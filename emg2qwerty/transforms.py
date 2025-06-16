# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, Sequence, Tuple

import numpy as np
import math
import torch
import torchaudio
import torch.nn.functional as F

TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16
    sample_rate: int = 200
    target_rate: int = 125

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=6,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        out = logspec.movedim(-1, 0)  # (T, ..., C, freq)
        if self.target_rate != self.sample_rate:
            ratio = self.target_rate / self.sample_rate
            T_old = out.shape[0]
            T_new = int(T_old * ratio)

            rest = out.shape[1:]
            out = out.reshape(T_old, -1).permute(1, 0).unsqueeze(1).unsqueeze(3)
            out = torch.nn.functional.interpolate(
                out, size=(T_new, 1), mode="bilinear", align_corners=False
            )
            out = out.squeeze(3).squeeze(1).permute(1, 0)
            # back into (T_new, …, C, 6)
            out = out.reshape((T_new,) + rest)
        return out


@dataclass
class NewLogSpectrogram:
    n_fft: int = 64
    hop_length: int = 16
    sample_rate: int = 200  # Critical addition for frequency calculations
    target_rate: int = 200

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )
        self._create_aggregation_matrix()

    def _create_aggregation_matrix(self) -> None:
        freq_ranges = [
            (31.25, 62.5), (62.5, 100),
            (100, 250), (250, 375),
            (375, 687.5), (687.5, 1000)
        ]
        # freq_ranges = [
        #     (31.25, 52.5), (52.5, 75),
        #     (75, 100), (100, 375),
        #     (375, 687.5), (687.5, 1000)
        # ]
        # freq_ranges = [
        #     (3.125, 6.25), (6.25, 12.5), 
        #     (12.5, 25), (25, 37.5),
        #     (37.5, 68.75), (68.75, 100)
        # ]
        
        num_bins = self.n_fft // 2 + 1
        aggregation = torch.zeros((6, num_bins))
        for i, (lower, upper) in enumerate(freq_ranges):
            lower_bin = int(lower * self.n_fft / self.sample_rate)
            upper_bin = int(upper * self.n_fft / self.sample_rate)
            if upper == self.sample_rate // 2:
                upper_bin = num_bins
            aggregation[i, lower_bin:upper_bin] = 1
        
        self.aggregation_matrix = aggregation

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)
        spec = self.spectrogram(x)
        logspec = torch.log10(spec + 1e-6).movedim(-1, 0)
        # Reshape for batch matrix multiplication
        orig_shape = logspec.shape
        logspec = logspec.view(-1, orig_shape[-2], orig_shape[-1])  # (N, C, freq)
        
        # Aggregate using precomputed matrix
        aggregated = torch.einsum('...cf,bf->...cb', 
                                logspec, 
                                self.aggregation_matrix)
        
        # Restore original dimensions
        out = aggregated.view(orig_shape[:-1] + (6,))  # (T, ..., C, 6)

        # For low Hz data, broadcast strongest low-frequency band to upper bins at each time stamp
        # second_band = out[..., 1:2] 
        # out[..., 2:] = second_band.expand_as(out[..., 2:])

        # Optional downsampling
        if self.target_rate != self.sample_rate:
            ratio = self.target_rate / self.sample_rate 
            T_old = out.shape[0]
            T_new = int(T_old * ratio)
      
            rest = out.shape[1:]
            out = out.view(T_old, -1).permute(1, 0).unsqueeze(1).unsqueeze(3)
            out = F.interpolate(
                out,
                size=(T_new, 1),
                mode='bilinear',
                align_corners=False
            )
            out = out.squeeze(3).squeeze(1).permute(1, 0)
            # back into (T_new, …, C, 6)
            out = out.view((T_new,) + rest)
        return out

    
@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

from dataclasses import dataclass
from typing import Sequence, Tuple
import torch
import numpy as np
import math

@dataclass
class RandomBandMobiusTransform:
    """
    Applies a Möbius-transformation style band augmentation.
    
    The 16 electrode channels are assumed to be arranged on the unit circle 
    at angles theta = 2*pi*k/16, k=0,...,15. For each application a random 
    inner circle is chosen (with radius r uniformly in [r_min, 1]) and a 
    random fixed point angle theta_fixed is chosen (uniformly from theta_fixed_range). 
    Then the Möbius transformation
        f(z) = (T*z + lambda*T^2) / (lambda*z + T)
    is applied to the complex positions representing the electrode locations, 
    where T = exp(i * theta_fixed) and lambda = (1 - r)/(1 + r). The new electrode 
    positions (given by f(z)) are converted into fractional indices, and the 
    output signal is obtained by linear interpolation of the original channels.
    
    Args:
        r_min (float): Minimum inner-circle radius to use. Must be in (0,1]. 
                       r=1 corresponds to the identity transform. (Default: 0.4)
        theta_fixed_range (tuple): A tuple (theta_fixed_min, theta_fixed_max) from 
                       which theta_fixed is drawn uniformly (in radians). 
                       (Default: (0, 2*pi))
        channel_dim (int): The dimension of the input tensor corresponding to the channels.
                           (Default: -1)
    """
    r_min: float = 0.4
    theta_fixed_range: Tuple[float, float] = (0, 2 * math.pi)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Get number of channels; we assume 16.
        C = tensor.shape[self.channel_dim]
        if C != 16:
            raise ValueError(f"Expected 16 channels, got {C}.")
        
        # Sample random parameters:
        r = np.random.uniform(self.r_min, 1.0)   # inner circle radius in [r_min, 1]
        theta_fixed = np.random.uniform(self.theta_fixed_range[0],
                                        self.theta_fixed_range[1])
        # Outer circle radius is 1.
        R = 1.0
        # Compute lambda = (1 - r) / (1 + r). Notice that if r==1 then lambda==0.
        lam = (R - r) / (R + r)
        # Fixed point on the unit circle:
        T = np.exp(1j * theta_fixed)
        
        # Compute the original electrode positions.
        # These are the 16 points uniformly arranged on the circle:
        angles_orig = np.linspace(0, 2 * np.pi, C, endpoint=False)
        z_orig = np.exp(1j * angles_orig)
        
        # Apply the Möbius transform:
        # f(z) = (T*z + lam*T^2) / (lam*z + T)
        new_z = (T * z_orig + lam * T**2) / (lam * z_orig + T)
        # Convert the complex positions back to angles in [0, 2*pi):
        new_angles = np.mod(np.angle(new_z), 2 * np.pi)
        
        # Interpret these new angles as fractional indices on the uniform grid.
        # (The uniform grid is defined by positions theta = 2*pi*k/16 for k=0,...,15.)
        fractional_indices = new_angles / (2 * np.pi) * C  # values in [0, C)
        
        # For each new position, compute the lower and upper indices and the linear interpolation weight.
        idx_lower = np.floor(fractional_indices).astype(int)       # integer indices
        idx_upper = (idx_lower + 1) % C                              # next index, wrapping around
        weight = fractional_indices - np.floor(fractional_indices)   # fractional part
        
        # --- Now, perform interpolation on the channel dimension.
        # We assume the input tensor's channel dimension holds the 16 electrode signals.
        # One simple (and efficient enough, since C=16) method is to unbind the channels,
        # interpolate channel by channel, and stack back.
        
        channels = list(torch.unbind(tensor, dim=self.channel_dim))
        new_channels = []
        for i in range(C):
            lower_idx = int(idx_lower[i])
            upper_idx = int(idx_upper[i])
            w = float(weight[i])
            # Linear interpolation: (1-w)*signal_at_lower + w*signal_at_upper.
            new_channel = (1 - w) * channels[lower_idx] + w * channels[upper_idx]
            new_channels.append(new_channel)
            
        # Stack the new channels along the channel dimension.
        out = torch.stack(new_channels, dim=self.channel_dim)
        return out
