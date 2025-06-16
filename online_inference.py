'''
Overview:
1. Build inference pipeline
2. Format waveform into chunks
3. Pass data through pipeline


what im thinking...
our model takes raw signal and does allat within its own module
do i just chunk first and then pass through model?
unsure.

need to stream myo data first
'''

import multiprocessing
import torch
import torchaudio
import emg2qwerty

print(torch.__version__)
print(torchaudio.__version__)

# import IPython
import matplotlib.pyplot as plt
# from torchaudio.io import StreamReader

######################################################################
# 3. Construct the pipeline
# -------------------------
#
# Pre-trained model weights and related pipeline components are
# bundled as :py:class:`torchaudio.pipelines.RNNTBundle`.
#
# We use :py:data:`torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH`,
# which is a Emformer RNN-T model trained on LibriSpeech dataset.
#

# bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

# feature_extractor = bundle.get_streaming_feature_extractor()
# decoder = bundle.get_decoder()
# token_processor = bundle.get_token_processor()

######################################################################
# Streaming inference works on input data with overlap.
# Emformer RNN-T model treats the newest portion of the input data
# as the "right context" — a preview of future context.
# In each inference call, the model expects the main segment
# to start from this right context from the previous inference call.
# The following figure illustrates this.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/emformer_rnnt_context.png
#
# The size of main segment and right context, along with
# the expected sample rate can be retrieved from bundle.
#

sample_rate = 200
segment_length = bundle.segment_length * bundle.hop_length
context_length = bundle.right_context_length * bundle.hop_length

print(f"Sample rate: {sample_rate}")
print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")

######################################################################
# 4. Configure the audio stream
# -----------------------------
#
# Next, we configure the input audio stream using :py:class:`torchaudio.io.StreamReader`.
#
# For the detail of this API, please refer to the
# `StreamReader Basic Usage <./streamreader_basic_tutorial.html>`__.
#

######################################################################
# The following audio file was originally published by LibriVox project,
# and it is in the public domain.
#
# https://librivox.org/great-pirate-stories-by-joseph-lewis-french/
#
# It was re-uploaded for the sake of the tutorial.
#


######################################################################
# As previously explained, Emformer RNN-T model expects input data with
# overlaps; however, `Streamer` iterates the source media without overlap,
# so we make a helper structure that caches a part of input data from
# `Streamer` as right context and then appends it to the next input data from
# `Streamer`.
#
# The following figure illustrates this.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/emformer_rnnt_streamer_context.png
#


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context


######################################################################
# 5. Run stream inference
# -----------------------
#
# Finally, we run the recognition.
#
# First, we initialize the stream iterator, context cacher, and
# state and hypothesis that are used by decoder to carry over the
# decoding state between inference calls.
#

cacher = ContextCacher(segment_length, context_length)

state, hypothesis = None, None

######################################################################
# Next we, run the inference.
#
# For the sake of better display, we create a helper function which
# processes the source stream up to the given times and call it
# repeatedly.
#

#make my own emg streamer
import sys

sys.path.append("../")  # Add parent directory to path
from pyomyo_main.src.pyomyo.pyomyo import Myo, emg_mode
import time

# ------------ Myo Setup ---------------
MODE = emg_mode.RAW
q_l = multiprocessing.Queue()
q_r = multiprocessing.Queue()
q = multiprocessing.Queue() # Queue to combine left and right EMG data


def worker(q, mac, tty):
    print("MAC", mac, flush=True)
    m = Myo(mode=MODE, tty=tty)
    m.connect(input_address=mac)

    def add_to_queue(emg, movement):
        curr_time = (time.time(),)
        emg = emg + curr_time
        # print("EMG TUP</LE", emg)
        q.put(emg)

    m.add_emg_handler(add_to_queue)

    # Orange logo and bar LEDs
    m.set_leds([128, 128, 0], [128, 128, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)

    """worker function"""
    while True:
        m.run()
    print("Worker Stopped")

#combines left and right emg data
def background_queue_insert(q_l, q_r, q):
    while not (q_l.empty() or q_r.empty()):
        # data_l = ['One', 'Two', 'Three', "Four", "Five", "Six", "Seven", "Eight", "Time"]
        get_q_l = q_l.get()
        get_q_r = q_r.get()

        q.put(get_q_l[:-1] + get_q_r[:-1])

def start_recording():
    p_recording = multiprocessing.Process(
                    target=background_queue_insert,
                    args=(q_l, q_r, q),
                )
    p_recording.start()

def start_connection():
    global p_l, p_r
    L_MAC = [176, 102, 48, 60, 192, 228]
    R_MAC = [67, 145, 190, 132, 38, 228]

    p_l = multiprocessing.Process(
        target=worker,
        args=(q_l, L_MAC, "/dev/ttyACM1"),
    )
    p_l.start()

    p_r = multiprocessing.Process(
        target=worker,
        args=(q_r, R_MAC, "/dev/ttyACM2"),
    )
    p_r.start()


def emg_generator():
    """Generator function to yield EMG data from the queue."""
    while True:
        chunk = q.get()
        yield (chunk,)

stream_iterator = emg_generator()


#order of operations is
'''
1. Connect both MyoBands (start_connection (calls worker for each band))
2. Start the recording of adding emg signals to the shared queue (start_recording (calls background_queue_insert))
3. Run inference (run_inference)
'''


def _plot(feats, num_iter, unit=25):
    unit_dur = segment_length / sample_rate * unit
    num_plots = num_iter // unit + (1 if num_iter % unit else 0)
    fig, axes = plt.subplots(num_plots, 1)
    t0 = 0
    for i, ax in enumerate(axes):
        feats_ = feats[i * unit : (i + 1) * unit]
        t1 = t0 + segment_length / sample_rate * len(feats_)
        feats_ = torch.cat([f[2:-2] for f in feats_])  # remove boundary effect and overlap
        ax.imshow(feats_.T, extent=[t0, t1, 0, 1], aspect="auto", origin="lower")
        ax.tick_params(which="both", left=False, labelleft=False)
        ax.set_xlim(t0, t0 + unit_dur)
        t0 = t1
    fig.suptitle("MelSpectrogram Feature")
    plt.tight_layout()


@torch.inference_mode()
def run_inference(num_iter=100):
    global hypothesis
    for i, (chunk,) in enumerate(stream_iterator, start=1):
        # segment = cacher(chunk[:, 0])
        # features, length = feature_extractor(segment)
        # hypos, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
        

        '''
        process the chunk by converting to spectrogram
        pass it through emg2qwerty model
        get the hypothesis
        process the token, print the character

        '''
        hypothesis = hypos
        transcript = token_processor(hypos[0][0], lstrip=False)
        print(transcript, end="\r", flush=True)

        chunks.append(chunk)
        feats.append(features)
        if i == num_iter:
            break

    # Plot the features
    _plot(feats, num_iter)
    return IPython.display.Audio(torch.cat(chunks).T.numpy(), rate=bundle.sample_rate)


######################################################################

if __name__ == "__main__":
    # Start the Myo connection
    start_connection()
    # Start the recording of EMG data
    start_recording()
    # Run inference
    run_inference(100)
    
    # Clean up processes
    p_l.join()
    p_r.join()

# Clean up the queues
q_l.close()
q_r.close()
q.close()  

######################################################################