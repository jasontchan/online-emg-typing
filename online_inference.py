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
import numpy as np
import emg2qwerty.modules
import torch
import torchaudio
from emg2qwerty.lightning import TDSConvCTCModule
from emg2qwerty.transforms import NewLogSpectrogram
from scipy.ndimage import zoom

print(torch.__version__)
print(torchaudio.__version__)

sample_rate = 200
window_length = 3200 # this is just window length

print(f"Sample rate: {sample_rate}")
print(f"Main segment: {window_length} frames ({window_length / sample_rate} seconds)")


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        window_length (int): how big the sliding window is. built-in stride of chunk's shape at dim 0.
    """

    def __init__(self, window_length: int):
        self.window_length = window_length
        self.curr_window = np.zeros((window_length, 16), dtype=np.float32)

    def __call__(self, chunk: np.array):
        self.curr_window = np.concatenate((chunk, self.curr_window[:-chunk.shape[0]]))
        assert self.curr_window.shape[0] == self.window_length
        return self.curr_window

hypothesis = None

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


def worker(raw_q, mac, tty):
    print("MAC", mac, flush=True)
    m = Myo(mode=MODE, tty=tty)
    m.connect(input_address=mac)

    def add_to_queue(emg, movement):
        # curr_time = (time.time(),)
        # emg = emg + curr_time
        # print("EMG TUP</LE", emg)
        raw_q.put(emg)

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
    while True:
        while not (q_l.empty() or q_r.empty()):
            get_q_l = list(q_l.get())
            get_q_r = list(q_r.get())

            #adjust channel orientation to match emg2qwerty
            get_q_l[0], get_q_l[4] = get_q_l[4], get_q_l[0]
            get_q_l[1], get_q_l[3] = get_q_l[3], get_q_l[1]
            get_q_l[7], get_q_l[5] = get_q_l[5], get_q_l[7]

            get_q_r = get_q_r[-2:] + get_q_r[:-2]

            q.put(tuple(get_q_l + get_q_r))

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
        while not q.empty():
            chunk = q.get()
            yield (chunk,)

def interpolate_segment_halves(segment):
    """Doubles the channel count by interpolating neighboring channels."""
    T, C = segment.shape
    half_channels = C // 2

    left_half = segment[:, :half_channels]
    right_half = segment[:, half_channels:]

    left_interpolated = zoom(left_half, (1, 2), order=1, mode='wrap')
    right_interpolated = zoom(right_half, (1, 2), order=1, mode='wrap')

    interpolated_segment = np.concatenate((left_interpolated, right_interpolated), axis=1)
    return interpolated_segment



#order of operations is
'''
1. Connect both MyoBands (start_connection (calls worker for each band))
2. Start the recording of adding emg signals to the shared queue (start_recording (calls background_queue_insert))
3. Run inference (run_inference)
'''


stream_iterator = emg_generator()
cacher = ContextCacher(window_length=window_length)

ckpt_path = "splashlast_125_small.ckpt"
model_packet = TDSConvCTCModule.load_from_checkpoint(ckpt_path)
model_packet.eval()

model = model_packet.model 
decoder = model_packet.decoder

nBands = 2
nChannels = 16
target_rate = 125
sample_rate = 200
@torch.inference_mode()
def run_inference(num_iter=400):
    global hypothesis

    log_spec = NewLogSpectrogram(
            n_fft=64,         
            hop_length=1,
            sample_rate=sample_rate,
            target_rate=target_rate
        )

    buffer = []
    stride = 4
    for i, (chunk,) in enumerate(stream_iterator, start=1):
        # print(f"Processing chunk {i}...", flush=True)
        # print(f"Chunk shape: {len(chunk)}", flush=True)
        # print(f"Chunk: {chunk}", flush=True)
        buffer.append(list(chunk))
        if len(buffer) < stride:
            continue
        segment = cacher(np.array(buffer)) #shape (T, C) where C is both hands channel count
        buffer = []
        #resample segment
        segment = torch.tensor(interpolate_segment_halves(segment), dtype=torch.float32) #shape (T, 2*C) 2*C should be 32
        # print(f"segment length {segment.shape}", flush=True)
        # print(f"segment {segment}", flush=True) 
        #process chunk and convert to spectrogram
        start = time.time()
        segment = log_spec(segment)
        log_spec_time = time.time() - start
        print(f"Log spectrogram took {log_spec_time:.4f} seconds", flush=True)
        # print(f"segment after log spec {segment.shape}", flush=True)
        T, total_channels, freq_bins = segment.shape
        segment = segment.reshape(T, nBands, nChannels, freq_bins)
        segment = segment.unsqueeze(1) #add batch dimension N=1 for online inference
        # print(f"segment after reshape {segment.shape}", flush=True)

        start = time.time()
        logits = model(segment)  # shape (T, C, V) where V is vocab size
        model_time = time.time() - start
        print(f"Model inference took {model_time:.4f} seconds", flush=True)
        logits = logits.squeeze(1)
        # print(f"logits shape {logits.shape}", flush=True)

        window_duration = window_length / sample_rate
        first_timestamp = time.time() - window_duration
        timestamps = first_timestamp + np.arange(logits.shape[0]) / target_rate

        start = time.time()
        hypothesis = decoder.decode(logits, timestamps=timestamps)  # shape (T, C, V) -> (T, C) -> (C,)
        decode_time = time.time() - start
        print(f"Decoded in {decode_time:.4f} seconds", flush=True)
        print(f"Hypothesis: {hypothesis}", flush=True)

        if i == num_iter:
            break


if __name__ == "__main__":
    # Start the Myo connection
    start_connection()
    # Start the recording of EMG data
    start_recording()
    # Run inference
    run_inference(12000)
    
    # Clean up processes
    p_l.join()
    p_r.join()

# Clean up the queues
q_l.close()
q_r.close()
q.close()  
