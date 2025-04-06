import matplotlib.pyplot as plt
from scipy.signal import get_window
from utils import *
import numpy as np


def fft_analysis(audio, frame_size, hop_size, window_type='hann'):
    """
    Task 1: FFT analysis: include 50% overlap, windowing, and SPL normalization.
    """
    window = get_window(window_type, frame_size)
    frames = frame_audio(audio, frame_size, hop_size)
    magnitude = apply_fft(frames, window)
    spl = spl_normalize(magnitude)

    return spl


def identify_maskers(spl_spectrum, sample_rate, frame_size):
    """
    Task 2: Identify tonal and noise maskers in each FFT frame.
    """

    return tonal_maskers, noise_maskers

def convertTo_dbFS(audio, spl, frame_size, hop_size, window_type):
    """
    Converts per-frame RMS to dBFS and zeroes SPL where dBFS is ≤ -96 dB.
    """
    frames = frame_audio(audio, frame_size, hop_size)
    window = get_window(window_type, frame_size)
    windowed = frames * window

    rms_vals = np.sqrt(np.mean(windowed**2, axis=1)) 
    dbfs = 20 * np.log10(rms_vals) 
    spl[dbfs <= -96] = 0

    return dbfs, spl

if __name__ == '__main__':
    filename = '/Users/nicolasadler/mue610/mp3_codec/mp3_codec/flute.wav'
    frame_size = 1024
    hop_size = frame_size // 2
    window_type = 'hann'

    fs, audio = load_audio(filename)

    # Task 2.1
    spl = fft_analysis(audio, frame_size, hop_size, window_type)

    frame_idx = 10  # arbitrary frame idx
    freqs = np.fft.rfftfreq(frame_size, d=1 / fs)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spl[frame_idx], label='Original SPL')
    convertTo_dbFS(audio, spl, frame_size, hop_size, window_type)
    plt.plot(freqs, spl[frame_idx], label='After -96 dBFS = 0 dB SPL')

    plt.title(f'FFT - Frame {frame_idx}')
    plt.xlim([0, fs / 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Task 2.2
   #tonal_maskers, noise_maskers = identify_maskers(spl, fs, frame_size)