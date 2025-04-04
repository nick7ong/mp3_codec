import matplotlib.pyplot as plt
from scipy.signal import get_window
from utils import *


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


if __name__ == '__main__':
    filename = 'audio/queen.wav'
    frame_size = 1024
    hop_size = frame_size // 2
    window_type = 'hann'

    fs, audio = load_audio(filename)

    # Task 2.1
    spl = fft_analysis(audio, frame_size, hop_size, window_type)

    frame_idx = 10  # arbitrary frame idx
    freqs = np.fft.rfftfreq(frame_size, d=1 / fs)

    # Plot SPL normalized FFT
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spl[frame_idx])
    plt.title(f'FFT - Frame {frame_idx} (Post SPL Normalization)')
    plt.xlim([0, fs / 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Task 2.2
    tonal_maskers, noise_maskers = identify_maskers(spl, fs, frame_size)


