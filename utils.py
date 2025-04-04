import numpy as np
import soundfile as sf


def load_audio(filename):
    audio, fs = sf.read(filename)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # modeling mono for now
    return fs, audio


def frame_audio(audio, frame_size, hop_size):
    num_frames = int((len(audio) - frame_size) / hop_size) + 1
    frames = np.stack([
        audio[i * hop_size: i * hop_size + frame_size]
        for i in range(num_frames)
    ])
    return frames


def apply_fft(frames, window):
    windowed = frames * window
    fft_result = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(fft_result)
    return magnitude


def spl_normalize(magnitude):
    power = magnitude ** 2 + 1e-10
    spl = 10 * np.log10(power)
    return spl


def is_tonal(spl, k):
    neighbors = [spl[k - 2], spl[k - 1], spl[k + 1], spl[k + 2]]
    if all(spl[k] - n > 7 for n in neighbors):
        return True
    return False


def bark_scale(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
