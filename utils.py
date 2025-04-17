import numpy as np
import soundfile as sf
from scipy.signal import get_window


def load_audio(filename):
    audio, fs = sf.read(filename)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # modeling mono for now
    return audio, fs


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
    spl = 20 * np.log10(magnitude + 1e-12)
    peak = np.max(spl, axis=1, keepdims=True)
    spl += 96.0 - peak
    return spl


def fft_and_normalize(audio, frame_size, hop_size, window_type='hann'):
    window = get_window(window_type, frame_size)
    frames = frame_audio(audio, frame_size, hop_size)
    magnitude = apply_fft(frames, window)
    spl = spl_normalize(magnitude)

    return spl


def dbspl_to_dbfs(audio, frame_size, hop_size, window_type):
    frames = frame_audio(audio, frame_size, hop_size)
    window = get_window(window_type, frame_size)
    rms = np.sqrt(np.mean((frames * window) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-12)


def bark_scale(f):
    # B = 13*atan(0.00076*f)+3.5*atan((f/7500).^2);
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def threshold_in_quiet(f):
    # (Zwicker & Fastl 1999)
    # LTQ=3.64*(f/1000.)**-0.8 -6.5*np.exp(-0.6*(f/1000.-3.3)**2.)+1e-3*((f/1000.)**4.)
    ratio = f / 1000.0
    ltq = (3.64 * (ratio ** -0.8) - 6.5 * np.exp(-0.6 * ((ratio - 3.3) ** 2)) + 0.001 * (ratio ** 4))
    return ltq


def compute_bark_and_threshold(fs, frame_size):
    freqs = np.fft.rfftfreq(frame_size, d=1 / fs)  # length = frame_size//2 + 1

    bark_map = bark_scale(freqs)

    # Compute thresholds instead of pre-baked table
    thresholds = np.zeros_like(freqs)
    for k, f in enumerate(freqs):
        if f < 20:  # anything below 20hz
            thresholds[k] = 80.0
        else:
            thresholds[k] = threshold_in_quiet(f)
    return bark_map, thresholds


def get_bark_boundaries(bark_map, freqs):
    bark_int = np.floor(bark_map).astype(int)
    idx = np.where(np.diff(bark_int))[0]  # indices before change
    return freqs[idx]  # Hz values


UNSET = 0
TONE = 1
NOISE = 2
IGNORE = 3


def add_db(db_values):
    linear_sum = np.sum(10 ** (np.array(db_values) / 10.0))
    return 10 * np.log10(linear_sum + 1e-12)


def identify_maskers(spl, threshold_in_quiet):
    n = len(spl)
    flags = np.zeros(n, dtype=np.uint8)
    tonal = []

    for k in range(2, n - 2):
        if spl[k] >= spl[k + 1] and spl[k] > spl[k - 1]:
            rng = (-2, -1, 1, 2) if k < 63 else (-3, -2, -1, 1, 2, 3) if k < 127 else \
                (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)

            if all(0 <= k + j < n and spl[k] - spl[k + j] >= 5  # ← 7 dB ⇒ 5 dB
                   for j in rng):

                if spl[k] - threshold_in_quiet[k] > -10:  # allow 10 dB below Tq
                    tonal.append(k)
                    flags[k] = TONE
                    for j in rng:
                        nb = k + j
                        if 0 <= nb < n and flags[nb] == UNSET:
                            flags[nb] = IGNORE

    noise = [k for k in range(n) if flags[k] == UNSET]
    for k in noise:
        flags[k] = NOISE
    return flags, tonal, noise


def decimate_maskers(spl, flags, tonal_maskers, noise_maskers, bark_map, threshold_in_quiet):
    # Decimate tonal maskers below threshold
    tonal_maskers = [k for k in tonal_maskers if spl[k] >= threshold_in_quiet[k]]
    for k in range(len(flags)):
        if flags[k] == TONE and k not in tonal_maskers:
            flags[k] = IGNORE

    # Decimate noise maskers below threshold
    noise_maskers = [k for k in noise_maskers if spl[k] >= threshold_in_quiet[k]]
    for k in range(len(flags)):
        if flags[k] == NOISE and k not in noise_maskers:
            flags[k] = IGNORE

    # Merge tonal maskers that are closer than 0.5 Bark
    # Sort tonal maskers by bark position
    tonal_maskers = sorted(tonal_maskers, key=lambda k: bark_map[k])
    i = 0
    while i < len(tonal_maskers) - 1:
        k1 = tonal_maskers[i]
        k2 = tonal_maskers[i + 1]
        if abs(bark_map[k1] - bark_map[k2]) < 0.5:
            # Remove weaker
            if spl[k1] >= spl[k2]:
                flags[k2] = IGNORE
                tonal_maskers.pop(i + 1)
            else:
                flags[k1] = IGNORE
                tonal_maskers.pop(i)
        else:
            i += 1

    # Merge noise maskers that are closer than 0.5 Bark
    noise_maskers = sorted(noise_maskers, key=lambda k: bark_map[k])
    i = 0
    while i < len(noise_maskers) - 1:
        k1 = noise_maskers[i]
        k2 = noise_maskers[i + 1]
        if abs(bark_map[k1] - bark_map[k2]) < 0.5:
            # Remove weaker
            if spl[k1] >= spl[k2]:
                flags[k2] = IGNORE
                noise_maskers.pop(i + 1)
            else:
                flags[k1] = IGNORE
                noise_maskers.pop(i)
        else:
            i += 1

    return flags, tonal_maskers, noise_maskers


def spreading_function(bark_distance, masker_spl, bark_mask_bin, masker_type='tonal'):
    if bark_distance < -3 or bark_distance > 8:
        return None  # negligible masking

    if masker_type == "tonal":
        av = -1.525 - 0.275 * bark_mask_bin - 4.5
    elif masker_type == "noise":
        av = -1.525 - 0.175 * bark_mask_bin - 0.5

    if bark_distance < -1:
        vf = 17 * (bark_distance + 1) - (0.4 * masker_spl + 6)
    elif bark_distance < 0:
        vf = bark_distance * (0.4 * masker_spl + 6)
    elif bark_distance < 1:
        vf = -17 * bark_distance
    else:
        vf = -(bark_distance - 1) * (17 - 0.15 * masker_spl) - 17

    return masker_spl + vf + av


def calc_individual_thresholds(spl, tonal_maskers, noise_maskers, bark_map):
    n_bins = len(spl)
    tonal_masking = [[] for _ in range(n_bins)]
    noise_masking = [[] for _ in range(n_bins)]

    for i in range(n_bins):
        bark_bin = bark_map[i]

        for m in tonal_maskers:
            mask_bin = bark_map[m]
            bark_dist = bark_bin - mask_bin
            mt = spreading_function(bark_dist, spl[m], mask_bin, masker_type='tonal')
            if mt is not None:
                tonal_masking[i].append(mt)

        for m in noise_maskers:
            mask_bin = bark_map[m]
            bark_dist = bark_bin - mask_bin
            mt = spreading_function(bark_dist, spl[m], mask_bin, masker_type='noise')
            if mt is not None:
                noise_masking[i].append(mt)

    return tonal_masking, noise_masking


def global_masking_threshold(tonal_masking, noise_masking, threshold_in_quiet):
    n_bins = len(threshold_in_quiet)
    global_mask = np.zeros(n_bins)

    for i in range(n_bins):
        all_maskers = [threshold_in_quiet[i]] + tonal_masking[i] + noise_masking[i]
        global_mask[i] = add_db(all_maskers)

    return global_mask


def get_uniform_subbands(n_bins=513, n_subbands=32):
    bins_per_band = n_bins // n_subbands
    boundaries = []
    for i in range(n_subbands):
        start = i * bins_per_band
        end = (i + 1) * bins_per_band if i < n_subbands - 1 else n_bins
        boundaries.append((start, end))
    return boundaries


def compute_subband_smr(spl, global_mask, subband_boundaries):
    n_subbands = len(subband_boundaries)
    smr = np.zeros(n_subbands)

    for sb, (start, end) in enumerate(subband_boundaries):
        if end > len(spl):  # just in case of boundary mismatch
            end = len(spl)

        signal_level = np.max(spl[start:end])
        mask_level = np.min(global_mask[start:end])

        smr[sb] = signal_level - mask_level

    return smr


def choose_informative_frame(spl, threshold_in_quiet, min_tonal=5, min_noise=5, rms_floor_db=-96):
    def spl_to_lin(spl_db):
        return 10 ** (spl_db / 20.0)

    rms_db = 20 * np.log10(np.sqrt(np.mean(spl_to_lin(spl), axis=1)) + 1e-12)

    best_idx, best_score = 0, -1
    for i, frame_spl in enumerate(spl):
        if rms_db[i] < rms_floor_db:
            continue

        _, tonal, noise = identify_maskers(frame_spl, threshold_in_quiet)

        if len(tonal) >= min_tonal and len(noise) >= min_noise:
            return i

        score = len(tonal) + len(noise)
        if score > best_score:
            best_idx, best_score = i, score

    return best_idx
