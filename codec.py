import argparse
import random

import matplotlib.pyplot as plt

from utils import *


def choose_random_frame(spl, energy_threshold=-70):
    frame_energies = np.mean(spl, axis=1)
    candidates = np.where(frame_energies > energy_threshold)[0]
    return random.choice(candidates) if len(candidates) > 0 else 0


def ear_model(audio_file, visualize=True, verbose=False):
    frame_size = 1024
    hop_size = frame_size // 2
    window_type = 'hann'

    verbose and print("Loading audio...")
    audio, fs = load_audio(audio_file)

    verbose and print("FFT and SPL normalization...")
    spl = fft_and_normalize(audio, frame_size, hop_size, window_type)

    verbose and print("Calculating dBFS per frame...")
    dbfs = dbspl_to_dbfs(audio, frame_size, hop_size, window_type)

    verbose and print("Computing Bark scale and Threshold in Quiet...")
    bark_map, threshold_in_quiet = compute_bark_and_threshold(fs, frame_size)

    subband_boundaries = get_uniform_subbands(len(spl[0]), n_subbands=32)

    global_masks = []
    smr_accum = []

    verbose and print("Selecting a frame with enough energy..")
    frame_idx = choose_random_frame(spl)

    for i in range(spl.shape[0]):
        if dbfs[i] <= -96:
            continue

        frame_spl = spl[i]

        verbose and i == frame_idx and print("Identifying maskers...")
        flags, tonal_maskers, noise_maskers = identify_maskers(frame_spl.copy(), threshold_in_quiet)

        verbose and i == frame_idx and print("Decimating maskers...")
        flags, tonal_maskers, noise_maskers = decimate_maskers(
            frame_spl, flags, tonal_maskers, noise_maskers, bark_map, threshold_in_quiet
        )

        verbose and i == frame_idx and print("Calculating individual masking thresholds...")
        tonal_masking, noise_masking = calc_individual_thresholds(
            frame_spl, tonal_maskers, noise_maskers, bark_map
        )

        verbose and i == frame_idx and print("Computing global masking threshold...")
        global_mask = global_masking_threshold(tonal_masking, noise_masking, threshold_in_quiet)
        global_masks.append(global_mask)

        verbose and i == frame_idx and print("Computing SMR per subband...")
        smr = compute_subband_smr(frame_spl, global_mask, subband_boundaries)
        smr_accum.append(smr)

        if i == frame_idx:
            captured_frame_spl = frame_spl.copy()
            captured_flags = flags.copy()
            captured_tonal_maskers = tonal_maskers[:]
            captured_noise_maskers = noise_maskers[:]
            captured_tonal_masking = tonal_masking
            captured_noise_masking = noise_masking

    if visualize:
        freqs = np.fft.rfftfreq(frame_size, d=1 / fs)

        # Tonal/Noise Masker Identification for One Frame
        plt.figure()
        plt.plot(freqs, captured_frame_spl, label="SPL")
        plt.scatter(freqs[captured_tonal_maskers], captured_frame_spl[captured_tonal_maskers], color='red',
                    label="Tonal")
        plt.scatter(freqs[captured_noise_maskers], captured_frame_spl[captured_noise_maskers], color='green',
                    label="Noise")
        plt.xscale("log")
        plt.title(f"Masker Identification - Frame {frame_idx}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB)")
        plt.legend()
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.show()

        # Individual Masking Thresholds
        tonal_vals = np.array([add_db(m) if m else np.nan for m in captured_tonal_masking])
        noise_vals = np.array([add_db(m) if m else np.nan for m in captured_noise_masking])

        plt.figure()
        plt.plot(freqs, captured_frame_spl, label="SPL")
        plt.plot(freqs, tonal_vals, label="Tonal Threshold", linestyle='--')
        plt.plot(freqs, noise_vals, label="Noise Threshold", linestyle=':')
        plt.xscale("log")
        plt.title(f"Individual Masking Thresholds - Frame {frame_idx}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB)")
        plt.legend()
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.show()

        # Average SPL, Global Mask, Threshold in Quiet
        avg_spl = np.mean(spl, axis=0)
        avg_mask = np.mean(global_masks, axis=0)
        avg_smr = np.mean(smr_accum, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(freqs, avg_spl, label="Average SPL")
        plt.plot(freqs, avg_mask, label="Average Global Mask", linestyle='--')
        plt.plot(freqs, threshold_in_quiet, label="Threshold in Quiet", linestyle=':')
        plt.xscale("log")
        plt.title("Average Frequency Content Over Time")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB)")
        plt.legend()
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.show()

        # SMR per Subband
        plt.figure()
        plt.stem(avg_smr)
        plt.title("Average SMR per Subband")
        plt.xlabel("Subband Index")
        plt.ylabel("SMR (dB)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Psychoacoustic Model 2 Analysis")
    parser.add_argument("filename", type=str, help="Path to input audio file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose step logging")
    args = parser.parse_args()

    ear_model(args.filename, visualize=True, verbose=args.verbose)
