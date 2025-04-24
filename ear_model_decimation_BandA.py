import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import *

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

    verbose and print("Selecting a frame with enough energy...")
    frame_idx = choose_informative_frame(spl, threshold_in_quiet)
    verbose and print(f"Picked frame {frame_idx} for plotting...")

    for i in range(spl.shape[0]):
        if dbfs[i] <= -96:
            continue

        frame_spl = spl[i]

        verbose and i == frame_idx and print("Identifying maskers...")
        flags, tonal_maskers, noise_maskers = identify_maskers(frame_spl.copy(), threshold_in_quiet)

        if i == frame_idx:
            raw_tonal_maskers = tonal_maskers[:]
            raw_noise_maskers = noise_maskers[:]

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
            captured_tonal_maskers = tonal_maskers[:]
            captured_noise_maskers = noise_maskers[:]
            captured_tonal_masking = tonal_masking
            captured_noise_masking = noise_masking

    if visualize:
        freqs = np.fft.rfftfreq(frame_size, d=1 / fs)
        bark_lines = get_bark_boundaries(bark_map, freqs)

        # Plot BEFORE decimation
        plt.figure()
        plt.plot(freqs, captured_frame_spl, label="SPL", linewidth=1)
        plt.plot(freqs, threshold_in_quiet, label="Threshold in Quiet", linestyle=':', linewidth=1)
        plt.scatter(freqs[raw_tonal_maskers], captured_frame_spl[raw_tonal_maskers], color='orange',
                    label="Raw Tonal Maskers", marker='^')
        plt.scatter(freqs[raw_noise_maskers], captured_frame_spl[raw_noise_maskers], color='cyan',
                    label="Raw Noise Maskers", marker='v')

        for f in bark_lines:
            plt.axvline(f, color='grey', linewidth=0.8, alpha=.4, zorder=0)

        plt.xscale("log")
        plt.title(f"Maskers Before Decimation - Frame {frame_idx}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB-SPL)")
        plt.ylim(-10, 100)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        # Plot AFTER decimation
        plt.figure()
        plt.plot(freqs, captured_frame_spl, label="SPL", linewidth=1)
        plt.plot(freqs, threshold_in_quiet, label="Threshold in Quiet", linestyle=':', linewidth=1)
        plt.scatter(freqs[captured_tonal_maskers], captured_frame_spl[captured_tonal_maskers], color='red',
                    label="Final Tonal Maskers", marker='o')
        plt.scatter(freqs[captured_noise_maskers], captured_frame_spl[captured_noise_maskers], color='green',
                    label="Final Noise Maskers", marker='x')

        for f in bark_lines:
            plt.axvline(f, color='grey', linewidth=0.8, alpha=.4, zorder=0)

        plt.xscale("log")
        plt.title(f"Maskers After Decimation - Frame {frame_idx}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB-SPL)")
        plt.ylim(-10, 100)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        tonal_vals = np.array([add_db(m) if m else np.nan for m in captured_tonal_masking])
        noise_vals = np.array([add_db(m) if m else np.nan for m in captured_noise_masking])

        plt.figure()
        plt.plot(freqs, captured_frame_spl, label="SPL", linewidth=1)
        plt.plot(freqs, threshold_in_quiet, label="Threshold in Quiet", linestyle=':', linewidth=1)
        plt.plot(freqs, tonal_vals, label="Tonal Threshold", linestyle='--', color='red', linewidth=1)
        plt.plot(freqs, noise_vals, label="Noise Threshold", linestyle='--', color='green', linewidth=1)
        plt.scatter(freqs[captured_tonal_maskers], captured_frame_spl[captured_tonal_maskers], color='red',
                    label="Tonal", marker='o')
        plt.scatter(freqs[captured_noise_maskers], captured_frame_spl[captured_noise_maskers], color='green',
                    label="Noise", marker='x')

        for f in bark_lines:
            plt.axvline(f, color='grey', linewidth=0.8, alpha=.4, zorder=0)

        plt.xscale("log")
        plt.title(f"Individual Masking Thresholds - Frame {frame_idx}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB-SPL)")
        plt.ylim(-10, 100)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        avg_smr = np.mean(smr_accum, axis=0)

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