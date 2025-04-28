import threading

import matplotlib.pyplot as plt

from utils import *


def process_channel(audio_channel, frame_size, hop_size, window_type, smr_threshold_db, fs):
    spl = fft_and_normalize(audio_channel, frame_size, hop_size, window_type)
    dbfs = dbspl_to_dbfs(audio_channel, frame_size, hop_size, window_type)
    bark_map, threshold_in_quiet = compute_bark_and_threshold(fs, frame_size)

    frames = frame_audio(audio_channel, frame_size, hop_size)
    window = get_window(window_type, frame_size)
    magnitude = apply_fft(frames, window)
    phase = np.angle(np.fft.rfft(frames * window, axis=1))

    output_frames = []
    smr_accum = []
    subband_boundaries = get_uniform_subbands(len(magnitude[0]), n_subbands=32)

    total_bins = 0
    zeroed_bins = 0

    for i in range(spl.shape[0]):
        if dbfs[i] <= -96:
            output_frames.append(np.zeros(frame_size))
            smr_accum.append(np.zeros(32))
            continue

        frame_spl = spl[i]

        flags, tonal_maskers, noise_maskers = identify_maskers(frame_spl.copy(), threshold_in_quiet)
        flags, tonal_maskers, noise_maskers = decimate_maskers(
            frame_spl, flags, tonal_maskers, noise_maskers, bark_map, threshold_in_quiet
        )
        tonal_masking, noise_masking = calc_individual_thresholds(
            frame_spl, tonal_maskers, noise_maskers, bark_map
        )
        global_mask = global_masking_threshold(tonal_masking, noise_masking, threshold_in_quiet)

        mask = frame_spl >= (global_mask + smr_threshold_db)
        new_magnitude = magnitude[i] * mask

        total_bins += len(mask)
        zeroed_bins += np.sum(~mask)

        new_spectrum = new_magnitude * np.exp(1j * phase[i])

        time_frame = np.fft.irfft(new_spectrum, n=frame_size)
        output_frames.append(time_frame)

        smr = compute_subband_smr(frame_spl, global_mask, subband_boundaries)
        smr_accum.append(smr)

    reconstructed_audio = overlap_add(output_frames, hop_size)
    return reconstructed_audio, np.array(smr_accum), total_bins, zeroed_bins


def quantize_audio(audio_file, frame_size=1024, hop_size=512, window_type='hann', smr_threshold_db=0, visualize=False):
    print("Loading audio...")
    audio, fs = load_audio(audio_file)

    if audio.ndim == 1:
        print("Processing mono audio...")
        quantized_audio, smr_data, total_bins, zeroed_bins = process_channel(audio, frame_size, hop_size, window_type,
                                                                             smr_threshold_db, fs)
    else:
        print("Processing stereo audio...")
        left_result = []
        right_result = []
        left_smr = []
        right_smr = []
        left_bins = []
        right_bins = []

        def process_left():
            print("Processing left channel...")
            audio_out, smr_out, total, zeroed = process_channel(audio[:, 0], frame_size, hop_size, window_type,
                                                                smr_threshold_db, fs)
            left_result.append(audio_out)
            left_smr.append(smr_out)
            left_bins.append((total, zeroed))

        def process_right():
            print("Processing right channel...")
            audio_out, smr_out, total, zeroed = process_channel(audio[:, 1], frame_size, hop_size, window_type,
                                                                smr_threshold_db, fs)
            right_result.append(audio_out)
            right_smr.append(smr_out)
            right_bins.append((total, zeroed))

        left_thread = threading.Thread(target=process_left)
        right_thread = threading.Thread(target=process_right)

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

        quantized_audio = np.stack((left_result[0], right_result[0]), axis=-1)

        # Analyze SMR and determine lowpass cutoff
        avg_smr = (np.mean(left_smr[0], axis=0) + np.mean(right_smr[0], axis=0)) / 2
        cutoff_band = np.argmax(avg_smr < smr_threshold_db)
        bandwidth_per_band = (fs / 2) / 32
        cutoff_freq = (cutoff_band + 1) * bandwidth_per_band

        if cutoff_freq < (fs / 2):
            print(f"Applying lowpass filter with cutoff frequency {cutoff_freq:.2f} Hz...")
            quantized_audio = apply_lowpass(quantized_audio, fs, cutoff_freq)

        total_bins = left_bins[0][0] + right_bins[0][0]
        zeroed_bins = left_bins[0][1] + right_bins[0][1]

    if visualize:
        print("Visualizing spectrograms...")
        visualize_spectrograms(audio, quantized_audio, fs)

    removed_percentage = (zeroed_bins / total_bins) * 100
    print(f"Percentage of spectral information removed: {removed_percentage:.2f}%")

    return quantized_audio, fs


def visualize_spectrograms(original, quantized, fs):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.specgram(original[:, 0] if original.ndim > 1 else original, Fs=fs, NFFT=1024, noverlap=512, cmap='magma')
    plt.title("Original Audio Spectrogram (Left Channel)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(2, 1, 2)
    plt.specgram(quantized[:, 0] if quantized.ndim > 1 else quantized, Fs=fs, NFFT=1024, noverlap=512, cmap='magma')
    plt.title("Quantized Audio Spectrogram (Left Channel)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


def compute_snr(original, quantized):
    noise = original - quantized
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize audio based on psychoacoustic masking model.")
    parser.add_argument("filename", type=str, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="quantized_output.wav", help="Output filename")
    parser.add_argument("--smr", type=float, default=0.0, help="SMR threshold in dB for more aggressive compression")
    parser.add_argument("--visualize", action="store_true", help="Visualize spectrograms before and after quantization")
    args = parser.parse_args()

    quantized_audio, fs = quantize_audio(args.filename, smr_threshold_db=args.smr, visualize=args.visualize)

    print("Loading original audio for SNR computation...")
    original_audio, _ = load_audio(args.filename)
    original_audio = original_audio[:len(quantized_audio)]

    print("Saving quantized audio...")
    sf.write(args.output, quantized_audio, fs)
    print(f"Quantized audio saved to {args.output}")

    print("Computing SNR between original and quantized audio...")
    snr_value = compute_snr(original_audio, quantized_audio)
    print(f"SNR between original and quantized audio: {snr_value:.2f} dB")
