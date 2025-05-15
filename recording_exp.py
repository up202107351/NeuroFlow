import numpy as np
import time # For simulating time if needed, not strictly necessary for offline
# from scipy.signal import welch # If you want to implement Welch manually
try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowFunctions
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("Warning: brainflow library not found. PSD calculation will be disabled.")
    exit()

import matplotlib.pyplot as plt

# --- Configuration ---
# Make these match what you'd use in your main app's backend
FILE_PATH = "eeg_recordings_lsl/eeg_recording_1.csv"  # OR .npy etc. <-- !!! SET THIS !!!
SAMPLING_RATE = 256.0  # Hz <-- !!! SET THIS TO YOUR RECORDING'S SAMPLING RATE !!!
EEG_CHANNEL_INDICES_IN_FILE = [0, 1, 2, 3] # Which columns in your file are the EEG channels
NUM_CHANNELS_USED = len(EEG_CHANNEL_INDICES_IN_FILE)

# Calibration and Windowing
BASELINE_DURATION_SECONDS = 4.0
ANALYSIS_WINDOW_SECONDS = 2.0  # How much data to use for each PSD calculation/classification
WINDOW_SLIDE_SECONDS = 0.5   # How much the analysis window slides forward each step (for overlap)

# BrainFlow PSD Parameters (match your backend)
NFFT = DataFilter.get_next_power_of_two(int(SAMPLING_RATE))
WELCH_NPERSEG = NFFT # Or int(SAMPLING_RATE * ANALYSIS_WINDOW_SECONDS) if window is long enough
WELCH_OVERLAP = WELCH_NPERSEG // 2

# Band definitions
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Classification Threshold Factors (relative to baseline) - TUNE THESE
RELAXATION_ALPHA_INCREASE_FACTOR = 1.2
RELAXATION_AB_RATIO_INCREASE_FACTOR = 1.15
FOCUS_BETA_THETA_RATIO_INCREASE_FACTOR = 1.2
FOCUS_BETA_INCREASE_FACTOR = 1.15
FOCUS_ALPHA_DECREASE_FACTOR = 0.8

# --- Helper Functions (can be adapted from your eeg_backend_processor.py) ---
def calculate_band_powers_for_window(data_window_channel, fs, nfft, nperseg, noverlap):
    """Calculates Theta, Alpha, Beta powers for a single channel window."""
    if not BRAINFLOW_AVAILABLE or data_window_channel.shape[0] < nperseg:
        print(f"Not enough data for PSD: have {data_window_channel.shape[0]}, need {nperseg}")
        return None

    DataFilter.detrend(data_window_channel, DetrendOperations.CONSTANT.value)
    try:
        psd_data = DataFilter.get_psd_welch(data_window_channel, nfft, nperseg,
                                            noverlap, int(fs), WindowFunctions.HANNING.value)
    except Exception as e:
        print(f"Error in get_psd_welch: {e}")
        return None

    theta = DataFilter.get_band_power(psd_data, THETA_BAND[0], THETA_BAND[1])
    alpha = DataFilter.get_band_power(psd_data, ALPHA_BAND[0], ALPHA_BAND[1])
    beta = DataFilter.get_band_power(psd_data, BETA_BAND[0], BETA_BAND[1])
    return {"theta": theta, "alpha": alpha, "beta": beta}

def process_eeg_segment(eeg_segment_all_channels, fs, nfft, nperseg, noverlap):
    """Processes a multi-channel EEG segment to get average band powers."""
    all_channel_powers = []
    for ch_idx in range(eeg_segment_all_channels.shape[0]):
        powers = calculate_band_powers_for_window(eeg_segment_all_channels[ch_idx, :], fs, nfft, nperseg, noverlap)
        if powers:
            all_channel_powers.append(powers)

    if not all_channel_powers or len(all_channel_powers) != eeg_segment_all_channels.shape[0]:
        return None # Failed to get powers for all channels

    # Average across channels (or select specific channels)
    avg_metrics = {
        'theta': np.mean([p['theta'] for p in all_channel_powers]),
        'alpha': np.mean([p['alpha'] for p in all_channel_powers]),
        'beta':  np.mean([p['beta']  for p in all_channel_powers])
    }
    avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
    avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
    return avg_metrics

# --- Main Test Logic ---
def main():
    if not BRAINFLOW_AVAILABLE:
        print("BrainFlow is required for this script. Please install it.")
        return

    # 1. Load EEG Data
    try:
        # Example for CSV: assumes first column is timestamp, then EEG channels
        # Adjust based on your file format
        print(f"Loading data from: {FILE_PATH}")
        raw_data = np.loadtxt(FILE_PATH, delimiter=',', skiprows=1) # Skip header if exists
        # Select only the EEG channels we want to use
        eeg_data = raw_data[:, EEG_CHANNEL_INDICES_IN_FILE].T # Transpose to (channels, samples)
        print(f"EEG data shape: {eeg_data.shape}")
        if eeg_data.shape[1] / SAMPLING_RATE < BASELINE_DURATION_SECONDS + ANALYSIS_WINDOW_SECONDS:
            print("Error: Data is too short for specified baseline and analysis window.")
            return
    except Exception as e:
        print(f"Error loading or processing data file: {e}")
        return

    total_samples = eeg_data.shape[1]
    time_vector = np.arange(total_samples) / SAMPLING_RATE

    # 2. Baseline Calculation Period
    baseline_samples = int(BASELINE_DURATION_SECONDS * SAMPLING_RATE)
    baseline_eeg_segment = eeg_data[:, :baseline_samples]

    print(f"\nCalculating baseline from first {BASELINE_DURATION_SECONDS} seconds ({baseline_samples} samples)...")
    baseline_metrics = process_eeg_segment(baseline_eeg_segment, SAMPLING_RATE, NFFT, WELCH_NPERSEG, WELCH_OVERLAP)

    if baseline_metrics is None:
        print("Failed to calculate baseline metrics.")
        return
    print(f"Baseline Metrics: {baseline_metrics}")

    # 3. Testing/Analysis Period (Sliding Window)
    analysis_window_samples = int(ANALYSIS_WINDOW_SECONDS * SAMPLING_RATE)
    window_slide_samples = int(WINDOW_SLIDE_SECONDS * SAMPLING_RATE)

    results = [] # To store (timestamp, current_metrics, classification)

    # Start processing after the baseline period
    current_sample_start = baseline_samples
    print(f"\nStarting analysis on subsequent data (window: {ANALYSIS_WINDOW_SECONDS}s, slide: {WINDOW_SLIDE_SECONDS}s)...")

    while current_sample_start + analysis_window_samples <= total_samples:
        window_end_sample = current_sample_start + analysis_window_samples
        current_eeg_window = eeg_data[:, current_sample_start:window_end_sample]
        window_center_time = time_vector[current_sample_start + analysis_window_samples // 2]

        print(f"\nProcessing window: {current_sample_start/SAMPLING_RATE:.2f}s - {window_end_sample/SAMPLING_RATE:.2f}s")
        current_metrics = process_eeg_segment(current_eeg_window, SAMPLING_RATE, NFFT, WELCH_NPERSEG, WELCH_OVERLAP)

        if current_metrics:
            # Classification logic (copied and adapted from backend)
            prediction = "Neutral" # Default
            # Relaxation indicators
            alpha_increased = current_metrics['alpha'] > baseline_metrics['alpha'] * RELAXATION_ALPHA_INCREASE_FACTOR
            ab_ratio_increased = current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * RELAXATION_AB_RATIO_INCREASE_FACTOR
            # Focus indicators
            bt_ratio_increased = current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * FOCUS_BETA_THETA_RATIO_INCREASE_FACTOR
            beta_increased = current_metrics['beta'] > baseline_metrics['beta'] * FOCUS_BETA_INCREASE_FACTOR
            alpha_decreased = current_metrics['alpha'] < baseline_metrics['alpha'] * FOCUS_ALPHA_DECREASE_FACTOR

            if alpha_increased and ab_ratio_increased:
                prediction = "Relaxed"
            elif bt_ratio_increased and beta_increased and alpha_decreased:
                prediction = "Focused"
            # Add more nuanced logic if needed based on your goals

            print(f"  Time: {window_center_time:.2f}s, Current A/B: {current_metrics['ab_ratio']:.2f}, B/T: {current_metrics['bt_ratio']:.2f}, Alpha: {current_metrics['alpha']:.2f}, Prediction: {prediction}")
            results.append({
                "time": window_center_time,
                "alpha": current_metrics['alpha'],
                "beta": current_metrics['beta'],
                "theta": current_metrics['theta'],
                "ab_ratio": current_metrics['ab_ratio'],
                "bt_ratio": current_metrics['bt_ratio'],
                "prediction": prediction
            })
        else:
            print(f"  Time: {window_center_time:.2f}s - Failed to process metrics for this window.")

        current_sample_start += window_slide_samples

    # 4. Plotting Results (Optional)
    if results:
        times = [r['time'] for r in results]
        alpha_powers = [r['alpha'] for r in results]
        beta_powers = [r['beta'] for r in results]
        theta_powers = [r['theta'] for r in results]
        ab_ratios = [r['ab_ratio'] for r in results]
        bt_ratios = [r['bt_ratio'] for r in results]
        predictions_text = [r['prediction'] for r in results]

        # Map predictions to numbers for easier plotting overlay
        prediction_map = {"Relaxed": 2, "Focused": 1, "Neutral": 0}
        prediction_values = [prediction_map.get(p, -1) for p in predictions_text]


        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Raw EEG (first channel for example)
        axs[0].plot(time_vector, eeg_data[0, :])
        axs[0].set_title(f"Raw EEG (Channel {EEG_CHANNEL_INDICES_IN_FILE[0]})")
        axs[0].set_ylabel("Amplitude (uV)")
        axs[0].axvspan(0, BASELINE_DURATION_SECONDS, color='gray', alpha=0.3, label='Baseline Period')
        axs[0].legend(loc='upper right')

        # Band Powers
        axs[1].plot(times, alpha_powers, label='Alpha', color='blue')
        axs[1].plot(times, beta_powers, label='Beta', color='red')
        axs[1].plot(times, theta_powers, label='Theta', color='green')
        axs[1].set_title("Band Powers Over Time (Post-Baseline)")
        axs[1].set_ylabel("Power (uV^2/Hz or arb.)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True, linestyle=':')

        # Ratios
        ax_ratios2 = axs[2].twinx() # Second y-axis for B/T ratio
        axs[2].plot(times, ab_ratios, label='Alpha/Beta Ratio', color='purple')
        ax_ratios2.plot(times, bt_ratios, label='Beta/Theta Ratio', color='orange', linestyle='--')
        axs[2].set_title("Key Ratios Over Time")
        axs[2].set_ylabel("Alpha/Beta Ratio", color='purple')
        ax_ratios2.set_ylabel("Beta/Theta Ratio", color='orange')
        axs[2].tick_params(axis='y', labelcolor='purple')
        ax_ratios2.tick_params(axis='y', labelcolor='orange')
        axs[2].legend(loc='upper left')
        ax_ratios2.legend(loc='upper right')
        axs[2].grid(True, linestyle=':')


        # Predictions
        axs[3].plot(times, prediction_values, drawstyle='steps-post', label='Classification', color='black', linewidth=2)
        axs[3].set_yticks(list(prediction_map.values()))
        axs[3].set_yticklabels(list(prediction_map.keys()))
        axs[3].set_title("Classification Over Time")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylim(min(prediction_map.values()) - 0.5, max(prediction_map.values()) + 0.5)
        axs[3].legend(loc='upper right')
        axs[3].grid(True, linestyle=':')


        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()