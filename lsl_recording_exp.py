# live_eeg_with_plotting.py
import time
import numpy as np
import pylsl
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd # For easier data handling in plots later
from scipy.signal import butter, filtfilt, welch
import signal
import os 
import pickle

try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("ERROR: brainflow library not found. This script requires BrainFlow. Please install it (pip install brainflow).")
    exit()

# --- Configuration ---
LSL_STREAM_TYPE = 'EEG'
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX_PULL = 128

EEG_CHANNEL_INDICES = [0, 1, 2, 3] # TP9, AF7, AF8, TP10 for Muse
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 6.0  # How often to give feedback
PSD_WINDOW_SECONDS = 2.0       # Data length for each individual PSD calculation

DEFAULT_SAMPLING_RATE = 256.0

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Feedback Logic Thresholds (RELATIVE TO CALIBRATED BASELINE) - TUNE THESE!
RELAX_ALPHA_INCREASE_FACTOR = 1.20
RELAX_AB_RATIO_INCREASE_FACTOR = 1.15
FOCUS_BT_RATIO_INCREASE_FACTOR = 1.20
FOCUS_BETA_INCREASE_FACTOR = 1.15
FOCUS_ALPHA_DECREASE_FACTOR = 0.85
LESS_RELAXED_AB_RATIO_DECREASE_FACTOR = 0.90 # If A/B drops below 90% of baseline A/B
LESS_FOCUSED_BT_RATIO_DECREASE_FACTOR = 0.90 # If B/T drops below 90% of baseline B/T

# --- Global State ---
running = True
lsl_inlet = None
sampling_rate = DEFAULT_SAMPLING_RATE
nfft = DataFilter.get_nearest_power_of_two(int(DEFAULT_SAMPLING_RATE * PSD_WINDOW_SECONDS))
welch_overlap_samples = nfft // 2
filter_order = 4
lowcut = 0.5
highcut = 30.0
nyq = 0.5 * sampling_rate
low = lowcut / nyq
b_hp, a_hp = butter(filter_order, low, btype='highpass', analog=False)

# Design Butterworth low-pass filter with SciPy
high = highcut / nyq
b_lp, a_lp = butter(filter_order, high, btype='lowpass', analog=False)


baseline_metrics = None
# For plotting at the end:
full_session_eeg_data = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
full_session_eeg_data_raw = np.array([]).reshape(NUM_EEG_CHANNELS, 0) # Raw EEG data
full_session_timestamps_lsl = [] # Store LSL timestamps of first sample in each chunk
feedback_log = [] # List of dicts: {"time_abs": wall_clock, "time_rel": rel_time, "metrics": current_metrics, "prediction": state}

# --- Data Saving Configuration ---
SAVE_DATA = True
SAVE_PATH = "live_session_data/"
# Filenames will be generated in main()
RAW_EEG_FILENAME = ""
METRICS_FILENAME = ""
BASELINE_FILENAME = ""

def save_session_data(full_session_eeg_data, full_session_eeg_data_raw, 
                     full_session_timestamps_lsl, feedback_log, baseline_metrics,
                     save_path="live_session_data/"):
    """
    Save the current session data to disk.
    
    Args:
        full_session_eeg_data: Processed EEG data array
        full_session_eeg_data_raw: Raw EEG data array
        full_session_timestamps_lsl: List of timestamps
        feedback_log: List of feedback dictionaries
        baseline_metrics: Dictionary of baseline metrics
        save_path: Directory to save files
    """
    # Create timestamp for filenames
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save raw and processed EEG data
    np.save(os.path.join(save_path, "eeg_data_raw.npy"), full_session_eeg_data_raw)
    np.save(os.path.join(save_path, "eeg_data_processed.npy"), full_session_eeg_data)
    np.save(os.path.join(save_path, "timestamps.npy"), np.array(full_session_timestamps_lsl))
    
    # Save baseline metrics
    with open(os.path.join(save_path, "baseline_metrics.pkl"), 'wb') as f:
        pickle.dump(baseline_metrics, f)
    
    # Save feedback log to CSV
    if feedback_log:
        # Flatten the metrics dictionary inside feedback_log for CSV export
        feedback_df = pd.DataFrame([
            {
                'time_abs': entry['time_abs'],
                'time_rel': entry['time_rel'],
                'prediction': entry['prediction'],
                'confidence': entry.get('confidence', ''),
                'alpha': entry['metrics']['alpha'],
                'beta': entry['metrics']['beta'],
                'theta': entry['metrics']['theta'],
                'ab_ratio': entry['metrics']['ab_ratio'],
                'bt_ratio': entry['metrics']['bt_ratio']
            } for entry in feedback_log
        ])
        feedback_df.to_csv(os.path.join(save_path, "feedback_log.csv"), index=False)

    if full_session_eeg_data.size == 0 and not feedback_log:
        print("No data recorded to plot.")
        return
    
    print("\n--- Generating Plots ---")
    
    # Ensure timestamps are sensible
    if not full_session_timestamps_lsl or len(full_session_timestamps_lsl) != full_session_eeg_data.shape[1]:
        print("Warning: LSL timestamps inconsistent or missing. Generating approximate time axis for raw EEG.")
        raw_time_axis = np.arange(full_session_eeg_data.shape[1]) / sampling_rate
    else:
        raw_time_axis = np.array(full_session_timestamps_lsl) - full_session_timestamps_lsl[0] # Relative to start

    num_raw_plots = NUM_EEG_CHANNELS
    num_metric_plots = 3 # Powers, Ratios, Predictions

    fig, axs = plt.subplots(num_raw_plots, 1, figsize=(12, 8), sharex=True)
    for i in range(num_raw_plots):
        frequencies, power_spectral_density = welch(full_session_eeg_data_raw[i], fs=256, nperseg=256, noverlap=128) # Adjust nperseg and noverlap as needed
        axs[i].plot(frequencies, 10 * np.log10(power_spectral_density)) # Convert to dB for better visualization
        axs[i].set_title(f'Power Spectrum (Welch) - Channel {i+1} (Before Filtering)')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Power/Frequency (dB/Hz)')
        axs[i].grid(True)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(SAVE_PATH, "power_spectrum_raw.png"), dpi=300)

    fig, axs = plt.subplots(num_raw_plots, 1, figsize=(12, 8), sharex=True)
    for i in range(num_raw_plots):
        frequencies, power_spectral_density = welch(full_session_eeg_data[i], fs=256, nperseg=256, noverlap=128) # Adjust nperseg and noverlap as needed
        axs[i].plot(frequencies, 10 * np.log10(power_spectral_density)) # Convert to dB for better visualization
        axs[i].set_title(f'Power Spectrum (Welch) - Channel {i+1} (After Filtering)')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Power/Frequency (dB/Hz)')
        axs[i].grid(True)


    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(SAVE_PATH, "power_spectrum_filtered.png"), dpi=300)

    fig, axs = plt.subplots(num_raw_plots, 1, figsize=(15, 3 * num_raw_plots), sharex=False)
    current_ax = 0

    # Plot Raw EEG
    for i in range(NUM_EEG_CHANNELS):
        ax = axs[current_ax]
        ax.plot(raw_time_axis, full_session_eeg_data[i, :], label=f'Channel {i+1}')
        ax.set_title(f'Raw EEG Data - Channel {i+1}')
        ax.set_ylabel('Amplitude (uV)')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5, label='Calibration')
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        current_ax += 1

    

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(SAVE_PATH, "eeg.png"), dpi=300)

    if current_ax > 0: axs[current_ax-1].set_xlabel('Time (s)')

    fig, axs = plt.subplots(num_metric_plots, 1, figsize=(15, 3 * num_metric_plots), sharex=False)
    current_ax = 0
    if feedback_log:
        metric_times = np.array([entry['time_rel'] for entry in feedback_log]) + CALIBRATION_DURATION_SECONDS # Align with raw EEG time
        
        # Band Powers
        ax = axs[current_ax]
        ax.plot(metric_times, [m['metrics']['alpha'] for m in feedback_log], label='Alpha', c='blue')
        ax.plot(metric_times, [m['metrics']['beta'] for m in feedback_log], label='Beta', c='red')
        ax.plot(metric_times, [m['metrics']['theta'] for m in feedback_log], label='Theta', c='green')
        if baseline_metrics:
            ax.axhline(baseline_metrics['alpha'],c='blue',ls='--',alpha=0.7, label='Alpha Base')
            ax.axhline(baseline_metrics['beta'],c='red',ls='--',alpha=0.7, label='Beta Base')
            ax.axhline(baseline_metrics['theta'],c='green',ls='--',alpha=0.7, label='Theta Base')
        ax.set_title('Band Powers (During Feedback Phase)')
        ax.set_ylabel('Power')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5) # Show calibration on this x-axis too
        ax.legend(); ax.grid(True, linestyle=':')
        current_ax+=1

        # Ratios
        ax = axs[current_ax]
        ax_twin = ax.twinx()
        ax.plot(metric_times, [m['metrics']['ab_ratio'] for m in feedback_log], label='A/B Ratio', c='purple')
        if baseline_metrics: ax.axhline(baseline_metrics['ab_ratio'],c='purple',ls='--',alpha=0.7, label='A/B Base')
        ax_twin.plot(metric_times, [m['metrics']['bt_ratio'] for m in feedback_log], label='B/T Ratio', c='orange', linestyle='--')
        if baseline_metrics: ax_twin.axhline(baseline_metrics['bt_ratio'],c='orange',ls=':',alpha=0.7, label='B/T Base')
        ax.set_title('Ratios (During Feedback Phase)')
        ax.set_ylabel('A/B Ratio', color='purple'); ax.tick_params(axis='y', labelcolor='purple')
        ax_twin.set_ylabel('B/T Ratio', color='orange'); ax_twin.tick_params(axis='y', labelcolor='orange')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
        ax.grid(True, linestyle=':')
        current_ax+=1

        # Predictions
        ax = axs[current_ax]
        pred_map_final = {
            "Neutral": 0,

            "Slightly Relaxed": 1,
            "Moderately Relaxed": 2,
            "Strongly Relaxed": 3,
            "Deeply Relaxed": 4, # Added from example rules

            "Slightly Alert / Less Relaxed": -1,
            "Moderately Alert / Less Relaxed": -2,

            "Slightly Focused": 5,
            "Moderately Focused": 6,
            "Strongly Focused": 7,
            "Highly Focused": 8, # Added from example rules

            "Slightly Distracted / Less Focused": -5,
            "Moderately Distracted / Less Focused": -6,

            "Slightly Drowsy": -10,
            "Moderately Drowsy": -11,
            "Drowsy": -11, # Alias

            "Internal Mental Activity": 9, # Could be positive or neutral depending on goal

            "Unknown": -99 # For any unmapped states
        }
        all_predictions_in_log = sorted(list(set(entry['prediction'] for entry in feedback_log)))
        for p_text in all_predictions_in_log:
            if p_text not in pred_map_final:
                print(f"Warning: Prediction '{p_text}' not found in pred_map_final. Assigning to 'Unknown'.")

        pred_values = [pred_map_final.get(entry['prediction'], pred_map_final["Unknown"]) for entry in feedback_log]

        ax.plot(metric_times, pred_values, drawstyle='steps-post', label='State', c='k')

        # For Y-axis ticks, dynamically use only the states that actually occurred
        # or a representative subset to keep it readable.
        unique_pred_texts_in_log = sorted(list(set(entry['prediction'] for entry in feedback_log)))
        used_ticks_values = sorted(list(set(pred_map_final.get(p, pred_map_final["Unknown"]) for p in unique_pred_texts_in_log)))

        # Filter labels to match used_ticks_values to avoid plotting labels for unused numerical values
        final_yticklabels = []
        for tick_val in used_ticks_values:
            found = False
            for text, val in pred_map_final.items():
                if val == tick_val and text in unique_pred_texts_in_log: # Ensure the text was actually predicted
                    final_yticklabels.append(text)
                    found = True
                    break
            if not found: # Fallback if a numeric value doesn't map back to a predicted text (shouldn't happen)
                final_yticklabels.append(f"Val: {tick_val}")


        if not used_ticks_values: # Handle case of no predictions logged
            used_ticks_values = [pred_map_final["Neutral"]]
            final_yticklabels = ["Neutral"]

        ax.set_yticks(used_ticks_values)
        ax.set_yticklabels(final_yticklabels, fontsize='small', rotation=0) # Adjust rotation if labels overlap
        ax.set_title('Predicted State (During Feedback Phase)')
        ax.set_ylabel('State')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5)
        ax.legend(); ax.grid(True, linestyle=':')
        current_ax+=1

    # Align X-axis for all plots
    max_time_overall = raw_time_axis[-1] if full_session_eeg_data.size > 0 else 0
    if feedback_log : max_time_overall = max(max_time_overall, metric_times[-1])

    for i in range(current_ax): # Iterate only up to plots actually made
        axs[i].set_xlim(0, max_time_overall + 1)
        if i < current_ax -1 : plt.setp(axs[i].get_xticklabels(), visible=False) # Hide x-labels for all but bottom

    if current_ax > 0 : axs[current_ax-1].set_xlabel('Time (s from start of recording)')


    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"band_powers_{timestamp_str}.png"), dpi=300)
    plt.show()
    
    print(f"\nSession data saved to: {save_path} at {datetime.now().strftime('%H:%M:%S')}")

def graceful_signal_handler(sig, frame):
    global running
    if running: # Prevent multiple calls if signal comes rapidly
        print(f'\nSignal {sig} (Ctrl+C) received. Finishing current operations and generating plots/saving data...')
        running = False # Signal loops to stop

signal.signal(signal.SIGINT, graceful_signal_handler)

# --- Helper Functions ---
def connect_to_lsl():
    global lsl_inlet, sampling_rate, nfft, welch_overlap_samples
    print(f"Looking for LSL stream (Type: '{LSL_STREAM_TYPE}')...")
    try:
        streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
        if not streams:
            print("LSL stream not found."); return False
        lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
        info = lsl_inlet.info()
        lsl_sr = info.nominal_srate()
        sampling_rate = lsl_sr if lsl_sr > 0 else DEFAULT_SAMPLING_RATE
        nfft = DataFilter.get_nearest_power_of_two(int(sampling_rate * PSD_WINDOW_SECONDS))
        welch_overlap_samples = nfft // 2
        print(f"Connected to '{info.name()}' @ {sampling_rate:.2f} Hz. NFFT={nfft}, Overlap={welch_overlap_samples}")
        if info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
            print(f"ERROR: LSL stream has insufficient channels."); return False
        return True
    except Exception as e:
        print(f"Error connecting to LSL: {e}"); return False

def calculate_band_powers_for_segment(eeg_segment_all_channels):
    global sampling_rate, nfft, welch_overlap_samples
    if eeg_segment_all_channels.shape[1] < nfft: return None
    metrics_list = []
    for ch_idx in range(NUM_EEG_CHANNELS):
        ch_data = eeg_segment_all_channels[ch_idx, :].copy()
        DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
        try:
            psd = DataFilter.get_psd_welch(ch_data, nfft, welch_overlap_samples, int(sampling_rate), WindowOperations.HANNING.value)
            metrics_list.append({
                'theta': DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1]),
                'alpha': DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1]),
                'beta': DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])})
        except Exception: return None # PSD failed for a channel
    if len(metrics_list) != NUM_EEG_CHANNELS: return None
    avg_metrics = {
        'theta': np.mean([m['theta'] for m in metrics_list]),
        'alpha': np.mean([m['alpha'] for m in metrics_list]),
        'beta': np.mean([m['beta'] for m in metrics_list])}
    avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
    avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
    return avg_metrics

def detect_eyes_closed(metrics, baseline_alpha, baseline_ab_ratio):
    """
    Eyes closed typically shows:
    1. Very strong alpha increase (often >100%)
    2. Alpha increase without corresponding theta increase
    3. Very sharp onset compared to gradual relaxation
    """
    is_likely_eyes_closed = (
        metrics['alpha'] > baseline_alpha * 2.0 and  # Alpha doubled
        metrics['ab_ratio'] > baseline_ab_ratio * 1.5 and  # A/B ratio increased 50%+
        metrics['theta'] < metrics['theta'] * 1.2  # Theta didn't increase significantly
    )
    
    return is_likely_eyes_closed

def collect_eeg_and_process_segments(duration_seconds, is_calibration_phase=False):
    global full_session_eeg_data, full_session_eeg_data_raw, full_session_timestamps_lsl, running
    
    collection_end_time = time.time() + duration_seconds
    collected_metrics_list = []
    local_eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
    samples_for_psd_window = int(PSD_WINDOW_SECONDS * sampling_rate)
    last_print_time = time.time()

    print(f"  Collecting data for {duration_seconds:.0f}s...")
    while time.time() < collection_end_time and running:
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.5, max_samples=LSL_CHUNK_MAX_PULL)
        if chunk:
            chunk_np = np.array(chunk, dtype=np.float64).T
            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
            eeg_chunk_lp = np.zeros_like(eeg_chunk)
            eeg_chunk_lhp = np.zeros_like(eeg_chunk)
            for i in range(NUM_EEG_CHANNELS):
                eeg_chunk_lp[i] = filtfilt(b_lp, a_lp, eeg_chunk[i])
                eeg_chunk_lhp[i] = filtfilt(b_hp, a_hp, eeg_chunk_lp[i])

            # Correct frontal channel (if needed)
            #eeg_chunk_lhp = correct_frontal_channel(eeg_chunk_lhp, problem_ch=2, reference_ch=1)
            
            # Store data
            full_session_eeg_data = np.append(full_session_eeg_data, eeg_chunk_lhp, axis=1)
            full_session_eeg_data_raw = np.append(full_session_eeg_data, eeg_chunk, axis=1)

            if timestamps:
                full_session_timestamps_lsl.extend(timestamps) # Store all LSL timestamps if needed for precise alignment
            else: # Fallback if LSL timestamps are missing from pull_chunk for some reason
                now = time.time()
                num_samples_in_chunk = eeg_chunk.shape[1]
                # Create synthetic timestamps based on wall clock and sampling rate
                full_session_timestamps_lsl.extend([now - (num_samples_in_chunk - 1 - i) / sampling_rate for i in range(num_samples_in_chunk)])


            local_eeg_buffer = np.append(local_eeg_buffer, eeg_chunk_lhp, axis=1)
            while local_eeg_buffer.shape[1] >= samples_for_psd_window:
                segment = local_eeg_buffer[:, :samples_for_psd_window]
                local_eeg_buffer = local_eeg_buffer[:, int(samples_for_psd_window * 0.5):] # 50% slide
                
                metrics = calculate_band_powers_for_segment(segment)
                if metrics:
                    collected_metrics_list.append(metrics)
                    if is_calibration_phase and time.time() - last_print_time > 2:
                        print(f"    Calibrating... Alpha:{metrics['alpha']:.1f} (Time left: {collection_end_time - time.time():.0f}s)")
                        last_print_time = time.time()
        elif not running: break
        else: time.sleep(0.01) # No data, brief pause
    
    if not collected_metrics_list:
        print(f"  Warning: No metrics calculated during this {duration_seconds}s period.")
        return None
    
    # Average all metrics collected during this period
    avg_period_metrics = {
        'alpha': np.mean([m['alpha'] for m in collected_metrics_list]),
        'beta':  np.mean([m['beta']  for m in collected_metrics_list]),
        'theta': np.mean([m['theta'] for m in collected_metrics_list])}
    avg_period_metrics['ab_ratio'] = avg_period_metrics['alpha'] / avg_period_metrics['beta'] if avg_period_metrics['beta'] > 1e-9 else 0
    avg_period_metrics['bt_ratio'] = avg_period_metrics['beta'] / avg_period_metrics['theta'] if avg_period_metrics['theta'] > 1e-9 else 0
    return avg_period_metrics

def perform_calibration_phase():
    global baseline_metrics
    print(f"\n--- Starting {CALIBRATION_DURATION_SECONDS:.0f} Second Calibration ---")
    print("Please remain in a neutral, resting state.")
    baseline_metrics = collect_eeg_and_process_segments(CALIBRATION_DURATION_SECONDS, is_calibration_phase=True)
    if baseline_metrics:
        print("\n--- Calibration Complete ---")
        for key, val in baseline_metrics.items(): print(f"Baseline {key.replace('_', ' ').title()}: {val:.2f}")
        return True
    else:
        print("Calibration failed: No metrics collected. Check LSL stream.")
        return False

# For channel 3 specifically:
def correct_frontal_channel(eeg_data, problem_ch=2, reference_ch=1):
    """Simple correction for frontal channels using symmetry"""
    corrected_data = eeg_data.copy()
    
    # Get amplitude difference between channels
    ch_diff = np.abs(eeg_data[problem_ch]) - np.abs(eeg_data[reference_ch])
    
    # Identify extreme differences (likely artifacts)
    extreme_diffs = ch_diff > 100  # Threshold in μV
    
    # Replace extreme values with estimates based on opposite channel
    # (preserving polarity but reducing amplitude)
    corrected_data[problem_ch, extreme_diffs] = (
        np.sign(eeg_data[problem_ch, extreme_diffs]) * 
        np.abs(eeg_data[reference_ch, extreme_diffs])
    )
    
    return corrected_data

def reject_artifacts(eeg_data, channel_thresholds=[150, 100, 100, 150]):
    """Flag samples with amplitudes exceeding thresholds"""
    # Check each channel against its threshold
    exceed_threshold = np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1)
    
    # Create a mask of "good" data points
    good_samples = ~np.any(exceed_threshold, axis=0)
    
    return good_samples

def feedback_loop():
    global baseline_metrics, running, feedback_log
    if not baseline_metrics: print("Cannot start feedback loop: baseline not calibrated."); return

    print(f"\n--- Starting Real-time Feedback (updates every {ANALYSIS_WINDOW_SECONDS:.0f}s) ---")
    session_start_time_abs = time.time() # Absolute start time of feedback phase
    total_loop_times = 0
    last_save_time = time.time()  # Initialize last save time
    save_interval = 120  # Save every 2 minutes (120 seconds)

    while running:
        loop_start_time = time.time()
        current_metrics = collect_eeg_and_process_segments(ANALYSIS_WINDOW_SECONDS, is_calibration_phase=False)
        
        if not current_metrics:
            print("  Skipping feedback: failed to get metrics for current window.")
            # Ensure we still wait for the full interval duration
            elapsed_in_loop = time.time() - loop_start_time
            wait_time = ANALYSIS_WINDOW_SECONDS - elapsed_in_loop
            if wait_time > 0 and running: time.sleep(wait_time)
            continue # Try next analysis window

        # --- Define Levels of Change from Baseline ---
        # Alpha Changes
        alpha_slight_incr = current_metrics['alpha'] > baseline_metrics['alpha'] * 1.10
        alpha_mod_incr = current_metrics['alpha'] > baseline_metrics['alpha'] * 1.25 # Was RELAX_ALPHA_INCREASE_FACTOR
        alpha_strong_incr = current_metrics['alpha'] > baseline_metrics['alpha'] * 1.50

        alpha_slight_decr = current_metrics['alpha'] < baseline_metrics['alpha'] * 0.90
        alpha_mod_decr = current_metrics['alpha'] < baseline_metrics['alpha'] * 0.80 # Was FOCUS_ALPHA_DECREASE_FACTOR
        alpha_strong_decr = current_metrics['alpha'] < baseline_metrics['alpha'] * 0.65

        # Beta Changes
        beta_slight_incr = current_metrics['beta'] > baseline_metrics['beta'] * 1.10
        beta_mod_incr = current_metrics['beta'] > baseline_metrics['beta'] * 1.20 # Was FOCUS_BETA_INCREASE_FACTOR
        beta_strong_incr = current_metrics['beta'] > baseline_metrics['beta'] * 1.40

        beta_slight_decr = current_metrics['beta'] < baseline_metrics['beta'] * 0.90
        beta_mod_decr = current_metrics['beta'] < baseline_metrics['beta'] * 0.80

        # Theta Changes
        theta_slight_incr = current_metrics['theta'] > baseline_metrics['theta'] * 1.15
        theta_mod_incr = current_metrics['theta'] > baseline_metrics['theta'] * 1.30

        theta_slight_decr = current_metrics['theta'] < baseline_metrics['theta'] * 0.85
        theta_mod_decr = current_metrics['theta'] < baseline_metrics['theta'] * 0.70

        # Alpha/Beta Ratio Changes
        ab_ratio_slight_incr = current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * 1.10
        ab_ratio_mod_incr = current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * 1.20 # Was RELAX_AB_RATIO_INCREASE_FACTOR
        ab_ratio_strong_incr = current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * 1.40

        ab_ratio_slight_decr = current_metrics['ab_ratio'] < baseline_metrics['ab_ratio'] * 0.90 # Was LESS_RELAXED_AB_RATIO_DECREASE_FACTOR
        ab_ratio_mod_decr = current_metrics['ab_ratio'] < baseline_metrics['ab_ratio'] * 0.75

        # Beta/Theta Ratio Changes
        bt_ratio_slight_incr = current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * 1.15
        bt_ratio_mod_incr = current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * 1.30 # Was FOCUS_BT_RATIO_INCREASE_FACTOR
        bt_ratio_strong_incr = current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * 1.60

        bt_ratio_slight_decr = current_metrics['bt_ratio'] < baseline_metrics['bt_ratio'] * 0.85 # Was LESS_FOCUSED_BT_RATIO_DECREASE_FACTOR
        bt_ratio_mod_decr = current_metrics['bt_ratio'] < baseline_metrics['bt_ratio'] * 0.70


       # --- Classification Logic using the flags ---
        state_base = "Neutral" # The core state
        confidence = ""      # "Slightly", "Moderately", "Strongly", or ""

        # **Order matters: more specific/stronger conditions first**

        # **Primary Focus Indicators**
        if beta_strong_incr and bt_ratio_strong_incr and alpha_mod_decr:
            state_base = "Focused"
            confidence = "Strongly"
        elif beta_mod_incr and bt_ratio_mod_incr and (alpha_slight_decr or alpha_mod_decr):
            state_base = "Focused"
            confidence = "Moderately"
        elif (beta_slight_incr and bt_ratio_slight_incr) and not alpha_mod_incr: # Alpha not also significantly up
            state_base = "Focused" # Or "Mentally Active" if preferred
            confidence = "Slightly"

        # **Primary Relaxation Indicators** (Can override slight/moderate focus if relaxation is strong)
        if alpha_strong_incr and ab_ratio_strong_incr:
            state_base = "Relaxed" # Override previous if this is stronger
            confidence = "Strongly"
        elif alpha_mod_incr and ab_ratio_mod_incr and not beta_mod_incr:
            # If not already strongly focused, then relaxed
            if not (state_base == "Focused" and confidence == "Strongly"):
                state_base = "Relaxed"
                confidence = "Moderately"
        elif (alpha_slight_incr and ab_ratio_slight_incr) or (alpha_mod_incr and not beta_slight_incr):
            # If not already focused or moderately/strongly relaxed
            if not (state_base.endswith("Focused") or (state_base == "Relaxed" and confidence in ["Moderately", "Strongly"])):
                state_base = "Relaxed"
                confidence = "Slightly"


        # **Handle "Less Relaxed" / "More Alert" states**
        # This should typically modify "Neutral" or a "Relaxed" state if a clear shift occurs
        if state_base == "Neutral" or state_base.endswith("Relaxed"): # Only if current base is neutral or relaxed
            if ab_ratio_mod_decr or (alpha_mod_decr and beta_slight_incr):
                state_base = "Alert / Less Relaxed" # Merged term
                confidence = "Moderately"
            elif ab_ratio_slight_decr or (alpha_slight_decr and not beta_strong_incr): # Avoid if strong Beta (could be focus)
                state_base = "Alert / Less Relaxed"
                confidence = "Slightly"

        # **Handle "Less Focused" / "Distracted" states**
        if state_base == "Neutral" or state_base.endswith("Focused"): # Only if current base is neutral or focused
            if bt_ratio_mod_decr or (beta_mod_decr and alpha_slight_incr):
                state_base = "Distracted / Less Focused"
                confidence = "Moderately"
            elif bt_ratio_slight_decr or (beta_slight_decr and not alpha_strong_incr): # Avoid if strong alpha
                state_base = "Distracted / Less Focused"
                confidence = "Slightly"

        # **Consider "Drowsy/Inattentive" (High Theta, Low Beta)**
        if theta_mod_incr and beta_mod_decr and not alpha_strong_incr:
            state_base = "Drowsy"
            confidence = "Moderately"
        elif theta_slight_incr and beta_slight_decr and not alpha_mod_incr:
            state_base = "Drowsy"
            confidence = "Slightly"

        # **Refine "Internal Focus" if Beta high but Alpha also high (or not suppressed)**
        if (state_base == "Focused" or state_base == "Mentally Active") and \
        (alpha_slight_incr or (alpha_mod_incr and confidence != "Strongly")): # if alpha is up and focus isn't already "Strongly Focused"
            state_base = "Internal Mental Activity" # More general term
            # Confidence level might already be set by the focus rule

        if detect_eyes_closed(current_metrics, baseline_metrics['alpha'], baseline_metrics['ab_ratio']):
            state_base = "Eyes Closed (Alpha)"
            confidence = "High"
        # --- Construct final state description ---
        if confidence and state_base != "Neutral": # Don't add confidence to "Neutral" unless you want "Slightly Neutral" etc.
            final_state_description = f"{confidence} {state_base}"
        else:
            final_state_description = state_base

        # This `final_state_description` is what you print and log
        state = final_state_description # Update the 'state' variable for logging

        time_rel_feedback = time.time() - session_start_time_abs
        print(f"\nFeedback @ {time_rel_feedback:6.1f}s ({time.strftime('%H:%M:%S')}): {state}")
        print(f"  Metrics: A/B {current_metrics['ab_ratio']:.2f}(B:{baseline_metrics['ab_ratio']:.2f}), "
              f"B/T {current_metrics['bt_ratio']:.2f}(B:{baseline_metrics['bt_ratio']:.2f}), "
              f"Alpha {current_metrics['alpha']:.1f}(B:{baseline_metrics['alpha']:.1f})")
        
        feedback_log.append({
            "time_abs": time.time(), "time_rel": time_rel_feedback,
            "metrics": current_metrics, "prediction": state, "confidence": confidence})
        
        # Check if it's time to save data (every 2 minutes)
        current_time = time.time()
        if current_time - last_save_time >= save_interval and SAVE_DATA:
            print("\n--- Saving session data (2-minute interval) ---")
            try:
                save_session_data(
                    full_session_eeg_data, 
                    full_session_eeg_data_raw, 
                    full_session_timestamps_lsl,
                    feedback_log, 
                    baseline_metrics,
                    SAVE_PATH
                )
                last_save_time = current_time
            except Exception as e:
                print(f"Error saving data: {e}")
        
        elapsed_in_loop = time.time() - loop_start_time
        wait_time = ANALYSIS_WINDOW_SECONDS - elapsed_in_loop
        total_loop_times = total_loop_times + elapsed_in_loop
        if wait_time > 0 and running:
            time.sleep(wait_time)
        if CALIBRATION_DURATION_SECONDS + total_loop_times > 40:
            print(f"  Total loops time: {total_loop_times:.2f}s")
            total_loop_times = 0
            plot_results() # Plot every 70 seconds to avoid too many plots
            print("  Waiting for next analysis window...")

def plot_results():
    global full_session_eeg_data, full_session_timestamps_lsl, feedback_log, baseline_metrics, sampling_rate

    if full_session_eeg_data.size == 0 and not feedback_log:
        print("No data recorded to plot.")
        return
    
    print("\n--- Generating Plots ---")
    
    # Ensure timestamps are sensible
    if not full_session_timestamps_lsl or len(full_session_timestamps_lsl) != full_session_eeg_data.shape[1]:
        print("Warning: LSL timestamps inconsistent or missing. Generating approximate time axis for raw EEG.")
        raw_time_axis = np.arange(full_session_eeg_data.shape[1]) / sampling_rate
    else:
        raw_time_axis = np.array(full_session_timestamps_lsl) - full_session_timestamps_lsl[0] # Relative to start

    num_raw_plots = NUM_EEG_CHANNELS
    num_metric_plots = 3 # Powers, Ratios, Predictions

    fig, axs = plt.subplots(num_raw_plots, 1, figsize=(12, 8), sharex=True)
    for i in range(num_raw_plots):
        frequencies, power_spectral_density = welch(full_session_eeg_data_raw[i], fs=256, nperseg=256, noverlap=128) # Adjust nperseg and noverlap as needed
        axs[i].plot(frequencies, 10 * np.log10(power_spectral_density)) # Convert to dB for better visualization
        axs[i].set_title(f'Power Spectrum (Welch) - Channel {i+1} (Before Filtering)')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Power/Frequency (dB/Hz)')
        axs[i].grid(True)

    fig, axs = plt.subplots(num_raw_plots, 1, figsize=(12, 8), sharex=True)
    for i in range(num_raw_plots):
        frequencies, power_spectral_density = welch(full_session_eeg_data[i], fs=256, nperseg=256, noverlap=128) # Adjust nperseg and noverlap as needed
        axs[i].plot(frequencies, 10 * np.log10(power_spectral_density)) # Convert to dB for better visualization
        axs[i].set_title(f'Power Spectrum (Welch) - Channel {i+1} (After Filtering)')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Power/Frequency (dB/Hz)')
        axs[i].grid(True)

    fig, axs = plt.subplots(num_raw_plots, 1, figsize=(15, 3 * num_raw_plots), sharex=False)
    current_ax = 0

    # Plot Raw EEG
    for i in range(NUM_EEG_CHANNELS):
        ax = axs[current_ax]
        ax.plot(raw_time_axis, full_session_eeg_data[i, :], label=f'Channel {i+1}')
        ax.set_title(f'Raw EEG Data - Channel {i+1}')
        ax.set_ylabel('Amplitude (uV)')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5, label='Calibration')
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        current_ax += 1
    if current_ax > 0: axs[current_ax-1].set_xlabel('Time (s)')

    fig, axs = plt.subplots(num_metric_plots, 1, figsize=(15, 3 * num_metric_plots), sharex=False)
    current_ax = 0
    if feedback_log:
        metric_times = np.array([entry['time_rel'] for entry in feedback_log]) + CALIBRATION_DURATION_SECONDS # Align with raw EEG time
        
        # Band Powers
        ax = axs[current_ax]
        ax.plot(metric_times, [m['metrics']['alpha'] for m in feedback_log], label='Alpha', c='blue')
        ax.plot(metric_times, [m['metrics']['beta'] for m in feedback_log], label='Beta', c='red')
        ax.plot(metric_times, [m['metrics']['theta'] for m in feedback_log], label='Theta', c='green')
        if baseline_metrics:
            ax.axhline(baseline_metrics['alpha'],c='blue',ls='--',alpha=0.7, label='Alpha Base')
            ax.axhline(baseline_metrics['beta'],c='red',ls='--',alpha=0.7, label='Beta Base')
            ax.axhline(baseline_metrics['theta'],c='green',ls='--',alpha=0.7, label='Theta Base')
        ax.set_title('Band Powers (During Feedback Phase)')
        ax.set_ylabel('Power')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5) # Show calibration on this x-axis too
        ax.legend(); ax.grid(True, linestyle=':')
        current_ax+=1

        # Ratios
        ax = axs[current_ax]
        ax_twin = ax.twinx()
        ax.plot(metric_times, [m['metrics']['ab_ratio'] for m in feedback_log], label='A/B Ratio', c='purple')
        if baseline_metrics: ax.axhline(baseline_metrics['ab_ratio'],c='purple',ls='--',alpha=0.7, label='A/B Base')
        ax_twin.plot(metric_times, [m['metrics']['bt_ratio'] for m in feedback_log], label='B/T Ratio', c='orange', linestyle='--')
        if baseline_metrics: ax_twin.axhline(baseline_metrics['bt_ratio'],c='orange',ls=':',alpha=0.7, label='B/T Base')
        ax.set_title('Ratios (During Feedback Phase)')
        ax.set_ylabel('A/B Ratio', color='purple'); ax.tick_params(axis='y', labelcolor='purple')
        ax_twin.set_ylabel('B/T Ratio', color='orange'); ax_twin.tick_params(axis='y', labelcolor='orange')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
        ax.grid(True, linestyle=':')
        current_ax+=1

        # Predictions
        ax = axs[current_ax]
        pred_map_final = {
            "Neutral": 0,

            "Slightly Relaxed": 1,
            "Moderately Relaxed": 2,
            "Strongly Relaxed": 3,
            "Deeply Relaxed": 4, # Added from example rules

            "Slightly Alert / Less Relaxed": -1,
            "Moderately Alert / Less Relaxed": -2,

            "Slightly Focused": 5,
            "Moderately Focused": 6,
            "Strongly Focused": 7,
            "Highly Focused": 8, # Added from example rules

            "Slightly Distracted / Less Focused": -5,
            "Moderately Distracted / Less Focused": -6,

            "Slightly Drowsy": -10,
            "Moderately Drowsy": -11,
            "Drowsy": -11, # Alias

            "Internal Mental Activity": 9, # Could be positive or neutral depending on goal

            "Unknown": -99 # For any unmapped states
        }
        all_predictions_in_log = sorted(list(set(entry['prediction'] for entry in feedback_log)))
        for p_text in all_predictions_in_log:
            if p_text not in pred_map_final:
                print(f"Warning: Prediction '{p_text}' not found in pred_map_final. Assigning to 'Unknown'.")

        pred_values = [pred_map_final.get(entry['prediction'], pred_map_final["Unknown"]) for entry in feedback_log]

        ax.plot(metric_times, pred_values, drawstyle='steps-post', label='State', c='k')

        # For Y-axis ticks, dynamically use only the states that actually occurred
        # or a representative subset to keep it readable.
        unique_pred_texts_in_log = sorted(list(set(entry['prediction'] for entry in feedback_log)))
        used_ticks_values = sorted(list(set(pred_map_final.get(p, pred_map_final["Unknown"]) for p in unique_pred_texts_in_log)))

        # Filter labels to match used_ticks_values to avoid plotting labels for unused numerical values
        final_yticklabels = []
        for tick_val in used_ticks_values:
            found = False
            for text, val in pred_map_final.items():
                if val == tick_val and text in unique_pred_texts_in_log: # Ensure the text was actually predicted
                    final_yticklabels.append(text)
                    found = True
                    break
            if not found: # Fallback if a numeric value doesn't map back to a predicted text (shouldn't happen)
                final_yticklabels.append(f"Val: {tick_val}")


        if not used_ticks_values: # Handle case of no predictions logged
            used_ticks_values = [pred_map_final["Neutral"]]
            final_yticklabels = ["Neutral"]

        ax.set_yticks(used_ticks_values)
        ax.set_yticklabels(final_yticklabels, fontsize='small', rotation=0) # Adjust rotation if labels overlap
        ax.set_title('Predicted State (During Feedback Phase)')
        ax.set_ylabel('State')
        ax.axvspan(0, CALIBRATION_DURATION_SECONDS, color='lightgray', alpha=0.5)
        ax.legend(); ax.grid(True, linestyle=':')
        current_ax+=1

    # Align X-axis for all plots
    max_time_overall = raw_time_axis[-1] if full_session_eeg_data.size > 0 else 0
    if feedback_log : max_time_overall = max(max_time_overall, metric_times[-1])

    for i in range(current_ax): # Iterate only up to plots actually made
        axs[i].set_xlim(0, max_time_overall + 1)
        if i < current_ax -1 : plt.setp(axs[i].get_xticklabels(), visible=False) # Hide x-labels for all but bottom

    if current_ax > 0 : axs[current_ax-1].set_xlabel('Time (s from start of recording)')


    plt.tight_layout()
    plt.show()


def main():
    global running, lsl_inlet # Make sure lsl_inlet is global if accessed in finally
    if not BRAINFLOW_AVAILABLE: return
    if not connect_to_lsl(): return
    if not perform_calibration_phase():
        if lsl_inlet: lsl_inlet.close_stream()
        return

    try:
        feedback_loop()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Session ending.")
    except pylsl.LostError:
        print("\nLSL connection lost during session. Session ending.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}. Session ending.")
    finally:
        running = False # Ensure all loops stop
        if lsl_inlet:
            print("Closing LSL stream...")
            lsl_inlet.close_stream()
        # Save final data
        if SAVE_DATA:
            print("\n--- Saving final session data ---")
            try:
                save_session_data(
                    full_session_eeg_data, 
                    full_session_eeg_data_raw, 
                    full_session_timestamps_lsl,
                    feedback_log, 
                    baseline_metrics,
                    SAVE_PATH
                )
            except Exception as e:
                print(f"Error saving final data: {e}")
        print("Session ended.")
        plot_results() # Plot everything at the very end

if __name__ == "__main__":

    main()