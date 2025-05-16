# live_eeg_with_plotting.py
import time
import numpy as np
import pylsl
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd # For easier data handling in plots later
from scipy.signal import butter, filtfilt
import signal
import os 

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
full_session_timestamps_lsl = [] # Store LSL timestamps of first sample in each chunk
feedback_log = [] # List of dicts: {"time_abs": wall_clock, "time_rel": rel_time, "metrics": current_metrics, "prediction": state}

# --- Data Saving Configuration ---
SAVE_DATA = True
SAVE_PATH = "live_session_data/"
# Filenames will be generated in main()
RAW_EEG_FILENAME = ""
METRICS_FILENAME = ""
BASELINE_FILENAME = ""

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

def collect_eeg_and_process_segments(duration_seconds, is_calibration_phase=False):
    global full_session_eeg_data, full_session_timestamps_lsl, running
    
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
            
            # Store raw data
            full_session_eeg_data = np.append(full_session_eeg_data, eeg_chunk_lhp, axis=1)
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

def feedback_loop():
    global baseline_metrics, running, feedback_log
    if not baseline_metrics: print("Cannot start feedback loop: baseline not calibrated."); return

    print(f"\n--- Starting Real-time Feedback (updates every {ANALYSIS_WINDOW_SECONDS:.0f}s) ---")
    session_start_time_abs = time.time() # Absolute start time of feedback phase

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

        state = "Neutral"
        # Relaxation checks
        if current_metrics['alpha'] > baseline_metrics['alpha'] * RELAX_ALPHA_INCREASE_FACTOR and \
           current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * RELAX_AB_RATIO_INCREASE_FACTOR:
            state = "Relaxed"
        elif current_metrics['alpha'] > baseline_metrics['alpha'] * ((RELAX_ALPHA_INCREASE_FACTOR + 1.0) / 2) or \
             current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * ((RELAX_AB_RATIO_INCREASE_FACTOR + 1.0) / 2):
            state = "Getting Relaxed"
        elif current_metrics['ab_ratio'] < baseline_metrics['ab_ratio'] * LESS_RELAXED_AB_RATIO_DECREASE_FACTOR:
            state = "Less Relaxed"

        # Focus checks (can potentially override relaxation if strong signals, or be combined)
        if current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * FOCUS_BT_RATIO_INCREASE_FACTOR and \
           current_metrics['beta'] > baseline_metrics['beta'] * FOCUS_BETA_INCREASE_FACTOR and \
           current_metrics['alpha'] < baseline_metrics['alpha'] * FOCUS_ALPHA_DECREASE_FACTOR:
            state = "Focused" # Strong focus
        elif current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * ((FOCUS_BT_RATIO_INCREASE_FACTOR + 1.0)/2) and \
             current_metrics['beta'] > baseline_metrics['beta'] * ((FOCUS_BETA_INCREASE_FACTOR + 1.0)/2):
            if state == "Relaxed" or state == "Getting Relaxed": state += " & Likely Focused" # Could be calm focus
            else: state = "Likely Focused"
        elif current_metrics['bt_ratio'] < baseline_metrics['bt_ratio'] * LESS_FOCUSED_BT_RATIO_DECREASE_FACTOR and \
             (state == "Neutral" or state.startswith("Less Relaxed")): # Avoid overriding strong relax state
            state = "Less Focused"


        time_rel_feedback = time.time() - session_start_time_abs
        print(f"\nFeedback @ {time_rel_feedback:6.1f}s ({time.strftime('%H:%M:%S')}): {state}")
        print(f"  Metrics: A/B {current_metrics['ab_ratio']:.2f}(B:{baseline_metrics['ab_ratio']:.2f}), "
              f"B/T {current_metrics['bt_ratio']:.2f}(B:{baseline_metrics['bt_ratio']:.2f}), "
              f"Alpha {current_metrics['alpha']:.1f}(B:{baseline_metrics['alpha']:.1f})")
        
        feedback_log.append({
            "time_abs": time.time(), "time_rel": time_rel_feedback,
            "metrics": current_metrics, "prediction": state
        })
        
        elapsed_in_loop = time.time() - loop_start_time
        wait_time = ANALYSIS_WINDOW_SECONDS - elapsed_in_loop
        if wait_time > 0 and running:
            time.sleep(wait_time)
        if session_start_time_abs + CALIBRATION_DURATION_SECONDS + ANALYSIS_WINDOW_SECONDS > 70:
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
    total_plots = num_raw_plots + num_metric_plots

    fig, axs = plt.subplots(total_plots, 1, figsize=(15, 3 * total_plots), sharex=False)
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
        pred_map = {"Relaxed":3, "Getting Relaxed":2, "Neutral":1, "Less Relaxed":0,
                    "Focused":-3, "Likely Focused":-2, "Less Focused":-1,
                    "Less Relaxed / More Alert": 0.5 } # Map to numbers
        pred_values = [pred_map.get(entry['prediction'], 0) for entry in feedback_log] # Default to neutral if unknown
        ax.plot(metric_times, pred_values, drawstyle='steps-post', label='State', c='black')
        ax.set_yticks(list(set(pred_map.values()))) # Use unique values from map
        ax.set_yticklabels([k for k,v in pred_map.items() if v in set(pred_map.values())]) # Show labels
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
        print("Session ended.")
        plot_results() # Plot everything at the very end

if __name__ == "__main__":
    main()