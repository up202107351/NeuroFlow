# live_eeg_feedback.py
import time
import numpy as np
import pylsl
import datetime
import csv
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
LSL_CHUNK_MAX_PULL = 128 # How many samples to pull at once from LSL (e.g., 0.5s worth at 256Hz)

# EEG Channel Info (Adapt to your Muse LSL stream - usually 4 EEG channels)
# These are indices *within the LSL stream data chunk*
EEG_CHANNEL_INDICES = [0, 1, 2, 3] # TP9, AF7, AF8, TP10 for Muse via MuseLSL/BlueMuse
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

# Calibration & Analysis Windows
CALIBRATION_DURATION_SECONDS = 60.0
ANALYSIS_WINDOW_SECONDS = 6.0  # Process data and give feedback every 6 seconds
# The amount of data used for each PSD calculation within the analysis window.
# Should be long enough for good frequency resolution, e.g., 2-4 seconds.
PSD_WINDOW_SECONDS = 2.0 # Each PSD is calculated over 2s of data

# BrainFlow PSD Parameters
# SAMPLING_RATE will be determined from LSL stream, or use a default if not available
DEFAULT_SAMPLING_RATE = 256.0
# NFFT and OVERLAP will be calculated once SAMPLING_RATE is known

# Band definitions
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0) # Could also define LOW_BETA (13-20) and HIGH_BETA (20-30)

# --- Feedback Logic Thresholds (RELATIVE TO CALIBRATED BASELINE) ---
# These factors determine how much change from baseline is considered significant.
# MUST BE TUNED THROUGH EXPERIMENTATION!

# For Relaxation
RELAX_ALPHA_INCREASE_FACTOR = 1.20  # Current Alpha > Baseline Alpha * 1.20 (20% increase)
RELAX_AB_RATIO_INCREASE_FACTOR = 1.15 # Current A/B > Baseline A/B * 1.15

# For Focus
FOCUS_BT_RATIO_INCREASE_FACTOR = 1.20 # Current B/T > Baseline B/T * 1.20
FOCUS_BETA_INCREASE_FACTOR = 1.15     # Current Beta > Baseline Beta * 1.15
FOCUS_ALPHA_DECREASE_FACTOR = 0.85    # Current Alpha < Baseline Alpha * 0.85 (15% decrease)

# --- Global State ---
running = True
lsl_inlet = None
sampling_rate = DEFAULT_SAMPLING_RATE
nfft = DataFilter.get_nearest_power_of_two(int(DEFAULT_SAMPLING_RATE * PSD_WINDOW_SECONDS))
welch_overlap_samples = nfft // 2 # 50% overlap for FFT windows in Welch

SAVE_DATA = True # Master switch to enable/disable saving
SAVE_PATH = "live_session_data/" # Directory to save data
if SAVE_DATA:
    import os
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    TIMESTAMP_FILENAME = datetime.now().strftime("%Y%m%d_%H%M%S")
    RAW_EEG_FILENAME = os.path.join(SAVE_PATH, f"raw_eeg_{TIMESTAMP_FILENAME}.csv")
    METRICS_FILENAME = os.path.join(SAVE_PATH, f"session_metrics_{TIMESTAMP_FILENAME}.csv")
    BASELINE_FILENAME = os.path.join(SAVE_PATH, f"baseline_{TIMESTAMP_FILENAME}.txt")


baseline_metrics = None # Will store dict: {'alpha':val, 'beta':val, 'theta':val, 'ab_ratio':val, 'bt_ratio':val}
eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0) # Buffer for incoming EEG data

# --- Helper Functions ---
def connect_to_lsl():
    global lsl_inlet, sampling_rate, nfft, welch_overlap_samples
    print(f"Looking for LSL stream (Type: '{LSL_STREAM_TYPE}')...")
    try:
        streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
        if not streams:
            print("LSL stream not found.")
            return False
        
        lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
        stream_info = lsl_inlet.info()
        lsl_sr = stream_info.nominal_srate()
        if lsl_sr > 0:
            sampling_rate = lsl_sr
        else:
            print(f"Warning: LSL stream reported 0 Hz sampling rate. Using default: {DEFAULT_SAMPLING_RATE} Hz")
            sampling_rate = DEFAULT_SAMPLING_RATE

        # Recalculate PSD params based on actual sampling rate
        nfft = DataFilter.get_nearest_power_of_two(int(sampling_rate * PSD_WINDOW_SECONDS))
        welch_overlap_samples = nfft // 2

        print(f"Connected to '{stream_info.name()}' (Type: {stream_info.type()})")
        print(f"  Sampling Rate: {sampling_rate:.2f} Hz")
        print(f"  Number of Channels in Stream: {stream_info.channel_count()}")
        if stream_info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
            print(f"ERROR: LSL stream has {stream_info.channel_count()} channels, script expects access up to index {np.max(EEG_CHANNEL_INDICES)}.")
            return False
        print(f"  Using LSL channels (0-indexed): {EEG_CHANNEL_INDICES}")
        print(f"  PSD NFFT: {nfft}, Welch Overlap: {welch_overlap_samples} samples")
        return True
    except Exception as e:
        print(f"Error connecting to LSL: {e}")
        return False

def calculate_band_powers_for_segment(eeg_segment_all_channels):
    """
    Calculates average band powers over all channels for a given EEG segment.
    eeg_segment_all_channels: (NUM_EEG_CHANNELS, num_samples)
                              This should be PSD_WINDOW_SECONDS long.
    """
    global sampling_rate, nfft, welch_overlap_samples
    
    if eeg_segment_all_channels.shape[1] < nfft:
        # print(f"  Debug: Not enough samples for one FFT in PSD: have {eeg_segment_all_channels.shape[1]}, need {nfft}")
        return None

    all_channel_powers = []
    for ch_idx in range(NUM_EEG_CHANNELS):
        channel_data = eeg_segment_all_channels[ch_idx, :].copy() # Work on a copy
        DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
        try:
            psd_data = DataFilter.get_psd_welch(
                data=channel_data,
                nfft=nfft,
                overlap=welch_overlap_samples,
                sampling_rate=int(sampling_rate),
                window=WindowOperations.HANNING.value
            )
        except Exception as e:
            # print(f"  Debug: Error in get_psd_welch for channel {ch_idx}: {e}")
            return None # If any channel fails, segment processing fails for now

        theta = DataFilter.get_band_power(psd_data, THETA_BAND[0], THETA_BAND[1])
        alpha = DataFilter.get_band_power(psd_data, ALPHA_BAND[0], ALPHA_BAND[1])
        beta = DataFilter.get_band_power(psd_data, BETA_BAND[0], BETA_BAND[1])
        all_channel_powers.append({"theta": theta, "alpha": alpha, "beta": beta})

    if not all_channel_powers or len(all_channel_powers) != NUM_EEG_CHANNELS:
        return None

    avg_metrics = {
        'theta': np.mean([p['theta'] for p in all_channel_powers]),
        'alpha': np.mean([p['alpha'] for p in all_channel_powers]),
        'beta':  np.mean([p['beta']  for p in all_channel_powers])
    }
    avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
    avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
    return avg_metrics

def perform_calibration():
    global eeg_buffer, baseline_metrics, running, sampling_rate
    print(f"\n--- Starting {CALIBRATION_DURATION_SECONDS:.0f} Second Calibration ---")
    print("Please remain in a neutral, resting state (e.g., eyes open, looking at a blank wall, or eyes closed if that's your preferred meditation start).")

    calibration_end_time = time.time() + CALIBRATION_DURATION_SECONDS
    collected_metrics_during_calibration = []
    
    # Ensure buffer is initially empty for calibration period or manage its content
    eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
    samples_for_psd_window = int(PSD_WINDOW_SECONDS * sampling_rate)

    last_metric_calc_time = time.time()

    while time.time() < calibration_end_time and running:
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=1.0, max_samples=LSL_CHUNK_MAX_PULL)
        if chunk:
            chunk_np = np.array(chunk, dtype=np.float64).T
            eeg_chunk_selected_channels = chunk_np[EEG_CHANNEL_INDICES, :]
            
            # Optional: Filter incoming chunk here if desired for calibration too
            # for i in range(NUM_EEG_CHANNELS):
            #     DataFilter.perform_bandpass(eeg_chunk_selected_channels[i,:], int(sampling_rate), 1.0, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0)

            eeg_buffer = np.append(eeg_buffer, eeg_chunk_selected_channels, axis=1)

            # Process in sliding PSD_WINDOW_SECONDS chunks during calibration
            # to get multiple readings for averaging
            while eeg_buffer.shape[1] >= samples_for_psd_window:
                current_segment = eeg_buffer[:, :samples_for_psd_window]
                # Slide buffer: keep overlapping part if desired, or just remove processed part
                # For simplicity, let's slide by half the PSD window for some overlap in readings
                eeg_buffer = eeg_buffer[:, int(samples_for_psd_window * 0.5):] 
                
                metrics = calculate_band_powers_for_segment(current_segment)
                if metrics:
                    collected_metrics_during_calibration.append(metrics)
                    # Provide some feedback during calibration
                    if time.time() - last_metric_calc_time > 2: # Print update every ~2s
                        print(f"  Calibration in progress... Alpha: {metrics['alpha']:.2f}, Beta: {metrics['beta']:.2f}, Theta: {metrics['theta']:.2f} (Time left: {calibration_end_time - time.time():.0f}s)")
                        last_metric_calc_time = time.time()
        else:
            time.sleep(0.01) # Brief pause if no data

    if not collected_metrics_during_calibration:
        print("Calibration failed: No metrics collected. Check LSL stream and connection.")
        return False

    # Calculate average of all collected metrics during calibration
    baseline_metrics = {
        'alpha': np.mean([m['alpha'] for m in collected_metrics_during_calibration]),
        'beta':  np.mean([m['beta']  for m in collected_metrics_during_calibration]),
        'theta': np.mean([m['theta'] for m in collected_metrics_during_calibration])
    }
    baseline_metrics['ab_ratio'] = baseline_metrics['alpha'] / baseline_metrics['beta'] if baseline_metrics['beta'] > 1e-9 else 0
    baseline_metrics['bt_ratio'] = baseline_metrics['beta'] / baseline_metrics['theta'] if baseline_metrics['theta'] > 1e-9 else 0

    print("\n--- Calibration Complete ---")
    print(f"Baseline Alpha Power: {baseline_metrics['alpha']:.2f}")
    print(f"Baseline Beta Power:  {baseline_metrics['beta']:.2f}")
    print(f"Baseline Theta Power: {baseline_metrics['theta']:.2f}")
    print(f"Baseline Alpha/Beta Ratio: {baseline_metrics['ab_ratio']:.2f}")
    print(f"Baseline Beta/Theta Ratio: {baseline_metrics['bt_ratio']:.2f}")
    return True

def analyze_and_feedback():
    global eeg_buffer, baseline_metrics, running, sampling_rate
    
    samples_for_analysis_window = int(ANALYSIS_WINDOW_SECONDS * sampling_rate)
    
    # Ensure buffer is trimmed to avoid using old calibration data directly for first analysis window
    # Or, ensure ANALYSIS_WINDOW_SECONDS starts accumulating *after* calibration
    # For now, we'll just ensure the buffer doesn't grow indefinitely
    max_buffer_samples = samples_for_analysis_window + int(sampling_rate * 2) # Keep a bit more than needed

    print(f"\n--- Starting Real-time Feedback (every {ANALYSIS_WINDOW_SECONDS:.0f} seconds) ---")

    while running:
        start_of_analysis_period = time.time()
        
        # 1. Accumulate data for ANALYSIS_WINDOW_SECONDS
        current_analysis_window_data = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        
        # Try to fill the analysis window. If LSL is slow, this might take longer than ANALYSIS_WINDOW_SECONDS.
        while current_analysis_window_data.shape[1] < samples_for_analysis_window and running:
            chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.2, max_samples=LSL_CHUNK_MAX_PULL)
            if chunk:
                chunk_np = np.array(chunk, dtype=np.float64).T
                eeg_chunk_selected_channels = chunk_np[EEG_CHANNEL_INDICES, :]
                
                # Optional: Filter incoming live chunk here
                # for i in range(NUM_EEG_CHANNELS):
                #     DataFilter.perform_bandpass(eeg_chunk_selected_channels[i,:], int(sampling_rate), 1.0, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0)

                current_analysis_window_data = np.append(current_analysis_window_data, eeg_chunk_selected_channels, axis=1)
            elif not timestamps: # Timeout and no data
                print("  Waiting for data to fill analysis window...")
                if not running: break # Exit if stop signal received
        
        if not running: break
        
        if current_analysis_window_data.shape[1] < samples_for_analysis_window:
            print("  Could not collect enough data for a full analysis window. Skipping this interval.")
            # Wait for the remainder of the nominal 6-second interval before trying again
            time_to_wait = ANALYSIS_WINDOW_SECONDS - (time.time() - start_of_analysis_period)
            if time_to_wait > 0: time.sleep(time_to_wait)
            continue

        # We have a full analysis window (e.g., 6s of data)
        # Now, calculate metrics using PSD_WINDOW_SECONDS (e.g., 2s) segments from this full window.
        # This gives an average state over the ANALYSIS_WINDOW_SECONDS.
        
        num_psd_windows_in_analysis = 0
        sum_metrics_in_analysis_window = {'alpha': 0, 'beta': 0, 'theta': 0}
        
        temp_psd_buffer = current_analysis_window_data.copy()
        samples_for_psd_window = int(PSD_WINDOW_SECONDS * sampling_rate)

        while temp_psd_buffer.shape[1] >= samples_for_psd_window:
            segment_for_psd = temp_psd_buffer[:, :samples_for_psd_window]
            # Slide by half for overlap, or full for no overlap of PSD windows
            temp_psd_buffer = temp_psd_buffer[:, int(samples_for_psd_window * 0.5):] 
            
            metrics = calculate_band_powers_for_segment(segment_for_psd)
            if metrics:
                sum_metrics_in_analysis_window['alpha'] += metrics['alpha']
                sum_metrics_in_analysis_window['beta']  += metrics['beta']
                sum_metrics_in_analysis_window['theta'] += metrics['theta']
                num_psd_windows_in_analysis += 1
        
        if num_psd_windows_in_analysis == 0:
            print("  Failed to calculate any PSD metrics for the current analysis window.")
            # Wait for the remainder of the nominal 6-second interval
            time_to_wait = ANALYSIS_WINDOW_SECONDS - (time.time() - start_of_analysis_period)
            if time_to_wait > 0: time.sleep(time_to_wait)
            continue

        # Average metrics over the entire ANALYSIS_WINDOW_SECONDS
        current_metrics = {
            'alpha': sum_metrics_in_analysis_window['alpha'] / num_psd_windows_in_analysis,
            'beta':  sum_metrics_in_analysis_window['beta'] / num_psd_windows_in_analysis,
            'theta': sum_metrics_in_analysis_window['theta'] / num_psd_windows_in_analysis
        }
        current_metrics['ab_ratio'] = current_metrics['alpha'] / current_metrics['beta'] if current_metrics['beta'] > 1e-9 else 0
        current_metrics['bt_ratio'] = current_metrics['beta'] / current_metrics['theta'] if current_metrics['theta'] > 1e-9 else 0

        # --- Compare to Baseline and Provide Feedback ---
        state_description = "Neutral"
        details = []

        # Relaxation Check
        alpha_significantly_higher = current_metrics['alpha'] > baseline_metrics['alpha'] * RELAX_ALPHA_INCREASE_FACTOR
        ab_ratio_significantly_higher = current_metrics['ab_ratio'] > baseline_metrics['ab_ratio'] * RELAX_AB_RATIO_INCREASE_FACTOR

        if alpha_significantly_higher and ab_ratio_significantly_higher:
            state_description = "Relaxed"
        elif alpha_significantly_higher or ab_ratio_significantly_higher: # Milder condition
            state_description = "Getting More Relaxed"
        elif current_metrics['ab_ratio'] < baseline_metrics['ab_ratio'] * 0.85: # Example for less relaxed
            state_description = "Less Relaxed / More Alert"


        # Focus Check (can override relaxation if strong focus signals, or report both if approriate)
        # Note: Stress and Focus can both show increased Beta. Ratios are key.
        bt_ratio_significantly_higher = current_metrics['bt_ratio'] > baseline_metrics['bt_ratio'] * FOCUS_BT_RATIO_INCREASE_FACTOR
        beta_significantly_higher = current_metrics['beta'] > baseline_metrics['beta'] * FOCUS_BETA_INCREASE_FACTOR
        alpha_significantly_lower = current_metrics['alpha'] < baseline_metrics['alpha'] * FOCUS_ALPHA_DECREASE_FACTOR

        if bt_ratio_significantly_higher and beta_significantly_higher and alpha_significantly_lower:
            state_description = "Focused" # Strongest focus indicator
        elif bt_ratio_significantly_higher and beta_significantly_higher:
            state_description = "Likely Focused"
        elif current_metrics['bt_ratio'] < baseline_metrics['bt_ratio'] * 0.85 and state_description.startswith("Neutral"): # Example for less focused
             state_description = "Less Focused"


        details.append(f"A/B: {current_metrics['ab_ratio']:.2f} (Base: {baseline_metrics['ab_ratio']:.2f})")
        details.append(f"B/T: {current_metrics['bt_ratio']:.2f} (Base: {baseline_metrics['bt_ratio']:.2f})")
        details.append(f"Alpha: {current_metrics['alpha']:.2f} (Base: {baseline_metrics['alpha']:.2f})")
        # details.append(f"Beta: {current_metrics['beta']:.2f} (Base: {baseline_metrics['beta']:.2f})")
        # details.append(f"Theta: {current_metrics['theta']:.2f} (Base: {baseline_metrics['theta']:.2f})")

        print(f"\nFeedback ({time.strftime('%H:%M:%S')}): {state_description}")
        print(f"  Metrics: {', '.join(details)}")
        
        # --- Crude buffer management: keep only last N seconds to prevent infinite growth ---
        # This is not ideal for continuous analysis but prevents memory overflow in this simple script.
        # A proper circular buffer or more refined segmenting would be better.
        # For now, we re-accumulate for each analysis window.
        # eeg_buffer = eeg_buffer[:, -max_buffer_samples:] if eeg_buffer.shape[1] > max_buffer_samples else eeg_buffer
        
        # Wait for the remainder of the 6-second interval
        time_spent_processing = time.time() - start_of_analysis_period
        time_to_wait = ANALYSIS_WINDOW_SECONDS - time_spent_processing
        if time_to_wait > 0 and running:
            time.sleep(time_to_wait)

def save_collected_data():
    global all_recorded_eeg_chunks, all_session_metrics_log
    if not SAVE_DATA:
        return

    print("\n--- Saving Collected Data ---")
    # Save Raw EEG Data
    if all_recorded_eeg_chunks:
        print(f"Saving raw EEG data to {RAW_EEG_FILENAME}...")
        # This is a simplified way to save. For perfect alignment, you'd interleave timestamps.
        # Here, we'll just concatenate all EEG data and note that timestamps are per-chunk.
        # A better way is to save each (timestamp, chunk_data) pair.
        # For simplicity, let's just save the concatenated data for now.
        full_eeg_data = np.concatenate([chunk_data for _, chunk_data in all_recorded_eeg_chunks], axis=1)
        # Add a time column based on first timestamp and sampling rate (approximate)
        first_lsl_timestamp = all_recorded_eeg_chunks[0][0]
        num_samples_total = full_eeg_data.shape[1]
        time_col = np.array([first_lsl_timestamp + (i / sampling_rate) for i in range(num_samples_total)])
        
        # Prepare data for CSV: time, ch1, ch2, ch3, ch4
        data_to_save = np.vstack((time_col, full_eeg_data)).T
        header = "Timestamp," + ",".join([f"EEG_Chan{i+1}" for i in range(NUM_EEG_CHANNELS)])
        np.savetxt(RAW_EEG_FILENAME, data_to_save, delimiter=",", header=header, comments='')
        print("Raw EEG data saved.")

    # Save Session Metrics Log
    if all_session_metrics_log:
        print(f"Saving session metrics to {METRICS_FILENAME}...")
        try:
            with open(METRICS_FILENAME, 'w', newline='') as csvfile:
                if all_session_metrics_log: # Check again in case it became empty
                    fieldnames = all_session_metrics_log[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_session_metrics_log)
            print("Session metrics saved.")
        except Exception as e:
            print(f"Error saving metrics: {e}")

def main():
    global running
    if not BRAINFLOW_AVAILABLE: return

    if not connect_to_lsl():
        return

    if not perform_calibration():
        if lsl_inlet: lsl_inlet.close_stream()
        return

    try:
        analyze_and_feedback()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")
    finally:
        running = False
        if lsl_inlet:
            print("Closing LSL stream...")
            lsl_inlet.close_stream()
        print("Application terminated.")

if __name__ == "__main__":
    main()