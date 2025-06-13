#!/usr/bin/env python3
"""
Simple Muse Testing Script

Uses your approach of keeping matplotlib in the main thread
while supporting the enhanced state labeling.
"""

import time
import numpy as np
import pylsl
import matplotlib.pyplot as plt
from datetime import datetime
import signal
import os
import pickle
import keyboard
from scipy.signal import butter, filtfilt, welch
import threading
import logging

# Import from backend modules
from backend.signal_quality_validator import SignalQualityValidator

try:
    from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("ERROR: brainflow library not found. Please install it (pip install brainflow).")
    exit()

# --- Configuration ---
LSL_STREAM_TYPE = 'EEG'
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX_PULL = 128

EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10 for Muse
ACC_CHANNEL_INDICES = [9, 10, 11]   # X, Y, Z for accelerometer
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0
PSD_WINDOW_SECONDS = 6.0

DEFAULT_SAMPLING_RATE = 256.0

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Session types
SESSION_TYPE_RELAX = "RELAXATION"
SESSION_TYPE_FOCUS = "FOCUS"

# --- Global State ---
running = True
lsl_inlet = None
sampling_rate = DEFAULT_SAMPLING_RATE
nfft = DataFilter.get_nearest_power_of_two(int(DEFAULT_SAMPLING_RATE * PSD_WINDOW_SECONDS))
welch_overlap_samples = nfft // 2
filter_order = 4
lowcut = 0.5
highcut = 30.0

# Signal quality related
signal_quality_validator = None

# Filter coefficients
nyq = 0.5 * sampling_rate
low = lowcut / nyq
b_hp, a_hp = butter(filter_order, low, btype='highpass', analog=False)
high = highcut / nyq
b_lp, a_lp = butter(filter_order, high, btype='lowpass', analog=False)

# Session data
baseline_metrics = None
full_session_eeg_data = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
full_session_eeg_data_raw = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
full_session_timestamps_lsl = []

# Enhanced labeling system
feedback_log = []
user_labels = []  # Event markers (artifacts, eyes open/closed)
user_state_changes = []  # Persistent state changes
current_user_state = "Unknown"  # Current persistent state
quality_log = []

# State tracking
previous_states = []
state_momentum = 0.75
state_velocity = 0.0
level_momentum = 0.8
level_velocity = 0.0

# Current session type
session_type = None
session_start_time = None

# --- Data Saving Configuration ---
SAVE_PATH = "test_session_data/"
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(SAVE_PATH, exist_ok=True)

# --- Enhanced User Labels ---
STATE_LABELS = {
    '1': 'Very Relaxed',
    '2': 'Relaxed', 
    '3': 'Neutral',
    '4': 'Focused',
    '5': 'Very Focused',
    '0': 'Unknown',
    'u': 'Uncertain'
}

EVENT_LABELS = {
    's': 'START_ACTIVITY',
    'e': 'END_ACTIVITY', 
    'b': 'BLINK_ARTIFACT',
    'm': 'MOVEMENT_ARTIFACT',
    'n': 'NOISE_ARTIFACT',
    'c': 'EYES_CLOSED',
    'o': 'EYES_OPEN'
}

# Color mapping for states
STATE_COLORS = {
    'Very Relaxed': '#0066CC',
    'Relaxed': '#3399FF', 
    'Neutral': '#CCCCCC',
    'Focused': '#FF6600',
    'Very Focused': '#CC3300',
    'Unknown': '#FFFFFF',
    'Uncertain': '#FFFF99'
}

print("\n=== Enhanced EEG Testing Script ===")
print("\nPersistent State Labels (press key to set current state):")
for key, label in STATE_LABELS.items():
    print(f"  [{key}] - {label}")
print("\nEvent Markers (single events):")
for key, label in EVENT_LABELS.items():
    print(f"  [{key}] - {label}")
print(f"\nCurrent State: {current_user_state}")
print("Press Ctrl+C to end the session")


def graceful_signal_handler(sig, frame):
    global running
    if running:
        print(f'\nSignal {sig} (Ctrl+C) received. Finishing operations and saving data...')
        running = False


signal.signal(signal.SIGINT, graceful_signal_handler)


def connect_to_lsl():
    global lsl_inlet, sampling_rate, nfft, welch_overlap_samples, signal_quality_validator
    print(f"Looking for LSL stream (Type: '{LSL_STREAM_TYPE}')...")
    
    try:
        streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
        if not streams:
            print("LSL stream not found.")
            return False
            
        lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
        info = lsl_inlet.info()
        lsl_sr = info.nominal_srate()
        sampling_rate = lsl_sr if lsl_sr > 0 else DEFAULT_SAMPLING_RATE
        
        # Update filter parameters
        nfft = DataFilter.get_nearest_power_of_two(int(sampling_rate * PSD_WINDOW_SECONDS))
        welch_overlap_samples = nfft // 2
        
        print(f"[DEBUG] Updated nfft to {nfft} for sampling rate {sampling_rate} Hz")
        print(f"[DEBUG] This requires {nfft/sampling_rate:.1f} seconds of data")
        
        # Create signal quality validator
        signal_quality_validator = SignalQualityValidator()
        
        print(f"Connected to '{info.name()}' @ {sampling_rate:.2f} Hz")
        
        if info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
            print(f"ERROR: LSL stream has insufficient channels.")
            return False
        return True
        
    except Exception as e:
        print(f"Error connecting to LSL: {e}")
        return False


def update_user_state(new_state):
    """Update the current user state and log the change"""
    global current_user_state, user_state_changes
    
    if new_state != current_user_state:
        timestamp = time.time()
        user_state_changes.append({
            "time": timestamp,
            "from_state": current_user_state,
            "to_state": new_state
        })
        current_user_state = new_state
        
        # Update session time if this is the first state change after calibration
        if session_start_time:
            time_rel = timestamp - session_start_time
            print(f"\n>>> State Change @ {time_rel:.1f}s: {new_state}")
        else:
            print(f"\n>>> State Change: {new_state}")


def get_current_user_state_at_time(timestamp):
    """Get what the user's reported state was at a given timestamp"""
    if not user_state_changes:
        return "Unknown"
    
    # Find the most recent state change before this timestamp
    current_state = "Unknown"
    for change in user_state_changes:
        if change["time"] <= timestamp:
            current_state = change["to_state"]
        else:
            break
    
    return current_state


def filter_eeg_data(eeg_data):
    """Apply bandpass filter to EEG data"""
    min_samples = 3 * filter_order + 1
    
    if eeg_data.shape[1] < min_samples:
        return eeg_data
        
    eeg_filtered = np.zeros_like(eeg_data)
    for i in range(NUM_EEG_CHANNELS):
        eeg_filtered[i] = filtfilt(b_lp, a_lp, eeg_data[i])
        eeg_filtered[i] = filtfilt(b_hp, a_hp, eeg_filtered[i])
    
    return eeg_filtered


def calculate_band_powers(eeg_segment):
    """Calculate band powers with more debug info"""
    if eeg_segment.shape[1] < nfft:
        print(f"[DEBUG] Segment too small: {eeg_segment.shape[1]} samples (need {nfft})")
        return None
    
    # Apply artifact rejection
    artifact_mask = improved_artifact_rejection(eeg_segment)
    valid_percentage = np.mean(artifact_mask) * 100
    
    if np.sum(artifact_mask) < 0.5 * eeg_segment.shape[1]:  # Lowered from 0.7
        print(f"[DEBUG] Too many artifacts: only {valid_percentage:.1f}% valid samples")
        return None
    
    metrics_list = []
    for ch_idx in range(NUM_EEG_CHANNELS):
        ch_data = eeg_segment[ch_idx, artifact_mask].copy() if np.any(artifact_mask) else eeg_segment[ch_idx].copy()
        
        if len(ch_data) < nfft:
            pad_length = nfft - len(ch_data)
            ch_data = np.pad(ch_data, (0, pad_length), mode='reflect')
            
        DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
        
        try:
            psd = DataFilter.get_psd_welch(
                ch_data, nfft, welch_overlap_samples, int(sampling_rate), 
                WindowOperations.HANNING.value
            )
            
            metrics_list.append({
                'theta': DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1]),
                'alpha': DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1]),
                'beta': DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])
            })
        except Exception as e:
            print(f"[DEBUG] PSD calculation error on channel {ch_idx}: {e}")
            return None
    
    if len(metrics_list) != NUM_EEG_CHANNELS:
        print(f"[DEBUG] Incomplete metrics: got {len(metrics_list)}/{NUM_EEG_CHANNELS} channels")
        return None
        
    # Calculate weighted average
    avg_metrics = {
        'theta': np.mean([m['theta'] for m in metrics_list]),
        'alpha': np.mean([m['alpha'] for m in metrics_list]),
        'beta': np.mean([m['beta'] for m in metrics_list])
    }
    
    avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
    avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
    
    return avg_metrics


def improved_artifact_rejection(eeg_data):
    """Advanced artifact rejection with less aggressive thresholds"""
    channel_thresholds = [300, 200, 200, 300]  # Increased from original values
    amplitude_mask = ~np.any(np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1), axis=0)
    
    diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
    if eeg_data.shape[1] > 1:
        diff_thresholds = [100, 60, 60, 100]  # Increased from original values
        diff_mask = ~np.any(
            np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
            np.array(diff_thresholds).reshape(-1, 1), axis=0
        )
    
    # Print artifact rejection stats during calibration
    valid_percentage = np.mean(amplitude_mask & diff_mask) * 100
    if valid_percentage < 70:
        print(f"[DEBUG] Artifact rejection is keeping only {valid_percentage:.1f}% of data")
    
    return amplitude_mask & diff_mask


def classify_mental_state(current_metrics):
    """Classify mental state based on current metrics"""
    global previous_states, state_velocity, session_type
    
    if not baseline_metrics or not current_metrics:
        return {
            "state": "Calibrating", "level": 0, "confidence": "N/A",
            "value": 0.5, "smooth_value": 0.5, "state_key": "calibrating"
        }
    
    # Determine most probable state
    level = 0
    
    if session_type == SESSION_TYPE_RELAX:
        # Calculate relaxation vs. alert signals
        ab_ratio_decrease = baseline_metrics['ab_ratio'] / current_metrics['ab_ratio'] - 1.0 if current_metrics['ab_ratio'] > 0 else 0
        beta_increase = current_metrics['beta'] / baseline_metrics['beta'] - 1.0
        alpha_decrease = 1.0 - current_metrics['alpha'] / baseline_metrics['alpha']
        
        alert_signal = (0.4 * ab_ratio_decrease + 0.3 * beta_increase + 0.3 * alpha_decrease) * 4.0
        
        # Calculate relaxation signals
        alpha_increase = current_metrics['alpha'] / baseline_metrics['alpha'] - 1.0
        ab_ratio_increase = current_metrics['ab_ratio'] / baseline_metrics['ab_ratio'] - 1.0 if baseline_metrics['ab_ratio'] > 0 else 0
        relax_signal = (0.5 * alpha_increase + 0.5 * ab_ratio_increase) * 5.0
        
        # Determine relaxation levels
        if (ab_ratio_decrease > 0.1 and (beta_increase > 0.1 or alpha_decrease > 0.1)):
            if alert_signal > 1.5:
                level = -3; state_name = "tense"
            elif alert_signal > 1.0:
                level = -2; state_name = "alert"
            else:
                level = -1; state_name = "less_relaxed"
        elif alpha_increase > 0.05 or ab_ratio_increase > 0.05:
            if relax_signal > 0.5:
                level = 4; state_name = "deeply_relaxed"
            elif relax_signal > 0.25:
                level = 3; state_name = "strongly_relaxed"
            elif relax_signal > 0.1:
                level = 2; state_name = "moderately_relaxed"
            else:
                level = 1; state_name = "slightly_relaxed"
        else:
            level = 0; state_name = "neutral"
        
        value = (level + 3) / 7.0
        
    elif session_type == SESSION_TYPE_FOCUS:
        # Calculate focus and distraction signals
        bt_ratio_decrease = baseline_metrics['bt_ratio'] / current_metrics['bt_ratio'] - 1.0 if current_metrics['bt_ratio'] > 0 else 0
        beta_decrease = 1.0 - current_metrics['beta'] / baseline_metrics['beta']
        theta_increase = current_metrics['theta'] / baseline_metrics['theta'] - 1.0
        
        distraction_signal = (0.4 * bt_ratio_decrease + 0.3 * beta_decrease + 0.3 * theta_increase) * 4.0
        
        # Calculate focus signals
        beta_increase = current_metrics['beta'] / baseline_metrics['beta'] - 1.0
        bt_ratio_increase = current_metrics['bt_ratio'] / baseline_metrics['bt_ratio'] - 1.0 if baseline_metrics['bt_ratio'] > 0 else 0
        focus_signal = (0.5 * beta_increase + 0.5 * bt_ratio_increase) * 5.0
        
        # Determine focus levels
        if (bt_ratio_decrease > 0.1 and (beta_decrease > 0.1 or theta_increase > 0.1)):
            if distraction_signal > 1.5:
                level = -3; state_name = "distracted"
            elif distraction_signal > 1.0:
                level = -2; state_name = "less_attentive"
            else:
                level = -1; state_name = "less_focused"
        elif beta_increase > 0.05 or bt_ratio_increase > 0.05:
            if focus_signal > 0.5:
                level = 4; state_name = "highly_focused"
            elif focus_signal > 0.25:
                level = 3; state_name = "strongly_focused"
            elif focus_signal > 0.1:
                level = 2; state_name = "moderately_focused"
            else:
                level = 1; state_name = "slightly_focused"
        else:
            level = 0; state_name = "neutral"
        
        value = (level + 3) / 7.0
    else:
        value = 0.5
        state_name = "neutral"
    
    # Apply temporal smoothing
    smooth_value = value
    
    previous_states.append((state_name, value, level))
    if len(previous_states) > 5:
        previous_states.pop(0)
    
    if len(previous_states) > 1:
        recent_values = [s[1] for s in previous_states]
        prev_value = recent_values[-2]
        
        target_velocity = value - prev_value
        state_velocity = (state_velocity * state_momentum + target_velocity * (1 - state_momentum))
        smooth_value = prev_value + state_velocity
        smooth_value = min(1.0, max(0.0, smooth_value))
    
    # Map state names to display names
    state_display_map = {
        'deeply_relaxed': "Deeply Relaxed", 'strongly_relaxed': "Strongly Relaxed",
        'moderately_relaxed': "Moderately Relaxed", 'slightly_relaxed': "Slightly Relaxed",
        'neutral': "Neutral", 'less_relaxed': "Less Relaxed", 'alert': "Alert", 'tense': "Tense",
        'calibrating': "Calibrating", 'highly_focused': "Highly Focused",
        'strongly_focused': "Strongly Focused", 'moderately_focused': "Moderately Focused",
        'slightly_focused': "Slightly Focused", 'less_focused': "Less Focused",
        'less_attentive': "Less Attentive", 'distracted': "Distracted"
    }
    
    display_state = state_display_map.get(state_name, state_name.title())
    
    return {
        "state": display_state, "state_key": state_name, "level": level,
        "confidence": "medium", "value": round(value, 3),
        "smooth_value": round(smooth_value, 3)
    }


def perform_calibration_phase():
    """Calibration phase with signal quality monitoring and DEBUGGING"""
    global baseline_metrics, signal_quality_validator, nfft

    print(f"\n--- Starting {CALIBRATION_DURATION_SECONDS:.0f} Second Calibration ---")
    print("Please remain in a neutral, resting state.")

    calibration_start_time = time.time()
    calibration_metrics_list = []
    chunk_counter = 0
    sample_counter = 0
    skip_small_chunk_counter = 0
    metrics_fail_counter = 0
    
    # Create a buffer to accumulate EEG data
    eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)

    # Process all metrics during calibration time
    while (time.time() - calibration_start_time < CALIBRATION_DURATION_SECONDS and running):
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)

        chunk_counter += 1
        if not chunk:
            print(f"[DEBUG] No chunk received at iteration {chunk_counter}")
            time.sleep(0.01)
            continue

        chunk_np = np.array(chunk, dtype=np.float64).T
        print(f"[DEBUG] Received chunk shape: {chunk_np.shape} at iteration {chunk_counter}")

        # Check if we have enough channels
        if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
            print(f"[DEBUG] Not enough channels in chunk: {chunk_np.shape[0]} (need >={max(EEG_CHANNEL_INDICES)+1}), skipping.")
            skip_small_chunk_counter += 1
            continue

        eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
        sample_counter += eeg_chunk.shape[1]

        # Extract accelerometer data if available
        if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
            try:
                acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                if acc_chunk.shape[1] > 0:
                    latest_acc_sample = acc_chunk[:, -1]
                    signal_quality_validator.add_accelerometer_data(latest_acc_sample)
            except Exception as e:
                print(f"[DEBUG] Could not extract accelerometer data: {e}")

        eeg_chunk_filtered = filter_eeg_data(eeg_chunk)

        # Add to global buffer for visualization
        global full_session_eeg_data, full_session_eeg_data_raw, full_session_timestamps_lsl
        full_session_eeg_data = np.append(full_session_eeg_data, eeg_chunk_filtered, axis=1)
        full_session_eeg_data_raw = np.append(full_session_eeg_data_raw, eeg_chunk, axis=1)

        if timestamps:
            full_session_timestamps_lsl.extend(timestamps)
            
        # Add to local EEG buffer
        eeg_buffer = np.append(eeg_buffer, eeg_chunk_filtered, axis=1)
        print(f"[DEBUG] Current buffer size: {eeg_buffer.shape[1]}/{nfft} samples")

        # If we have enough data for PSD calculation
        if eeg_buffer.shape[1] >= nfft:
            segment = eeg_buffer[:, -nfft:]
            print(f"[DEBUG] Trying to calculate band powers on segment shape: {segment.shape}")
            metrics = calculate_band_powers(segment)

            if metrics:
                print(f"[DEBUG] Got metrics: {metrics}")
                # Add to quality validator
                signal_quality_validator.add_band_power_data(metrics)
                signal_quality_validator.add_raw_eeg_data(segment)
                calibration_metrics_list.append(metrics)
                
                # Keep a sliding window with 50% overlap
                eeg_buffer = eeg_buffer[:, -int(nfft/2):]
            else:
                metrics_fail_counter += 1
                print(f"[DEBUG] Failed to calculate metrics for segment at iteration {chunk_counter}")

            # Check signal quality every 5 metrics
            if len(calibration_metrics_list) % 5 == 0 and len(calibration_metrics_list) > 0:
                quality = signal_quality_validator.assess_overall_quality()
                print(f"Calibration progress: {((time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS) * 100:.0f}% - "
                      f"Quality: {quality.quality_level} ({quality.overall_score:.2f})")
                if quality.overall_score < 0.3:
                    print(f"WARNING: Poor signal quality - {quality.recommendations[0] if quality.recommendations else 'Please adjust headband'}")

    print(f"\n[DEBUG SUMMARY] Total chunks: {chunk_counter}, samples: {sample_counter}, small chunks skipped: {skip_small_chunk_counter}, metrics failed: {metrics_fail_counter}")
    print(f"[DEBUG] Total metrics collected: {len(calibration_metrics_list)}")
    print(f"[DEBUG] Final buffer size: {eeg_buffer.shape[1]} samples")
    print(f"[DEBUG] Required nfft size: {nfft} samples")

    if calibration_metrics_list:
        print(f"Creating baseline from {len(calibration_metrics_list)} metrics")
        baseline_metrics = {
            'alpha': np.mean([m['alpha'] for m in calibration_metrics_list]),
            'beta': np.mean([m['beta'] for m in calibration_metrics_list]),
            'theta': np.mean([m['theta'] for m in calibration_metrics_list])
        }
        baseline_metrics['ab_ratio'] = (
            baseline_metrics['alpha'] / baseline_metrics['beta']
            if baseline_metrics['beta'] > 1e-9 else 0
        )
        baseline_metrics['bt_ratio'] = (
            baseline_metrics['beta'] / baseline_metrics['theta']
            if baseline_metrics['theta'] > 1e-9 else 0
        )

        print("\n--- Calibration Complete ---")
        for key, val in baseline_metrics.items():
            print(f"Baseline {key.replace('_', ' ').title()}: {val:.2f}")
        return True
    else:
        print("Calibration failed: No metrics collected. Check LSL stream.")
        # Additional diagnostic info
        print(f"[DEBUG] Make sure your Muse is properly connected and sending data.")
        print(f"[DEBUG] Check that PSD_WINDOW_SECONDS ({PSD_WINDOW_SECONDS}) isn't too large for your sample rate ({sampling_rate}).")
        print(f"[DEBUG] Current nfft setting requires {nfft/sampling_rate:.1f} seconds of continuous data.")
        return False


def monitor_keyboard_input():
    """Monitor keyboard for user labels and state changes"""
    global running, user_labels, current_user_state
    
    while running:
        # Check for state labels
        for key in STATE_LABELS:
            if keyboard.is_pressed(key):
                new_state = STATE_LABELS[key]
                update_user_state(new_state)
                time.sleep(0.5)  # Prevent multiple detections
        
        # Check for event labels
        for key in EVENT_LABELS:
            if keyboard.is_pressed(key):
                label = EVENT_LABELS[key]
                timestamp = time.time()
                user_labels.append({"time": timestamp, "label": label})
                
                if session_start_time:
                    time_rel = timestamp - session_start_time
                    print(f"\n>>> Event @ {time_rel:.1f}s: {label}")
                else:
                    print(f"\n>>> Event: {label}")
                time.sleep(0.5)
                
        time.sleep(0.1)


def feedback_loop():
    """Real-time feedback loop with data collection and periodic plotting"""
    global baseline_metrics, running, feedback_log, session_start_time
    
    if not baseline_metrics:
        print("Cannot start feedback loop: baseline not calibrated.")
        return
    
    print(f"\n--- Starting Real-time Feedback ---")
    session_start_time = time.time()
    
    # Start keyboard monitoring thread
    keyboard_thread = threading.Thread(target=monitor_keyboard_input)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
    last_analysis_time = 0
    total_loop_times = 0
    last_plot_time = 0
    
    while running:
        loop_start_time = time.time()
        
        # Pull data chunk
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
        
        if not chunk:
            time.sleep(0.01)
            continue
            
        chunk_np = np.array(chunk, dtype=np.float64).T
        
        if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
            continue
            
        eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
        
        # Extract accelerometer data
        if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
            try:
                acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                if acc_chunk.shape[1] > 0:
                    latest_acc_sample = acc_chunk[:, -1]
                    signal_quality_validator.add_accelerometer_data(latest_acc_sample)
            except Exception as e:
                pass
        
        eeg_chunk_filtered = filter_eeg_data(eeg_chunk)
        
        # Add to global buffer
        global full_session_eeg_data, full_session_eeg_data_raw, full_session_timestamps_lsl
        full_session_eeg_data = np.append(full_session_eeg_data, eeg_chunk_filtered, axis=1)
        full_session_eeg_data_raw = np.append(full_session_eeg_data_raw, eeg_chunk, axis=1)
        
        if timestamps:
            full_session_timestamps_lsl.extend(timestamps)
        
        eeg_buffer = np.append(eeg_buffer, eeg_chunk_filtered, axis=1)
        
        # Process if enough time has passed
        current_time = time.time()
        if current_time - last_analysis_time >= ANALYSIS_WINDOW_SECONDS:
            if eeg_buffer.shape[1] >= nfft:
                segment = eeg_buffer[:, -nfft:]
                current_metrics = calculate_band_powers(segment)
                
                if current_metrics:
                    signal_quality_validator.add_band_power_data(current_metrics)
                    signal_quality_validator.add_raw_eeg_data(segment)
                    
                    quality = signal_quality_validator.assess_overall_quality()
                    classification = classify_mental_state(current_metrics)
                    
                    time_rel_feedback = current_time - session_start_time
                    current_reported_state = get_current_user_state_at_time(current_time)
                    
                    feedback_log.append({
                        "time_abs": current_time,
                        "time_rel": time_rel_feedback,
                        "metrics": current_metrics,
                        "prediction": classification["state"],
                        "state_key": classification["state_key"],
                        "level": classification["level"],
                        "confidence": classification["confidence"],
                        "smooth_value": classification["smooth_value"],
                        "user_reported_state": current_reported_state
                    })
                    
                    quality_log.append({
                        "time": current_time,
                        "metrics": {
                            "movement_score": quality.movement_score,
                            "band_power_score": quality.band_power_score,
                            "electrode_contact_score": quality.electrode_contact_score,
                            "overall_score": quality.overall_score,
                            "quality_level": quality.quality_level,
                            "recommendations": quality.recommendations
                        }
                    })
                    
                    # Print feedback with current user state
                    print(f"\nFeedback @ {time_rel_feedback:6.1f}s: {classification['state']} (Level {classification['level']})")
                    print(f"  User State: {current_reported_state} | Quality: {quality.quality_level} ({quality.overall_score:.2f})")
                    
                    # Track total time for plotting interval
                    total_loop_times += (current_time - last_analysis_time)
                    
                # Reset buffer with overlap
                eeg_buffer = eeg_buffer[:, -int(nfft * 0.5):]
                last_analysis_time = current_time
                
                # Plot periodically (every 30 seconds)
                if total_loop_times >= 30 and (current_time - last_plot_time) >= 30:
                    print("\n--- Generating periodic plot ---")
                    plot_results()
                    total_loop_times = 0
                    last_plot_time = current_time
                
            
def add_state_background(ax, time_range):
    """Add background coloring for user states"""
    if not session_start_time or not user_state_changes or not time_range:
        return
        
    for i, change in enumerate(user_state_changes):
        if change["time"] >= session_start_time:
            start_time_rel = change["time"] - session_start_time
            
            # Find end time
            if i + 1 < len(user_state_changes):
                next_change = user_state_changes[i + 1]
                if next_change["time"] >= session_start_time:
                    end_time_rel = next_change["time"] - session_start_time
                else:
                    continue
            else:
                end_time_rel = max(time_range) if isinstance(time_range, list) else time_range[-1]
            
            # Only draw if within current view
            min_time = min(time_range) if isinstance(time_range, list) else time_range[0]
            max_time = max(time_range) if isinstance(time_range, list) else time_range[-1]
            
            if start_time_rel <= max_time and end_time_rel >= min_time:
                color = STATE_COLORS.get(change["to_state"], '#FFFFFF')
                ax.axvspan(start_time_rel, end_time_rel, alpha=0.2, color=color)


def add_event_markers(ax, times):
    """Add event markers to plot"""
    if not session_start_time or not user_labels or not times:
        return
        
    for label_event in user_labels:
        if label_event['time'] >= session_start_time:
            label_time = label_event['time'] - session_start_time
            if times[0] <= label_time <= times[-1]:
                ax.axvline(x=label_time, color='purple', linestyle=':', alpha=0.7)
                ax.text(label_time, ax.get_ylim()[1] * 0.9, label_event['label'], 
                       rotation=90, ha='right', va='top', alpha=0.8, fontsize=8)


def plot_results():
    """Create plots for visualization - RUNS IN MAIN THREAD"""
    if full_session_eeg_data.size == 0 and not feedback_log:
        print("No data to plot yet.")
        return
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Raw EEG with user state background
    ax = axs[0]
    if full_session_eeg_data.shape[1] > 0:
        window_samples = min(full_session_eeg_data.shape[1], int(sampling_rate * 10))
        recent_data = full_session_eeg_data[:, -window_samples:]
        
        if len(full_session_timestamps_lsl) >= window_samples:
            time_axis = np.array(full_session_timestamps_lsl[-window_samples:]) - session_start_time
        else:
            time_axis = np.linspace(-window_samples/sampling_rate, 0, recent_data.shape[1])
        
        # Add background coloring for user states
        add_state_background(ax, time_axis)
        
        for i in range(NUM_EEG_CHANNELS):
            ax.plot(time_axis, recent_data[i], label=f'Ch {i+1}', alpha=0.8)
    
    ax.set_title(f'Real-time EEG - Current User State: {current_user_state}')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Band Powers with user state background  
    ax = axs[1]
    if feedback_log:
        recent_feedback = feedback_log[-60:]  # Last 60 entries
        times = [entry['time_rel'] for entry in recent_feedback]
        
        if times:
            # Add background coloring
            add_state_background(ax, times)
            
            ax.plot(times, [m['metrics']['alpha'] for m in recent_feedback], 
                   label='Alpha', c='blue', linewidth=1.5)
            ax.plot(times, [m['metrics']['beta'] for m in recent_feedback], 
                   label='Beta', c='red', linewidth=1.5)
            ax.plot(times, [m['metrics']['theta'] for m in recent_feedback], 
                   label='Theta', c='green', linewidth=1.5)
            
            # Plot baselines
            if baseline_metrics:
                baseline_time = [times[0], times[-1]]
                ax.plot(baseline_time, [baseline_metrics['alpha']] * 2, 'b--', alpha=0.5)
                ax.plot(baseline_time, [baseline_metrics['beta']] * 2, 'r--', alpha=0.5)
                ax.plot(baseline_time, [baseline_metrics['theta']] * 2, 'g--', alpha=0.5)
            
            # Add event markers
            add_event_markers(ax, times)
    
    ax.set_title('Band Powers with User States')
    ax.set_ylabel('Power')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Classification vs User State
    ax = axs[2]
    if feedback_log:
        ax_twin = ax.twinx()
        
        recent_feedback = feedback_log[-60:]
        times = [entry['time_rel'] for entry in recent_feedback]
        levels = [entry['level'] for entry in recent_feedback]
        
        if times:
            # Add background coloring
            add_state_background(ax, times)
            
            ax.plot(times, levels, 'k-', label='Predicted Level', 
                   drawstyle='steps-post', linewidth=2)
            ax.set_ylabel('Prediction Level')
            ax.set_ylim(-3.5, 4.5)
            
            # Plot signal quality
            if quality_log:
                quality_times = []
                quality_scores = []
                for entry in quality_log[-60:]:
                    if session_start_time and entry['time'] >= session_start_time:
                        quality_times.append(entry['time'] - session_start_time)
                        quality_scores.append(entry['metrics']['overall_score'] * 10)
                
                if quality_times:
                    ax_twin.plot(quality_times, quality_scores, 'c-.', 
                               label='Signal Quality', alpha=0.7)
                    ax_twin.set_ylabel('Signal Quality (0-10)')
                    ax_twin.set_ylim(0, 10)
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # State level labels
            state_levels = [-3, -2, -1, 0, 1, 2, 3, 4]
            if session_type == SESSION_TYPE_RELAX:
                state_names = ["Tense", "Alert", "Less Relaxed", "Neutral", 
                             "Slightly Relaxed", "Moderately Relaxed", 
                             "Strongly Relaxed", "Deeply Relaxed"]
            else:
                state_names = ["Distracted", "Less Attentive", "Less Focused", "Neutral", 
                             "Slightly Focused", "Moderately Focused", 
                             "Strongly Focused", "Highly Focused"]
            
            ax.set_yticks(state_levels)
            ax.set_yticklabels(state_names, fontsize=8)
    
    ax.set_title('Mental State Classification vs User Reports')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.1)  # Brief pause to update the plot
    plt.show(block=False)  # Show but don't block execution


def save_session_data():
    """Save session data to disk"""
    print("\n--- Saving session data ---")
    
    # Create directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save raw and processed EEG data
    np.save(os.path.join(SAVE_PATH, "eeg_data_raw.npy"), full_session_eeg_data_raw)
    np.save(os.path.join(SAVE_PATH, "eeg_data_processed.npy"), full_session_eeg_data)
    
    # Save feedback and labels
    import pandas as pd
    
    # Save feedback log
    if feedback_log:
        feedback_df = pd.DataFrame()
        
        # Extract basic fields
        feedback_df['time_abs'] = [entry['time_abs'] for entry in feedback_log]
        feedback_df['time_rel'] = [entry['time_rel'] for entry in feedback_log]
        feedback_df['state'] = [entry['prediction'] for entry in feedback_log]
        feedback_df['level'] = [entry['level'] for entry in feedback_log]
        feedback_df['user_reported_state'] = [entry['user_reported_state'] for entry in feedback_log]
        
        # Extract metrics
        for metric in ['alpha', 'beta', 'theta', 'ab_ratio', 'bt_ratio']:
            feedback_df[metric] = [entry['metrics'][metric] for entry in feedback_log]
        
        feedback_df.to_csv(os.path.join(SAVE_PATH, "feedback_log.csv"), index=False)
    
    # Save user state changes
    if user_state_changes:
        state_changes_df = pd.DataFrame(user_state_changes)
        state_changes_df.to_csv(os.path.join(SAVE_PATH, "user_state_changes.csv"), index=False)
    
    # Save event labels
    if user_labels:
        user_labels_df = pd.DataFrame(user_labels)
        user_labels_df.to_csv(os.path.join(SAVE_PATH, "event_labels.csv"), index=False)
    
    print(f"Session data saved to: {SAVE_PATH}")


def select_session_type():
    """Select session type from command line"""
    global session_type
    
    print("\nSelect Session Type:")
    print("1. RELAXATION")
    print("2. FOCUS")
    
    while True:
        choice = input("Enter choice (1 or 2): ")
        if choice == "1":
            session_type = SESSION_TYPE_RELAX
            break
        elif choice == "2":
            session_type = SESSION_TYPE_FOCUS
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    global running, session_start_time
    
    # Select session type
    select_session_type()
    
    # Connect to LSL
    if not connect_to_lsl():
        print("Failed to connect to LSL. Exiting.")
        return
    
    # Perform calibration in main thread
    if not perform_calibration_phase():
        print("Calibration failed. Exiting.")
        return
    
    # Set initial user state
    update_user_state("Unknown")
    
    try:
        # Start feedback loop (also in main thread)
        feedback_loop()
    except KeyboardInterrupt:
        running = False
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Save data and generate final plots
        if feedback_log:
            save_session_data()
            plot_results()
        
        if lsl_inlet:
            lsl_inlet.close_stream()
        
        print("\nSession complete.")
        
        # Keep plot window open until user closes it
        if plt.get_fignums():
            print("Plots will remain open until you close the window.")
            plt.show(block=True)


if __name__ == "__main__":
    main()