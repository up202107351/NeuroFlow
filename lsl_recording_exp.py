#!/usr/bin/env python3
"""
Enhanced EEG Testing Script

Features:
- Signal quality validation using accelerometer data
- Advanced classification logic from eeg_processing_worker
- Real-time user state labeling via keyboard
- Latency measurement and visualization
- Comprehensive data logging for validation
"""

import time
import numpy as np
import pylsl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import signal
import os
import pickle
import keyboard
from scipy.signal import butter, filtfilt, welch
import threading
import logging
from collections import deque

# Import from backend modules
from backend.signal_quality_validator import SignalQualityValidator, SignalQualityMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EEG_Test')

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

EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10 for Muse
ACC_CHANNEL_INDICES = [9, 10, 11]   # X, Y, Z for accelerometer
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0  # How often to give feedback (shorter for better responsiveness)
PSD_WINDOW_SECONDS = 6.0       # Data length for each individual PSD calculation

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
nyq = 0.5 * sampling_rate
low = lowcut / nyq

# Signal quality related
signal_quality_validator = None

# Butterworth filter design
b_hp, a_hp = butter(filter_order, low, btype='highpass', analog=False)
high = highcut / nyq
b_lp, a_lp = butter(filter_order, high, btype='lowpass', analog=False)

# Session data
baseline_metrics = None
full_session_eeg_data = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
full_session_eeg_data_raw = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
full_session_timestamps_lsl = []

# Feedback and user annotation
feedback_log = [] 
user_labels = []  # List of {"time": timestamp, "label": label_text} dicts
quality_log = []  # List of {"time": timestamp, "metrics": quality_metrics} dicts

# State tracking
previous_states = []
state_momentum = 0.75
state_velocity = 0.0
level_momentum = 0.8
level_velocity = 0.0

# Current session type
session_type = None

# --- Data Saving Configuration ---
SAVE_PATH = "test_session_data/"
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(SAVE_PATH, exist_ok=True)

# --- Visualization ---
plt.ion()  # Enable interactive mode
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.canvas.manager.set_window_title('EEG Testing Interface')

# --- Thresholds from eeg_processing_worker.py ---
THRESHOLDS = {
    'alpha_incr': [1.10, 1.25, 1.50],
    'alpha_decr': [0.90, 0.80, 0.65],
    'beta_incr': [1.10, 1.20, 1.40],
    'beta_decr': [0.90, 0.80],
    'theta_incr': [1.15, 1.30],
    'theta_decr': [0.85, 0.70],
    'ab_ratio_incr': [1.10, 1.20, 1.40],
    'ab_ratio_decr': [0.90, 0.75],
    'bt_ratio_incr': [1.15, 1.30, 1.60],
    'bt_ratio_decr': [0.85, 0.70]
}

# --- User Labeling Setup ---
USER_LABELS = {
    '1': 'Very Relaxed',
    '2': 'Relaxed',
    '3': 'Neutral',
    '4': 'Focused',
    '5': 'Very Focused',
    's': 'START_ACTIVITY',
    'e': 'END_ACTIVITY',
    'b': 'BLINK_ARTIFACT',
    'm': 'MOVEMENT_ARTIFACT',
    'n': 'NOISE_ARTIFACT',
    'c': 'EYES_CLOSED',
    'o': 'EYES_OPEN'
}

print("\n=== EEG Testing Script with Signal Quality Validation ===")
print("\nUser Labels (press key to mark):")
for key, label in USER_LABELS.items():
    print(f"  [{key}] - {label}")
print("\nPress Ctrl+C to end the session")


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
        
        # Create signal quality validator
        signal_quality_validator = SignalQualityValidator()
        
        print(f"Connected to '{info.name()}' @ {sampling_rate:.2f} Hz. NFFT={nfft}, Overlap={welch_overlap_samples}")
        
        # Debug: Print channel info
        print(f"Channel count: {info.channel_count()}")
        print(f"EEG channels: {EEG_CHANNEL_INDICES}")
        
        if info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
            print(f"ERROR: LSL stream has insufficient channels.")
            return False
        return True
        
    except Exception as e:
        print(f"Error connecting to LSL: {e}")
        return False


def filter_eeg_data(eeg_data):
    """Apply bandpass filter to EEG data"""
    min_samples = 3 * filter_order + 1
    
    if eeg_data.shape[1] < min_samples:
        return eeg_data
        
    eeg_filtered = np.zeros_like(eeg_data)
    for i in range(NUM_EEG_CHANNELS):
        # Apply low-pass filter first
        eeg_filtered[i] = filtfilt(b_lp, a_lp, eeg_data[i])
        # Then apply high-pass filter
        eeg_filtered[i] = filtfilt(b_hp, a_hp, eeg_filtered[i])
    
    return eeg_filtered


def calculate_band_powers(eeg_segment):
    """Calculate band powers for EEG segment with artifact rejection"""
    if eeg_segment.shape[1] < nfft:
        return None
    
    # Apply artifact rejection from eeg_processing_worker
    artifact_mask = improved_artifact_rejection(eeg_segment)
    if np.sum(artifact_mask) < 0.7 * eeg_segment.shape[1]:
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
                ch_data, 
                nfft, 
                welch_overlap_samples, 
                int(sampling_rate), 
                WindowOperations.HANNING.value
            )
            
            metrics_list.append({
                'theta': DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1]),
                'alpha': DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1]),
                'beta': DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])
            })
        except Exception as e:
            print(f"PSD calculation error: {e}")
            return None
    
    if len(metrics_list) != NUM_EEG_CHANNELS:
        return None
        
    # Calculate weighted average (assuming equal weights)
    avg_metrics = {
        'theta': np.mean([m['theta'] for m in metrics_list]),
        'alpha': np.mean([m['alpha'] for m in metrics_list]),
        'beta': np.mean([m['beta'] for m in metrics_list])
    }
    
    avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
    avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
    
    return avg_metrics


def improved_artifact_rejection(eeg_data):
    """Advanced artifact rejection from eeg_processing_worker.py"""
    channel_thresholds = [150, 100, 100, 150]
    amplitude_mask = ~np.any(np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1), axis=0)
    
    diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
    if eeg_data.shape[1] > 1:
        diff_thresholds = [50, 30, 30, 50]
        diff_mask = ~np.any(
            np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
            np.array(diff_thresholds).reshape(-1, 1), 
            axis=0
        )
    
    return amplitude_mask & diff_mask


def calculate_state_probabilities(current_metrics):
    """Calculate probabilities for different mental states (from eeg_processing_worker)"""
    if not baseline_metrics:
        return {
            'relaxed': 0.5,
            'focused': 0.5,
            'drowsy': 0.2,
            'internal_focus': 0.3,
            'neutral': 0.7
        }
    
    def sigmoid(x, k=5):
        return 1 / (1 + np.exp(-k * x))
    
    # Calculate normalized differences from baseline
    alpha_diff = current_metrics['alpha'] / baseline_metrics['alpha'] - 1.0
    beta_diff = current_metrics['beta'] / baseline_metrics['beta'] - 1.0
    theta_diff = current_metrics['theta'] / baseline_metrics['theta'] - 1.0
    ab_ratio_diff = current_metrics['ab_ratio'] / baseline_metrics['ab_ratio'] - 1.0 if baseline_metrics['ab_ratio'] > 1e-9 else 0
    bt_ratio_diff = current_metrics['bt_ratio'] / baseline_metrics['bt_ratio'] - 1.0 if baseline_metrics['bt_ratio'] > 1e-9 else 0
    
    # Calculate state probabilities
    relaxation_signal = 0.4 * alpha_diff + 0.6 * ab_ratio_diff
    relaxed_prob = sigmoid(relaxation_signal, k=3)
    
    alpha_inverse_diff = 1.0 - (current_metrics['alpha'] / baseline_metrics['alpha'])
    focus_signal = 0.3 * beta_diff + 0.5 * bt_ratio_diff + 0.2 * max(0, alpha_inverse_diff)
    focused_prob = sigmoid(focus_signal, k=3)
    
    drowsy_signal = 0.6 * theta_diff - 0.4 * beta_diff
    drowsy_prob = sigmoid(drowsy_signal, k=3)
    
    internal_focus_signal = 0.5 * beta_diff + 0.5 * alpha_diff
    internal_focus_prob = sigmoid(internal_focus_signal, k=3)
    
    baseline_closeness = -2.0 * (abs(alpha_diff) + abs(beta_diff) + abs(theta_diff))
    neutral_prob = sigmoid(baseline_closeness, k=5)
    
    return {
        'relaxed': relaxed_prob,
        'focused': focused_prob,
        'drowsy': drowsy_prob,
        'internal_focus': internal_focus_prob,
        'neutral': neutral_prob
    }


def classify_mental_state(current_metrics):
    """Classify mental state based on current metrics (from eeg_processing_worker)"""
    global previous_states, state_velocity, level_velocity, session_type
    
    if not baseline_metrics or not current_metrics:
        return {
            "state": "Calibrating",
            "level": 0,
            "confidence": "N/A",
            "value": 0.5,
            "smooth_value": 0.5,
            "state_key": "calibrating"
        }
    
    # Get state probabilities
    state_probs = calculate_state_probabilities(current_metrics)
    
    # Determine most probable state
    most_probable_state = max(state_probs.items(), key=lambda x: x[1])
    state_name, prob_value = most_probable_state
    
    level = 0
    
    if session_type == SESSION_TYPE_RELAX:
        relaxation_value = min(1.0, max(0.0, state_probs['relaxed']))
        
        # Calculate alert signal
        ab_ratio_decrease = baseline_metrics['ab_ratio'] / current_metrics['ab_ratio'] - 1.0 if current_metrics['ab_ratio'] > 0 else 0
        beta_increase = current_metrics['beta'] / baseline_metrics['beta'] - 1.0
        alpha_decrease = 1.0 - current_metrics['alpha'] / baseline_metrics['alpha']
        
        alert_signal = (0.4 * ab_ratio_decrease + 0.3 * beta_increase + 0.3 * alpha_decrease) * 4.0
        
        # Determine relaxation levels
        if (ab_ratio_decrease > 0.1 and (beta_increase > 0.1 or alpha_decrease > 0.1)):
            if alert_signal > 1.5:
                level = -3
                state_name = "tense"
            elif alert_signal > 1.0:
                level = -2
                state_name = "alert"
            else:
                level = -1
                state_name = "less_relaxed"
        elif relaxation_value > 0.3:
            alpha_increase = current_metrics['alpha'] / baseline_metrics['alpha'] - 1.0
            ab_ratio_increase = current_metrics['ab_ratio'] / baseline_metrics['ab_ratio'] - 1.0 if baseline_metrics['ab_ratio'] > 0 else 0
            relax_signal = (0.5 * alpha_increase + 0.5 * ab_ratio_increase) * 5.0
            
            if relax_signal > 0.5:
                level = 4
                state_name = "deeply_relaxed"
            elif relax_signal > 0.25:
                level = 3
                state_name = "strongly_relaxed"
            elif relax_signal > 0.1:
                level = 2
                state_name = "moderately_relaxed"
            else:
                level = 1
                state_name = "slightly_relaxed"
        else:
            level = 0
            state_name = "neutral"
        
        value = (level + 3) / 7.0
        
    elif session_type == SESSION_TYPE_FOCUS:
        # Focus classification logic
        focus_value = min(1.0, max(0.0, state_probs['focused']))
        
        # Calculate focus and distraction signals
        bt_ratio_decrease = baseline_metrics['bt_ratio'] / current_metrics['bt_ratio'] - 1.0 if current_metrics['bt_ratio'] > 0 else 0
        beta_decrease = 1.0 - current_metrics['beta'] / baseline_metrics['beta']
        theta_increase = current_metrics['theta'] / baseline_metrics['theta'] - 1.0
        
        distraction_signal = (0.4 * bt_ratio_decrease + 0.3 * beta_decrease + 0.3 * theta_increase) * 4.0
        
        # Determine focus levels
        if (bt_ratio_decrease > 0.1 and (beta_decrease > 0.1 or theta_increase > 0.1)):
            if distraction_signal > 1.5:
                level = -3
                state_name = "distracted"
            elif distraction_signal > 1.0:
                level = -2
                state_name = "less_attentive"
            else:
                level = -1
                state_name = "less_focused"
        elif focus_value > 0.3:
            beta_increase = current_metrics['beta'] / baseline_metrics['beta'] - 1.0
            bt_ratio_increase = current_metrics['bt_ratio'] / baseline_metrics['bt_ratio'] - 1.0 if baseline_metrics['bt_ratio'] > 0 else 0
            focus_signal = (0.5 * beta_increase + 0.5 * bt_ratio_increase) * 5.0
            
            if focus_signal > 0.5:
                level = 4
                state_name = "highly_focused"
            elif focus_signal > 0.25:
                level = 3
                state_name = "strongly_focused"
            elif focus_signal > 0.1:
                level = 2
                state_name = "moderately_focused"
            else:
                level = 1
                state_name = "slightly_focused"
        else:
            level = 0
            state_name = "neutral"
        
        value = (level + 3) / 7.0
    else:
        value = prob_value
    
    # Determine confidence
    if prob_value > 0.8:
        confidence = "high"
    elif prob_value > 0.65:
        confidence = "medium"
    elif prob_value > 0.55:
        confidence = "low"
    else:
        confidence = "very_low"
    
    # Apply temporal smoothing
    smooth_value = value
    
    previous_states.append((state_name, value, level))
    if len(previous_states) > 5:  # MAX_STATE_HISTORY
        previous_states.pop(0)
    
    if len(previous_states) > 1:
        current_value = value
        recent_values = [s[1] for s in previous_states]
        prev_value = recent_values[-2]
        
        target_velocity = current_value - prev_value
        state_velocity = (state_velocity * state_momentum + 
                        target_velocity * (1 - state_momentum))
        smooth_value = prev_value + state_velocity
        smooth_value = min(1.0, max(0.0, smooth_value))
    
    # Map state names to display names
    state_display_map = {
        'deeply_relaxed': "Deeply Relaxed",
        'strongly_relaxed': "Strongly Relaxed",
        'moderately_relaxed': "Moderately Relaxed",
        'slightly_relaxed': "Slightly Relaxed",
        'neutral': "Neutral",
        'less_relaxed': "Less Relaxed",
        'alert': "Alert",
        'tense': "Tense",
        'calibrating': "Calibrating",
        'highly_focused': "Highly Focused",
        'strongly_focused': "Strongly Focused",
        'moderately_focused': "Moderately Focused",
        'slightly_focused': "Slightly Focused",
        'less_focused': "Less Focused",
        'less_attentive': "Less Attentive",
        'distracted': "Distracted"
    }
    
    display_state = state_display_map.get(state_name, state_name.title())
    
    return {
        "state": display_state,
        "state_key": state_name,
        "level": level,
        "confidence": confidence,
        "value": round(value, 3),
        "smooth_value": round(smooth_value, 3),
        "probabilities": state_probs
    }


def detect_eyes_closed(metrics, baseline_alpha, baseline_ab_ratio):
    """
    Eyes closed detection (improved from lsl_recording_exp.py)
    """
    is_likely_eyes_closed = (
        metrics['alpha'] > baseline_alpha * 2.0 and  # Alpha doubled
        metrics['ab_ratio'] > baseline_ab_ratio * 1.5 and  # A/B ratio increased 50%+
        metrics['theta'] < metrics['theta'] * 1.2  # Theta didn't increase significantly
    )
    
    return is_likely_eyes_closed


def perform_calibration_phase():
    """Calibration phase with signal quality monitoring"""
    global baseline_metrics, signal_quality_validator
    
    print(f"\n--- Starting {CALIBRATION_DURATION_SECONDS:.0f} Second Calibration ---")
    print("Please remain in a neutral, resting state.")
    
    calibration_start_time = time.time()
    calibration_metrics_list = []
    
    # Process all metrics during calibration time
    while (time.time() - calibration_start_time < CALIBRATION_DURATION_SECONDS and running):
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
        
        if not chunk:
            time.sleep(0.01)
            continue
            
        # Convert to numpy array
        chunk_np = np.array(chunk, dtype=np.float64).T
        
        # Check if we have enough channels
        if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
            print(f"Not enough channels in chunk: {chunk_np.shape[0]}")
            continue
            
        # Extract EEG data
        eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
        
        # Extract accelerometer data if available
        if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
            try:
                acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                if acc_chunk.shape[1] > 0:
                    latest_acc_sample = acc_chunk[:, -1]
                    signal_quality_validator.add_accelerometer_data(latest_acc_sample)
            except Exception as e:
                print(f"Could not extract accelerometer data: {e}")
        
        # Process EEG data
        eeg_chunk_filtered = filter_eeg_data(eeg_chunk)
        
        # Add to global buffer for visualization
        global full_session_eeg_data, full_session_eeg_data_raw, full_session_timestamps_lsl
        full_session_eeg_data = np.append(full_session_eeg_data, eeg_chunk_filtered, axis=1)
        full_session_eeg_data_raw = np.append(full_session_eeg_data_raw, eeg_chunk, axis=1)
        
        if timestamps:
            full_session_timestamps_lsl.extend(timestamps)
        
        # If we have enough data for PSD calculation
        if eeg_chunk_filtered.shape[1] >= nfft:
            segment = eeg_chunk_filtered[:, -nfft:]
            metrics = calculate_band_powers(segment)
            
            if metrics:
                # Add to quality validator
                signal_quality_validator.add_band_power_data(metrics)
                signal_quality_validator.add_raw_eeg_data(segment)
                
                calibration_metrics_list.append(metrics)
                
                # Check signal quality every second
                if len(calibration_metrics_list) % 5 == 0:  # Assuming ~5 calculations per second
                    quality = signal_quality_validator.assess_overall_quality()
                    print(f"Calibration progress: {((time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS) * 100:.0f}% - "
                          f"Quality: {quality.quality_level} ({quality.overall_score:.2f})")
                    
                    # Store quality info
                    quality_log.append({
                        "time": time.time(),
                        "metrics": {
                            "movement_score": quality.movement_score,
                            "band_power_score": quality.band_power_score,
                            "electrode_contact_score": quality.electrode_contact_score,
                            "overall_score": quality.overall_score,
                            "quality_level": quality.quality_level,
                            "recommendations": quality.recommendations
                        }
                    })
                    
                    # If quality is very poor, alert user
                    if quality.overall_score < 0.3:
                        print(f"WARNING: Poor signal quality - {quality.recommendations[0] if quality.recommendations else 'Please adjust headband'}")
    
    # Calculate baseline from calibration data
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
        return False


def monitor_keyboard_input():
    """Monitor keyboard for user labels"""
    global running, user_labels
    
    while running:
        for key in USER_LABELS:
            if keyboard.is_pressed(key):
                label = USER_LABELS[key]
                timestamp = time.time()
                user_labels.append({
                    "time": timestamp,
                    "label": label
                })
                print(f"\n>>> User Label: {label} at {timestamp:.2f}s")
                time.sleep(0.5)  # Prevent multiple detections
        time.sleep(0.1)


def feedback_loop():
    """Real-time feedback loop with signal quality monitoring and user labeling"""
    global baseline_metrics, running, feedback_log, quality_log
    
    if not baseline_metrics:
        print("Cannot start feedback loop: baseline not calibrated.")
        return
    
    print(f"\n--- Starting Real-time Feedback (updates every {ANALYSIS_WINDOW_SECONDS:.1f}s) ---")
    session_start_time = time.time()  # Start time of feedback phase
    
    # Start keyboard monitoring thread
    keyboard_thread = threading.Thread(target=monitor_keyboard_input)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    # Buffer for more stable processing
    eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
    
    # Initial visualization
    update_visualization()
    
    while running:
        loop_start_time = time.time()
        
        # Pull data chunk
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
        
        if not chunk:
            time.sleep(0.01)
            continue
            
        # Convert to numpy array
        chunk_np = np.array(chunk, dtype=np.float64).T
        
        # Check if we have enough channels
        if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
            continue
            
        # Extract EEG data
        eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
        
        # Extract accelerometer data if available
        if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
            try:
                acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                if acc_chunk.shape[1] > 0:
                    latest_acc_sample = acc_chunk[:, -1]
                    signal_quality_validator.add_accelerometer_data(latest_acc_sample)
            except Exception as e:
                pass  # Non-critical error
        
        # Process EEG data
        eeg_chunk_filtered = filter_eeg_data(eeg_chunk)
        
        # Add to global buffer for visualization
        global full_session_eeg_data, full_session_eeg_data_raw, full_session_timestamps_lsl
        full_session_eeg_data = np.append(full_session_eeg_data, eeg_chunk_filtered, axis=1)
        full_session_eeg_data_raw = np.append(full_session_eeg_data_raw, eeg_chunk, axis=1)
        
        if timestamps:
            full_session_timestamps_lsl.extend(timestamps)
        
        # Add to local buffer
        eeg_buffer = np.append(eeg_buffer, eeg_chunk_filtered, axis=1)
        
        # Only process if enough time has passed
        elapsed = time.time() - loop_start_time
        if elapsed >= ANALYSIS_WINDOW_SECONDS:
            # If we have enough data for PSD calculation
            if eeg_buffer.shape[1] >= nfft:
                segment = eeg_buffer[:, -nfft:]
                current_metrics = calculate_band_powers(segment)
                
                if current_metrics:
                    # Add to quality validator
                    signal_quality_validator.add_band_power_data(current_metrics)
                    signal_quality_validator.add_raw_eeg_data(segment)
                    
                    # Get signal quality assessment
                    quality = signal_quality_validator.assess_overall_quality()
                    
                    # Classify mental state
                    classification = classify_mental_state(current_metrics)
                    
                    # Log states
                    time_rel_feedback = time.time() - session_start_time
                    feedback_log.append({
                        "time_abs": time.time(),
                        "time_rel": time_rel_feedback,
                        "metrics": current_metrics,
                        "prediction": classification["state"],
                        "state_key": classification["state_key"],
                        "level": classification["level"],
                        "confidence": classification["confidence"],
                        "smooth_value": classification["smooth_value"],
                        "probabilities": classification["probabilities"],
                        "signal_quality": quality.quality_level
                    })
                    
                    # Log quality
                    quality_log.append({
                        "time": time.time(),
                        "metrics": {
                            "movement_score": quality.movement_score,
                            "band_power_score": quality.band_power_score,
                            "electrode_contact_score": quality.electrode_contact_score,
                            "overall_score": quality.overall_score,
                            "quality_level": quality.quality_level,
                            "recommendations": quality.recommendations
                        }
                    })
                    
                    # Print feedback
                    print(f"\nFeedback @ {time_rel_feedback:6.1f}s: {classification['state']} (Level {classification['level']}, Confidence: {classification['confidence']})")
                    print(f"  Metrics: A/B {current_metrics['ab_ratio']:.2f}(B:{baseline_metrics['ab_ratio']:.2f}), "
                          f"B/T {current_metrics['bt_ratio']:.2f}(B:{baseline_metrics['bt_ratio']:.2f}), "
                          f"Alpha {current_metrics['alpha']:.1f}(B:{baseline_metrics['alpha']:.1f})")
                    print(f"  Quality: {quality.quality_level} ({quality.overall_score:.2f})")
                    
                    if quality.recommendations:
                        print(f"  Recommendations: {quality.recommendations[0]}")
                    
                    # Update visualization
                    if len(feedback_log) % 4 == 0:  # Don't update too frequently
                        update_visualization()
                    
                # Reset buffer with overlap
                eeg_buffer = eeg_buffer[:, -int(nfft * 0.5):]
            
            loop_start_time = time.time()  # Reset loop timer


def update_visualization():
    """Update the visualization plots"""
    plt.figure(fig.number)
    
    # Clear all axes
    for ax in axs:
        ax.clear()
    
    # Plot 1: Raw EEG for most recent 5 seconds
    ax = axs[0]
    window_samples = int(sampling_rate * 5)  # 5 seconds
    if full_session_eeg_data.shape[1] > window_samples:
        recent_data = full_session_eeg_data[:, -window_samples:]
        time_axis = np.linspace(-5, 0, recent_data.shape[1])
        
        for i in range(NUM_EEG_CHANNELS):
            ax.plot(time_axis, recent_data[i], label=f'Channel {i+1}')
    else:
        ax.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('Real-time EEG')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Plot 2: Band Powers
    ax = axs[1]
    if feedback_log:
        # Extract last 20 entries
        recent_feedback = feedback_log[-30:]
        times = [entry['time_rel'] for entry in recent_feedback]
        baseline_time = [times[0], times[-1]]
        
        # Plot band powers
        ax.plot(times, [m['metrics']['alpha'] for m in recent_feedback], label='Alpha', c='blue')
        ax.plot(times, [m['metrics']['beta'] for m in recent_feedback], label='Beta', c='red')
        ax.plot(times, [m['metrics']['theta'] for m in recent_feedback], label='Theta', c='green')
        
        # Plot baselines
        if baseline_metrics:
            ax.plot(baseline_time, [baseline_metrics['alpha'], baseline_metrics['alpha']], 'b--', alpha=0.5)
            ax.plot(baseline_time, [baseline_metrics['beta'], baseline_metrics['beta']], 'r--', alpha=0.5)
            ax.plot(baseline_time, [baseline_metrics['theta'], baseline_metrics['theta']], 'g--', alpha=0.5)
            
        # Add user labels
        for label_event in user_labels:
            if label_event['time'] >= session_start_time and label_event['time'] - session_start_time <= times[-1]:
                label_time = label_event['time'] - session_start_time
                ax.axvline(x=label_time, color='purple', linestyle=':', alpha=0.7)
                ax.text(label_time, ax.get_ylim()[1] * 0.9, label_event['label'], 
                        rotation=90, ha='right', va='top', alpha=0.8, fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Waiting for feedback...', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('Band Powers')
    ax.set_ylabel('Power')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Plot 3: Classification and Signal Quality
    ax = axs[2]
    if feedback_log:
        # Create twin axis
        ax_twin = ax.twinx()
        
        # Extract data
        times = [entry['time_rel'] for entry in feedback_log[-30:]]
        levels = [entry['level'] for entry in feedback_log[-30:]]
        smooth_values = [entry['smooth_value'] * 10 for entry in feedback_log[-30:]]  # Scale to 0-10
        
        # Plot classification level
        ax.plot(times, levels, 'k-', label='State Level', drawstyle='steps-post')
        ax.set_ylabel('State Level')
        ax.set_ylim(-3.5, 4.5)
        
        # Plot signal quality
        if quality_log:
            quality_times = [entry['time'] - session_start_time for entry in quality_log[-30:]]
            quality_scores = [entry['metrics']['overall_score'] * 10 for entry in quality_log[-30:]]  # Scale to 0-10
            ax_twin.plot(quality_times, quality_scores, 'c-.', label='Signal Quality', alpha=0.7)
            ax_twin.set_ylabel('Signal Quality (0-10)')
            ax_twin.set_ylim(0, 10)
            
        # Create combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add state labels on y-axis
        state_levels = [-3, -2, -1, 0, 1, 2, 3, 4]
        state_names = [
            "Tense/Distracted" if session_type == SESSION_TYPE_RELAX else "Distracted",
            "Alert/Less Attentive" if session_type == SESSION_TYPE_RELAX else "Less Attentive",
            "Less Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Less Focused",
            "Neutral",
            "Slightly Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Slightly Focused",
            "Moderately Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Moderately Focused",
            "Strongly Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Strongly Focused",
            "Deeply Relaxed/Highly Focused" if session_type == SESSION_TYPE_RELAX else "Highly Focused"
        ]
        ax.set_yticks(state_levels)
        ax.set_yticklabels(state_names, fontsize=8)
        
        # Add user labels
        for label_event in user_labels:
            if label_event['time'] >= session_start_time and label_event['time'] - session_start_time <= times[-1]:
                label_time = label_event['time'] - session_start_time
                ax.axvline(x=label_time, color='purple', linestyle=':', alpha=0.7)
                ax.text(label_time, ax.get_ylim()[1] * 0.9, label_event['label'], 
                        rotation=90, ha='right', va='top', alpha=0.8, fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Waiting for classification...', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('Mental State Classification & Signal Quality')
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # Small delay to update the plot


def save_session_data():
    """Save all session data to disk for later analysis"""
    global full_session_eeg_data, full_session_eeg_data_raw, full_session_timestamps_lsl, feedback_log, user_labels, quality_log, baseline_metrics
    
    # Create timestamp for filenames
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save raw and processed EEG data
    np.save(os.path.join(SAVE_PATH, f"eeg_data_raw_{timestamp_str}.npy"), full_session_eeg_data_raw)
    np.save(os.path.join(SAVE_PATH, f"eeg_data_processed_{timestamp_str}.npy"), full_session_eeg_data)
    np.save(os.path.join(SAVE_PATH, f"timestamps_{timestamp_str}.npy"), np.array(full_session_timestamps_lsl))
    
    # Save baseline metrics
    with open(os.path.join(SAVE_PATH, f"baseline_metrics_{timestamp_str}.pkl"), 'wb') as f:
        pickle.dump(baseline_metrics, f)
    
    # Save feedback log to CSV
    if feedback_log:
        # Create a flattened DataFrame
        feedback_df = pd.DataFrame()
        
        # Extract basic fields
        feedback_df['time_abs'] = [entry['time_abs'] for entry in feedback_log]
        feedback_df['time_rel'] = [entry['time_rel'] for entry in feedback_log]
        feedback_df['state'] = [entry['prediction'] for entry in feedback_log]
        feedback_df['state_key'] = [entry['state_key'] for entry in feedback_log]
        feedback_df['level'] = [entry['level'] for entry in feedback_log]
        feedback_df['confidence'] = [entry['confidence'] for entry in feedback_log]
        feedback_df['smooth_value'] = [entry['smooth_value'] for entry in feedback_log]
        feedback_df['signal_quality'] = [entry['signal_quality'] for entry in feedback_log]
        
        # Extract metrics
        feedback_df['alpha'] = [entry['metrics']['alpha'] for entry in feedback_log]
        feedback_df['beta'] = [entry['metrics']['beta'] for entry in feedback_log]
        feedback_df['theta'] = [entry['metrics']['theta'] for entry in feedback_log]
        feedback_df['ab_ratio'] = [entry['metrics']['ab_ratio'] for entry in feedback_log]
        feedback_df['bt_ratio'] = [entry['metrics']['bt_ratio'] for entry in feedback_log]
        
        # Extract probabilities
        if 'probabilities' in feedback_log[0]:
            feedback_df['p_relaxed'] = [entry['probabilities']['relaxed'] for entry in feedback_log]
            feedback_df['p_focused'] = [entry['probabilities']['focused'] for entry in feedback_log]
            feedback_df['p_drowsy'] = [entry['probabilities']['drowsy'] for entry in feedback_log]
            feedback_df['p_internal_focus'] = [entry['probabilities']['internal_focus'] for entry in feedback_log]
            feedback_df['p_neutral'] = [entry['probabilities']['neutral'] for entry in feedback_log]
        
        # Save to CSV
        feedback_df.to_csv(os.path.join(SAVE_PATH, f"feedback_log_{timestamp_str}.csv"), index=False)
    
    # Save user labels to CSV
    if user_labels:
        user_labels_df = pd.DataFrame(user_labels)
        user_labels_df.to_csv(os.path.join(SAVE_PATH, f"user_labels_{timestamp_str}.csv"), index=False)
    
    # Save quality log to CSV
    if quality_log:
        quality_df = pd.DataFrame()
        
        quality_df['time'] = [entry['time'] for entry in quality_log]
        quality_df['movement_score'] = [entry['metrics']['movement_score'] for entry in quality_log]
        quality_df['band_power_score'] = [entry['metrics']['band_power_score'] for entry in quality_log]
        quality_df['electrode_contact_score'] = [entry['metrics']['electrode_contact_score'] for entry in quality_log]
        quality_df['overall_score'] = [entry['metrics']['overall_score'] for entry in quality_log]
        quality_df['quality_level'] = [entry['metrics']['quality_level'] for entry in quality_log]
        
        # Extract first recommendation if available
        quality_df['recommendation'] = [
            entry['metrics']['recommendations'][0] if entry['metrics']['recommendations'] else ""
            for entry in quality_log
        ]
        
        quality_df.to_csv(os.path.join(SAVE_PATH, f"quality_log_{timestamp_str}.csv"), index=False)
    
    # Save metadata
    metadata = {
        'session_type': session_type,
        'timestamp': timestamp_str,
        'sampling_rate': sampling_rate,
        'calibration_duration': CALIBRATION_DURATION_SECONDS,
        'analysis_window': ANALYSIS_WINDOW_SECONDS,
        'psd_window': PSD_WINDOW_SECONDS,
        'thresholds': THRESHOLDS
    }
    
    with open(os.path.join(SAVE_PATH, f"metadata_{timestamp_str}.json"), 'w') as f:
        import json
        json.dump(metadata, f, indent=4)
    
    print(f"\nSession data saved to: {SAVE_PATH} at {datetime.now().strftime('%H:%M:%S')}")


def create_analysis_plots():
    """Create comprehensive plots for analysis"""
    if full_session_eeg_data.size == 0 and not feedback_log:
        print("No data recorded to plot.")
        return
    
    print("\n--- Generating Analysis Plots ---")
    
    # Create a new figure for the analysis
    fig_analysis = plt.figure(figsize=(15, 14))
    gs = plt.GridSpec(5, 2, figure=fig_analysis)
    
    # 1. Raw Data Plot (Top Left)
    ax1 = fig_analysis.add_subplot(gs[0, 0])
    time_axis = np.arange(full_session_eeg_data.shape[1]) / sampling_rate
    for i in range(NUM_EEG_CHANNELS):
        ax1.plot(time_axis, full_session_eeg_data[i], label=f'Ch {i+1}')
    ax1.set_title('Filtered EEG Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV)')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True)
    
    # 2. Power Spectrum (Top Right)
    ax2 = fig_analysis.add_subplot(gs[0, 1])
    for i in range(NUM_EEG_CHANNELS):
        frequencies, power_spectral_density = welch(
            full_session_eeg_data[i], fs=sampling_rate, 
            nperseg=int(sampling_rate * 4), noverlap=int(sampling_rate * 2)
        )
        ax2.semilogy(frequencies, power_spectral_density, label=f'Ch {i+1}')
        
    ax2.set_title('Power Spectrum (Entire Session)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (µV²/Hz)')
    ax2.set_xlim(0, 40)  # Only show up to 40Hz
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True)
    
    # 3. Band Powers Over Time (Middle Left)
    if feedback_log:
        ax3 = fig_analysis.add_subplot(gs[1, 0])
        times = [entry['time_rel'] for entry in feedback_log]
        
        ax3.plot(times, [m['metrics']['alpha'] for m in feedback_log], label='Alpha', c='blue')
        ax3.plot(times, [m['metrics']['beta'] for m in feedback_log], label='Beta', c='red')
        ax3.plot(times, [m['metrics']['theta'] for m in feedback_log], label='Theta', c='green')
        
        # Add baseline lines
        if baseline_metrics:
            ax3.axhline(y=baseline_metrics['alpha'], color='blue', linestyle='--', alpha=0.5, label='Alpha Base')
            ax3.axhline(y=baseline_metrics['beta'], color='red', linestyle='--', alpha=0.5, label='Beta Base')
            ax3.axhline(y=baseline_metrics['theta'], color='green', linestyle='--', alpha=0.5, label='Theta Base')
        
        ax3.set_title('Band Powers Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power')
        ax3.legend(loc='upper right', fontsize='small')
        ax3.grid(True)
    
    # 4. Alpha/Beta and Beta/Theta Ratios (Middle Right)
    if feedback_log:
        ax4 = fig_analysis.add_subplot(gs[1, 1])
        ax4_twin = ax4.twinx()
        
        times = [entry['time_rel'] for entry in feedback_log]
        
        # Alpha/Beta ratio
        ax4.plot(times, [m['metrics']['ab_ratio'] for m in feedback_log], label='A/B Ratio', c='purple')
        if baseline_metrics:
            ax4.axhline(y=baseline_metrics['ab_ratio'], color='purple', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Alpha/Beta Ratio', color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')
        
        # Beta/Theta ratio
        ax4_twin.plot(times, [m['metrics']['bt_ratio'] for m in feedback_log], label='B/T Ratio', c='orange')
        if baseline_metrics:
            ax4_twin.axhline(y=baseline_metrics['bt_ratio'], color='orange', linestyle='--', alpha=0.5)
        ax4_twin.set_ylabel('Beta/Theta Ratio', color='orange')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        
        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')
        
        ax4.set_title('Frequency Band Ratios')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True)
    
    # 5. Mental State Classification (Bottom Left)
    if feedback_log:
        ax5 = fig_analysis.add_subplot(gs[2, 0])
        
        times = [entry['time_rel'] for entry in feedback_log]
        levels = [entry['level'] for entry in feedback_log]
        
        ax5.plot(times, levels, 'k-', drawstyle='steps-post')
        
        # Add user labels
        for label_event in user_labels:
            if label_event['time'] >= session_start_time:
                label_time = label_event['time'] - session_start_time
                ax5.axvline(x=label_time, color='purple', linestyle=':', alpha=0.7)
                ax5.text(label_time, ax5.get_ylim()[1] * 0.9, label_event['label'], 
                        rotation=90, ha='right', va='top', alpha=0.8, fontsize=8)
        
        # State levels
        state_levels = [-3, -2, -1, 0, 1, 2, 3, 4]
        state_names = [
            "Tense/Distracted" if session_type == SESSION_TYPE_RELAX else "Distracted",
            "Alert/Less Attentive" if session_type == SESSION_TYPE_RELAX else "Less Attentive",
            "Less Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Less Focused",
            "Neutral",
            "Slightly Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Slightly Focused",
            "Moderately Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Moderately Focused",
            "Strongly Relaxed/Focused" if session_type == SESSION_TYPE_RELAX else "Strongly Focused",
            "Deeply Relaxed/Highly Focused" if session_type == SESSION_TYPE_RELAX else "Highly Focused"
        ]
        ax5.set_yticks(state_levels)
        ax5.set_yticklabels(state_names, fontsize=8)
        
        ax5.set_title('Mental State Classification')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('State Level')
        ax5.grid(True)
    
    # 6. Signal Quality (Bottom Right)
    if quality_log:
        ax6 = fig_analysis.add_subplot(gs[2, 1])
        
        quality_times = [entry['time'] - session_start_time for entry in quality_log]
        overall_scores = [entry['metrics']['overall_score'] for entry in quality_log]
        movement_scores = [entry['metrics']['movement_score'] for entry in quality_log]
        bp_scores = [entry['metrics']['band_power_score'] for entry in quality_log]
        electrode_scores = [entry['metrics']['electrode_contact_score'] for entry in quality_log]
        
        ax6.plot(quality_times, overall_scores, 'k-', label='Overall Quality')
        ax6.plot(quality_times, movement_scores, 'b-', alpha=0.6, label='Movement')
        ax6.plot(quality_times, bp_scores, 'g-', alpha=0.6, label='Band Power')
        ax6.plot(quality_times, electrode_scores, 'r-', alpha=0.6, label='Electrode')
        
        # Add horizontal lines for quality thresholds
        ax6.axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='Good')
        ax6.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, label='Poor')
        
        ax6.set_title('Signal Quality Metrics')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Quality Score (0-1)')
        ax6.set_ylim(0, 1)
        ax6.legend(loc='lower right', fontsize='small')
        ax6.grid(True)
    
    # 7. Latency Analysis (Bottom)
    if feedback_log and full_session_timestamps_lsl:
        ax7 = fig_analysis.add_subplot(gs[3, :])
        
        # Calculate and plot latencies between LSL timestamps and processing
        lsl_times = np.array(full_session_timestamps_lsl)
        processing_times = np.array([entry['time_abs'] for entry in feedback_log])
        
        # Find closest LSL timestamp for each processing timestamp
        latencies = []
        for proc_time in processing_times:
            closest_idx = np.argmin(np.abs(lsl_times - proc_time))
            latency = (proc_time - lsl_times[closest_idx]) * 1000  # in ms
            latencies.append(latency)
        
        # Plot latency histogram
        ax7.hist(latencies, bins=20, alpha=0.7)
        ax7.axvline(x=np.mean(latencies), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(latencies):.1f} ms')
        
        ax7.set_title('Processing Latency Distribution')
        ax7.set_xlabel('Latency (ms)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True)
    
    # 8. State Probability Analysis
    if feedback_log and 'probabilities' in feedback_log[0]:
        ax8 = fig_analysis.add_subplot(gs[4, :])
        
        times = [entry['time_rel'] for entry in feedback_log]
        
        # Plot all state probabilities
        ax8.plot(times, [entry['probabilities']['relaxed'] for entry in feedback_log], 
                label='Relaxed', color='blue')
        ax8.plot(times, [entry['probabilities']['focused'] for entry in feedback_log],
                label='Focused', color='red')
        ax8.plot(times, [entry['probabilities']['drowsy'] for entry in feedback_log],
                label='Drowsy', color='purple')
        ax8.plot(times, [entry['probabilities']['internal_focus'] for entry in feedback_log],
                label='Internal Focus', color='orange')
        ax8.plot(times, [entry['probabilities']['neutral'] for entry in feedback_log],
                label='Neutral', color='green')
        
        # Add user labels
        for label_event in user_labels:
            if label_event['time'] >= session_start_time:
                label_time = label_event['time'] - session_start_time
                ax8.axvline(x=label_time, color='black', linestyle=':', alpha=0.7)
                ax8.text(label_time, ax8.get_ylim()[1] * 0.9, label_event['label'], 
                        rotation=90, ha='right', va='top', alpha=0.8, fontsize=8)
        
        ax8.set_title('Mental State Probabilities')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Probability')
        ax8.set_ylim(0, 1)
        ax8.legend(loc='upper right')
        ax8.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"analysis_plots_{timestamp_str}.png"), dpi=300)
    plt.show()


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
    global running, lsl_inlet, session_start_time, full_session_eeg_data, full_session_eeg_