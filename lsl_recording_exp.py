#!/usr/bin/env python3
"""
Enhanced EEG Testing Script v2 - Final Threading Fix

Features:
- Matplotlib in main thread with timer-based updates
- Persistent user state labeling with visual feedback
- Advanced classification validation with accuracy metrics
- Signal quality validation using accelerometer data
- Real-time visualization with color-coded user states
- Comprehensive analysis and reporting
"""

import time
import numpy as np
import pylsl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import signal
import os
import pickle
import keyboard
from scipy.signal import butter, filtfilt, welch
import threading
import logging
from collections import deque
import queue

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

# Threading control
data_processing_thread = None
calibration_complete = False

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

print("\n=== Enhanced EEG Testing Script v2 ===")
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
    """Calculate band powers for EEG segment with artifact rejection"""
    if eeg_segment.shape[1] < nfft:
        return None
    
    # Apply artifact rejection
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
                ch_data, nfft, welch_overlap_samples, int(sampling_rate), 
                WindowOperations.HANNING.value
            )
            
            metrics_list.append({
                'theta': DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1]),
                'alpha': DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1]),
                'beta': DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])
            })
        except Exception as e:
            return None
    
    if len(metrics_list) != NUM_EEG_CHANNELS:
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
    """Advanced artifact rejection"""
    channel_thresholds = [150, 100, 100, 150]
    amplitude_mask = ~np.any(np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1), axis=0)
    
    diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
    if eeg_data.shape[1] > 1:
        diff_thresholds = [50, 30, 30, 50]
        diff_mask = ~np.any(
            np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
            np.array(diff_thresholds).reshape(-1, 1), axis=0
        )
    
    return amplitude_mask & diff_mask


def calculate_state_probabilities(current_metrics):
    """Calculate probabilities for different mental states"""
    if not baseline_metrics:
        return {'relaxed': 0.5, 'focused': 0.5, 'drowsy': 0.2, 'internal_focus': 0.3, 'neutral': 0.7}
    
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
        'relaxed': relaxed_prob, 'focused': focused_prob, 'drowsy': drowsy_prob,
        'internal_focus': internal_focus_prob, 'neutral': neutral_prob
    }


def classify_mental_state(current_metrics):
    """Classify mental state based on current metrics"""
    global previous_states, state_velocity, session_type
    
    if not baseline_metrics or not current_metrics:
        return {
            "state": "Calibrating", "level": 0, "confidence": "N/A",
            "value": 0.5, "smooth_value": 0.5, "state_key": "calibrating"
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
                level = -3; state_name = "tense"
            elif alert_signal > 1.0:
                level = -2; state_name = "alert"
            else:
                level = -1; state_name = "less_relaxed"
        elif relaxation_value > 0.3:
            alpha_increase = current_metrics['alpha'] / baseline_metrics['alpha'] - 1.0
            ab_ratio_increase = current_metrics['ab_ratio'] / baseline_metrics['ab_ratio'] - 1.0 if baseline_metrics['ab_ratio'] > 0 else 0
            relax_signal = (0.5 * alpha_increase + 0.5 * ab_ratio_increase) * 5.0
            
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
        focus_value = min(1.0, max(0.0, state_probs['focused']))
        
        # Calculate focus and distraction signals
        bt_ratio_decrease = baseline_metrics['bt_ratio'] / current_metrics['bt_ratio'] - 1.0 if current_metrics['bt_ratio'] > 0 else 0
        beta_decrease = 1.0 - current_metrics['beta'] / baseline_metrics['beta']
        theta_increase = current_metrics['theta'] / baseline_metrics['theta'] - 1.0
        
        distraction_signal = (0.4 * bt_ratio_decrease + 0.3 * beta_decrease + 0.3 * theta_increase) * 4.0
        
        # Determine focus levels
        if (bt_ratio_decrease > 0.1 and (beta_decrease > 0.1 or theta_increase > 0.1)):
            if distraction_signal > 1.5:
                level = -3; state_name = "distracted"
            elif distraction_signal > 1.0:
                level = -2; state_name = "less_attentive"
            else:
                level = -1; state_name = "less_focused"
        elif focus_value > 0.3:
            beta_increase = current_metrics['beta'] / baseline_metrics['beta'] - 1.0
            bt_ratio_increase = current_metrics['bt_ratio'] / baseline_metrics['bt_ratio'] - 1.0 if baseline_metrics['bt_ratio'] > 0 else 0
            focus_signal = (0.5 * beta_increase + 0.5 * bt_ratio_increase) * 5.0
            
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
        "confidence": confidence, "value": round(value, 3),
        "smooth_value": round(smooth_value, 3), "probabilities": state_probs
    }


def data_processing_thread_function():
    """Thread function for processing EEG data"""
    global baseline_metrics, running, feedback_log, session_start_time, calibration_complete
    
    eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
    last_analysis_time = 0
    
    while running:
        try:
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
            
            # Process if enough time has passed and we have baseline
            current_time = time.time()
            if (current_time - last_analysis_time >= ANALYSIS_WINDOW_SECONDS and 
                eeg_buffer.shape[1] >= nfft and calibration_complete):
                
                segment = eeg_buffer[:, -nfft:]
                current_metrics = calculate_band_powers(segment)
                
                if current_metrics:
                    signal_quality_validator.add_band_power_data(current_metrics)
                    signal_quality_validator.add_raw_eeg_data(segment)
                    
                    quality = signal_quality_validator.assess_overall_quality()
                    classification = classify_mental_state(current_metrics)
                    
                    time_rel_feedback = current_time - session_start_time if session_start_time else 0
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
                        "probabilities": classification["probabilities"],
                        "signal_quality": quality.quality_level,
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
                    
                # Reset buffer with overlap
                eeg_buffer = eeg_buffer[:, -int(nfft * 0.5):]
                last_analysis_time = current_time
                
        except Exception as e:
            print(f"Data processing error: {e}")
            time.sleep(0.1)


def perform_calibration_phase():
    """Calibration phase with signal quality monitoring"""
    global baseline_metrics, signal_quality_validator, calibration_complete
    
    print(f"\n--- Starting {CALIBRATION_DURATION_SECONDS:.0f} Second Calibration ---")
    print("Please remain in a neutral, resting state.")
    
    calibration_start_time = time.time()
    calibration_metrics_list = []
    
    while (time.time() - calibration_start_time < CALIBRATION_DURATION_SECONDS and running):
        chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
        
        if not chunk:
            time.sleep(0.01)
            continue
            
        chunk_np = np.array(chunk, dtype=np.float64).T
        
        if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
            print(f"Not enough channels in chunk: {chunk_np.shape[0]} <= {max(EEG_CHANNEL_INDICES)}")
            continue
            
        eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
        
        # Extract accelerometer data if available
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
        
        # If we have enough data for PSD calculation
        if eeg_chunk_filtered.shape[1] >= nfft:
            segment = eeg_chunk_filtered[:, -nfft:]
            metrics = calculate_band_powers(segment)
            
            if metrics:
                signal_quality_validator.add_band_power_data(metrics)
                signal_quality_validator.add_raw_eeg_data(segment)
                calibration_metrics_list.append(metrics)
                
                # Check signal quality periodically
                if len(calibration_metrics_list) % 5 == 0:
                    quality = signal_quality_validator.assess_overall_quality()
                    progress = ((time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS) * 100
                    print(f"Calibration progress: {progress:.0f}% - Quality: {quality.quality_level} ({quality.overall_score:.2f})")
                    
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
                    
                    if quality.overall_score < 0.3:
                        print(f"WARNING: Poor signal quality - {quality.recommendations[0] if quality.recommendations else 'Please adjust headband'}")
    
    # Calculate baseline
    if calibration_metrics_list:
        print(f"Creating baseline from {len(calibration_metrics_list)} metrics")
        
        baseline_metrics = {
            'alpha': np.mean([m['alpha'] for m in calibration_metrics_list]),
            'beta': np.mean([m['beta'] for m in calibration_metrics_list]),
            'theta': np.mean([m['theta'] for m in calibration_metrics_list])
        }
        baseline_metrics['ab_ratio'] = baseline_metrics['alpha'] / baseline_metrics['beta'] if baseline_metrics['beta'] > 1e-9 else 0
        baseline_metrics['bt_ratio'] = baseline_metrics['beta'] / baseline_metrics['theta'] if baseline_metrics['theta'] > 1e-9 else 0
        
        print("\n--- Calibration Complete ---")
        for key, val in baseline_metrics.items(): 
            print(f"Baseline {key.replace('_', ' ').title()}: {val:.2f}")
            
        calibration_complete = True
        return True
    else:
        print("Calibration failed: No metrics collected. Check LSL stream.")
        calibration_complete = False
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


def update_plots():
    """Update all plots with current data"""
    try:
        if not plt.get_fignums():  # Check if window was closed
            return
            
        for ax in axs:
            ax.clear()
        
        # Plot 1: Raw EEG with user state background
        ax = axs[0]
        if full_session_eeg_data.shape[1] > 0:
            window_samples = min(full_session_eeg_data.shape[1], int(sampling_rate * 10))
            recent_data = full_session_eeg_data[:, -window_samples:]
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
        plt.draw()
        plt.pause(0.01)
        
    except Exception as e:
        print(f"Plot update error: {e}")


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


def generate_accuracy_report():
    """Generate classification accuracy report"""
    if not feedback_log:
        print("No feedback data available for analysis.")
        return
    
    print("\n=== Classification Accuracy Analysis ===")
    
    # Create simplified state mappings for comparison
    def simplify_predicted_state(state_key, level):
        if level <= -2:
            return "Negative"  # Tense/Distracted
        elif level == -1:
            return "Slightly Negative"
        elif level == 0:
            return "Neutral"
        elif level == 1:
            return "Slightly Positive"
        else:  # level >= 2
            return "Positive"  # Relaxed/Focused
    
    def simplify_user_state(user_state):
        mapping = {
            'Very Relaxed': 'Positive',
            'Relaxed': 'Slightly Positive',
            'Neutral': 'Neutral',
            'Focused': 'Slightly Positive',
            'Very Focused': 'Positive',
            'Unknown': 'Unknown',
            'Uncertain': 'Unknown'
        }
        return mapping.get(user_state, 'Unknown')
    
    # Extract predictions and user states
    predicted_states = []
    user_states = []
    
    for entry in feedback_log:
        if entry['user_reported_state'] != 'Unknown':
            predicted = simplify_predicted_state(entry['state_key'], entry['level'])
            user = simplify_user_state(entry['user_reported_state'])
            
            if user != 'Unknown':
                predicted_states.append(predicted)
                user_states.append(user)
    
    if len(predicted_states) < 10:
        print("Insufficient labeled data for meaningful analysis.")
        return
    
    # Calculate accuracy metrics
    from collections import Counter
    
    correct_predictions = sum(1 for p, u in zip(predicted_states, user_states) if p == u)
    total_predictions = len(predicted_states)
    accuracy = correct_predictions / total_predictions
    
    print(f"Total labeled predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.2%}")
    
    # State distribution
    user_counter = Counter(user_states)
    pred_counter = Counter(predicted_states)
    
    print(f"\nUser state distribution: {dict(user_counter)}")
    print(f"Predicted state distribution: {dict(pred_counter)}")
    
    # Create confusion matrix if we have sklearn
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Get unique labels
        labels = sorted(list(set(user_states + predicted_states)))
        
        cm = confusion_matrix(user_states, predicted_states, labels=labels)
        
        print(f"\nConfusion Matrix:")
        print(f"{'':>15} " + " ".join(f"{label:>10}" for label in labels))
        for i, true_label in enumerate(labels):
            print(f"{true_label:>15} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(user_states, predicted_states, labels=labels))
        
    except ImportError:
        print("sklearn not available for detailed metrics.")


def save_enhanced_session_data():
    """Save enhanced session data with user states"""
    global feedback_log, user_labels, user_state_changes, quality_log, baseline_metrics
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save raw and processed EEG data
    np.save(os.path.join(SAVE_PATH, f"eeg_data_raw_{timestamp_str}.npy"), full_session_eeg_data_raw)
    np.save(os.path.join(SAVE_PATH, f"eeg_data_processed_{timestamp_str}.npy"), full_session_eeg_data)
    np.save(os.path.join(SAVE_PATH, f"timestamps_{timestamp_str}.npy"), np.array(full_session_timestamps_lsl))
    
    # Save baseline metrics
    if baseline_metrics:
        with open(os.path.join(SAVE_PATH, f"baseline_metrics_{timestamp_str}.pkl"), 'wb') as f:
            pickle.dump(baseline_metrics, f)
    
    # Save enhanced feedback log
    if feedback_log:
        feedback_df = pd.DataFrame()
        
        # Basic fields
        feedback_df['time_abs'] = [entry['time_abs'] for entry in feedback_log]
        feedback_df['time_rel'] = [entry['time_rel'] for entry in feedback_log]
        feedback_df['state'] = [entry['prediction'] for entry in feedback_log]
        feedback_df['state_key'] = [entry['state_key'] for entry in feedback_log]
        feedback_df['level'] = [entry['level'] for entry in feedback_log]
        feedback_df['confidence'] = [entry['confidence'] for entry in feedback_log]
        feedback_df['smooth_value'] = [entry['smooth_value'] for entry in feedback_log]
        feedback_df['signal_quality'] = [entry['signal_quality'] for entry in feedback_log]
        feedback_df['user_reported_state'] = [entry['user_reported_state'] for entry in feedback_log]
        
        # Metrics
        for metric in ['alpha', 'beta', 'theta', 'ab_ratio', 'bt_ratio']:
            feedback_df[metric] = [entry['metrics'][metric] for entry in feedback_log]
        
        # Probabilities
        if 'probabilities' in feedback_log[0]:
            for prob in ['relaxed', 'focused', 'drowsy', 'internal_focus', 'neutral']:
                feedback_df[f'p_{prob}'] = [entry['probabilities'][prob] for entry in feedback_log]
        
        feedback_df.to_csv(os.path.join(SAVE_PATH, f"enhanced_feedback_log_{timestamp_str}.csv"), index=False)
    
    # Save user state changes
    if user_state_changes:
        state_changes_df = pd.DataFrame(user_state_changes)
        state_changes_df.to_csv(os.path.join(SAVE_PATH, f"user_state_changes_{timestamp_str}.csv"), index=False)
    
    # Save event labels
    if user_labels:
        user_labels_df = pd.DataFrame(user_labels)
        user_labels_df.to_csv(os.path.join(SAVE_PATH, f"event_labels_{timestamp_str}.csv"), index=False)
    
    # Save quality log
    if quality_log:
        quality_df = pd.DataFrame()
        quality_df['time'] = [entry['time'] for entry in quality_log]
        for metric in ['movement_score', 'band_power_score', 'electrode_contact_score', 'overall_score']:
            quality_df[metric] = [entry['metrics'][metric] for entry in quality_log]
        quality_df['quality_level'] = [entry['metrics']['quality_level'] for entry in quality_log]
        quality_df['recommendation'] = [
            entry['metrics']['recommendations'][0] if entry['metrics']['recommendations'] else ""
            for entry in quality_log
        ]
        quality_df.to_csv(os.path.join(SAVE_PATH, f"quality_log_{timestamp_str}.csv"), index=False)
    
    # Save enhanced metadata
    metadata = {
        'session_type': session_type,
        'timestamp': timestamp_str,
        'sampling_rate': sampling_rate,
        'calibration_duration': CALIBRATION_DURATION_SECONDS,
        'analysis_window': ANALYSIS_WINDOW_SECONDS,
        'psd_window': PSD_WINDOW_SECONDS,
        'total_feedback_entries': len(feedback_log),
        'total_state_changes': len(user_state_changes),
        'total_event_labels': len(user_labels),
        'state_labels_used': STATE_LABELS,
        'event_labels_used': EVENT_LABELS
    }
    
    import json
    with open(os.path.join(SAVE_PATH, f"enhanced_metadata_{timestamp_str}.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nEnhanced session data saved to: {SAVE_PATH}")
    
    # Generate accuracy report
    generate_accuracy_report()


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
    global running, session_start_time, data_processing_thread, fig, axs
    
    # Select session type
    select_session_type()
    
    # Connect to LSL
    if not connect_to_lsl():
        print("Failed to connect to LSL. Exiting.")
        return
    
    # Initialize matplotlib in main thread
    print("Setting up visualization...")
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.canvas.manager.set_window_title('EEG Testing Interface - Enhanced')
    plt.show(block=False)
    
    # Perform calibration
    if not perform_calibration_phase():
        print("Calibration failed. Exiting.")
        return
    
    # Set initial user state and session start time
    update_user_state("Unknown")
    session_start_time = time.time()
    
    print(f"\n--- Starting Real-time Feedback ---")
    
    # Start data processing thread
    data_processing_thread = threading.Thread(target=data_processing_thread_function)
    data_processing_thread.daemon = True
    data_processing_thread.start()
    
    # Start keyboard monitoring thread
    keyboard_thread = threading.Thread(target=monitor_keyboard_input)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    # Main loop - handle plotting and user input
    last_plot_update = 0
    try:
        while running:
            current_time = time.time()
            
            # Update plots every 2 seconds
            if current_time - last_plot_update >= 2.0:
                update_plots()
                last_plot_update = current_time
            
            # Process matplotlib events to keep window responsive
            fig.canvas.flush_events()
            
            # Check if plot window was closed
            if not plt.get_fignums():
                print("Plot window closed. Ending session.")
                running = False
                break
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        running = False
    finally:
        # Save data and generate reports
        save_enhanced_session_data()
        
        if lsl_inlet:
            lsl_inlet.close_stream()
        
        print("\nSession complete. Check the generated plots and CSV files for analysis.")
        
        # Keep plot window open for a bit
        print("Plots will remain open for 10 seconds...")
        for i in range(100):  # 10 seconds
            if plt.get_fignums():
                fig.canvas.flush_events()
                time.sleep(0.1)
            else:
                break


if __name__ == "__main__":
    main()