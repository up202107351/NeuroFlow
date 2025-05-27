#!/usr/bin/env python3
"""
EEG Backend Processor - ZMQ Publisher for EEG-based Feedback

This script handles:
1. LSL connection to EEG device
2. Baseline calibration
3. Real-time EEG processing and state classification
4. ZMQ publishing of mental state predictions
"""

import time
import json
import numpy as np
import zmq
import pylsl
import signal
import os
import logging
from datetime import datetime
from scipy.signal import butter, filtfilt, welch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eeg_processor.log")
    ]
)
logger = logging.getLogger('EEG_Processor')

try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logger.error("BrainFlow library not found. Please install it (pip install brainflow).")
    exit(1)

# --- Configuration ---
LSL_STREAM_TYPE = 'EEG'
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX_PULL = 128

EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10 for Muse
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0  # More frequent updates for smooth feedback
PSD_WINDOW_SECONDS = 6.0       # Data length for each individual PSD calculation

DEFAULT_SAMPLING_RATE = 256.0

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# ZMQ Settings
ZMQ_PUBLISHER_ADDRESS = "tcp://*:5556"  # Address for app to connect to
ZMQ_HEARTBEAT_INTERVAL = 1.0  # Send heartbeat every second
ZMQ_PUBLISHER_SYNC_PORT = 5557  # Port for REP socket to confirm subscribers are ready

PREDICTION_PUBLISH_INTERVAL = 1.0   # Base interval between published predictions (seconds)
SIGNIFICANT_CHANGE_THRESHOLD = 0.15  # Only publish if state value changes by at least this amount

# --- Global State ---
running = True
lsl_inlet = None
sampling_rate = DEFAULT_SAMPLING_RATE
nfft = DataFilter.get_nearest_power_of_two(int(DEFAULT_SAMPLING_RATE * PSD_WINDOW_SECONDS))
welch_overlap_samples = nfft // 2

# Design Butterworth bandpass filter (0.5 - 30 Hz)
filter_order = 4
lowcut = 0.5
highcut = 30.0
nyq = 0.5 * sampling_rate
low = lowcut / nyq
high = highcut / nyq
b, a = butter(filter_order, [low, high], btype='band', analog=False)

# Classification state
baseline_metrics = None
is_calibrated = False
is_calibrating = False

# For temporal smoothing of classifications
previous_states = []
MAX_STATE_HISTORY = 5

# For adaptive thresholds
recent_metrics_history = []
MAX_HISTORY_SIZE = 15

# Channel quality weights
channel_weights = np.ones(NUM_EEG_CHANNELS) / NUM_EEG_CHANNELS

# Session state
current_session_type = None  # "RELAXATION" or "FOCUS"
session_start_time = None

# --- Signal Handler for Graceful Shutdown ---
def graceful_signal_handler(sig, frame):
    global running
    if running:
        logger.info(f"Signal {sig} received. Shutting down...")
        running = False

signal.signal(signal.SIGINT, graceful_signal_handler)
signal.signal(signal.SIGTERM, graceful_signal_handler)

# --- ZMQ Setup ---
class ZMQPublisher:
    def __init__(self, publisher_address=ZMQ_PUBLISHER_ADDRESS, sync_port=ZMQ_PUBLISHER_SYNC_PORT):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(publisher_address)
        logger.info(f"ZMQ Publisher bound to {publisher_address}")
        
        # Setup synchronization socket
        self.sync_socket = self.context.socket(zmq.REP)
        self.sync_socket.bind(f"tcp://*:{sync_port}")
        logger.info(f"ZMQ Sync socket bound to port {sync_port}")
        
        # Last message timestamp for heartbeat
        self.last_heartbeat_time = 0
        
    def wait_for_subscribers(self, timeout=5.0):
        """Wait for at least one subscriber to connect"""
        logger.info("Waiting for subscribers...")
        
        # Set timeout for sync socket receive
        self.sync_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        
        try:
            # Wait for sync request
            message = self.sync_socket.recv_string()
            logger.info(f"Received sync request: {message}")
            self.sync_socket.send_string("READY")
            logger.info("Subscriber synchronized!")
            return True
        except zmq.error.Again:
            logger.warning(f"No subscribers connected within {timeout} seconds")
            return False
            
    def publish_message(self, message_dict):
        """Publish a message to subscribers"""
        try:
            self.publisher.send_json(message_dict)
            return True
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
            
    def send_heartbeat(self):
        """Send a heartbeat message to subscribers"""
        current_time = time.time()
        if current_time - self.last_heartbeat_time >= ZMQ_HEARTBEAT_INTERVAL:
            heartbeat_msg = {
                "message_type": "HEARTBEAT",
                "timestamp": current_time
            }
            self.publish_message(heartbeat_msg)
            self.last_heartbeat_time = current_time
            
    def cleanup(self):
        """Close sockets and terminate context"""
        if self.publisher:
            self.publisher.close()
        if self.sync_socket:
            self.sync_socket.close()
        if self.context:
            self.context.term()
        logger.info("ZMQ publisher cleaned up")

# --- EEG Processing ---
class EEGProcessor:
    def __init__(self, zmq_publisher, command_socket):
        self.zmq_publisher = zmq_publisher
        self.command_socket = command_socket  # Add a reference to command socket
        self.lsl_inlet = None
        self.sampling_rate = DEFAULT_SAMPLING_RATE
        self.running = True
        self.is_calibrated = False
        self.is_calibrating = False
        self.current_session_type = None
        self.session_start_time = None

        self.last_command_check = 0
        self.command_check_interval = 0.2  # Check commands every 200ms
        
        # EEG processing state
        self.baseline_metrics = None
        self.recent_metrics_history = []
        self.previous_states = []
        self.channel_weights = np.ones(NUM_EEG_CHANNELS) / NUM_EEG_CHANNELS
        
        # For buffer management
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        self.buffer_timestamps = []

        self.state_momentum = 0.75  # Higher = more resistance to state changes
        self.state_velocity = 0.0   # Current rate of change
        self.level_momentum = 0.8   # Higher = smoother level transitions
        self.level_velocity = 0.0   # Current rate of level change
        
    def connect_to_lsl(self):
        """Connect to the LSL stream"""
        logger.info(f"Looking for LSL stream (Type: '{LSL_STREAM_TYPE}')...")
        try:
            streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
            if not streams:
                logger.error("LSL stream not found.")
                return False
                
            self.lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
            info = self.lsl_inlet.info()
            lsl_sr = info.nominal_srate()
            self.sampling_rate = lsl_sr if lsl_sr > 0 else DEFAULT_SAMPLING_RATE
            self.nfft = DataFilter.get_nearest_power_of_two(int(self.sampling_rate * PSD_WINDOW_SECONDS))
            self.welch_overlap_samples = self.nfft // 2
            
            # Update filter coefficients based on actual sampling rate
            self.nyq = 0.5 * self.sampling_rate
            self.low = lowcut / self.nyq
            self.high = highcut / self.nyq
            self.b, self.a = butter(filter_order, [self.low, self.high], btype='band', analog=False)
            
            logger.info(f"Connected to '{info.name()}' @ {self.sampling_rate:.2f} Hz. NFFT={self.nfft}")
            
            # Publish connection status
            self.zmq_publisher.publish_message({
                "message_type": "CONNECTION_STATUS",
                "status": "CONNECTED",
                "device_name": info.name(),
                "sampling_rate": self.sampling_rate,
                "timestamp": time.time()
            })
            
            if info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
                logger.error(f"LSL stream has insufficient channels.")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to LSL: {e}")
            
            # Publish connection error
            self.zmq_publisher.publish_message({
                "message_type": "CONNECTION_STATUS",
                "status": "ERROR",
                "error_message": str(e),
                "timestamp": time.time()
            })
            
            return False
            
    def improved_artifact_rejection(self, eeg_data):
        """More sophisticated artifact rejection"""
        # Statistical thresholds based on running variance
        window_size = min(64, eeg_data.shape[1])  # ~250ms at 256Hz or smaller if data is smaller
        channel_thresholds = [150, 100, 100, 150]  # Base thresholds
        
        # Check amplitude thresholds (existing approach)
        amplitude_mask = ~np.any(np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1), axis=0)
        
        # Check for rapid changes (derivatives/gradients) if we have enough data
        diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
        if eeg_data.shape[1] > 1:
            diff_thresholds = [50, 30, 30, 50]  # μV per sample
            diff_mask = ~np.any(
                np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
                np.array(diff_thresholds).reshape(-1, 1), 
                axis=0
            )
        
        # Variance-based detection (for muscle artifacts) if we have enough data
        var_mask = np.ones(eeg_data.shape[1], dtype=bool)
        if eeg_data.shape[1] >= window_size:
            for ch in range(eeg_data.shape[0]):
                rolling_var = np.array([
                    np.var(eeg_data[ch, max(0, i-window_size):i+1]) 
                    for i in range(eeg_data.shape[1])
                ])
                # Mark samples where variance exceeds 3 standard deviations
                var_threshold = np.mean(rolling_var) + 3 * np.std(rolling_var)
                var_mask = var_mask & (rolling_var < var_threshold)
        
        # Combine all masks
        good_samples = amplitude_mask & diff_mask & var_mask
        
        return good_samples
        
    def update_channel_weights(self, eeg_data):
        """Update channel weights based on signal quality"""
        # Calculate signal quality per channel
        signal_quality = np.zeros(NUM_EEG_CHANNELS)
        
        for ch in range(NUM_EEG_CHANNELS):
            # Simple signal-to-noise ratio estimate
            if eeg_data.shape[1] >= self.nfft:
                frequencies, psd = welch(eeg_data[ch], fs=self.sampling_rate, nperseg=self.nfft, noverlap=self.welch_overlap_samples)
                
                # Find indices for bands of interest
                eeg_band_mask = (frequencies >= 1) & (frequencies <= 30)
                noise_band_mask = (frequencies > 40) & (frequencies < min(100, self.sampling_rate/2 - 1))
                
                # Calculate power ratio
                eeg_power = np.mean(psd[eeg_band_mask]) if np.any(eeg_band_mask) else 0
                noise_power = np.mean(psd[noise_band_mask]) if np.any(noise_band_mask) else 0
                
                # Higher ratio = better signal quality
                if noise_power > 0 and eeg_power > 0:
                    snr = eeg_power / noise_power
                    signal_quality[ch] = min(5, snr)  # Cap at 5
                else:
                    signal_quality[ch] = 1.0  # Default
            else:
                signal_quality[ch] = 1.0  # Default for short segments
        
        # Normalize to create weights
        signal_quality = np.clip(signal_quality, 0.5, 5)  # Ensure minimum weight
        self.channel_weights = signal_quality / np.sum(signal_quality)
        
        return self.channel_weights
        
    def filter_eeg_data(self, eeg_data):
        """Apply bandpass filter to EEG data"""
        # Minimum samples needed for filtfilt
        min_samples = 3 * filter_order + 1
        
        if eeg_data.shape[1] < min_samples:
            logger.warning(f"Not enough data for filtfilt: {eeg_data.shape[1]} < {min_samples}")
            return eeg_data  # Return unfiltered if not enough data
            
        eeg_filtered = np.zeros_like(eeg_data)
        for i in range(NUM_EEG_CHANNELS):
            eeg_filtered[i] = filtfilt(self.b, self.a, eeg_data[i])
        return eeg_filtered
        
    def calculate_band_powers(self, eeg_segment):
        """Calculate band powers for EEG segment with artifact rejection"""
        if eeg_segment.shape[1] < self.nfft:
            logger.warning(f"Segment too short for PSD: {eeg_segment.shape[1]} < {self.nfft}")
            return None
            
        # Apply artifact rejection
        artifact_mask = self.improved_artifact_rejection(eeg_segment)
        if np.sum(artifact_mask) < 0.7 * eeg_segment.shape[1]:
            logger.debug(f"Excessive artifacts: {np.sum(artifact_mask)}/{eeg_segment.shape[1]} good samples")
            return None
            
        # Update channel weights based on signal quality
        self.update_channel_weights(eeg_segment)
        
        # Calculate band powers for each channel
        metrics_list = []
        for ch_idx in range(NUM_EEG_CHANNELS):
            ch_data = eeg_segment[ch_idx, artifact_mask].copy() if np.any(artifact_mask) else eeg_segment[ch_idx].copy()
            
            # Ensure enough data after artifact rejection
            if len(ch_data) < self.nfft:
                # Pad data if needed
                pad_length = self.nfft - len(ch_data)
                ch_data = np.pad(ch_data, (0, pad_length), mode='reflect')
                
            DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
            
            try:
                psd = DataFilter.get_psd_welch(
                    ch_data, 
                    self.nfft, 
                    self.welch_overlap_samples, 
                    int(self.sampling_rate), 
                    WindowOperations.HANNING.value
                )
                
                metrics_list.append({
                    'theta': DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1]),
                    'alpha': DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1]),
                    'beta': DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])
                })
            except Exception as e:
                logger.error(f"PSD calculation failed: {e}")
                return None
                
        if len(metrics_list) != NUM_EEG_CHANNELS:
            return None
            
        # Calculate weighted average of band powers
        avg_metrics = {
            'theta': np.sum([m['theta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
            'alpha': np.sum([m['alpha'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
            'beta': np.sum([m['beta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)])
        }
        
        # Calculate ratios
        avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
        avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
        
        return avg_metrics
        
    def calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds based on recent history"""
        if len(self.recent_metrics_history) < 5:
            # Not enough history, use default thresholds
            return {
                'alpha_incr': [1.10, 1.25, 1.50],  # slight, moderate, strong
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
        
        # Calculate recent variability
        alpha_values = [m['alpha'] for m in self.recent_metrics_history]
        beta_values = [m['beta'] for m in self.recent_metrics_history]
        theta_values = [m['theta'] for m in self.recent_metrics_history]
        
        alpha_std = np.std(alpha_values) / self.baseline_metrics['alpha'] if self.baseline_metrics else 0.1
        beta_std = np.std(beta_values) / self.baseline_metrics['beta'] if self.baseline_metrics else 0.1
        theta_std = np.std(theta_values) / self.baseline_metrics['theta'] if self.baseline_metrics else 0.1
        
        # Adjust thresholds based on observed variability (more variable = wider thresholds)
        alpha_factor = min(1.0, max(0.5, 1.0 - alpha_std))
        beta_factor = min(1.0, max(0.5, 1.0 - beta_std))
        theta_factor = min(1.0, max(0.5, 1.0 - theta_std))
        
        return {
            'alpha_incr': [1.0 + 0.10/alpha_factor, 1.0 + 0.25/alpha_factor, 1.0 + 0.50/alpha_factor],
            'alpha_decr': [1.0 - 0.10/alpha_factor, 1.0 - 0.20/alpha_factor, 1.0 - 0.35/alpha_factor],
            'beta_incr': [1.0 + 0.10/beta_factor, 1.0 + 0.20/beta_factor, 1.0 + 0.40/beta_factor],
            'beta_decr': [1.0 - 0.10/beta_factor, 1.0 - 0.20/beta_factor],
            'theta_incr': [1.0 + 0.15/theta_factor, 1.0 + 0.30/theta_factor],
            'theta_decr': [1.0 - 0.15/theta_factor, 1.0 - 0.30/theta_factor],
            'ab_ratio_incr': [1.10, 1.20, 1.40],
            'ab_ratio_decr': [0.90, 0.75],
            'bt_ratio_incr': [1.15, 1.30, 1.60],
            'bt_ratio_decr': [0.85, 0.70]
        }
        
    def calculate_state_probabilities(self, current_metrics):
        """Calculate probabilities for different mental states"""
        if not self.baseline_metrics:
            return {
                'relaxed': 0.5,
                'focused': 0.5,
                'drowsy': 0.2,
                'internal_focus': 0.3,
                'neutral': 0.7
            }
            
        probs = {}
        
        # Define sigmoid function to map differences to probabilities
        def sigmoid(x, k=5):
            return 1 / (1 + np.exp(-k * x))
        
        # Calculate normalized differences from baseline
        alpha_diff = current_metrics['alpha'] / self.baseline_metrics['alpha'] - 1.0
        beta_diff = current_metrics['beta'] / self.baseline_metrics['beta'] - 1.0
        theta_diff = current_metrics['theta'] / self.baseline_metrics['theta'] - 1.0
        ab_ratio_diff = current_metrics['ab_ratio'] / self.baseline_metrics['ab_ratio'] - 1.0
        bt_ratio_diff = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] - 1.0
        
        # Calculate "relaxation probability" based on alpha and alpha/beta ratio
        # More weight to ratio for relaxation
        relaxation_signal = 0.4 * alpha_diff + 0.6 * ab_ratio_diff
        probs['relaxed'] = sigmoid(relaxation_signal, k=3)
        
        # Calculate "focus probability" based on beta and beta/theta ratio
        # Focus has inverse relationship with alpha
        alpha_inverse_diff = 1.0 - (current_metrics['alpha'] / self.baseline_metrics['alpha'])
        focus_signal = 0.3 * beta_diff + 0.5 * bt_ratio_diff + 0.2 * max(0, alpha_inverse_diff)
        probs['focused'] = sigmoid(focus_signal, k=3)
        
        # Drowsiness - high theta, low beta
        drowsy_signal = 0.6 * theta_diff - 0.4 * beta_diff
        probs['drowsy'] = sigmoid(drowsy_signal, k=3)
        
        # Calculate "internal focus" probability
        internal_focus_signal = 0.5 * beta_diff + 0.5 * alpha_diff
        probs['internal_focus'] = sigmoid(internal_focus_signal, k=3)
        
        # Neutral state - everything close to baseline
        baseline_closeness = -2.0 * (abs(alpha_diff) + abs(beta_diff) + abs(theta_diff))
        probs['neutral'] = sigmoid(baseline_closeness, k=5)
        
        return probs
        
    def detect_eyes_closed(self, current_metrics):
        """Detect eyes closed state based on alpha characteristics"""
        if not self.baseline_metrics:
            return False
            
        is_likely_eyes_closed = (
            current_metrics['alpha'] > self.baseline_metrics['alpha'] * 2.0 and  # Alpha doubled
            current_metrics['ab_ratio'] > self.baseline_metrics['ab_ratio'] * 1.5 and  # A/B ratio increased 50%+
            current_metrics['theta'] < current_metrics['theta'] * 1.2  # Theta didn't increase significantly
        )
        
        # Additional check for pattern consistency across frontal vs temporal electrodes
        frontal_channels = [1, 2]  # AF7, AF8
        temporal_channels = [0, 3]  # TP9, TP10
        
        # This would require per-channel alpha power in buffer, simplified check for now
        return is_likely_eyes_closed
        
    def adapt_baseline(self, current_metrics, adaptation_rate=0.02):
        """Slowly adapt baseline to gradual shifts"""
        if not self.baseline_metrics:
            return
            
        # Detect if we're in a neutral state (close to baseline)
        is_neutral = (
            0.9 < current_metrics['alpha'] / self.baseline_metrics['alpha'] < 1.1 and
            0.9 < current_metrics['beta'] / self.baseline_metrics['beta'] < 1.1 and
            0.9 < current_metrics['theta'] / self.baseline_metrics['theta'] < 1.1
        )
        
        # Only adapt if in neutral state and enough time has passed since calibration
        if is_neutral and self.session_start_time and (time.time() - self.session_start_time) > 90:
            # Slowly adapt baseline (weighted average)
            for key in ['alpha', 'beta', 'theta']:
                self.baseline_metrics[key] = (1 - adaptation_rate) * self.baseline_metrics[key] + adaptation_rate * current_metrics[key]
            
            # Recalculate ratios
            self.baseline_metrics['ab_ratio'] = self.baseline_metrics['alpha'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 1e-9 else 0
            self.baseline_metrics['bt_ratio'] = self.baseline_metrics['beta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 1e-9 else 0
            
    def classify_mental_state(self, current_metrics):
        """Classify mental state based on current metrics and baseline with more granular levels"""
        if not self.baseline_metrics or not current_metrics:
            return {
                "state": "Calibrating",
                "level": 0,
                "confidence": "N/A",
                "value": 0.5,
                "smooth_value": 0.5
            }
        
        # Get state probabilities
        state_probs = self.calculate_state_probabilities(current_metrics)
        
        # Initial determination of most probable state
        most_probable_state = max(state_probs.items(), key=lambda x: x[1])
        state_name, prob_value = most_probable_state
        
        # Special case for eyes closed
        if self.detect_eyes_closed(current_metrics):
            state_name = "eyes_closed"
            prob_value = 0.9
        
        # Default level is 0 (neutral)
        level = 0
        
        # Adjust classification for session goal
        if self.current_session_type == "RELAXATION":
            # Calculate normalized relaxation value (0.0 to 1.0)
            relaxation_value = min(1.0, max(0.0, state_probs['relaxed']))
            
            # Determine relaxation level (from -3 to 4)
            # -3 to -1: Less relaxed (alert/tense)
            # 0: Neutral
            # 1 to 4: Increasingly relaxed
            
            # Measure degree of being less relaxed (alert/tense)
            ab_ratio_decrease = self.baseline_metrics['ab_ratio'] / current_metrics['ab_ratio'] - 1.0 if current_metrics['ab_ratio'] > 0 else 0
            beta_increase = current_metrics['beta'] / self.baseline_metrics['beta'] - 1.0
            alpha_decrease = 1.0 - current_metrics['alpha'] / self.baseline_metrics['alpha']
            
            alert_signal = (0.4 * ab_ratio_decrease + 0.3 * beta_increase + 0.3 * alpha_decrease) * 4.0
            
            # Less relaxed levels
            if ab_ratio_decrease > 0.15 and (beta_increase > 0.1 or alpha_decrease > 0.1):
                if alert_signal > 1.5:
                    level = -3  # Significantly less relaxed / tense
                    state_name = "tense"
                elif alert_signal > 1.0:
                    level = -2  # Moderately less relaxed / alert
                    state_name = "alert"
                else:
                    level = -1  # Slightly less relaxed
                    state_name = "less_relaxed"
            
            # More relaxed levels
            elif relaxation_value > 0.3:
                # Determine relaxation level based on alpha and ab_ratio increases
                alpha_increase = current_metrics['alpha'] / self.baseline_metrics['alpha'] - 1.0
                ab_ratio_increase = current_metrics['ab_ratio'] / self.baseline_metrics['ab_ratio'] - 1.0 if self.baseline_metrics['ab_ratio'] > 0 else 0
                
                relax_signal = (0.5 * alpha_increase + 0.5 * ab_ratio_increase) * 5.0
                
                if relax_signal > 2.0:
                    level = 4  # Deeply relaxed
                    state_name = "deeply_relaxed"
                elif relax_signal > 1.3:
                    level = 3  # Strongly relaxed
                    state_name = "strongly_relaxed"
                elif relax_signal > 0.7:
                    level = 2  # Moderately relaxed
                    state_name = "moderately_relaxed"
                else:
                    level = 1  # Slightly relaxed
                    state_name = "slightly_relaxed"
            else:
                level = 0  # Neutral
                state_name = "neutral"
            
            # Map to a smooth value from 0-1 for visualization
            # -3 = 0.0, 0 = 0.5, 4 = 1.0
            value = (level + 3) / 7.0
            
        elif self.current_session_type == "FOCUS":
            # Calculate normalized focus value (0.0 to 1.0)
            focus_value = min(1.0, max(0.0, state_probs['focused']))
            
            # Determine focus level (from -3 to 4)
            # -3 to -1: Less focused (distracted)
            # 0: Neutral
            # 1 to 4: Increasingly focused
            
            # Measure degree of being less focused (distracted)
            bt_ratio_decrease = self.baseline_metrics['bt_ratio'] / current_metrics['bt_ratio'] - 1.0 if current_metrics['bt_ratio'] > 0 else 0
            beta_decrease = 1.0 - current_metrics['beta'] / self.baseline_metrics['beta']
            theta_increase = current_metrics['theta'] / self.baseline_metrics['theta'] - 1.0
            
            distracted_signal = (0.4 * bt_ratio_decrease + 0.3 * beta_decrease + 0.3 * theta_increase) * 4.0
            
            # Less focused levels
            if bt_ratio_decrease > 0.15 and (beta_decrease > 0.1 or theta_increase > 0.1):
                if distracted_signal > 1.5:
                    level = -3  # Significantly distracted
                    state_name = "very_distracted"
                elif distracted_signal > 1.0:
                    level = -2  # Moderately distracted
                    state_name = "distracted"
                else:
                    level = -1  # Slightly distracted
                    state_name = "less_focused"
            
            # More focused levels
            elif focus_value > 0.3:
                # Determine focus level based on beta and bt_ratio increases
                beta_increase = current_metrics['beta'] / self.baseline_metrics['beta'] - 1.0
                bt_ratio_increase = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] - 1.0 if self.baseline_metrics['bt_ratio'] > 0 else 0
                
                focus_signal = (0.5 * beta_increase + 0.5 * bt_ratio_increase) * 5.0
                
                if focus_signal > 2.0:
                    level = 4  # Deeply focused
                    state_name = "deeply_focused"
                elif focus_signal > 1.3:
                    level = 3  # Strongly focused
                    state_name = "strongly_focused"
                elif focus_signal > 0.7:
                    level = 2  # Moderately focused
                    state_name = "moderately_focused"
                else:
                    level = 1  # Slightly focused
                    state_name = "slightly_focused"
            else:
                level = 0  # Neutral
                state_name = "neutral"
            
            # Map to a smooth value from 0-1 for visualization
            # -3 = 0.0, 0 = 0.5, 4 = 1.0
            value = (level + 3) / 7.0
        else:
            # Default mapping
            value = prob_value
        
        # Determine confidence based on probability
        if prob_value > 0.8:
            confidence = "high"
        elif prob_value > 0.65:
            confidence = "medium"
        elif prob_value > 0.55:
            confidence = "low"
        else:
            confidence = "very_low"
        
        # Apply temporal smoothing for the normalized value
        smooth_value = value
        
        # Store state and value for history
        self.previous_states.append((state_name, value, level))
        if len(self.previous_states) > MAX_STATE_HISTORY:
            self.previous_states.pop(0)
        
        # Smooth classification with moving average if we have history
        if len(self.previous_states) > 1:
            # Smooth the value with momentum-based approach
            current_value = value
            recent_values = [s[1] for s in self.previous_states]
            prev_value = recent_values[-2]  # Previous value
            
            # Calculate target velocity (difference from previous value)
            target_velocity = current_value - prev_value
            
            # Apply momentum to velocity (gradual change in value)
            self.state_velocity = (self.state_velocity * self.state_momentum + 
                                target_velocity * (1 - self.state_momentum))
            
            # Apply smoothed velocity to get new value
            smooth_value = prev_value + self.state_velocity
            
            # Ensure smooth_value stays in valid range
            smooth_value = min(1.0, max(0.0, smooth_value))
            
            # Smooth the level with momentum-based approach
            target_level = level
            recent_levels = [s[2] for s in self.previous_states]
            prev_level = recent_levels[-2]
            
            # Only apply heavy smoothing for large jumps
            level_diff = target_level - prev_level
            if abs(level_diff) > 1:
                # Apply momentum to level changes
                self.level_velocity = (self.level_velocity * self.level_momentum + 
                                    level_diff * (1 - self.level_momentum))
                
                # Calculate smoothed level (allow partial steps)
                smoothed_level_float = prev_level + self.level_velocity
                
                # Convert to integer but with special handling for transitions
                if abs(smoothed_level_float - prev_level) < 0.5:
                    # Not enough change to justify a level change yet
                    level = prev_level
                else:
                    # Move one step in the right direction
                    level = prev_level + (1 if level_diff > 0 else -1)
            else:
                # For small changes (1 level or less), allow immediate transitions
                self.level_velocity = level_diff  # Reset velocity to match current change
        
        # Map state name to display name
        state_display_map = {
            # Relaxation states
            'deeply_relaxed': "Deeply Relaxed",
            'strongly_relaxed': "Strongly Relaxed",
            'moderately_relaxed': "Moderately Relaxed",
            'slightly_relaxed': "Slightly Relaxed",
            'less_relaxed': "Less Relaxed",
            'alert': "Alert",
            'tense': "Tense",
            
            # Focus states
            'deeply_focused': "Deeply Focused",
            'strongly_focused': "Strongly Focused",
            'moderately_focused': "Moderately Focused",
            'slightly_focused': "Slightly Focused",
            'less_focused': "Less Focused",
            'distracted': "Distracted",
            'very_distracted': "Very Distracted",
            
            # Other states
            'neutral': "Neutral",
            'eyes_closed': "Eyes Closed",
            'relaxed': "Relaxed",
            'focused': "Focused",
            'drowsy': "Drowsy",
            'internal_focus': "Internal Mental Activity"
        }
        
        display_state = state_display_map.get(state_name, state_name.title())
        
        result = {
            "state": display_state,
            "state_key": state_name,
            "level": level,
            "confidence": confidence,
            "value": round(value, 3),
            "smooth_value": round(smooth_value, 3)
        }
        
        return result
        
    def perform_calibration(self, session_type):
        """Perform calibration for current session type"""
        self.is_calibrating = True
        self.is_calibrated = False
        self.current_session_type = session_type
        
        # Send calibration start message
        self.zmq_publisher.publish_message({
            "message_type": "CALIBRATION_STATUS",
            "status": "STARTED",
            "session_type": session_type,
            "timestamp": time.time()
        })
        
        logger.info(f"Starting calibration for session type: {session_type}")
        
        # Clear buffer
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        
        calibration_start_time = time.time()
        last_heartbeat_time = time.time()
        last_command_check = time.time()
        calibration_metrics_list = []
        enough_samples = False
        
        # Collect calibration data
        while time.time() - calibration_start_time < CALIBRATION_DURATION_SECONDS and self.running:
            current_time = time.time()
            
            # Send heartbeat more frequently
            if current_time - last_heartbeat_time >= 0.2:  # More frequent heartbeats (200ms)
                self.zmq_publisher.send_heartbeat()
                last_heartbeat_time = current_time
            
            # Check for commands periodically
            if current_time - last_command_check >= 0.2:
                self.process_commands()
                last_command_check = current_time
                
            # Get data chunk
            chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
            
            if not chunk:
                continue
                
            # Process chunk - ADD UNFILTERED DATA TO BUFFER
            chunk_np = np.array(chunk, dtype=np.float64).T
            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
            
            # Critically important: Add to buffer EVERY TIME we get data
            self.eeg_buffer = np.append(self.eeg_buffer, eeg_chunk, axis=1)
            
            # Check if we have enough data yet
            if not enough_samples and self.eeg_buffer.shape[1] >= self.nfft:
                logger.info(f"Collected enough samples for processing: {self.eeg_buffer.shape[1]}")
                enough_samples = True
            
            # Only process data when we have enough
            if enough_samples:
                # Get window from buffer (full window)
                eeg_window = self.eeg_buffer[:, -self.nfft:]
                
                # Filter and calculate metrics
                filtered_window = self.filter_eeg_data(eeg_window)
                metrics = self.calculate_band_powers(filtered_window)
                
                if metrics:
                    calibration_metrics_list.append(metrics)
                    
                    # Publish progress update
                    progress = min(1.0, (time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS)
                    self.zmq_publisher.publish_message({
                        "message_type": "CALIBRATION_PROGRESS",
                        "progress": round(progress, 2),
                        "current_alpha": metrics['alpha'],
                        "current_beta": metrics['beta'],
                        "timestamp": time.time()
                    })
                    
            # Prevent excessive buffer growth
            max_buffer_size = int(self.sampling_rate * 10)  # Keep 10 seconds max
            if self.eeg_buffer.shape[1] > max_buffer_size:
                self.eeg_buffer = self.eeg_buffer[:, -max_buffer_size:]

            time.sleep(0.001)
        
        # Calculate baseline from calibration data
        if calibration_metrics_list:
            self.baseline_metrics = {
                'alpha': np.mean([m['alpha'] for m in calibration_metrics_list]),
                'beta': np.mean([m['beta'] for m in calibration_metrics_list]),
                'theta': np.mean([m['theta'] for m in calibration_metrics_list])
            }
            self.baseline_metrics['ab_ratio'] = (
                self.baseline_metrics['alpha'] / self.baseline_metrics['beta'] 
                if self.baseline_metrics['beta'] > 1e-9 else 0
            )
            self.baseline_metrics['bt_ratio'] = (
                self.baseline_metrics['beta'] / self.baseline_metrics['theta'] 
                if self.baseline_metrics['theta'] > 1e-9 else 0
            )
            
            self.is_calibrated = True
            self.is_calibrating = False
            
            # Record session start time
            self.session_start_time = time.time()
            
            # Send calibration complete message
            self.zmq_publisher.publish_message({
                "message_type": "CALIBRATION_STATUS",
                "status": "COMPLETED",
                "session_type": session_type,
                "baseline": self.baseline_metrics,
                "timestamp": time.time()
            })
            
            logger.info(f"Calibration complete: Alpha={self.baseline_metrics['alpha']:.2f}, Beta={self.baseline_metrics['beta']:.2f}")
            return True
        else:
            logger.error("Calibration failed: No metrics collected")
            self.is_calibrating = False
            
            # Send calibration failed message
            self.zmq_publisher.publish_message({
                "message_type": "CALIBRATION_STATUS",
                "status": "FAILED",
                "error_message": "No valid EEG data collected during calibration",
                "timestamp": time.time()
            })
            
            return False
            
    def process_commands(self):
        """Check for and process incoming commands (non-blocking)"""
        if not self.command_socket:
            return
            
        try:
            # Use poll with a short timeout to make this non-blocking
            poller = zmq.Poller()
            poller.register(self.command_socket, zmq.POLLIN)
            
            if poller.poll(10):  # 10ms timeout
                command = self.command_socket.recv_json(zmq.NOBLOCK)
                logger.info(f"Received command during calibration: {command}")
                
                # Send immediate acknowledgement
                self.command_socket.send_json({"status": "SUCCESS", "message": "Command received"})
                
                # We don't need to do anything else here since we're already calibrating
                # Just log that we got the command
        except zmq.Again:
            # No command waiting
            pass
        except Exception as e:
            logger.error(f"Error processing commands: {e}")
        
    def run_prediction_loop(self):
        """Main processing loop for EEG prediction with adaptive publishing"""
        last_prediction_time = 0
        last_published_time = 0
        last_published_value = None
        prediction_interval = ANALYSIS_WINDOW_SECONDS  # How often we analyze data
        
        while self.running:
            try:
                # Process commands & heartbeats (unchanged)
                self.process_commands()
                self.zmq_publisher.send_heartbeat()
                
                # Skip if not connected
                if not self.lsl_inlet or not self.current_session_type:
                    time.sleep(0.1)
                    continue
                    
                # Get and process data (unchanged)
                chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
                if not chunk:
                    continue
                    
                # Add to buffer (unchanged)
                chunk_np = np.array(chunk, dtype=np.float64).T
                eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
                self.eeg_buffer = np.append(self.eeg_buffer, eeg_chunk, axis=1)
                
                # Prune buffer (unchanged)
                max_buffer_size = int(self.sampling_rate * 10)
                if self.eeg_buffer.shape[1] > max_buffer_size:
                    self.eeg_buffer = self.eeg_buffer[:, -max_buffer_size:]
                
                # Analyze data at regular intervals
                current_time = time.time()
                if (current_time - last_prediction_time >= prediction_interval and 
                        self.eeg_buffer.shape[1] >= self.nfft):
                    
                    # Process the data (unchanged)
                    window_for_analysis = self.eeg_buffer[:, -self.nfft:]
                    filtered_window = self.filter_eeg_data(window_for_analysis)
                    current_metrics = self.calculate_band_powers(filtered_window)
                    
                    if current_metrics:
                        # Update history (unchanged)
                        self.recent_metrics_history.append(current_metrics)
                        if len(self.recent_metrics_history) > MAX_HISTORY_SIZE:
                            self.recent_metrics_history.pop(0)
                            
                        # Adapt baseline (unchanged)
                        if self.is_calibrated and not self.is_calibrating:
                            self.adapt_baseline(current_metrics)
                            
                        # Classify mental state
                        classification = self.classify_mental_state(current_metrics)
                        
                        # NEW: Determine if we should publish this prediction
                        should_publish = False
                        
                        # 1. Publish if minimum time has elapsed
                        if current_time - last_published_time >= PREDICTION_PUBLISH_INTERVAL:
                            should_publish = True
                        
                        # 2. Or publish if state has changed significantly
                        current_value = classification.get("smooth_value", 0.5)
                        if (last_published_value is not None and 
                                abs(current_value - last_published_value) > SIGNIFICANT_CHANGE_THRESHOLD):
                            should_publish = True
                        
                        # 3. Always publish the first prediction
                        if last_published_value is None:
                            should_publish = True
                        
                        # Publish if criteria met
                        if should_publish:
                            prediction_data = {
                                "message_type": "PREDICTION",
                                "timestamp": current_time,
                                "session_type": self.current_session_type,
                                "classification": classification,
                                "metrics": {
                                    "alpha": round(current_metrics['alpha'], 3),
                                    "beta": round(current_metrics['beta'], 3),
                                    "theta": round(current_metrics['theta'], 3),
                                    "ab_ratio": round(current_metrics['ab_ratio'], 3),
                                    "bt_ratio": round(current_metrics['bt_ratio'], 3)
                                }
                            }
                            
                            self.zmq_publisher.publish_message(prediction_data)
                            last_published_time = current_time
                            last_published_value = current_value
                        
                        # Update prediction time regardless of publishing
                        last_prediction_time = current_time
                        
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                if not self.running:
                    break
                time.sleep(0.1)

    def start_session(self, session_type):
        """Start a new session of the specified type"""
        if session_type not in ["RELAXATION", "FOCUS"]:
            logger.error(f"Invalid session type: {session_type}")
            return False
            
        # Reset state
        self.is_calibrated = False
        self.is_calibrating = False
        self.current_session_type = session_type
        self.session_start_time = None
        self.recent_metrics_history = []
        self.previous_states = []
        
        # Connect to LSL if not already connected
        if not self.lsl_inlet:
            if not self.connect_to_lsl():
                logger.error("Failed to connect to LSL. Session aborted.")
                return False
                
        # Perform calibration
        return self.perform_calibration(session_type)
        
    def stop_session(self):
        """Stop the current session"""
        self.current_session_type = None
        self.session_start_time = None
        
        # Send session ended message
        self.zmq_publisher.publish_message({
            "message_type": "SESSION_STATUS",
            "status": "ENDED",
            "timestamp": time.time()
        })
        
        logger.info("Session stopped")
        return True
        
    def cleanup(self):
        """Cleanup resources"""
        if self.lsl_inlet:
            self.lsl_inlet.close_stream()
            self.lsl_inlet = None
            
# --- Command socket for control ---
class CommandReceiver:
    def __init__(self, eeg_processor, command_address="tcp://*:5558"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(command_address)
        logger.info(f"Command receiver bound to {command_address}")
        self.eeg_processor = eeg_processor
        self.running = True
        
    def process_commands(self):
        """Process incoming commands"""
        try:
            # Non-blocking check for messages
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            
            # Wait for up to 500ms for a command
            if poller.poll(500):
                # Receive command as JSON
                command = self.socket.recv_json()
                logger.info(f"Received command: {command}")
                
                # Process command
                response = {"status": "ERROR", "message": "Unknown command"}
                
                if 'command' in command:
                    if command['command'] == 'START_SESSION':
                        if 'session_type' in command:
                            success = self.eeg_processor.start_session(command['session_type'])
                            if success:
                                response = {"status": "SUCCESS", "message": f"Started {command['session_type']} session"}
                            else:
                                response = {"status": "ERROR", "message": "Failed to start session"}
                        else:
                            response = {"status": "ERROR", "message": "Missing session_type parameter"}
                            
                    elif command['command'] == 'STOP_SESSION':
                        success = self.eeg_processor.stop_session()
                        if success:
                            response = {"status": "SUCCESS", "message": "Session stopped"}
                        else:
                            response = {"status": "ERROR", "message": "Failed to stop session"}
                            
                    elif command['command'] == 'SHUTDOWN':
                        self.running = False
                        self.eeg_processor.running = False
                        response = {"status": "SUCCESS", "message": "Shutting down"}
                        
                # Send response
                self.socket.send_json(response)
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            if self.socket:
                try:
                    # Send error response if socket is still open
                    self.socket.send_json({"status": "ERROR", "message": f"Exception: {str(e)}"})
                except:
                    pass
                    
    def cleanup(self):
        """Cleanup resources"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

# --- Main Function ---
def main():
    # Create ZMQ context and publisher
    zmq_context = zmq.Context()
    zmq_publisher = ZMQPublisher(ZMQ_PUBLISHER_ADDRESS, ZMQ_PUBLISHER_SYNC_PORT)
    
    # Create command socket
    command_socket = zmq_context.socket(zmq.REP)
    command_socket.bind("tcp://*:5558")
    logger.info("Command receiver bound to tcp://*:5558")
    
    # Create EEG processor with reference to command socket
    eeg_processor = EEGProcessor(zmq_publisher, command_socket)
    
    # Wait for subscribers
    logger.info("Waiting for subscribers...")
    zmq_publisher.wait_for_subscribers()
    
    # Setup poller for command socket
    poller = zmq.Poller()
    poller.register(command_socket, zmq.POLLIN)
    
    # Main processing loop
    try:
        while eeg_processor.running:
            # Check for commands with timeout
            socks = dict(poller.poll(timeout=100))  # 100ms timeout
            
            if command_socket in socks and socks[command_socket] == zmq.POLLIN:
                try:
                    command = command_socket.recv_json()
                    logger.info(f"Received command: {command}")
                    
                    if command.get('command') == 'START_SESSION':
                        session_type = command.get('session_type', 'GENERIC')
                        
                        # Start LSL stream if not already running
                        if eeg_processor.lsl_inlet is None:
                            eeg_processor.connect_to_lsl()
                        
                        # Send an immediate acknowledgement
                        command_socket.send_json({
                            'status': 'SUCCESS',
                            'message': f'Starting {session_type} session'
                        })
                        
                        # Perform calibration
                        eeg_processor.perform_calibration(session_type)
                        
                    elif command.get('command') == 'STOP_SESSION':
                        # Handle stop command
                        eeg_processor.stop_session()
                        command_socket.send_json({
                            'status': 'SUCCESS',
                            'message': 'Session stopped'
                        })
                        
                    elif command.get('command') == 'SHUTDOWN':
                        # Handle shutdown command
                        eeg_processor.running = False
                        command_socket.send_json({
                            'status': 'SUCCESS',
                            'message': 'Shutting down'
                        })
                        
                    else:
                        # Unknown command
                        command_socket.send_json({
                            'status': 'ERROR',
                            'message': 'Unknown command'
                        })
                        
                except Exception as e:
                    logger.error(f"Error handling command: {e}")
                    try:
                        command_socket.send_json({
                            'status': 'ERROR',
                            'message': str(e)
                        })
                    except:
                        pass
                        
            # Process incoming EEG data (if not calibrating)
            if eeg_processor.lsl_inlet and eeg_processor.is_calibrated and not eeg_processor.is_calibrating:
                eeg_processor.run_prediction_loop()
                
            # Send heartbeat periodically
            zmq_publisher.send_heartbeat()
            
            # Small sleep to prevent CPU spiking
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down.")
    finally:
        # Cleanup
        eeg_processor.running = False
        zmq_publisher.cleanup()
        if command_socket:
            command_socket.close()
        if zmq_context:
            zmq_context.term()

if __name__ == "__main__":
    main()