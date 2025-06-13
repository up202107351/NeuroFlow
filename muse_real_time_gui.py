#!/usr/bin/env python3
"""
Multi-Stream Muse EEG Monitor

Enhanced to properly handle separate LSL streams for EEG, accelerometer, 
gyroscope, and PPG data from the Muse headband.
"""

import time
import numpy as np
import pylsl
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import datetime
from collections import deque
import traceback
import scipy.signal
from pathlib import Path
import json

# --- Matplotlib Integration ---
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Import from backend modules
try:
    from backend.signal_quality_validator import SignalQualityValidator
    from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("WARNING: BrainFlow library not found. Using basic filtering.")

# --- Multi-Stream Configuration ---
STREAM_TYPES = {
    'EEG': 'EEG',
    'ACCELEROMETER': 'Accelerometer', 
    'GYROSCOPE': 'Gyroscope',
    'PPG': 'PPG'
}

# Expected sampling rates for each stream
EXPECTED_RATES = {
    'EEG': 256.0,
    'ACCELEROMETER': 52.0,
    'GYROSCOPE': 52.0,
    'PPG': 64.0
}

# Channel configurations
EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10
EEG_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

ACCELEROMETER_CHANNELS = ['X', 'Y', 'Z']
GYROSCOPE_CHANNELS = ['X', 'Y', 'Z']
PPG_CHANNELS = ['PPG1', 'PPG2', 'PPG3']

# Rest of configuration constants...
FILTER_ORDER = 2
LOWCUT = 0.5
HIGHCUT = 30.0
PLOT_DURATION_S = 5.0
PLOT_UPDATE_INTERVAL_MS = 150

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0

SESSION_TYPE_RELAX = "RELAXATION"
SESSION_TYPE_FOCUS = "FOCUS"

OUTPUT_DIR = "muse_test_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EYES_CLOSED_ALPHA_THRESHOLD = 1.5
EYES_CLOSED_MIN_DURATION = 2.0

# Dynamic state labels (same as before)
RELAXATION_STATES = {
    '1': 'Deeply Relaxed', '2': 'Relaxed', '3': 'Slightly Relaxed',
    '4': 'Neutral', '5': 'Slightly Tense', '6': 'Tense'
}

FOCUS_STATES = {
    '1': 'Highly Focused', '2': 'Focused', '3': 'Slightly Focused',
    '4': 'Neutral', '5': 'Slightly Distracted', '6': 'Distracted'
}

EVENT_LABELS = {
    's': 'START_ACTIVITY', 'e': 'END_ACTIVITY', 'b': 'BLINK_ARTIFACT',
    'm': 'MOVEMENT_ARTIFACT', 'c': 'EYES_CLOSED', 'o': 'EYES_OPEN'
}

class MultiStreamMuseEEGMonitorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Multi-Stream Muse EEG Monitor v3.0")
        master.geometry("1400x900")
        
        # Configure grid weights
        master.grid_rowconfigure(2, weight=1)
        master.grid_columnconfigure(0, weight=1)
        
        # --- Multi-Stream State Variables ---
        self.running = True
        
        # Separate LSL inlets for each stream type
        self.lsl_inlets = {
            'EEG': None,
            'ACCELEROMETER': None,
            'GYROSCOPE': None,
            'PPG': None
        }
        self.lsl_inlet = self.lsl_inlets['EEG']  # Default to EEG inlet
        
        # Stream-specific sampling rates
        self.sampling_rates = {
            'EEG': EXPECTED_RATES['EEG'],
            'ACCELEROMETER': EXPECTED_RATES['ACCELEROMETER'],
            'GYROSCOPE': EXPECTED_RATES['GYROSCOPE'],
            'PPG': EXPECTED_RATES['PPG']
        }

        self.sampling_rate = self.sampling_rates['EEG']  # Default to EEG rate
        
        # Connection status for each stream
        self.stream_status = {
            'EEG': 'Disconnected',
            'ACCELEROMETER': 'Disconnected',
            'GYROSCOPE': 'Disconnected',
            'PPG': 'Disconnected'
        }
        
        # Session variables
        self.session_type = SESSION_TYPE_RELAX
        self.baseline_metrics = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.session_start_time = None
        
        # Multi-stream data buffers
        self.eeg_data_buffer = [deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['EEG'])) for _ in range(NUM_EEG_CHANNELS)]
        self.accelerometer_buffer = {
            'x': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['ACCELEROMETER'])),
            'y': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['ACCELEROMETER'])),
            'z': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['ACCELEROMETER'])),
            'timestamps': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['ACCELEROMETER']))
        }
        self.gyroscope_buffer = {
            'x': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['GYROSCOPE'])),
            'y': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['GYROSCOPE'])),
            'z': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['GYROSCOPE'])),
            'timestamps': deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['GYROSCOPE']))
        }
        
        self.band_power_buffer = {
            'alpha': deque(maxlen=100),
            'beta': deque(maxlen=100),
            'theta': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
        self.time_buffer = deque(maxlen=int(PLOT_DURATION_S * EXPECTED_RATES['EEG']))
        
        # Mental state prediction
        self.current_prediction = {
            'state': 'Unknown', 'level': 0, 'confidence': 0.0, 'smooth_value': 0.5
        }
        self.prediction_history = deque(maxlen=50)
        
        # Eyes closed detection
        self.eyes_state = "Unknown"
        self.eyes_closed_start_time = None
        self.eyes_state_history = deque(maxlen=20)
        
        # Annotation tracking
        self.current_user_state = "Unknown"
        self.user_state_changes = []
        self.user_events = []
        
        # Enhanced session data with multi-stream support
        self.session_data = {
            'eeg_raw': [],
            'accelerometer_data': [],
            'gyroscope_data': [],
            'ppg_data': [],
            'timestamps': {
                'eeg': [],
                'accelerometer': [],
                'gyroscope': [],
                'ppg': []
            },
            'band_powers': [],
            'predictions': [],
            'state_changes': [],
            'events': [],
            'eyes_states': [],
            'session_metadata': {}
        }
        
        # Enhanced signal quality validator
        self.signal_quality_validator = SignalQualityValidator()

        # Enhanced prediction smoothing
        self.prediction_smoothing_window = 5  # Number of predictions to average
        self.min_confidence_for_change = 0.6   # Minimum confidence to change state
        self.state_change_threshold = 1.0      # Minimum level change to trigger state change
        self.last_stable_prediction = None
        self.stable_prediction_count = 0
        self.min_stable_count = 3              # Need 3 consistent predictions before changing
        
        # Enhanced eyes state tracking
        self.eyes_detection_history = deque(maxlen=20)
        self.eyes_closed_confidence = 0.0
        self.eyes_state_persistence = 5        # Number of samples to maintain state
        self.current_eyes_persistence = 0
        
        # Event marker tracking for plots
        self.plot_events = []  # Store events with relative timestamps for plotting
        
        # Filtering parameters
        self.filter_b = None
        self.filter_a = None
        self.update_filter_coefficients()
        
        # Dynamic state buttons storage
        self.state_buttons = []
        
        # Multi-stream threading
        self.stream_threads = {}
        
        # Set up GUI layout
        self.setup_gui()
        
        # Start multi-stream connections
        self.start_multi_stream_connections()
        
        # Set up plot update scheduler
        self.update_plots_scheduler()
        
        # Set up keyboard monitoring
        self.setup_keyboard_monitoring()
        
        # Cleanup on close
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """Set up enhanced GUI with multi-stream status"""
        
        # Enhanced top frame with multi-stream status
        self.top_frame = ttk.Frame(self.master, padding=5)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.top_frame.grid_columnconfigure(1, weight=1)
        
        # Multi-stream status display
        status_frame = ttk.LabelFrame(self.top_frame, text="Stream Status", padding=3)
        status_frame.grid(row=0, column=0, sticky="w", padx=5)
        
        self.status_vars = {}
        for i, stream_type in enumerate(['EEG', 'ACCELEROMETER', 'GYROSCOPE', 'PPG']):
            self.status_vars[stream_type] = tk.StringVar(value=f"{stream_type}: Disconnected")
            label = ttk.Label(status_frame, textvariable=self.status_vars[stream_type], font=("Arial", 8))
            label.grid(row=i//2, column=i%2, sticky="w", padx=3, pady=1)
        
        # Session type in middle
        session_frame = ttk.Frame(self.top_frame)
        session_frame.grid(row=0, column=1, padx=20)
        
        ttk.Label(session_frame, text="Session:").pack(side=tk.LEFT, padx=(0, 5))
        self.session_type_var = tk.StringVar(value=SESSION_TYPE_RELAX)
        session_combo = ttk.Combobox(session_frame, textvariable=self.session_type_var, state="readonly", width=12)
        session_combo['values'] = (SESSION_TYPE_RELAX, SESSION_TYPE_FOCUS)
        session_combo.bind('<<ComboboxSelected>>', self.on_session_type_changed)
        session_combo.pack(side=tk.LEFT)
        
        # Current state and quality on right
        right_frame = ttk.Frame(self.top_frame)
        right_frame.grid(row=0, column=2, sticky="e", padx=5)
        
        self.current_state_var = tk.StringVar(value=f"State: {self.current_user_state}")
        self.current_state_label = ttk.Label(right_frame, textvariable=self.current_state_var, font=("Arial", 10, "bold"))
        self.current_state_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.eyes_state_var = tk.StringVar(value="Eyes: Unknown")
        self.eyes_state_label = ttk.Label(right_frame, textvariable=self.eyes_state_var, font=("Arial", 10))
        self.eyes_state_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quality_var = tk.StringVar(value="Quality: Unknown")
        self.quality_label = ttk.Label(right_frame, textvariable=self.quality_var, font=("Arial", 9))
        self.quality_label.pack(side=tk.LEFT)
        
        # Enhanced prediction frame
        self.prediction_frame = ttk.LabelFrame(self.master, text="Mental State & Physiological Monitoring", padding=5)
        self.prediction_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        
        # Prediction info frame (same layout as before)
        pred_info_frame = ttk.Frame(self.prediction_frame)
        pred_info_frame.grid(row=0, column=0, sticky="ew")
        pred_info_frame.grid_columnconfigure(1, weight=1)
        
        # Prediction state
        self.prediction_state_var = tk.StringVar(value="Unknown")
        self.prediction_state_label = ttk.Label(pred_info_frame, textvariable=self.prediction_state_var, font=("Arial", 14, "bold"))
        self.prediction_state_label.grid(row=0, column=0, sticky="w", padx=(0, 20))
        
        # Progress and details
        progress_frame = ttk.Frame(pred_info_frame)
        progress_frame.grid(row=0, column=1, sticky="ew", padx=10)
        progress_frame.grid_columnconfigure(0, weight=1)
        
        self.prediction_details_var = tk.StringVar(value="Level: 0 | Confidence: N/A | Eyes: Unknown")
        ttk.Label(progress_frame, textvariable=self.prediction_details_var, font=("Arial", 9)).grid(row=0, column=0, sticky="ew")
        
        self.prediction_progress = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.prediction_progress.grid(row=1, column=0, sticky="ew", pady=2)
        
        # Baseline comparison
        baseline_frame = ttk.Frame(pred_info_frame)
        baseline_frame.grid(row=0, column=2, sticky="e")
        
        ttk.Label(baseline_frame, text="vs Baseline:", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky="e")
        
        self.alpha_comparison_var = tk.StringVar(value="α: N/A")
        self.beta_comparison_var = tk.StringVar(value="β: N/A")
        self.theta_comparison_var = tk.StringVar(value="θ: N/A")
        
        ttk.Label(baseline_frame, textvariable=self.alpha_comparison_var, font=("Arial", 8)).grid(row=1, column=0, sticky="e")
        ttk.Label(baseline_frame, textvariable=self.beta_comparison_var, font=("Arial", 8)).grid(row=2, column=0, sticky="e")
        ttk.Label(baseline_frame, textvariable=self.theta_comparison_var, font=("Arial", 8)).grid(row=3, column=0, sticky="e")
        
        # Plot frame
        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=2)
        self.plot_frame.grid_rowconfigure(0, weight=2)
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.master, padding=5)
        self.control_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
        
        # Main control buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        self.connect_button = ttk.Button(button_frame, text="Connect to Muse", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        self.calibrate_button = ttk.Button(button_frame, text="Start Calibration", command=self.start_calibration, state=tk.DISABLED)
        self.calibrate_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Multi-Stream Session", command=self.save_session_data, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Dynamic state buttons frame
        self.state_frame = ttk.LabelFrame(self.control_frame, text="Report Current State", padding=3)
        self.state_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Event buttons frame
        self.event_frame = ttk.LabelFrame(self.control_frame, text="Mark Events", padding=3)
        self.event_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        for i, (key, label) in enumerate(EVENT_LABELS.items()):
            btn = ttk.Button(
                self.event_frame, 
                text=f"{key}: {label.replace('_', ' ')}", 
                command=lambda l=label: self.add_event_marker(l),
                width=15
            )
            btn.grid(row=0, column=i, padx=1, pady=2, sticky="ew")
        
        for col in range(len(EVENT_LABELS)):
            self.event_frame.grid_columnconfigure(col, weight=1)
        
        # Set up plots
        self.setup_plots()
        
        # Create initial state buttons
        self.create_state_buttons()
    
    def start_multi_stream_connections(self):
        """Start separate threads for each LSL stream type"""
        # Start EEG stream thread (highest priority)
        self.stream_threads['EEG'] = threading.Thread(target=self.eeg_stream_loop, daemon=True)
        self.stream_threads['EEG'].start()
        
        # Start accelerometer stream thread
        self.stream_threads['ACCELEROMETER'] = threading.Thread(target=self.accelerometer_stream_loop, daemon=True)
        self.stream_threads['ACCELEROMETER'].start()
        
        # Start gyroscope stream thread (optional)
        self.stream_threads['GYROSCOPE'] = threading.Thread(target=self.gyroscope_stream_loop, daemon=True)
        self.stream_threads['GYROSCOPE'].start()
        
        # Start PPG stream thread (optional)
        self.stream_threads['PPG'] = threading.Thread(target=self.ppg_stream_loop, daemon=True)
        self.stream_threads['PPG'].start()
    
    def connect_to_stream(self, stream_type):
        """Connect to a specific LSL stream type"""
        try:
            print(f"Looking for {stream_type} stream...")
            streams = pylsl.resolve_byprop('type', STREAM_TYPES[stream_type], 1, timeout=3.0)
            
            if not streams:
                self.master.after(0, lambda: self.update_stream_status(stream_type, "Not Found"))
                return None
            
            inlet = pylsl.StreamInlet(streams[0], max_chunklen=128)
            info = inlet.info()
            
            # Update sampling rate from stream
            actual_rate = info.nominal_srate()
            if actual_rate > 0:
                self.sampling_rates[stream_type] = actual_rate
            
            self.master.after(0, lambda: self.update_stream_status(stream_type, f"Connected ({actual_rate:.1f} Hz)"))
            return inlet
            
        except Exception as e:
            self.master.after(0, lambda: self.update_stream_status(stream_type, f"Error: {str(e)[:20]}"))
            return None
    
    def update_stream_status(self, stream_type, status):
        """Update the status display for a specific stream"""
        self.stream_status[stream_type] = status
        self.status_vars[stream_type].set(f"{stream_type}: {status}")
        
        # Update color based on status
        if "Connected" in status:
            # Find the label widget and update color - this is a simplified approach
            pass  # Could add color coding here
    
    def eeg_stream_loop(self):
        """Main EEG data processing loop"""
        connection_attempts = 0
        max_attempts = 3
        
        while self.running and connection_attempts < max_attempts:
            # Try to connect to EEG stream
            self.lsl_inlets['EEG'] = self.connect_to_stream('EEG')
            
            if self.lsl_inlets['EEG'] is None:
                connection_attempts += 1
                time.sleep(2)
                continue
            
            # Update filter coefficients based on actual EEG sampling rate
            self.update_filter_coefficients()
            
            # Enable calibration button
            self.master.after(0, lambda: self.calibrate_button.config(state=tk.NORMAL))
            
            # Main EEG processing loop
            last_band_power_calc_time = 0
            last_quality_check_time = 0
            
            while self.running and self.lsl_inlets['EEG']:
                try:
                    chunk, timestamps = self.lsl_inlets['EEG'].pull_chunk(timeout=0.1, max_samples=64)
                    
                    if chunk and len(chunk) > 0:
                        chunk_np = np.array(chunk, dtype=np.float64).T
                        
                        # Extract EEG channels (should be first 4 channels)
                        if chunk_np.shape[0] >= NUM_EEG_CHANNELS:
                            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
                            
                            # Filter EEG data
                            eeg_filtered = self.filter_eeg_data(eeg_chunk)
                            
                            # Store in buffer
                            for i in range(NUM_EEG_CHANNELS):
                                self.eeg_data_buffer[i].extend(eeg_filtered[i, :])
                            
                            for ts in timestamps:
                                self.time_buffer.append(ts)
                            
                            # Process band powers periodically
                            current_time = time.time()
                            if current_time - last_band_power_calc_time > 0.5:
                                if all(len(buf) >= int(self.sampling_rates['EEG'] * 2) for buf in self.eeg_data_buffer):
                                    data_for_analysis = np.array([list(buf)[-int(self.sampling_rates['EEG'] * 2):] for buf in self.eeg_data_buffer])
                                    
                                    metrics = self.calculate_band_powers(data_for_analysis)
                                    
                                    if metrics:
                                        # Store band power data
                                        self.band_power_buffer['alpha'].append(metrics['alpha'])
                                        self.band_power_buffer['beta'].append(metrics['beta'])
                                        self.band_power_buffer['theta'].append(metrics['theta'])
                                        self.band_power_buffer['timestamps'].append(current_time)
                                        
                                        # Eyes state detection
                                        eyes_state = self.detect_eyes_closed(metrics)
                                        if eyes_state != self.eyes_state:
                                            self.eyes_state = eyes_state
                                            self.master.after(0, lambda: self.update_eyes_display())
                                            
                                            if self.is_calibrated:
                                                self.session_data['eyes_states'].append({
                                                    'timestamp': current_time,
                                                    'state': eyes_state
                                                })
                                        
                                        # Mental state prediction if calibrated
                                        if self.is_calibrated:
                                            prediction = self.classify_mental_state(metrics)
                                            self.current_prediction = prediction
                                            self.prediction_history.append(prediction)
                                            
                                            self.master.after(0, lambda: self.update_prediction_display())
                                            self.master.after(0, lambda m=metrics: self.update_baseline_comparison(m))
                                            
                                            self.session_data['predictions'].append({
                                                'timestamp': current_time,
                                                'prediction': prediction,
                                                'metrics': metrics,
                                                'eyes_state': eyes_state
                                            })
                                        
                                        # Store band power data
                                        self.session_data['band_powers'].append({
                                            'timestamp': current_time,
                                            'alpha': metrics['alpha'],
                                            'beta': metrics['beta'],
                                            'theta': metrics['theta']
                                        })
                                
                                last_band_power_calc_time = current_time
                            
                            # Store raw EEG data if recording
                            if self.is_calibrated and self.session_start_time:
                                self.session_data['eeg_raw'].append(eeg_chunk.copy())
                                self.session_data['timestamps']['eeg'].extend(timestamps)
                
                except Exception as e:
                    print(f"Error in EEG processing: {e}")
                    time.sleep(0.1)
            
            # Connection lost, try to reconnect
            self.lsl_inlets['EEG'] = None
            self.master.after(0, lambda: self.update_stream_status('EEG', "Reconnecting..."))
            connection_attempts += 1
            time.sleep(1)
        
        # Failed to maintain connection
        self.master.after(0, lambda: self.update_stream_status('EEG', "Failed"))
        print("EEG stream loop exiting")
    
    def accelerometer_stream_loop(self):
        """Accelerometer data processing loop"""
        connection_attempts = 0
        max_attempts = 3
        
        while self.running and connection_attempts < max_attempts:
            self.lsl_inlets['ACCELEROMETER'] = self.connect_to_stream('ACCELEROMETER')
            
            if self.lsl_inlets['ACCELEROMETER'] is None:
                connection_attempts += 1
                time.sleep(2)
                continue
            
            # Main accelerometer processing loop
            while self.running and self.lsl_inlets['ACCELEROMETER']:
                try:
                    chunk, timestamps = self.lsl_inlets['ACCELEROMETER'].pull_chunk(timeout=0.1, max_samples=32)
                    
                    if chunk and len(chunk) > 0:
                        chunk_np = np.array(chunk, dtype=np.float64).T
                        
                        # Store accelerometer data
                        if chunk_np.shape[0] >= 3:  # X, Y, Z
                            self.accelerometer_buffer['x'].extend(chunk_np[0, :])
                            self.accelerometer_buffer['y'].extend(chunk_np[1, :])
                            self.accelerometer_buffer['z'].extend(chunk_np[2, :])
                            self.accelerometer_buffer['timestamps'].extend(timestamps)
                            
                            # Send to signal quality validator
                            for i in range(chunk_np.shape[1]):
                                acc_sample = chunk_np[:3, i]  # X, Y, Z
                                self.signal_quality_validator.add_accelerometer_data(acc_sample)
                            
                            # Store in session data if recording
                            if self.is_calibrated and self.session_start_time:
                                self.session_data['accelerometer_data'].append(chunk_np.copy())
                                self.session_data['timestamps']['accelerometer'].extend(timestamps)
                
                except Exception as e:
                    print(f"Error in accelerometer processing: {e}")
                    time.sleep(0.1)
            
            self.lsl_inlets['ACCELEROMETER'] = None
            self.master.after(0, lambda: self.update_stream_status('ACCELEROMETER', "Reconnecting..."))
            connection_attempts += 1
            time.sleep(1)
        
        self.master.after(0, lambda: self.update_stream_status('ACCELEROMETER', "Failed"))
        print("Accelerometer stream loop exiting")
    
    def gyroscope_stream_loop(self):
        """Gyroscope data processing loop"""
        connection_attempts = 0
        max_attempts = 3
        
        while self.running and connection_attempts < max_attempts:
            self.lsl_inlets['GYROSCOPE'] = self.connect_to_stream('GYROSCOPE')
            
            if self.lsl_inlets['GYROSCOPE'] is None:
                connection_attempts += 1
                time.sleep(2)
                continue
            
            while self.running and self.lsl_inlets['GYROSCOPE']:
                try:
                    chunk, timestamps = self.lsl_inlets['GYROSCOPE'].pull_chunk(timeout=0.1, max_samples=32)
                    
                    if chunk and len(chunk) > 0:
                        chunk_np = np.array(chunk, dtype=np.float64).T
                        
                        if chunk_np.shape[0] >= 3:  # X, Y, Z
                            self.gyroscope_buffer['x'].extend(chunk_np[0, :])
                            self.gyroscope_buffer['y'].extend(chunk_np[1, :])
                            self.gyroscope_buffer['z'].extend(chunk_np[2, :])
                            self.gyroscope_buffer['timestamps'].extend(timestamps)
                            
                            # Store in session data if recording
                            if self.is_calibrated and self.session_start_time:
                                self.session_data['gyroscope_data'].append(chunk_np.copy())
                                self.session_data['timestamps']['gyroscope'].extend(timestamps)
                
                except Exception as e:
                    print(f"Error in gyroscope processing: {e}")
                    time.sleep(0.1)
            
            self.lsl_inlets['GYROSCOPE'] = None
            self.master.after(0, lambda: self.update_stream_status('GYROSCOPE', "Reconnecting..."))
            connection_attempts += 1
            time.sleep(1)
        
        self.master.after(0, lambda: self.update_stream_status('GYROSCOPE', "Failed"))
        print("Gyroscope stream loop exiting")
    
    def ppg_stream_loop(self):
        """PPG data processing loop (optional)"""
        connection_attempts = 0
        max_attempts = 3
        
        while self.running and connection_attempts < max_attempts:
            self.lsl_inlets['PPG'] = self.connect_to_stream('PPG')
            
            if self.lsl_inlets['PPG'] is None:
                connection_attempts += 1
                time.sleep(2)
                continue
            
            while self.running and self.lsl_inlets['PPG']:
                try:
                    chunk, timestamps = self.lsl_inlets['PPG'].pull_chunk(timeout=0.1, max_samples=32)
                    
                    if chunk and len(chunk) > 0:
                        chunk_np = np.array(chunk, dtype=np.float64).T
                        
                        # Store in session data if recording
                        if self.is_calibrated and self.session_start_time:
                            self.session_data['ppg_data'].append(chunk_np.copy())
                            self.session_data['timestamps']['ppg'].extend(timestamps)
                
                except Exception as e:
                    print(f"Error in PPG processing: {e}")
                    time.sleep(0.1)
            
            self.lsl_inlets['PPG'] = None
            self.master.after(0, lambda: self.update_stream_status('PPG', "Reconnecting..."))
            connection_attempts += 1
            time.sleep(1)
        
        self.master.after(0, lambda: self.update_stream_status('PPG', "Failed"))
        print("PPG stream loop exiting")
    
    # ... Include other methods from previous version ...
    # (get_current_state_labels, on_session_type_changed, create_state_buttons, 
    #  setup_plots, setup_keyboard_monitoring, update_filter_coefficients, 
    #  detect_eyes_closed, etc.)
    
    def get_current_state_labels(self):
        """Get state labels based on current session type"""
        if self.session_type == SESSION_TYPE_RELAX:
            return RELAXATION_STATES
        else:
            return FOCUS_STATES
    
    def toggle_connection(self):
        """Toggle multi-stream connections"""
        if all(inlet is None for inlet in self.lsl_inlets.values()):
            # Start connections
            self.connect_button.config(text="Connecting...", state=tk.DISABLED)
            # Connections are handled by the stream threads
            time.sleep(1)  # Give threads time to connect
            self.connect_button.config(text="Disconnect", state=tk.NORMAL)
        else:
            # Disconnect all streams
            self.disconnect_all_streams()
            self.connect_button.config(text="Connect to Muse")
            self.calibrate_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
    
    def disconnect_all_streams(self):
        """Disconnect from all LSL streams"""
        for stream_type, inlet in self.lsl_inlets.items():
            if inlet:
                try:
                    inlet.close_stream()
                except:
                    pass
                self.lsl_inlets[stream_type] = None
                self.update_stream_status(stream_type, "Disconnected")
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit? Unsaved data will be lost."):
            self.running = False
            
            # Disconnect all streams
            self.disconnect_all_streams()
            
            # Wait for threads to finish
            for thread in self.stream_threads.values():
                if thread and thread.is_alive():
                    thread.join(timeout=1.0)
            
            self.master.destroy()
    
    def get_current_state_labels(self):
        """Get state labels based on current session type"""
        if self.session_type == SESSION_TYPE_RELAX:
            return RELAXATION_STATES
        else:
            return FOCUS_STATES
    
    
    def on_session_type_changed(self, event=None):
        """Handle session type change"""
        self.session_type = self.session_type_var.get()
        self.create_state_buttons()  # Recreate buttons for new session type
        print(f"Session type changed to: {self.session_type}")
    
    def create_state_buttons(self):
        """Create state buttons based on current session type"""
        # Clear existing buttons
        for btn in self.state_buttons:
            btn.destroy()
        self.state_buttons.clear()
        
        # Get appropriate state labels
        state_labels = self.get_current_state_labels()
        
        # Update frame title
        session_name = "Relaxation" if self.session_type == SESSION_TYPE_RELAX else "Focus"
        self.state_frame.config(text=f"Report Current {session_name} State")
        
        # Create new buttons in a grid (2 rows x 3 columns)
        for i, (key, label) in enumerate(state_labels.items()):
            row = i // 3
            col = i % 3
            btn = ttk.Button(
                self.state_frame, 
                text=f"{key}: {label}", 
                command=lambda l=label: self.update_user_state(l),
                width=18
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            self.state_buttons.append(btn)
        
        # Configure state frame columns to expand equally
        for col in range(3):
            self.state_frame.grid_columnconfigure(col, weight=1)
    
    def setup_plots(self):
        """Set up compact matplotlib plot canvases"""
        # EEG Plot
        self.eeg_fig = Figure(figsize=(12, 4), dpi=80)
        self.eeg_axes = []
        
        for i in range(NUM_EEG_CHANNELS):
            ax = self.eeg_fig.add_subplot(NUM_EEG_CHANNELS, 1, i+1)
            ax.set_ylabel(EEG_CHANNEL_NAMES[i], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if i < NUM_EEG_CHANNELS - 1:
                ax.set_xticklabels([])
            self.eeg_axes.append(ax)
        
        self.eeg_lines = [ax.plot([], [], 'b-', linewidth=1)[0] for ax in self.eeg_axes]
        self.eeg_axes[-1].set_xlabel('Time (s)', fontsize=9)
        self.eeg_fig.tight_layout(pad=1.0)
        
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=self.plot_frame)
        self.eeg_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Band Powers Plot
        self.band_fig = Figure(figsize=(12, 2.5), dpi=80)
        self.band_ax = self.band_fig.add_subplot(111)
        self.band_ax.set_title("Band Powers with Baseline & Eyes State", fontsize=10)
        self.band_ax.set_ylabel("Power", fontsize=9)
        self.band_ax.set_xlabel("Time (samples)", fontsize=9)
        self.band_ax.grid(True, alpha=0.3)
        self.band_ax.tick_params(labelsize=8)
        
        self.alpha_line, = self.band_ax.plot([], [], 'b-', label='Alpha', linewidth=2)
        self.beta_line, = self.band_ax.plot([], [], 'r-', label='Beta', linewidth=2)
        self.theta_line, = self.band_ax.plot([], [], 'g-', label='Theta', linewidth=2)
        self.band_ax.legend(fontsize=8)
        
        # Store baseline line references
        self.baseline_lines = {'alpha': None, 'beta': None, 'theta': None}
        
        self.band_fig.tight_layout(pad=1.0)
        self.band_canvas = FigureCanvasTkAgg(self.band_fig, master=self.plot_frame)
        self.band_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
    
    def setup_keyboard_monitoring(self):
        """Set up keyboard shortcuts for state and event marking"""
        def on_key_press(event):
            key = event.char
            
            state_labels = self.get_current_state_labels()
            if key in state_labels:
                self.update_user_state(state_labels[key])
            elif key in EVENT_LABELS:
                self.add_event_marker(EVENT_LABELS[key])
        
        self.master.bind('<Key>', on_key_press)
        self.master.focus_set()
    
    def update_filter_coefficients(self):
        """Update filter coefficients based on current sampling rate"""
        nyq = 0.5 * self.sampling_rate
        low = LOWCUT / nyq
        high = HIGHCUT / nyq
        self.filter_b, self.filter_a = scipy.signal.butter(FILTER_ORDER, [low, high], btype='band', analog=False)
    
    def filter_eeg_data(self, eeg_data):
        """Apply bandpass filter to EEG data with improved error handling"""
        min_samples = max(3 * FILTER_ORDER + 1, 10)
        
        if eeg_data.shape[1] < min_samples:
            return eeg_data
        
        eeg_filtered = np.zeros_like(eeg_data)
        for i in range(NUM_EEG_CHANNELS):
            try:
                padlen = min(3 * FILTER_ORDER, eeg_data.shape[1] // 4)
                if padlen < 1:
                    padlen = None
                
                eeg_filtered[i] = scipy.signal.filtfilt(
                    self.filter_b, self.filter_a, 
                    eeg_data[i],
                    padlen=padlen
                )
            except Exception as e:
                eeg_filtered[i] = eeg_data[i]
        
        return eeg_filtered
    
    def calculate_band_powers(self, eeg_segment):
        """Calculate band powers from EEG segment"""
        if eeg_segment.shape[1] < int(self.sampling_rate):
            return None
        
        metrics_list = []
        for ch_idx in range(NUM_EEG_CHANNELS):
            ch_data = eeg_segment[ch_idx, :].copy()
            
            try:
                freqs, psd = scipy.signal.welch(ch_data, fs=self.sampling_rate, nperseg=min(int(self.sampling_rate*2), len(ch_data)))
                
                theta_idx = np.logical_and(freqs >= THETA_BAND[0], freqs <= THETA_BAND[1])
                alpha_idx = np.logical_and(freqs >= ALPHA_BAND[0], freqs <= ALPHA_BAND[1])
                beta_idx = np.logical_and(freqs >= BETA_BAND[0], freqs <= BETA_BAND[1])
                
                metrics_list.append({
                    'theta': np.mean(psd[theta_idx]) if np.any(theta_idx) else 0,
                    'alpha': np.mean(psd[alpha_idx]) if np.any(alpha_idx) else 0,
                    'beta': np.mean(psd[beta_idx]) if np.any(beta_idx) else 0
                })
            except Exception as e:
                print(f"Error calculating PSD for channel {ch_idx}: {e}")
                return None
        
        if not metrics_list:
            return None
        
        avg_metrics = {
            'theta': np.mean([m['theta'] for m in metrics_list]),
            'alpha': np.mean([m['alpha'] for m in metrics_list]),
            'beta': np.mean([m['beta'] for m in metrics_list])
        }
        
        avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
        avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
        
        return avg_metrics
    
    def classify_mental_state(self, current_metrics):
        """Enhanced classify mental state with smoothing and stability"""
        if not self.baseline_metrics or not current_metrics:
            return {
                "state": "Unknown", "level": 0, "confidence": 0.0,
                "value": 0.5, "smooth_value": 0.5, "state_key": "unknown"
            }
        
        # Calculate ratios relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha'] if self.baseline_metrics['alpha'] > 0 else 1
        beta_ratio = current_metrics['beta'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 0 else 1
        theta_ratio = current_metrics['theta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 0 else 1
        
        # Session-specific classification
        if self.session_type == SESSION_TYPE_RELAX:
            # For relaxation: higher alpha is better, lower beta/theta is better
            if alpha_ratio > 1.4 and beta_ratio < 0.8:
                raw_state, raw_level = "Deeply Relaxed", 3
                confidence = min(0.95, 0.5 + (alpha_ratio - 1.4) * 0.3)
            elif alpha_ratio > 1.2:
                raw_state, raw_level = "Relaxed", 2
                confidence = min(0.9, 0.5 + (alpha_ratio - 1.2) * 0.5)
            elif alpha_ratio > 1.05:
                raw_state, raw_level = "Slightly Relaxed", 1
                confidence = min(0.8, 0.4 + (alpha_ratio - 1.05) * 2.0)
            elif alpha_ratio < 0.8 and beta_ratio > 1.2:
                raw_state, raw_level = "Tense", -2
                confidence = min(0.9, 0.5 + (1.2 - alpha_ratio) * 0.5)
            elif alpha_ratio < 0.9:
                raw_state, raw_level = "Slightly Tense", -1
                confidence = min(0.8, 0.4 + (0.9 - alpha_ratio) * 2.0)
            else:
                raw_state, raw_level = "Neutral", 0
                confidence = 0.6
        else:  # FOCUS
            bt_ratio = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] if self.baseline_metrics['bt_ratio'] > 0 else 1
            if bt_ratio > 1.4 and beta_ratio > 1.2:
                raw_state, raw_level = "Highly Focused", 3
                confidence = min(0.95, 0.5 + (bt_ratio - 1.4) * 0.3)
            elif bt_ratio > 1.2:
                raw_state, raw_level = "Focused", 2
                confidence = min(0.9, 0.5 + (bt_ratio - 1.2) * 0.5)
            elif bt_ratio > 1.05:
                raw_state, raw_level = "Slightly Focused", 1
                confidence = min(0.8, 0.4 + (bt_ratio - 1.05) * 2.0)
            elif bt_ratio < 0.8 or theta_ratio > 1.3:
                raw_state, raw_level = "Distracted", -2
                confidence = min(0.9, 0.5 + (0.8 - bt_ratio) * 0.5)
            elif bt_ratio < 0.9:
                raw_state, raw_level = "Slightly Distracted", -1
                confidence = min(0.8, 0.4 + (0.9 - bt_ratio) * 2.0)
            else:
                raw_state, raw_level = "Neutral", 0
                confidence = 0.6
        
        # Apply smoothing and stability logic
        smoothed_prediction = self.apply_prediction_smoothing(raw_state, raw_level, confidence)
        
        return smoothed_prediction
    
    def apply_prediction_smoothing(self, raw_state, raw_level, confidence):
        """Apply smoothing and stability requirements to predictions"""
        # Calculate smoothed value
        raw_value = (raw_level + 3) / 6.0  # Normalize to 0-1
        
        # Get previous smoothed value
        if self.prediction_history:
            prev_smooth_value = self.prediction_history[-1]['smooth_value']
            prev_state = self.prediction_history[-1]['state']
            prev_level = self.prediction_history[-1]['level']
        else:
            prev_smooth_value = 0.5
            prev_state = "Unknown"
            prev_level = 0
        
        # Apply exponential smoothing
        smoothing_factor = 0.3 if confidence > 0.8 else 0.1  # Faster smoothing for high confidence
        smooth_value = smoothing_factor * raw_value + (1 - smoothing_factor) * prev_smooth_value
        
        # Determine if we should change the displayed state
        level_change = abs(raw_level - prev_level)
        
        # Check stability requirements
        if (confidence >= self.min_confidence_for_change and 
            level_change >= self.state_change_threshold):
            
            # Check if this is consistent with recent predictions
            if self.last_stable_prediction == raw_state:
                self.stable_prediction_count += 1
            else:
                self.last_stable_prediction = raw_state
                self.stable_prediction_count = 1
            
            # Only change state if we have enough consistent predictions
            if self.stable_prediction_count >= self.min_stable_count:
                final_state = raw_state
                final_level = raw_level
                final_confidence = confidence
            else:
                # Keep previous state but update smooth_value
                final_state = prev_state
                final_level = prev_level
                final_confidence = confidence * 0.7  # Reduce confidence during transition
        else:
            # Not confident enough or change too small - keep previous state
            final_state = prev_state
            final_level = prev_level
            final_confidence = confidence * 0.8
        
        return {
            "state": final_state,
            "state_key": final_state.lower().replace(' ', '_'),
            "level": final_level,
            "confidence": final_confidence,
            "value": raw_value,
            "smooth_value": smooth_value,
            "raw_state": raw_state,
            "raw_level": raw_level,
            "raw_confidence": confidence
        }
    
    def detect_eyes_closed(self, current_metrics):
        """Enhanced eyes detection with persistence and confidence"""
        if not self.baseline_metrics or not current_metrics:
            return "Unknown"
        
        # Calculate alpha ratio relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha']
        
        # Build confidence score based on alpha elevation
        if alpha_ratio > EYES_CLOSED_ALPHA_THRESHOLD:
            confidence_increase = min(0.3, (alpha_ratio - EYES_CLOSED_ALPHA_THRESHOLD) * 0.2)
            self.eyes_closed_confidence = min(1.0, self.eyes_closed_confidence + confidence_increase)
        else:
            confidence_decrease = min(0.2, (EYES_CLOSED_ALPHA_THRESHOLD - alpha_ratio) * 0.1)
            self.eyes_closed_confidence = max(0.0, self.eyes_closed_confidence - confidence_decrease)
        
        # Store detection data
        self.eyes_detection_history.append({
            'timestamp': time.time(),
            'alpha_ratio': alpha_ratio,
            'confidence': self.eyes_closed_confidence
        })
        
        # Determine state with persistence
        if self.eyes_closed_confidence > 0.7:
            if self.eyes_state != "Closed":
                self.current_eyes_persistence = 0
            self.current_eyes_persistence += 1
            
            if self.current_eyes_persistence >= self.eyes_state_persistence:
                return "Closed"
            else:
                return "Closing"
        elif self.eyes_closed_confidence < 0.3:
            if self.eyes_state != "Open":
                self.current_eyes_persistence = 0
            self.current_eyes_persistence += 1
            
            if self.current_eyes_persistence >= self.eyes_state_persistence:
                return "Open"
            else:
                return "Opening"
        else:
            return "Uncertain"
    
    def add_event_marker(self, event_label):
        """Enhanced event marker with plot tracking"""
        timestamp = time.time()
        
        # Calculate relative timestamp for plots
        if self.session_start_time:
            relative_time = timestamp - self.session_start_time
            self.plot_events.append({
                'time': relative_time,
                'label': event_label,
                'absolute_time': timestamp
            })
        
        # Store event in session data
        self.user_events.append({
            "time": timestamp,
            "label": event_label
        })
        
        if self.is_calibrated:
            self.session_data['events'].append({
                "time": timestamp,
                "label": event_label
            })
        
        # Print event with context
        if self.session_start_time:
            time_rel = timestamp - self.session_start_time
            print(f"\n>>> Event @ {time_rel:.1f}s: {event_label}")
        else:
            print(f"\n>>> Event: {event_label}")
    
    def update_plots(self):
        """Enhanced plots with event markers and smoother updates"""
        try:
            # Update EEG plot (same as before)
            if len(self.time_buffer) > 0 and all(len(buf) > 0 for buf in self.eeg_data_buffer):
                if self.session_start_time:
                    relative_times = np.array([t - self.session_start_time for t in self.time_buffer])
                else:
                    relative_times = np.array([t - self.time_buffer[0] for t in self.time_buffer])
                
                for i, line in enumerate(self.eeg_lines):
                    if i < len(self.eeg_data_buffer):
                        line.set_xdata(relative_times)
                        line.set_ydata(list(self.eeg_data_buffer[i]))
                
                # Add event markers to EEG plot
                self.clear_event_markers()
                for event in self.plot_events[-10:]:  # Show last 10 events
                    if event['time'] >= min(relative_times) and event['time'] <= max(relative_times):
                        for ax in self.eeg_axes:
                            line = ax.axvline(event['time'], color='red', linestyle='--', alpha=0.7, linewidth=1)
                            ax.text(event['time'], ax.get_ylim()[1]*0.9, event['label'][:10], 
                                   rotation=90, ha='right', va='top', fontsize=7, color='red')
                
                if len(relative_times) > 0:
                    for ax in self.eeg_axes:
                        ax.set_xlim(min(relative_times), max(relative_times))
                        
                        if len(self.eeg_data_buffer[0]) > 0:
                            all_data = []
                            for buf in self.eeg_data_buffer:
                                all_data.extend(buf)
                            if all_data:
                                ymin, ymax = min(all_data), max(all_data)
                                padding = (ymax - ymin) * 0.1 if ymax != ymin else 1
                                ax.set_ylim(ymin - padding, ymax + padding)
                
                self.eeg_canvas.draw_idle()
            
            # Update band powers plot with event markers
            if all(len(self.band_power_buffer[key]) > 0 for key in ['alpha', 'beta', 'theta']):
                x = np.arange(len(self.band_power_buffer['alpha']))
                
                self.alpha_line.set_xdata(x)
                self.alpha_line.set_ydata(list(self.band_power_buffer['alpha']))
                
                self.beta_line.set_xdata(x)
                self.beta_line.set_ydata(list(self.band_power_buffer['beta']))
                
                self.theta_line.set_xdata(x)
                self.theta_line.set_ydata(list(self.band_power_buffer['theta']))
                
                # Add baseline lines if calibrated
                if self.baseline_metrics:
                    for band in ['alpha', 'beta', 'theta']:
                        if hasattr(self, 'baseline_lines') and self.baseline_lines.get(band) is not None:
                            try:
                                self.baseline_lines[band].remove()
                            except:
                                pass
                    
                    if not hasattr(self, 'baseline_lines'):
                        self.baseline_lines = {}
                    
                    colors = {'alpha': 'blue', 'beta': 'red', 'theta': 'green'}
                    for band, value in self.baseline_metrics.items():
                        if band in colors:
                            line = self.band_ax.axhline(value, color=colors[band], alpha=0.7, linestyle=':', linewidth=2)
                            self.baseline_lines[band] = line
                
                # Add event markers to band power plot
                for event in self.plot_events[-5:]:  # Show last 5 events
                    if 0 <= event['time'] * 2 <= len(x):  # Rough time scaling
                        event_x = event['time'] * 2  # Approximate scaling
                        line = self.band_ax.axvline(event_x, color='green', linestyle='-.', alpha=0.7)
                        self.band_ax.text(event_x, self.band_ax.get_ylim()[1]*0.9, event['label'][:8], 
                                         rotation=90, ha='right', va='top', fontsize=7, color='green')
                
                if len(x) > 0:
                    self.band_ax.set_xlim(0, len(x))
                    
                    all_values = (list(self.band_power_buffer['alpha']) + 
                                 list(self.band_power_buffer['beta']) + 
                                 list(self.band_power_buffer['theta']))
                    if all_values:
                        ymin, ymax = min(all_values), max(all_values)
                        padding = (ymax - ymin) * 0.1 if ymax != ymin else 1
                        self.band_ax.set_ylim(max(0, ymin - padding), ymax + padding)
                
                self.band_canvas.draw_idle()
                
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def clear_event_markers(self):
        """Clear previous event markers from plots"""
        # This is a simplified version - you might need to track markers more carefully
        pass
    
    def update_prediction_display(self):
        """Enhanced prediction display with stability info"""
        pred = self.current_prediction
        
        # Update state text with stability indicator
        confidence_indicator = "●" if pred['confidence'] > 0.8 else "◐" if pred['confidence'] > 0.6 else "○"
        self.prediction_state_var.set(f"{pred['state']} {confidence_indicator}")
        
        # Update details with more info
        confidence_pct = pred['confidence'] * 100
        stability = "Stable" if self.stable_prediction_count >= self.min_stable_count else f"Stabilizing ({self.stable_prediction_count}/{self.min_stable_count})"
        
        self.prediction_details_var.set(f"Level: {pred['level']} | Confidence: {confidence_pct:.1f}% | {stability} | Eyes: {self.eyes_state}")
        
        # Update progress bar
        progress_value = pred['smooth_value'] * 100
        self.prediction_progress['value'] = progress_value
        
        # Set color based on confidence
        if pred['confidence'] > 0.8:
            color = "green" if pred['level'] > 0 else "red" if pred['level'] < 0 else "gray"
        else:
            color = "orange"  # Low confidence
        
        self.prediction_state_label.config(foreground=color)
    
    def update_eyes_display(self):
        """Update the eyes state display"""
        self.eyes_state_var.set(f"Eyes: {self.eyes_state}")
        
        # Set color based on eyes state
        if self.eyes_state == "Closed":
            self.eyes_state_label.config(foreground="red")
        elif self.eyes_state == "Open":
            self.eyes_state_label.config(foreground="green")
        elif self.eyes_state == "Closing":
            self.eyes_state_label.config(foreground="orange")
        else:
            self.eyes_state_label.config(foreground="gray")
    
    def update_baseline_comparison(self, current_metrics):
        """Update the baseline comparison display"""
        if not self.baseline_metrics:
            return
        
        alpha_pct = ((current_metrics['alpha'] / self.baseline_metrics['alpha']) - 1) * 100 if self.baseline_metrics['alpha'] > 0 else 0
        beta_pct = ((current_metrics['beta'] / self.baseline_metrics['beta']) - 1) * 100 if self.baseline_metrics['beta'] > 0 else 0
        theta_pct = ((current_metrics['theta'] / self.baseline_metrics['theta']) - 1) * 100 if self.baseline_metrics['theta'] > 0 else 0
        
        self.alpha_comparison_var.set(f"α: {alpha_pct:+.0f}%")
        self.beta_comparison_var.set(f"β: {beta_pct:+.0f}%")
        self.theta_comparison_var.set(f"θ: {theta_pct:+.0f}%")
    
    def update_quality_display(self, quality):
        """Update the signal quality display"""
        quality_text = f"Quality: {quality.quality_level.title()} ({quality.overall_score:.2f})"
        self.quality_var.set(quality_text)
        
        if quality.quality_level == "excellent":
            self.quality_label.config(foreground="green")
        elif quality.quality_level == "good":
            self.quality_label.config(foreground="darkgreen")
        elif quality.quality_level == "fair":
            self.quality_label.config(foreground="orange")
        else:
            self.quality_label.config(foreground="red")
    
    def update_plots_scheduler(self):
        """Schedule plot updates"""
        self.update_plots()
        self.master.after(PLOT_UPDATE_INTERVAL_MS, self.update_plots_scheduler)
    
    
    def start_calibration(self):
        """Start the enhanced calibration process"""
        self.lsl_inlet = self.lsl_inlets['EEG']
        if self.lsl_inlet is None:
            messagebox.showwarning("Not Connected", "Please connect to Muse first")
            return
        
        if self.is_calibrating:
            messagebox.showinfo("Already Calibrating", "Calibration is already in progress")
            return
        
        self.session_type = self.session_type_var.get()
        
        self.is_calibrating = True
        self.calibrate_button.config(text="Calibrating...", state=tk.DISABLED)
        
        # Option 1: Update the prediction state label to show calibration status
        self.prediction_state_var.set("Calibrating...")
        self.prediction_details_var.set(f"Collecting baseline for {CALIBRATION_DURATION_SECONDS} seconds...")
        
        # Option 2: Or update one of the existing status displays
        # self.current_state_var.set(f"Calibrating... ({CALIBRATION_DURATION_SECONDS}s)")
        
        threading.Thread(target=self.perform_calibration, daemon=True).start()

    def perform_calibration(self):
        """Enhanced calibration with metadata collection"""
        # Reset state
        self.baseline_metrics = None
        self.signal_quality_validator.reset()
        self.session_data = {
            'eeg_raw': [],
            'accelerometer_data': [],
            'gyroscope_data': [],
            'ppg_data': [],
            'timestamps': {
                'eeg': [],
                'accelerometer': [],
                'gyroscope': [],
                'ppg': []
            },
            'band_powers': [],
            'predictions': [],
            'state_changes': [],
            'events': [],
            'eyes_states': [],
            'session_metadata': {
                'session_type': self.session_type,
                'calibration_start': datetime.datetime.now().isoformat(),
                'sampling_rate': self.sampling_rates['EEG'],  # Use the correct sampling rate
                'filter_params': {
                    'lowcut': LOWCUT,
                    'highcut': HIGHCUT,
                    'order': FILTER_ORDER
                },
                'detection_params': {
                    'eyes_closed_threshold': EYES_CLOSED_ALPHA_THRESHOLD,
                    'eyes_closed_min_duration': EYES_CLOSED_MIN_DURATION
                }
            }
        }
        
        # Clear buffers
        for buf in self.eeg_data_buffer:
            buf.clear()
        for buf in self.band_power_buffer.values():
            buf.clear()
        self.time_buffer.clear()
        
        # Collect calibration data
        calibration_start_time = time.time()
        calibration_metrics_list = []
        
        while time.time() - calibration_start_time < CALIBRATION_DURATION_SECONDS and self.is_calibrating:
            progress = (time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS
            self.master.after(0, lambda p=progress: self.update_calibration_progress(p))
            
            if all(len(buf) >= int(self.sampling_rates['EEG'] * 2) for buf in self.eeg_data_buffer):
                data_for_analysis = np.array([list(buf)[-int(self.sampling_rates['EEG'] * 2):] for buf in self.eeg_data_buffer])
                metrics = self.calculate_band_powers(data_for_analysis)
                
                if metrics:
                    calibration_metrics_list.append(metrics)
            
            time.sleep(0.1)
        
        # Calculate baseline
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
            self.session_start_time = time.time()
            
            # Store calibration results in metadata
            self.session_data['session_metadata']['calibration_end'] = datetime.datetime.now().isoformat()
            self.session_data['session_metadata']['baseline_metrics'] = self.baseline_metrics
            self.session_data['session_metadata']['calibration_samples'] = len(calibration_metrics_list)
            
            # Update UI to show calibration complete
            self.master.after(0, lambda: self.prediction_state_var.set("Ready"))
            self.master.after(0, lambda: self.prediction_details_var.set(f"Calibration complete - {self.session_type} session active"))
            self.master.after(0, lambda: self.current_state_var.set("State: Ready"))
            self.master.after(0, lambda: self.calibrate_button.config(text="Recalibrate", state=tk.NORMAL))
            self.master.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            
            print("\n--- Enhanced Calibration Complete ---")
            for key, val in self.baseline_metrics.items():
                print(f"Baseline {key.replace('_', ' ').title()}: {val:.2f}")
            print(f"Eyes detection threshold: {EYES_CLOSED_ALPHA_THRESHOLD}x alpha baseline")
        else:
            # Calibration failed
            self.master.after(0, lambda: self.prediction_state_var.set("Calibration Failed"))
            self.master.after(0, lambda: self.prediction_details_var.set("No valid data collected - try again"))
            self.master.after(0, lambda: self.calibrate_button.config(text="Start Calibration", state=tk.NORMAL))
        
        self.is_calibrating = False

    def update_calibration_progress(self, progress):
        """Update the calibration progress display"""
        time_remaining = CALIBRATION_DURATION_SECONDS * (1 - progress)
        self.prediction_details_var.set(f"Calibrating... {progress*100:.0f}% complete ({time_remaining:.1f}s remaining)")
        
        # Update the progress bar if it exists
        if hasattr(self, 'prediction_progress'):
            self.prediction_progress['value'] = progress * 100
    
    def update_user_state(self, new_state):
        """Update the current user state"""
        if new_state != self.current_user_state:
            timestamp = time.time()
            self.user_state_changes.append({
                "time": timestamp,
                "from_state": self.current_user_state,
                "to_state": new_state
            })
            self.current_user_state = new_state
            
            self.current_state_var.set(f"State: {self.current_user_state}")
            
            if self.is_calibrated:
                self.session_data['state_changes'].append({
                    "time": timestamp,
                    "from_state": self.current_user_state,
                    "to_state": new_state
                })
            
            if self.session_start_time:
                time_rel = timestamp - self.session_start_time
                print(f"\n>>> State Change @ {time_rel:.1f}s: {new_state}")
            else:
                print(f"\n>>> State Change: {new_state}")
    
    def add_event_marker(self, event_label):
        """Add an event marker"""
        timestamp = time.time()
        self.user_events.append({
            "time": timestamp,
            "label": event_label
        })
        
        if self.is_calibrated:
            self.session_data['events'].append({
                "time": timestamp,
                "label": event_label
            })
        
        if self.session_start_time:
            time_rel = timestamp - self.session_start_time
            print(f"\n>>> Event @ {time_rel:.1f}s: {event_label}")
        else:
            print(f"\n>>> Event: {event_label}")
    
    def generate_session_plots(self, save_path_base):
        """Generate comprehensive session analysis plots with level legend"""
        if not self.is_calibrated or not self.session_data['predictions']:
            return []
        
        saved_plots = []
        
        # Define level-to-state mappings based on session type
        if self.session_type == SESSION_TYPE_RELAX:
            level_legend = {
                3: "Deeply Relaxed",
                2: "Relaxed", 
                1: "Slightly Relaxed",
                0: "Neutral",
                -1: "Slightly Tense",
                -2: "Tense",
                -3: "Very Tense"
            }
            positive_color = 'lightblue'
            negative_color = 'lightcoral'
        else:  # FOCUS
            level_legend = {
                3: "Highly Focused",
                2: "Focused", 
                1: "Slightly Focused",
                0: "Neutral",
                -1: "Slightly Distracted",
                -2: "Distracted",
                -3: "Very Distracted"
            }
            positive_color = 'lightgreen'
            negative_color = 'lightyellow'
        
        # Extract data for plotting
        timestamps = [p['timestamp'] for p in self.session_data['predictions']]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_session_time = end_time - start_time
        rel_times = [(t - start_time) for t in timestamps]
        
        prediction_levels = [p['prediction']['level'] for p in self.session_data['predictions']]
        prediction_states = [p['prediction']['state'] for p in self.session_data['predictions']]
        confidence_scores = [p['prediction']['confidence'] for p in self.session_data['predictions']]
        
        # Extract band powers
        band_times = [bp['timestamp'] - start_time for bp in self.session_data['band_powers']]
        alphas = [bp['alpha'] for bp in self.session_data['band_powers']]
        betas = [bp['beta'] for bp in self.session_data['band_powers']]
        thetas = [bp['theta'] for bp in self.session_data['band_powers']]
        
        # Plot 1: Complete session overview with enhanced prediction plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        # Plot 1a: Band Powers with correlation explanation
        axes[0].plot(band_times, alphas, 'b-', label='Alpha (8-13 Hz)', linewidth=2)
        axes[0].plot(band_times, betas, 'r-', label='Beta (13-30 Hz)', linewidth=2)
        axes[0].plot(band_times, thetas, 'g-', label='Theta (4-8 Hz)', linewidth=2)
        
        if self.baseline_metrics:
            axes[0].axhline(self.baseline_metrics['alpha'], color='blue', linestyle='--', alpha=0.7, label='Alpha baseline')
            axes[0].axhline(self.baseline_metrics['beta'], color='red', linestyle='--', alpha=0.7, label='Beta baseline')
            axes[0].axhline(self.baseline_metrics['theta'], color='green', linestyle='--', alpha=0.7, label='Theta baseline')
        
        axes[0].set_ylabel('Band Power (μV²)')
        axes[0].set_title(f'{self.session_type} Session Analysis - EEG Band Powers')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 1b: Enhanced Prediction Levels with background coloring and legend
        # Create background coloring for different level ranges
        for level, state_name in level_legend.items():
            if level > 0:
                axes[1].axhspan(level-0.4, level+0.4, alpha=0.1, color=positive_color)
            elif level < 0:
                axes[1].axhspan(level-0.4, level+0.4, alpha=0.1, color=negative_color)
            else:  # level == 0
                axes[1].axhspan(level-0.4, level+0.4, alpha=0.1, color='lightgray')
        
        # Plot the prediction line
        axes[1].plot(rel_times, prediction_levels, 'k-', linewidth=2, label='AI Prediction Level')
        axes[1].fill_between(rel_times, prediction_levels, alpha=0.3, color='gray')
        
        # Add level labels on the right side
        for level, state_name in level_legend.items():
            if level >= min(prediction_levels) - 0.5 and level <= max(prediction_levels) + 0.5:
                axes[1].text(max(rel_times) * 1.02, level, f"{level}: {state_name}", 
                            fontsize=9, va='center', ha='left')
        
        axes[1].set_ylabel('Prediction Level')
        axes[1].set_title(f'Mental State Predictions - {self.session_type} Session')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-3.5, 3.5)
        axes[1].legend(loc='upper left')
        
        # Add level reference lines
        for level in level_legend.keys():
            axes[1].axhline(level, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
        
        # Plot 1c: Confidence Scores
        axes[2].plot(rel_times, confidence_scores, 'orange', linewidth=2, label='Prediction Confidence')
        axes[2].axhline(np.mean(confidence_scores), color='orange', linestyle='--', alpha=0.7, 
                    label=f'Mean: {np.mean(confidence_scores):.2f}')
        axes[2].set_ylabel('Confidence Score')
        axes[2].set_title('AI Prediction Confidence (Higher = More Certain)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        # Add confidence interpretation
        axes[2].axhspan(0.8, 1.0, alpha=0.1, color='green')
        axes[2].axhspan(0.6, 0.8, alpha=0.1, color='yellow')
        axes[2].axhspan(0.0, 0.6, alpha=0.1, color='red')
        axes[2].text(max(rel_times) * 0.02, 0.9, 'High Confidence', fontsize=8, alpha=0.7)
        axes[2].text(max(rel_times) * 0.02, 0.7, 'Medium Confidence', fontsize=8, alpha=0.7)
        axes[2].text(max(rel_times) * 0.02, 0.3, 'Low Confidence', fontsize=8, alpha=0.7)
        
        # Plot 1d: Eyes State Timeline
        if self.session_data['eyes_states']:
            eyes_times = [es['timestamp'] - start_time for es in self.session_data['eyes_states']]
            eyes_states = [1 if es['state'] == 'Closed' else 0 for es in self.session_data['eyes_states']]
            axes[3].plot(eyes_times, eyes_states, 'purple', linewidth=2, marker='o', markersize=4, label='Eyes State')
            axes[3].set_ylabel('Eyes State')
            axes[3].set_title('Eyes Open/Closed Detection (Alpha waves increase when eyes close)')
            axes[3].set_ylim(-0.1, 1.1)
            axes[3].set_yticks([0, 1])
            axes[3].set_yticklabels(['Open', 'Closed'])
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, 'No eyes state data collected', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Eyes Open/Closed Detection (No Data)')
        
        axes[3].grid(True, alpha=0.3)
        
        # Add user state change markers to all plots
        if self.session_data['state_changes']:
            for change in self.session_data['state_changes']:
                change_time = change['time'] - start_time
                for i, ax in enumerate(axes):
                    ax.axvline(change_time, color='red', linestyle=':', alpha=0.7, linewidth=2)
                    # Only add text label to the first plot to avoid clutter
                    if i == 0:
                        ax.text(change_time, ax.get_ylim()[1]*0.9, f"User: {change['to_state']}", 
                            rotation=90, ha='right', va='top', fontsize=8, color='red', weight='bold')
        
        # Add event markers
        if self.session_data['events']:
            for event in self.session_data['events']:
                event_time = event['time'] - start_time
                for i, ax in enumerate(axes):
                    ax.axvline(event_time, color='green', linestyle='-.', alpha=0.7)
                    # Only add text label to the bottom plot
                    if i == len(axes) - 1:
                        ax.text(event_time, ax.get_ylim()[0]*0.9, event['label'], 
                            rotation=90, ha='left', va='bottom', fontsize=7, color='green')
        
        axes[3].set_xlabel('Time (seconds)')
        plt.tight_layout()
        
        plot1_path = f"{save_path_base}_session_overview.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot1_path)
        plt.close()
        
        # Plot 2: Enhanced Analysis with correlation explanation
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 2a: Time-based Predicted State Distribution
        from collections import defaultdict
        prediction_time_distribution = defaultdict(float)
        for i in range(len(prediction_states) - 1):
            current_state = prediction_states[i]
            current_time = timestamps[i]
            next_time = timestamps[i + 1]
            duration = next_time - current_time
            prediction_time_distribution[current_state] += duration
        
        if prediction_states:
            last_state = prediction_states[-1]
            last_duration = end_time - timestamps[-1]
            prediction_time_distribution[last_state] += last_duration
        
        if prediction_time_distribution:
            total_pred_time = sum(prediction_time_distribution.values())
            pred_percentages = {state: (duration/total_pred_time)*100 for state, duration in prediction_time_distribution.items()}
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(pred_percentages)))
            axes[0,0].pie(pred_percentages.values(), labels=[f"{state}\n({pct:.1f}%)" for state, pct in pred_percentages.items()], 
                        autopct='', colors=colors, startangle=90)
            axes[0,0].set_title('AI Predicted State Distribution\n(Time-based)')
        else:
            axes[0,0].text(0.5, 0.5, 'No prediction data', ha='center', va='center')
            axes[0,0].set_title('AI Predicted States (No Data)')
        
        # Plot 2b: Time-based User Reported State Distribution
        if self.session_data['state_changes']:
            user_time_distribution = defaultdict(float)
            
            for i in range(len(self.session_data['state_changes'])):
                current_change = self.session_data['state_changes'][i]
                current_state = current_change['to_state']
                current_time = current_change['time']
                
                if i < len(self.session_data['state_changes']) - 1:
                    next_time = self.session_data['state_changes'][i + 1]['time']
                else:
                    next_time = end_time
                
                duration = next_time - current_time
                user_time_distribution[current_state] += duration
            
            if user_time_distribution:
                total_user_time = sum(user_time_distribution.values())
                user_percentages = {state: (duration/total_user_time)*100 for state, duration in user_time_distribution.items()}
                
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(user_percentages)))
                axes[0,1].pie(user_percentages.values(), labels=[f"{state}\n({pct:.1f}%)" for state, pct in user_percentages.items()], 
                            autopct='', colors=colors, startangle=90)
                axes[0,1].set_title('Your Reported State Distribution\n(Time-based)')
            else:
                axes[0,1].text(0.5, 0.5, 'No user state data', ha='center', va='center')
                axes[0,1].set_title('Your Reported States (No Data)')
        else:
            axes[0,1].text(0.5, 0.5, 'No user state changes recorded', ha='center', va='center')
            axes[0,1].set_title('Your Reported States (No Data)')
        
        # Plot 2c: Enhanced Band Power Correlations with explanation
        import pandas as pd
        df_bands = pd.DataFrame({
            'Alpha': alphas,
            'Beta': betas,
            'Theta': thetas
        })
        corr_matrix = df_bands.corr()
        
        im = axes[1,0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1,0].set_xticks(range(len(corr_matrix.columns)))
        axes[1,0].set_yticks(range(len(corr_matrix.columns)))
        axes[1,0].set_xticklabels(corr_matrix.columns)
        axes[1,0].set_yticklabels(corr_matrix.columns)
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.6 else 'black'
                axes[1,0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', color=color, weight='bold')
        
        plt.colorbar(im, ax=axes[1,0], label='Correlation Coefficient')
        axes[1,0].set_title('EEG Band Power Correlations\n(How brain waves relate to each other)')
        
        # Add interpretation text
        interpretation = """
    Interpretation Guide:
    • High positive correlation (>0.7): Bands move together
    • Low correlation (~0.0): Independent activity  
    • High negative correlation (<-0.7): Opposite patterns
    • Very high correlations (>0.9) may indicate artifacts
    """
        axes[1,0].text(1.1, 0.5, interpretation, transform=axes[1,0].transAxes, 
                    fontsize=8, va='center', ha='left', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # Plot 2d: Eyes Closed Analysis
        if self.session_data['eyes_states']:
            alpha_during_closed = []
            alpha_during_open = []
            
            for i, bp in enumerate(self.session_data['band_powers']):
                bp_time = bp['timestamp'] - start_time
                
                closest_eyes_state = None
                min_time_diff = float('inf')
                for es in self.session_data['eyes_states']:
                    time_diff = abs((es['timestamp'] - start_time) - bp_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_eyes_state = es['state']
                
                if closest_eyes_state == 'Closed':
                    alpha_during_closed.append(bp['alpha'])
                elif closest_eyes_state == 'Open':
                    alpha_during_open.append(bp['alpha'])
            
            if alpha_during_closed and alpha_during_open:
                box_data = [alpha_during_open, alpha_during_closed]
                bp = axes[1,1].boxplot(box_data, labels=['Eyes Open', 'Eyes Closed'], patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                axes[1,1].set_ylabel('Alpha Power (μV²)')
                axes[1,1].set_title('Alpha Power: Eyes Open vs Closed\n(Alpha increases when eyes close)')
                
                mean_open = np.mean(alpha_during_open)
                mean_closed = np.mean(alpha_during_closed)
                ratio = mean_closed / mean_open if mean_open > 0 else 0
                
                stats_text = f"""Alpha Ratio: {ratio:.2f}x
    Open: {mean_open:.2f} μV²
    Closed: {mean_closed:.2f} μV²

    Normal ratio: 2-10x
    Your ratio: {'Normal' if 2 <= ratio <= 10 else 'Unusual'}"""
                
                axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                            fontsize=9, va='top', ha='left',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            else:
                axes[1,1].text(0.5, 0.5, 'Insufficient eyes state data\nfor meaningful analysis', 
                            ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Alpha Power Analysis\n(Insufficient Data)')
        else:
            axes[1,1].text(0.5, 0.5, 'No eyes state data collected\nTry closing your eyes during recording', 
                        ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Eyes State Analysis\n(No Data)')
        
        plt.tight_layout()
        
        plot2_path = f"{save_path_base}_detailed_analysis.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot2_path)
        plt.close()
        
        # Plot 3: Enhanced Prediction vs User State Comparison with level legend
        if self.session_data['state_changes']:
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            # Create a twin axis for the level legend
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_ylabel('State Names', fontsize=12)
            
            # Plot prediction timeline
            ax.plot(rel_times, prediction_levels, 'b-', linewidth=3, label='AI Prediction Level', alpha=0.8)
            ax.fill_between(rel_times, prediction_levels, alpha=0.2, color='blue')
            
            # Add user state regions as background colors
            state_colors = {
                'Deeply Relaxed': 'lightblue', 'Relaxed': 'lightgreen', 'Slightly Relaxed': 'lightcyan',
                'Neutral': 'lightgray',
                'Slightly Tense': 'lightyellow', 'Tense': 'lightcoral',
                'Highly Focused': 'darkgreen', 'Focused': 'lightgreen', 'Slightly Focused': 'lightcyan',
                'Slightly Distracted': 'lightyellow', 'Distracted': 'lightcoral'
            }
            
            prev_change_time = 0
            for i, change in enumerate(self.session_data['state_changes']):
                change_time = change['time'] - start_time
                user_state = change['to_state']
                
                if i < len(self.session_data['state_changes']) - 1:
                    next_change_time = self.session_data['state_changes'][i + 1]['time'] - start_time
                else:
                    next_change_time = max(rel_times)
                
                color = state_colors.get(user_state, 'white')
                ax.axvspan(prev_change_time, next_change_time, alpha=0.3, color=color)
                
                # Add state change markers
                ax.axvline(change_time, color='red', linestyle=':', alpha=0.8, linewidth=2)
                ax.text(change_time, ax.get_ylim()[1]*0.95, user_state, 
                    rotation=90, ha='right', va='top', fontsize=10, color='red', weight='bold')
                
                prev_change_time = change_time
            
            # Add level reference lines and labels
            for level, state_name in level_legend.items():
                if level >= min(prediction_levels) - 0.5 and level <= max(prediction_levels) + 0.5:
                    ax.axhline(level, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                    # Add state name labels on the right axis
                    ax2.text(1.02, level, f"{level}: {state_name}", transform=ax.get_yaxis_transform(),
                            fontsize=10, va='center', ha='left', 
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('AI Prediction Level', fontsize=12)
            ax.set_title(f'AI Predictions vs Your Reported States - {self.session_type} Session\n' +
                        'Background colors show your reported states, blue line shows AI predictions', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            ax.set_ylim(-3.5, 3.5)
            
            # Hide the right axis ticks but keep the labels
            ax2.set_yticks([])
            
            plt.tight_layout()
            
            plot3_path = f"{save_path_base}_ai_vs_user_comparison.png"
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            saved_plots.append(plot3_path)
            plt.close()
        
        return saved_plots
    
    def save_session_data(self):
        """Enhanced save session data with proper multi-stream timestamp handling"""
        if not self.is_calibrated:
            messagebox.showwarning("Not Calibrated", "Please calibrate before saving data")
            return
        
        # Create filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_type}_{timestamp_str}"
        
        # Create target directory
        target_dir = os.path.join(os.getcwd(), "muse_test_data")
        os.makedirs(target_dir, exist_ok=True)
        
        # Full save path
        save_path = os.path.join(target_dir, filename)
        
        try:
            # Update session metadata with end time
            self.session_data['session_metadata']['session_end'] = datetime.datetime.now().isoformat()
            self.session_data['session_metadata']['total_duration'] = time.time() - self.session_start_time if self.session_start_time else 0
            
            # Calculate session statistics
            if self.session_data['predictions']:
                prediction_levels = [p['prediction']['level'] for p in self.session_data['predictions']]
                confidence_scores = [p['prediction']['confidence'] for p in self.session_data['predictions']]
                
                self.session_data['session_metadata']['statistics'] = {
                    'mean_prediction_level': np.mean(prediction_levels),
                    'std_prediction_level': np.std(prediction_levels),
                    'mean_confidence': np.mean(confidence_scores),
                    'num_predictions': len(self.session_data['predictions']),
                    'num_state_changes': len(self.session_data['state_changes']),
                    'num_events': len(self.session_data['events']),
                    'num_eyes_state_changes': len(self.session_data['eyes_states'])
                }
            
            # Prepare EEG data for saving
            if self.session_data['eeg_raw']:
                eeg_raw_combined = np.concatenate(self.session_data['eeg_raw'], axis=1)
                print(f"EEG data shape: {eeg_raw_combined.shape}")
            else:
                eeg_raw_combined = np.array([])
                print("Warning: No EEG data collected!")
            
            # Prepare timestamps - extract EEG timestamps specifically
            eeg_timestamps = np.array(self.session_data['timestamps']['eeg']) if self.session_data['timestamps']['eeg'] else np.array([])
            accelerometer_timestamps = np.array(self.session_data['timestamps']['accelerometer']) if self.session_data['timestamps']['accelerometer'] else np.array([])
            gyroscope_timestamps = np.array(self.session_data['timestamps']['gyroscope']) if self.session_data['timestamps']['gyroscope'] else np.array([])
            ppg_timestamps = np.array(self.session_data['timestamps']['ppg']) if self.session_data['timestamps']['ppg'] else np.array([])
            
            print(f"EEG timestamps: {len(eeg_timestamps)} samples")
            print(f"Accelerometer timestamps: {len(accelerometer_timestamps)} samples")
            print(f"Gyroscope timestamps: {len(gyroscope_timestamps)} samples")
            print(f"PPG timestamps: {len(ppg_timestamps)} samples")
            
            # Prepare accelerometer data
            if self.session_data['accelerometer_data']:
                accelerometer_combined = np.concatenate(self.session_data['accelerometer_data'], axis=1)
                print(f"Accelerometer data shape: {accelerometer_combined.shape}")
            else:
                accelerometer_combined = np.array([])
            
            # Prepare gyroscope data
            if self.session_data['gyroscope_data']:
                gyroscope_combined = np.concatenate(self.session_data['gyroscope_data'], axis=1)
                print(f"Gyroscope data shape: {gyroscope_combined.shape}")
            else:
                gyroscope_combined = np.array([])
            
            # Prepare PPG data
            if self.session_data['ppg_data']:
                ppg_combined = np.concatenate(self.session_data['ppg_data'], axis=1)
                print(f"PPG data shape: {ppg_combined.shape}")
            else:
                ppg_combined = np.array([])
            
            # Save to npz file with proper multi-stream data
            np.savez_compressed(
                save_path,
                # Raw sensor data with timestamps
                eeg_raw=eeg_raw_combined,
                eeg_timestamps=eeg_timestamps,
                accelerometer_data=accelerometer_combined,
                accelerometer_timestamps=accelerometer_timestamps,
                gyroscope_data=gyroscope_combined,
                gyroscope_timestamps=gyroscope_timestamps,
                ppg_data=ppg_combined,
                ppg_timestamps=ppg_timestamps,
                
                # Processed data
                band_powers=self.session_data['band_powers'],
                predictions=self.session_data['predictions'],
                state_changes=self.session_data['state_changes'],
                events=self.session_data['events'],
                eyes_states=self.session_data['eyes_states'],
                
                # Metadata
                baseline_metrics=self.baseline_metrics,
                session_type=self.session_type,
                sampling_rates=self.sampling_rates,  # Save all sampling rates
                channel_names=EEG_CHANNEL_NAMES,
                session_metadata=self.session_data['session_metadata']
            )
            
            # Generate and save plots
            save_path_base = Path(save_path).stem
            save_dir = Path(save_path).parent
            plot_base_path = save_dir / save_path_base
            
            saved_plots = self.generate_session_plots(str(plot_base_path))
            
            # Save session summary as JSON
            summary_path = f"{plot_base_path}_summary.json"
            with open(summary_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                metadata_copy = json.loads(json.dumps(self.session_data['session_metadata'], default=str))
                json.dump(metadata_copy, f, indent=2)
            
            # Show success message with details
            plot_count = len(saved_plots)
            
            # Data size summary
            data_summary = f"""
    EEG: {eeg_raw_combined.shape if eeg_raw_combined.size > 0 else 'No data'} 
    Accelerometer: {accelerometer_combined.shape if accelerometer_combined.size > 0 else 'No data'}
    Gyroscope: {gyroscope_combined.shape if gyroscope_combined.size > 0 else 'No data'}
    PPG: {ppg_combined.shape if ppg_combined.size > 0 else 'No data'}"""
            
            success_msg = f"""Session data saved successfully!

    Files created:
    • Data file: {Path(save_path).name}.npz
    • Analysis plots: {plot_count} files
    • Session summary: {Path(summary_path).name}

    Data Summary:{data_summary}

    Session Statistics:
    • Duration: {self.session_data['session_metadata'].get('total_duration', 0):.1f} seconds
    • Predictions: {self.session_data['session_metadata'].get('statistics', {}).get('num_predictions', 0)}
    • State changes: {self.session_data['session_metadata'].get('statistics', {}).get('num_state_changes', 0)}
    • Events: {self.session_data['session_metadata'].get('statistics', {}).get('num_events', 0)}
    • Eyes state changes: {self.session_data['session_metadata'].get('statistics', {}).get('num_eyes_state_changes', 0)}"""
            
            messagebox.showinfo("Save Complete", success_msg)
            print(f"Session data and {plot_count} plots saved to: {save_dir}")
            
            # Print file info for debugging
            print(f"\nSaved NPZ file contents:")
            print(f"- eeg_raw: {eeg_raw_combined.shape if eeg_raw_combined.size > 0 else 'Empty'}")
            print(f"- eeg_timestamps: {len(eeg_timestamps)} samples")
            print(f"- accelerometer_data: {accelerometer_combined.shape if accelerometer_combined.size > 0 else 'Empty'}")
            print(f"- Total file size: ~{os.path.getsize(save_path + '.npz') / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving data: {str(e)}")
            traceback.print_exc()

def main():
    """Main entry point"""
    root = tk.Tk()
    app = MultiStreamMuseEEGMonitorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()