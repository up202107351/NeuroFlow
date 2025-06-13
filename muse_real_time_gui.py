#!/usr/bin/env python3
"""
Compact Enhanced Muse Real-time EEG Visualization

This version has a more compact layout with all controls visible.
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

# --- Configuration ---
LSL_STREAM_TYPE = 'EEG'
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX_PULL = 128

EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10 for Muse
ACC_CHANNEL_INDICES = [9, 10, 11]   # X, Y, Z for accelerometer
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
DEFAULT_SAMPLING_RATE = 256.0

# Filter parameters
FILTER_ORDER = 2  # Reduced for better real-time performance
LOWCUT = 0.5
HIGHCUT = 30.0

# Time window for plots
PLOT_DURATION_S = 5.0
PLOT_UPDATE_INTERVAL_MS = 150

# Band power parameters
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Calibration and analysis
CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0

# Session types
SESSION_TYPE_RELAX = "RELAXATION"
SESSION_TYPE_FOCUS = "FOCUS"

# Data saving configuration
OUTPUT_DIR = "muse_test_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- State Labels ---
STATE_LABELS = {
    '1': 'Very Relaxed',
    '2': 'Relaxed', 
    '3': 'Neutral',
    '4': 'Focused',
    '5': 'Very Focused',
    '0': 'Unknown'
}

EVENT_LABELS = {
    's': 'START_ACTIVITY',
    'e': 'END_ACTIVITY', 
    'b': 'BLINK_ARTIFACT',
    'm': 'MOVEMENT_ARTIFACT',
    'c': 'EYES_CLOSED',
    'o': 'EYES_OPEN'
}

class CompactMuseEEGMonitorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Compact Muse EEG Monitor")
        master.geometry("1400x800")  # Slightly larger window
        
        # Configure grid weights for proper resizing
        master.grid_rowconfigure(2, weight=1)  # Plot frame should expand
        master.grid_columnconfigure(0, weight=1)
        
        # --- State Variables ---
        self.running = True
        self.lsl_inlet = None
        self.sampling_rate = DEFAULT_SAMPLING_RATE
        self.session_type = SESSION_TYPE_RELAX
        self.baseline_metrics = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.session_start_time = None
        
        # Data buffers
        self.eeg_data_buffer = [deque(maxlen=int(PLOT_DURATION_S * DEFAULT_SAMPLING_RATE)) for _ in range(NUM_EEG_CHANNELS)]
        self.band_power_buffer = {
            'alpha': deque(maxlen=100),
            'beta': deque(maxlen=100),
            'theta': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
        self.time_buffer = deque(maxlen=int(PLOT_DURATION_S * DEFAULT_SAMPLING_RATE))
        
        # Mental state prediction
        self.current_prediction = {
            'state': 'Unknown',
            'level': 0,
            'confidence': 0.0,
            'smooth_value': 0.5
        }
        self.prediction_history = deque(maxlen=50)
        
        # Annotation tracking
        self.current_user_state = "Unknown"
        self.user_state_changes = []
        self.user_events = []
        
        # Session data
        self.session_data = {
            'eeg_raw': [],
            'timestamps': [],
            'band_powers': [],
            'predictions': [],
            'state_changes': [],
            'events': []
        }
        
        # Signal quality validator
        self.signal_quality_validator = SignalQualityValidator()
        
        # Filtering parameters
        self.filter_b = None
        self.filter_a = None
        self.update_filter_coefficients()
        
        # Set up GUI layout
        self.setup_gui()
        
        # Initialize LSL connection thread
        self.lsl_thread = threading.Thread(target=self.lsl_connection_loop, daemon=True)
        self.lsl_thread.start()
        
        # Set up plot update scheduler
        self.update_plots_scheduler()
        
        # Set up keyboard monitoring
        self.setup_keyboard_monitoring()
        
        # Cleanup on close
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """Set up the compact GUI layout using grid"""
        
        # Top frame - Status and controls (FIXED HEIGHT)
        self.top_frame = ttk.Frame(self.master, padding=5)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.top_frame.grid_columnconfigure(1, weight=1)  # Middle column expands
        
        # Status on left
        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.status_label = ttk.Label(self.top_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.grid(row=0, column=0, sticky="w", padx=5)
        
        # Session type in middle
        session_frame = ttk.Frame(self.top_frame)
        session_frame.grid(row=0, column=1, padx=20)
        
        ttk.Label(session_frame, text="Session:").pack(side=tk.LEFT, padx=(0, 5))
        self.session_type_var = tk.StringVar(value=SESSION_TYPE_RELAX)
        session_combo = ttk.Combobox(session_frame, textvariable=self.session_type_var, state="readonly", width=12)
        session_combo['values'] = (SESSION_TYPE_RELAX, SESSION_TYPE_FOCUS)
        session_combo.pack(side=tk.LEFT)
        
        # Current state and quality on right
        right_frame = ttk.Frame(self.top_frame)
        right_frame.grid(row=0, column=2, sticky="e", padx=5)
        
        self.current_state_var = tk.StringVar(value=f"State: {self.current_user_state}")
        self.current_state_label = ttk.Label(right_frame, textvariable=self.current_state_var, font=("Arial", 10, "bold"))
        self.current_state_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quality_var = tk.StringVar(value="Quality: Unknown")
        self.quality_label = ttk.Label(right_frame, textvariable=self.quality_var, font=("Arial", 9))
        self.quality_label.pack(side=tk.LEFT)
        
        # Compact prediction frame (FIXED HEIGHT)
        self.prediction_frame = ttk.LabelFrame(self.master, text="Mental State", padding=5)
        self.prediction_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        
        # Single row layout for prediction info
        pred_info_frame = ttk.Frame(self.prediction_frame)
        pred_info_frame.grid(row=0, column=0, sticky="ew")
        pred_info_frame.grid_columnconfigure(1, weight=1)
        
        # Prediction state (left)
        self.prediction_state_var = tk.StringVar(value="Unknown")
        self.prediction_state_label = ttk.Label(pred_info_frame, textvariable=self.prediction_state_var, font=("Arial", 14, "bold"))
        self.prediction_state_label.grid(row=0, column=0, sticky="w", padx=(0, 20))
        
        # Progress bar and details (center)
        progress_frame = ttk.Frame(pred_info_frame)
        progress_frame.grid(row=0, column=1, sticky="ew", padx=10)
        progress_frame.grid_columnconfigure(0, weight=1)
        
        self.prediction_details_var = tk.StringVar(value="Level: 0 | Confidence: N/A")
        ttk.Label(progress_frame, textvariable=self.prediction_details_var, font=("Arial", 9)).grid(row=0, column=0, sticky="ew")
        
        self.prediction_progress = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.prediction_progress.grid(row=1, column=0, sticky="ew", pady=2)
        
        # Baseline comparison (right)
        baseline_frame = ttk.Frame(pred_info_frame)
        baseline_frame.grid(row=0, column=2, sticky="e")
        
        ttk.Label(baseline_frame, text="vs Baseline:", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky="e")
        
        self.alpha_comparison_var = tk.StringVar(value="α: N/A")
        self.beta_comparison_var = tk.StringVar(value="β: N/A")
        self.theta_comparison_var = tk.StringVar(value="θ: N/A")
        
        ttk.Label(baseline_frame, textvariable=self.alpha_comparison_var, font=("Arial", 8)).grid(row=1, column=0, sticky="e")
        ttk.Label(baseline_frame, textvariable=self.beta_comparison_var, font=("Arial", 8)).grid(row=2, column=0, sticky="e")
        ttk.Label(baseline_frame, textvariable=self.theta_comparison_var, font=("Arial", 8)).grid(row=3, column=0, sticky="e")
        
        # Plot frame (EXPANDABLE)
        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=2)
        self.plot_frame.grid_rowconfigure(0, weight=2)  # EEG plot gets more space
        self.plot_frame.grid_rowconfigure(1, weight=1)  # Band power plot gets less space
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        # Control buttons frame (FIXED HEIGHT AT BOTTOM)
        self.control_frame = ttk.Frame(self.master, padding=5)
        self.control_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
        
        # Main control buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        self.connect_button = ttk.Button(button_frame, text="Connect to Muse", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        self.calibrate_button = ttk.Button(button_frame, text="Start Calibration", command=self.start_calibration, state=tk.DISABLED)
        self.calibrate_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Session", command=self.save_session_data, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # State change buttons in a compact grid
        state_frame = ttk.LabelFrame(self.control_frame, text="Quick State Change", padding=3)
        state_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Create a grid of state buttons (2 rows)
        state_buttons = list(STATE_LABELS.items())
        for i, (key, label) in enumerate(state_buttons):
            row = i // 3
            col = i % 3
            btn = ttk.Button(
                state_frame, 
                text=f"{key}: {label}", 
                command=lambda l=label: self.update_user_state(l),
                width=18
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        
        # Configure state frame columns to expand equally
        for col in range(3):
            state_frame.grid_columnconfigure(col, weight=1)
        
        # Event buttons in a single row
        event_frame = ttk.LabelFrame(self.control_frame, text="Quick Events", padding=3)
        event_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        for i, (key, label) in enumerate(EVENT_LABELS.items()):
            btn = ttk.Button(
                event_frame, 
                text=f"{key}: {label.replace('_', ' ')}", 
                command=lambda l=label: self.add_event_marker(l),
                width=15
            )
            btn.grid(row=0, column=i, padx=1, pady=2, sticky="ew")
        
        # Configure event frame columns to expand equally
        for col in range(len(EVENT_LABELS)):
            event_frame.grid_columnconfigure(col, weight=1)
        
        # Initialize plot canvases
        self.setup_plots()
    
    def setup_plots(self):
        """Set up compact matplotlib plot canvases"""
        # EEG Plot (smaller figure size)
        self.eeg_fig = Figure(figsize=(12, 4), dpi=80)  # Reduced DPI and height
        self.eeg_axes = []
        
        for i in range(NUM_EEG_CHANNELS):
            ax = self.eeg_fig.add_subplot(NUM_EEG_CHANNELS, 1, i+1)
            ax.set_ylabel(CHANNEL_NAMES[i], fontsize=9)
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
        
        # Band Powers Plot (smaller figure size)
        self.band_fig = Figure(figsize=(12, 2.5), dpi=80)  # Reduced height
        self.band_ax = self.band_fig.add_subplot(111)
        self.band_ax.set_title("Band Powers with Baseline", fontsize=10)
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
            
            if key in STATE_LABELS:
                self.update_user_state(STATE_LABELS[key])
            elif key in EVENT_LABELS:
                self.add_event_marker(EVENT_LABELS[key])
        
        self.master.bind('<Key>', on_key_press)
        self.master.focus_set()  # Ensure window can receive key events
    
    def update_filter_coefficients(self):
        """Update filter coefficients based on current sampling rate"""
        nyq = 0.5 * self.sampling_rate
        low = LOWCUT / nyq
        high = HIGHCUT / nyq
        self.filter_b, self.filter_a = scipy.signal.butter(FILTER_ORDER, [low, high], btype='band', analog=False)
    
    def toggle_connection(self):
        """Connect to or disconnect from the Muse headband"""
        if self.lsl_inlet is None:
            # Connect
            self.status_var.set("Status: Connecting to Muse...")
            self.connect_button.config(text="Connecting...", state=tk.DISABLED)
            
            # Start connection in a separate thread
            threading.Thread(target=self.connect_to_lsl, daemon=True).start()
        else:
            # Disconnect
            self.disconnect_from_lsl()
            self.connect_button.config(text="Connect to Muse")
            self.calibrate_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
    
    def connect_to_lsl(self):
        """Connect to the LSL stream"""
        try:
            streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
            if not streams:
                self.master.after(0, lambda: self.status_var.set("Status: No EEG stream found"))
                self.master.after(0, lambda: self.connect_button.config(text="Connect to Muse", state=tk.NORMAL))
                return
                
            self.lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
            info = self.lsl_inlet.info()
            
            # Get stream info
            self.sampling_rate = info.nominal_srate() if info.nominal_srate() > 0 else DEFAULT_SAMPLING_RATE
            
            # Update filter coefficients
            self.update_filter_coefficients()
            
            # Update UI
            self.master.after(0, lambda: self.status_var.set(f"Status: Connected to {info.name()}"))
            self.master.after(0, lambda: self.connect_button.config(text="Disconnect", state=tk.NORMAL))
            self.master.after(0, lambda: self.calibrate_button.config(state=tk.NORMAL))
            
            # Start data processing
            self.running = True
            
        except Exception as e:
            self.master.after(0, lambda: self.status_var.set(f"Status: Connection error: {str(e)}"))
            self.master.after(0, lambda: self.connect_button.config(text="Connect to Muse", state=tk.NORMAL))
            traceback.print_exc()
    
    def disconnect_from_lsl(self):
        """Disconnect from the LSL stream"""
        if self.lsl_inlet:
            try:
                self.lsl_inlet.close_stream()
            except:
                pass
            self.lsl_inlet = None
            self.status_var.set("Status: Disconnected")
    
    def filter_eeg_data(self, eeg_data):
        """Apply bandpass filter to EEG data with improved error handling"""
        # Minimum number of samples needed for safe filtering
        min_samples = max(3 * FILTER_ORDER + 1, 10)
        
        # Check if we have enough samples
        if eeg_data.shape[1] < min_samples:
            return eeg_data
        
        # Apply filter
        eeg_filtered = np.zeros_like(eeg_data)
        for i in range(NUM_EEG_CHANNELS):
            try:
                # Calculate safe padding
                padlen = min(3 * FILTER_ORDER, eeg_data.shape[1] // 4)
                if padlen < 1:
                    padlen = None
                
                eeg_filtered[i] = scipy.signal.filtfilt(
                    self.filter_b, self.filter_a, 
                    eeg_data[i],
                    padlen=padlen
                )
            except Exception as e:
                # Fall back to original data if filtering fails
                eeg_filtered[i] = eeg_data[i]
        
        return eeg_filtered
    
    def calculate_band_powers(self, eeg_segment):
        """Calculate band powers from EEG segment"""
        if eeg_segment.shape[1] < int(self.sampling_rate):  # Need at least 1 second
            return None
        
        metrics_list = []
        for ch_idx in range(NUM_EEG_CHANNELS):
            ch_data = eeg_segment[ch_idx, :].copy()
            
            # Calculate PSD
            try:
                freqs, psd = scipy.signal.welch(ch_data, fs=self.sampling_rate, nperseg=min(int(self.sampling_rate*2), len(ch_data)))
                
                # Calculate band powers
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
        
        # Calculate weighted average
        avg_metrics = {
            'theta': np.mean([m['theta'] for m in metrics_list]),
            'alpha': np.mean([m['alpha'] for m in metrics_list]),
            'beta': np.mean([m['beta'] for m in metrics_list])
        }
        
        avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
        avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
        
        return avg_metrics
    
    def classify_mental_state(self, current_metrics):
        """Classify mental state based on current metrics"""
        if not self.baseline_metrics or not current_metrics:
            return {
                "state": "Unknown", "level": 0, "confidence": 0.0,
                "value": 0.5, "smooth_value": 0.5, "state_key": "unknown"
            }
        
        # Calculate ratios relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha'] if self.baseline_metrics['alpha'] > 0 else 1
        beta_ratio = current_metrics['beta'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 0 else 1
        theta_ratio = current_metrics['theta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 0 else 1
        
        # Simple classification logic
        if self.session_type == SESSION_TYPE_RELAX:
            # For relaxation, higher alpha is better
            if alpha_ratio > 1.3:
                state, level = "Very Relaxed", 4
            elif alpha_ratio > 1.1:
                state, level = "Relaxed", 2
            elif alpha_ratio < 0.8:
                state, level = "Tense", -2
            else:
                state, level = "Neutral", 0
        else:  # FOCUS
            # For focus, higher beta/theta ratio is better
            bt_ratio = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] if self.baseline_metrics['bt_ratio'] > 0 else 1
            if bt_ratio > 1.3:
                state, level = "Very Focused", 4
            elif bt_ratio > 1.1:
                state, level = "Focused", 2
            elif bt_ratio < 0.8:
                state, level = "Distracted", -2
            else:
                state, level = "Neutral", 0
        
        # Calculate confidence and smoothed value
        confidence = min(1.0, abs(level) / 4.0 + 0.5)
        value = (level + 4) / 8.0  # Normalize to 0-1
        
        # Simple smoothing
        if self.prediction_history:
            prev_value = self.prediction_history[-1]['smooth_value']
            smooth_value = 0.7 * prev_value + 0.3 * value
        else:
            smooth_value = value
        
        return {
            "state": state,
            "state_key": state.lower().replace(' ', '_'),
            "level": level,
            "confidence": confidence,
            "value": value,
            "smooth_value": smooth_value
        }
    
    # ... Continue with the rest of the methods (lsl_connection_loop, update_prediction_display, etc.)
    # but I'll include the key ones for layout fixes:
    
    def lsl_connection_loop(self):
        """Main thread for LSL data processing"""
        last_quality_check_time = 0
        last_band_power_calc_time = 0
        
        while self.running:
            # Skip if not connected
            if self.lsl_inlet is None:
                time.sleep(0.1)
                continue
            
            try:
                # Pull data from LSL
                chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
                
                if chunk and len(chunk) > 0:
                    # Convert to numpy array
                    chunk_np = np.array(chunk, dtype=np.float64).T
                    
                    # Extract EEG data
                    if chunk_np.shape[0] > max(EEG_CHANNEL_INDICES):
                        eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
                        
                        # Extract accelerometer data if available
                        if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
                            try:
                                acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                                if acc_chunk.shape[1] > 0:
                                    latest_acc_sample = acc_chunk[:, -1]
                                    self.signal_quality_validator.add_accelerometer_data(latest_acc_sample)
                            except Exception as e:
                                pass
                        
                        # Filter EEG data
                        eeg_filtered = self.filter_eeg_data(eeg_chunk)
                        
                        # Store in buffer for visualization
                        for i in range(NUM_EEG_CHANNELS):
                            self.eeg_data_buffer[i].extend(eeg_filtered[i, :])
                        
                        # Add timestamps
                        for ts in timestamps:
                            self.time_buffer.append(ts)
                        
                        # Calculate band powers periodically
                        current_time = time.time()
                        if current_time - last_band_power_calc_time > 0.5:
                            # Ensure we have enough data
                            if all(len(buf) >= int(self.sampling_rate * 2) for buf in self.eeg_data_buffer):
                                # Get the most recent 2 seconds
                                data_for_analysis = np.array([list(buf)[-int(self.sampling_rate * 2):] for buf in self.eeg_data_buffer])
                                
                                # Calculate band powers
                                metrics = self.calculate_band_powers(data_for_analysis)
                                
                                if metrics:
                                    # Add to band power buffers
                                    self.band_power_buffer['alpha'].append(metrics['alpha'])
                                    self.band_power_buffer['beta'].append(metrics['beta'])
                                    self.band_power_buffer['theta'].append(metrics['theta'])
                                    self.band_power_buffer['timestamps'].append(current_time)
                                    
                                    # Calculate mental state prediction if calibrated
                                    if self.is_calibrated:
                                        prediction = self.classify_mental_state(metrics)
                                        self.current_prediction = prediction
                                        self.prediction_history.append(prediction)
                                        
                                        # Update prediction display
                                        self.master.after(0, lambda: self.update_prediction_display())
                                        
                                        # Update baseline comparison
                                        self.master.after(0, lambda m=metrics: self.update_baseline_comparison(m))
                                        
                                        # Store in session data
                                        self.session_data['predictions'].append({
                                            'timestamp': current_time,
                                            'prediction': prediction,
                                            'metrics': metrics
                                        })
                                    
                                    # Store in session data
                                    self.session_data['band_powers'].append({
                                        'timestamp': current_time,
                                        'alpha': metrics['alpha'],
                                        'beta': metrics['beta'],
                                        'theta': metrics['theta']
                                    })
                            
                            last_band_power_calc_time = current_time
                        
                        # Check signal quality every second
                        if current_time - last_quality_check_time > 1.0:
                            # Add data to signal quality validator
                            if len(self.eeg_data_buffer[0]) > int(self.sampling_rate):
                                data_for_quality = np.array([list(buf)[-int(self.sampling_rate):] for buf in self.eeg_data_buffer])
                                self.signal_quality_validator.add_raw_eeg_data(data_for_quality)
                                
                                # Get quality assessment
                                quality = self.signal_quality_validator.assess_overall_quality()
                                self.master.after(0, lambda q=quality: self.update_quality_display(q))
                            
                            last_quality_check_time = current_time
                        
                        # Store raw data if recording
                        if self.is_calibrated and self.session_start_time:
                            self.session_data['eeg_raw'].append(eeg_chunk.copy())
                            self.session_data['timestamps'].extend(timestamps)
            
            except Exception as e:
                print(f"Error in LSL processing loop: {e}")
                time.sleep(0.1)
        
        print("LSL connection loop exiting")
    
    def update_prediction_display(self):
        """Update the mental state prediction display"""
        pred = self.current_prediction
        
        # Update state text
        self.prediction_state_var.set(pred['state'])
        
        # Update details
        confidence_pct = pred['confidence'] * 100
        self.prediction_details_var.set(f"Level: {pred['level']} | Confidence: {confidence_pct:.1f}%")
        
        # Update progress bar (0-100)
        progress_value = pred['smooth_value'] * 100
        self.prediction_progress['value'] = progress_value
        
        # Set color based on state
        if pred['level'] > 2:
            color = "green"
        elif pred['level'] > 0:
            color = "darkgreen"
        elif pred['level'] < -2:
            color = "red"
        elif pred['level'] < 0:
            color = "orange"
        else:
            color = "gray"
        
        self.prediction_state_label.config(foreground=color)
    
    def update_baseline_comparison(self, current_metrics):
        """Update the baseline comparison display"""
        if not self.baseline_metrics:
            return
        
        # Calculate percentage differences
        alpha_pct = ((current_metrics['alpha'] / self.baseline_metrics['alpha']) - 1) * 100 if self.baseline_metrics['alpha'] > 0 else 0
        beta_pct = ((current_metrics['beta'] / self.baseline_metrics['beta']) - 1) * 100 if self.baseline_metrics['beta'] > 0 else 0
        theta_pct = ((current_metrics['theta'] / self.baseline_metrics['theta']) - 1) * 100 if self.baseline_metrics['theta'] > 0 else 0
        
        # Update display with shorter labels
        self.alpha_comparison_var.set(f"α: {alpha_pct:+.0f}%")
        self.beta_comparison_var.set(f"β: {beta_pct:+.0f}%")
        self.theta_comparison_var.set(f"θ: {theta_pct:+.0f}%")
    
    def update_quality_display(self, quality):
        """Update the signal quality display"""
        quality_text = f"Quality: {quality.quality_level.title()} ({quality.overall_score:.2f})"
        
        self.quality_var.set(quality_text)
        
        # Set color based on quality
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
    
    def update_plots(self):
        """Update the matplotlib plots with proper error handling"""
        try:
            # Update EEG plot
            if len(self.time_buffer) > 0 and all(len(buf) > 0 for buf in self.eeg_data_buffer):
                # Create time axis relative to session start
                if self.session_start_time:
                    relative_times = np.array([t - self.session_start_time for t in self.time_buffer])
                else:
                    relative_times = np.array([t - self.time_buffer[0] for t in self.time_buffer])
                
                # Update each channel plot
                for i, line in enumerate(self.eeg_lines):
                    if i < len(self.eeg_data_buffer):
                        line.set_xdata(relative_times)
                        line.set_ydata(list(self.eeg_data_buffer[i]))
                
                # Adjust x-axis
                if len(relative_times) > 0:
                    for ax in self.eeg_axes:
                        ax.set_xlim(min(relative_times), max(relative_times))
                        
                        # Auto-scale y-axis
                        if len(self.eeg_data_buffer[0]) > 0:
                            all_data = []
                            for buf in self.eeg_data_buffer:
                                all_data.extend(buf)
                            if all_data:
                                ymin, ymax = min(all_data), max(all_data)
                                padding = (ymax - ymin) * 0.1 if ymax != ymin else 1
                                ax.set_ylim(ymin - padding, ymax + padding)
                
                # Update canvas
                self.eeg_canvas.draw_idle()
            
            # Update band powers plot
            if all(len(self.band_power_buffer[key]) > 0 for key in ['alpha', 'beta', 'theta']):
                # Create x-axis
                x = np.arange(len(self.band_power_buffer['alpha']))
                
                # Update lines
                self.alpha_line.set_xdata(x)
                self.alpha_line.set_ydata(list(self.band_power_buffer['alpha']))
                
                self.beta_line.set_xdata(x)
                self.beta_line.set_ydata(list(self.band_power_buffer['beta']))
                
                self.theta_line.set_xdata(x)
                self.theta_line.set_ydata(list(self.band_power_buffer['theta']))
                
                # Adjust axes
                if len(x) > 0:
                    self.band_ax.set_xlim(0, len(x))
                    
                    # Find min/max across all bands
                    all_values = (list(self.band_power_buffer['alpha']) + 
                                 list(self.band_power_buffer['beta']) + 
                                 list(self.band_power_buffer['theta']))
                    if all_values:
                        ymin, ymax = min(all_values), max(all_values)
                        padding = (ymax - ymin) * 0.1 if ymax != ymin else 1
                        self.band_ax.set_ylim(max(0, ymin - padding), ymax + padding)
                
                # Draw baseline lines if calibrated
                if self.baseline_metrics:
                    # Remove old baseline lines
                    for band in ['alpha', 'beta', 'theta']:
                        if self.baseline_lines[band] is not None:
                            try:
                                self.baseline_lines[band].remove()
                            except:
                                pass
                    
                    # Add new baseline lines
                    colors = {'alpha': 'blue', 'beta': 'red', 'theta': 'green'}
                    for band, value in self.baseline_metrics.items():
                        if band in colors:
                            line = self.band_ax.axhline(value, color=colors[band], alpha=0.7, linestyle=':', linewidth=2)
                            self.baseline_lines[band] = line
                
                # Update canvas
                self.band_canvas.draw_idle()
                
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def start_calibration(self):
        """Start the calibration process"""
        if self.lsl_inlet is None:
            messagebox.showwarning("Not Connected", "Please connect to Muse first")
            return
        
        if self.is_calibrating:
            messagebox.showinfo("Already Calibrating", "Calibration is already in progress")
            return
        
        # Get session type
        self.session_type = self.session_type_var.get()
        
        # Start calibration thread
        self.is_calibrating = True
        self.calibrate_button.config(text="Calibrating...", state=tk.DISABLED)
        self.status_var.set(f"Status: Calibrating for {CALIBRATION_DURATION_SECONDS} seconds...")
        
        threading.Thread(target=self.perform_calibration, daemon=True).start()
    
    def perform_calibration(self):
        """Perform the calibration process"""
        # Reset state
        self.baseline_metrics = None
        self.signal_quality_validator.reset()
        self.session_data = {
            'eeg_raw': [],
            'timestamps': [],
            'band_powers': [],
            'predictions': [],
            'state_changes': [],
            'events': []
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
            # Update progress
            progress = (time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS
            self.master.after(0, lambda p=progress: self.update_calibration_progress(p))
            
            # Check if we have enough data for band power calculation
            if all(len(buf) >= int(self.sampling_rate * 2) for buf in self.eeg_data_buffer):
                data_for_analysis = np.array([list(buf)[-int(self.sampling_rate * 2):] for buf in self.eeg_data_buffer])
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
            
            self.master.after(0, lambda: self.status_var.set(f"Status: Calibration complete - {self.session_type} session active"))
            self.master.after(0, lambda: self.calibrate_button.config(text="Recalibrate", state=tk.NORMAL))
            self.master.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            
            # Print baseline
            print("\n--- Calibration Complete ---")
            for key, val in self.baseline_metrics.items():
                print(f"Baseline {key.replace('_', ' ').title()}: {val:.2f}")
        else:
            self.master.after(0, lambda: self.status_var.set("Status: Calibration failed - No valid data"))
            self.master.after(0, lambda: self.calibrate_button.config(text="Start Calibration", state=tk.NORMAL))
        
        self.is_calibrating = False
    
    def update_calibration_progress(self, progress):
        """Update the calibration progress display"""
        self.status_var.set(f"Status: Calibrating... {progress*100:.0f}%")
    
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
            
            # Update UI
            self.current_state_var.set(f"State: {self.current_user_state}")
            
            # Update session data
            if self.is_calibrated:
                self.session_data['state_changes'].append({
                    "time": timestamp,
                    "from_state": self.current_user_state,
                    "to_state": new_state
                })
            
            # Print state change
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
        
        # Update session data
        if self.is_calibrated:
            self.session_data['events'].append({
                "time": timestamp,
                "label": event_label
            })
        
        # Print event
        if self.session_start_time:
            time_rel = timestamp - self.session_start_time
            print(f"\n>>> Event @ {time_rel:.1f}s: {event_label}")
        else:
            print(f"\n>>> Event: {event_label}")
    
    def save_session_data(self):
        """Save the current session data"""
        if not self.is_calibrated:
            messagebox.showwarning("Not Calibrated", "Please calibrate before saving data")
            return
        
        # Create filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_type}_{timestamp_str}"
        
        # Ask for save location
        target_dir = os.path.join(os.getcwd(), "muse_Test_data")
        os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Full save path
        save_path = os.path.join(target_dir, filename)
        
        if not save_path:
            return
        
        try:
            # Prepare data for saving
            # Combine raw EEG chunks
            if self.session_data['eeg_raw']:
                eeg_raw_combined = np.concatenate(self.session_data['eeg_raw'], axis=1)
            else:
                eeg_raw_combined = np.array([])
            
            # Save to npz file
            np.savez_compressed(
                save_path,
                eeg_raw=eeg_raw_combined,
                timestamps=np.array(self.session_data['timestamps']),
                band_powers=self.session_data['band_powers'],
                predictions=self.session_data['predictions'],
                state_changes=self.session_data['state_changes'],
                events=self.session_data['events'],
                baseline_metrics=self.baseline_metrics,
                session_type=self.session_type,
                sampling_rate=self.sampling_rate,
                channel_names=CHANNEL_NAMES
            )
            
            messagebox.showinfo("Save Complete", f"Session data saved to {save_path}")
            print(f"Session data saved to: {save_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving data: {str(e)}")
            traceback.print_exc()
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit? Unsaved data will be lost."):
            self.running = False
            
            # Disconnect from LSL
            if self.lsl_inlet:
                try:
                    self.lsl_inlet.close_stream()
                except:
                    pass
                self.lsl_inlet = None
            
            # Wait for LSL thread to finish
            if self.lsl_thread and self.lsl_thread.is_alive():
                self.lsl_thread.join(timeout=1.0)
            
            self.master.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()
    app = CompactMuseEEGMonitorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()