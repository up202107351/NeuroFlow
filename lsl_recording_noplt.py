#!/usr/bin/env python3
"""
Enhanced Muse EEG Monitor with Advanced Features

New features:
1. Automatic plot generation and export on save
2. Dynamic state buttons based on session type
3. Eyes open/closed detection
4. Comprehensive session reports
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
import matplotlib.dates as mdates

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
FILTER_ORDER = 2
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

# Eyes closed detection parameters
EYES_CLOSED_ALPHA_THRESHOLD = 1.5  # Multiplier of baseline alpha for eyes closed detection
EYES_CLOSED_MIN_DURATION = 2.0     # Minimum duration in seconds to confirm eyes closed

# --- Dynamic State Labels Based on Session Type ---
RELAXATION_STATES = {
    '1': 'Deeply Relaxed',
    '2': 'Relaxed', 
    '3': 'Slightly Relaxed',
    '4': 'Neutral',
    '5': 'Slightly Tense',
    '6': 'Tense'
}

FOCUS_STATES = {
    '1': 'Highly Focused',
    '2': 'Focused', 
    '3': 'Slightly Focused',
    '4': 'Neutral',
    '5': 'Slightly Distracted',
    '6': 'Distracted'
}

EVENT_LABELS = {
    's': 'START_ACTIVITY',
    'e': 'END_ACTIVITY', 
    'b': 'BLINK_ARTIFACT',
    'm': 'MOVEMENT_ARTIFACT',
    'c': 'EYES_CLOSED',
    'o': 'EYES_OPEN'
}

class EnhancedMuseEEGMonitorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced Muse EEG Monitor v2.0")
        master.geometry("1400x900")
        
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
        
        # Eyes closed detection
        self.eyes_state = "Unknown"  # "Open", "Closed", "Unknown"
        self.eyes_closed_start_time = None
        self.eyes_state_history = deque(maxlen=20)
        
        # Annotation tracking
        self.current_user_state = "Unknown"
        self.user_state_changes = []
        self.user_events = []
        
        # Session data with enhanced tracking
        self.session_data = {
            'eeg_raw': [],
            'timestamps': [],
            'band_powers': [],
            'predictions': [],
            'state_changes': [],
            'events': [],
            'eyes_states': [],  # New: track eyes open/closed
            'session_metadata': {}
        }
        
        # Signal quality validator
        self.signal_quality_validator = SignalQualityValidator()
        
        # Filtering parameters
        self.filter_b = None
        self.filter_a = None
        self.update_filter_coefficients()
        
        # Dynamic state buttons storage
        self.state_buttons = []
        
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
    
    def get_current_state_labels(self):
        """Get state labels based on current session type"""
        if self.session_type == SESSION_TYPE_RELAX:
            return RELAXATION_STATES
        else:
            return FOCUS_STATES
    
    def setup_gui(self):
        """Set up the enhanced GUI layout using grid"""
        
        # Top frame - Status and controls (FIXED HEIGHT)
        self.top_frame = ttk.Frame(self.master, padding=5)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.top_frame.grid_columnconfigure(1, weight=1)
        
        # Status on left
        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.status_label = ttk.Label(self.top_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.grid(row=0, column=0, sticky="w", padx=5)
        
        # Session type in middle with callback
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
        
        # Eyes state indicator
        self.eyes_state_var = tk.StringVar(value="Eyes: Unknown")
        self.eyes_state_label = ttk.Label(right_frame, textvariable=self.eyes_state_var, font=("Arial", 10))
        self.eyes_state_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quality_var = tk.StringVar(value="Quality: Unknown")
        self.quality_label = ttk.Label(right_frame, textvariable=self.quality_var, font=("Arial", 9))
        self.quality_label.pack(side=tk.LEFT)
        
        # Enhanced prediction frame with eyes state
        self.prediction_frame = ttk.LabelFrame(self.master, text="Mental State & Eyes Detection", padding=5)
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
        
        self.prediction_details_var = tk.StringVar(value="Level: 0 | Confidence: N/A | Eyes: Unknown")
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
        self.plot_frame.grid_rowconfigure(0, weight=2)
        self.plot_frame.grid_rowconfigure(1, weight=1)
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
        
        self.save_button = ttk.Button(button_frame, text="Save Session + Plots", command=self.save_session_data, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Dynamic state change buttons frame
        self.state_frame = ttk.LabelFrame(self.control_frame, text="Report Current State", padding=3)
        self.state_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Event buttons in a single row
        event_frame = ttk.LabelFrame(self.control_frame, text="Mark Events", padding=3)
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
        
        # Create initial state buttons
        self.create_state_buttons()
    
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
    
    def detect_eyes_closed(self, current_metrics):
        """Detect if eyes are closed based on alpha power"""
        if not self.baseline_metrics or not current_metrics:
            return "Unknown"
        
        # Calculate alpha ratio relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha']
        
        current_time = time.time()
        
        # Eyes likely closed if alpha power is significantly elevated
        if alpha_ratio > EYES_CLOSED_ALPHA_THRESHOLD:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
            elif current_time - self.eyes_closed_start_time > EYES_CLOSED_MIN_DURATION:
                return "Closed"
            else:
                return "Closing"  # Transitional state
        else:
            self.eyes_closed_start_time = None
            return "Open"
        
        return "Unknown"
    
    def toggle_connection(self):
        """Connect to or disconnect from the Muse headband"""
        if self.lsl_inlet is None:
            self.status_var.set("Status: Connecting to Muse...")
            self.connect_button.config(text="Connecting...", state=tk.DISABLED)
            threading.Thread(target=self.connect_to_lsl, daemon=True).start()
        else:
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
            
            self.sampling_rate = info.nominal_srate() if info.nominal_srate() > 0 else DEFAULT_SAMPLING_RATE
            self.update_filter_coefficients()
            
            self.master.after(0, lambda: self.status_var.set(f"Status: Connected to {info.name()}"))
            self.master.after(0, lambda: self.connect_button.config(text="Disconnect", state=tk.NORMAL))
            self.master.after(0, lambda: self.calibrate_button.config(state=tk.NORMAL))
            
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
        """Enhanced classify mental state with session-specific logic"""
        if not self.baseline_metrics or not current_metrics:
            return {
                "state": "Unknown", "level": 0, "confidence": 0.0,
                "value": 0.5, "smooth_value": 0.5, "state_key": "unknown"
            }
        
        # Calculate ratios relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha'] if self.baseline_metrics['alpha'] > 0 else 1
        beta_ratio = current_metrics['beta'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 0 else 1
        theta_ratio = current_metrics['theta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 0 else 1
        
        # Session-specific classification that matches the user state labels
        if self.session_type == SESSION_TYPE_RELAX:
            # For relaxation: higher alpha is better, lower beta/theta is better
            if alpha_ratio > 1.4 and beta_ratio < 0.8:
                state, level = "Deeply Relaxed", 3
            elif alpha_ratio > 1.2:
                state, level = "Relaxed", 2
            elif alpha_ratio > 1.05:
                state, level = "Slightly Relaxed", 1
            elif alpha_ratio < 0.8 and beta_ratio > 1.2:
                state, level = "Tense", -2
            elif alpha_ratio < 0.9:
                state, level = "Slightly Tense", -1
            else:
                state, level = "Neutral", 0
        else:  # FOCUS
            # For focus: higher beta/theta ratio is better, sustained attention
            bt_ratio = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] if self.baseline_metrics['bt_ratio'] > 0 else 1
            if bt_ratio > 1.4 and beta_ratio > 1.2:
                state, level = "Highly Focused", 3
            elif bt_ratio > 1.2:
                state, level = "Focused", 2
            elif bt_ratio > 1.05:
                state, level = "Slightly Focused", 1
            elif bt_ratio < 0.8 or theta_ratio > 1.3:
                state, level = "Distracted", -2
            elif bt_ratio < 0.9:
                state, level = "Slightly Distracted", -1
            else:
                state, level = "Neutral", 0
        
        # Calculate confidence and smoothed value
        confidence = min(1.0, abs(level) / 3.0 + 0.5)
        value = (level + 3) / 6.0  # Normalize to 0-1
        
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
    
    def lsl_connection_loop(self):
        """Enhanced LSL processing loop with eyes detection"""
        last_quality_check_time = 0
        last_band_power_calc_time = 0
        
        while self.running:
            if self.lsl_inlet is None:
                time.sleep(0.1)
                continue
            
            try:
                chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
                
                if chunk and len(chunk) > 0:
                    chunk_np = np.array(chunk, dtype=np.float64).T
                    
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
                        
                        for ts in timestamps:
                            self.time_buffer.append(ts)
                        
                        # Calculate band powers and predictions periodically
                        current_time = time.time()
                        if current_time - last_band_power_calc_time > 0.5:
                            if all(len(buf) >= int(self.sampling_rate * 2) for buf in self.eeg_data_buffer):
                                data_for_analysis = np.array([list(buf)[-int(self.sampling_rate * 2):] for buf in self.eeg_data_buffer])
                                
                                metrics = self.calculate_band_powers(data_for_analysis)
                                
                                if metrics:
                                    # Add to band power buffers
                                    self.band_power_buffer['alpha'].append(metrics['alpha'])
                                    self.band_power_buffer['beta'].append(metrics['beta'])
                                    self.band_power_buffer['theta'].append(metrics['theta'])
                                    self.band_power_buffer['timestamps'].append(current_time)
                                    
                                    # Detect eyes state
                                    eyes_state = self.detect_eyes_closed(metrics)
                                    if eyes_state != self.eyes_state:
                                        self.eyes_state = eyes_state
                                        self.master.after(0, lambda: self.update_eyes_display())
                                        
                                        # Store eyes state change
                                        if self.is_calibrated:
                                            self.session_data['eyes_states'].append({
                                                'timestamp': current_time,
                                                'state': eyes_state
                                            })
                                    
                                    # Calculate mental state prediction if calibrated
                                    if self.is_calibrated:
                                        prediction = self.classify_mental_state(metrics)
                                        self.current_prediction = prediction
                                        self.prediction_history.append(prediction)
                                        
                                        # Update prediction display
                                        self.master.after(0, lambda: self.update_prediction_display())
                                        self.master.after(0, lambda m=metrics: self.update_baseline_comparison(m))
                                        
                                        # Store in session data
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
                        
                        # Check signal quality every second
                        if current_time - last_quality_check_time > 1.0:
                            if len(self.eeg_data_buffer[0]) > int(self.sampling_rate):
                                data_for_quality = np.array([list(buf)[-int(self.sampling_rate):] for buf in self.eeg_data_buffer])
                                self.signal_quality_validator.add_raw_eeg_data(data_for_quality)
                                
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
        """Update the enhanced prediction display with eyes state"""
        pred = self.current_prediction
        
        # Update state text
        self.prediction_state_var.set(pred['state'])
        
        # Update details with eyes state
        confidence_pct = pred['confidence'] * 100
        self.prediction_details_var.set(f"Level: {pred['level']} | Confidence: {confidence_pct:.1f}% | Eyes: {self.eyes_state}")
        
        # Update progress bar
        progress_value = pred['smooth_value'] * 100
        self.prediction_progress['value'] = progress_value
        
        # Set color based on state
        if pred['level'] > 1:
            color = "green"
        elif pred['level'] > 0:
            color = "darkgreen"
        elif pred['level'] < -1:
            color = "red"
        elif pred['level'] < 0:
            color = "orange"
        else:
            color = "gray"
        
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
    
    def update_plots(self):
        """Update the matplotlib plots with enhanced markers"""
        try:
            # Update EEG plot
            if len(self.time_buffer) > 0 and all(len(buf) > 0 for buf in self.eeg_data_buffer):
                if self.session_start_time:
                    relative_times = np.array([t - self.session_start_time for t in self.time_buffer])
                else:
                    relative_times = np.array([t - self.time_buffer[0] for t in self.time_buffer])
                
                for i, line in enumerate(self.eeg_lines):
                    if i < len(self.eeg_data_buffer):
                        line.set_xdata(relative_times)
                        line.set_ydata(list(self.eeg_data_buffer[i]))
                
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
            
            # Update band powers plot with enhanced markers
            if all(len(self.band_power_buffer[key]) > 0 for key in ['alpha', 'beta', 'theta']):
                x = np.arange(len(self.band_power_buffer['alpha']))
                
                self.alpha_line.set_xdata(x)
                self.alpha_line.set_ydata(list(self.band_power_buffer['alpha']))
                
                self.beta_line.set_xdata(x)
                self.beta_line.set_ydata(list(self.band_power_buffer['beta']))
                
                self.theta_line.set_xdata(x)
                self.theta_line.set_ydata(list(self.band_power_buffer['theta']))
                
                if len(x) > 0:
                    self.band_ax.set_xlim(0, len(x))
                    
                    all_values = (list(self.band_power_buffer['alpha']) + 
                                 list(self.band_power_buffer['beta']) + 
                                 list(self.band_power_buffer['theta']))
                    if all_values:
                        ymin, ymax = min(all_values), max(all_values)
                        padding = (ymax - ymin) * 0.1 if ymax != ymin else 1
                        self.band_ax.set_ylim(max(0, ymin - padding), ymax + padding)
                
                # Draw baseline lines if calibrated
                if self.baseline_metrics:
                    for band in ['alpha', 'beta', 'theta']:
                        if self.baseline_lines[band] is not None:
                            try:
                                self.baseline_lines[band].remove()
                            except:
                                pass
                    
                    colors = {'alpha': 'blue', 'beta': 'red', 'theta': 'green'}
                    for band, value in self.baseline_metrics.items():
                        if band in colors:
                            line = self.band_ax.axhline(value, color=colors[band], alpha=0.7, linestyle=':', linewidth=2)
                            self.baseline_lines[band] = line
                
                self.band_canvas.draw_idle()
                
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def start_calibration(self):
        """Start the enhanced calibration process"""
        if self.lsl_inlet is None:
            messagebox.showwarning("Not Connected", "Please connect to Muse first")
            return
        
        if self.is_calibrating:
            messagebox.showinfo("Already Calibrating", "Calibration is already in progress")
            return
        
        self.session_type = self.session_type_var.get()
        
        self.is_calibrating = True
        self.calibrate_button.config(text="Calibrating...", state=tk.DISABLED)
        self.status_var.set(f"Status: Calibrating for {CALIBRATION_DURATION_SECONDS} seconds...")
        
        threading.Thread(target=self.perform_calibration, daemon=True).start()
    
    def perform_calibration(self):
        """Enhanced calibration with metadata collection"""
        # Reset state
        self.baseline_metrics = None
        self.signal_quality_validator.reset()
        self.session_data = {
            'eeg_raw': [],
            'timestamps': [],
            'band_powers': [],
            'predictions': [],
            'state_changes': [],
            'events': [],
            'eyes_states': [],
            'session_metadata': {
                'session_type': self.session_type,
                'calibration_start': datetime.datetime.now().isoformat(),
                'sampling_rate': self.sampling_rate,
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
            
            # Store calibration results in metadata
            self.session_data['session_metadata']['calibration_end'] = datetime.datetime.now().isoformat()
            self.session_data['session_metadata']['baseline_metrics'] = self.baseline_metrics
            self.session_data['session_metadata']['calibration_samples'] = len(calibration_metrics_list)
            
            self.master.after(0, lambda: self.status_var.set(f"Status: Calibration complete - {self.session_type} session active"))
            self.master.after(0, lambda: self.calibrate_button.config(text="Recalibrate", state=tk.NORMAL))
            self.master.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            
            print("\n--- Enhanced Calibration Complete ---")
            for key, val in self.baseline_metrics.items():
                print(f"Baseline {key.replace('_', ' ').title()}: {val:.2f}")
            print(f"Eyes detection threshold: {EYES_CLOSED_ALPHA_THRESHOLD}x alpha baseline")
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
        """Generate comprehensive session analysis plots"""
        if not self.is_calibrated or not self.session_data['predictions']:
            return []
        
        saved_plots = []
        
        # Plot 1: Complete session overview
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        # Extract data for plotting
        timestamps = [p['timestamp'] for p in self.session_data['predictions']]
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        prediction_levels = [p['prediction']['level'] for p in self.session_data['predictions']]
        prediction_states = [p['prediction']['state'] for p in self.session_data['predictions']]
        confidence_scores = [p['prediction']['confidence'] for p in self.session_data['predictions']]
        
        # Extract band powers
        band_times = [bp['timestamp'] - start_time for bp in self.session_data['band_powers']]
        alphas = [bp['alpha'] for bp in self.session_data['band_powers']]
        betas = [bp['beta'] for bp in self.session_data['band_powers']]
        thetas = [bp['theta'] for bp in self.session_data['band_powers']]
        
        # Plot 1a: Band Powers
        axes[0].plot(band_times, alphas, 'b-', label='Alpha', linewidth=2)
        axes[0].plot(band_times, betas, 'r-', label='Beta', linewidth=2)
        axes[0].plot(band_times, thetas, 'g-', label='Theta', linewidth=2)
        
        # Add baseline lines
        if self.baseline_metrics:
            axes[0].axhline(self.baseline_metrics['alpha'], color='blue', linestyle='--', alpha=0.7, label='Alpha baseline')
            axes[0].axhline(self.baseline_metrics['beta'], color='red', linestyle='--', alpha=0.7, label='Beta baseline')
            axes[0].axhline(self.baseline_metrics['theta'], color='green', linestyle='--', alpha=0.7, label='Theta baseline')
        
        axes[0].set_ylabel('Band Power')
        axes[0].set_title(f'{self.session_type} Session Analysis - Band Powers')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 1b: Prediction Levels
        axes[1].plot(rel_times, prediction_levels, 'k-', linewidth=2, label='Predicted Level')
        axes[1].fill_between(rel_times, prediction_levels, alpha=0.3)
        axes[1].set_ylabel('Prediction Level')
        axes[1].set_title('Mental State Predictions')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-3.5, 3.5)
        
        # Plot 1c: Confidence Scores
        axes[2].plot(rel_times, confidence_scores, 'orange', linewidth=2, label='Confidence')
        axes[2].axhline(np.mean(confidence_scores), color='orange', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(confidence_scores):.2f}')
        axes[2].set_ylabel('Confidence')
        axes[2].set_title('Prediction Confidence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        # Plot 1d: Eyes State Timeline
        if self.session_data['eyes_states']:
            eyes_times = [es['timestamp'] - start_time for es in self.session_data['eyes_states']]
            eyes_states = [1 if es['state'] == 'Closed' else 0 for es in self.session_data['eyes_states']]
            axes[3].plot(eyes_times, eyes_states, 'purple', linewidth=2, marker='o', markersize=4)
            axes[3].set_ylabel('Eyes State')
            axes[3].set_title('Eyes Open/Closed Detection')
            axes[3].set_ylim(-0.1, 1.1)
            axes[3].set_yticks([0, 1])
            axes[3].set_yticklabels(['Open', 'Closed'])
        else:
            axes[3].text(0.5, 0.5, 'No eyes state data', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Eyes Open/Closed Detection (No Data)')
        
        axes[3].grid(True, alpha=0.3)
        
        # Add user state change markers to all plots
        if self.session_data['state_changes']:
            for change in self.session_data['state_changes']:
                change_time = change['time'] - start_time
                for ax in axes:
                    ax.axvline(change_time, color='red', linestyle=':', alpha=0.7, linewidth=2)
                    ax.text(change_time, ax.get_ylim()[1]*0.9, change['to_state'], 
                           rotation=90, ha='right', va='top', fontsize=8, color='red')
        
        # Add event markers
        if self.session_data['events']:
            for event in self.session_data['events']:
                event_time = event['time'] - start_time
                for ax in axes:
                    ax.axvline(event_time, color='green', linestyle='-.', alpha=0.7)
                    ax.text(event_time, ax.get_ylim()[0]*0.9, event['label'], 
                           rotation=90, ha='left', va='bottom', fontsize=7, color='green')
        
        axes[3].set_xlabel('Time (seconds)')
        plt.tight_layout()
        
        plot1_path = f"{save_path_base}_session_overview.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot1_path)
        plt.close()
        
        # Plot 2: State Distribution and Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 2a: Predicted State Distribution
        from collections import Counter
        state_counts = Counter(prediction_states)
        colors = plt.cm.Set3(np.linspace(0, 1, len(state_counts)))
        axes[0,0].pie(state_counts.values(), labels=state_counts.keys(), autopct='%1.1f%%', colors=colors)
        axes[0,0].set_title('Predicted State Distribution')
        
        # Plot 2b: User Reported State Distribution
        if self.session_data['state_changes']:
            user_states = [change['to_state'] for change in self.session_data['state_changes']]
            user_state_counts = Counter(user_states)
            axes[0,1].pie(user_state_counts.values(), labels=user_state_counts.keys(), autopct='%1.1f%%')
            axes[0,1].set_title('User Reported State Distribution')
        else:
            axes[0,1].text(0.5, 0.5, 'No user state data', ha='center', va='center')
            axes[0,1].set_title('User Reported States (No Data)')
        
        # Plot 2c: Band Power Correlations
        import pandas as pd
        df_bands = pd.DataFrame({
            'Alpha': alphas,
            'Beta': betas,
            'Theta': thetas
        })
        corr_matrix = df_bands.corr()
        
        im = axes[1,0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1,0].set_xticks(range(len(corr_matrix.columns)))
        axes[1,0].set_yticks(range(len(corr_matrix.columns)))
        axes[1,0].set_xticklabels(corr_matrix.columns)
        axes[1,0].set_yticklabels(corr_matrix.columns)
        axes[1,0].set_title('Band Power Correlations')
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[1,0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                              ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=axes[1,0])
        
        # Plot 2d: Eyes Closed Analysis
        if self.session_data['eyes_states']:
            eyes_closed_times = []
            eyes_open_times = []
            
            for es in self.session_data['eyes_states']:
                if es['state'] == 'Closed':
                    eyes_closed_times.append(es['timestamp'] - start_time)
                else:
                    eyes_open_times.append(es['timestamp'] - start_time)
            
            # Calculate alpha power during eyes closed vs open
            alpha_during_closed = []
            alpha_during_open = []
            
            for i, bp in enumerate(self.session_data['band_powers']):
                bp_time = bp['timestamp'] - start_time
                
                # Find closest eyes state
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
            
            # Box plot comparison
            if alpha_during_closed and alpha_during_open:
                axes[1,1].boxplot([alpha_during_open, alpha_during_closed], 
                                 labels=['Eyes Open', 'Eyes Closed'])
                axes[1,1].set_ylabel('Alpha Power')
                axes[1,1].set_title('Alpha Power: Eyes Open vs Closed')
                
                # Add statistical info
                mean_open = np.mean(alpha_during_open)
                mean_closed = np.mean(alpha_during_closed)
                ratio = mean_closed / mean_open if mean_open > 0 else 0
                axes[1,1].text(0.05, 0.95, f'Ratio: {ratio:.2f}x', 
                              transform=axes[1,1].transAxes, fontsize=10, 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            else:
                axes[1,1].text(0.5, 0.5, 'Insufficient eyes state data for analysis', 
                              ha='center', va='center')
                axes[1,1].set_title('Alpha Power Analysis (Insufficient Data)')
        else:
            axes[1,1].text(0.5, 0.5, 'No eyes state data', ha='center', va='center')
            axes[1,1].set_title('Eyes State Analysis (No Data)')
        
        plt.tight_layout()
        
        plot2_path = f"{save_path_base}_analysis.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot2_path)
        plt.close()
        
        # Plot 3: Prediction vs User State Comparison (if user states exist)
        if self.session_data['state_changes']:
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Plot prediction timeline
            ax.plot(rel_times, prediction_levels, 'b-', linewidth=2, label='AI Prediction', alpha=0.7)
            ax.fill_between(rel_times, prediction_levels, alpha=0.2, color='blue')
            
            # Add user state regions as background colors
            prev_change_time = 0
            state_colors = {
                'Deeply Relaxed': 'lightblue', 'Relaxed': 'lightgreen', 'Slightly Relaxed': 'lightcyan',
                'Neutral': 'lightgray',
                'Slightly Tense': 'lightyellow', 'Tense': 'lightcoral',
                'Highly Focused': 'darkgreen', 'Focused': 'lightgreen', 'Slightly Focused': 'lightcyan',
                'Slightly Distracted': 'lightyellow', 'Distracted': 'lightcoral'
            }
            
            for i, change in enumerate(self.session_data['state_changes']):
                change_time = change['time'] - start_time
                user_state = change['to_state']
                
                # Fill background color for this state period
                if i < len(self.session_data['state_changes']) - 1:
                    next_change_time = self.session_data['state_changes'][i + 1]['time'] - start_time
                else:
                    next_change_time = max(rel_times)
                
                color = state_colors.get(user_state, 'white')
                ax.axvspan(prev_change_time, next_change_time, alpha=0.3, color=color, label=f'User: {user_state}' if i == 0 else '')
                
                # Add state change markers
                ax.axvline(change_time, color='red', linestyle=':', alpha=0.8, linewidth=2)
                ax.text(change_time, ax.get_ylim()[1]*0.95, user_state, 
                       rotation=90, ha='right', va='top', fontsize=9, color='red', weight='bold')
                
                prev_change_time = change_time
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Prediction Level')
            ax.set_title('AI Predictions vs User Reported States')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-3.5, 3.5)
            
            plt.tight_layout()
            
            plot3_path = f"{save_path_base}_comparison.png"
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            saved_plots.append(plot3_path)
            plt.close()
        
        return saved_plots
    
    def save_session_data(self):
        """Enhanced save session data with automatic plot generation"""
        if not self.is_calibrated:
            messagebox.showwarning("Not Calibrated", "Please calibrate before saving data")
            return
        
        # Create filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_type}_{timestamp_str}"
        
        # Don't ask for save location
        target_dir = os.path.join(os.getcwd(), "muse_Test_data")
        os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Full save path
        save_path = os.path.join(target_dir, filename)
        
        if not save_path:
            return
        
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
            
            # Prepare data for saving
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
                eyes_states=self.session_data['eyes_states'],
                baseline_metrics=self.baseline_metrics,
                session_type=self.session_type,
                sampling_rate=self.sampling_rate,
                channel_names=CHANNEL_NAMES,
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
            success_msg = f"""Session data saved successfully!

Files created:
• Data file: {Path(save_path).name}
• Analysis plots: {plot_count} files
• Session summary: {Path(summary_path).name}

Session Statistics:
• Duration: {self.session_data['session_metadata'].get('total_duration', 0):.1f} seconds
• Predictions: {self.session_data['session_metadata'].get('statistics', {}).get('num_predictions', 0)}
• State changes: {self.session_data['session_metadata'].get('statistics', {}).get('num_state_changes', 0)}
• Events: {self.session_data['session_metadata'].get('statistics', {}).get('num_events', 0)}
• Eyes state changes: {self.session_data['session_metadata'].get('statistics', {}).get('num_eyes_state_changes', 0)}"""
            
            messagebox.showinfo("Save Complete", success_msg)
            print(f"Session data and {plot_count} plots saved to: {save_dir}")
            
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
    app = EnhancedMuseEEGMonitorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()