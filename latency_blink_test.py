import sys
import time
import numpy as np
import pylsl
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import signal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import deque
import random
import datetime
import os
from pathlib import Path
import json

# Output directory for results
LATENCY_TEST_DIR = "eeg_latency_test_data"

class EEGPlotWidget(FigureCanvas):
    """Real-time EEG plot widget showing filtered data"""
    def __init__(self, parent=None, width=10, height=4, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(EEGPlotWidget, self).__init__(self.fig)
        
        # Create subplots for 4 EEG channels (TP9, AF7, AF8, TP10)
        self.axes = []
        channel_names = ["TP9", "AF7", "AF8", "TP10"]
        
        for i in range(4):
            ax = self.fig.add_subplot(4, 1, i+1)
            ax.set_ylabel(channel_names[i], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if i < 3:
                ax.set_xticklabels([])
            self.axes.append(ax)
        
        self.axes[-1].set_xlabel('Time (s)', fontsize=9)
        
        # Two lines per channel: raw (light) and filtered (bold)
        self.raw_lines = [ax.plot([], [], 'lightblue', linewidth=0.5, alpha=0.6, label='Raw')[0] for ax in self.axes]
        self.filtered_lines = [ax.plot([], [], 'blue', linewidth=1.5, label='Filtered')[0] for ax in self.axes]
        
        # Add legend to first subplot
        self.axes[0].legend(loc='upper right', fontsize=8)
        
        # Data buffers for plotting (2 seconds of data for better movement visibility)
        self.plot_buffer_size = 512  # 2 seconds at 256 Hz
        self.raw_data = [deque(maxlen=self.plot_buffer_size) for _ in range(4)]
        self.filtered_data = [deque(maxlen=self.plot_buffer_size) for _ in range(4)]
        self.time_data = deque(maxlen=self.plot_buffer_size)
        
        # Movement markers (renamed from blink markers)
        self.movement_markers = []
        self.prompt_markers = []
        
        self.fig.tight_layout(pad=1.0)
    
    def update_plot(self, raw_sample, filtered_sample, timestamp):
        """Update the plot with new raw and filtered EEG data"""
        # Add data to buffers
        for i in range(min(4, len(raw_sample))):
            self.raw_data[i].append(raw_sample[i])
            self.filtered_data[i].append(filtered_sample[i])
        self.time_data.append(timestamp)
        
        # Update plot lines
        if len(self.time_data) > 1:
            time_array = np.array(self.time_data) - self.time_data[0]  # Relative time
            
            for i in range(4):
                if i < len(self.raw_data):
                    # Update raw data line
                    self.raw_lines[i].set_xdata(time_array)
                    self.raw_lines[i].set_ydata(self.raw_data[i])
                    
                    # Update filtered data line
                    self.filtered_lines[i].set_xdata(time_array)
                    self.filtered_lines[i].set_ydata(self.filtered_data[i])
                    
                    # Auto-scale Y axis based on filtered data
                    if len(self.filtered_data[i]) > 10:
                        filtered_array = np.array(self.filtered_data[i])
                        data_min, data_max = np.min(filtered_array), np.max(filtered_array)
                        padding = (data_max - data_min) * 0.1 if data_max != data_min else 10
                        self.axes[i].set_ylim(data_min - padding, data_max + padding)
                    
                    # Set X axis
                    self.axes[i].set_xlim(0, max(time_array) if len(time_array) > 0 else 1)
        
        self.draw_idle()
    
    def add_prompt_marker(self, timestamp):
        """Add marker when prompt is shown"""
        # Clear old markers
        for marker in self.prompt_markers:
            marker.remove()
        self.prompt_markers.clear()
        
        # Add new markers
        if len(self.time_data) > 0:
            relative_time = timestamp - self.time_data[0]
            for ax in self.axes:
                marker = ax.axvline(relative_time, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Prompt')
                self.prompt_markers.append(marker)
        
        self.draw_idle()
    
    def add_movement_marker(self, timestamp):
        """Add marker when movement is detected"""
        # Clear old markers
        for marker in self.movement_markers:
            marker.remove()
        self.movement_markers.clear()
        
        # Add new markers
        if len(self.time_data) > 0:
            relative_time = timestamp - self.time_data[0]
            for ax in self.axes:
                marker = ax.axvline(relative_time, color='green', linestyle=':', alpha=0.8, linewidth=2, label='Movement')
                self.movement_markers.append(marker)
        
        self.draw_idle()
    
    def clear_markers(self):
        """Clear all markers"""
        for marker in self.prompt_markers + self.movement_markers:
            marker.remove()
        self.prompt_markers.clear()
        self.movement_markers.clear()
        self.draw_idle()


class MovementDetector:
    """Simple movement detector for headband adjustments - much more reliable than blinks"""
    def __init__(self, sampling_rate=256.0, threshold_multiplier=5.0):
        """
        Movement detector for large EEG amplitude changes
        
        Args:
            sampling_rate: The sampling rate of the EEG data
            threshold_multiplier: Factor to multiply std dev (5-10 for movement vs 2.5 for blinks)
        """
        self.sampling_rate = sampling_rate
        self.threshold_multiplier = threshold_multiplier
        
        # Much shorter baseline - just need recent stable values
        self.baseline_window = int(sampling_rate * 2.0)  # 2 seconds
        self.detection_window = int(sampling_rate * 0.5)  # 500ms detection window
        
        # Buffers for all 4 channels - movement can happen on any channel
        self.channel_buffers = [deque(maxlen=self.baseline_window) for _ in range(4)]
        self.filtered_buffers = [deque(maxlen=self.baseline_window) for _ in range(4)]
        
        # Rolling baseline stats
        self.baseline_means = [0.0] * 4
        self.baseline_stds = [1.0] * 4
        self.baseline_ready = False
        self.baseline_calculated = False  # For compatibility with main code
        
        # Filtering (same as before)
        self.filter_order = 2
        self.lowcut = 0.5
        self.highcut = 30.0
        self.update_filter_coefficients()
        self.raw_buffers = [deque(maxlen=50) for _ in range(4)]  # Small buffer for filtering
        
        # Detection state
        self.last_detection_time = 0
        self.debounce_time = 0.5  # Reduce from 1.0 to 0.5 seconds
        self.in_movement = False
        self.detection_enabled = True  # Add this flag
        
        # Minimum samples for startup (much faster than blinks)
        self.min_baseline_samples = max(50, int(sampling_rate * 0.5))  # 0.5 seconds
        
        print(f"MovementDetector initialized: threshold={threshold_multiplier}x, need {self.min_baseline_samples} samples")
    
    def update_filter_coefficients(self):
        """Update filter coefficients based on sampling rate"""
        nyq = 0.5 * self.sampling_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.filter_b, self.filter_a = signal.butter(self.filter_order, [low, high], btype='band')
    
    def filter_eeg_sample(self, raw_sample):
        """Apply real-time filtering to a single EEG sample"""
        filtered_sample = raw_sample.copy()
        
        # Store raw data
        for i in range(min(4, len(raw_sample))):
            self.raw_buffers[i].append(raw_sample[i])
        
        # Apply filtering if we have enough samples
        min_samples_for_filter = max(20, 3 * self.filter_order)
        
        for i in range(min(4, len(raw_sample))):
            if len(self.raw_buffers[i]) >= min_samples_for_filter:
                try:
                    recent_raw = np.array(list(self.raw_buffers[i])[-min_samples_for_filter:])
                    padlen = min(3 * self.filter_order, len(recent_raw) // 4)
                    if padlen < 1:
                        padlen = None
                    filtered_segment = signal.filtfilt(self.filter_b, self.filter_a, recent_raw, padlen=padlen)
                    filtered_sample[i] = filtered_segment[-1]
                except:
                    filtered_sample[i] = raw_sample[i]
            else:
                filtered_sample[i] = raw_sample[i]
        
        return filtered_sample
    
    def update(self, raw_sample, timestamp):
        """Update with new sample and check for movement"""
        if len(raw_sample) < 4:
            return raw_sample, False, 0
        
        # Apply filtering
        filtered_sample = self.filter_eeg_sample(raw_sample)
        
        # Add to buffers
        for i in range(4):
            self.channel_buffers[i].append(raw_sample[i])
            self.filtered_buffers[i].append(filtered_sample[i])
        
        # Calculate rolling baseline (much faster than blinks)
        if len(self.channel_buffers[0]) >= self.min_baseline_samples:
            self.update_baseline()
        
        # Check for movement if baseline ready
        if self.baseline_ready:
            is_movement, confidence = self.detect_movement(timestamp)
            return filtered_sample, is_movement, confidence
        
        return filtered_sample, False, 0
    
    def calculate_baseline(self):
        """Calculate baseline EEG values for movement detection"""
        if len(self.channel_buffers[0]) < self.min_baseline_samples:
            return
            
        # Calculate baseline for all 4 channels using filtered data
        for i in range(4):
            if len(self.filtered_buffers[i]) >= self.min_baseline_samples:
                baseline_data = np.array(list(self.filtered_buffers[i])[-self.min_baseline_samples:])
                
                # Get mean and std for this channel
                self.baseline_means[i] = np.mean(baseline_data)
                self.baseline_stds[i] = max(1.0, np.std(baseline_data))  # Prevent division by zero
        
        self.baseline_ready = True
        self.baseline_calculated = True  # Add this attribute for compatibility
        
        print(f"Movement baseline calculated from {self.min_baseline_samples} filtered samples:")
        channel_names = ["TP9", "AF7", "AF8", "TP10"]
        for i in range(4):
            print(f"  {channel_names[i]}: {self.baseline_means[i]:.2f} ± {self.baseline_stds[i]:.2f}")
    
    def update_baseline(self):
        """Update rolling baseline with recent stable data"""
        for i in range(4):
            if len(self.filtered_buffers[i]) >= self.min_baseline_samples:
                # Use filtered data for baseline (more stable)
                recent_data = np.array(list(self.filtered_buffers[i])[-self.min_baseline_samples:])
                self.baseline_means[i] = np.mean(recent_data)
                self.baseline_stds[i] = max(1.0, np.std(recent_data))
        
        if not self.baseline_ready:
            self.baseline_ready = True
            self.baseline_calculated = True
            print(f"Movement baseline calculated:")
            for i, (mean, std) in enumerate(zip(self.baseline_means, self.baseline_stds)):
                print(f"  CH{i+1}: {mean:.2f} ± {std:.2f}")
    
    def enable_detection(self):
        """Enable movement detection"""
        self.detection_enabled = True
        self.in_movement = False  # Reset movement state
        print("Movement detection ENABLED")
        
    def disable_detection(self):
        """Disable movement detection"""
        self.detection_enabled = False
        self.in_movement = False
        print("Movement detection DISABLED")
    
    def detect_movement(self, timestamp):
        """Detect large amplitude changes indicating headband movement"""
        # Don't detect if disabled
        if not self.detection_enabled:
            return False, 0
            
        # Don't detect movements too close together
        if timestamp - self.last_detection_time < self.debounce_time:
            return False, 0
        
        # Check recent samples (last 500ms)
        check_samples = min(self.detection_window, len(self.filtered_buffers[0]))
        if check_samples < 10:
            return False, 0
        
        detected_channels = []
        max_deviation = 0
        
        # Check all 4 channels for movement
        for i in range(4):
            if len(self.filtered_buffers[i]) >= check_samples:
                recent = list(self.filtered_buffers[i])[-check_samples:]
                
                # Calculate z-scores relative to baseline
                z_scores = [(x - self.baseline_means[i]) / self.baseline_stds[i] for x in recent]
                
                # Find maximum absolute deviation (movement can be + or -)
                max_abs_z = max(abs(min(z_scores)), abs(max(z_scores)))
                
                # Movement detected if exceeds threshold
                if max_abs_z > self.threshold_multiplier:
                    detected_channels.append(i)
                    max_deviation = max(max_deviation, max_abs_z)
        
        # Movement detected if any channel shows large change
        if detected_channels and not self.in_movement:
            self.in_movement = True
            self.last_detection_time = timestamp
            confidence = min(100, (max_deviation / self.threshold_multiplier) * 100)
            
            channel_names = ["TP9", "AF7", "AF8", "TP10"]
            detected_names = [channel_names[i] for i in detected_channels]
            print(f"Movement detected on {detected_names}, max deviation: {max_deviation:.1f}σ, confidence: {confidence:.1f}%")
            return True, confidence
        elif max_deviation < 2.0:  # Reset if values return to normal
            self.in_movement = False
        
        return False, 0
    
    def reset_baseline(self):
        """Reset the baseline calculation"""
        self.baseline_ready = False
        self.baseline_calculated = False
        self.baseline_means = [0.0] * 4
        self.baseline_stds = [1.0] * 4
        self.in_movement = False
        
        # Clear all buffers
        for buf in self.channel_buffers:
            buf.clear()
        for buf in self.filtered_buffers:
            buf.clear()
        for buf in self.raw_buffers:
            buf.clear()
            
        print("Movement detector reset")
    
    def get_stats(self):
        """Get current detector statistics"""
        return {
            'baseline_ready': self.baseline_ready,
            'baseline_means': self.baseline_means.copy(),
            'baseline_stds': self.baseline_stds.copy(),
            'buffer_size': len(self.channel_buffers[0]) if self.channel_buffers else 0,
            'threshold_multiplier': self.threshold_multiplier,
            'sampling_rate': self.sampling_rate,
            'detection_type': 'movement',
            'filter_params': {
                'lowcut': self.lowcut,
                'highcut': self.highcut,
                'order': self.filter_order
            }
        }


class EnhancedLatencyTestWindow(QtWidgets.QMainWindow):
    """Enhanced EEG latency test with automatic baseline and filtered data display"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Muse EEG Latency Test v3.0 - Movement Detection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create output directory
        self.output_dir = Path(LATENCY_TEST_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        # LSL connection state
        self.inlet = None
        self.connected = False
        self.sampling_rate = 256.0
        self.channel_count = 4
        
        # Test parameters
        self.num_trials = 10
        self.current_trial = 0
        self.trial_results = []
        self.movement_request_time = None  # Changed from blink_request_time
        self.movement_request_lsl_time = None  # Changed from blink_request_lsl_time
        self.test_running = False
        self.trial_running = False
        self.baseline_phase = False
        self.waiting_for_lsl_timestamp = False
        
        # Enhanced movement detector
        self.movement_detector = MovementDetector(sampling_rate=self.sampling_rate)
        
        # Session data for analysis
        self.session_data = {
            'start_time': None,
            'end_time': None,
            'trials': [],
            'detector_stats': {},
            'system_info': {
                'sampling_rate': self.sampling_rate,
                'channel_count': self.channel_count,
                'user': 'up202107351',
                'test_date': datetime.datetime.now().isoformat()
            }
        }
        
        # Setup UI
        self.setup_ui()
        
        # Setup LSL inlet timer
        self.lsl_timer = QtCore.QTimer()
        self.lsl_timer.timeout.connect(self.process_eeg_data)
        self.lsl_timer.setInterval(10)  # Process data every 10ms
        
        # Setup result variables
        self.movement_detected = False  # Changed from blink_detected
        self.detection_time = None
        self.countdown_timer = None
        self.baseline_countdown_timer = None

    def setup_ui(self):
        """Set up the enhanced user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Enhanced title
        title_label = QtWidgets.QLabel("Enhanced Muse EEG Latency Test v3.0")
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        subtitle_label = QtWidgets.QLabel("Hardware Latency Measurement via Headband Movement Detection • Filtered Data Display")
        subtitle_label.setFont(QtGui.QFont("Arial", 10))
        subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        main_layout.addWidget(subtitle_label)
        
        # Enhanced status indicators
        status_layout = QtWidgets.QHBoxLayout()
        
        self.connection_status = QtWidgets.QLabel("Status: Not Connected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        status_layout.addWidget(self.connection_status)
        
        self.sample_rate_label = QtWidgets.QLabel("Sample Rate: N/A")
        status_layout.addWidget(self.sample_rate_label)
        
        self.detection_status = QtWidgets.QLabel("Detection: Ready for baseline")
        status_layout.addWidget(self.detection_status)
        
        self.output_dir_label = QtWidgets.QLabel(f"Output: {self.output_dir}")
        self.output_dir_label.setStyleSheet("color: blue; font-size: 9px;")
        status_layout.addWidget(self.output_dir_label)
        
        main_layout.addLayout(status_layout)
        
        # Add real-time EEG plot showing filtered data
        self.eeg_plot = EEGPlotWidget()
        main_layout.addWidget(self.eeg_plot)
        
        # Main instruction display
        self.instruction_label = QtWidgets.QLabel("Click 'Connect' to start")
        self.instruction_label.setFont(QtGui.QFont("Arial", 14))
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        self.instruction_label.setWordWrap(True)
        main_layout.addWidget(self.instruction_label)
        
        # Countdown timer display
        self.countdown_display = QtWidgets.QLabel("")
        self.countdown_display.setFont(QtGui.QFont("Arial", 36, QtGui.QFont.Bold))
        self.countdown_display.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.countdown_display)
        
        # Enhanced results display
        results_group = QtWidgets.QGroupBox("Latency Test Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        
        self.results_table = QtWidgets.QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Trial #", "Latency (ms)", "Confidence (%)", "Status"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        # Statistics display
        stats_layout = QtWidgets.QHBoxLayout()
        
        self.avg_latency_label = QtWidgets.QLabel("Average Latency: N/A")
        self.avg_latency_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        stats_layout.addWidget(self.avg_latency_label)
        
        self.success_rate_label = QtWidgets.QLabel("Success Rate: N/A")
        stats_layout.addWidget(self.success_rate_label)
        
        self.std_dev_label = QtWidgets.QLabel("Std Dev: N/A")
        stats_layout.addWidget(self.std_dev_label)
        
        results_layout.addLayout(stats_layout)
        main_layout.addWidget(results_group)
        
        # Enhanced buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.connect_button = QtWidgets.QPushButton("Connect to Muse")
        self.connect_button.clicked.connect(self.toggle_connection)
        button_layout.addWidget(self.connect_button)
        
        self.start_button = QtWidgets.QPushButton("Start Latency Test")
        self.start_button.clicked.connect(self.start_test)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_test)
        self.reset_button.setEnabled(False)
        button_layout.addWidget(self.reset_button)
        
        self.save_button = QtWidgets.QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)
        
        # Enhanced options for movement detection
        options_layout = QtWidgets.QHBoxLayout()
        
        options_layout.addWidget(QtWidgets.QLabel("Number of Trials:"))
        self.trials_spinbox = QtWidgets.QSpinBox()
        self.trials_spinbox.setRange(1, 50)
        self.trials_spinbox.setValue(10)
        self.trials_spinbox.valueChanged.connect(self.update_trial_count)
        options_layout.addWidget(self.trials_spinbox)
        
        options_layout.addWidget(QtWidgets.QLabel("Movement Threshold:"))
        self.threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.threshold_spinbox.setRange(1.0, 15.0)
        self.threshold_spinbox.setValue(3.0)  # Lower default for better sensitivity
        self.threshold_spinbox.setSingleStep(0.5)
        self.threshold_spinbox.valueChanged.connect(self.update_threshold)
        options_layout.addWidget(self.threshold_spinbox)
        
        options_layout.addWidget(QtWidgets.QLabel("Baseline Duration (s):"))
        self.baseline_duration_spinbox = QtWidgets.QDoubleSpinBox()
        self.baseline_duration_spinbox.setRange(1.0, 5.0)
        self.baseline_duration_spinbox.setValue(2.0)  # Shorter for movement
        self.baseline_duration_spinbox.setSingleStep(0.5)
        options_layout.addWidget(self.baseline_duration_spinbox)
        
        main_layout.addLayout(options_layout)
        
    def toggle_connection(self):
        """Connect to or disconnect from the Muse headset via LSL"""
        if not self.connected:
            self.connect_to_muse()
        else:
            self.disconnect_from_muse()
    
    def connect_to_muse(self):
        """Connect to Muse headset via LSL with enhanced error handling"""
        try:
            self.connection_status.setText("Status: Searching for EEG stream...")
            self.connection_status.setStyleSheet("color: orange; font-weight: bold")
            QtWidgets.QApplication.processEvents()
            
            # Look for EEG streams
            print("Looking for EEG stream...")
            streams = pylsl.resolve_byprop('type', 'EEG', timeout=5)
            
            if not streams:
                self.connection_status.setText("Status: No EEG stream found")
                self.connection_status.setStyleSheet("color: red; font-weight: bold")
                QtWidgets.QMessageBox.warning(self, "Connection Error", 
                                          "No EEG stream found. Make sure Muse is streaming data.")
                return
                
            # Create an inlet from the first stream
            self.inlet = pylsl.StreamInlet(streams[0], max_chunklen=128)
            
            # Get stream info
            info = self.inlet.info()
            self.sampling_rate = info.nominal_srate()
            self.channel_count = info.channel_count()
            
            # Update session data
            self.session_data['system_info']['sampling_rate'] = self.sampling_rate
            self.session_data['system_info']['channel_count'] = self.channel_count
            self.session_data['system_info']['stream_name'] = info.name()
            
            # Update UI
            self.connection_status.setText(f"Status: Connected to {info.name()}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold")
            self.sample_rate_label.setText(f"Sample Rate: {self.sampling_rate} Hz")
            
            # Start processing EEG data
            self.connected = True
            self.connect_button.setText("Disconnect")
            self.start_button.setEnabled(True)
            self.lsl_timer.start()
            
            # Create new movement detector with correct sampling rate
            self.movement_detector = MovementDetector(
                sampling_rate=self.sampling_rate,
                threshold_multiplier=self.threshold_spinbox.value()
            )
            
            # Update instructions
            self.instruction_label.setText(
                "Connected! EEG data is now streaming with filtering.\n\n"
                "Click 'Start Latency Test' to begin.\n"
                "The test will automatically:\n"
                "1. Calculate baseline (2 seconds)\n"
                "2. Run movement detection trials\n\n"
                "You can see both raw (light) and filtered (bold) data in the plot."
            )
            self.detection_status.setText("Detection: Ready for baseline")
            self.detection_status.setStyleSheet("color: blue; font-weight: bold")
            
            print(f"Connected to EEG stream: {info.name()}")
            print(f"Sampling rate: {self.sampling_rate} Hz, Channels: {self.channel_count}")
            
        except Exception as e:
            self.connection_status.setText(f"Status: Error - {str(e)}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold")
            QtWidgets.QMessageBox.critical(self, "Connection Error", f"Error: {str(e)}")
    
    def disconnect_from_muse(self):
        """Disconnect from the Muse headset"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if self.inlet:
            try:
                self.inlet.close_stream()
            except:
                pass
            self.inlet = None
        
        self.connected = False
        self.connect_button.setText("Connect to Muse")
        self.start_button.setEnabled(False)
        
        self.connection_status.setText("Status: Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        self.detection_status.setText("Detection: Not Ready")
        self.detection_status.setStyleSheet("color: gray")
        
    def update_trial_count(self, value):
        """Update the number of trials"""
        self.num_trials = value
        
    def update_threshold(self, value):
        """Update the movement detection threshold factor"""
        if hasattr(self, 'movement_detector'):
            self.movement_detector.threshold_multiplier = value
    
    def start_test(self):
        """Start the enhanced latency test with automatic baseline calculation"""
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", 
                                      "Please connect to Muse headset first.")
            return
        
        # Start the test regardless of baseline status
        self.test_running = True
        self.baseline_phase = True
        self.current_trial = 0
        self.trial_results = []
        
        # Reset and prepare for new baseline calculation
        self.movement_detector.reset_baseline()
        
        # Initialize session data
        self.session_data['start_time'] = datetime.datetime.now().isoformat()
        self.session_data['trials'] = []
        
        # Clear results table
        self.results_table.setRowCount(0)
        self.update_statistics_display()
        
        # Update UI
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(False) 
        self.connect_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # Clear plot markers
        self.eeg_plot.clear_markers()
        
        # Start baseline calculation phase
        self.start_baseline_calculation()

    def start_baseline_calculation(self):
        """Start the baseline calculation phase"""
        baseline_duration = self.baseline_duration_spinbox.value()
        
        self.instruction_label.setText(
            "Calculating baseline for movement detection...\n\n"
            "Please keep the headband still and stable.\n"
            f"This will take {baseline_duration:.0f} seconds.\n\n"
            "Watch the filtered EEG data (bold lines) stabilize."
        )
        self.instruction_label.setStyleSheet(
            "margin: 20px 0; padding: 20px; background-color: #fff3cd; "
            "border-radius: 10px; color: #856404;"
        )
        
        # Start countdown for baseline
        self.baseline_countdown_timer = QtCore.QTimer()
        self.baseline_remaining = int(baseline_duration)
        self.countdown_display.setText(f"Baseline: {self.baseline_remaining}s")
        
        def update_baseline_countdown():
            self.baseline_remaining -= 1
            if self.baseline_remaining > 0:
                self.countdown_display.setText(f"Baseline: {self.baseline_remaining}s")
                # Update progress
                total_duration = self.baseline_duration_spinbox.value()
                progress = ((total_duration - self.baseline_remaining) / total_duration) * 100
                self.detection_status.setText(f"Baseline: {progress:.0f}% complete")
                self.detection_status.setStyleSheet("color: orange; font-weight: bold")
            else:
                self.baseline_countdown_timer.stop()
                self.finish_baseline_calculation()
        
        self.baseline_countdown_timer.timeout.connect(update_baseline_countdown)
        self.baseline_countdown_timer.start(1000)

    def finish_baseline_calculation(self):
        """Finish baseline calculation and start the actual test"""
        try:
            self.baseline_phase = False
            
            # Check if we have any data at all
            if not hasattr(self, 'movement_detector') or self.movement_detector is None:
                raise Exception("Movement detector not initialized")
            
            buffer_size = len(self.movement_detector.channel_buffers[0])
            print(f"Finishing baseline with {buffer_size} samples")
            
            # Force baseline calculation with current data (lowered requirement)
            if buffer_size >= 25:  # Much lower requirement
                print("Attempting baseline calculation...")
                self.movement_detector.calculate_baseline()
            else:
                print(f"Not enough samples for baseline: {buffer_size} < 25")
                # Force it anyway with whatever we have
                if buffer_size >= 10:
                    print("Forcing baseline calculation with limited data...")
                    self.movement_detector.min_baseline_samples = buffer_size
                    self.movement_detector.calculate_baseline()
            
            # Check if baseline was successful
            if hasattr(self.movement_detector, 'baseline_calculated') and self.movement_detector.baseline_calculated:
                # Baseline successful
                print("Baseline calculation successful!")
                self.instruction_label.setText(
                    "Baseline calculation complete!\n\n"
                    "Starting latency test trials...\n"
                    "When prompted, quickly tap or adjust the headband."
                )
                self.instruction_label.setStyleSheet(
                    "margin: 20px 0; padding: 20px; background-color: #d4edda; "
                    "border-radius: 10px; color: #155724;"
                )
                self.countdown_display.setText("Ready!")
                self.detection_status.setText("Detection: Active")
                self.detection_status.setStyleSheet("color: green; font-weight: bold")
                
                # Store baseline stats safely
                try:
                    self.session_data['detector_stats'] = self.movement_detector.get_stats()
                except Exception as e:
                    print(f"Warning: Could not save detector stats: {e}")
                    self.session_data['detector_stats'] = {}
                
                # Start first trial after short delay
                print("Starting first trial in 2 seconds...")
                QtCore.QTimer.singleShot(2000, self.start_trial)
                
            else:
                # Baseline failed - but continue anyway for testing
                print("Baseline calculation failed, but continuing for testing...")
                
                # Force baseline to be "calculated" 
                self.movement_detector.baseline_calculated = True
                self.movement_detector.baseline_ready = True
                self.movement_detector.baseline_means = [0.0, 0.0, 0.0, 0.0]  # Dummy values
                self.movement_detector.baseline_stds = [1.0, 1.0, 1.0, 1.0]    # Dummy values
                
                self.instruction_label.setText(
                    "Baseline calculation incomplete but continuing...\n\n"
                    "Detection may be less reliable.\n"
                    "When prompted, quickly tap or adjust the headband."
                )
                self.instruction_label.setStyleSheet(
                    "margin: 20px 0; padding: 20px; background-color: #fff3cd; "
                    "border-radius: 10px; color: #856404;"
                )
                self.countdown_display.setText("Ready!")
                self.detection_status.setText("Detection: Active (Limited)")
                self.detection_status.setStyleSheet("color: orange; font-weight: bold")
                
                # Start first trial after short delay
                print("Starting first trial in 2 seconds...")
                QtCore.QTimer.singleShot(2000, self.start_trial)
                
        except Exception as e:
            print(f"Error in finish_baseline_calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error to user
            self.instruction_label.setText(
                f"Error during baseline calculation:\n\n"
                f"{str(e)}\n\n"
                "Please try again or check the console for details."
            )
            self.instruction_label.setStyleSheet(
                "margin: 20px 0; padding: 20px; background-color: #f8d7da; "
                "border-radius: 10px; color: #721c24;"
            )
            self.countdown_display.setText("ERROR")
            
            # Re-enable start button
            self.test_running = False
            self.start_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.connect_button.setEnabled(True)

    def start_trial(self):
        """Start a single trial of the latency test"""
        if self.current_trial >= self.num_trials:
            self.finish_test()
            return
            
        self.current_trial += 1
        
        # Reset state for this trial
        self.movement_detected = False
        self.detection_time = None
        self.movement_request_time = None
        self.movement_request_lsl_time = None
        self.waiting_for_lsl_timestamp = False
        
        # Reset detector state
        self.movement_detector.disable_detection()  # Start with detection disabled
        
        # Show instructions
        self.instruction_label.setText(
            f"Trial {self.current_trial} of {self.num_trials}\n\n"
            "Keep the headband still and stable.\n"
            "When prompted, quickly tap or adjust the headband.\n\n"
            "Watch for large spikes in any EEG channel."
        )
        self.instruction_label.setStyleSheet(
            "margin: 20px 0; padding: 20px; background-color: #e3f2fd; "
            "border-radius: 10px; color: #0d47a1;"
        )
        
        # Start countdown to prompt
        self.start_countdown(random.randint(3, 6))  # Random delay between 3-6 seconds
        
    def start_countdown(self, seconds):
        """Start countdown timer before prompting for movement"""
        self.countdown_display.setText(str(seconds))
        
        self.trial_running = True
        
        # DISABLE detection during countdown
        self.movement_detector.disable_detection()
        
        # Create countdown timer
        self.countdown_timer = QtCore.QTimer()
        remaining_time = seconds
        
        def update_countdown():
            nonlocal remaining_time
            remaining_time -= 1
            
            if remaining_time > 0:
                self.countdown_display.setText(str(remaining_time))
            else:
                self.countdown_timer.stop()
                self.countdown_display.setText("")
                self.prompt_for_movement()
                
        self.countdown_timer.timeout.connect(update_countdown)
        self.countdown_timer.start(1000)

    def prompt_for_movement(self):
        """Prompt the user to move the headband and record the time"""
        self.instruction_label.setText("ADJUST HEADBAND NOW!")
        self.instruction_label.setStyleSheet(
            "margin: 20px 0; padding: 20px; background-color: #ffebee; "
            "border-radius: 10px; font-size: 24pt; font-weight: bold; color: #c62828;"
        )
        
        # Record the time of the prompt
        self.movement_request_time = time.time()
        self.waiting_for_lsl_timestamp = True
        
        # ENABLE detection right before prompt
        self.movement_detector.enable_detection()
        
        # Set timeout for this trial
        QtCore.QTimer.singleShot(5000, self.check_trial_timeout)
        
    def check_trial_timeout(self):
        """Check if the trial has timed out"""
        if self.trial_running and not self.movement_detected:  # Changed from blink_detected
            # Trial timed out
            self.movement_detector.disable_detection()
            self.trial_running = False
            self.instruction_label.setText(
                "No movement detected - trial timed out.\n\n"
                "Try to tap/adjust the headband more clearly next time.\n"
                "Moving to next trial..."
            )
            self.instruction_label.setStyleSheet(
                "margin: 20px 0; padding: 20px; background-color: #fff3e0; "
                "border-radius: 10px; color: #ef6c00;"
            )
            
            # Record failed trial
            trial_data = {
                'trial': self.current_trial,
                'latency_ms': -1,
                'confidence': 0,
                'status': 'timeout',
                'timestamp': time.time()
            }
            
            # Add failed trial to results
            self.add_trial_result(-1, 0, 'Timeout')
            self.session_data['trials'].append(trial_data)
            
            # Continue to next trial after delay
            QtCore.QTimer.singleShot(2000, self.start_trial)
            
    def add_trial_result(self, latency_ms, confidence, status='Success'):
        """Add trial result to the table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Add data to table
        self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.current_trial)))
        
        if latency_ms < 0:
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem("--"))
        else:
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{latency_ms:.1f}"))
            
        self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{confidence:.1f}"))
        self.results_table.setItem(row, 3, QtWidgets.QTableWidgetItem(status))
        
        # Color code the row based on status
        if status == 'Success':
            for col in range(4):
                self.results_table.item(row, col).setBackground(QtGui.QColor(200, 255, 200))
        elif status == 'Timeout':
            for col in range(4):
                self.results_table.item(row, col).setBackground(QtGui.QColor(255, 200, 200))
        
        # Store result
        self.trial_results.append((latency_ms, confidence, status))
        
        # Update statistics
        self.update_statistics_display()
    
    def update_statistics_display(self):
        """Update the statistics display"""
        if not self.trial_results:
            return
            
        # Calculate successful trials only
        successful_trials = [t for t in self.trial_results if t[0] >= 0]
        
        if successful_trials:
            latencies = [t[0] for t in successful_trials]
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            self.avg_latency_label.setText(f"Average Latency: {avg_latency:.1f} ms")
            self.std_dev_label.setText(f"Std Dev: {std_latency:.1f} ms")
        else:
            self.avg_latency_label.setText("Average Latency: N/A")
            self.std_dev_label.setText("Std Dev: N/A")
        
        # Success rate
        success_rate = len(successful_trials) / len(self.trial_results) * 100
        self.success_rate_label.setText(f"Success Rate: {success_rate:.1f}%")
    
    def process_eeg_data(self):
        """Process incoming EEG data with filtering and visualization"""
        if not self.connected or not self.inlet:
            return
            
        try:
            # Get all available samples with LSL timestamps
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=100)
            
            if not chunk:
                return
                
            # Track movement detection across all samples in this chunk
            chunk_movement_detected = False  # Changed from chunk_blink_detected
            chunk_confidence = 0
            chunk_detection_timestamp = None
            
            # Process each sample
            for i, raw_sample in enumerate(chunk):
                if len(raw_sample) < 4:
                    print(f"Warning: Sample {i} has only {len(raw_sample)} channels")
                    continue
                    
                lsl_timestamp = timestamps[i]
                
                # Apply filtering and get detection results
                try:
                    filtered_sample, is_movement, confidence = self.movement_detector.update(raw_sample, lsl_timestamp)
                    
                    # Track any movement detection in this chunk
                    if is_movement and not chunk_movement_detected:  # Changed from blink
                        chunk_movement_detected = True
                        chunk_confidence = confidence
                        chunk_detection_timestamp = lsl_timestamp
                        
                except Exception as e:
                    print(f"Error in movement detector update: {e}")
                    filtered_sample = raw_sample  # Fallback to raw data
                    continue
                
                # Update EEG plot with both raw and filtered data
                try:
                    self.eeg_plot.update_plot(raw_sample, filtered_sample, lsl_timestamp)
                except Exception as e:
                    print(f"Error updating plot: {e}")
                
                # Update detection status based on current state
                if self.baseline_phase:
                    # During baseline calculation
                    try:
                        buffer_size = len(self.movement_detector.channel_buffers[0])
                        min_required = self.movement_detector.min_baseline_samples
                        if buffer_size < min_required:
                            progress = (buffer_size / min_required) * 100
                            self.detection_status.setText(f"Baseline: {progress:.0f}% samples")
                            self.detection_status.setStyleSheet("color: orange; font-weight: bold")
                    except Exception as e:
                        print(f"Error updating baseline status: {e}")
                elif hasattr(self.movement_detector, 'baseline_calculated') and self.movement_detector.baseline_calculated:
                    self.detection_status.setText("Detection: Active")
                    self.detection_status.setStyleSheet("color: green; font-weight: bold")
                
                # Capture LSL timestamp right after prompt (only needs to happen once)
                if (self.trial_running and not self.baseline_phase and 
                    self.waiting_for_lsl_timestamp and self.movement_request_time and  # Changed variable names
                    not self.movement_request_lsl_time):  # Changed variable name
                    self.movement_request_lsl_time = lsl_timestamp  # Changed variable name
                    self.waiting_for_lsl_timestamp = False
                    self.eeg_plot.add_prompt_marker(lsl_timestamp)
                    print(f"Prompt marker added at LSL timestamp: {lsl_timestamp}")
            
            # Handle movement detection AFTER processing all samples in chunk
            if (self.trial_running and not self.baseline_phase and 
                self.movement_request_lsl_time and not self.movement_detected and  # Changed variable names
                chunk_movement_detected):  # Changed from chunk_blink_detected
                
                self.movement_detector.disable_detection()
        
                self.detection_time = chunk_detection_timestamp
                self.movement_detected = True
                
                # Calculate latency using LSL timestamps for better precision
                latency_ms = (self.detection_time - self.movement_request_lsl_time) * 1000.0  # Changed variable name
                
                print(f"Movement detected! Latency: {latency_ms:.1f} ms, Confidence: {chunk_confidence:.1f}%")
                
                # Add detection marker to plot
                self.eeg_plot.add_movement_marker(chunk_detection_timestamp)  # Changed from add_blink_marker
                
                # Update UI
                self.instruction_label.setText(
                    f"Movement detected!\n\n"
                    f"Latency: {latency_ms:.1f} ms\n"
                    f"Confidence: {chunk_confidence:.1f}%\n\n"
                    f"Great! Moving to next trial..."
                )
                self.instruction_label.setStyleSheet(
                    "margin: 20px 0; padding: 20px; background-color: #e8f5e8; "
                    "border-radius: 10px; color: #2e7d32; font-weight: bold;"
                )
                
                # Record successful trial
                trial_data = {
                    'trial': self.current_trial,
                    'latency_ms': latency_ms,
                    'confidence': chunk_confidence,
                    'status': 'success',
                    'lsl_timestamp': chunk_detection_timestamp,
                    'prompt_timestamp': self.movement_request_lsl_time,  # Changed variable name
                    'detection_timestamp': self.detection_time
                }
                
                # Add result
                self.add_trial_result(latency_ms, chunk_confidence, 'Success')
                self.session_data['trials'].append(trial_data)
                
                # End this trial
                self.trial_running = False
                
                # Move to next trial after a delay
                QtCore.QTimer.singleShot(2000, self.start_trial)
                
        except Exception as e:
            print(f"Error in process_eeg_data: {e}")
            import traceback
            traceback.print_exc()
    
    def finish_test(self):
        """Complete the test and show final results"""
        self.test_running = False
        self.session_data['end_time'] = datetime.datetime.now().isoformat()
        
        # Calculate final statistics
        successful_trials = [t for t in self.trial_results if t[0] >= 0]
        
        if successful_trials:
            latencies = [t[0] for t in successful_trials]
            avg_latency = np.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            std_dev = np.std(latencies) if len(latencies) > 1 else 0
            
            success_rate = len(successful_trials) / len(self.trial_results) * 100
            
            result_text = (
                f"Movement Latency Test Complete!\n\n"
                f"Successful Trials: {len(successful_trials)}/{len(self.trial_results)}\n"
                f"Success Rate: {success_rate:.1f}%\n\n"
                f"Results Summary:\n"
                f"• Average Latency: {avg_latency:.1f} ms\n"
                f"• Minimum Latency: {min_latency:.1f} ms\n"
                f"• Maximum Latency: {max_latency:.1f} ms\n"
                f"• Standard Deviation: {std_dev:.1f} ms\n\n"
                f"This measures hardware transmission latency\n"
                f"from Muse sensors to your computer via EEG movement detection."
            )
        else:
            result_text = (
                f"Test Complete!\n\n"
                f"No successful movements were detected.\n\n"
                f"Suggestions:\n"
                f"• Lower the threshold factor (currently {self.threshold_spinbox.value()})\n"
                f"• Make larger, more deliberate movements\n"
                f"• Check headset positioning\n"
                f"• Try tapping or adjusting the headband more firmly"
            )
        
        self.instruction_label.setText(result_text)
        self.instruction_label.setStyleSheet(
            "margin: 20px 0; padding: 20px; background-color: #e3f2fd; "
            "border-radius: 10px; color: #0d47a1;"
        )
        self.countdown_display.setText("COMPLETE")
        self.countdown_display.setStyleSheet("color: #1976d2; font-weight: bold;")
        
        # Re-enable buttons
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        print(f"Movement latency test completed. {len(successful_trials)}/{len(self.trial_results)} successful trials.")
    
    def save_results(self):
        """Save the enhanced latency test results"""
        if not self.trial_results:
            QtWidgets.QMessageBox.warning(self, "No Data", "No test results to save.")
            return
        
        try:
            # Create timestamped filename
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"eeg_movement_latency_test_{timestamp_str}"
            
            # Create organized folder structure
            test_dir = self.output_dir / timestamp_str
            test_dir.mkdir(exist_ok=True)
            
            # Save detailed results as CSV
            csv_path = test_dir / f"{filename_base}.csv"
            with open(csv_path, 'w') as f:
                f.write("Trial,Latency_ms,Confidence,Status\n")
                for i, (latency, confidence, status) in enumerate(self.trial_results):
                    latency_str = f"{latency:.1f}" if latency >= 0 else "timeout"
                    f.write(f"{i+1},{latency_str},{confidence:.1f},{status}\n")
            
            # Save session metadata as JSON
            json_path = test_dir / f"{filename_base}_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
            
            # Generate summary statistics
            successful_trials = [t for t in self.trial_results if t[0] >= 0]
            summary = {
                'test_info': {
                    'user': 'up202107351',
                    'test_date': datetime.datetime.now().isoformat(),
                    'test_type': 'EEG Hardware Latency via Movement Detection',
                    'version': '3.0'
                },
                'hardware_info': {
                    'sampling_rate_hz': self.sampling_rate,
                    'channel_count': self.channel_count,
                    'stream_name': self.session_data['system_info'].get('stream_name', 'Unknown')
                },
                'test_parameters': {
                    'threshold_factor': self.movement_detector.threshold_multiplier,
                    'baseline_duration_s': self.baseline_duration_spinbox.value(),
                    'filter_lowcut_hz': self.movement_detector.lowcut,
                    'filter_highcut_hz': self.movement_detector.highcut,
                    'detection_channels': ['TP9', 'AF7', 'AF8', 'TP10']
                },
                'results': {
                    'total_trials': len(self.trial_results),
                    'successful_trials': len(successful_trials),
                    'success_rate_percent': len(successful_trials) / len(self.trial_results) * 100,
                    'average_latency_ms': np.mean([t[0] for t in successful_trials]) if successful_trials else None,
                    'std_dev_latency_ms': np.std([t[0] for t in successful_trials]) if successful_trials else None,
                    'min_latency_ms': min([t[0] for t in successful_trials]) if successful_trials else None,
                    'max_latency_ms': max([t[0] for t in successful_trials]) if successful_trials else None,
                }
            }
            
            # Save summary
            summary_path = test_dir / f"{filename_base}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Show success message
            if successful_trials:
                avg_lat = np.mean([t[0] for t in successful_trials])
                std_lat = np.std([t[0] for t in successful_trials])
                success_msg = f"""Results saved successfully!

Files created in: {test_dir}

• Detailed results: {csv_path.name}
• Session metadata: {json_path.name}
• Summary statistics: {summary_path.name}

Test Summary:
• Total trials: {len(self.trial_results)}
• Successful: {len(successful_trials)} ({len(successful_trials)/len(self.trial_results)*100:.1f}%)
• Average latency: {avg_lat:.1f} ms (±{std_lat:.1f})
• Hardware: {self.sampling_rate} Hz EEG sampling
• Filter: {self.movement_detector.lowcut}-{self.movement_detector.highcut} Hz bandpass

This measures the time from Muse sensor detection to computer reception."""
            else:
                success_msg = f"""Results saved successfully!

Files created in: {test_dir}

• Detailed results: {csv_path.name}
• Session metadata: {json_path.name}
• Summary statistics: {summary_path.name}

Note: No successful trials were recorded.
Consider adjusting detection parameters."""
            
            QtWidgets.QMessageBox.information(self, "Save Complete", success_msg)
            print(f"EEG movement latency test results saved to: {test_dir}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving results: {str(e)}")
            print(f"Error saving results: {e}")
    
    def reset_test(self):
        """Reset the test completely"""
        self.test_running = False
        self.trial_running = False
        self.baseline_phase = False
        self.current_trial = 0
        self.trial_results = []
        
        # Stop any running timers
        if hasattr(self, 'countdown_timer') and self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
        if hasattr(self, 'baseline_countdown_timer') and self.baseline_countdown_timer and self.baseline_countdown_timer.isActive():
            self.baseline_countdown_timer.stop()
        
        # Reset detector
        if hasattr(self, 'movement_detector'):
            self.movement_detector.reset_baseline()
        
        # Clear UI
        self.results_table.setRowCount(0)
        self.update_statistics_display()
        self.countdown_display.setText("")
        self.countdown_display.setStyleSheet("")
        
        self.instruction_label.setText(
            "Ready to start movement latency test.\n\n"
            "The test will automatically:\n"
            "1. Calculate baseline (filtered EEG analysis)\n"
            "2. Run movement detection trials\n\n"
            "Both raw and filtered data are displayed in the plot."
        )
        self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        
        self.detection_status.setText("Detection: Ready for baseline")
        self.detection_status.setStyleSheet("color: blue; font-weight: bold")
        
        # Clear plot markers
        self.eeg_plot.clear_markers()
        
        # Enable buttons
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.save_button.setEnabled(False)
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if hasattr(self, 'countdown_timer') and self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            
        if hasattr(self, 'baseline_countdown_timer') and self.baseline_countdown_timer and self.baseline_countdown_timer.isActive():
            self.baseline_countdown_timer.stop()
            
        if self.inlet:
            try:
                self.inlet.close_stream()
            except:
                pass
            
        event.accept()


def main():
    """Main entry point for the enhanced EEG movement latency test"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Enhanced Muse EEG Movement Latency Test")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("NeuroFlow")
    
    # Create and show the main window
    window = EnhancedLatencyTestWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()