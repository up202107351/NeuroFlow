import sys
import time
import numpy as np
import pylsl
from PyQt5 import QtWidgets, QtCore, QtGui
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import datetime
from pathlib import Path
import json

# Configuration to match main GUI
EXPECTED_ACC_RATE = 52.0  # Match main GUI expected rate
STREAM_TYPE = 'Accelerometer'  # Match main GUI stream type
LATENCY_TEST_DIR = "latency_test_data"

class MovementDetector:
    def __init__(self, window_size=0.5, threshold=1.5, min_samples=5):
        """
        Enhanced movement detector for accelerometer data - matches main GUI approach
        
        Args:
            window_size: Size of analysis window in seconds
            threshold: Acceleration threshold as factor of baseline std dev
            min_samples: Minimum samples exceeding threshold to confirm movement
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        
        # Buffers for accelerometer data (x, y, z) - match main GUI buffer size
        max_buffer_size = int(window_size * EXPECTED_ACC_RATE * 20)  # 20 seconds worth
        self.acc_buffer_x = deque(maxlen=max_buffer_size)
        self.acc_buffer_y = deque(maxlen=max_buffer_size)
        self.acc_buffer_z = deque(maxlen=max_buffer_size)
        self.timestamps = deque(maxlen=max_buffer_size)
        
        # State variables
        self.baseline_calculated = False
        self.baseline_mean = [0, 0, 0]
        self.baseline_std = [0, 0, 0]
        self.last_movement_time = 0
        self.debounce_time = 0.5  # Seconds between possible movements
        self.in_movement = False
        
        # Enhanced detection parameters
        self.min_baseline_samples = int(EXPECTED_ACC_RATE * 2)  # 2 seconds of data
        
    def update(self, acc_data, timestamp):
        """
        Update the detector with new accelerometer data - enhanced version
        
        Args:
            acc_data: List or array of accelerometer values [x, y, z]
            timestamp: LSL timestamp for this sample
            
        Returns:
            (movement_detected, magnitude, timestamp, confidence) tuple
        """
        # Add data to buffers
        self.acc_buffer_x.append(acc_data[0])
        self.acc_buffer_y.append(acc_data[1])
        self.acc_buffer_z.append(acc_data[2])
        self.timestamps.append(timestamp)
        
        # Wait until we have enough data for baseline
        if len(self.acc_buffer_x) < self.min_baseline_samples:
            return False, 0, timestamp, 0.0
            
        # Calculate baseline if needed
        if not self.baseline_calculated:
            self.calculate_baseline()
            return False, 0, timestamp, 0.0
        
        # Check for movement
        movement, magnitude, confidence = self.detect_movement()
        
        # Debounce to avoid multiple detections of same movement
        current_time = time.time()
        if movement and not self.in_movement and current_time - self.last_movement_time > self.debounce_time:
            self.last_movement_time = current_time
            self.in_movement = True
            return True, magnitude, timestamp, confidence
        
        # Reset movement state once acceleration returns to near baseline
        if self.in_movement:
            # Calculate recent average acceleration magnitude
            recent_samples = 5
            recent_x = list(self.acc_buffer_x)[-recent_samples:]
            recent_y = list(self.acc_buffer_y)[-recent_samples:]
            recent_z = list(self.acc_buffer_z)[-recent_samples:]
            
            recent_magnitude = np.mean(np.sqrt(
                [(x-self.baseline_mean[0])**2 + 
                 (y-self.baseline_mean[1])**2 + 
                 (z-self.baseline_mean[2])**2 
                 for x, y, z in zip(recent_x, recent_y, recent_z)]
            ))
            
            if recent_magnitude < self.threshold * np.mean(self.baseline_std) * 0.5:
                self.in_movement = False
            
        return False, 0, timestamp, 0.0
        
    def calculate_baseline(self):
        """Calculate baseline accelerometer values - enhanced version"""
        if len(self.acc_buffer_x) < self.min_baseline_samples:
            return
            
        x_array = np.array(self.acc_buffer_x)
        y_array = np.array(self.acc_buffer_y)
        z_array = np.array(self.acc_buffer_z)
        
        # Get mean and std for each axis
        self.baseline_mean = [
            np.mean(x_array),
            np.mean(y_array),
            np.mean(z_array)
        ]
        
        self.baseline_std = [
            max(0.01, np.std(x_array)),  # Prevent division by zero
            max(0.01, np.std(y_array)),
            max(0.01, np.std(z_array))
        ]
        
        self.baseline_calculated = True
        print(f"Baseline calculated from {len(x_array)} samples:")
        print(f"  Mean: [{self.baseline_mean[0]:.3f}, {self.baseline_mean[1]:.3f}, {self.baseline_mean[2]:.3f}]")
        print(f"  StdDev: [{self.baseline_std[0]:.3f}, {self.baseline_std[1]:.3f}, {self.baseline_std[2]:.3f}]")
    
    def detect_movement(self):
        """
        Enhanced movement detection with confidence scoring
        
        Returns:
            (movement_detected, magnitude, confidence) tuple
        """
        # Get recent samples for analysis
        samples_to_check = min(int(EXPECTED_ACC_RATE * 0.2), 15)  # ~200ms of data
        recent_x = list(self.acc_buffer_x)[-samples_to_check:]
        recent_y = list(self.acc_buffer_y)[-samples_to_check:]
        recent_z = list(self.acc_buffer_z)[-samples_to_check:]
        
        # Calculate z-scores for each axis
        z_x = [(x - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_x]
        z_y = [(y - self.baseline_mean[1]) / self.baseline_std[1] for y in recent_y]
        z_z = [(z - self.baseline_mean[2]) / self.baseline_std[2] for z in recent_z]
        
        # Calculate magnitude of deviation for each sample
        z_magnitudes = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(z_x, z_y, z_z)]
        
        # Count samples exceeding threshold
        exceeding_samples = sum(1 for mag in z_magnitudes if mag > self.threshold)
        max_magnitude = max(z_magnitudes)
        avg_magnitude = np.mean(z_magnitudes)
        
        # Calculate confidence based on consistency and magnitude
        consistency = exceeding_samples / len(z_magnitudes)
        magnitude_factor = min(1.0, max_magnitude / (self.threshold * 2))
        confidence = (consistency * 0.7 + magnitude_factor * 0.3)
        
        # Return True if enough samples exceed threshold
        movement_detected = exceeding_samples >= self.min_samples
        
        return movement_detected, max_magnitude, confidence
        
    def reset_baseline(self):
        """Reset the baseline calculation"""
        self.baseline_calculated = False
        self.acc_buffer_x.clear()
        self.acc_buffer_y.clear()
        self.acc_buffer_z.clear()
        self.timestamps.clear()
        self.in_movement = False
        
    def get_stats(self):
        """Get current detector statistics"""
        return {
            'baseline_calculated': self.baseline_calculated,
            'baseline_mean': self.baseline_mean.copy() if self.baseline_calculated else None,
            'baseline_std': self.baseline_std.copy() if self.baseline_calculated else None,
            'buffer_size': len(self.acc_buffer_x),
            'threshold': self.threshold,
            'min_samples': self.min_samples
        }


class AccelerometerPlot(FigureCanvas):
    """Enhanced live plot of accelerometer data"""
    def __init__(self, parent=None, width=5, height=3, dpi=100, buffer_size=500):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
        super(AccelerometerPlot, self).__init__(self.fig)
        
        self.buffer_size = buffer_size
        self.time_data = np.linspace(-10, 0, buffer_size)  # Last 10 seconds
        self.x_data = np.zeros(buffer_size)
        self.y_data = np.zeros(buffer_size)
        self.z_data = np.zeros(buffer_size)
        self.magnitude_data = np.zeros(buffer_size)
        
        # Top plot: Raw accelerometer data
        ax1 = self.axes[0]
        self.line_x, = ax1.plot(self.time_data, self.x_data, 'r-', alpha=0.7, label='X', linewidth=1.5)
        self.line_y, = ax1.plot(self.time_data, self.y_data, 'g-', alpha=0.7, label='Y', linewidth=1.5)
        self.line_z, = ax1.plot(self.time_data, self.z_data, 'b-', alpha=0.7, label='Z', linewidth=1.5)
        
        ax1.set_ylim(-3, 3)
        ax1.set_ylabel('Acceleration (g)')
        ax1.set_title('Raw Accelerometer Data')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Movement magnitude
        ax2 = self.axes[1]
        self.line_mag, = ax2.plot(self.time_data, self.magnitude_data, 'k-', label='Movement Magnitude', linewidth=2)
        self.threshold_line = None
        
        ax2.set_ylim(0, 5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Movement Detection')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Event markers
        self.prompt_lines = []
        self.detection_lines = []
        
        self.fig.tight_layout()
        
    def update_plot(self, x, y, z, baseline_mean=None, threshold=None):
        """Update plot with new data"""
        # Shift data arrays
        self.x_data[:-1] = self.x_data[1:]
        self.y_data[:-1] = self.y_data[1:]
        self.z_data[:-1] = self.z_data[1:]
        self.magnitude_data[:-1] = self.magnitude_data[1:]
        
        # Add new data
        self.x_data[-1] = x
        self.y_data[-1] = y
        self.z_data[-1] = z
        
        # Calculate magnitude (relative to baseline if available)
        if baseline_mean:
            x_rel = x - baseline_mean[0]
            y_rel = y - baseline_mean[1]
            z_rel = z - baseline_mean[2]
            self.magnitude_data[-1] = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        else:
            self.magnitude_data[-1] = np.sqrt(x**2 + y**2 + z**2)
        
        # Update plot data
        self.line_x.set_ydata(self.x_data)
        self.line_y.set_ydata(self.y_data)
        self.line_z.set_ydata(self.z_data)
        self.line_mag.set_ydata(self.magnitude_data)
        
        # Update threshold line if provided
        if threshold and baseline_mean:
            if self.threshold_line:
                self.threshold_line.remove()
            baseline_std = np.std([x, y, z])  # Rough estimate
            threshold_value = threshold * baseline_std
            self.threshold_line = self.axes[1].axhline(y=threshold_value, color='r', 
                                                    linestyle='--', alpha=0.7, 
                                                    label=f'Threshold ({threshold_value:.2f})')
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def add_prompt_marker(self):
        """Add vertical line to mark when prompt was shown"""
        for line in self.prompt_lines:
            line.remove()
        self.prompt_lines.clear()
        
        for ax in self.axes:
            line = ax.axvline(x=0, color='m', linestyle='-', alpha=0.8, linewidth=2, label='Prompt')
            self.prompt_lines.append(line)
        
        self.fig.canvas.draw_idle()
        
    def add_detection_marker(self):
        """Add marker for movement detection"""
        for line in self.detection_lines:
            line.remove()
        self.detection_lines.clear()
        
        for ax in self.axes:
            line = ax.axvline(x=0, color='g', linestyle=':', alpha=0.8, linewidth=2, label='Detection')
            self.detection_lines.append(line)
            
        self.fig.canvas.draw_idle()
        
    def clear_markers(self):
        """Remove all markers"""
        for line in self.prompt_lines + self.detection_lines:
            line.remove()
        self.prompt_lines.clear()
        self.detection_lines.clear()
        self.fig.canvas.draw_idle()


class LatencyTestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse Accelerometer Latency Test - Enhanced v2.0")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create output directory (matching main GUI approach)
        self.output_dir = Path(LATENCY_TEST_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        # LSL connection state - updated to match main GUI
        self.acc_inlet = None
        self.connected = False
        self.actual_sampling_rate = EXPECTED_ACC_RATE
        
        # Test parameters
        self.num_trials = 10
        self.current_trial = 0
        self.trial_results = []
        self.movement_request_time = None
        self.test_running = False
        self.trial_running = False
        
        # Enhanced movement detector
        self.movement_detector = MovementDetector()
        
        # Session data for enhanced reporting
        self.session_data = {
            'start_time': None,
            'end_time': None,
            'trials': [],
            'detector_stats': {},
            'system_info': {
                'expected_rate': EXPECTED_ACC_RATE,
                'actual_rate': None,
                'stream_type': STREAM_TYPE
            }
        }
        
        # Setup UI
        self.setup_ui()
        
        # Setup LSL inlet timer
        self.lsl_timer = QtCore.QTimer()
        self.lsl_timer.timeout.connect(self.process_acc_data)
        self.lsl_timer.setInterval(20)  # Process data every 20ms
        
        # Setup result variables
        self.movement_detected = False
        self.detection_time = None
        self.countdown_timer = None

    def setup_ui(self):
        """Set up the enhanced user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Enhanced title
        title_label = QtWidgets.QLabel("Muse Accelerometer Latency Test - Enhanced v2.0")
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Status indicators with more info
        status_layout = QtWidgets.QHBoxLayout()
        
        self.connection_status = QtWidgets.QLabel("Status: Not Connected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        status_layout.addWidget(self.connection_status)
        
        self.sample_rate_label = QtWidgets.QLabel("Sample Rate: N/A")
        status_layout.addWidget(self.sample_rate_label)
        
        self.baseline_status = QtWidgets.QLabel("Baseline: Not Calculated")
        status_layout.addWidget(self.baseline_status)
        
        self.output_dir_label = QtWidgets.QLabel(f"Output: {self.output_dir}")
        self.output_dir_label.setStyleSheet("color: blue; font-size: 10px;")
        status_layout.addWidget(self.output_dir_label)
        
        main_layout.addLayout(status_layout)
        
        # Enhanced plot widget
        self.plot_widget = AccelerometerPlot(width=10, height=4)
        main_layout.addWidget(self.plot_widget)
        
        # Main instruction display
        self.instruction_label = QtWidgets.QLabel("Click 'Connect' to start")
        self.instruction_label.setFont(QtGui.QFont("Arial", 16))
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
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        
        self.results_table = QtWidgets.QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Trial #", "Latency (ms)", "Magnitude", "Confidence"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        # Enhanced statistics display
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
        
        # Enhanced options
        options_layout = QtWidgets.QHBoxLayout()
        
        options_layout.addWidget(QtWidgets.QLabel("Number of Trials:"))
        self.trials_spinbox = QtWidgets.QSpinBox()
        self.trials_spinbox.setRange(1, 50)
        self.trials_spinbox.setValue(10)
        self.trials_spinbox.valueChanged.connect(self.update_trial_count)
        options_layout.addWidget(self.trials_spinbox)
        
        options_layout.addWidget(QtWidgets.QLabel("Movement Threshold:"))
        self.threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.threshold_spinbox.setRange(1.0, 10.0)
        self.threshold_spinbox.setValue(1.5)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.valueChanged.connect(self.update_threshold)
        options_layout.addWidget(self.threshold_spinbox)
        
        options_layout.addWidget(QtWidgets.QLabel("Movement Type:"))
        self.movement_combo = QtWidgets.QComboBox()
        self.movement_combo.addItems(["Nod Head", "Tap Headset", "Tilt Head", "Shake Head"])
        options_layout.addWidget(self.movement_combo)
        
        main_layout.addLayout(options_layout)
        
    def toggle_connection(self):
        """Connect to or disconnect from the Muse headset via LSL - updated to match main GUI"""
        if not self.connected:
            self.connect_to_muse()
        else:
            self.disconnect_from_muse()
    
    def connect_to_muse(self):
        """Connect to Muse headset accelerometer via LSL - updated to match main GUI approach"""
        try:
            self.connection_status.setText("Status: Searching for accelerometer stream...")
            self.connection_status.setStyleSheet("color: orange; font-weight: bold")
            QtWidgets.QApplication.processEvents()
            
            # Updated to match main GUI approach
            print("Looking for accelerometer stream...")
            streams = pylsl.resolve_byprop('type', STREAM_TYPE, 1, timeout=3.0)
            
            if not streams:
                self.connection_status.setText("Status: No accelerometer stream found")
                self.connection_status.setStyleSheet("color: red; font-weight: bold")
                QtWidgets.QMessageBox.warning(self, "Connection Error", 
                                          "No accelerometer stream found. Make sure Muse is streaming data.")
                return
                
            # Create an inlet from the first stream
            self.acc_inlet = pylsl.StreamInlet(streams[0], max_chunklen=128)
            
            # Get stream info
            info = self.acc_inlet.info()
            self.actual_sampling_rate = info.nominal_srate()
            if self.actual_sampling_rate <= 0:
                self.actual_sampling_rate = EXPECTED_ACC_RATE
            
            # Update UI
            self.connection_status.setText(f"Status: Connected to {info.name()}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold")
            self.sample_rate_label.setText(f"Sample Rate: {self.actual_sampling_rate} Hz")
            
            # Update session data
            self.session_data['system_info']['actual_rate'] = self.actual_sampling_rate
            
            # Start processing accelerometer data
            self.connected = True
            self.connect_button.setText("Disconnect")
            self.start_button.setEnabled(True)
            self.lsl_timer.start()
            
            # Create new movement detector with correct sampling rate
            self.movement_detector = MovementDetector()
            
            # Update instructions
            self.instruction_label.setText("Ready to start latency test.\n\nYou will be asked to move your head when prompted.")
            
            print(f"Connected to accelerometer stream: {info.name()} at {self.actual_sampling_rate} Hz")
            
        except Exception as e:
            self.connection_status.setText(f"Status: Error - {str(e)}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold")
            QtWidgets.QMessageBox.critical(self, "Connection Error", f"Error: {str(e)}")
    
    def disconnect_from_muse(self):
        """Disconnect from the Muse headset"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if self.acc_inlet:
            try:
                self.acc_inlet.close_stream()
            except:
                pass
            self.acc_inlet = None
        
        self.connected = False
        self.connect_button.setText("Connect to Muse")
        self.start_button.setEnabled(False)
        
        self.connection_status.setText("Status: Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        self.baseline_status.setText("Baseline: Not Calculated")
        
    def update_trial_count(self, value):
        """Update the number of trials"""
        self.num_trials = value
        
    def update_threshold(self, value):
        """Update the movement detection threshold"""
        if hasattr(self, 'movement_detector'):
            self.movement_detector.threshold = value
    
    def start_test(self):
        """Start the enhanced latency test"""
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", 
                                      "Please connect to Muse headset first.")
            return
            
        self.test_running = True
        self.current_trial = 0
        self.trial_results = []
        
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
        
        # Reset baseline calculation
        self.movement_detector.reset_baseline()
        self.baseline_status.setText("Baseline: Calculating...")
        
        # Start first trial after collecting baseline data
        self.instruction_label.setText(
            "Getting baseline readings.\n\n"
            "Please hold still for a few seconds."
        )
        
        # Start first trial after a delay for baseline collection
        QtCore.QTimer.singleShot(5000, self.start_trial)
        
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
        
        # Clear previous markers
        self.plot_widget.clear_markers()
        
        # Get selected movement type
        movement_type = self.movement_combo.currentText()
        
        # Update baseline status
        if self.movement_detector.baseline_calculated:
            self.baseline_status.setText("Baseline: Calculated")
            self.baseline_status.setStyleSheet("color: green; font-weight: bold")
        
        # Show instructions
        self.instruction_label.setText(
            f"Trial {self.current_trial} of {self.num_trials}\n\n"
            f"When prompted, {movement_type.lower()} once quickly."
        )
        
        # Start countdown to prompt
        self.start_countdown(random.randint(3, 6))  # Random delay between 3-6 seconds
        
    def start_countdown(self, seconds):
        """Start countdown timer before prompting for movement"""
        self.countdown_display.setText(str(seconds))
        
        self.trial_running = True
        
        # Create countdown timer
        self.countdown_timer = QtCore.QTimer()
        remaining_time = seconds
        
        def update_countdown():