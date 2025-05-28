import sys
import time
import numpy as np
import pylsl
from PyQt5 import QtWidgets, QtCore, QtGui
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MovementDetector:
    def __init__(self, window_size=0.5, threshold=1.5, min_samples=5):
        """
        Movement detector for accelerometer data
        
        Args:
            window_size: Size of analysis window in seconds
            threshold: Acceleration threshold as factor of baseline std dev
            min_samples: Minimum samples exceeding threshold to confirm movement
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        
        # Buffers for accelerometer data (x, y, z)
        self.acc_buffer_x = deque(maxlen=500)  # Should be enough for ~10s at 50Hz
        self.acc_buffer_y = deque(maxlen=500)
        self.acc_buffer_z = deque(maxlen=500)
        
        # State variables
        self.baseline_calculated = False
        self.baseline_mean = [0, 0, 0]
        self.baseline_std = [0, 0, 0]
        self.last_movement_time = 0
        self.debounce_time = 0.5  # Seconds between possible movements
        self.in_movement = False
        
    def update(self, acc_data, timestamp):
        """
        Update the detector with new accelerometer data
        
        Args:
            acc_data: List or array of accelerometer values [x, y, z]
            timestamp: LSL timestamp for this sample
            
        Returns:
            (movement_detected, magnitude, timestamp) tuple
        """
        # Add data to buffers
        self.acc_buffer_x.append(acc_data[0])
        self.acc_buffer_y.append(acc_data[1])
        self.acc_buffer_z.append(acc_data[2])
        
        # Wait until we have enough data
        if len(self.acc_buffer_x) < 50:  # Need at least ~1 second of data at 50Hz
            return False, 0, timestamp
            
        # Calculate baseline if needed
        if not self.baseline_calculated:
            self.calculate_baseline()
            return False, 0, timestamp
        
        # Check for movement
        movement, magnitude = self.detect_movement()
        
        # Debounce to avoid multiple detections of same movement
        current_time = time.time()
        if movement and not self.in_movement and current_time - self.last_movement_time > self.debounce_time:
            self.last_movement_time = current_time
            self.in_movement = True
            return True, magnitude, timestamp
        
        # Reset movement state once acceleration returns to near baseline
        if self.in_movement:
            # Calculate recent average acceleration magnitude
            recent_x = list(self.acc_buffer_x)[-5:]
            recent_y = list(self.acc_buffer_y)[-5:]
            recent_z = list(self.acc_buffer_z)[-5:]
            
            recent_magnitude = np.mean(np.sqrt(
                [(x-self.baseline_mean[0])**2 + 
                 (y-self.baseline_mean[1])**2 + 
                 (z-self.baseline_mean[2])**2 
                 for x, y, z in zip(recent_x, recent_y, recent_z)]
            ))
            
            if recent_magnitude < self.threshold * np.mean(self.baseline_std) * 0.5:
                self.in_movement = False
            
        return False, 0, timestamp
        
    def calculate_baseline(self):
        """Calculate baseline accelerometer values for movement detection"""
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
            np.std(x_array),
            np.std(y_array),
            np.std(z_array)
        ]
        
        self.baseline_calculated = True
        print(f"Baseline calculated: Mean=[{self.baseline_mean[0]:.2f}, {self.baseline_mean[1]:.2f}, {self.baseline_mean[2]:.2f}]")
        print(f"Baseline calculated: StdDev=[{self.baseline_std[0]:.2f}, {self.baseline_std[1]:.2f}, {self.baseline_std[2]:.2f}]")
    
    def detect_movement(self):
        """
        Detect if movement occurred in the latest data
        
        Returns:
            (movement_detected, magnitude) tuple
        """
        # Get recent samples
        samples_to_check = 10  # Look at last ~200ms of data (assuming 50Hz)
        recent_x = list(self.acc_buffer_x)[-samples_to_check:]
        recent_y = list(self.acc_buffer_y)[-samples_to_check:]
        recent_z = list(self.acc_buffer_z)[-samples_to_check:]
        
        # Calculate z-scores for each axis
        z_x = [(x - self.baseline_mean[0]) / max(0.01, self.baseline_std[0]) for x in recent_x]
        z_y = [(y - self.baseline_mean[1]) / max(0.01, self.baseline_std[1]) for y in recent_y]
        z_z = [(z - self.baseline_mean[2]) / max(0.01, self.baseline_std[2]) for z in recent_z]
        
        # Calculate magnitude of deviation for each sample
        z_magnitudes = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(z_x, z_y, z_z)]
        
        # Count samples exceeding threshold
        exceeding_samples = sum(1 for mag in z_magnitudes if mag > self.threshold)
        max_magnitude = max(z_magnitudes)
        
        # Return True if enough samples exceed threshold
        if exceeding_samples >= self.min_samples:
            return True, max_magnitude
        else:
            return False, max_magnitude
        
    def reset_baseline(self):
        """Reset the baseline calculation"""
        self.baseline_calculated = False
        self.acc_buffer_x.clear()
        self.acc_buffer_y.clear()
        self.acc_buffer_z.clear()
        self.in_movement = False


class AccelerometerPlot(FigureCanvas):
    """Live plot of accelerometer data"""
    def __init__(self, parent=None, width=5, height=3, dpi=100, buffer_size=500):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(AccelerometerPlot, self).__init__(self.fig)
        
        self.buffer_size = buffer_size
        self.time_data = np.linspace(-10, 0, buffer_size)  # Last 10 seconds
        self.x_data = np.zeros(buffer_size)
        self.y_data = np.zeros(buffer_size)
        self.z_data = np.zeros(buffer_size)
        self.magnitude_data = np.zeros(buffer_size)
        
        # Set up plot lines
        self.line_x, = self.ax.plot(self.time_data, self.x_data, 'r-', alpha=0.7, label='X')
        self.line_y, = self.ax.plot(self.time_data, self.y_data, 'g-', alpha=0.7, label='Y')
        self.line_z, = self.ax.plot(self.time_data, self.z_data, 'b-', alpha=0.7, label='Z')
        self.line_mag, = self.ax.plot(self.time_data, self.magnitude_data, 'k-', label='Magnitude')
        
        # Add prompt markers
        self.prompt_line = None
        
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Acceleration (g)')
        self.ax.set_title('Accelerometer Data')
        self.ax.legend(loc='upper left')
        self.ax.grid(True)
        self.fig.tight_layout()
        
    def update_plot(self, x, y, z, baseline_mean=None):
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
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def add_prompt_marker(self):
        """Add vertical line to mark when prompt was shown"""
        if self.prompt_line:
            self.prompt_line.remove()
        self.prompt_line = self.ax.axvline(x=0, color='m', linestyle='-', alpha=0.7)
        self.fig.canvas.draw_idle()
        
    def clear_prompt_marker(self):
        """Remove prompt marker"""
        if self.prompt_line:
            self.prompt_line.remove()
            self.prompt_line = None
            self.fig.canvas.draw_idle()


class LatencyTestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse Accelerometer Latency Test")
        self.setGeometry(100, 100, 900, 700)
        
        # LSL connection state
        self.acc_inlet = None
        self.connected = False
        self.sampling_rate = 50.0  # Default for Muse accelerometer
        
        # Test parameters
        self.num_trials = 10
        self.current_trial = 0
        self.trial_results = []
        self.movement_request_time = None
        self.test_running = False
        self.trial_running = False
        
        # Movement detector
        self.movement_detector = MovementDetector()
        
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
        """Set up the user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Title
        title_label = QtWidgets.QLabel("Muse Accelerometer Latency Test")
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Status indicators
        status_layout = QtWidgets.QHBoxLayout()
        self.connection_status = QtWidgets.QLabel("Status: Not Connected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        status_layout.addWidget(self.connection_status)
        
        self.sample_rate_label = QtWidgets.QLabel("Sample Rate: N/A")
        status_layout.addWidget(self.sample_rate_label)
        main_layout.addLayout(status_layout)
        
        # Create plot widget
        self.plot_widget = AccelerometerPlot(width=8, height=3)
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
        
        # Results display
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        
        self.results_table = QtWidgets.QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Trial #", "Latency (ms)", "Magnitude"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        self.avg_latency_label = QtWidgets.QLabel("Average Latency: N/A")
        self.avg_latency_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        results_layout.addWidget(self.avg_latency_label)
        
        main_layout.addWidget(results_group)
        
        # Buttons
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
        
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)
        
        # Options
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
        self.movement_combo.addItems(["Nod Head", "Tap Headset", "Tilt Head"])
        options_layout.addWidget(self.movement_combo)
        
        main_layout.addLayout(options_layout)
        
    def toggle_connection(self):
        """Connect to or disconnect from the Muse headset via LSL"""
        if not self.connected:
            self.connect_to_muse()
        else:
            self.disconnect_from_muse()
    
    def connect_to_muse(self):
        """Connect to Muse headset accelerometer via LSL"""
        try:
            self.connection_status.setText("Status: Searching for accelerometer stream...")
            self.connection_status.setStyleSheet("color: orange; font-weight: bold")
            QtWidgets.QApplication.processEvents()
            
            # Look for accelerometer streams
            print("Looking for accelerometer stream...")
            streams = pylsl.resolve_byprop('type', 'Accelerometer', timeout=5)
            
            if not streams:
                self.connection_status.setText("Status: No accelerometer stream found")
                self.connection_status.setStyleSheet("color: red; font-weight: bold")
                QtWidgets.QMessageBox.warning(self, "Connection Error", 
                                          "No accelerometer stream found. Make sure Muse is streaming data.")
                return
                
            # Create an inlet from the first stream
            self.acc_inlet = pylsl.StreamInlet(streams[0])
            
            # Get stream info
            info = self.acc_inlet.info()
            self.sampling_rate = info.nominal_srate()
            
            # Update UI
            self.connection_status.setText(f"Status: Connected to {info.name()}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold")
            self.sample_rate_label.setText(f"Sample Rate: {self.sampling_rate} Hz")
            
            # Start processing accelerometer data
            self.connected = True
            self.connect_button.setText("Disconnect")
            self.start_button.setEnabled(True)
            self.lsl_timer.start()
            
            # Create new movement detector with correct sampling rate
            self.movement_detector = MovementDetector()
            
            # Update instructions
            self.instruction_label.setText("Ready to start latency test.\n\nYou will be asked to move your head when prompted.")
            
        except Exception as e:
            self.connection_status.setText(f"Status: Error - {str(e)}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold")
            QtWidgets.QMessageBox.critical(self, "Connection Error", f"Error: {str(e)}")
    
    def disconnect_from_muse(self):
        """Disconnect from the Muse headset"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if self.acc_inlet:
            self.acc_inlet.close_stream()
            self.acc_inlet = None
        
        self.connected = False
        self.connect_button.setText("Connect to Muse")
        self.start_button.setEnabled(False)
        
        self.connection_status.setText("Status: Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        
    def update_trial_count(self, value):
        """Update the number of trials"""
        self.num_trials = value
        
    def update_threshold(self, value):
        """Update the movement detection threshold"""
        if hasattr(self, 'movement_detector'):
            self.movement_detector.threshold = value
    
    def start_test(self):
        """Start the latency test"""
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", 
                                      "Please connect to Muse headset first.")
            return
            
        self.test_running = True
        self.current_trial = 0
        self.trial_results = []
        
        # Clear results table
        self.results_table.setRowCount(0)
        self.avg_latency_label.setText("Average Latency: N/A")
        
        # Update UI
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        
        # Reset baseline calculation
        self.movement_detector.reset_baseline()
        
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
        
        # Get selected movement type
        movement_type = self.movement_combo.currentText()
        
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
            nonlocal remaining_time
            remaining_time -= 1
            
            if remaining_time > 0:
                self.countdown_display.setText(str(remaining_time))
            else:
                self.countdown_timer.stop()
                self.countdown_display.setText("")
                self.prompt_for_movement()
                
        self.countdown_timer.timeout.connect(update_countdown)
        self.countdown_timer.start(1000)  # 1 second interval
        
    def prompt_for_movement(self):
        """Prompt the user to move and record the time"""
        movement_type = self.movement_combo.currentText().upper()
        
        self.instruction_label.setText(f"{movement_type} NOW!")
        self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #ffcccc; border-radius: 10px; font-size: 24pt; font-weight: bold;")
        
        # Record the time of the prompt
        self.movement_request_time = time.time()
        
        # Add a marker to the plot
        self.plot_widget.add_prompt_marker()
        
        # Set timeout for this trial
        QtCore.QTimer.singleShot(3000, self.check_trial_timeout)
        
    def check_trial_timeout(self):
        """Check if the trial has timed out"""
        if self.trial_running and not self.movement_detected:
            # Trial timed out
            self.trial_running = False
            self.instruction_label.setText("No movement detected. Moving to next trial...")
            self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
            
            # Clear any prompt markers
            self.plot_widget.clear_prompt_marker()
            
            # Add failed trial to results
            self.add_trial_result(-1, 0)  # -1 indicates timeout
            
            # Continue to next trial after delay
            QtCore.QTimer.singleShot(2000, self.start_trial)
            
    def add_trial_result(self, latency_ms, magnitude):
        """Add trial result to the table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Add data to table
        self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.current_trial)))
        
        if latency_ms < 0:
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem("Timeout"))
        else:
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{latency_ms:.1f}"))
            
        self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{magnitude:.2f}"))
        
        # Store result
        self.trial_results.append((latency_ms, magnitude))
        
        # Update average latency
        valid_latencies = [lat for lat, _ in self.trial_results if lat >= 0]
        if valid_latencies:
            avg_latency = sum(valid_latencies) / len(valid_latencies)
            self.avg_latency_label.setText(f"Average Latency: {avg_latency:.1f} ms")
    
    def process_acc_data(self):
        """Process incoming accelerometer data and detect movement"""
        if not self.connected or not self.acc_inlet:
            return
            
        # Get all available samples
        chunk, timestamps = self.acc_inlet.pull_chunk(timeout=0.0, max_samples=100)
        
        if not chunk:
            return
            
        # Process each sample
        for i, sample in enumerate(chunk):
            # Update plot (even if test not running)
            self.plot_widget.update_plot(
                sample[0], sample[1], sample[2], 
                self.movement_detector.baseline_mean if self.movement_detector.baseline_calculated else None
            )
            
            # Only look for movement if test is running
            if self.trial_running and self.movement_request_time and not self.movement_detected:
                # Process sample with movement detector
                is_movement, magnitude, sample_timestamp = self.movement_detector.update(sample, timestamps[i])
                
                if is_movement:
                    self.detection_time = time.time()
                    self.movement_detected = True
                    
                    # Calculate latency using both local time and LSL timestamp
                    system_latency_ms = (self.detection_time - self.movement_request_time) * 1000.0
                    
                    # Estimated user reaction time (time from prompt to when movement was picked up by device)
                    user_latency_ms = (timestamps[i] - self.movement_request_time) * 1000.0
                    
                    # Calculate processing latency (time from sample acquisition to detection)
                    processing_latency_ms = (self.detection_time - timestamps[i]) * 1000.0
                    
                    print(f"Movement detected! Total Latency: {system_latency_ms:.1f} ms")
                    print(f"- User reaction time: ~{user_latency_ms:.1f} ms")
                    print(f"- Processing latency: ~{processing_latency_ms:.1f} ms")
                    print(f"- Magnitude: {magnitude:.2f}")
                    
                    # Update UI
                    self.instruction_label.setText(
                        f"Movement detected!\n"
                        f"Total Latency: {system_latency_ms:.1f} ms\n"
                        f"User Reaction: ~{user_latency_ms:.1f} ms\n"
                        f"Processing: ~{processing_latency_ms:.1f} ms"
                    )
                    self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #ccffcc; border-radius: 10px;")
                    
                    # Add result (using total system latency)
                    self.add_trial_result(system_latency_ms, magnitude)
                    
                    # End this trial
                    self.trial_running = False
                    
                    # Clear any prompt markers after a short delay
                    QtCore.QTimer.singleShot(1000, self.plot_widget.clear_prompt_marker)
                    
                    # Move to next trial after a delay
                    QtCore.QTimer.singleShot(3000, self.start_trial)
    
    def finish_test(self):
        """Complete the test and show final results"""
        self.test_running = False
        
        # Calculate final statistics
        valid_latencies = [lat for lat, _ in self.trial_results if lat >= 0]
        
        if valid_latencies:
            avg_latency = sum(valid_latencies) / len(valid_latencies)
            min_latency = min(valid_latencies)
            max_latency = max(valid_latencies)
            std_dev = np.std(valid_latencies) if len(valid_latencies) > 1 else 0
            
            success_rate = len(valid_latencies) / len(self.trial_results) * 100
            
            result_text = (
                f"Test Complete!\n\n"
                f"Average Latency: {avg_latency:.1f} ms\n"
                f"Minimum Latency: {min_latency:.1f} ms\n"
                f"Maximum Latency: {max_latency:.1f} ms\n"
                f"Standard Deviation: {std_dev:.1f} ms\n\n"
                f"Success Rate: {success_rate:.1f}% ({len(valid_latencies)}/{len(self.trial_results)})"
            )
        else:
            result_text = "Test Complete, but no successful movements were detected."
        
        self.instruction_label.setText(result_text)
        self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        
        # Re-enable buttons
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.connect_button.setEnabled(True)
        
        # Offer to save results
        self.offer_save_results()
        
    def offer_save_results(self):
        """Ask user if they want to save the results to a file"""
        reply = QtWidgets.QMessageBox.question(self, 'Save Results', 
                                           'Would you like to save the results to a file?',
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                           QtWidgets.QMessageBox.Yes)
                                           
        if reply == QtWidgets.QMessageBox.Yes:
            self.save_results()
    
    def save_results(self):
        """Save results to a CSV file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)")
            
        if filename:
            try:
                with open(filename, 'w') as f:
                    # Get movement type
                    movement_type = self.movement_combo.currentText()
                    
                    # Write header and test info
                    f.write("Muse Accelerometer Latency Test Results\n")
                    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Movement Type: {movement_type}\n")
                    f.write(f"Threshold: {self.movement_detector.threshold}\n\n")
                    
                    # Write data header
                    f.write("Trial,Latency_ms,Magnitude\n")
                    
                    # Write data
                    for i, (latency, magnitude) in enumerate(self.trial_results):
                        latency_str = str(latency) if latency >= 0 else "Timeout"
                        f.write(f"{i+1},{latency_str},{magnitude}\n")
                    
                    # Write summary statistics
                    valid_latencies = [lat for lat, _ in self.trial_results if lat >= 0]
                    if valid_latencies:
                        avg_latency = sum(valid_latencies) / len(valid_latencies)
                        min_latency = min(valid_latencies)
                        max_latency = max(valid_latencies)
                        std_dev = np.std(valid_latencies) if len(valid_latencies) > 1 else 0
                        
                        f.write("\nSummary Statistics:\n")
                        f.write(f"Average Latency,{avg_latency}\n")
                        f.write(f"Minimum Latency,{min_latency}\n")
                        f.write(f"Maximum Latency,{max_latency}\n")
                        f.write(f"Standard Deviation,{std_dev}\n")
                        f.write(f"Success Rate,{len(valid_latencies)/len(self.trial_results)*100}%\n")
                    
                QtWidgets.QMessageBox.information(self, "Success", f"Results saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
    
    def reset_test(self):
        """Reset the test"""
        self.current_trial = 0
        self.trial_results = []
        
        # Clear results table
        self.results_table.setRowCount(0)
        self.avg_latency_label.setText("Average Latency: N/A")
        
        # Reset UI
        self.instruction_label.setText("Ready to start latency test.\n\nYou will be asked to move when prompted.")
        self.instruction_label.setStyleSheet("margin: 20px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        
        # Clear any prompt markers
        self.plot_widget.clear_prompt_marker()
        
    def closeEvent(self, event):
        """Clean up when window is closed"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            
        if self.acc_inlet:
            self.acc_inlet.close_stream()
            
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LatencyTestWindow()
    window.show()
    sys.exit(app.exec_())