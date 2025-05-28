import sys
import time
import numpy as np
import pylsl
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import signal
import matplotlib.pyplot as plt
from collections import deque
import random

class BlinkDetector:
    def __init__(self, sampling_rate=256.0, window_size=1.0, threshold_factor=3.5):
        """
        Blink detector for Muse EEG data
        
        Args:
            sampling_rate: The sampling rate of the EEG data
            window_size: Size of the window in seconds for analysis
            threshold_factor: Factor to multiply std dev for threshold calculation
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.buffer_size = int(window_size * sampling_rate)
        self.threshold_factor = threshold_factor
        
        # Buffer for frontal electrodes (AF7, AF8 on Muse headset)
        self.front_left_buffer = deque(maxlen=self.buffer_size)
        self.front_right_buffer = deque(maxlen=self.buffer_size)
        
        # State variables
        self.baseline_calculated = False
        self.baseline_values = None
        self.baseline_std = None
        self.last_blink_time = 0
        self.blink_debounce_time = 0.5  # Seconds between possible blinks
        self.in_blink = False
        
        # Debug info
        self.debug_values = []

    def update(self, eeg_data, current_time):
        """
        Update the detector with new EEG data
        
        Args:
            eeg_data: List or array of EEG values [TP9, AF7, AF8, TP10]
            current_time: Current timestamp
            
        Returns:
            (is_blink, confidence) tuple
        """
        # Extract frontal electrode data (AF7 = index 1, AF8 = index 2 in Muse)
        front_left = eeg_data[1]
        front_right = eeg_data[2]
        
        # Add to buffers
        self.front_left_buffer.append(front_left)
        self.front_right_buffer.append(front_right)
        
        # Wait until buffer is filled
        if len(self.front_left_buffer) < self.buffer_size:
            return False, 0
            
        # Calculate baseline if needed
        if not self.baseline_calculated and len(self.front_left_buffer) == self.buffer_size:
            self.calculate_baseline()
            return False, 0
        
        # Check for blink
        return self.detect_blink(current_time)
    
    def calculate_baseline(self):
        """Calculate baseline EEG values for blink detection"""
        left_array = np.array(self.front_left_buffer)
        right_array = np.array(self.front_right_buffer)
        
        # Get mean and std for both channels
        left_mean = np.mean(left_array)
        left_std = np.std(left_array)
        right_mean = np.mean(right_array)
        right_std = np.std(right_array)
        
        self.baseline_values = [left_mean, right_mean]
        self.baseline_std = [left_std, right_std]
        
        self.baseline_calculated = True
        print(f"Baseline calculated: [{left_mean:.2f} ± {left_std:.2f}, {right_mean:.2f} ± {right_std:.2f}]")
    
    def detect_blink(self, current_time):
        """
        Detect if a blink occurred in the latest data
        
        Returns:
            (is_blink, confidence) tuple
        """
        # Don't detect blinks too close together
        if current_time - self.last_blink_time < self.blink_debounce_time:
            return False, 0
            
        # Check the most recent samples (last ~100ms)
        samples_to_check = min(25, len(self.front_left_buffer))
        recent_left = list(self.front_left_buffer)[-samples_to_check:]
        recent_right = list(self.front_right_buffer)[-samples_to_check:]
        
        # Calculate z-scores relative to baseline
        z_left = [(x - self.baseline_values[0]) / self.baseline_std[0] for x in recent_left]
        z_right = [(x - self.baseline_values[1]) / self.baseline_std[1] for x in recent_right]
        
        # Blink detection logic:
        # 1. Both channels must show significant deviation
        # 2. Deviations must be in the same direction (usually negative for Muse)
        # 3. Magnitude must exceed threshold
        
        # Get maximum deviation
        max_dev_left = min(z_left) if min(z_left) < 0 else max(z_left)
        max_dev_right = min(z_right) if min(z_right) < 0 else max(z_right)
        
        # Record for debugging
        self.debug_values.append((max_dev_left, max_dev_right))
        if len(self.debug_values) > 500:
            self.debug_values.pop(0)
        
        # Both channels show significant negative deviation (typical for blinks)
        threshold = -self.threshold_factor  # Looking for negative deviations
        
        if max_dev_left < threshold and max_dev_right < threshold:
            if not self.in_blink:  # Only trigger once per blink
                self.in_blink = True
                self.last_blink_time = current_time
                
                # Calculate confidence based on deviation magnitude
                confidence = min(100, (abs(max_dev_left) + abs(max_dev_right)) / (2 * self.threshold_factor) * 100)
                return True, confidence
        else:
            # Reset blink state if values return to normal
            if self.in_blink and max_dev_left > -1.0 and max_dev_right > -1.0:
                self.in_blink = False
                
        return False, 0
        
    def reset_baseline(self):
        """Reset the baseline calculation"""
        self.baseline_calculated = False
        self.front_left_buffer.clear()
        self.front_right_buffer.clear()


class LatencyTestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse EEG Latency Test")
        self.setGeometry(100, 100, 800, 600)
        
        # LSL connection state
        self.inlet = None
        self.connected = False
        self.sampling_rate = 256.0
        self.channel_count = 4
        
        # Test parameters
        self.num_trials = 10
        self.current_trial = 0
        self.trial_results = []
        self.blink_request_time = None
        self.test_running = False
        self.trial_running = False
        
        # Blink detector
        self.blink_detector = BlinkDetector(sampling_rate=self.sampling_rate)
        
        # Setup UI
        self.setup_ui()
        
        # Setup LSL inlet timer
        self.lsl_timer = QtCore.QTimer()
        self.lsl_timer.timeout.connect(self.process_eeg_data)
        self.lsl_timer.setInterval(10)  # Process data every 10ms
        
        # Setup result variables
        self.blink_detected = False
        self.detection_time = None
        self.countdown_timer = None

    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Title
        title_label = QtWidgets.QLabel("Muse EEG Latency Test")
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Status indicators
        status_layout = QtWidgets.QHBoxLayout()
        self.connection_status = QtWidgets.QLabel("Status: Not Connected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        status_layout.addWidget(self.connection_status)
        
        self.sample_rate_label = QtWidgets.QLabel("Sample Rate: N/A")
        status_layout.addWidget(self.sample_rate_label)
        layout.addLayout(status_layout)
        
        # Main instruction display
        self.instruction_label = QtWidgets.QLabel("Click 'Connect' to start")
        self.instruction_label.setFont(QtGui.QFont("Arial", 16))
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instruction_label.setStyleSheet("margin: 40px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)
        
        # Countdown timer display
        self.countdown_display = QtWidgets.QLabel("")
        self.countdown_display.setFont(QtGui.QFont("Arial", 36, QtGui.QFont.Bold))
        self.countdown_display.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.countdown_display)
        
        # Results display
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        
        self.results_table = QtWidgets.QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Trial #", "Latency (ms)", "Confidence"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        self.avg_latency_label = QtWidgets.QLabel("Average Latency: N/A")
        self.avg_latency_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        results_layout.addWidget(self.avg_latency_label)
        
        layout.addWidget(results_group)
        
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
        
        layout.addLayout(button_layout)
        
        # Options
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.addWidget(QtWidgets.QLabel("Number of Trials:"))
        
        self.trials_spinbox = QtWidgets.QSpinBox()
        self.trials_spinbox.setRange(1, 50)
        self.trials_spinbox.setValue(10)
        self.trials_spinbox.valueChanged.connect(self.update_trial_count)
        options_layout.addWidget(self.trials_spinbox)
        
        options_layout.addWidget(QtWidgets.QLabel("Threshold Factor:"))
        
        self.threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.threshold_spinbox.setRange(1.0, 10.0)
        self.threshold_spinbox.setValue(3.5)
        self.threshold_spinbox.setSingleStep(0.5)
        self.threshold_spinbox.valueChanged.connect(self.update_threshold)
        options_layout.addWidget(self.threshold_spinbox)
        
        layout.addLayout(options_layout)
        
    def toggle_connection(self):
        """Connect to or disconnect from the Muse headset via LSL"""
        if not self.connected:
            self.connect_to_muse()
        else:
            self.disconnect_from_muse()
    
    def connect_to_muse(self):
        """Connect to Muse headset via LSL"""
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
            self.inlet = pylsl.StreamInlet(streams[0])
            
            # Get stream info
            info = self.inlet.info()
            self.sampling_rate = info.nominal_srate()
            self.channel_count = info.channel_count()
            
            # Update UI
            self.connection_status.setText(f"Status: Connected to {info.name()}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold")
            self.sample_rate_label.setText(f"Sample Rate: {self.sampling_rate} Hz")
            
            # Start processing EEG data
            self.connected = True
            self.connect_button.setText("Disconnect")
            self.start_button.setEnabled(True)
            self.lsl_timer.start()
            
            # Create new blink detector with correct sampling rate
            self.blink_detector = BlinkDetector(sampling_rate=self.sampling_rate)
            
            # Update instructions
            self.instruction_label.setText("Ready to start latency test.\n\nYou will be asked to blink when prompted.")
            
        except Exception as e:
            self.connection_status.setText(f"Status: Error - {str(e)}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold")
            QtWidgets.QMessageBox.critical(self, "Connection Error", f"Error: {str(e)}")
    
    def disconnect_from_muse(self):
        """Disconnect from the Muse headset"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
        
        self.connected = False
        self.connect_button.setText("Connect to Muse")
        self.start_button.setEnabled(False)
        
        self.connection_status.setText("Status: Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold")
        
    def update_trial_count(self, value):
        """Update the number of trials"""
        self.num_trials = value
        
    def update_threshold(self, value):
        """Update the blink detection threshold factor"""
        self.blink_detector.threshold_factor = value
    
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
        
        # Start first trial after a short delay
        QtCore.QTimer.singleShot(2000, self.start_trial)
        
    def start_trial(self):
        """Start a single trial of the latency test"""
        if self.current_trial >= self.num_trials:
            self.finish_test()
            return
            
        self.current_trial += 1
        
        # Reset state for this trial
        self.blink_detected = False
        self.detection_time = None
        self.blink_request_time = None
        
        # Reset baseline detection
        self.blink_detector.reset_baseline()
        
        # Show instructions
        self.instruction_label.setText(
            f"Trial {self.current_trial} of {self.num_trials}\n\n"
            "Stay still and look forward.\n"
            "When prompted, blink once quickly."
        )
        
        # Start countdown to prompt
        self.start_countdown(random.randint(3, 6))  # Random delay between 3-6 seconds
        
    def start_countdown(self, seconds):
        """Start countdown timer before prompting for blink"""
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
                self.prompt_for_blink()
                
        self.countdown_timer.timeout.connect(update_countdown)
        self.countdown_timer.start(1000)  # 1 second interval
        
    def prompt_for_blink(self):
        """Prompt the user to blink and record the time"""
        self.instruction_label.setText("BLINK NOW!")
        self.instruction_label.setStyleSheet("margin: 40px 0; padding: 20px; background-color: #ffcccc; border-radius: 10px; font-size: 24pt; font-weight: bold;")
        
        # Record the time of the prompt
        self.blink_request_time = time.time()
        
        # Set timeout for this trial
        QtCore.QTimer.singleShot(3000, self.check_trial_timeout)
        
    def check_trial_timeout(self):
        """Check if the trial has timed out"""
        if self.trial_running and not self.blink_detected:
            # Trial timed out
            self.trial_running = False
            self.instruction_label.setText("No blink detected. Moving to next trial...")
            self.instruction_label.setStyleSheet("margin: 40px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
            
            # Add failed trial to results
            self.add_trial_result(-1, 0)  # -1 indicates timeout
            
            # Continue to next trial after delay
            QtCore.QTimer.singleShot(2000, self.start_trial)
            
    def add_trial_result(self, latency_ms, confidence):
        """Add trial result to the table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Add data to table
        self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.current_trial)))
        
        if latency_ms < 0:
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem("Timeout"))
        else:
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{latency_ms:.1f}"))
            
        self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{confidence:.1f}%"))
        
        # Store result
        self.trial_results.append((latency_ms, confidence))
        
        # Update average latency
        valid_latencies = [lat for lat, _ in self.trial_results if lat >= 0]
        if valid_latencies:
            avg_latency = sum(valid_latencies) / len(valid_latencies)
            self.avg_latency_label.setText(f"Average Latency: {avg_latency:.1f} ms")
    
    def process_eeg_data(self):
        """Process incoming EEG data and detect blinks"""
        if not self.connected or not self.inlet:
            return
            
        # Get all available samples
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=100)
        
        if not chunk:
            return
            
        # Process each sample
        for i, sample in enumerate(chunk):
            current_time = time.time()  # Use local time for precision
            
            # Only look for blinks if test is running
            if self.trial_running and self.blink_request_time and not self.blink_detected:
                # Process sample with blink detector
                is_blink, confidence = self.blink_detector.update(sample, current_time)
                
                if is_blink:
                    self.detection_time = current_time
                    self.blink_detected = True
                    
                    # Calculate latency
                    latency_ms = (self.detection_time - self.blink_request_time) * 1000.0
                    
                    print(f"Blink detected! Latency: {latency_ms:.1f} ms, Confidence: {confidence:.1f}%")
                    
                    # Update UI
                    self.instruction_label.setText(f"Blink detected!\nLatency: {latency_ms:.1f} ms")
                    self.instruction_label.setStyleSheet("margin: 40px 0; padding: 20px; background-color: #ccffcc; border-radius: 10px;")
                    
                    # Add result
                    self.add_trial_result(latency_ms, confidence)
                    
                    # End this trial
                    self.trial_running = False
                    
                    # Move to next trial after a delay
                    QtCore.QTimer.singleShot(2000, self.start_trial)
    
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
            result_text = "Test Complete, but no successful blinks were detected."
        
        self.instruction_label.setText(result_text)
        self.instruction_label.setStyleSheet("margin: 40px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        
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
                    # Write header
                    f.write("Trial,Latency_ms,Confidence\n")
                    
                    # Write data
                    for i, (latency, confidence) in enumerate(self.trial_results):
                        latency_str = str(latency) if latency >= 0 else "Timeout"
                        f.write(f"{i+1},{latency_str},{confidence}\n")
                    
                    # Write summary statistics
                    valid_latencies = [lat for lat, _ in self.trial_results if lat >= 0]
                    if valid_latencies:
                        avg_latency = sum(valid_latencies) / len(valid_latencies)
                        min_latency = min(valid_latencies)
                        max_latency = max(valid_latencies)
                        std_dev = np.std(valid_latencies) if len(valid_latencies) > 1 else 0
                        
                        f.write("\n")
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
        self.instruction_label.setText("Ready to start latency test.\n\nYou will be asked to blink when prompted.")
        self.instruction_label.setStyleSheet("margin: 40px 0; padding: 20px; background-color: #f0f0f0; border-radius: 10px;")
        
    def closeEvent(self, event):
        """Clean up when window is closed"""
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
            
        if self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            
        if self.inlet:
            self.inlet.close_stream()
            
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LatencyTestWindow()
    window.show()
    sys.exit(app.exec_())