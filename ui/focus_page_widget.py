import os
import subprocess
import signal
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import subprocess # To launch the backend script
import matplotlib
from datetime import datetime
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from ui.video_player_window import VideoPlayerWindow
from backend.eeg_prediction_subscriber import EEGPredictionSubscriber
from backend import database_manager as db

class FocusPageWidget(QtWidgets.QWidget):
    def __init__(self, parent = None, main_app_window_ref = None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.user_id = None  # Will be set when user logs in
        self.backend_process = None
        self.prediction_subscriber = None
        self.prediction_thread = None
        self.video_player_window = None
        self.current_session_id = None
        self.current_session_start_time = None
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.initUI()
        if self.main_app_window:
            self.update_button_tooltips(self.main_app_window.is_lsl_connected)
        else:
            self.update_button_tooltips(False)

    def initUI(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        # --- Title ---
        title_label = QtWidgets.QLabel("Choose Your Focus Session")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(25)

        # --- First Row of Focus Options (Work, Video) ---
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(30) # Spacing between items in this row

        # Option 1: Work Focus Session
        work_focus_layout = self.create_focus_option_layout(
            title="Work Session",
            image_path="./assets/work.jpg", # <-- SET PATH!
            button_text="Start",
            action_slot=self.start_work_focus_session
        )
        row1_layout.addLayout(work_focus_layout)

        # Option 2: Video Focus Session
        video_focus_layout = self.create_focus_option_layout(
            title="Video Session",
            image_path="./assets/focus.jpg", # <-- SET PATH!
            button_text="Start",
            action_slot=self.start_video_focus_session
        )
        row1_layout.addLayout(video_focus_layout)
        self.main_layout.addLayout(row1_layout)
        self.main_layout.addSpacing(25)

        # --- Second Row of Focus Options (Game) - Centered ---
        row2_outer_layout = QtWidgets.QHBoxLayout() # To help with centering
        row2_outer_layout.addStretch(1) # Push game option to center

        game_focus_layout = self.create_focus_option_layout(
            title="Game Session",
            image_path="./assets/focus_game.jpg", # <-- SET PATH!
            button_text="Start",
            action_slot=self.start_game_focus_session,
            is_single_item_row=True # Hint for potential size adjustment
        )
        row2_outer_layout.addLayout(game_focus_layout)
        row2_outer_layout.addStretch(1) # Push game option to center
        self.main_layout.addLayout(row2_outer_layout)


        self.main_layout.addStretch(1) # Push all content towards the top

    def create_focus_option_layout(self, title, image_path, button_text, action_slot, is_single_item_row=False):
        """Helper function to create a consistent layout for each focus option."""
        option_layout = QtWidgets.QVBoxLayout()
        option_layout.setAlignment(QtCore.Qt.AlignCenter)

        title_label = QtWidgets.QLabel(title)
        title_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Medium))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        option_layout.addWidget(title_label)
        option_layout.addSpacing(10)

        image_label = QtWidgets.QLabel()
        image_width = 250 if not is_single_item_row else 300 # Slightly larger if it's the only one in row
        image_height = 150 if not is_single_item_row else 180

        if os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            image_label.setPixmap(pixmap.scaled(image_width, image_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            image_label.setText(f"({title} Image Not Found)")
            image_label.setStyleSheet("background-color: #444; border: 1px solid #555; color: #ccc;")
        image_label.setFixedSize(image_width, image_height)
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        option_layout.addWidget(image_label)
        option_layout.addSpacing(10)

        button = QtWidgets.QPushButton(button_text)
        button.setStyleSheet("font-size: 11pt; padding: 8px 15px;")
        button.clicked.connect(action_slot)
        # Store button reference if needed for tooltips/disabling later
        setattr(self, f"btn_{title.lower().replace(' ', '_')}", button) # e.g., self.btn_work_focus
        option_layout.addWidget(button)

        return option_layout

    def update_button_tooltips(self, is_lsl_connected_from_main_app):
        tooltip_text = "Requires Muse connection." if not is_lsl_connected_from_main_app else ""
        
        # Check if user is logged in
        user_tooltip = "You must be logged in to start a session." if not self.user_id else ""
        
        # Combine tooltips if both apply
        if tooltip_text and user_tooltip:
            tooltip_text = f"{tooltip_text} {user_tooltip}"
        elif user_tooltip and not tooltip_text:
            tooltip_text = user_tooltip
        
        # Update button states
        enabled = is_lsl_connected_from_main_app and self.user_id is not None
        
        if hasattr(self, 'btn_work_session'):
            self.btn_work_session.setToolTip(tooltip_text)
            self.btn_work_session.setEnabled(enabled)
            
        if hasattr(self, 'btn_video_session'):
            self.btn_video_session.setToolTip(tooltip_text)
            self.btn_video_session.setEnabled(enabled)
            
        if hasattr(self, 'btn_game_session'):
            self.btn_game_session.setToolTip(tooltip_text)
            self.btn_game_session.setEnabled(enabled)

    # --- Action Slots for Focus Page Buttons ---
    def start_work_focus_session(self):
        print("Focus Page: Clicked Start Work Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
            
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In",
                                        "You must be logged in to start a session.")
            return
        
        # Create a new database session
        session_type = "Focus-Work"
        target_metric = "Concentration"
        self.current_session_id, self.current_session_start_time = db.start_new_session(
            self.user_id, session_type, target_metric
        )
        
        # Set session active
        self.session_goal = "FOCUS"
        
        # Launch the work focus monitor window
        # This would be a simple window with a timer and focus indicators
        self.work_monitor_window = QtWidgets.QDialog(self)
        self.work_monitor_window.setWindowTitle("Work Focus Monitor")
        self.work_monitor_window.setFixedSize(400, 300)
        
        # Set up the UI for the work monitor window
        monitor_layout = QtWidgets.QVBoxLayout(self.work_monitor_window)
        
        # Timer label
        self.timer_label = QtWidgets.QLabel("00:00")
        self.timer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        monitor_layout.addWidget(self.timer_label)
        
        # Focus indicator
        self.focus_indicator = QtWidgets.QProgressBar()
        self.focus_indicator.setRange(0, 100)
        self.focus_indicator.setValue(50)
        self.focus_indicator.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #8A2BE2;
                width: 10px;
                margin: 0.5px;
            }
        """)
        monitor_layout.addWidget(self.focus_indicator)
        
        # Status label
        self.focus_status_label = QtWidgets.QLabel("Monitoring focus...")
        self.focus_status_label.setAlignment(QtCore.Qt.AlignCenter)
        monitor_layout.addWidget(self.focus_status_label)
        
        # Add some space
        monitor_layout.addSpacing(20)
        
        # Stop button
        stop_button = QtWidgets.QPushButton("End Session")
        stop_button.setStyleSheet("background-color: #c0392b; color: white; padding: 8px;")
        stop_button.clicked.connect(self.stop_active_session)
        monitor_layout.addWidget(stop_button)
        
        # Launch backend processor
        backend_script_path = "eeg_backend_processor.py"
        try:
            print(f"Focus Page: Launching backend script: {backend_script_path}")
            self.backend_process = subprocess.Popen([sys.executable, "-u", backend_script_path])
            print(f"Focus Page: Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Focus Page: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            return
            
        # Start timer for session
        self.session_timer = QtCore.QTimer(self)
        self.session_timer.timeout.connect(self.update_session_timer)
        self.session_timer.start(1000)  # Update every second
        self.session_start_time = QtCore.QDateTime.currentDateTime()
        
        # Start prediction subscriber
        QtCore.QTimer.singleShot(1500, self._start_prediction_subscriber_for_focus)
        
        # Show the monitor window
        self.work_monitor_window.show()
        
        # Update UI
        self.btn_work_session.setEnabled(False)
        self.btn_video_session.setEnabled(False)
        self.btn_game_session.setEnabled(False)

    def update_session_timer(self):
        """Update the timer display for the active session"""
        if hasattr(self, 'session_start_time') and hasattr(self, 'timer_label'):
            elapsed = self.session_start_time.secsTo(QtCore.QDateTime.currentDateTime())
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.timer_label.setText(f"{minutes:02}:{seconds:02}")

    def _start_prediction_subscriber_for_focus(self):
        """Starts the ZMQ subscriber thread for focus sessions"""
        if self.backend_process is None or self.backend_process.poll() is not None:
            print("Focus Page: Backend process not running. Cannot start subscriber.")
            self.stop_active_session()
            return
            
        print("Focus Page: Starting ZMQ prediction subscriber thread.")
        self.prediction_subscriber = EEGPredictionSubscriber()
        self.prediction_thread = QtCore.QThread()
        self.prediction_subscriber.moveToThread(self.prediction_thread)
        
        # Connect signals
        self.prediction_subscriber.new_prediction_received.connect(self.on_new_eeg_prediction)
        self.prediction_subscriber.subscriber_error.connect(self.on_subscriber_error)
        self.prediction_subscriber.connection_status.connect(self.on_subscriber_connection_status)
        
        # Thread management
        self.prediction_thread.started.connect(self.prediction_subscriber.run)
        self.prediction_subscriber.finished.connect(self.prediction_thread.quit)
        self.prediction_subscriber.finished.connect(self.prediction_subscriber.deleteLater)
        self.prediction_thread.finished.connect(self.prediction_thread.deleteLater)
        
        # Start the thread
        self.prediction_thread.start()
        
        # Start focus session after a moment
        QtCore.QTimer.singleShot(2000, self._start_focus_calibration)

    def _start_focus_calibration(self):
        """Start focus calibration"""
        if not self.prediction_subscriber:
            return
            
        # Start the focus session
        success = self.prediction_subscriber.start_focus_session()
        
        if success:
            self.focus_status_label.setText("Calibrating EEG... Please focus.")
            print("Focus Page: Focus session started, calibration in progress.")
        else:
            self.focus_status_label.setText("Failed to start session.")
            print("Focus Page: Failed to start focus session.")
            self.stop_active_session()

    def start_video_focus_session(self):
        print("Focus Page: Clicked Start Video Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
            
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In",
                                        "You must be logged in to start a session.")
            return
            
        # Create a new session in the database
        session_type = "Focus-Video"
        target_metric = "Concentration"
        self.current_session_id, self.current_session_start_time = db.start_new_session(
            self.user_id, session_type, target_metric
        )
        
        # Set session active
        self.session_goal = "FOCUS"
        self.is_calibrating = True
        self.is_calibrated = False
        
        # Create video player window
        if self.video_player_window:
            self.video_player_window.set_status("Connecting to EEG...")
            self.video_player_window.show()
        else:
            self.video_player_window = VideoPlayerWindow(self)
            self.video_player_window.show()
            self.video_player_window.set_status("Connecting to EEG...")
            
        # Launch backend processor
        backend_script_path = "eeg_backend_processor.py"
        try:
            print(f"Focus Page: Launching backend script: {backend_script_path}")
            self.backend_process = subprocess.Popen([sys.executable, "-u", backend_script_path])
            print(f"Focus Page: Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Focus Page: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            return
            
        # Start prediction subscriber
        QtCore.QTimer.singleShot(1500, self._start_prediction_subscriber_for_focus)
        
        # Update UI
        self.btn_work_session.setEnabled(False)
        self.btn_video_session.setEnabled(False)
        self.btn_game_session.setEnabled(False)

    def start_game_focus_session(self):
        print("Focus Page: Clicked Start Game Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
            
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In",
                                        "You must be logged in to start a session.")
            return
            
        # Create a new session in the database
        session_type = "Focus-Game"
        target_metric = "Concentration"
        self.current_session_id, self.current_session_start_time = db.start_new_session(
            self.user_id, session_type, target_metric
        )
        
        # Launch the Unity game
        unity_game_path = r"C:/NeuroFlow/Neuro/NeuroFlowFocus.exe" # <-- SET PATH!
        if not os.path.exists(unity_game_path):
            QtWidgets.QMessageBox.warning(self, "Error", f"Focus game not found at:\n{unity_game_path}")
            # Clean up session
            if self.current_session_id:
                db.end_session(self.current_session_id)
                self.current_session_id = None
            return
            
        try:
            subprocess.Popen([unity_game_path])
            QtWidgets.QMessageBox.information(self, "Game Focus", 
                "Focus Game is launching. The session will be recorded in your history.")
                
            # Set a timer to end the session after a fixed time (e.g., 15 minutes)
            QtCore.QTimer.singleShot(15 * 60 * 1000, self.stop_active_session)
            
            # Update UI
            self.btn_work_session.setEnabled(False)
            self.btn_video_session.setEnabled(False)
            self.btn_game_session.setEnabled(False)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error launching focus game:\n{e}")
            # Clean up session
            if self.current_session_id:
                db.end_session(self.current_session_id)
                self.current_session_id = None

    def stop_active_session(self):
        """Stop the active focus session"""
        print("Focus Page: Stopping active session...")
        
        # Stop the subscriber thread
        if self.prediction_subscriber:
            try:
                self.prediction_subscriber.stop()
            except Exception as e:
                print(f"Error stopping subscriber: {e}")
        
        # Terminate the backend process
        if self.backend_process and self.backend_process.poll() is None:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(3)  # Wait up to 3 seconds for graceful termination
                
                # Force kill if still running
                if self.backend_process.poll() is None:
                    if os.name == 'nt':  # Windows
                        os.kill(self.backend_process.pid, signal.CTRL_BREAK_EVENT)
                    else:  # Unix/Linux
                        os.kill(self.backend_process.pid, signal.SIGKILL)
            except Exception as e:
                print(f"Error terminating backend process: {e}")
        
        # Close the video player window if it exists
        if hasattr(self, 'video_player_window') and self.video_player_window:
            self.video_player_window.close()
            self.video_player_window = None
            
        # Close the work monitor window if it exists
        if hasattr(self, 'work_monitor_window') and self.work_monitor_window:
            self.work_monitor_window.close()
            
        # Stop the session timer if it exists
        if hasattr(self, 'session_timer') and self.session_timer:
            self.session_timer.stop()
        
        # End the session in the database
        if self.current_session_id:
            db.end_session(self.current_session_id)
            self.current_session_id = None
            self.current_session_start_time = None
        
        # Reset session state
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        
        
        # Re-enable start buttons if LSL is connected
        if self.main_app_window and self.main_app_window.is_lsl_connected and self.user_id:
            if hasattr(self, 'btn_work_session'):
                self.btn_work_session.setEnabled(True)
            if hasattr(self, 'btn_video_session'):
                self.btn_video_session.setEnabled(True)
            if hasattr(self, 'btn_game_session'):
                self.btn_game_session.setEnabled(True)
                
        print("Focus Page: Session stopped")

    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction(self, prediction_data):
        """Handle new EEG predictions from the subscriber"""
        # Only process PREDICTION messages
        if prediction_data.get("message_type") != "PREDICTION":
            return
        
        # Extract metrics and classification from prediction
        classification = prediction_data.get("classification", {})
        metrics = prediction_data.get("metrics", {})
        
        # Get the display state, level, and smooth value
        state = classification.get("state", "Unknown")
        level = classification.get("level", 0) 
        smooth_value = classification.get("smooth_value", 0.5)
        
        # Log the prediction for debugging
        print(f"Focus Prediction: {state} (Level: {level}, Value: {smooth_value:.2f})")
        
        # Update UI based on session type
        if self.session_goal == "FOCUS":
            # Determine if on target for focus
            is_on_target = level > 0
            
            # Update database
            if self.current_session_id:
                db.add_session_metric(
                    self.current_session_id,
                    state,
                    is_on_target,
                    smooth_value
                )
            
            # Update work monitor if active
            if hasattr(self, 'focus_indicator') and hasattr(self, 'focus_status_label'):
                # Scale the smooth value to 0-100 for the progress bar
                focus_percent = int(smooth_value * 100)
                self.focus_indicator.setValue(focus_percent)
                
                # Update status message
                if level <= -3:
                    status = "Very distracted - try to refocus"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
                elif level == -2:
                    status = "Distracted - bring attention back"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #e67e22; }")
                elif level == -1:
                    status = "Slightly distracted - stay with it"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #f1c40f; }")
                elif level == 0:
                    status = "Neutral - continue focusing"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #3498db; }")
                elif level == 1:
                    status = "Slightly focused - good start"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
                elif level == 2:
                    status = "Moderately focused - well done"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #27ae60; }")
                elif level >= 3:
                    status = "Strongly focused - excellent"
                    self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #8A2BE2; }")
                    
                self.focus_status_label.setText(status)
            
            # Update video player if active
            if self.video_player_window and self.video_player_window.isVisible():
                # Video feedback for focus session
                if self.is_calibrating:
                    self.video_player_window.set_status(f"Calibrating... ({state})")
                else:
                    # Update the video based on focus level
                    if level <= -3:
                        self.video_player_window.set_scene("very_distracted")
                        self.video_player_window.set_status(f"Status: {state} (Try to refocus)")
                    elif level == -2:
                        self.video_player_window.set_scene("distracted")
                        self.video_player_window.set_status(f"Status: {state} (Bring attention back)")
                    elif level == -1:
                        self.video_player_window.set_scene("less_focused")
                        self.video_player_window.set_status(f"Status: {state} (Stay with it)")
                    elif level == 0:
                        self.video_player_window.set_scene("neutral")
                        self.video_player_window.set_status(f"Status: {state} (Continue focusing)")
                    elif level == 1:
                        self.video_player_window.set_scene("slightly_focused")
                        self.video_player_window.set_status(f"Status: {state} (Good start)")
                    elif level == 2:
                        self.video_player_window.set_scene("moderately_focused")
                        self.video_player_window.set_status(f"Status: {state} (Well done)")
                    elif level == 3:
                        self.video_player_window.set_scene("strongly_focused")
                        self.video_player_window.set_status(f"Status: {state} (Excellent)")
                    elif level >= 4:
                        self.video_player_window.set_scene("deeply_focused")
                        self.video_player_window.set_status(f"Status: {state} (Perfect focus!)")
                    
                    # Adjust video parameters
                    self.video_player_window.set_focus_level(smooth_value)

    @QtCore.pyqtSlot(str)
    def on_subscriber_connection_status(self, status):
        """Handle connection status updates from the subscriber"""
        print(f"EEG Connection Status: {status}")
        
        # Update UI based on status
        if "Connected" in status:
            pass  # Connected to backend
        elif "Connecting" in status:
            pass  # Connecting to backend
        elif "Disconnected" in status or "Failed" in status or "Error" in status:
            # Connection lost or failed
            if self.session_goal:
                QtWidgets.QMessageBox.warning(self, "Connection Lost",
                    "Connection to EEG Backend lost. Session will be stopped.")
                self.stop_active_session()

    @QtCore.pyqtSlot(str)
    def on_subscriber_error(self, error_message):
        """Handle error messages from the subscriber"""
        print(f"EEG Subscriber Error: {error_message}")
        
        # Show error to user if serious
        if "connection" in error_message.lower() or "timeout" in error_message.lower():
            QtWidgets.QMessageBox.warning(self, "EEG Connection Error", error_message)

    def clean_up_session(self):
        """If focus sessions launch persistent backends, add cleanup here."""
        print("FocusPageWidget: Cleaning up active session if any.")
        if self.backend_process and self.backend_process.poll() is None:
            self.stop_active_session()