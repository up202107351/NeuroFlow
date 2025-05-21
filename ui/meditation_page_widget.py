import signal
import sys
import os
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
import subprocess # To launch the backend script
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from ui.video_player_window import VideoPlayerWindow
from backend.eeg_prediction_subscriber import EEGPredictionSubscriber
from backend import database_manager as db_manager
import pythonosc
from pythonosc.udp_client import SimpleUDPClient

UNITY_IP = "127.0.0.1"
UNITY_OSC_PORT = 9000
UNITY_OSC_ADDRESS = "/muse/relaxation"


class MeditationPageWidget(QtWidgets.QWidget):
    # Signals could be added if actions here need to communicate broadly
    # e.g., eeg_session_start_requested = QtCore.pyqtSignal(str) # "video" or "game"

    def __init__(self, parent=None, main_app_window_ref=None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.backend_process = None # To hold the subprocess reference
        self.prediction_subscriber = None
        self.prediction_thread = None
        self.video_player_window = None
        self.current_session_id = None # To store active session ID
        self.current_session_start_time = None
        self.session_target_label = ""
        self.is_calibrating = False
        self.is_calibrated = False
        self.session_goal = None # "RELAXATION" or "FOCUS"
        self.initUI()

        self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)
        # Add a stop button to the Meditation Page UI
        self.btn_stop_video_feedback = QtWidgets.QPushButton("Stop Video Session")
        self.btn_stop_video_feedback.setStyleSheet("font-size: 11pt; padding: 8px 15px; background-color: #c0392b; color: white;")
        self.btn_stop_video_feedback.clicked.connect(self.stop_video_session)
        if self.main_app_window: # Check if reference is valid
            self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
            # This case should ideally not happen if instantiated correctly
            print("Warning: MeditationPageWidget initialized without a valid main_app_window reference.")
            self.update_button_states(False)

    def initUI(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30) # Padding
        main_layout.setAlignment(QtCore.Qt.AlignTop) # Align content to top

        # --- Title ---
        title_label = QtWidgets.QLabel("Choose Your Meditation Experience")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        main_layout.addWidget(title_label)
        main_layout.addSpacing(20)

        # --- Horizontal Layout for Teasers ---
        teasers_layout = QtWidgets.QHBoxLayout()

        # --- Left Teaser (Video Feedback) ---
        video_teaser_layout = QtWidgets.QVBoxLayout()
        video_teaser_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.video_trailer_placeholder = QtWidgets.QLabel() # Remove placeholder text
        video_trailer_image_path = "relax.jpg" # <-- SET PATH!
        if os.path.exists(video_trailer_image_path):
            pixmap = QtGui.QPixmap(video_trailer_image_path)
            self.video_trailer_placeholder.setPixmap(pixmap.scaled(300, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.video_trailer_placeholder.setText("(Video Teaser Image Not Found)")
            self.video_trailer_placeholder.setStyleSheet("background-color: #444; border: 1px solid #555; color: #ccc;")
        self.video_trailer_placeholder.setFixedSize(300, 180)
        self.video_trailer_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        video_teaser_layout.addWidget(self.video_trailer_placeholder)
        video_teaser_layout.addSpacing(10)

        self.btn_start_video_feedback = QtWidgets.QPushButton("Start Video Relaxation")
        self.btn_start_video_feedback.setStyleSheet("font-size: 11pt; padding: 8px 15px;")
        self.btn_start_video_feedback.clicked.connect(self.start_video_session)
        video_teaser_layout.addWidget(self.btn_start_video_feedback)

        teasers_layout.addLayout(video_teaser_layout)
        teasers_layout.addSpacing(30) # Space between teasers

        # --- Right Teaser (Unity Game) ---
        game_teaser_layout = QtWidgets.QVBoxLayout()
        game_teaser_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.game_teaser_placeholder = QtWidgets.QLabel()
        game_teaser_image_path = "game.png" # <-- SET PATH!
        if os.path.exists(game_teaser_image_path):
            pixmap = QtGui.QPixmap(game_teaser_image_path)
            self.game_teaser_placeholder.setPixmap(pixmap.scaled(300, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.game_teaser_placeholder.setText("(Game Teaser Image Not Found)")
            self.game_teaser_placeholder.setStyleSheet("background-color: #444; border: 1px solid #555; color: #ccc;")
        self.game_teaser_placeholder.setFixedSize(300, 180)
        self.game_teaser_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        game_teaser_layout.addWidget(self.game_teaser_placeholder)

        game_teaser_layout.addSpacing(10)

        self.connection_status_label = QtWidgets.QLabel("Not connected to EEG")
        self.connection_status_label.setStyleSheet("color: gray;")
        
        # Add it to your layout, for example:
        # (if you have a bottom area or status area in your layout)
        # main_layout.addWidget(self.connection_status_label)
        
        # If you don't want to display this label, but still need it to avoid the error,
        # you can create it but not add it to any layout:
        self.connection_status_label = QtWidgets.QLabel()
        self.connection_status_label.hide()  # Hide it if you don't want to display it

        self.btn_start_unity_game = QtWidgets.QPushButton("Launch Unity Game")
        self.btn_start_unity_game.setStyleSheet("font-size: 11pt; padding: 8px 15px;")
        self.btn_start_unity_game.clicked.connect(self.launch_unity_game)
        game_teaser_layout.addWidget(self.btn_start_unity_game)

        teasers_layout.addLayout(game_teaser_layout)
        main_layout.addLayout(teasers_layout)
        main_layout.addStretch(1) # Push content up

    def update_button_states(self, is_lsl_connected):
        # This applies to buttons that *start* a session requiring LSL.
        # The "Stop Session" button's enabled state is managed by session start/stop.
        if hasattr(self, 'btn_start_video_feedback'):
            self.btn_start_video_feedback.setToolTip("Muse must be connected." if not is_lsl_connected else "")
        if hasattr(self, 'btn_start_unity_game'):
            self.btn_start_unity_game.setToolTip("Muse must be connected." if not is_lsl_connected else "")

    def start_video_session(self):
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                        "Cannot start session.")
            return
        
        print("Meditation Page: Start Video Feedback clicked.")

        self.session_goal = "RELAXATION"  # Set the goal for this session
        self.is_calibrating = True  # UI should show "Calibrating..."
        self.is_calibrated = False
        
        if self.video_player_window:
            self.video_player_window.set_status("Connecting to EEG...")
            self.video_player_window.show()
        else:
            self.video_player_window = VideoPlayerWindow(self)
            self.video_player_window.show()
            self.video_player_window.set_status("Connecting to EEG...")

        self.session_target_label = "Relaxed"  # Example for meditation
        session_type_for_db = "Meditation-Video"
        target_metric_for_db = "Relaxation"  # Generic name for what's being tracked

        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            session_type_for_db, target_metric_for_db
        )

        if self.backend_process and self.backend_process.poll() is None:
            QtWidgets.QMessageBox.warning(self, "Session Active", "An EEG backend process is already running.")
            return
        if self.prediction_thread and self.prediction_thread.isRunning():
            QtWidgets.QMessageBox.warning(self, "Session Active", "Prediction subscriber already running.")
            return

        # --- 1. Launch the backend EEG processor script ---
        backend_script_path = "eeg_backend_processor.py"  # Assuming it's in the same dir or on PATH
        try:
            print(f"Frontend: Launching backend script: {backend_script_path}")
            # Use python -u for unbuffered output if you want to see backend prints immediately
            self.backend_process = subprocess.Popen([sys.executable, "-u", backend_script_path])
            print(f"Frontend: Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Frontend: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            return

        # Give backend a moment to start and bind the ZMQ publisher
        QtCore.QTimer.singleShot(1500, self._start_prediction_subscriber_for_relaxation)

        # --- UI Updates ---
        self.btn_start_video_feedback.setEnabled(False)
        self.btn_stop_video_feedback.setEnabled(True)

    def _start_prediction_subscriber_for_relaxation(self):
        """Starts the ZMQ subscriber thread and initiates a relaxation session"""
        if self.backend_process is None or self.backend_process.poll() is not None:
            print("Frontend: Backend process not running. Cannot start subscriber.")
            if self.video_player_window:
                self.video_player_window.set_status("Backend failed to start.")
            self.stop_video_session()  # Clean up UI
            return

        print("Frontend: Starting ZMQ prediction subscriber thread.")
        self.prediction_subscriber = EEGPredictionSubscriber()  # ZMQ address is default
        self.prediction_thread = QtCore.QThread()
        self.prediction_subscriber.moveToThread(self.prediction_thread)

        # Connect signals
        self.prediction_subscriber.new_prediction_received.connect(self.on_new_eeg_prediction)
        self.prediction_subscriber.subscriber_error.connect(self.on_subscriber_error)
        self.prediction_subscriber.connection_status.connect(self.on_subscriber_connection_status)
        self.prediction_subscriber.calibration_progress.connect(self.on_calibration_progress)
        self.prediction_subscriber.calibration_status.connect(self.on_calibration_status)

        # Thread management
        self.prediction_thread.started.connect(self.prediction_subscriber.run)
        self.prediction_subscriber.finished.connect(self.prediction_thread.quit)
        self.prediction_subscriber.finished.connect(self.prediction_subscriber.deleteLater)
        self.prediction_thread.finished.connect(self.prediction_thread.deleteLater)

        # Start the thread
        self.prediction_thread.start()
        
        # Wait a moment for the thread to connect, then start relaxation session
        QtCore.QTimer.singleShot(2000, self._start_relaxation_calibration)

    def _start_relaxation_calibration(self):
        """Start the actual relaxation session after subscriber is connected"""
        if not self.prediction_subscriber:
            return
            
        # Start the relaxation session
        success = self.prediction_subscriber.start_relaxation_session()
        
        if success:
            if self.video_player_window:
                self.video_player_window.set_status("Calibrating EEG... Please relax.")
            print("Frontend: Relaxation session started, calibration in progress.")
        else:
            if self.video_player_window:
                self.video_player_window.set_status("Failed to start session.")
            print("Frontend: Failed to start relaxation session.")
            self.stop_video_session()

    def stop_video_session(self):
        """Stop the current video session"""
        print("Meditation Page: Stopping video session...")
        
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
        
        # Close the video player window
        if self.video_player_window:
            self.video_player_window.close()
            self.video_player_window = None
        
        # Reset session state
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        
        # End the session in the database
        if self.current_session_id:
            db_manager.end_session(self.current_session_id)
            self.current_session_id = None
            self.current_session_start_time = None
        
        # Reset UI
        self.btn_start_video_feedback.setEnabled(True)
        self.btn_stop_video_feedback.setEnabled(False)
        
        print("Meditation Page: Video session stopped")

    @QtCore.pyqtSlot(dict) # Slot for calibration status messages
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
        state_key = classification.get("state_key", "neutral")
        
        # Log the prediction for debugging
        print(f"Prediction: {state} (Level: {level}, Value: {smooth_value:.2f})")
        
        # --- Send smooth_value to Unity via OSC ---
        if self.osc_client:
            # Scale smooth_value (0.0-1.0) to 0-100 for Unity slider
            scaled_relaxation_level = smooth_value * 100.0
            try:
                self.osc_client.send_message(self.UNITY_OSC_ADDRESS, scaled_relaxation_level)
                # print(f"Sent OSC: {self.UNITY_OSC_ADDRESS}, Value: {scaled_relaxation_level:.2f}") # Uncomment for debugging OSC sends
            except Exception as e:
                print(f"Error sending OSC message to Unity: {e}")
        else:
            print("OSC Client not initialized. Cannot send data to Unity.")
        # --- End OSC Sending ---

        # Store the last prediction for later queries
        self.last_prediction = classification
        
        # Update the video player if it's active
        if self.video_player_window and self.video_player_window.isVisible():
            # Update video state based on the level and smooth value
            self.update_video_feedback(state, level, smooth_value, state_key)
        
        # Log to database if session is active
        if self.current_session_id:
            # Save prediction to the database
            db_manager.log_session_event(
                self.current_session_id, 
                "prediction", 
                {
                    "state": state,
                    "level": level, 
                    "value": smooth_value,
                    "metrics": metrics
                }
            )

    def update_video_feedback(self, state, level, smooth_value, state_key):
        """Update the video feedback based on the mental state prediction"""
        # Check if a session is active
        if not self.session_goal:
            return
        
        if self.is_calibrating:
            # During calibration, just show status
            self.video_player_window.set_status(f"Calibrating... ({state})")
            return
        
        # Different logic for relaxation vs focus sessions
        if self.session_goal == "RELAXATION":
            # Handle relaxation feedback
            if level <= -3:
                # Very tense/alert
                self.video_player_window.set_scene("very_tense")
                self.video_player_window.set_status(f"Status: {state} (Try to relax)")
            elif level == -2:
                # Moderately tense/alert
                self.video_player_window.set_scene("tense")
                self.video_player_window.set_status(f"Status: {state} (Breathe deeply)")
            elif level == -1:
                # Slightly tense/less relaxed
                self.video_player_window.set_scene("less_relaxed")
                self.video_player_window.set_status(f"Status: {state} (Find calmness)")
            elif level == 0:
                # Neutral
                self.video_player_window.set_scene("neutral")
                self.video_player_window.set_status(f"Status: {state} (Continue relaxing)")
            elif level == 1:
                # Slightly relaxed
                self.video_player_window.set_scene("slightly_relaxed")
                self.video_player_window.set_status(f"Status: {state} (Good start)")
            elif level == 2:
                # Moderately relaxed
                self.video_player_window.set_scene("moderately_relaxed")
                self.video_player_window.set_status(f"Status: {state} (Well done)")
            elif level == 3:
                # Strongly relaxed
                self.video_player_window.set_scene("strongly_relaxed")
                self.video_player_window.set_status(f"Status: {state} (Excellent)")
            elif level >= 4:
                # Deeply relaxed
                self.video_player_window.set_scene("deeply_relaxed")
                self.video_player_window.set_status(f"Status: {state} (Perfect!)")
            
            # Adjust video parameters based on smooth_value (0.0 to 1.0)
            # This allows for more fine-grained visual feedback
            self.video_player_window.set_relaxation_level(smooth_value)
            
        elif self.session_goal == "FOCUS": # COPY THIS BLOCK TO THE FOCUS PAGE
            # Handle focus feedback
            if level <= -3:
                # Very distracted
                self.video_player_window.set_scene("very_distracted")
                self.video_player_window.set_status(f"Status: {state} (Try to refocus)")
            elif level == -2:
                # Distracted
                self.video_player_window.set_scene("distracted")
                self.video_player_window.set_status(f"Status: {state} (Bring attention back)")
            elif level == -1:
                # Slightly distracted / less focused
                self.video_player_window.set_scene("less_focused")
                self.video_player_window.set_status(f"Status: {state} (Stay with it)")
            elif level == 0:
                # Neutral
                self.video_player_window.set_scene("neutral")
                self.video_player_window.set_status(f"Status: {state} (Continue focusing)")
            elif level == 1:
                # Slightly focused
                self.video_player_window.set_scene("slightly_focused")
                self.video_player_window.set_status(f"Status: {state} (Good start)")
            elif level == 2:
                # Moderately focused
                self.video_player_window.set_scene("moderately_focused")
                self.video_player_window.set_status(f"Status: {state} (Well done)")
            elif level == 3:
                # Strongly focused
                self.video_player_window.set_scene("strongly_focused")
                self.video_player_window.set_status(f"Status: {state} (Excellent)")
            elif level >= 4:
                # Deeply focused
                self.video_player_window.set_scene("deeply_focused")
                self.video_player_window.set_status(f"Status: {state} (Perfect focus!)")
            
            # Adjust video parameters based on smooth_value (0.0 to 1.0)
            self.video_player_window.set_focus_level(smooth_value)

    @QtCore.pyqtSlot(str)
    def on_subscriber_connection_status(self, status):
        """Handle connection status updates from the subscriber"""
        print(f"EEG Connection Status: {status}")
        
        # Update UI based on status
        if "Connected" in status:
            # Connected to backend
            self.connection_status_label.setText("Connected to EEG Backend")
            self.connection_status_label.setStyleSheet("color: green;")
        elif "Connecting" in status:
            # Connecting to backend
            self.connection_status_label.setText("Connecting to EEG Backend...")
            self.connection_status_label.setStyleSheet("color: orange;")
        elif "Disconnected" in status or "Failed" in status or "Error" in status:
            # Connection lost or failed
            self.connection_status_label.setText("Disconnected from EEG Backend")
            self.connection_status_label.setStyleSheet("color: red;")
            
            # If session was active, stop it
            if self.session_goal:
                QtWidgets.QMessageBox.warning(self, "Connection Lost",
                    "Connection to EEG Backend lost. Session will be stopped.")
                self.stop_video_session()

    @QtCore.pyqtSlot(str)
    def on_subscriber_error(self, error_message):
        """Handle error messages from the subscriber"""
        print(f"EEG Subscriber Error: {error_message}")
        
        # Show error to user if serious
        if "connection" in error_message.lower() or "timeout" in error_message.lower():
            QtWidgets.QMessageBox.warning(self, "EEG Connection Error", error_message)

    @QtCore.pyqtSlot(float)
    def on_calibration_progress(self, progress):
        """Handle calibration progress updates"""
        # progress is a float from 0.0 to 1.0
        if self.video_player_window:
            percent = int(progress * 100)
            self.video_player_window.set_status(f"Calibrating EEG: {percent}% complete")
            
            # Update progress bar if you have one
            if hasattr(self.video_player_window, 'calibration_progress_bar'):
                self.video_player_window.calibration_progress_bar.setValue(percent)

    @QtCore.pyqtSlot(str, dict)
    def on_calibration_status(self, status, baseline_data):
        """Handle calibration status updates"""
        print(f"Calibration Status: {status}, Baseline: {baseline_data}")
        
        if status == "COMPLETED":
            # Calibration completed successfully
            self.is_calibrating = False
            self.is_calibrated = True
            
            if self.video_player_window:
                self.video_player_window.set_status("Calibration complete. Starting session...")
                
                # Hide progress bar if you have one
                if hasattr(self.video_player_window, 'calibration_progress_bar'):
                    self.video_player_window.calibration_progress_bar.hide()
                
                # Start appropriate video based on session goal
                if self.session_goal == "RELAXATION":
                    self.video_player_window.start_relaxation_video()
                elif self.session_goal == "FOCUS":
                    self.video_player_window.start_focus_video()
        
        elif status == "FAILED":
            # Calibration failed
            self.is_calibrating = False
            self.is_calibrated = False
            
            QtWidgets.QMessageBox.warning(self, "Calibration Failed",
                "Failed to calibrate EEG. Please try again.")
            
            self.stop_video_session()

    # Ensure to stop the backend if the main window closes while a session is active
    def clean_up_session(self): # Call this from main window's closeEvent or page visibility change
        print("Meditation Page: Cleaning up active session if any.")
        if self.backend_process and self.backend_process.poll() is None:
            self.stop_video_session()

    def launch_unity_game(self):
        print("Meditation Page: Launch Unity Game clicked.")
        # --- LAUNCH UNITY GAME (Copied from MeditationSelectionDialog for direct launch) ---
        unity_game_path = r"C:\Neuro\NeuroFlow.exe" # <-- !!! IMPORTANT: SET THIS PATH !!!
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # unity_game_path = os.path.join(script_dir, "YourGameFolder", "game.exe")

        if not os.path.exists(unity_game_path):
            print(f"Error: Game executable not found at calculated path: {unity_game_path}")
            QtWidgets.QMessageBox.warning(self, "Error",
                                          f"Could not find the Unity game executable at the specified path:\n{unity_game_path}\n\nPlease check the path in MeditationPageWidget.")
            return

        try:
            print(f"Attempting to launch: {unity_game_path}")
            subprocess.Popen([unity_game_path]) # Non-blocking launch
            # Optionally show a brief confirmation
            # QtWidgets.QMessageBox.information(self, "Game Launched", "Unity game is launching.")
        except Exception as e:
            print(f"Error launching game: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An unexpected error occurred launching the game:\n{e}")