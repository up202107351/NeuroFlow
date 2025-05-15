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
        if not self.main_app_window.is_lsl_connected: # Access main window's state
             QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                           "Cannot start session.")
             return
        
        print("Meditation Page: Start Video Feedback clicked.")

        self.session_goal = "RELAXATION" # Set the goal for this session
        self.is_calibrating = True # UI should show "Calibrating..."
        self.is_calibrated = False
        self.video_player_window.set_status("Calibrating EEG... Please relax.")

        self.session_target_label = "Relaxed" # Example for meditation
        session_type_for_db = "Meditation-Video"
        target_metric_for_db = "Relaxation" # Generic name for what's being tracked

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
        backend_script_path = "eeg_backend_processor.py" # Assuming it's in the same dir or on PATH
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
        QtCore.QTimer.singleShot(1500, self._start_prediction_subscriber) # Delay subscriber start

        # --- UI Updates ---
        if not self.video_player_window:
            self.video_player_window = VideoPlayerWindow(self)
        self.video_player_window.show()
        self.video_player_window.set_status("Launching EEG Backend...")

        self.btn_start_video_feedback.setEnabled(False)
        self.btn_stop_video_feedback.setEnabled(True)


    def _start_prediction_subscriber(self):
        """Starts the ZMQ subscriber thread after a short delay."""
        if self.backend_process is None or self.backend_process.poll() is not None:
            print("Frontend: Backend process not running. Cannot start subscriber.")
            self.video_player_window.set_status("Backend failed to start.")
            self.stop_video_session() # Clean up UI
            return

        print("Frontend: Starting ZMQ prediction subscriber thread.")
        self.prediction_subscriber = EEGPredictionSubscriber() # ZMQ address is default
        self.prediction_thread = QtCore.QThread()
        self.prediction_subscriber.moveToThread(self.prediction_thread)

        self.prediction_subscriber.new_prediction_received.connect(self.on_new_eeg_prediction)
        self.prediction_subscriber.subscriber_error.connect(self.on_subscriber_error)
        self.prediction_subscriber.connection_status.connect(self.on_subscriber_connection_status)

        self.prediction_thread.started.connect(self.prediction_subscriber.run)
        self.prediction_subscriber.finished.connect(self.prediction_thread.quit) # Assuming subscriber emits finished
        self.prediction_subscriber.finished.connect(self.prediction_subscriber.deleteLater)
        self.prediction_thread.finished.connect(self.prediction_thread.deleteLater)

        self.prediction_thread.start()

    def stop_video_session(self):
        print("Meditation Page: Stop Video Feedback requested.")
        if self.current_session_id is not None:
            db_manager.end_session_and_summarize(self.current_session_id, datetime.now())
            self.current_session_id = None
            self.current_session_start_time = None

        self.update_button_states(self.parent().is_lsl_connected)
        # Stop the ZMQ subscriber thread first
        if self.prediction_subscriber:
            self.prediction_subscriber.stop() # Signal it to stop its loop
        if self.prediction_thread and self.prediction_thread.isRunning():
            # Give it a moment to shut down, then force quit if necessary.
            # Proper shutdown involves the subscriber's run loop exiting.
            self.prediction_thread.quit()
            if not self.prediction_thread.wait(2000): # Wait up to 2 seconds
                 print("Frontend: Prediction thread did not quit gracefully, terminating.")
                 self.prediction_thread.terminate()
                 self.prediction_thread.wait()


        # Terminate the backend process
        if self.backend_process:
            if self.backend_process.poll() is None: # Check if still running
                print(f"Frontend: Terminating backend process PID: {self.backend_process.pid}")
                self.backend_process.terminate() # Send SIGTERM
                try:
                    self.backend_process.wait(timeout=5) # Wait for it to terminate
                except subprocess.TimeoutExpired:
                    print(f"Frontend: Backend process PID: {self.backend_process.pid} did not terminate, killing.")
                    self.backend_process.kill() # Force kill
                print("Frontend: Backend process stopped.")
            self.backend_process = None

        if self.video_player_window:
            self.video_player_window.set_status("Session stopped.")
            # self.video_player_window.close()

        self.btn_start_video_feedback.setEnabled(True)
        self.btn_stop_video_feedback.setEnabled(False)
        self.prediction_thread = None
        self.prediction_subscriber = None

    @QtCore.pyqtSlot(dict) # Slot for calibration status messages
    def on_calibration_status(self, status_data):
        status = status_data.get("status")
        print(f"UI: Calibration Status Update: {status}")
        if status == "calibration_started":
            self.is_calibrating = True
            self.video_player_window.set_status(f"Calibrating... ({status_data.get('duration')}s remaining - estimate)")
        elif status == "calibration_complete":
            self.is_calibrating = False
            self.is_calibrated = True
            baselines = status_data.get("baselines")
            if self.video_player_window and baselines and 'ab_ratio' in baselines:
                self.video_player_window.set_ab_ratio_baseline(baselines['ab_ratio']) # SET BASELINE
                self.video_player_window.set_status("Calibration Complete. Session starting.")
            elif self.video_player_window:
                 self.video_player_window.set_status("Calibration Complete (no A/B baseline). Session starting.")
        elif status == "calibration_failed":
            self.is_calibrating = False
            self.is_calibrated = False
            self.video_player_window.set_status(f"Calibration Failed: {status_data.get('reason', 'Unknown')}. Please try again.")
            QtWidgets.QMessageBox.critical(self, "Calibration Failed", f"EEG baseline calibration failed. {status_data.get('reason', '')}")
            self.stop_video_session() # Stop the attempt

    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction_received(self, prediction_dict):
        if self.current_session_id is None:
            return # No active session to log for
        if self.is_calibrating or not self.is_calibrated:
            return # Ignore predictions during calibration or if not calibrated

        prediction_label = prediction_dict.get("prediction_label", "Unknown")        
        ab_ratio = prediction_dict.get("current_ab_ratio") 

        is_on_target = (prediction_label == self.session_target_label)

        db_manager.add_session_metric(self.current_session_id, prediction_label, is_on_target, ab_ratio)

        # Update video player
        if self.video_player_window:
            self.video_player_window.update_based_on_prediction(prediction_label)

    @QtCore.pyqtSlot(str)
    def on_subscriber_error(self, error_message):
        print(f"UI: Subscriber Error: {error_message}")
        if self.video_player_window:
            self.video_player_window.set_status(f"Comms Error: {error_message}")
        # Optionally try to reconnect or inform user more prominently

    @QtCore.pyqtSlot(str)
    def on_subscriber_connection_status(self, status_message):
        print(f"UI: Subscriber Connection Status: {status_message}")
        if self.video_player_window:
             self.video_player_window.set_status(status_message)
        if "Failed" in status_message or "Disconnected" in status_message:
            # Consider if you want to automatically stop the session or try to reconnect
            pass

    # Ensure to stop the backend if the main window closes while a session is active
    def clean_up_session(self): # Call this from main window's closeEvent or page visibility change
        print("Meditation Page: Cleaning up active session if any.")
        if self.backend_process and self.backend_process.poll() is None:
            self.stop_video_session()

    def launch_unity_game(self):
        print("Meditation Page: Launch Unity Game clicked.")
        # --- LAUNCH UNITY GAME (Copied from MeditationSelectionDialog for direct launch) ---
        unity_game_path = r"C:/path/to/your/unity/game.exe" # <-- !!! IMPORTANT: SET THIS PATH !!!
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