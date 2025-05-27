import signal
import sys
import os
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
# import subprocess # Duplicate import
from datetime import datetime
import time
import random
import matplotlib
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from ui.video_player_window import VideoPlayerWindow
from backend.eeg_prediction_subscriber import EEGPredictionSubscriber
from backend import database_manager as db_manager
# import pythonosc # Not used directly if SimpleUDPClient is used
from pythonosc.udp_client import SimpleUDPClient
from backend.zmq_port_cleanup import cleanup_all_zmq_ports

UNITY_IP = "127.0.0.1"
UNITY_OSC_PORT = 9000
UNITY_OSC_ADDRESS = "/muse/relaxation"
UNITY_OSC_SCENE_ADDRESS = "/muse/scene"


class MeditationPageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, main_app_window_ref=None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.backend_process = None
        self.prediction_subscriber = None
        self.prediction_thread = None
        self.video_player_window = None
        self.current_session_id = None
        self.current_session_start_time = None
        self.session_target_label = ""
        self.is_calibrating = False # Tracks if calibration process is active
        self.is_calibrated = False  # Tracks if calibration completed successfully
        self.session_goal = None
        self.user_id = None
        self.last_sent_scene_index = -1
        self.UNITY_OSC_SCENE_ADDRESS = "/muse/scene"
        self.initUI()

        self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)

        if self.main_app_window:
            self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
            print("Warning: MeditationPageWidget initialized without a valid main_app_window reference.")
            self.update_button_states(False)

        # Timers - define them here to manage them better
        self.connection_timeout_timer = QtCore.QTimer(self)
        self.connection_timeout_timer.setSingleShot(True)
        self.connection_timeout_timer.timeout.connect(self._handle_connection_timeout)

        self.calibration_update_timer = QtCore.QTimer(self)
        self.calibration_update_timer.timeout.connect(self._update_fake_calibration_progress)
        self.calibration_progress_value = 0


    def initUI(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        title_label = QtWidgets.QLabel("Choose Your Meditation Experience")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(20)

        teasers_layout = QtWidgets.QHBoxLayout()

        video_teaser_layout = QtWidgets.QVBoxLayout()
        video_teaser_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.video_trailer_placeholder = QtWidgets.QLabel()
        video_trailer_image_path = "./assets/relax.jpg"
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
        teasers_layout.addSpacing(30)

        game_teaser_layout = QtWidgets.QVBoxLayout()
        game_teaser_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.game_teaser_placeholder = QtWidgets.QLabel()
        game_teaser_image_path = "./assets/game.png"
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

        # self.connection_status_label = QtWidgets.QLabel("Not connected to EEG") # This seems better placed in main window status bar
        # self.connection_status_label.setStyleSheet("color: gray;")
        # self.connection_status_label.hide()

        self.btn_start_unity_game = QtWidgets.QPushButton("Launch Unity Game")
        self.btn_start_unity_game.setStyleSheet("font-size: 11pt; padding: 8px 15px;")
        self.btn_start_unity_game.clicked.connect(self.launch_unity_game)
        game_teaser_layout.addWidget(self.btn_start_unity_game)

        teasers_layout.addLayout(game_teaser_layout)
        self.main_layout.addLayout(teasers_layout)
        self.main_layout.addStretch(1)

    def update_button_states(self, is_lsl_connected):
        is_session_active = bool(self.session_goal) # True if a session is running

        if hasattr(self, 'btn_start_video_feedback'):
            self.btn_start_video_feedback.setEnabled(is_lsl_connected and not is_session_active)
            if not is_lsl_connected:
                self.btn_start_video_feedback.setToolTip("Muse must be connected.")
            elif is_session_active:
                self.btn_start_video_feedback.setToolTip("A session is already active.")
            else:
                self.btn_start_video_feedback.setToolTip("")

        if hasattr(self, 'btn_start_unity_game'):
            # Assuming Unity game doesn't run concurrently with video feedback managed by this widget
            self.btn_start_unity_game.setEnabled(is_lsl_connected and not is_session_active)
            if not is_lsl_connected:
                self.btn_start_unity_game.setToolTip("Muse must be connected.")
            elif is_session_active:
                 self.btn_start_unity_game.setToolTip("A video feedback session is active.")
            else:
                self.btn_start_unity_game.setToolTip("")


    def start_video_session(self):
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected", "Cannot start session.")
            return
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In", "You must be logged in to start a session.")
            return
        if self.session_goal: # A session is already active
            QtWidgets.QMessageBox.warning(self, "Session Active", "A video feedback session is already running.")
            return

        print("Meditation Page: Start Video Feedback clicked.")
        try:
            print("Cleaning up ZMQ ports before starting backend...")
            cleanup_all_zmq_ports()
        except Exception as e:
            print(f"Warning: Port cleanup failed: {e}")

        self.session_goal = "RELAXATION"
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0 # Reset progress

        if not self.video_player_window: # Create if it doesn't exist
            self.video_player_window = VideoPlayerWindow(parent=self) # Parent it correctly for modality if needed
            self.video_player_window.session_stopped.connect(self.handle_video_session_stopped_signal)
        
        self.video_player_window.set_status("Connecting to EEG...")
        self.video_player_window.show_calibration_progress(0) # Show bar immediately
        self.video_player_window.show()
        self.video_player_window.activateWindow() # Bring to front

        self.session_target_label = "Relaxed"
        session_type_for_db = "Meditation-Video"
        target_metric_for_db = "Relaxation"

        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            self.user_id, session_type_for_db, target_metric_for_db
        )

        if self.backend_process and self.backend_process.poll() is None:
            QtWidgets.QMessageBox.warning(self, "Backend Active", "An EEG backend process is already running. Please stop it first or wait.")
            # self._reset_ui_and_state() # Clean up if we can't proceed
            return
        if self.prediction_thread and self.prediction_thread.isRunning():
            QtWidgets.QMessageBox.warning(self, "Subscriber Active", "Prediction subscriber already running.")
            # self._reset_ui_and_state()
            return

        backend_script_path = "eeg_backend_processor.py"
        try:
            print(f"Frontend: Launching backend script: {backend_script_path}")
            self.backend_process = subprocess.Popen([sys.executable, "-u", backend_script_path],
                                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0) # For better termination on Windows
            print(f"Frontend: Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Frontend: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            self._reset_ui_and_state() # Call a helper to reset
            return

        self.update_button_states(self.main_app_window.is_lsl_connected) # Disable start buttons
        QtCore.QTimer.singleShot(2000, self._start_prediction_subscriber_for_relaxation) # Increased delay slightly

    def handle_video_session_stopped_signal(self):
        """Called when video player emits session_stopped"""
        print("MeditationPage: Received session_stopped signal from VideoPlayerWindow.")
        self.stop_video_session_logic()


    def stop_video_session_logic(self, triggered_by_error=False):
        """
        Core logic to stop the video session.
        Can be called by user action (stop button, close window) or by error.
        """
        print("Meditation Page: Stopping video session logic...")

        if self.connection_timeout_timer.isActive():
            self.connection_timeout_timer.stop()
        if self.calibration_update_timer.isActive():
            self.calibration_update_timer.stop()

        if self.prediction_subscriber:
            print("Meditation Page: Requesting subscriber to stop...")
            self.prediction_subscriber.stop() # This should make the run() loop exit
            if self.prediction_thread and self.prediction_thread.isRunning():
                print("Meditation Page: Waiting for prediction thread to quit...")
                if not self.prediction_thread.wait(2000): # Wait up to 2s
                    print("Meditation Page: Prediction thread did not quit gracefully, terminating.")
                    self.prediction_thread.terminate() # Force if necessary
                    self.prediction_thread.wait() # Wait for termination
            self.prediction_subscriber = None # Clear reference
            self.prediction_thread = None

        if self.backend_process and self.backend_process.poll() is None:
            print(f"Meditation Page: Terminating backend process PID: {self.backend_process.pid}")
            try:
                if os.name == 'nt':
                    # Send CTRL_BREAK_EVENT to the process group on Windows
                    # This is generally more effective for console apps that catch SIGINT/CTRL_C
                    os.kill(self.backend_process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    # Send SIGINT (Ctrl+C) first for graceful shutdown
                    self.backend_process.send_signal(signal.SIGINT)
                
                self.backend_process.wait(timeout=3) # Wait for graceful exit
            except subprocess.TimeoutExpired:
                print("Meditation Page: Backend process did not terminate gracefully, killing.")
                self.backend_process.kill() # Force kill
                self.backend_process.wait() # Wait for kill
            except Exception as e: # Catch other errors like process already exited
                print(f"Error during backend termination: {e}")
            self.backend_process = None

        # Video player window is closed by its own logic (closeEvent or _delayed_stop)
        # We just need to nullify our reference if it exists.
        if self.video_player_window:
            try:
                self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped_signal)
            except TypeError: # Already disconnected
                pass
            if not triggered_by_error and self.video_player_window.isVisible():
                 # If not an error, and window is visible, it means user might have closed it.
                 # If it was an error, the video window might already be handling its closure or showing an error.
                 # Generally, the video window should close itself. This is a fallback.
                 print("MeditationPage: Ensuring video player window is closed if still visible.")
                 # self.video_player_window.close() # Let the video player handle its own close
            self.video_player_window = None


        if self.current_session_id:
            db_manager.end_session(self.current_session_id)
            print(f"Meditation Page: Session {self.current_session_id} ended in DB.")
            self.current_session_id = None
            self.current_session_start_time = None

        self._reset_ui_and_state()
        print("Meditation Page: Video session stopped logic completed.")

    def _reset_ui_and_state(self):
        """Resets UI elements and internal state flags after a session."""
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.calibration_progress_value = 0

        # Close video player if it's still around and this method is called
        # (e.g. due to an error before the player is fully managed by stop_video_session_logic)
        if self.video_player_window:
            print("MeditationPage: _reset_ui_and_state closing video_player_window.")
            # Disconnect first to avoid re-triggering stop logic
            try:
                self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped_signal)
            except TypeError:
                pass # Already disconnected
            self.video_player_window.close()
            self.video_player_window = None

        # Re-enable buttons
        if hasattr(self.main_app_window, 'is_lsl_connected'):
             self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
             self.update_button_states(False) # Fallback

        # Reset status in video player (if it were to be reused, but we destroy it)
        # if self.video_player_window:
        # self.video_player_window.set_status("Session ended.")
        # self.video_player_window.calibration_progress_bar.hide()

    def _start_prediction_subscriber_for_relaxation(self):
        """Start the prediction subscriber for relaxation session"""
        if self.backend_process is None or self.backend_process.poll() is not None:
            print("Frontend: Backend process not running or exited. Cannot start subscriber.")
            if self.video_player_window:
                self.video_player_window.set_status("Backend failed to start.")
            self.stop_video_session_logic(triggered_by_error=True)
            return

        print("Frontend: Starting ZMQ prediction subscriber thread.")
        self.connection_timeout_timer.start(10000)  # 10 second timeout

        try:
            # Create video player window first to improve user experience
            if self.video_player_window:
                self.video_player_window.set_status("Connecting to EEG...")
                self.video_player_window.show_calibration_progress(0)
                # Start UI update timer immediately for responsiveness
                self.video_player_window.start_ui_updates()

            # Set up the prediction subscriber in a non-blocking way
            self.prediction_subscriber = EEGPredictionSubscriber()
            self.prediction_thread = QtCore.QThread()
            self.prediction_subscriber.moveToThread(self.prediction_thread)

            self.prediction_subscriber.new_prediction_received.connect(self.on_new_eeg_prediction)
            self.prediction_subscriber.subscriber_error.connect(self.on_subscriber_error)
            self.prediction_subscriber.connection_status.connect(self.on_subscriber_connection_status)
            self.prediction_subscriber.calibration_progress.connect(self.on_calibration_progress)
            self.prediction_subscriber.calibration_status.connect(self.on_calibration_status)

            self.prediction_thread.started.connect(self.prediction_subscriber.run)
            self.prediction_subscriber.finished.connect(self.prediction_thread.quit)

            # Start the thread
            self.prediction_thread.start()
            
            # Use a non-blocking timer to start calibration after thread is running
            QtCore.QTimer.singleShot(1000, self._initiate_calibration_in_subscriber)
        
        except Exception as e:
            print(f"Error starting prediction subscriber: {e}")
            if self.connection_timeout_timer.isActive():
                self.connection_timeout_timer.stop()
            self.stop_video_session_logic(triggered_by_error=True)


    def _initiate_calibration_in_subscriber(self):
        """Sends command to subscriber to start calibration after connection is likely established."""
        if not self.prediction_subscriber or not self.prediction_thread or not self.prediction_thread.isRunning():
            print("Frontend: Subscriber not ready for calibration.")
            if self.video_player_window:
                self.video_player_window.set_status("Error: EEG subscriber not ready.")
            self.stop_video_session_logic(triggered_by_error=True)
            return

        try:
            # Make sure the video player window is responsive
            if self.video_player_window:
                self.video_player_window.set_status("Calibrating EEG... Please relax.")
                
                # Start the UI updates in the video player window
                if hasattr(self.video_player_window, 'start_ui_updates'):
                    self.video_player_window.start_ui_updates()

            # Use direct method call - it's now non-blocking with the command queue
            print("Frontend: Requesting subscriber to start relaxation session (calibration)...")
            if not self.prediction_subscriber.start_relaxation_session():
                print("Frontend: Failed to queue start relaxation session command.")
                if self.video_player_window:
                    self.video_player_window.set_status("Failed to initiate calibration.")
                self.stop_video_session_logic(triggered_by_error=True)
                return

            print(f"{QtCore.QTime.currentTime().toString('hh:mm:ss.zzz')} - Frontend: Calibration requested")
            
        except Exception as e:
            print(f"{QtCore.QTime.currentTime().toString('hh:mm:ss.zzz')} - Error initiating calibration: {e}")
            if self.video_player_window:
                self.video_player_window.set_status(f"Calibration Error: {str(e)}")
            self.stop_video_session_logic(triggered_by_error=True)


    def _handle_connection_timeout(self):
        print("Frontend: ZMQ connection timeout.")
        if self.video_player_window:
            self.video_player_window.set_status("Connection timeout. Please try again.")
        QtWidgets.QMessageBox.warning(self, "Connection Timeout",
                                "Failed to connect to EEG backend. Please try again.")
        self.stop_video_session_logic(triggered_by_error=True)


    def _update_fake_calibration_progress(self):
        if not self.is_calibrating or self.calibration_progress_value >= 100:
            self.calibration_update_timer.stop()
            return

        if self.calibration_progress_value < 30: increment = 2
        elif self.calibration_progress_value < 60: increment = 1
        else: increment = 1
        self.calibration_progress_value = min(100, self.calibration_progress_value + increment)

        if self.video_player_window:
            self.video_player_window.show_calibration_progress(self.calibration_progress_value)
        
        # If it reaches 100% here, it means the backend never sent a "COMPLETED" status
        if self.calibration_progress_value >= 100:
            self.calibration_update_timer.stop()
            # This is a fallback, ideally on_calibration_status("COMPLETED",...) handles this
            print("Fake calibration reached 100%. Waiting for backend confirmation or timeout.")


    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction(self, prediction_data):
        if self.is_calibrating or not self.is_calibrated: # Ignore predictions during calibration or if not calibrated
            # print("Skipping prediction, still calibrating or not calibrated.")
            return
        if prediction_data.get("message_type") != "PREDICTION":
            return

        classification = prediction_data.get("classification", {})
        # metrics = prediction_data.get("metrics", {}) # metrics not used directly here
        state = classification.get("state", "Unknown")
        level = classification.get("level", 0)
        smooth_value = classification.get("smooth_value", 0.5)
        state_key = classification.get("state_key", "neutral")

        self.last_prediction = classification  # This line is crucial
    
        print(f"Got new prediction: state={state}, level={level}, smooth={smooth_value:.2f}")

        # print(f"Prediction: {state} (Level: {level}, Value: {smooth_value:.2f})")

        if self.client:
            scaled_relaxation_level = smooth_value * 100.0
            try:
                self.client.send_message(UNITY_OSC_ADDRESS, scaled_relaxation_level)
            except Exception as e:
                print(f"Error sending relaxation OSC message to Unity: {e}")

            target_scene_index = -1
            if scaled_relaxation_level >= 80: target_scene_index = 2
            elif scaled_relaxation_level >= 50: target_scene_index = 1
            else: target_scene_index = 0

            if target_scene_index != -1 and target_scene_index != self.last_sent_scene_index:
                try:
                    self.client.send_message(self.UNITY_OSC_SCENE_ADDRESS, target_scene_index)
                    self.last_sent_scene_index = target_scene_index
                    # print(f"Sent OSC Scene Change: {self.UNITY_OSC_SCENE_ADDRESS}, Index: {target_scene_index}")
                except Exception as e:
                    print(f"Error sending scene OSC message to Unity: {e}")
        # else: # Covered by init
        #     print("OSC Client not initialized!")

        self.last_prediction = classification # For potential future use

        if self.video_player_window and self.video_player_window.isVisible():
            self.update_video_feedback(state, level, smooth_value, state_key)

        if self.current_session_id:
            is_on_target = (self.session_goal == "RELAXATION" and level > 0) or \
                           (self.session_goal == "FOCUS" and level > 0)
            db_manager.add_session_metric(self.current_session_id, state, is_on_target, smooth_value)

    def update_video_feedback(self, state, level, smooth_value, state_key):
        if not self.session_goal or not self.video_player_window: # or self.is_calibrating (handled by on_new_eeg_prediction)
            return

        # Logic for relaxation
        if self.session_goal == "RELAXATION":
            if level <= -3: scene, status_msg = "very_tense", f"{state} (Try to relax)"
            elif level == -2: scene, status_msg = "tense", f"{state} (Breathe deeply)"
            elif level == -1: scene, status_msg = "less_relaxed", f"{state} (Find calmness)"
            elif level == 0: scene, status_msg = "neutral", f"{state} (Continue relaxing)"
            elif level == 1: scene, status_msg = "slightly_relaxed", f"{state} (Good start)"
            elif level == 2: scene, status_msg = "moderately_relaxed", f"{state} (Well done)"
            elif level == 3: scene, status_msg = "strongly_relaxed", f"{state} (Excellent)"
            else: scene, status_msg = "deeply_relaxed", f"{state} (Perfect!)" # level >= 4
            
            self.video_player_window.set_scene(scene)
            self.video_player_window.set_status(f"Status: {status_msg}")
            self.video_player_window.set_relaxation_level(smooth_value)

        elif self.session_goal == "FOCUS": # Placeholder for focus logic if adapted
            if level <= -3: scene, status_msg = "very_distracted", f"{state} (Try to refocus)"
            # ... (rest of focus logic)
            else: scene, status_msg = "deeply_focused", f"{state} (Perfect focus!)"

            self.video_player_window.set_scene(scene)
            self.video_player_window.set_status(f"Status: {status_msg}")
            self.video_player_window.set_focus_level(smooth_value)


    @QtCore.pyqtSlot(str)
    def on_subscriber_connection_status(self, status):
        print(f"EEG Connection Status from Subscriber: {status}")
        if "Connected" in status:
            if self.connection_timeout_timer.isActive():
                self.connection_timeout_timer.stop() # Connection successful
            if self.video_player_window: # Update status only if window exists
                self.video_player_window.set_status("EEG Connected. Initializing calibration...")
            # No need to call _initiate_calibration_in_subscriber here, it's timed after thread start.
        elif "Error" in status or "Failed" in status or "Disconnected" in status:
            if self.connection_timeout_timer.isActive():
                self.connection_timeout_timer.stop()
            QtWidgets.QMessageBox.critical(self, "EEG Connection Error",
                                      f"Subscriber reported: {status}. Stopping session.")
            self.stop_video_session_logic(triggered_by_error=True)


    @QtCore.pyqtSlot(str)
    def on_subscriber_error(self, error_message):
        print(f"EEG Subscriber Error: {error_message}")
        QtWidgets.QMessageBox.warning(self, "EEG Subscriber Error", error_message)
        # Decide if this error is critical enough to stop the session
        if "fatal" in error_message.lower() or "cannot recover" in error_message.lower():
             self.stop_video_session_logic(triggered_by_error=True)


    @QtCore.pyqtSlot(float)
    def on_calibration_progress(self, progress):
        if not self.is_calibrating: return  # Ignore if not in calibration phase

        self.calibration_progress_value = int(progress * 100)
        if self.video_player_window:
            # Just update the value - the video player's timer will refresh the UI
            self.video_player_window.show_calibration_progress(self.calibration_progress_value)
            
            # Only update the status text occasionally to avoid too many forced updates
            if self.calibration_progress_value % 10 == 0 or self.calibration_progress_value >= 100:
                self.video_player_window.set_status(f"Calibrating EEG: {self.calibration_progress_value}% complete")

        if self.calibration_update_timer.isActive():
            self.calibration_update_timer.stop()


    @QtCore.pyqtSlot(str, dict)
    def on_calibration_status(self, status, baseline_data):
        print(f"Calibration Status from Subscriber: {status}, Baseline: {baseline_data if baseline_data else 'N/A'}")
        
        if self.calibration_update_timer.isActive(): # Stop fake progress if real status comes
            self.calibration_update_timer.stop()

        if status == "COMPLETED":
            self.is_calibrating = False
            self.is_calibrated = True
            if self.video_player_window:
                self.video_player_window.set_status("Calibration complete. Starting session...")
                self.video_player_window.hide_calibration_progress_bar() # Method to hide bar
                
                # Video player should ideally start its video after its own _finish_calibration
                # Or we can trigger it here if needed
                if self.session_goal == "RELAXATION":
                    self.video_player_window.start_relaxation_video()
                # elif self.session_goal == "FOCUS":
                #     self.video_player_window.start_focus_video()
        elif status == "FAILED":
            self.is_calibrating = False
            self.is_calibrated = False
            QtWidgets.QMessageBox.warning(self, "Calibration Failed",
                "Failed to calibrate EEG. Please check the Muse connection and try again.")
            self.stop_video_session_logic(triggered_by_error=True)
        # Other statuses like "STARTED" could be logged or update UI
        elif status == "STARTED":
            self.is_calibrating = True # Ensure it's set
            if self.video_player_window:
                self.video_player_window.set_status("Calibration process initiated...")
                self.video_player_window.show_calibration_progress(0) # Show bar at 0 or small value


    def clean_up_session(self):
        print("Meditation Page: clean_up_session called (e.g., on main window close).")
        if self.session_goal: # If a video session is active
            self.stop_video_session_logic(triggered_by_error=False) # Or True if it's an unexpected cleanup
        if self.current_session_id and not self.session_goal: # For Unity game session that might be running
            self.end_unity_game_session() # Ensure Unity game session is ended


    def launch_unity_game(self):
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In", "You must be logged in to start a session.")
            return
        if self.session_goal:
            QtWidgets.QMessageBox.warning(self, "Session Active", "A session is already running.")
            return

        print("Meditation Page: Launch Unity Game clicked.")

        self._force_cleanup_zmq_ports()
        
        # Show a dialog to inform user about calibration
        QtWidgets.QMessageBox.information(self, "Calibration Required", 
            "We'll first calibrate your EEG data before launching the game. Please relax for a moment.")
        
        # Start a session in the database
        session_type_for_db = "Meditation-Unity"
        target_metric_for_db = "Relaxation"
        
        # Launch backend and calibrate BEFORE starting Unity
        backend_script_path = "eeg_backend_processor.py"
        try:
            print(f"Launching backend script for calibration: {backend_script_path}")
            self.backend_process = subprocess.Popen(
                [sys.executable, "-u", backend_script_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            print(f"Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Error: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            return
        
        # Start a calibration dialog with progress bar
        self.calibration_dialog = QtWidgets.QProgressDialog("Calibrating EEG data...", None, 0, 100, self)
        self.calibration_dialog.setWindowTitle("Calibrating for Unity Game")
        self.calibration_dialog.setWindowModality(Qt.WindowModal)
        self.calibration_dialog.setMinimumDuration(0)
        self.calibration_dialog.setValue(0)
        self.calibration_dialog.setAutoClose(True)
        self.calibration_dialog.show()
        
        # Create and start the prediction subscriber
        self.prediction_subscriber = EEGPredictionSubscriber()
        self.prediction_thread = QtCore.QThread()
        self.prediction_subscriber.moveToThread(self.prediction_thread)
        
        # Connect signals
        self.prediction_subscriber.calibration_progress.connect(self.on_unity_calibration_progress)
        self.prediction_subscriber.calibration_status.connect(self.on_unity_calibration_status)
        self.prediction_subscriber.new_prediction_received.connect(self.on_new_eeg_prediction)
        
        # Start thread
        self.prediction_thread.started.connect(self.prediction_subscriber.run)
        self.prediction_thread.start()
        
        # Request calibration
        QtCore.QTimer.singleShot(1000, lambda: self.prediction_subscriber.start_relaxation_session())

    def on_unity_calibration_progress(self, progress):
        """Update calibration dialog with progress"""
        if hasattr(self, 'calibration_dialog'):
            self.calibration_dialog.setValue(int(progress * 100))

    def on_unity_calibration_status(self, status, baseline_data):
        """Handle calibration completion for Unity game"""
        if self.backend_process and self.backend_process.poll() is not None:
            # Backend has terminated unexpectedly
            QtWidgets.QMessageBox.critical(self, "Backend Error", 
                "The EEG backend process has terminated unexpectedly. Please try again.")
            self.stop_unity_session()
            return
        if status == "COMPLETED":
            # Setup OSC client
            self.is_calibrating = False
            self.is_calibrated = True
            try:
                self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)
                print(f"Created OSC client to Unity at {UNITY_IP}:{UNITY_OSC_PORT}")
                
                # Send initial data immediately
                self.client.send_message(UNITY_OSC_ADDRESS, 50.0)
                self.client.send_message(UNITY_OSC_SCENE_ADDRESS, 0)
                print("Sent initial data to Unity")
                
                # IMPORTANT: Wait a moment for the first few predictions to arrive
                QtWidgets.QMessageBox.information(self, "Calibration Complete",
                    "EEG calibration complete! The Unity game will launch in 5 seconds.\n"
                    "This delay allows the EEG system to start generating predictions.")
                    
                # Use the delay to process events and gather initial predictions
                for i in range(5):
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(1)
                    print(f"Waiting for predictions... ({i+1}/5)")
                
                # Close calibration dialog
                if hasattr(self, 'calibration_dialog'):
                    self.calibration_dialog.close()
                
                # Start database session
                self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
                    self.user_id, "Meditation-Unity", "Relaxation"
                )
                
                # Now launch Unity game
                unity_game_path = r"C:\Users\berna\OneDrive\Documentos\GitHub\NeuroFlow\game\NeuroFlow.exe"
                if not os.path.exists(unity_game_path):
                    msg = f"Could not find the Unity game executable at:\n{unity_game_path}"
                    print(f"Error: {msg}")
                    QtWidgets.QMessageBox.warning(self, "Error", msg)
                    self.stop_unity_session()
                    return
                    
                try:
                    print(f"Launching Unity game: {unity_game_path}")
                    subprocess.Popen([unity_game_path])
                    
                    # Set session as active
                    self.session_goal = "UNITY_GAME"
                    self.update_button_states(self.main_app_window.is_lsl_connected)
                    
                    # Start heartbeat timer
                    self.unity_data_timer = QtCore.QTimer(self)
                    self.unity_data_timer.timeout.connect(self.send_unity_heartbeat)
                    self.unity_data_timer.start(2000)  # Every 2 seconds
                    
                    QtWidgets.QMessageBox.information(self, "Game Launched",
                        "Unity game is launching with EEG connection ready.\nClose this message to continue.")
                        
                except Exception as e:
                    print(f"Error launching game: {e}")
                    QtWidgets.QMessageBox.critical(self, "Error", f"Failed to launch Unity game:\n{e}")
                    self.stop_unity_session()
            except Exception as e:
                print(f"Error setting up OSC client: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to setup OSC connection:\n{e}")
                self.stop_unity_session()
        
        elif status == "FAILED":
            QtWidgets.QMessageBox.critical(self, "Calibration Failed", 
                "Failed to calibrate EEG. Please check your Muse connection and try again.")
            self.stop_unity_session()

    def send_unity_heartbeat(self):
        """Send regular data to prevent Unity from timing out"""
        if self.client and self.session_goal == "UNITY_GAME":
            try:
                # Get the most recent prediction data
                if hasattr(self, 'last_prediction') and self.last_prediction:
                    smooth_value = self.last_prediction.get("smooth_value", 0.5)
                    # Log what we're actually using
                    print(f"Using actual prediction value: {smooth_value:.2f}")
                else:
                    smooth_value = 0.5  # Default value
                    print("No prediction data available, using default value: 0.5")
                    
                # Send the data
                scaled_value = smooth_value * 100.0
                self.client.send_message(UNITY_OSC_ADDRESS, scaled_value)
                print(f"Sent to Unity: {UNITY_OSC_ADDRESS} = {scaled_value:.1f}")
                
                # Also send scene index occasionally
                if random.random() < 0.2:  # ~20% chance to send scene update
                    scene_index = 0
                    if scaled_value >= 80:
                        scene_index = 2
                    elif scaled_value >= 50:
                        scene_index = 1
                        
                    self.client.send_message(UNITY_OSC_SCENE_ADDRESS, scene_index)
                    print(f"Sent scene index to Unity: {scene_index}")
                    
            except Exception as e:
                print(f"Error sending Unity data: {e}")

    def stop_unity_session(self):
        """Clean up Unity game session resources"""
        print("Stopping Unity session and cleaning up resources...")
        
        # Stop heartbeat timer
        if hasattr(self, 'unity_data_timer') and self.unity_data_timer.isActive():
            self.unity_data_timer.stop()
        
        # Stop prediction subscriber
        if self.prediction_subscriber:
            print("Stopping prediction subscriber...")
            try:
                self.prediction_subscriber.stop()
                if self.prediction_thread and self.prediction_thread.isRunning():
                    if not self.prediction_thread.wait(2000):  # Wait 2 seconds max
                        print("Forcefully terminating prediction thread...")
                        self.prediction_thread.terminate()
            except Exception as e:
                print(f"Error stopping subscriber: {e}")
        
        # Kill backend process - BE MORE AGGRESSIVE
        if self.backend_process:
            print(f"Terminating backend process PID: {self.backend_process.pid}")
            try:
                # First try graceful termination
                if os.name == 'nt':  # Windows
                    os.kill(self.backend_process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self.backend_process.send_signal(signal.SIGINT)
                
                # Wait a short time for graceful exit
                time.sleep(1)
                
                # If still running, force kill
                if self.backend_process.poll() is None:
                    print("Force killing backend process...")
                    if os.name == 'nt':
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.backend_process.pid)])
                    else:
                        os.kill(self.backend_process.pid, signal.SIGKILL)
            except Exception as e:
                print(f"Error terminating backend: {e}")
        
        # Run ZMQ cleanup as safety measure
        self._force_cleanup_zmq_ports()
        
        # End database session
        if self.current_session_id:
            db_manager.end_session(self.current_session_id)
            self.current_session_id = None
        
        # Reset state
        self.session_goal = None
        self.update_button_states(self.main_app_window.is_lsl_connected)
        
        print("Unity session cleanup complete")

    def _force_cleanup_zmq_ports(self):
        """Force cleanup of any processes using our ZMQ ports"""
        print("Performing emergency ZMQ port cleanup...")
        
        # First try using the ZMQ cleanup utility
        try:
            from backend.zmq_port_cleanup import cleanup_all_zmq_ports
            cleanup_all_zmq_ports()
        except Exception as e:
            print(f"ZMQ utility cleanup failed: {e}")
        
        # As a backup, find and kill any Python processes with eeg_backend in the name
        try:
            if os.name == 'nt':  # Windows
                # Find PIDs of Python processes running our backend
                output = subprocess.check_output('tasklist /FI "IMAGENAME eq python.exe" /FO CSV', shell=True).decode()
                lines = output.strip().split('\n')[1:]  # Skip header
                
                for line in lines:
                    if 'eeg_backend' in line.lower():
                        try:
                            pid = int(line.split(',')[1].strip('"'))
                            print(f"Killing Python process with PID {pid}")
                            os.kill(pid, signal.SIGTERM)
                        except:
                            pass
            else:  # Unix-like
                os.system("pkill -f eeg_backend_processor")
        except:
            pass
        
        # Just to be safe, give the system a moment to fully release ports
        time.sleep(1)