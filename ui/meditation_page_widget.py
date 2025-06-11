import signal
import sys
import os
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from datetime import datetime
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from ui.video_player_window import VideoPlayerWindow
from backend.eeg_processing_worker import EEGProcessingWorker
from backend import database_manager as db_manager
from pythonosc.udp_client import SimpleUDPClient
from ui.signal_quality_widget import SignalQualityWidget
from backend.signal_quality_validator import SignalQualityValidator

UNITY_IP = "127.0.0.1"
UNITY_OSC_PORT = 9000
UNITY_OSC_ADDRESS = "/muse/relaxation"
UNITY_OSC_SCENE_ADDRESS = "/muse/scene"


class MeditationPageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, main_app_window_ref=None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.video_player_window = None
        self.current_session_id = None
        self.current_session_start_time = None
        self.session_target_label = ""
        self.is_calibrating = False
        self.is_calibrated = False
        self.session_goal = None
        self.user_id = None
        self.last_sent_scene_index = -1
        self.UNITY_OSC_SCENE_ADDRESS = "/muse/scene"
        
        # Threading components
        self.eeg_thread = None
        self.eeg_worker = None
        
        # UI state
        self.calibration_progress_value = 0
        self.last_prediction = None
        
        self.initUI()

        self.signal_quality_validator = SignalQualityValidator()
        self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)

        if self.main_app_window:
            self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
            print("Warning: MeditationPageWidget initialized without a valid main_app_window reference.")
            self.update_button_states(False)

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

        # Video teaser
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

        # Game teaser
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

        self.btn_start_unity_game = QtWidgets.QPushButton("Launch Unity Game")
        self.btn_start_unity_game.setStyleSheet("font-size: 11pt; padding: 8px 15px;")
        self.btn_start_unity_game.clicked.connect(self.launch_unity_game)
        game_teaser_layout.addWidget(self.btn_start_unity_game)

        self.latency_test_button = QtWidgets.QPushButton("Test EEG Processing")
        self.latency_test_button.clicked.connect(self.test_eeg_processing)
        game_teaser_layout.addWidget(self.latency_test_button)

        teasers_layout.addLayout(game_teaser_layout)
        self.main_layout.addLayout(teasers_layout)
        self.main_layout.addStretch(1)

    def _setup_eeg_worker(self):
        """Set up the EEG processing worker and thread"""
        if self.eeg_worker is not None:
            print("EEG worker already exists")
            return True

        print("Setting up EEG processing worker...")
        
        try:
            # Create thread and worker
            self.eeg_thread = QtCore.QThread()
            self.eeg_worker = EEGProcessingWorker()
            
            # Move worker to thread
            self.eeg_worker.moveToThread(self.eeg_thread)
            
            # Connect worker signals
            self.eeg_worker.connection_status_changed.connect(self.on_connection_status_changed)
            self.eeg_worker.calibration_progress.connect(self.on_calibration_progress)
            self.eeg_worker.calibration_status_changed.connect(self.on_calibration_status_changed)
            self.eeg_worker.new_prediction.connect(self.on_new_eeg_prediction)
            self.eeg_worker.signal_quality_update.connect(self.on_signal_quality_update)
            self.eeg_worker.error_occurred.connect(self.on_eeg_error)
            self.eeg_worker.session_data_ready.connect(self.on_session_data_ready)
            
            # Connect thread signals
            self.eeg_thread.started.connect(self.eeg_worker.initialize)
            self.eeg_thread.finished.connect(self.eeg_worker.cleanup)
            
            # Start the thread
            self.eeg_thread.start()
            
            print("EEG worker setup complete")
            return True
            
        except Exception as e:
            print(f"Error setting up EEG worker: {e}")
            self._cleanup_eeg_worker()
            return False

    def _cleanup_eeg_worker(self):
        """Clean up EEG worker and thread"""
        print("Cleaning up EEG worker...")
        
        if self.eeg_worker:
            # Stop any active session
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)
            
            # Disconnect signals
            try:
                self.eeg_worker.connection_status_changed.disconnect()
                self.eeg_worker.calibration_progress.disconnect()
                self.eeg_worker.calibration_status_changed.disconnect()
                self.eeg_worker.new_prediction.disconnect()
                self.eeg_worker.signal_quality_update.disconnect()
                self.eeg_worker.error_occurred.disconnect()
                self.eeg_worker.session_data_ready.disconnect()
            except TypeError:
                pass  # Already disconnected
            
            self.eeg_worker = None
        
        if self.eeg_thread:
            if self.eeg_thread.isRunning():
                self.eeg_thread.quit()
                if not self.eeg_thread.wait(3000):  # Wait up to 3 seconds
                    print("Warning: EEG thread did not quit gracefully")
                    self.eeg_thread.terminate()
                    self.eeg_thread.wait()
            self.eeg_thread = None
        
        print("EEG worker cleanup complete")

    def update_button_states(self, is_lsl_connected):
        is_session_active = bool(self.session_goal)

        if hasattr(self, 'btn_start_video_feedback'):
            self.btn_start_video_feedback.setEnabled(is_lsl_connected and not is_session_active)
            if not is_lsl_connected:
                self.btn_start_video_feedback.setToolTip("Muse must be connected.")
            elif is_session_active:
                self.btn_start_video_feedback.setToolTip("A session is already active.")
            else:
                self.btn_start_video_feedback.setToolTip("")

        if hasattr(self, 'btn_start_unity_game'):
            self.btn_start_unity_game.setEnabled(is_lsl_connected and not is_session_active)
            if not is_lsl_connected:
                self.btn_start_unity_game.setToolTip("Muse must be connected.")
            elif is_session_active:
                 self.btn_start_unity_game.setToolTip("A video feedback session is active.")
            else:
                self.btn_start_unity_game.setToolTip("")

    def start_video_session(self):
        """Start a video feedback session using the threaded EEG worker"""
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected", "Cannot start session.")
            return
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In", "You must be logged in to start a session.")
            return
        if self.session_goal:
            QtWidgets.QMessageBox.warning(self, "Session Active", "A video feedback session is already running.")
            return

        print("Meditation Page: Start Video Feedback clicked.")
        
        # Set up EEG worker if not already done
        if not self._setup_eeg_worker():
            QtWidgets.QMessageBox.critical(self, "EEG Setup Error", "Failed to initialize EEG processing system.")
            return

        self.session_goal = "RELAXATION"
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        self.signal_quality_validator.reset()
        
        # Create video player window with signal quality widget
        if not self.video_player_window:
            self.video_player_window = VideoPlayerWindow(parent=self)
            self.video_player_window.session_stopped.connect(self.handle_video_session_stopped_signal)
            
            # Connect recalibration signal
            self.video_player_window.recalibration_requested.connect(self.handle_recalibration_request)
        
        # Add signal quality widget to video player
        if not hasattr(self.video_player_window, 'signal_quality_widget'):
            self.video_player_window.signal_quality_widget = SignalQualityWidget()
            self.video_player_window.signal_quality_widget.recalibrate_requested.connect(self.handle_recalibration_request)
            self.video_player_window.add_signal_quality_widget(self.video_player_window.signal_quality_widget)

        self.video_player_window.set_status("Connecting to EEG...")
        self.video_player_window.show_calibration_progress(0)
        self.video_player_window.show()
        self.video_player_window.activateWindow()

        self.session_target_label = "Relaxed"
        session_type_for_db = "Meditation-Video"
        target_metric_for_db = "Relaxation"

        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            self.user_id, session_type_for_db, target_metric_for_db
        )

        self.update_button_states(self.main_app_window.is_lsl_connected)
        
        # Start the EEG session
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "start_session", 
                                      QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "RELAXATION"))

    def handle_video_session_stopped_signal(self):
        """Called when video player emits session_stopped"""
        print("MeditationPage: Received session_stopped signal from VideoPlayerWindow.")
        self.stop_video_session_logic()


    def _reset_ui_and_state(self):
        """Resets UI elements and internal state flags after a session."""
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.calibration_progress_value = 0

        if self.video_player_window:
            print("MeditationPage: _reset_ui_and_state closing video_player_window.")
            try:
                self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped_signal)
                self.video_player_window.recalibration_requested.disconnect(self.handle_recalibration_request)
            except TypeError:
                pass
            self.video_player_window.close()
            self.video_player_window = None

        if hasattr(self.main_app_window, 'is_lsl_connected'):
             self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
             self.update_button_states(False)

    def handle_recalibration_request(self):
        """Handle user request to recalibrate due to poor signal quality"""
        print("MeditationPageWidget: Recalibration requested by user")
        
        reply = QtWidgets.QMessageBox.question(
            self, 
            "Recalibrate EEG?", 
            "This will restart the calibration process due to poor signal quality.\n\n"
            "Please adjust your headband and ensure good electrode contact.\n\n"
            "Continue with recalibration?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self._restart_calibration()

    def _restart_calibration(self):
        """Restart the calibration process"""
        print("MeditationPageWidget: Restarting calibration...")
        
        # Reset calibration state
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        
        # Reset UI elements
        if self.video_player_window:
            self.video_player_window.set_status("Restarting calibration...")
            self.video_player_window.show_calibration_progress(0)
            self.video_player_window.show_signal_quality_panel()
            
            # Reset signal quality widget
            if self.video_player_window.signal_quality_widget:
                self.video_player_window.signal_quality_widget.reset()
        
        # Request recalibration from worker
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "recalibrate", QtCore.Qt.QueuedConnection)


    @QtCore.pyqtSlot(float)
    def on_calibration_progress(self, progress):
        """Handle calibration progress updates from EEG worker"""
        if not self.is_calibrating:
            return

        self.calibration_progress_value = int(progress * 100)
        if self.video_player_window:
            self.video_player_window.show_calibration_progress(self.calibration_progress_value)
            
            if self.calibration_progress_value % 10 == 0 or self.calibration_progress_value >= 100:
                self.video_player_window.set_status(f"Calibrating EEG: {self.calibration_progress_value}% complete")

    @QtCore.pyqtSlot(str, dict)
    def on_calibration_status_changed(self, status, data):
        """Handle calibration status changes from EEG worker with better error handling"""
        print(f"Calibration Status: {status}, Data: {data}")
        
        # Check if video player window still exists before using it
        if not self.video_player_window:
            print("Warning: Received calibration status but video player window is None")
            return
        
        if status == "PAUSED":
            self.video_player_window.set_status("Poor signal quality - adjust headband")
            
        elif status == "WAITING_FOR_QUALITY":
            message = data.get('message', 'Waiting for better signal quality...')
            self.video_player_window.set_status(message)
            
        elif status == "COMPLETED":
            self.is_calibrating = False
            self.is_calibrated = True
            self.video_player_window.set_status("Calibration complete. Starting session...")
            self.video_player_window.hide_calibration_progress_bar()
            
            if self.session_goal == "RELAXATION":
                self.video_player_window.start_relaxation_video()
            elif self.session_goal == "FOCUS":
                self.video_player_window.start_focus_video()
                
        elif status == "FAILED":
            self.is_calibrating = False
            self.is_calibrated = False
            error_msg = data.get('error_message', 'Calibration failed')
            
            # Show error message before stopping session
            QtWidgets.QMessageBox.warning(self, "Calibration Failed", 
                f"{error_msg}\n\nPlease check:\n"
                "• Muse headband is connected\n"
                "• LSL stream is running\n"
                "• Electrodes have good contact")
            
            # Stop session after user acknowledges error
            self.stop_video_session_logic(triggered_by_error=True)
            
        elif status == "STARTED":
            self.is_calibrating = True
            # Only update UI if video player window exists
            if self.video_player_window:
                self.video_player_window.set_status("Calibration process initiated...")
                self.video_player_window.show_calibration_progress(0)

    @QtCore.pyqtSlot(str, str)
    def on_connection_status_changed(self, status, message):
        """Handle connection status updates from EEG worker with better error handling"""
        print(f"EEG Connection Status: {status} - {message}")
        
        # Check if video player window still exists
        if not self.video_player_window:
            print("Warning: Received connection status but video player window is None")
            return
        
        if status == "CONNECTED":
            self.video_player_window.set_status("EEG Connected. Preparing calibration...")
        elif status == "CONNECTING":
            self.video_player_window.set_status(f"Connecting: {message}")
        elif status == "ERROR":
            self.video_player_window.set_status(f"Connection Error: {message}")
            
            # Show detailed error message
            QtWidgets.QMessageBox.critical(self, "EEG Connection Error", 
                f"Failed to connect to EEG device:\n{message}\n\n"
                f"Please check:\n"
                f"• Your Muse headband is powered on\n"
                f"• Muse is paired and connected\n"
                f"• LSL stream is running (MuseSimulator or MuseLSL)\n"
                f"• No other applications are using the Muse")
            
            # Stop session after error
            self.stop_video_session_logic(triggered_by_error=True)

    def stop_video_session_logic(self, triggered_by_error=False):
        """Core logic to stop the video session with better cleanup"""
        print("Meditation Page: Stopping video session logic...")

        # Stop the EEG worker session first
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)

        # Close video player window with null checks
        if self.video_player_window:
            try:
                # Disconnect signals safely
                try:
                    self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped_signal)
                except (TypeError, RuntimeError):
                    pass  # Already disconnected or widget destroyed
                
                try:
                    self.video_player_window.recalibration_requested.disconnect(self.handle_recalibration_request)
                except (TypeError, RuntimeError):
                    pass
                
                # Close window if still visible and not triggered by error
                if not triggered_by_error and self.video_player_window.isVisible():
                    print("MeditationPage: Closing video player window...")
                    self.video_player_window.close()
                
            except RuntimeError:
                # Widget was already destroyed
                print("MeditationPage: Video player window was already destroyed")
            
            # Set to None regardless
            self.video_player_window = None

        # End database session
        if self.current_session_id:
            db_manager.end_session(self.current_session_id)
            print(f"Meditation Page: Session {self.current_session_id} ended in DB.")
            self.current_session_id = None
            self.current_session_start_time = None

        self._reset_ui_and_state()
        print("Meditation Page: Video session stopped logic completed.")

    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction(self, prediction_data):
        """Handle new EEG predictions from worker"""
        if self.is_calibrating or not self.is_calibrated:
            return
            
        if prediction_data.get("message_type") != "PREDICTION":
            return

        classification = prediction_data.get("classification", {})
        state = classification.get("state", "Unknown")
        level = classification.get("level", 0)
        smooth_value = classification.get("smooth_value", 0.5)
        state_key = classification.get("state_key", "neutral")

        self.last_prediction = classification

        print(f"Got new prediction: state={state}, level={level}, smooth={smooth_value:.2f}")

        # Send to Unity if connected
        if self.client:
            scaled_relaxation_level = smooth_value * 100.0
            try:
                self.client.send_message(UNITY_OSC_ADDRESS, scaled_relaxation_level)
            except Exception as e:
                print(f"Error sending relaxation OSC message to Unity: {e}")

            target_scene_index = -1
            if scaled_relaxation_level >= 80:
                target_scene_index = 2
            elif scaled_relaxation_level >= 50:
                target_scene_index = 1
            else:
                target_scene_index = 0

            if target_scene_index != -1 and target_scene_index != self.last_sent_scene_index:
                try:
                    self.client.send_message(self.UNITY_OSC_SCENE_ADDRESS, target_scene_index)
                    self.last_sent_scene_index = target_scene_index
                except Exception as e:
                    print(f"Error sending scene OSC message to Unity: {e}")

        # Update video feedback
        if self.video_player_window and self.video_player_window.isVisible():
            self.update_video_feedback(state, level, smooth_value, state_key)

        # Store session metrics
        if self.current_session_id:
            is_on_target = (self.session_goal == "RELAXATION" and level > 0) or \
                           (self.session_goal == "FOCUS" and level > 0)
            db_manager.add_session_metric(self.current_session_id, state, is_on_target, smooth_value)

    @QtCore.pyqtSlot(dict)
    def on_signal_quality_update(self, quality_data):
        """Handle signal quality updates from EEG worker"""
        if (self.video_player_window and 
            hasattr(self.video_player_window, 'signal_quality_widget') and
            self.video_player_window.signal_quality_widget):
            
            # Update the signal quality widget
            self.video_player_window.signal_quality_widget.update_metrics(quality_data)

    @QtCore.pyqtSlot(str)
    def on_eeg_error(self, error_message):
        """Handle errors from EEG worker"""
        print(f"EEG Worker Error: {error_message}")
        QtWidgets.QMessageBox.warning(self, "EEG Processing Error", error_message)
        
        if "fatal" in error_message.lower() or "cannot recover" in error_message.lower():
             self.stop_video_session_logic(triggered_by_error=True)

    @QtCore.pyqtSlot(dict)
    def on_session_data_ready(self, session_data):
        """Handle session data from EEG worker for saving"""
        if self.current_session_id:
            print(f"Saving session data for session {self.current_session_id}")
            
            # Save band data to database
            try:
                bands_dict = {
                    "session_id": self.current_session_id,
                    "alpha": session_data["band_data"]["alpha"],
                    "beta": session_data["band_data"]["beta"],
                    "theta": session_data["band_data"]["theta"],
                    "ab_ratio": session_data["band_data"]["ab_ratio"],
                    "bt_ratio": session_data["band_data"]["bt_ratio"],
                    "timestamps": session_data["timestamps"]
                }
                
                db_manager.save_session_band_data(bands_dict)
                
                # Optionally save EEG data
                if session_data["eeg_data"]:
                    db_manager.save_session_eeg_data(
                        self.current_session_id, 
                        session_data["eeg_data"], 
                        session_data["timestamps"]
                    )
                    
                print("Session data saved successfully")
                
            except Exception as e:
                print(f"Error saving session data: {e}")

    def update_video_feedback(self, state, level, smooth_value, state_key):
        """Update video feedback based on EEG state"""
        if not self.session_goal or not self.video_player_window:
            return

        if self.session_goal == "RELAXATION":
            if level <= -3:
                scene, status_msg = "very_tense", f"{state} (Try to relax)"
            elif level == -2:
                scene, status_msg = "tense", f"{state} (Breathe deeply)"
            elif level == -1:
                scene, status_msg = "less_relaxed", f"{state} (Find calmness)"
            elif level == 0:
                scene, status_msg = "neutral", f"{state} (Continue relaxing)"
            elif level == 1:
                scene, status_msg = "slightly_relaxed", f"{state} (Good start)"
            elif level == 2:
                scene, status_msg = "moderately_relaxed", f"{state} (Well done)"
            elif level == 3:
                scene, status_msg = "strongly_relaxed", f"{state} (Excellent)"
            else:
                scene, status_msg = "deeply_relaxed", f"{state} (Perfect!)"
            
            self.video_player_window.set_scene(scene)
            self.video_player_window.set_status(f"Status: {status_msg}")
            self.video_player_window.set_relaxation_level(smooth_value)

        elif self.session_goal == "FOCUS":
            if level <= -3:
                scene, status_msg = "very_distracted", f"{state} (Try to refocus)"
            elif level == -2:
                scene, status_msg = "distracted", f"{state} (Clear your mind)"
            elif level == -1:
                scene, status_msg = "less_focused", f"{state} (Concentrate)"
            elif level == 0:
                scene, status_msg = "neutral", f"{state} (Find your focus)"
            elif level == 1:
                scene, status_msg = "slightly_focused", f"{state} (Good start)"
            elif level == 2:
                scene, status_msg = "moderately_focused", f"{state} (Well done)"
            elif level == 3:
                scene, status_msg = "strongly_focused", f"{state} (Excellent)"
            else:
                scene, status_msg = "deeply_focused", f"{state} (Perfect focus!)"

            self.video_player_window.set_scene(scene)
            self.video_player_window.set_status(f"Status: {status_msg}")
            self.video_player_window.set_focus_level(smooth_value)

    def launch_unity_game(self):
        """Launch Unity game with EEG connection"""
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In", "You must be logged in to start a session.")
            return
        if self.session_goal:
            QtWidgets.QMessageBox.warning(self, "Session Active", "A session is already running.")
            return

        print("Meditation Page: Launch Unity Game clicked.")
        
        # Set up EEG worker if not already done
        if not self._setup_eeg_worker():
            QtWidgets.QMessageBox.critical(self, "EEG Setup Error", "Failed to initialize EEG processing system.")
            return

        QtWidgets.QMessageBox.information(self, "Calibration Required", 
            "We'll first calibrate your EEG data before launching the game. Please relax for a moment.")
        
        self.session_goal = "UNITY_GAME"
        self.is_calibrating = True
        self.is_calibrated = False
        
        # Start database session
        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            self.user_id, "Meditation-Unity", "Relaxation"
        )
        
        # Show calibration dialog
        self.calibration_dialog = QtWidgets.QProgressDialog("Calibrating EEG data...", None, 0, 100, self)
        self.calibration_dialog.setWindowTitle("Calibrating for Unity Game")
        self.calibration_dialog.setWindowModality(Qt.WindowModal)
        self.calibration_dialog.setMinimumDuration(0)
        self.calibration_dialog.setValue(0)
        self.calibration_dialog.setAutoClose(True)
        self.calibration_dialog.show()
        
        # Connect calibration signals for Unity
        self.eeg_worker.calibration_progress.connect(self.on_unity_calibration_progress)
        self.eeg_worker.calibration_status_changed.connect(self.on_unity_calibration_status)
        
        # Start calibration
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "start_session", 
                                      QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "RELAXATION"))

    def on_unity_calibration_progress(self, progress):
        """Update calibration dialog with progress"""
        if hasattr(self, 'calibration_dialog'):
            self.calibration_dialog.setValue(int(progress * 100))

    def on_unity_calibration_status(self, status, baseline_data):
        """Handle calibration completion for Unity game"""
        if status == "COMPLETED":
            self.is_calibrating = False
            self.is_calibrated = True
            
            try:
                # Close calibration dialog
                if hasattr(self, 'calibration_dialog'):
                    self.calibration_dialog.close()
                
                # Setup OSC client
                self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)
                print(f"Created OSC client to Unity at {UNITY_IP}:{UNITY_OSC_PORT}")
                
                # Send initial data
                self.client.send_message(UNITY_OSC_ADDRESS, 50.0)
                self.client.send_message(UNITY_OSC_SCENE_ADDRESS, 0)
                print("Sent initial data to Unity")
                
                # Wait and process events
                QtWidgets.QMessageBox.information(self, "Calibration Complete",
                    "EEG calibration complete! The Unity game will launch in 5 seconds.\n"
                    "This delay allows the EEG system to start generating predictions.")
                    
                for i in range(5):
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(1)
                    print(f"Waiting for predictions... ({i+1}/5)")
                
                # Launch Unity game
                unity_game_path = r"C:\Users\berna\OneDrive\Documentos\GitHub\NeuroFlow\game\NeuroFlow.exe"
                if not os.path.exists(unity_game_path):
                    msg = f"Could not find the Unity game executable at:\n{unity_game_path}"
                    print(f"Error: {msg}")
                    QtWidgets.QMessageBox.warning(self, "Error", msg)
                    self.stop_unity_session()
                    return
                    
                print(f"Launching Unity game: {unity_game_path}")
                subprocess.Popen([unity_game_path])
                
                # Start heartbeat timer
                self.unity_data_timer = QtCore.QTimer(self)
                self.unity_data_timer.timeout.connect(self.send_unity_heartbeat)
                self.unity_data_timer.start(2000)  # Every 2 seconds
                
                self.update_button_states(self.main_app_window.is_lsl_connected)
                
                QtWidgets.QMessageBox.information(self, "Game Launched",
                    "Unity game is launching with EEG connection ready.\nClose this message to continue.")
                    
            except Exception as e:
                print(f"Error setting up Unity game: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to setup Unity game:\n{e}")
                self.stop_unity_session()
        
        elif status == "FAILED":
            QtWidgets.QMessageBox.critical(self, "Calibration Failed", 
                "Failed to calibrate EEG. Please check your Muse connection and try again.")
            self.stop_unity_session()

    def send_unity_heartbeat(self):
        """Send regular data to Unity to prevent timeout"""
        if self.client and self.session_goal == "UNITY_GAME":
            try:
                if hasattr(self, 'last_prediction') and self.last_prediction:
                    smooth_value = self.last_prediction.get("smooth_value", 0.5)
                    print(f"Using actual prediction value: {smooth_value:.2f}")
                else:
                    smooth_value = 0.5
                    print("No prediction data available, using default value: 0.5")
                    
                scaled_value = smooth_value * 100.0
                self.client.send_message(UNITY_OSC_ADDRESS, scaled_value)
                print(f"Sent to Unity: {UNITY_OSC_ADDRESS} = {scaled_value:.1f}")
                
                if random.random() < 0.2:  # 20% chance to send scene update
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
        
        # Stop EEG worker session
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)
        
        # End database session
        if self.current_session_id:
            db_manager.end_session(self.current_session_id)
            self.current_session_id = None
        
        # Reset state
        self.session_goal = None
        self.update_button_states(self.main_app_window.is_lsl_connected)
        
        print("Unity session cleanup complete")

    def test_eeg_processing(self):
        """Test EEG processing system"""
        if not self._setup_eeg_worker():
            QtWidgets.QMessageBox.critical(self, "EEG Setup Error", "Failed to initialize EEG processing system.")
            return
        
        # Connect to LSL and run a basic test
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "connect_to_lsl", QtCore.Qt.QueuedConnection)
        
        QtWidgets.QMessageBox.information(self, "EEG Test", 
            "Testing EEG connection. Check console for status updates.")

    def clean_up_session(self):
        """Clean up sessions on main window close"""
        print("Meditation Page: clean_up_session called")
        
        if self.session_goal:
            if self.session_goal in ["RELAXATION", "FOCUS"]:
                self.stop_video_session_logic(triggered_by_error=False)
            elif self.session_goal == "UNITY_GAME":
                self.stop_unity_session()
        
        # Clean up EEG worker
        self._cleanup_eeg_worker()