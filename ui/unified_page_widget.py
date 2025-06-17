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
UNITY_OSC_ADDRESS_RELAXATION = "/muse/relaxation"
UNITY_OSC_ADDRESS_FOCUS = "/neuroflow/focus"
UNITY_OSC_SCENE_ADDRESS = "/muse/scene"


class UnifiedEEGPageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, main_app_window_ref=None, page_type="meditation"):
        super().__init__(parent)
        self.page_type = page_type  # "meditation" or "focus"
        self.main_app_window = main_app_window_ref
        
        # Common attributes
        self.video_player_window = None
        self.current_session_id = None
        self.current_session_start_time = None
        self.session_target_label = ""
        self.is_calibrating = False
        self.is_calibrated = False
        self.session_goal = None
        self.user_id = None
        self.last_sent_scene_index = -1
        
        # Threading components
        self.eeg_thread = None
        self.eeg_worker = None
        
        # UI state
        self.calibration_progress_value = 0
        self.last_prediction = None
        
        # Focus-specific attributes
        self.work_monitor_window = None
        self.focus_monitoring_active = False
        self.focus_history = []
        self.focus_alert_shown = False
        self.focus_drop_counter = 0
        self.session_timer = None
        self.session_start_time = None
        self.focus_monitor_timer = None
        self.unity_data_timer = None
        
        # Page-specific configuration
        self._setup_page_config()
        
        self.initUI()
        
        self.signal_quality_validator = SignalQualityValidator()
        self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)

        if self.main_app_window:
            self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
            print(f"Warning: {self.page_type.title()}PageWidget initialized without main_app_window reference.")
            self.update_button_states(False)

    def _setup_page_config(self):
        """Configure page-specific settings"""
        if self.page_type == "meditation":
            self.primary_color = "#3498db"
            self.page_title = "Choose Your Meditation Experience"
            self.eeg_session_type = "RELAXATION"
            self.unity_osc_address = UNITY_OSC_ADDRESS_RELAXATION
        else:  # focus
            self.primary_color = "#8A2BE2"
            self.page_title = "Choose Your Focus Session"
            self.eeg_session_type = "FOCUS"
            self.unity_osc_address = UNITY_OSC_ADDRESS_FOCUS

    def initUI(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        # Dynamic title based on page type
        title_label = QtWidgets.QLabel(self.page_title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(20)

        if self.page_type == "meditation":
            self._setup_meditation_ui()
        else:
            self._setup_focus_ui()

        self.main_layout.addStretch(1)

    def _setup_meditation_ui(self):
        """Setup meditation-specific UI elements"""
        teasers_layout = QtWidgets.QHBoxLayout()

        # Video meditation option
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
        self.btn_start_video_feedback.clicked.connect(lambda: self.start_session("video"))
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
        self.btn_start_unity_game.clicked.connect(lambda: self.start_session("unity"))
        game_teaser_layout.addWidget(self.btn_start_unity_game)

        self.latency_test_button = QtWidgets.QPushButton("Test EEG Processing")
        self.latency_test_button.clicked.connect(self.test_eeg_processing)
        game_teaser_layout.addWidget(self.latency_test_button)

        teasers_layout.addLayout(game_teaser_layout)
        self.main_layout.addLayout(teasers_layout)

    def _setup_focus_ui(self):
        """Setup focus-specific UI elements"""
        # First row of focus options (Work, Video)
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(30)

        # Work session option
        work_focus_layout = self._create_focus_option_layout(
            title="Work Session",
            image_path="./assets/work.jpg",
            button_text="Start",
            action_slot=lambda: self.start_session("work")
        )
        row1_layout.addLayout(work_focus_layout)

        # Video session option
        video_focus_layout = self._create_focus_option_layout(
            title="Video Session",
            image_path="./assets/focus.jpg",
            button_text="Start",
            action_slot=lambda: self.start_session("video")
        )
        row1_layout.addLayout(video_focus_layout)
        
        self.main_layout.addLayout(row1_layout)
        self.main_layout.addSpacing(25)

        # Second row - Game session (centered)
        row2_outer_layout = QtWidgets.QHBoxLayout()
        row2_outer_layout.addStretch(1)

        game_focus_layout = self._create_focus_option_layout(
            title="Game Session",
            image_path="./assets/focus_game.jpg",
            button_text="Start",
            action_slot=lambda: self.start_session("unity"),
            is_single_item_row=True
        )
        row2_outer_layout.addLayout(game_focus_layout)
        row2_outer_layout.addStretch(1)
        
        self.main_layout.addLayout(row2_outer_layout)

    def _create_focus_option_layout(self, title, image_path, button_text, action_slot, is_single_item_row=False):
        """Helper function to create a consistent layout for each focus option."""
        option_layout = QtWidgets.QVBoxLayout()
        option_layout.setAlignment(QtCore.Qt.AlignCenter)

        title_label = QtWidgets.QLabel(title)
        title_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Medium))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        option_layout.addWidget(title_label)
        option_layout.addSpacing(10)

        image_label = QtWidgets.QLabel()
        image_width = 250 if not is_single_item_row else 300
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
        
        # Store button reference for tooltips/disabling later
        button_name = f"btn_{title.lower().replace(' ', '_')}"
        setattr(self, button_name, button)
        option_layout.addWidget(button)

        return option_layout

    def update_button_states(self, is_lsl_connected):
        """Update button states based on connection and session status"""
        is_session_active = bool(self.session_goal)

        if self.page_type == "meditation":
            # Meditation buttons
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
                    self.btn_start_unity_game.setToolTip("A session is already active.")
                else:
                    self.btn_start_unity_game.setToolTip("")
        else:
            # Focus buttons
            tooltip_text = "Requires Muse connection." if not is_lsl_connected else ""
            user_tooltip = "You must be logged in to start a session." if not self.user_id else ""
            
            if tooltip_text and user_tooltip:
                tooltip_text = f"{tooltip_text} {user_tooltip}"
            elif user_tooltip and not tooltip_text:
                tooltip_text = user_tooltip
            
            enabled = is_lsl_connected and not is_session_active
            
            for button_name in ['btn_work_session', 'btn_video_session', 'btn_game_session']:
                if hasattr(self, button_name):
                    button = getattr(self, button_name)
                    button.setToolTip(tooltip_text)
                    button.setEnabled(enabled)

    # EEG Worker Management - SIMPLIFIED
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
            
            # Connect worker signals - SIMPLIFIED
            self.eeg_worker.connection_status_changed.connect(self.on_connection_status_changed)
            self.eeg_worker.calibration_progress.connect(self.on_calibration_progress)
            self.eeg_worker.calibration_status_changed.connect(self.on_calibration_status_changed)
            self.eeg_worker.new_prediction.connect(self.on_new_eeg_prediction)
            self.eeg_worker.signal_quality_update.connect(self.on_signal_quality_update)
            self.eeg_worker.error_occurred.connect(self.on_eeg_error)
            self.eeg_worker.session_saved.connect(self.on_session_saved)  # NEW SIGNAL
            
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
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)
            
            try:
                self.eeg_worker.connection_status_changed.disconnect()
                self.eeg_worker.calibration_progress.disconnect()
                self.eeg_worker.calibration_status_changed.disconnect()
                self.eeg_worker.new_prediction.disconnect()
                self.eeg_worker.signal_quality_update.disconnect()
                self.eeg_worker.error_occurred.disconnect()
                self.eeg_worker.session_saved.disconnect()
            except TypeError:
                pass
            
            self.eeg_worker = None
        
        if self.eeg_thread:
            if self.eeg_thread.isRunning():
                self.eeg_thread.quit()
                if not self.eeg_thread.wait(3000):
                    print("Warning: EEG thread did not quit gracefully")
                    self.eeg_thread.terminate()
                    self.eeg_thread.wait()
            self.eeg_thread = None
        
        print("EEG worker cleanup complete")

    # Session Management - SIMPLIFIED
    def start_session(self, session_subtype):
        """Unified session start method - SIMPLIFIED"""
        print(f"{self.page_type.title()} Page: Starting {session_subtype} session")
        
        # Common validation
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected", "Cannot start session.")
            return
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In", "You must be logged in to start a session.")
            return
        if self.session_goal:
            QtWidgets.QMessageBox.warning(self, "Session Active", "A session is already running.")
            return

        # Setup EEG worker
        if not self._setup_eeg_worker():
            QtWidgets.QMessageBox.critical(self, "EEG Setup Error", "Failed to initialize EEG processing system.")
            return

        # Route to appropriate session type
        if session_subtype == "video":
            self._start_video_session()
        elif session_subtype == "unity":
            self._start_unity_session()
        elif session_subtype == "work" and self.page_type == "focus":
            self._start_work_session()

    def _start_video_session(self):
        """Start video feedback session - SIMPLIFIED"""
        self.session_goal = self.eeg_session_type
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        self.signal_quality_validator.reset()
        
        # Create database session FIRST
        if self.page_type == "meditation":
            self.session_target_label = "Relaxed"
            session_type_for_db = "Meditation-Video"
            target_metric_for_db = "Relaxation"
        else:
            self.session_target_label = "Focused"
            session_type_for_db = "Focus-Video"
            target_metric_for_db = "Concentration"

        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            self.user_id, session_type_for_db, target_metric_for_db
        )
        print(f"Page Widget: Created database session {self.current_session_id}")
        
        # Create video player window with signal quality widget
        if not self.video_player_window:
            self.video_player_window = VideoPlayerWindow(parent=self)
            self.video_player_window.session_stopped.connect(self.handle_video_session_stopped_signal)
            self.video_player_window.recalibration_requested.connect(self.handle_recalibration_request)

        if hasattr(self.video_player_window, 'relaxation_circle'):
            self.video_player_window.relaxation_circle.session_type = self.eeg_session_type

        # Add signal quality widget to video player
        if not hasattr(self.video_player_window, 'signal_quality_widget'):
            self.video_player_window.signal_quality_widget = SignalQualityWidget()
            self.video_player_window.signal_quality_widget.recalibrate_requested.connect(self.handle_recalibration_request)
            self.video_player_window.add_signal_quality_widget(self.video_player_window.signal_quality_widget)

        self.video_player_window.set_status("Connecting to EEG...")
        self.video_player_window.show_calibration_progress(0)
        self.video_player_window.show()
        self.video_player_window.activateWindow()

        self.update_button_states(self.main_app_window.is_lsl_connected)
        
        # Start the EEG session - PASS SESSION ID
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "start_session", 
                                      QtCore.Qt.QueuedConnection, 
                                      QtCore.Q_ARG(str, self.eeg_session_type),
                                      QtCore.Q_ARG(int, self.current_session_id))

    def _start_unity_session(self):
        """Start Unity game session - SIMPLIFIED"""
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In", "You must be logged in to start a session.")
            return
        if self.session_goal:
            QtWidgets.QMessageBox.warning(self, "Session Active", "A session is already running.")
            return

        print(f"{self.page_type.title()} Page: Launch Unity Game clicked.")
        
        if not self._setup_eeg_worker():
            QtWidgets.QMessageBox.critical(self, "EEG Setup Error", "Failed to initialize EEG processing system.")
            return

        QtWidgets.QMessageBox.information(self, "Calibration Required", 
            "We'll first calibrate your EEG data before launching the game. Please stay relaxed for a moment.")
        
        self.session_goal = f"UNITY_{self.eeg_session_type}"
        self.is_calibrating = True
        self.is_calibrated = False
        
        # Start database session FIRST
        session_type = f"{self.page_type.title()}-Unity"
        target_metric = "Relaxation" if self.page_type == "meditation" else "Concentration"
        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            self.user_id, session_type, target_metric
        )
        print(f"Page Widget: Created database session {self.current_session_id}")
        
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
        
        # Start calibration - PASS SESSION ID
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "start_session", 
                                      QtCore.Qt.QueuedConnection, 
                                      QtCore.Q_ARG(str, self.eeg_session_type),
                                      QtCore.Q_ARG(int, self.current_session_id))

    def _start_work_session(self):
        """Start work session (focus page only) - SIMPLIFIED"""
        if self.page_type != "focus":
            return
            
        self.session_goal = "FOCUS"
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        self.focus_history = []
        self.focus_drop_counter = 0
        self.focus_alert_shown = False
        self.focus_monitoring_active = False
        
        # Create database session FIRST
        self.current_session_id, self.current_session_start_time = db_manager.start_new_session(
            self.user_id, "Focus-Work", "Concentration"
        )
        print(f"Page Widget: Created database session {self.current_session_id}")
        
        # Launch the work focus monitor window
        self.work_monitor_window = QtWidgets.QDialog(self)
        self.work_monitor_window.setWindowTitle("Work Focus Monitor")
        self.work_monitor_window.setFixedSize(400, 300)
        self.work_monitor_window.closeEvent = self.handle_work_window_closed
        
        # Set up the UI for the work monitor window
        self._setup_work_monitor_ui()
        
        # Start session timer
        self.session_timer = QtCore.QTimer(self)
        self.session_timer.timeout.connect(self.update_session_timer)
        self.session_timer.start(1000)
        self.session_start_time = QtCore.QDateTime.currentDateTime()
        
        # Show the monitor window
        self.work_monitor_window.show()
        
        # Update UI buttons
        self.update_button_states(self.main_app_window.is_lsl_connected)
        
        # Start EEG processing - PASS SESSION ID
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "start_session", 
                                      QtCore.Qt.QueuedConnection, 
                                      QtCore.Q_ARG(str, "FOCUS"),
                                      QtCore.Q_ARG(int, self.current_session_id))

    def _setup_work_monitor_ui(self):
        """Setup UI for work monitor window"""
        monitor_layout = QtWidgets.QVBoxLayout(self.work_monitor_window)
        
        # Timer label
        self.timer_label = QtWidgets.QLabel("00:00")
        self.timer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        monitor_layout.addWidget(self.timer_label)
        
        # Calibration progress bar
        self.calibration_progress_bar = QtWidgets.QProgressBar()
        self.calibration_progress_bar.setRange(0, 100)
        self.calibration_progress_bar.setValue(0)
        self.calibration_progress_bar.setFormat("Calibrating EEG: %p%")
        self.calibration_progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        monitor_layout.addWidget(self.calibration_progress_bar)
        
        # Focus indicator
        self.focus_indicator = QtWidgets.QProgressBar()
        self.focus_indicator.setRange(0, 100)
        self.focus_indicator.setValue(50)
        self.focus_indicator.setFormat("Focus Level: %p%")
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
        self.focus_indicator.hide()
        monitor_layout.addWidget(self.focus_indicator)
        
        # Status label
        self.focus_status_label = QtWidgets.QLabel("Connecting to EEG...")
        self.focus_status_label.setAlignment(QtCore.Qt.AlignCenter)
        monitor_layout.addWidget(self.focus_status_label)
        
        monitor_layout.addSpacing(20)
        
        # Stop button
        stop_button = QtWidgets.QPushButton("End Session")
        stop_button.setStyleSheet("background-color: #c0392b; color: white; padding: 8px;")
        stop_button.clicked.connect(self.stop_active_session)
        monitor_layout.addWidget(stop_button)

    # Signal Handlers - SIMPLIFIED (no more data handling)
    @QtCore.pyqtSlot(str, str)
    def on_connection_status_changed(self, status, message):
        """Handle connection status updates from EEG worker"""
        print(f"EEG Connection Status: {status} - {message}")
        
        if self.video_player_window:
            if status == "CONNECTED":
                self.video_player_window.set_status("EEG Connected. Starting calibration...")
            elif status == "ERROR":
                self.video_player_window.set_status(f"Connection Error: {message}")
                QtWidgets.QMessageBox.critical(self, "EEG Connection Error", message)
                self.stop_video_session_logic(triggered_by_error=True)
        
        if hasattr(self, 'focus_status_label'):
            if status == "CONNECTED":
                self.focus_status_label.setText("EEG Connected. Initializing calibration...")
            elif status == "ERROR":
                QtWidgets.QMessageBox.warning(self, "Connection Lost", f"EEG Connection Issue: {status}")
                self.stop_active_session()

    @QtCore.pyqtSlot(float)
    def on_calibration_progress(self, progress):
        """Handle calibration progress updates from EEG worker"""
        if not self.is_calibrating:
            return

        self.calibration_progress_value = int(progress * 100)
        
        # Update video player UI
        if self.video_player_window:
            self.video_player_window.show_calibration_progress(self.calibration_progress_value)
            if self.calibration_progress_value % 10 == 0 or self.calibration_progress_value >= 100:
                self.video_player_window.set_status(f"Calibrating EEG: {self.calibration_progress_value}% complete")
        
        # Update work monitor UI
        if hasattr(self, 'calibration_progress_bar'):
            self.calibration_progress_bar.setValue(self.calibration_progress_value)
            if self.calibration_progress_value % 10 == 0 or self.calibration_progress_value >= 100:
                self.focus_status_label.setText(f"Calibrating EEG: {self.calibration_progress_value}% complete")
        
        # Update Unity calibration dialog
        if hasattr(self, 'calibration_dialog'):
            self.calibration_dialog.setValue(self.calibration_progress_value)

    @QtCore.pyqtSlot(str, dict)
    def on_calibration_status_changed(self, status, data):
        """Handle calibration status changes from EEG worker"""
        print(f"Calibration Status: {status}, Data: {data}")
        
        if status == "COMPLETED":
            self.is_calibrating = False
            self.is_calibrated = True
            
            # Handle video session completion
            if self.video_player_window:
                self.video_player_window.set_status("Calibration complete. Starting session...")
                self.video_player_window.hide_calibration_progress_bar()
                
                if self.eeg_session_type == "RELAXATION":
                    self.video_player_window.start_relaxation_video()
                else:
                    self.video_player_window.start_focus_video()
            
            # Handle work session completion
            if hasattr(self, 'calibration_progress_bar'):
                self.calibration_progress_bar.hide()
                self.focus_indicator.show()
                self.focus_status_label.setText("Calibration complete. Monitoring focus...")
                
                # Start focus monitoring
                self.focus_monitoring_active = True
                if not self.focus_monitor_timer:
                    self.focus_monitor_timer = QtCore.QTimer(self)
                    self.focus_monitor_timer.timeout.connect(self._check_focus_levels)
                self.focus_monitor_timer.start(2000)
            
            # Handle Unity session completion
            if hasattr(self, 'calibration_dialog'):
                self._handle_unity_calibration_complete(data)
                
        elif status == "FAILED":
            self.is_calibrating = False
            self.is_calibrated = False
            error_msg = data.get('error_message', 'Calibration failed')
            QtWidgets.QMessageBox.warning(self, "Calibration Failed", error_msg)
            
            if self.video_player_window:
                self.stop_video_session_logic(triggered_by_error=True)
            else:
                self.stop_active_session()

    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction(self, prediction_data):
        """Handle new EEG predictions from worker - UI FEEDBACK ONLY"""
        if self.is_calibrating or not self.is_calibrated:
            return
            
        if prediction_data.get("message_type") != "PREDICTION":
            return

        classification = prediction_data.get("classification", {})
        state = classification.get("state", "Unknown")
        level = classification.get("level", 0)
        smooth_value = classification.get("smooth_value", 0.5)
        state_key = classification.get("state_key", "neutral")
        confidence = classification.get("confidence", 0.0)

        self.last_prediction = classification

        # Send to Unity if connected
        if self.client and self.session_goal and "UNITY" in self.session_goal:
            scaled_level = smooth_value * 100.0
            try:
                self.client.send_message(self.unity_osc_address, scaled_level)
            except Exception as e:
                print(f"Error sending OSC message to Unity: {e}")

        # Update video feedback
        if self.video_player_window and self.video_player_window.isVisible():
            self.update_video_feedback(state, level, smooth_value, state_key)
        
        # Update work session feedback
        if hasattr(self, 'focus_indicator') and self.session_goal == "FOCUS":
            self.update_work_feedback(state, level, smooth_value)

    @QtCore.pyqtSlot(dict)
    def on_signal_quality_update(self, quality_data):
        """Handle signal quality updates from EEG worker"""
        if (self.video_player_window and 
            hasattr(self.video_player_window, 'signal_quality_widget') and
            self.video_player_window.signal_quality_widget):
            
            self.video_player_window.signal_quality_widget.update_metrics(quality_data)

    @QtCore.pyqtSlot(str)
    def on_eeg_error(self, error_message):
        """Handle errors from EEG worker"""
        print(f"EEG Worker Error: {error_message}")
        QtWidgets.QMessageBox.warning(self, "EEG Processing Error", error_message)
        
        if "fatal" in error_message.lower():
            if self.video_player_window:
                self.stop_video_session_logic(triggered_by_error=True)
            else:
                self.stop_active_session()

    @QtCore.pyqtSlot(int, dict)
    def on_session_saved(self, session_id, summary_stats):
        """Handle session saved notification from EEG worker - NEW"""
        print(f"Page Widget: Session {session_id} saved successfully!")
        print(f"  - Total predictions: {summary_stats.get('total_predictions', 0)}")
        print(f"  - Percent on target: {summary_stats.get('percent_on_target', 0.0):.1f}%")
        print(f"  - Band data points: {summary_stats.get('band_data_points', 0)}")
        print(f"  - EEG data points: {summary_stats.get('eeg_data_points', 0)}")
        
        # Show success message to user (optional)
        if summary_stats.get('total_predictions', 0) > 0:
            QtWidgets.QMessageBox.information(
                self, 
                "Session Saved", 
                f"Session completed successfully!\n\n"
                f"Duration: {summary_stats.get('total_predictions', 0)} predictions\n"
                f"On target: {summary_stats.get('percent_on_target', 0.0):.1f}%\n"
                f"Data points saved: {summary_stats.get('band_data_points', 0)} band + {summary_stats.get('eeg_data_points', 0)} EEG"
            )

    # Feedback Methods (unchanged)
    def update_video_feedback(self, state, level, smooth_value, state_key):
        """Update video feedback based on EEG state and page type"""
        if not self.session_goal or not self.video_player_window:
            return

        if self.page_type == "meditation":
            # Relaxation-focused feedback
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
            
        else:
            # Focus-focused feedback
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

        if (level != getattr(self, 'last_scene_level', None) and time.time() - getattr(self, 'last_scene_change', 0) > 2.0):  # 2 sec minimum
            self.video_player_window.set_scene(scene)
            self.video_player_window.set_status(f"Status: {status_msg}")
            
            self.last_scene_level = level
            self.last_scene_change = time.time()
    
        # Circle can update more frequently but with level-aware smoothing
        self._update_circle_with_level_awareness(level, smooth_value)

            
    def _update_circle_with_level_awareness(self, level, smooth_value):
        """Blend level-based targets with smooth values"""
        
        # Define target circle values for each level
        level_targets = {
            -3: 0.05, -2: 0.15, -1: 0.35, 0: 0.50,
            1: 0.65, 2: 0.80, 3: 0.92, 4: 0.98
        }
        
        # Get ideal target for this level
        level_target = level_targets.get(level, 0.5)
        
        # Blend with smooth_value (you can adjust the ratio)
        final_value = 0.7 * smooth_value + 0.3 * level_target
        
        if self.page_type == "meditation":
            self.video_player_window.set_relaxation_level(final_value)
        else:  # focus
            self.video_player_window.set_focus_level(final_value)

    def update_work_feedback(self, state, level, smooth_value):
        """Update work session feedback"""
        if not hasattr(self, 'focus_indicator'):
            return
            
        focus_percent = int(smooth_value * 100)
        self.focus_indicator.setValue(focus_percent)
        
        # Set status text and color based on focus level
        if level <= -3:
            status = "Very distracted - try to refocus"
            color = "#e74c3c"
        elif level == -2:
            status = "Distracted - bring attention back"
            color = "#e67e22"
        elif level == -1:
            status = "Slightly distracted - stay with it"
            color = "#f1c40f"
        elif level == 0:
            status = "Neutral - continue focusing"
            color = "#3498db"
        elif level == 1:
            status = "Slightly focused - good start"
            color = "#2ecc71"
        elif level == 2:
            status = "Moderately focused - well done"
            color = "#27ae60"
        else:
            status = "Strongly focused - excellent"
            color = "#8A2BE2"
            
        self.focus_indicator.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
        if hasattr(self, 'focus_status_label'):
            self.focus_status_label.setText(status)
        
        # Add to focus history for monitoring
        if self.focus_monitoring_active:
            self.focus_history.append(smooth_value)
            if len(self.focus_history) > 60:
                self.focus_history.pop(0)

    # Session Management - SIMPLIFIED
    def handle_video_session_stopped_signal(self):
        """Called when video player emits session_stopped"""
        print(f"{self.page_type.title()}Page: Received session_stopped signal from VideoPlayerWindow.")
        self.stop_video_session_logic()

    def stop_video_session_logic(self, triggered_by_error=False):
        """Core logic to stop the video session - SIMPLIFIED"""
        print(f"{self.page_type.title()} Page: Stopping video session logic...")

        # Tell EEG worker to stop (it will handle database saving)
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)

        if self.video_player_window:
            try:
                self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped_signal)
                self.video_player_window.recalibration_requested.disconnect(self.handle_recalibration_request)
            except TypeError:
                pass
            if not triggered_by_error and self.video_player_window.isVisible():
                self.video_player_window.close()
            self.video_player_window = None

        # Reset session state (don't need to handle DB - worker does it)
        self.current_session_id = None
        self._reset_ui_and_state()

    def stop_active_session(self):
        """Stop any active session - SIMPLIFIED"""
        print(f"{self.page_type.title()} Page: Stopping active session...")
        
        # Stop timers
        if self.session_timer and self.session_timer.isActive():
            self.session_timer.stop()
        if self.focus_monitor_timer and self.focus_monitor_timer.isActive():
            self.focus_monitor_timer.stop()
        if hasattr(self, 'unity_data_timer') and self.unity_data_timer and self.unity_data_timer.isActive():
            self.unity_data_timer.stop()
        
        # Tell EEG worker to stop (it will handle database saving)
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)
        
        # Close windows
        if self.work_monitor_window:
            self.work_monitor_window.close()
            self.work_monitor_window = None
        
        if self.video_player_window:
            try:
                self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped_signal)
                self.video_player_window.recalibration_requested.disconnect(self.handle_recalibration_request)
            except TypeError:
                pass
            self.video_player_window.close()
            self.video_player_window = None
        
        # Reset session state (don't need to handle DB - worker does it)
        self.current_session_id = None
        self._reset_ui_and_state()

    def _reset_ui_and_state(self):
        """Reset UI elements and internal state - SIMPLIFIED"""
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.calibration_progress_value = 0
        self.focus_monitoring_active = False
        self.focus_history = []
        self.focus_drop_counter = 0
        self.focus_alert_shown = False
        self.last_sent_scene_index = -1
        
        if self.main_app_window and hasattr(self.main_app_window, 'is_lsl_connected'):
            self.update_button_states(self.main_app_window.is_lsl_connected)
        else:
            self.update_button_states(False)

    # Focus-specific methods (unchanged)
    def _check_focus_levels(self):
        """Analyze focus history to detect significant drops (focus page only)"""
        if not self.focus_monitoring_active or len(self.focus_history) < 10:
            return
        
        recent_focus = self.focus_history[-10:]
        first_half_avg = sum(recent_focus[:5]) / 5
        second_half_avg = sum(recent_focus[-5:]) / 5
        
        if second_half_avg < first_half_avg * 0.7:
            self.focus_drop_counter += 1
            if self.focus_drop_counter >= 3 and not self.focus_alert_shown:
                self.show_focus_alert()
        else:
            self.focus_drop_counter = max(0, self.focus_drop_counter - 1)

    def show_focus_alert(self):
        """Show focus drop alert (focus page only)"""
        self.focus_alert_shown = True
        
        reply = QtWidgets.QMessageBox.question(
            self, "Focus Alert", 
            "Your focus level is dropping.\nWould you like to take a break or start a focus exercise?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            if self.current_session_id:
                db_manager.add_session_note(self.current_session_id, "User took a break due to focus drop")
        
        # Reset alert flag after delay
        QtCore.QTimer.singleShot(5 * 60 * 1000, self.reset_focus_alert)

    def reset_focus_alert(self):
        """Reset focus alert flag"""
        self.focus_alert_shown = False

    def update_session_timer(self):
        """Update the timer display for the active session"""
        if hasattr(self, 'session_start_time') and hasattr(self, 'timer_label'):
            elapsed = self.session_start_time.secsTo(QtCore.QDateTime.currentDateTime())
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.timer_label.setText(f"{minutes:02}:{seconds:02}")

    def handle_work_window_closed(self, event):
        """Handle when work monitor window is closed by user"""
        print("Focus Page: Work monitor window closed.")
        self.stop_active_session()
        event.accept()

    # Unity-specific methods (unchanged)
    def on_unity_calibration_progress(self, progress):
        """Update Unity calibration dialog with progress"""
        if hasattr(self, 'calibration_dialog'):
            self.calibration_dialog.setValue(int(progress * 100))

    def on_unity_calibration_status(self, status, data):
        """Handle calibration completion for Unity game"""
        if status == "COMPLETED":
            self._handle_unity_calibration_complete(data)
        elif status == "FAILED":
            QtWidgets.QMessageBox.critical(self, "Calibration Failed", 
                "Failed to calibrate EEG. Please check your Muse connection and try again.")
            self.stop_unity_session()

    def _handle_unity_calibration_complete(self, data):
        """Handle successful Unity calibration completion"""
        self.is_calibrating = False
        self.is_calibrated = True
        
        try:
            if hasattr(self, 'calibration_dialog'):
                self.calibration_dialog.close()
            
            # Setup OSC client
            self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)
            
            # Send initial data
            self.client.send_message(self.unity_osc_address, 50.0)
            self.client.send_message(UNITY_OSC_SCENE_ADDRESS, 0)
            
            # Launch Unity game
            if self.page_type == "meditation":
                unity_game_path = r"C:\Users\berna\OneDrive\Documentos\GitHub\NeuroFlow\game\NeuroFlow.exe"
            else:
                unity_game_path = r"C:/NeuroFlow/Neuro/NeuroFlowFocus.exe"
                
            if not os.path.exists(unity_game_path):
                QtWidgets.QMessageBox.warning(self, "Error", f"Game not found at:\n{unity_game_path}")
                self.stop_unity_session()
                return
                
            subprocess.Popen([unity_game_path])
            
            # Start heartbeat timer
            self.unity_data_timer = QtCore.QTimer(self)
            self.unity_data_timer.timeout.connect(self.send_unity_heartbeat)
            self.unity_data_timer.start(2000)
            
            self.update_button_states(self.main_app_window.is_lsl_connected)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to setup Unity game:\n{e}")
            self.stop_unity_session()

    def send_unity_heartbeat(self):
        """Send regular data to Unity to prevent timeout"""
        if self.client and self.session_goal and "UNITY" in self.session_goal:
            try:
                if hasattr(self, 'last_prediction') and self.last_prediction:
                    smooth_value = self.last_prediction.get("smooth_value", 0.5)
                else:
                    smooth_value = 0.5
                    
                scaled_value = smooth_value * 100.0
                self.client.send_message(self.unity_osc_address, scaled_value)
                
            except Exception as e:
                print(f"Error sending Unity data: {e}")

    def stop_unity_session(self):
        """Clean up Unity game session resources"""
        print("Stopping Unity session and cleaning up resources...")
        
        if hasattr(self, 'unity_data_timer') and self.unity_data_timer and self.unity_data_timer.isActive():
            self.unity_data_timer.stop()
        
        # Tell EEG worker to stop (it will handle database saving)
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "stop_session", QtCore.Qt.QueuedConnection)
        
        # Reset session state (don't need to handle DB - worker does it)
        self.current_session_id = None
        self.session_goal = None
        self.update_button_states(self.main_app_window.is_lsl_connected)

    # Utility methods (unchanged)
    def handle_recalibration_request(self):
        """Handle user request to recalibrate due to poor signal quality"""
        reply = QtWidgets.QMessageBox.question(
            self, "Recalibrate EEG?", 
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
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        
        if self.video_player_window:
            self.video_player_window.set_status("Restarting calibration...")
            self.video_player_window.show_calibration_progress(0)
            self.video_player_window.show_signal_quality_panel()
            
            if self.video_player_window.signal_quality_widget:
                self.video_player_window.signal_quality_widget.reset()
        
        if self.eeg_worker:
            QtCore.QMetaObject.invokeMethod(self.eeg_worker, "recalibrate", QtCore.Qt.QueuedConnection)

    def test_eeg_processing(self):
        """Test EEG processing system"""
        if not self._setup_eeg_worker():
            QtWidgets.QMessageBox.critical(self, "EEG Setup Error", "Failed to initialize EEG processing system.")
            return
        
        QtCore.QMetaObject.invokeMethod(self.eeg_worker, "connect_to_lsl", QtCore.Qt.QueuedConnection)
        QtWidgets.QMessageBox.information(self, "EEG Test", "Testing EEG connection. Check console for status updates.")

    def clean_up_session(self):
        """Clean up any active sessions when widget is closed or app exits"""
        print(f"{self.page_type.title()} Page: Cleaning up active session if any.")
        if self.session_goal:
            if "UNITY" in self.session_goal:
                self.stop_unity_session()
            else:
                self.stop_active_session()
        
        self._cleanup_eeg_worker()