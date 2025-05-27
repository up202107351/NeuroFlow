import os
import subprocess
import signal
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import matplotlib
import random
from datetime import datetime
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from ui.video_player_window import VideoPlayerWindow
from backend.eeg_prediction_subscriber import EEGPredictionSubscriber
from backend import database_manager as db
from pythonosc.udp_client import SimpleUDPClient
from backend.zmq_port_cleanup import cleanup_all_zmq_ports

# OSC configuration for Unity
UNITY_IP = "127.0.0.1"
UNITY_OSC_PORT = 9000
UNITY_OSC_ADDRESS = "/neuroflow/focus"  # Changed from muse/relaxation to neuroflow/focus
UNITY_OSC_SCENE_ADDRESS = "/neuroflow/scene"  # Changed address for better identification

class FocusPageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, main_app_window_ref=None):
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
        self.calibration_progress_value = 0
        self.last_sent_scene_index = -1
        self.focus_monitoring_active = False
        self.focus_history = []
        self.focus_alert_shown = False
        self.focus_drop_counter = 0
        
        # Initialize OSC client for Unity
        self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)
        
        # Initialize UI
        self.initUI()
        
        # Set up timers
        self.connection_timeout_timer = QtCore.QTimer(self)
        self.connection_timeout_timer.setSingleShot(True)
        self.connection_timeout_timer.timeout.connect(self._handle_connection_timeout)
        
        self.calibration_update_timer = QtCore.QTimer(self)
        self.calibration_update_timer.timeout.connect(self._update_fake_calibration_progress)
        
        self.focus_monitor_timer = QtCore.QTimer(self)
        self.focus_monitor_timer.timeout.connect(self._check_focus_levels)
        
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
            image_path="./assets/work.jpg",
            button_text="Start",
            action_slot=self.start_work_focus_session
        )
        row1_layout.addLayout(work_focus_layout)

        # Option 2: Video Focus Session
        video_focus_layout = self.create_focus_option_layout(
            title="Video Session",
            image_path="./assets/focus.jpg",
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
            image_path="./assets/focus_game.jpg",
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
        # Store button reference for tooltips/disabling later
        button_name = f"btn_{title.lower().replace(' ', '_')}"
        setattr(self, button_name, button)
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
        
        # Clean up ZMQ ports to prevent conflicts
        try:
            print("Cleaning up ZMQ ports before starting backend...")
            cleanup_all_zmq_ports()
        except Exception as e:
            print(f"Warning: Port cleanup failed: {e}")
            
        # Create a new database session
        session_type = "Focus-Work"
        target_metric = "Concentration"
        self.current_session_id, self.current_session_start_time = db.start_new_session(
            self.user_id, session_type, target_metric
        )
        
        # Set session active
        self.session_goal = "FOCUS"
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        self.focus_history = []
        self.focus_drop_counter = 0
        self.focus_alert_shown = False
        self.focus_monitoring_active = False
        
        # Launch the work focus monitor window
        self.work_monitor_window = QtWidgets.QDialog(self)
        self.work_monitor_window.setWindowTitle("Work Focus Monitor")
        self.work_monitor_window.setFixedSize(400, 300)
        self.work_monitor_window.closeEvent = self.handle_work_window_closed
        
        # Set up the UI for the work monitor window
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
        self.focus_indicator.hide()  # Hide until calibrated
        monitor_layout.addWidget(self.focus_indicator)
        
        # Status label
        self.focus_status_label = QtWidgets.QLabel("Connecting to EEG...")
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
            self.backend_process = subprocess.Popen(
                [sys.executable, "-u", backend_script_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            print(f"Focus Page: Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Focus Page: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            self._reset_ui_and_state()
            return
            
        # Start timer for session
        self.session_timer = QtCore.QTimer(self)
        self.session_timer.timeout.connect(self.update_session_timer)
        self.session_timer.start(1000)  # Update every second
        self.session_start_time = QtCore.QDateTime.currentDateTime()
        
        # Show the monitor window
        self.work_monitor_window.show()
        
        # Update UI buttons
        self.update_button_states()
        
        # Start connection timeout timer
        self.connection_timeout_timer.start(10000)  # 10 seconds
        
        # Start prediction subscriber after a delay
        QtCore.QTimer.singleShot(1500, self._start_prediction_subscriber_for_focus)

    def update_button_states(self):
        """Update button states based on session activity"""
        active_session = self.session_goal is not None
        
        if hasattr(self, 'btn_work_session'):
            self.btn_work_session.setEnabled(not active_session)
            
        if hasattr(self, 'btn_video_session'):
            self.btn_video_session.setEnabled(not active_session)
            
        if hasattr(self, 'btn_game_session'):
            self.btn_game_session.setEnabled(not active_session)
    
    def update_session_timer(self):
        """Update the timer display for the active session"""
        if hasattr(self, 'session_start_time') and hasattr(self, 'timer_label'):
            elapsed = self.session_start_time.secsTo(QtCore.QDateTime.currentDateTime())
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.timer_label.setText(f"{minutes:02}:{seconds:02}")

    def _handle_connection_timeout(self):
        """Handle timeout when connecting to EEG backend"""
        print("Focus Page: ZMQ connection timeout.")
        
        if hasattr(self, 'focus_status_label'):
            self.focus_status_label.setText("Connection timeout. Please try again.")
            
        QtWidgets.QMessageBox.warning(self, "Connection Timeout",
                                "Failed to connect to EEG backend. Please try again.")
        self.stop_active_session()

    def _start_prediction_subscriber_for_focus(self):
        """Starts the ZMQ subscriber thread for focus sessions"""
        if self.backend_process is None or self.backend_process.poll() is not None:
            print("Focus Page: Backend process not running. Cannot start subscriber.")
            self.focus_status_label.setText("Backend failed to start.")
            self.stop_active_session()
            return
            
        print("Focus Page: Starting ZMQ prediction subscriber thread.")
        
        try:
            # Set up the prediction subscriber in a non-blocking way
            self.prediction_subscriber = EEGPredictionSubscriber()
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
            
            # Start the thread
            self.prediction_thread.start()
            
            # Start focus session after a moment
            QtCore.QTimer.singleShot(1000, self._initiate_calibration_in_subscriber)
            
        except Exception as e:
            print(f"Error starting prediction subscriber: {e}")
            if self.connection_timeout_timer.isActive():
                self.connection_timeout_timer.stop()
            self.stop_active_session()

    def _initiate_calibration_in_subscriber(self):
        """Sends command to subscriber to start calibration after connection is likely established."""
        if not self.prediction_subscriber or not self.prediction_thread or not self.prediction_thread.isRunning():
            print("Focus Page: Subscriber not ready for calibration.")
            if hasattr(self, 'focus_status_label'):
                self.focus_status_label.setText("Error: EEG subscriber not ready.")
            self.stop_active_session()
            return

        try:
            # Update status
            if hasattr(self, 'focus_status_label'):
                self.focus_status_label.setText("Calibrating EEG... Please focus.")
                
            # Request focus calibration - non-blocking with command queue
            print("Focus Page: Requesting subscriber to start focus session (calibration)...")
            if not self.prediction_subscriber.start_focus_session():
                print("Focus Page: Failed to queue start focus session command.")
                if hasattr(self, 'focus_status_label'):
                    self.focus_status_label.setText("Failed to initiate calibration.")
                self.stop_active_session()
                return

            print(f"{QtCore.QTime.currentTime().toString('hh:mm:ss.zzz')} - Focus Page: Calibration requested")
            
        except Exception as e:
            print(f"{QtCore.QTime.currentTime().toString('hh:mm:ss.zzz')} - Error initiating calibration: {e}")
            if hasattr(self, 'focus_status_label'):
                self.focus_status_label.setText(f"Calibration Error: {str(e)}")
            self.stop_active_session()

    def _update_fake_calibration_progress(self):
        """Update calibration progress bar with fallback animation if backend is slow"""
        if not self.is_calibrating or self.calibration_progress_value >= 100:
            self.calibration_update_timer.stop()
            return

        # Increment progress with decreasing speed
        if self.calibration_progress_value < 30: increment = 2
        elif self.calibration_progress_value < 60: increment = 1
        else: increment = 1
        
        self.calibration_progress_value = min(100, self.calibration_progress_value + increment)

        # Update UI
        if hasattr(self, 'calibration_progress_bar'):
            self.calibration_progress_bar.setValue(self.calibration_progress_value)
            
        # If it hits 100%, stop timer but don't switch UI yet (wait for backend confirmation)
        if self.calibration_progress_value >= 100:
            self.calibration_update_timer.stop()
            print("Focus Page: Fake calibration reached 100%. Waiting for backend confirmation.")

    def start_video_focus_session(self):
        """Start a video focus session"""
        print("Focus Page: Clicked Start Video Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
            
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In",
                                        "You must be logged in to start a session.")
            return
        
        # Clean up ZMQ ports
        try:
            print("Cleaning up ZMQ ports before starting backend...")
            cleanup_all_zmq_ports()
        except Exception as e:
            print(f"Warning: Port cleanup failed: {e}")
            
        # Set session parameters
        self.session_goal = "FOCUS"
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_progress_value = 0
        
        # Create or update video player window
        if not self.video_player_window:
            self.video_player_window = VideoPlayerWindow(parent=self)
            self.video_player_window.session_stopped.connect(self.handle_video_session_stopped)
        
        self.video_player_window.set_status("Connecting to EEG...")
        self.video_player_window.show_calibration_progress(0)
        self.video_player_window.show()
        self.video_player_window.activateWindow()
        
        # Create session in database
        session_type = "Focus-Video"
        target_metric = "Concentration"
        self.current_session_id, self.current_session_start_time = db.start_new_session(
            self.user_id, session_type, target_metric
        )
        
        # Launch backend processor
        backend_script_path = "eeg_backend_processor.py"
        try:
            print(f"Focus Page: Launching backend script: {backend_script_path}")
            self.backend_process = subprocess.Popen(
                [sys.executable, "-u", backend_script_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            print(f"Focus Page: Backend process started with PID: {self.backend_process.pid}")
        except Exception as e:
            error_msg = f"Failed to launch backend EEG script: {e}"
            print(f"Focus Page: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Backend Error", error_msg)
            self._reset_ui_and_state()
            return
        
        # Update UI buttons
        self.update_button_states()
        
        # Start connection timeout timer
        self.connection_timeout_timer.start(10000)  # 10 seconds
        
        # Start prediction subscriber after a delay
        QtCore.QTimer.singleShot(2000, self._start_prediction_subscriber_for_focus)

    def handle_video_session_stopped(self):
        """Called when video player emits session_stopped signal"""
        print("Focus Page: Received session_stopped signal from VideoPlayerWindow.")
        self.stop_active_session()

    def handle_work_window_closed(self, event):
        """Handle when work monitor window is closed by user"""
        print("Focus Page: Work monitor window closed.")
        self.stop_active_session()
        event.accept()

    def start_game_focus_session(self):
        """Start a focus game session with Unity integration"""
        print("Focus Page: Clicked Start Game Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
            
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "User Not Logged In",
                                        "You must be logged in to start a session.")
            return
        
        # VERY IMPORTANT: Make sure we have a valid OSC client for Unity
        try:
            # Recreate the OSC client to ensure it's fresh
            self.client = SimpleUDPClient(UNITY_IP, UNITY_OSC_PORT)
            print(f"Created OSC client to Unity at {UNITY_IP}:{UNITY_OSC_PORT}")
            
            # Send a test message to verify connection
            self.client.send_message(UNITY_OSC_ADDRESS, 50.0)  # Send 50% focus level to test
            self.client.send_message(UNITY_OSC_SCENE_ADDRESS, 1)  # Test scene change
            print("Sent test messages to Unity")
        except Exception as e:
            print(f"Error setting up OSC client: {e}")
            QtWidgets.QMessageBox.warning(self, "Connection Error", 
                f"Failed to initialize OSC connection to Unity: {e}\nGame may not receive EEG data.")
            return
            
        # Create a new session in the database
        session_type = "Focus-Game"
        target_metric = "Concentration"
        
        # Launch the Unity game
        unity_game_path = r"C:/NeuroFlow/Neuro/NeuroFlowFocus.exe" # Adjust path as needed
        if not os.path.exists(unity_game_path):
            QtWidgets.QMessageBox.warning(self, "Error", f"Focus game not found at:\n{unity_game_path}")
            return
            
        try:
            # Launch game
            subprocess.Popen([unity_game_path])
            
            # Create session in database only after successful launch
            self.current_session_id, self.current_session_start_time = db.start_new_session(
                self.user_id, session_type, target_metric
            )
            
            # Set session active
            self.session_goal = "FOCUS_GAME"
            self.update_button_states()

            # Inform user
            QtWidgets.QMessageBox.information(self, "Game Launched",
                "Focus Game is launching. Make sure your Muse is connected.\n\n" +
                "The focus data will be sent automatically to the game.")
                
            print(f"Focus game session {self.current_session_id} started for user {self.user_id}.")
            
        except Exception as e:
            print(f"Error launching game: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error launching focus game:\n{e}")
            self._reset_ui_and_state()

    def _check_focus_levels(self):
        """Analyze focus history to detect significant drops"""
        if not self.focus_monitoring_active or len(self.focus_history) < 10:
            return
        
        # Get the last 10 focus values
        recent_focus = self.focus_history[-10:]
        
        # Calculate average of first half and second half
        first_half_avg = sum(recent_focus[:5]) / 5
        second_half_avg = sum(recent_focus[-5:]) / 5
        
        # Check for significant drop in focus
        if second_half_avg < first_half_avg * 0.7:  # Focus dropped by 30% or more
            self.focus_drop_counter += 1
            
            # Only show alert if drop persists for multiple checks and alert not recently shown
            if self.focus_drop_counter >= 3 and not self.focus_alert_shown:
                self.show_focus_alert()
        else:
            self.focus_drop_counter = max(0, self.focus_drop_counter - 1)  # Gradually reduce counter

    def show_focus_alert(self):
        """Show an alert when focus has significantly dropped"""
        self.focus_alert_shown = True
        
        # Create alert dialog that stays on top of other windows
        alert_dialog = QtWidgets.QDialog(self)
        alert_dialog.setWindowTitle("Focus Alert")
        alert_dialog.setWindowFlags(alert_dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        alert_dialog.setFixedSize(400, 200)
        
        # Dialog layout
        layout = QtWidgets.QVBoxLayout(alert_dialog)
        
        # Warning icon
        icon_label = QtWidgets.QLabel()
        icon = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        icon_label.setPixmap(icon.pixmap(48, 48))
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Message
        message = QtWidgets.QLabel("Your focus level is dropping.\nWould you like to take a break or start a focus exercise?")
        message.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(message)
        
        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        
        # Take break button
        break_button = QtWidgets.QPushButton("Take a Break")
        break_button.clicked.connect(lambda: self.handle_alert_response("break", alert_dialog))
        button_layout.addWidget(break_button)
        
        # Start focus exercise button
        focus_button = QtWidgets.QPushButton("Start Focus Exercise")
        focus_button.clicked.connect(lambda: self.handle_alert_response("focus", alert_dialog))
        button_layout.addWidget(focus_button)
        
        # Ignore button
        ignore_button = QtWidgets.QPushButton("Ignore")
        ignore_button.clicked.connect(lambda: self.handle_alert_response("ignore", alert_dialog))
        button_layout.addWidget(ignore_button)
        
        layout.addLayout(button_layout)
        
        # Play alert sound if available
        try:
            QtWidgets.QApplication.beep()
        except:
            pass
            
        # Force window to be shown on top of all others
        alert_dialog.show()
        alert_dialog.raise_()
        alert_dialog.activateWindow()
        
        # Make it modal to ensure user sees it
        alert_dialog.exec_()

    # Update the handle_alert_response method to properly close the dialog
    def handle_alert_response(self, response, alert_dialog):
        """Handle user response to focus alert"""
        # Close the dialog first
        alert_dialog.accept()
        
        if response == "break":
            # Log the break in database
            if self.current_session_id:
                db.add_session_note(self.current_session_id, "User took a break due to focus drop")
            
            # Show break timer dialog (simple implementation)
            break_timer = QtWidgets.QDialog(self)
            break_timer.setWindowTitle("Break Timer")
            break_timer.setFixedSize(300, 150)
            break_timer.setWindowFlags(break_timer.windowFlags() | Qt.WindowStaysOnTopHint)
            
            layout = QtWidgets.QVBoxLayout(break_timer)
            
            time_label = QtWidgets.QLabel("05:00")
            time_label.setAlignment(QtCore.Qt.AlignCenter)
            time_label.setStyleSheet("font-size: 36px; font-weight: bold;")
            layout.addWidget(time_label)
            
            message = QtWidgets.QLabel("Take a short break. Timer will remind you to return.")
            message.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(message)
            
            end_break_button = QtWidgets.QPushButton("End Break")
            end_break_button.clicked.connect(break_timer.accept)
            layout.addWidget(end_break_button)
            
            # Break timer countdown
            break_seconds = 5 * 60
            
            def update_break_timer():
                nonlocal break_seconds
                break_seconds -= 1
                mins = break_seconds // 60
                secs = break_seconds % 60
                time_label.setText(f"{mins:02}:{secs:02}")
                
                if break_seconds <= 0:
                    timer.stop()
                    break_timer.accept()
            
            timer = QtCore.QTimer(break_timer)
            timer.timeout.connect(update_break_timer)
            timer.start(1000)
            
            break_timer.show()
            break_timer.raise_()
            break_timer.activateWindow()
            break_timer.exec_()
            
        elif response == "focus":
            # Transfer to video session without stopping backend
            self.transfer_to_video_session()
            
        else:  # ignore
            # Just log that user ignored the alert
            if self.current_session_id:
                db.add_session_note(self.current_session_id, "User ignored focus drop alert")
        
        # Reset the alert flag after a delay
        QtCore.QTimer.singleShot(5 * 60 * 1000, self.reset_focus_alert)  # 5 minutes

    # Add a new method to transfer from work monitor to video session
    def transfer_to_video_session(self):
        """Transfer from work session to video session without recalibrating"""
        # Log the action in database
        if self.current_session_id:
            db.add_session_note(self.current_session_id, "User switched to focus video exercise")
            db.end_session(self.current_session_id)
        
        # Create a new database session for the video
        session_type = "Focus-Video"
        target_metric = "Concentration"
        self.current_session_id, self.current_session_start_time = db.start_new_session(
            self.user_id, session_type, target_metric
        )
        
        # Close work monitor window if it exists
        if hasattr(self, 'work_monitor_window') and self.work_monitor_window:
            self.work_monitor_window.close()
            self.work_monitor_window = None
        
        # Stop focus monitoring timer
        if self.focus_monitor_timer.isActive():
            self.focus_monitor_timer.stop()
        
        # Create and set up video player window
        if not self.video_player_window:
            self.video_player_window = VideoPlayerWindow(parent=self)
            self.video_player_window.session_stopped.connect(self.handle_video_session_stopped)
        
        # Set status and show window
        self.video_player_window.set_status("Starting focus video session...")
        self.video_player_window.show()
        self.video_player_window.activateWindow()
        
        # Start the video immediately - no need to calibrate again
        # We're already calibrated from the work session
        QtCore.QTimer.singleShot(500, self.video_player_window.start_focus_video)
        
        print("Transferred from work session to video session")

    # Modify the stop_active_session method to check if we're transferring
    def stop_active_session(self, transferring_to_video=False):
        """Stop any active focus session"""
        print("Focus Page: Stopping active session...")
        
        # Stop all active timers
        if self.connection_timeout_timer.isActive():
            self.connection_timeout_timer.stop()
            
        if self.calibration_update_timer.isActive():
            self.calibration_update_timer.stop()
            
        if self.focus_monitor_timer.isActive():
            self.focus_monitor_timer.stop()
            
        if hasattr(self, 'session_timer') and self.session_timer.isActive():
            self.session_timer.stop()
        
        # Only stop ZMQ and backend if we're not transferring to video
        if not transferring_to_video:
            # Stop the subscriber thread
            if self.prediction_subscriber:
                try:
                    print("Focus Page: Stopping prediction subscriber...")
                    self.prediction_subscriber.stop()
                    if self.prediction_thread and self.prediction_thread.isRunning():
                        if not self.prediction_thread.wait(2000):  # Wait up to 2 seconds
                            print("Focus Page: Terminating prediction thread...")
                            self.prediction_thread.terminate()
                            self.prediction_thread.wait()
                except Exception as e:
                    print(f"Error stopping subscriber: {e}")
                
                self.prediction_subscriber = None
                self.prediction_thread = None
            
            # Terminate the backend process
            if self.backend_process and self.backend_process.poll() is None:
                try:
                    print(f"Focus Page: Terminating backend process PID: {self.backend_process.pid}")
                    if os.name == 'nt':
                        # Send CTRL_BREAK_EVENT to the process group on Windows
                        os.kill(self.backend_process.pid, signal.CTRL_BREAK_EVENT)
                    else:
                        # Send SIGINT for graceful shutdown
                        self.backend_process.send_signal(signal.SIGINT)
                    
                    self.backend_process.wait(timeout=3)  # Wait for graceful exit
                except subprocess.TimeoutExpired:
                    print("Focus Page: Backend process did not terminate gracefully, killing.")
                    self.backend_process.kill()
                    self.backend_process.wait()
                except Exception as e:
                    print(f"Error terminating backend process: {e}")
                    
                self.backend_process = None
        
        # Close the work monitor window if it exists
        if hasattr(self, 'work_monitor_window') and self.work_monitor_window:
            self.work_monitor_window.close()
            self.work_monitor_window = None
        
        # Close the video player window if not transferring to video
        if not transferring_to_video and self.video_player_window:
            try:
                if hasattr(self.video_player_window, 'session_stopped'):
                    self.video_player_window.session_stopped.disconnect(self.handle_video_session_stopped)
            except TypeError:
                pass  # Already disconnected
                
            self.video_player_window.close()
            self.video_player_window = None
        
        # End the session in the database if not transferring
        if not transferring_to_video and self.current_session_id:
            db.end_session(self.current_session_id)
            self.current_session_id = None
            self.current_session_start_time = None
        
        # Reset UI and state if not transferring
        if not transferring_to_video:
            self._reset_ui_and_state()

    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction(self, prediction_data):
        """Handle new EEG predictions from the subscriber"""
        if self.is_calibrating or not self.is_calibrated:
            return  # Ignore predictions during calibration
            
        if prediction_data.get("message_type") != "PREDICTION":
            return  # Only process prediction messages
        
        # Extract data from prediction
        classification = prediction_data.get("classification", {})
        state = classification.get("state", "Unknown")
        level = classification.get("level", 0) 
        smooth_value = classification.get("smooth_value", 0.5)
        state_key = classification.get("state_key", "neutral")
        
        # Send focus data to Unity game if active
        if self.session_goal == "FOCUS_GAME" and self.client:
            # Scale focus value for Unity (0-100)
            scaled_focus_level = smooth_value * 100.0
            
            try:
                # Send focus level
                self.client.send_message(UNITY_OSC_ADDRESS, scaled_focus_level)
                
                # Determine scene index based on focus level
                target_scene_index = -1
                if scaled_focus_level >= 80: target_scene_index = 2
                elif scaled_focus_level >= 50: target_scene_index = 1
                else: target_scene_index = 0
                
                # Only send scene changes when necessary
                if target_scene_index != self.last_sent_scene_index:
                    self.client.send_message(UNITY_OSC_SCENE_ADDRESS, target_scene_index)
                    self.last_sent_scene_index = target_scene_index
                    print(f"Sent OSC Scene Change: {UNITY_OSC_SCENE_ADDRESS}, Index: {target_scene_index}")
            except Exception as e:
                print(f"Error sending OSC message to Unity: {e}")
        
        # Update work session UI if active
        if self.session_goal == "FOCUS" and hasattr(self, 'focus_indicator'):
            # Scale to percent for progress bar
            focus_percent = int(smooth_value * 100)
            self.focus_indicator.setValue(focus_percent)
            
            # Set status text and color based on focus level
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
            else:  # level >= 3
                status = "Strongly focused - excellent"
                self.focus_indicator.setStyleSheet("QProgressBar::chunk { background-color: #8A2BE2; }")
                
            if hasattr(self, 'focus_status_label'):
                self.focus_status_label.setText(status)
            
            # Add to focus history for monitoring
            if self.focus_monitoring_active:
                self.focus_history.append(smooth_value)
                
                # Keep history from growing too large
                if len(self.focus_history) > 60:  # Store ~2 minutes at 2s intervals
                    self.focus_history.pop(0)
        
        # Update video player if active
        if self.session_goal == "FOCUS" and self.video_player_window and self.video_player_window.isVisible():
            # Set scene based on focus level
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
            else:  # level >= 4
                self.video_player_window.set_scene("deeply_focused")
                self.video_player_window.set_status(f"Status: {state} (Perfect focus!)")
            
            # Update focus level
            self.video_player_window.set_focus_level(smooth_value)
            
        # Record session data to database
        if self.current_session_id:
            is_on_target = level > 0  # Focus is on target when level is positive
            db.add_session_metric(self.current_session_id, state, is_on_target, smooth_value)

    @QtCore.pyqtSlot(float)
    def on_calibration_progress(self, progress):
        """Handle calibration progress updates"""
        if not self.is_calibrating:
            return  # Ignore if not in calibration phase
            
        self.calibration_progress_value = int(progress * 100)
        
        # Update work monitor UI
        if hasattr(self, 'calibration_progress_bar'):
            self.calibration_progress_bar.setValue(self.calibration_progress_value)
            
            if self.calibration_progress_value % 10 == 0 or self.calibration_progress_value >= 100:
                self.focus_status_label.setText(f"Calibrating EEG: {self.calibration_progress_value}% complete")
        
        # Update video player UI
        if self.video_player_window and self.video_player_window.isVisible():
            self.video_player_window.show_calibration_progress(self.calibration_progress_value)
            
            if self.calibration_progress_value % 10 == 0 or self.calibration_progress_value >= 100:
                self.video_player_window.set_status(f"Calibrating EEG: {self.calibration_progress_value}% complete")
        
        # Stop fake progress timer if we're getting real updates
        if self.calibration_update_timer.isActive():
            self.calibration_update_timer.stop()

    @QtCore.pyqtSlot(str, dict)
    def on_calibration_status(self, status, baseline_data):
        """Handle calibration status updates"""
        print(f"Focus Page: Calibration Status: {status}, Baseline: {baseline_data if baseline_data else 'N/A'}")
        
        # Stop fake progress timer
        if self.calibration_update_timer.isActive():
            self.calibration_update_timer.stop()
            
        if status == "COMPLETED":
            self.is_calibrating = False
            self.is_calibrated = True
            
            # Update work monitor UI
            if hasattr(self, 'calibration_progress_bar'):
                self.calibration_progress_bar.hide()
                self.focus_indicator.show()
                self.focus_status_label.setText("Calibration complete. Monitoring focus...")
                
                # Start focus monitoring
                self.focus_monitoring_active = True
                self.focus_monitor_timer.start(2000)  # Check focus every 2 seconds
            
            # Update video player UI
            if self.video_player_window and self.video_player_window.isVisible():
                self.video_player_window.set_status("Calibration complete. Starting focus session...")
                self.video_player_window.hide_calibration_progress_bar()
                
                # Start the appropriate video
                self.video_player_window.start_focus_video()
                
            # Stop the connection timeout timer if still active
            if self.connection_timeout_timer.isActive():
                self.connection_timeout_timer.stop()
                
        elif status == "FAILED":
            self.is_calibrating = False
            self.is_calibrated = False
            
            QtWidgets.QMessageBox.warning(self, "Calibration Failed",
                "Failed to calibrate EEG. Please check the Muse connection and try again.")
            self.stop_active_session()
            
        elif status == "STARTED":
            self.is_calibrating = True
            
            # Update work monitor UI
            if hasattr(self, 'calibration_progress_bar'):
                self.calibration_progress_bar.setValue(0)
                self.focus_status_label.setText("Calibration process initiated...")
            
            # Update video player UI
            if self.video_player_window and self.video_player_window.isVisible():
                self.video_player_window.set_status("Calibration process initiated...")
                self.video_player_window.show_calibration_progress(0)
                
            # Start fake progress timer for UI feedback
            if not self.calibration_update_timer.isActive():
                self.calibration_update_timer.start(200)

    @QtCore.pyqtSlot(str)
    def on_subscriber_connection_status(self, status):
        """Handle connection status updates from the subscriber"""
        print(f"Focus Page: EEG Connection Status: {status}")
        
        if "Connected" in status:
            # Connection established
            if self.connection_timeout_timer.isActive():
                self.connection_timeout_timer.stop()
                
            # Update work monitor UI
            if hasattr(self, 'focus_status_label'):
                self.focus_status_label.setText("EEG Connected. Initializing calibration...")
                
            # Update video player UI
            if self.video_player_window and self.video_player_window.isVisible():
                self.video_player_window.set_status("EEG Connected. Initializing calibration...")
                
        elif "Disconnected" in status or "Failed" in status or "Error" in status:
            # Connection lost
            QtWidgets.QMessageBox.warning(self, "Connection Lost",
                f"EEG Connection Issue: {status}")
            self.stop_active_session()

    @QtCore.pyqtSlot(str)
    def on_subscriber_error(self, error_message):
        """Handle error messages from the subscriber"""
        print(f"Focus Page: EEG Subscriber Error: {error_message}")
        
        # Only show critical errors to user
        if "fatal" in error_message.lower() or "cannot recover" in error_message.lower():
            QtWidgets.QMessageBox.critical(self, "EEG Error", error_message)
            self.stop_active_session()

    def _reset_ui_and_state(self):
        """Reset UI elements and internal state"""
        # Reset state flags
        self.session_goal = None
        self.is_calibrating = False
        self.is_calibrated = False
        self.calibration_progress_value = 0
        self.focus_monitoring_active = False
        self.focus_history = []
        self.focus_drop_counter = 0
        self.focus_alert_shown = False
        self.last_sent_scene_index = -1
        
        # Re-enable buttons
        if self.main_app_window and hasattr(self.main_app_window, 'is_lsl_connected'):
            self.update_button_tooltips(self.main_app_window.is_lsl_connected)
        else:
            self.update_button_tooltips(False)

    def clean_up_session(self):
        """Clean up any active sessions when widget is closed or app exits"""
        print("Focus Page: Cleaning up active session if any.")
        if self.session_goal:
            self.stop_active_session()

    def send_test_signals_to_unity(self):
        """Send test signals to Unity to verify connection"""
        try:
            # Create a simple sine wave oscillation between 0-100
            import math
            import time
            
            print("Sending test signals to Unity at 1-second intervals...")
            print(f"Using address: {UNITY_OSC_ADDRESS} on port {UNITY_OSC_PORT}")
            
            for i in range(30):  # Send for 30 seconds
                # Calculate a value that oscillates between 0-100
                value = 50 + 50 * math.sin(i * 0.2)
                
                # Send to Unity
                self.client.send_message(UNITY_OSC_ADDRESS, value)
                print(f"Sent test value: {value:.1f}")
                
                # Also send scene changes
                scene = int(value // 33)  # 0, 1, or 2 based on value
                self.client.send_message(UNITY_OSC_SCENE_ADDRESS, scene)
                
                time.sleep(1)  # Wait 1 second between signals
                
            print("Test signal sequence completed")
            
        except Exception as e:
            print(f"Error sending test signals: {e}")