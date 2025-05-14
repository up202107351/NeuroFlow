import sys
import os
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
import qtmodern.styles
import qtmodern.windows
import zmq
import json
import subprocess # To launch the backend script
import time
import pylsl 
import time  
import database_manager as db_manager 
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import seaborn as sns # For nicer plots
import pandas as pd 

# --- EEG Prediction Subscriber Thread (for PyQt app) ---
class EEGPredictionSubscriber(QtCore.QObject):
    new_prediction_received = QtCore.pyqtSignal(dict) # Emit the raw dict
    subscriber_error = QtCore.pyqtSignal(str)
    connection_status = QtCore.pyqtSignal(str) # "Connecting", "Connected", "Disconnected"
    finished = QtCore.pyqtSignal()

    def __init__(self, zmq_sub_address="tcp://localhost:5556"):
        super().__init__()
        self.zmq_sub_address = zmq_sub_address
        self._running = False
        self.context = None
        self.subscriber = None

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        print(f"Frontend: Attempting to connect to ZMQ PUB at {self.zmq_sub_address}")
        self.connection_status.emit(f"Connecting to {self.zmq_sub_address}...")
        try:
            self.subscriber.connect(self.zmq_sub_address)
            self.subscriber.subscribe("") # Subscribe to all messages
            self.connection_status.emit(f"Connected to EEG Backend.")
            print(f"Frontend: Connected to ZMQ Publisher.")
        except zmq.error.ZMQError as e:
            err_msg = f"Frontend: ZMQ connection error: {e}"
            print(err_msg)
            self.subscriber_error.emit(err_msg)
            self.connection_status.emit(f"Connection Failed: {e}")
            self._running = False # Stop if connection fails initially

        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN)

        while self._running:
            try:
                # Poll with a timeout (e.g., 1000ms) to allow checking _running flag
                socks = dict(poller.poll(timeout=1000))
                if self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN:
                    message_json = self.subscriber.recv_json()
                    self.new_prediction_received.emit(message_json)
            except zmq.error.ContextTerminated:
                print("Frontend: ZMQ Context terminated, subscriber stopping.")
                self._running = False # Exit loop if context is terminated
            except Exception as e:
                # Avoid flooding with errors if backend is down after initial connect
                if self._running: # Only log if we are supposed to be running
                    print(f"Frontend: Error receiving/parsing ZMQ message: {e}")
                    self.subscriber_error.emit(f"ZMQ receive error: {e}")
                    # Optionally emit a disconnected status here if errors persist
                time.sleep(0.1) # Small pause

        # --- Ensure 'finished' is emitted when the loop exits ---
        if self.subscriber:
            self.subscriber.close()
        if self.context:
            # Be careful with term() if other sockets from this context might still be in use
            # If this is the only user of self.context, then it's okay.
            if not self.context.closed: # Check if not already closed
                try:
                    self.context.term()
                except zmq.error.ZMQError as e:
                    print(f"Frontend: Error terminating ZMQ context: {e}")

        print("Frontend: EEGPredictionSubscriber run loop finished.")
        self.connection_status.emit("Disconnected from EEG Backend.") # Update status
        self.finished.emit() # <--- EMIT THE SIGNAL HERE

    def stop(self):
        print("Frontend: EEGPredictionSubscriber stop requested.")
        self._running = False
        # Note: The 'finished' signal will be emitted from the run() method
        # when its loop naturally exits due to _running being False.

class LSLStatusChecker(QtCore.QObject):
    status_update = QtCore.pyqtSignal(bool, str) # connected (bool), message (str)
    finished = QtCore.pyqtSignal()

    def __init__(self, stream_type='EEG', resolve_timeout=1, check_interval=3): # Short timeout for check
        super().__init__()
        self.stream_type = stream_type
        self.resolve_timeout = resolve_timeout
        self.check_interval = check_interval # Seconds between checks
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        print("LSLStatusChecker: Thread started.")
        while self._running:
            try:
                streams = pylsl.resolve_byprop('type', self.stream_type, 1, timeout=self.resolve_timeout)
                if streams:
                    # To be more robust, you could even try to open an inlet briefly
                    # inlet = pylsl.StreamInlet(streams[0], max_chunklen=1, max_buffered=1, processing_flags=pylsl.proc_dejitter)
                    # inlet.open_stream(timeout=0.5) # Try to open with short timeout
                    # inlet.close_stream()
                    self.status_update.emit(True, "Connected")
                else:
                    self.status_update.emit(False, "Not Detected")
            except Exception as e:
                print(f"LSLStatusChecker: Error during LSL check: {e}")
                self.status_update.emit(False, "Error Checking")

            # Wait for the check_interval, but break early if stop is requested
            for _ in range(int(self.check_interval / 0.1)): # Check stop flag every 0.1s
                if not self._running:
                    break
                time.sleep(0.1)
            if not self._running: # Ensure loop breaks if stop() was called during sleep
                break

        print("LSLStatusChecker: Thread finished.")
        self.finished.emit()

    def stop(self):
        print("LSLStatusChecker: Stop requested.")
        self._running = False


class VideoPlayerWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Feedback Session")
        self.setGeometry(200, 200, 640, 480)
        layout = QtWidgets.QVBoxLayout(self)

        self.video_widget_placeholder = QtWidgets.QLabel("Video Player Area")
        self.video_widget_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.video_widget_placeholder.setMinimumSize(600, 400)
        self.video_widget_placeholder.setStyleSheet("background-color: black; color: white; font-size: 20pt;")
        layout.addWidget(self.video_widget_placeholder)

        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def set_status(self, message):
        if hasattr(self, 'status_label'): # Check if UI elements are initialized
            self.status_label.setText(message)
        else:
            print(f"VideoPlayerWindow Status (UI not ready): {message}")


    def update_based_on_prediction(self, prediction_label):
        self.set_status(f"Current State: {prediction_label}")
        # TODO: Actual video control logic
        pass

    def closeEvent(self, event):
        # Optional: notify parent or clean up player
        print("VideoPlayerWindow closing.")
        event.accept()

# --- Placeholder Dialogs (Keep these from previous code) ---
# --- MeditationSelectionDialog (Modified) ---
class MeditationSelectionDialog(QtWidgets.QDialog):
    VIDEO_FEEDBACK = 1
    UNITY_GAME = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Meditation Type")
        self.setModal(True)
        self.selection = None # To store what the user selected
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Choose your meditation feedback method:")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFont(QtGui.QFont("Arial", 12))
        layout.addWidget(label)

        self.btn_video = QtWidgets.QPushButton("Video Feedback (EEG Based)")
        self.btn_game = QtWidgets.QPushButton("Start Unity Game")
        button_style = "QPushButton { font-size: 11pt; padding: 8px; }"
        self.btn_video.setStyleSheet(button_style)
        self.btn_game.setStyleSheet(button_style)

        layout.addSpacing(15)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.btn_game)
        layout.addSpacing(10)

        self.btn_video.clicked.connect(self.select_video)
        self.btn_game.clicked.connect(self.select_game)

    def select_video(self):
        print("Dialog: Video Feedback selected!")
        self.selection = MeditationSelectionDialog.VIDEO_FEEDBACK
        self.accept() # Closes dialog, signals success

    def select_game(self):
        print("Dialog: Unity Game selected!")
        self.selection = MeditationSelectionDialog.UNITY_GAME
        self.accept() # Closes dialog, signals success

    def get_selection(self):
        return self.selection
    
class HistoryPageWidget(QtWidgets.QWidget):
    def __init__(self, parent = None, main_app_window_ref = None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.initUI()

    def initUI(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QtWidgets.QLabel("Session History")
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # --- Controls (e.g., refresh button, date filters - optional for now) ---
        self.btn_refresh = QtWidgets.QPushButton("Refresh History")
        self.btn_refresh.clicked.connect(self.load_history_data)
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addStretch()
        control_layout.addWidget(self.btn_refresh)
        main_layout.addLayout(control_layout)


        # --- Session List (Table View) ---
        self.sessions_table = QtWidgets.QTableWidget()
        self.sessions_table.setColumnCount(5) # session_id, type, start_time, duration, % on target
        self.sessions_table.setHorizontalHeaderLabels(["ID", "Type", "Start Time", "Duration (s)", "% On Target"])
        self.sessions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sessions_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sessions_table.itemSelectionChanged.connect(self.on_session_selected)
        main_layout.addWidget(self.sessions_table)

        # --- Details Area (for selected session's graph) ---
        self.details_area_label = QtWidgets.QLabel("Select a session to view details.")
        self.details_area_label.setAlignment(QtCore.Qt.AlignCenter)

        # Matplotlib Canvas
        self.figure = Figure(figsize=(5, 3), dpi=100) # Smaller figure for embedding
        self.canvas = FigureCanvas(self.figure)
        # Initially hide canvas until data is loaded
        self.canvas.setVisible(False)


        details_layout = QtWidgets.QVBoxLayout()
        details_layout.addWidget(self.details_area_label)
        details_layout.addWidget(self.canvas) # Add canvas here

        main_layout.addLayout(details_layout)

        self.load_history_data() # Load data when page is created


    def load_history_data(self):
        self.sessions_table.setRowCount(0) # Clear existing rows
        self.canvas.setVisible(False) # Hide graph
        self.details_area_label.setText("Select a session to view details.")
        self.details_area_label.setVisible(True)

        try:
            sessions_data = db_manager.get_all_sessions_summary()
            self.sessions_table.setRowCount(len(sessions_data))

            for row_idx, session_row in enumerate(sessions_data):
                self.sessions_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(session_row['session_id'])))
                self.sessions_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(session_row['session_type']))
                start_time_dt = datetime.fromisoformat(session_row['start_time'])
                self.sessions_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(start_time_dt.strftime("%Y-%m-%d %H:%M")))
                self.sessions_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(session_row['duration_seconds'])))
                percent_on_target = f"{session_row['percent_on_target']:.1f}%" if session_row['percent_on_target'] is not None else "N/A"
                self.sessions_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(percent_on_target))
            self.sessions_table.resizeColumnsToContents()
        except Exception as e:
            print(f"Error loading history data: {e}")
            QtWidgets.QMessageBox.critical(self, "DB Error", f"Could not load session history: {e}")

    def on_session_selected(self):
        selected_items = self.sessions_table.selectedItems()
        if not selected_items:
            self.canvas.setVisible(False)
            self.details_area_label.setText("Select a session to view details.")
            self.details_area_label.setVisible(True)
            return

        selected_row = selected_items[0].row()
        session_id_item = self.sessions_table.item(selected_row, 0)
        if not session_id_item: return

        try:
            session_id = int(session_id_item.text())
            session_details = db_manager.get_session_details(session_id)

            if not session_details:
                self.details_area_label.setText(f"No detailed metrics found for session {session_id}.")
                self.details_area_label.setVisible(True)
                self.canvas.setVisible(False)
                return

            self.plot_session_details(session_details)
            self.details_area_label.setVisible(False) # Hide placeholder text
            self.canvas.setVisible(True) # Show graph

        except Exception as e:
            print(f"Error loading/plotting session details: {e}")
            self.details_area_label.setText(f"Error loading details: {e}")
            self.details_area_label.setVisible(True)
            self.canvas.setVisible(False)


    def plot_session_details(self, details_data):
        """Plots session metrics (e.g., prediction over time)."""
        self.figure.clear() # Clear previous plot
        ax = self.figure.add_subplot(111)

        if not details_data:
            ax.text(0.5, 0.5, 'No data to display for this session.',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()
            return

        # Convert to DataFrame for easier plotting with Seaborn/Matplotlib
        df = pd.DataFrame(details_data, columns=['timestamp', 'prediction_label', 'is_on_target', 'raw_score'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

        # Example plot: prediction_label over time (as a categorical plot or step plot)
        # For simplicity, let's map predictions to numerical values for a line plot
        # This needs to be adapted based on what 'prediction_label' contains
        prediction_map = {"Relaxed": 1, "Neutral": 0, "Agitated/Focused": -1, "Focused": -1, "Unknown": 0} # Example
        df['prediction_value'] = df['prediction_label'].map(prediction_map).fillna(0)

        # sns.lineplot(x='timestamp', y='prediction_value', data=df, ax=ax, marker='o', hue='is_on_target') # Hue by on_target
        ax.plot(df['timestamp'], df['prediction_value'], marker='.', linestyle='-')
        ax.set_yticks(list(prediction_map.values()))
        ax.set_yticklabels(list(prediction_map.keys()))


        ax.set_title("Session State Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Detected State")
        self.figure.autofmt_xdate() # Auto format date on x-axis
        ax.grid(True, linestyle='--', alpha=0.7)

        self.canvas.draw() # Redraw the canvas

    def update_button_tooltips(self, is_lsl_connected_from_main_app):
        # History page doesn't typically have LSL-dependent start buttons
        pass

    def clean_up_session(self): # Usually not needed for history page
        pass
    
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


    @QtCore.pyqtSlot(dict)
    def on_new_eeg_prediction_received(self, prediction_data):
        if self.current_session_id is None:
            return # No active session to log for

        prediction_label = prediction_data.get("prediction", "Unknown")
        raw_score = prediction_data.get("score") # If your backend sends a numeric score

        is_on_target = (prediction_label == self.session_target_label)

        db_manager.add_session_metric(self.current_session_id, prediction_label, is_on_target, raw_score)

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

class FocusPageWidget(QtWidgets.QWidget):
    def __init__(self, parent = None, main_app_window_ref = None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.initUI()
        if self.main_app_window:
            self.update_button_tooltips(self.main_app_window.is_lsl_connected)
        else:
            self.update_button_tooltips(False)

    def initUI(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        # --- Title ---
        title_label = QtWidgets.QLabel("Choose Your Focus Session")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        main_layout.addWidget(title_label)
        main_layout.addSpacing(25)

        # --- First Row of Focus Options (Work, Video) ---
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(30) # Spacing between items in this row

        # Option 1: Work Focus Session
        work_focus_layout = self.create_focus_option_layout(
            title="Work Session",
            image_path="work.jpg", # <-- SET PATH!
            button_text="Start",
            action_slot=self.start_work_focus_session
        )
        row1_layout.addLayout(work_focus_layout)

        # Option 2: Video Focus Session
        video_focus_layout = self.create_focus_option_layout(
            title="Video Session",
            image_path="focus.jpg", # <-- SET PATH!
            button_text="Start",
            action_slot=self.start_video_focus_session
        )
        row1_layout.addLayout(video_focus_layout)
        main_layout.addLayout(row1_layout)
        main_layout.addSpacing(25)

        # --- Second Row of Focus Options (Game) - Centered ---
        row2_outer_layout = QtWidgets.QHBoxLayout() # To help with centering
        row2_outer_layout.addStretch(1) # Push game option to center

        game_focus_layout = self.create_focus_option_layout(
            title="Game Session",
            image_path="focus_game.jpg", # <-- SET PATH!
            button_text="Start",
            action_slot=self.start_game_focus_session,
            is_single_item_row=True # Hint for potential size adjustment
        )
        row2_outer_layout.addLayout(game_focus_layout)
        row2_outer_layout.addStretch(1) # Push game option to center
        main_layout.addLayout(row2_outer_layout)

        main_layout.addStretch(1) # Push all content towards the top


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
        if hasattr(self, 'btn_work_focus'):
            self.btn_work_focus.setToolTip(tooltip_text)
        if hasattr(self, 'btn_video_focus'):
            self.btn_video_focus.setToolTip(tooltip_text)
        if hasattr(self, 'btn_game_focus'):
            self.btn_game_focus.setToolTip(tooltip_text)

    # --- Action Slots for Focus Page Buttons ---
    def start_work_focus_session(self):
        print("Focus Page: Clicked Start Work Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
        # TODO: Implement Work Focus Session logic
        # This might involve background LSL processing (similar to video feedback)
        # and periodic notifications/alerts if focus drops.
        QtWidgets.QMessageBox.information(self, "Work Focus", "Work Focus Session starting... (Not yet implemented)")

    def start_video_focus_session(self):
        print("Focus Page: Clicked Start Video Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
        # TODO: Implement Video Focus Session logic
        # This could be similar to Meditation video feedback, but with different videos/goals
        # Potentially reusing or adapting the MeditationPageWidget's video session logic.
        QtWidgets.QMessageBox.information(self, "Video Focus", "Video Focus Session starting... (Not yet implemented)")
        # Example: Could launch the same backend process if the logic is similar
        # if hasattr(self.main_app_window.meditation_page, 'start_video_session'): # A bit hacky
        #     print("Attempting to use Meditation Page's video session logic for focus...")
        #     # This would require careful state management if both use the same backend instance
        #     # self.main_app_window.meditation_page.start_video_session()

    def start_game_focus_session(self):
        print("Focus Page: Clicked Start Game Focus Session.")
        if not self.main_app_window.is_lsl_connected:
            QtWidgets.QMessageBox.warning(self, "Muse Not Connected",
                                          "Please connect Muse to start.")
            return
        # TODO: Implement Game Focus Session logic
        # This could launch the same or a different Unity game.
        # Re-use launch_unity_game logic from MeditationPageWidget if applicable.
        unity_game_path = r"C:/path/to/your/focus_unity_game.exe" # <-- SET PATH! (or use a general game path)
        if not os.path.exists(unity_game_path):
            QtWidgets.QMessageBox.warning(self, "Error", f"Focus game not found at:\n{unity_game_path}")
            return
        try:
            subprocess.Popen([unity_game_path])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error launching focus game:\n{e}")
        QtWidgets.QMessageBox.information(self, "Game Focus", "Game Focus Session starting... (Not yet implemented)")


    def clean_up_session(self):
        """If focus sessions launch persistent backends, add cleanup here."""
        print("FocusPageWidget: Cleaning up active session if any.")
        # TODO: Add logic similar to MeditationPageWidget.clean_up_session if needed
        # e.g., if self.work_focus_backend_process: self.work_focus_backend_process.terminate()
        pass

class HomePageWidget(QtWidgets.QWidget):
    # Define signals to emit when buttons are clicked
    meditate_requested = QtCore.pyqtSignal()
    focus_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40) # Add padding around the content
        layout.setAlignment(QtCore.Qt.AlignCenter) # Center content vertically

        # --- Logo ---
        logo_label = QtWidgets.QLabel()
        # IMPORTANT: Replace with the actual path to your logo image
        logo_path = "logo.png" 
        if os.path.exists(logo_path):
             pixmap = QtGui.QPixmap(logo_path)
             # Scale pixmap smoothly while keeping aspect ratio
             scaled_pixmap = pixmap.scaled(128, 128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
             logo_label.setPixmap(scaled_pixmap)
        else:
             print(f"Warning: Logo image not found at {logo_path}")
             logo_label.setText("(Logo Not Found)") # Placeholder text
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)
        layout.addSpacing(20)

        # --- Text ---
        title_label = QtWidgets.QLabel("Welcome to NeuroFlow")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        # Adjust font size and weight as needed
        title_font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        subtitle_label = QtWidgets.QLabel("Your app to improve relaxation and focus.")
        subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        subtitle_font = QtGui.QFont("Arial", 11)
        subtitle_label.setFont(subtitle_font)
        # Optionally make subtitle slightly grey
        # subtitle_label.setStyleSheet("color: #cccccc;") # Example grey color
        layout.addWidget(subtitle_label)

        layout.addSpacing(30) # Space before buttons

        # --- Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setAlignment(QtCore.Qt.AlignCenter) # Center buttons horizontally

        self.btn_meditate = QtWidgets.QPushButton("Start Meditating")
        self.btn_focus = QtWidgets.QPushButton("Start Focusing")

        # Style for main action buttons
        action_button_style = "QPushButton { font-size: 12pt; padding: 12px 25px; }" # Wider padding
        self.btn_meditate.setStyleSheet(action_button_style)
        self.btn_focus.setStyleSheet(action_button_style)

        button_layout.addWidget(self.btn_meditate)
        button_layout.addSpacing(20) # Space between buttons
        button_layout.addWidget(self.btn_focus)

        layout.addLayout(button_layout) # Add button row to main vertical layout
        layout.addStretch(1) # Push content towards the center/top

        # Connect internal buttons to emit signals
        self.btn_meditate.clicked.connect(self.meditate_requested.emit)
        self.btn_focus.clicked.connect(self.focus_requested.emit)
    

class PlaceholderWidget(QtWidgets.QWidget):
    """Simple widget to show for pages not yet implemented"""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        label = QtWidgets.QLabel(f"{text} Page Content")
        label.setFont(QtGui.QFont("Arial", 16))
        layout.addWidget(label)


# --- Main Application Window ---
class NeuroAppMainWindow(QtWidgets.QMainWindow):
    lsl_status_changed_signal = QtCore.pyqtSignal(bool, str) # To update status from LSL checker

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFlow")
        self.setGeometry(50, 50, 900, 600)
        self.sidebar_buttons = {}
        self.is_lsl_connected = False # Track LSL connection state
        self.initUI()
        self.update_active_button(self.sidebar_buttons["Home"])

        # LSL Status Checker
        self.lsl_checker_thread = None
        self.lsl_checker_worker = None
        self.start_lsl_status_checker()

        # Connect the signal for updating UI
        self.lsl_status_changed_signal.connect(self.update_lsl_status_ui)


    def initUI(self):
        # --- Main Widget and Layout ---
        main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar ---
        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(180)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(10, 15, 10, 10) # Adjusted bottom margin
        sidebar_layout.setSpacing(10)
        sidebar_layout.setAlignment(QtCore.Qt.AlignTop)

        nav_items = ["Home", "Meditation", "Focus", "History"]
        for item_text in nav_items:
            button = QtWidgets.QPushButton(item_text)
            # ... (rest of button setup as before) ...
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setStyleSheet("""
                QPushButton { font-size: 11pt; text-align: left; padding: 10px; border: none; background-color: transparent; }
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:checked { background-color: #555555; font-weight: bold; }
            """)
            button.clicked.connect(self.change_page)
            sidebar_layout.addWidget(button)
            self.sidebar_buttons[item_text] = button

        sidebar_layout.addStretch(1) # Push nav items up

        # --- LSL Status Label in Sidebar ---
        self.lsl_status_label = QtWidgets.QLabel("Muse: Checking...")
        self.lsl_status_label.setObjectName("lslStatusLabel")
        self.lsl_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.lsl_status_label.setWordWrap(True)
        self.lsl_status_label.setStyleSheet("""
            QLabel#lslStatusLabel {
                font-size: 9pt;
                color: #aaa; /* Neutral color */
                padding: 5px;
                border-top: 1px solid #444; /* Separator line */
                margin-top: 10px;
            }
        """)
        sidebar_layout.addWidget(self.lsl_status_label)
        # --- END Sidebar LSL Status ---

        # --- Content Area (Stacked Widget) ---
        self.stacked_widget = QtWidgets.QStackedWidget()
        # ... (rest of stacked widget and page setup as before) ...
        self.home_page = HomePageWidget()
        self.meditation_page = MeditationPageWidget(self, main_app_window_ref=self)
        self.focus_page = FocusPageWidget(self, main_app_window_ref=self)
        self.history_page = HistoryPageWidget(self, main_app_window_ref=self) # Placeholder for now

        self.page_map = {
            "Home": self.home_page,
            "Meditation": self.meditation_page,
            "Focus": self.focus_page,
            "History": self.history_page,
        }
        for page_widget in self.page_map.values():
            self.stacked_widget.addWidget(page_widget)

        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.stacked_widget)

        self.home_page.meditate_requested.connect(self.go_to_meditation_page) 
        self.home_page.focus_requested.connect(self.go_to_focus_page) 

        # Remove the main window status bar as it's no longer used for LSL status
        # self.statusBar().showMessage("Ready") # Remove or repurpose
        self.setStatusBar(None) # Completely remove status bar


    def update_lsl_status_ui(self, connected, message):
        """Updates the LSL status label in the sidebar and internal state."""
        self.is_lsl_connected = connected
        self.lsl_status_label.setText(f"Muse: {message}")
        if connected:
            self.lsl_status_label.setStyleSheet("""
                QLabel#lslStatusLabel {
                    font-size: 9pt; color: #2ecc71; /* Green for connected */
                    padding: 5px; border-top: 1px solid #444; margin-top: 10px;
                }
            """)
        else:
            self.lsl_status_label.setStyleSheet("""
                QLabel#lslStatusLabel {
                    font-size: 9pt; color: #e74c3c; /* Red for disconnected */
                    padding: 5px; border-top: 1px solid #444; margin-top: 10px;
                }
            """)

        # If on meditation/focus page, you might want to update their buttons too
        if self.stacked_widget.currentWidget() == self.meditation_page:
             self.meditation_page.update_button_states(self.is_lsl_connected)
        # Similarly for focus page if it has start buttons


    def start_lsl_status_checker(self):
        if self.lsl_checker_thread and self.lsl_checker_thread.isRunning():
            return # Already running

        print("MainApp: Starting LSL status checker thread.")
        self.lsl_checker_worker = LSLStatusChecker()
        self.lsl_checker_thread = QtCore.QThread()
        self.lsl_checker_worker.moveToThread(self.lsl_checker_thread)

        self.lsl_checker_worker.status_update.connect(self.lsl_status_changed_signal.emit) # Connect worker's signal
        self.lsl_checker_thread.started.connect(self.lsl_checker_worker.run)
        self.lsl_checker_worker.finished.connect(self.lsl_checker_thread.quit)
        self.lsl_checker_worker.finished.connect(self.lsl_checker_worker.deleteLater)
        self.lsl_checker_thread.finished.connect(self.lsl_checker_thread.deleteLater)
        self.lsl_checker_thread.start()

    def stop_lsl_status_checker(self):
        print("MainApp: Stopping LSL status checker.")
        if self.lsl_checker_worker:
            self.lsl_checker_worker.stop()
        # Thread will quit and clean up via connected 'finished' signals

    # ... (go_to_meditation_page, go_to_focus_page, change_page, update_active_button as before) ...
    def go_to_meditation_page(self):
        self.stacked_widget.setCurrentWidget(self.meditation_page)
        self.update_active_button(self.sidebar_buttons["Meditation"])
        self.meditation_page.update_button_states(self.is_lsl_connected) # Ensure buttons on page are correct

    def go_to_focus_page(self):
        self.stacked_widget.setCurrentWidget(self.focus_page)
        self.update_active_button(self.sidebar_buttons["Focus"])
        self.focus_page.update_button_tooltips(self.is_lsl_connected) # Update tooltips based on LSL status
        # If focus page has buttons needing LSL, call its update_button_states here

    def closeEvent(self, event):
        print("Main window closing...")
        self.stop_lsl_status_checker() # Stop the checker thread
        if hasattr(self, 'meditation_page') and self.meditation_page:
            self.meditation_page.clean_up_session() # If a session was running
        event.accept()

    def change_page(self):
        """Slot activated when a sidebar button is clicked."""
        sender_button = self.sender() # Get the button that triggered the signal
        if sender_button:
            page_name = sender_button.text()
            if page_name in self.page_map:
                target_widget = self.page_map[page_name]
                self.stacked_widget.setCurrentWidget(target_widget)
                self.update_active_button(sender_button)
                self.statusBar().showMessage(f"{page_name} page loaded")
            else:
                print(f"Warning: No page mapped for button '{page_name}'")

    def update_active_button(self, active_button):
        """ Ensures the clicked button is visually marked as active (checked) """
        # The autoExclusive property handles the visual check state,
        # but we might add more styling logic here if needed in the future.
        active_button.setChecked(True) # Ensure it's checked programmatically too


    # --- Action Methods ---
    # These methods are now triggered by signals from the HomePageWidget or potentially
    # directly if we add corresponding buttons to the Meditation/Focus pages later.

    def start_meditation_dialog(self):
        """Shows the dialog to choose meditation type."""
        print("Meditation requested from Home Page")
        self.statusBar().showMessage("Opening Meditation setup...")
        dialog = MeditationSelectionDialog(self)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            print("Meditation selection dialog closed successfully.")
            self.statusBar().showMessage("Meditation setup completed or initiated.")
        else:
            print("Meditation selection dialog cancelled.")
            self.statusBar().showMessage("Meditation setup cancelled.")

    def start_focus_dialog(self):
        """Placeholder for focus setup."""
        print("Focus requested from Home Page")
        self.statusBar().showMessage("Starting Focus setup...")
        # TODO: Implement focus selection dialog (Work monitoring vs Session)
        QtWidgets.QMessageBox.information(self, "Focus", "Focus Session setup not yet implemented.")


# --- Main Execution Block ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    main_window = NeuroAppMainWindow()

    # Apply qtmodern style (ensure qtmodern is installed: pip install qtmodern)
    mw = qtmodern.windows.ModernWindow(main_window)
    qtmodern.styles.dark(app) # Explicitly set dark theme

    # Add specific styling for the sidebar (optional, qtmodern does a lot)
    # You can load from a .qss file or define here
    # Using objectName allows specific targeting
    app.setStyleSheet("""
        #main_widget {
            background-color: #17121C;ZXCZXC
        }
        #sidebar {
            background-color: #2a2a2a; /* Slightly different background for sidebar */
        }
        /* Add other global styles if needed */
    """)

    mw.show()
    sys.exit(app.exec_())