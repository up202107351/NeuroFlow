import sys
import os # For path manipulation
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
import qtmodern.styles
import qtmodern.windows
import zmq
import json
import subprocess # To launch the backend script
import time

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
    
class MeditationPageWidget(QtWidgets.QWidget):
    # Signals could be added if actions here need to communicate broadly
    # e.g., eeg_session_start_requested = QtCore.pyqtSignal(str) # "video" or "game"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend_process = None # To hold the subprocess reference
        self.prediction_subscriber = None
        self.prediction_thread = None
        self.video_player_window = None
        self.initUI()
        # Add a stop button to the Meditation Page UI
        self.btn_stop_video_feedback = QtWidgets.QPushButton("Stop Video Session")
        self.btn_stop_video_feedback.setStyleSheet("font-size: 11pt; padding: 8px 15px; background-color: #c0392b; color: white;")
        self.btn_stop_video_feedback.clicked.connect(self.stop_video_session)
        self.btn_stop_video_feedback.setEnabled(False) # Enabled when session starts

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


    def start_video_session(self):
        print("Meditation Page: Start Video Feedback clicked.")

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
    def on_new_eeg_prediction(self, prediction_data):
        # prediction_data is a dict like {"timestamp": ..., "prediction": "Relaxed"}
        prediction_label = prediction_data.get("prediction", "Unknown")
        # print(f"UI: Received Prediction: {prediction_label}")
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

# --- Content Area Widgets ---

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFlow")
        self.setGeometry(50, 50, 900, 600) # Adjusted size for dashboard layout
        self.sidebar_buttons = {} # To keep track of sidebar buttons for styling
        self.initUI()
        # Set initial active button
        self.update_active_button(self.sidebar_buttons["Home"])

    def initUI(self):
        # --- Main Widget and Layout ---
        main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0) # No margins for the main layout
        main_layout.setSpacing(0) # No spacing between sidebar and content
        main_widget.setObjectName("main_widget")

        # --- Sidebar ---
        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setObjectName("sidebar") # For potential styling
        sidebar_widget.setFixedWidth(180) # Fixed width for the sidebar
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(10, 15, 10, 15) # Internal padding
        sidebar_layout.setSpacing(10) # Spacing between buttons
        sidebar_layout.setAlignment(QtCore.Qt.AlignTop) # Align buttons to the top


        # Sidebar Buttons
        nav_items = ["Home", "Meditation", "Focus", "History"]
        for item_text in nav_items:
            button = QtWidgets.QPushButton(item_text)
            button.setCheckable(True) # Allows styling based on checked state (optional)
            button.setAutoExclusive(True) # Only one button checked at a time
            # Basic style for sidebar buttons (qtmodern provides base)
            button.setStyleSheet("""
                QPushButton {
                    font-size: 11pt;
                    text-align: left;
                    padding: 10px;
                    border: none;
                    background-color: transparent; /* Let qtmodern handle base */
                    /* color: #e0e0e0; Default text color in dark mode often ok */
                }
                QPushButton:hover {
                    background-color: #4a4a4a; /* Slightly lighter grey on hover */
                }
                QPushButton:checked {
                    background-color: #555555; /* Darker grey when selected/checked */
                    font-weight: bold;
                    /* border-left: 3px solid #007bff; Example active indicator */
                }
            """)
            button.clicked.connect(self.change_page) # Connect all buttons to the same slot
            sidebar_layout.addWidget(button)
            self.sidebar_buttons[item_text] = button # Store reference

        sidebar_layout.addStretch(1) # Push buttons to the top

        # --- Content Area (Stacked Widget) ---
        self.stacked_widget = QtWidgets.QStackedWidget()

        self.home_page = HomePageWidget()
        self.meditation_page = MeditationPageWidget() # <--- Use the new MeditationPageWidget
        self.focus_page = PlaceholderWidget("Focus")
        self.history_page = PlaceholderWidget("History")

        self.page_map = {
            "Home": self.home_page,
            "Meditation": self.meditation_page, # <--- Mapped here
            "Focus": self.focus_page,
            "History": self.history_page,
        }
        for page_widget in self.page_map.values():
            self.stacked_widget.addWidget(page_widget)

        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.stacked_widget)

        self.home_page.meditate_requested.connect(self.go_to_meditation_page) # Changed
        self.home_page.focus_requested.connect(self.go_to_focus_page)       # Changed

        self.statusBar().showMessage("Ready")

    def change_page(self):
        sender_button = self.sender()
        if sender_button:
            page_name = sender_button.text()
            if page_name in self.page_map:
                target_widget = self.page_map[page_name]
                self.stacked_widget.setCurrentWidget(target_widget)
                self.update_active_button(sender_button)
                self.statusBar().showMessage(f"{page_name} page loaded")

    def update_active_button(self, active_button):
        active_button.setChecked(True)

    def go_to_meditation_page(self):
        """Navigates directly to the Meditation page from Home."""
        print("Navigating to Meditation Page from Home")
        self.stacked_widget.setCurrentWidget(self.meditation_page)
        self.update_active_button(self.sidebar_buttons["Meditation"])
        self.statusBar().showMessage("Meditation page loaded")

    def go_to_focus_page(self):
        """Navigates directly to the Focus page from Home (or shows dialog)."""
        print("Navigating to Focus Page from Home")
        self.stacked_widget.setCurrentWidget(self.focus_page)
        self.update_active_button(self.sidebar_buttons["Focus"])
        self.statusBar().showMessage("Focus page loaded")
        # You might still want a dialog for Focus options if that page gets complex:
        # QtWidgets.QMessageBox.information(self, "Focus", "Focus Session setup (dialog) not yet implemented.")

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

    def closeEvent(self, event):
        print("Main window closing...")
        # Access the meditation page and call its cleanup
        if hasattr(self, 'meditation_page') and self.meditation_page:
            self.meditation_page.clean_up_session()
        event.accept()


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