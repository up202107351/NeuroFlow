import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia, QtMultimediaWidgets
import qtmodern.styles
import qtmodern.windows
import matplotlib
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from ui.home_page_widget import HomePageWidget
from ui.meditation_page_widget import MeditationPageWidget
from ui.focus_page_widget import FocusPageWidget
from ui.history_page_widget import HistoryPageWidget
from backend.lsl_status_checker import LSLStatusChecker

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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