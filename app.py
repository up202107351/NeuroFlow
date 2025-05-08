import sys
import os # For path manipulation
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
import qtmodern.styles
import qtmodern.windows

# --- Placeholder Dialogs (Keep these from previous code) ---
class MeditationSelectionDialog(QtWidgets.QDialog):
    # (Keep the MeditationSelectionDialog class code from the previous example here)
    # ... (Ensure it includes initUI, select_video, select_game)
    # ... (Make sure the game path is updated if necessary)
    def __init__(self, parent=None): # parent allows association with main window
        super().__init__(parent)
        self.setWindowTitle("Select Meditation Type")
        self.setModal(True) # Block main window while this is open
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self) # Apply layout directly to the dialog

        label = QtWidgets.QLabel("Choose your meditation feedback method:")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFont(QtGui.QFont("Arial", 12))
        layout.addWidget(label)

        # Buttons
        self.btn_video = QtWidgets.QPushButton("Video Feedback (EEG Based)")
        self.btn_game = QtWidgets.QPushButton("Start Unity Game")

        # Style buttons (optional - qtmodern handles base styling)
        button_style = "QPushButton { font-size: 11pt; padding: 8px; }"
        self.btn_video.setStyleSheet(button_style)
        self.btn_game.setStyleSheet(button_style)

        layout.addSpacing(15) # Add some space
        layout.addWidget(self.btn_video)
        layout.addWidget(self.btn_game)
        layout.addSpacing(10)

        # Connect buttons
        self.btn_video.clicked.connect(self.select_video)
        self.btn_game.clicked.connect(self.select_game)

    def select_video(self):
        print("Video Feedback selected!")
        QtWidgets.QMessageBox.information(self, "Video Feedback",
                                          "Video feedback selected.\n\n(Backend communication and video player not yet implemented)")
        self.accept() # Close the dialog signaling success/selection made

    def select_game(self):
        print("Unity Game selected!")
        # --- LAUNCH UNITY GAME ---
        # IMPORTANT: SET THIS PATH! Use forward slashes or escaped backslashes
        unity_game_path = r"C:/path/to/your/unity/game.exe" # <-- EXAMPLE PATH! CHANGE THIS!
        # Or use relative path if game is nearby:
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # unity_game_path = os.path.join(script_dir, "YourGameFolder", "game.exe")

        if not os.path.exists(unity_game_path):
             print(f"Error: Game executable not found at calculated path: {unity_game_path}")
             QtWidgets.QMessageBox.warning(self, "Error",
                                        f"Could not find the Unity game executable at the specified path:\n{unity_game_path}\n\nPlease check the path in the code (MeditationSelectionDialog class).")
             return # Don't close dialog

        try:
            print(f"Attempting to launch: {unity_game_path}")
            subprocess.Popen([unity_game_path])
            # No message box here, just launch and close dialog
            self.accept() # Close the dialog
        except Exception as e:
            print(f"Error launching game: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An unexpected error occurred launching the game:\n{e}")
            # Don't close the dialog on error

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
        logo_path = "logo.png" # <-- !!! SET THIS PATH !!!
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

        # Create pages
        self.home_page = HomePageWidget()
        self.meditation_page = PlaceholderWidget("Meditation") # Replace later
        self.focus_page = PlaceholderWidget("Focus")       # Replace later
        self.history_page = PlaceholderWidget("History")     # Replace later

        # Add pages to stacked widget (order matters for index)
        self.page_map = { # Map button text to page widget
            "Home": self.home_page,
            "Meditation": self.meditation_page,
            "Focus": self.focus_page,
            "History": self.history_page,
        }
        for page_widget in self.page_map.values():
            self.stacked_widget.addWidget(page_widget)

        # --- Assemble Main Layout ---
        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.stacked_widget)

        # --- Connect Home Page Signals ---
        self.home_page.meditate_requested.connect(self.start_meditation_dialog)
        self.home_page.focus_requested.connect(self.start_focus_dialog)

        # --- Status Bar ---
        self.statusBar().showMessage("Ready")


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