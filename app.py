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
from ui.login_widget import LoginWidget
from backend.lsl_status_checker import LSLStatusChecker

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

       
# --- Main Application Window ---
class NeuroAppMainWindow(QtWidgets.QMainWindow):
    lsl_status_changed_signal = QtCore.pyqtSignal(bool, str) # To update status from LSL checker

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFlow")
        self.setGeometry(50, 50, 900, 600)
        self.sidebar_buttons = {}
        self.is_lsl_connected = False # Track LSL connection state
        self.current_user_id = None
        self.current_username = None
        
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

        # --- User info display ---
        self.user_info_label = QtWidgets.QLabel("")
        self.user_info_label.setObjectName("userInfoLabel")
        self.user_info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.user_info_label.setStyleSheet("""
            QLabel#userInfoLabel {
                font-size: 9pt;
                color: #ccc;
                padding: 5px;
                border-top: 1px solid #444;
                font-weight: bold;
            }
        """)
        sidebar_layout.addWidget(self.user_info_label)
        
        # --- Logout button ---
        self.logout_button = QtWidgets.QPushButton("Logout")
        self.logout_button.setObjectName("logoutButton")
        self.logout_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.logout_button.setStyleSheet("""
            QPushButton#logoutButton {
                font-size: 9pt;
                color: #e74c3c;
                background-color: transparent;
                border: none;
                padding: 5px;
            }
            QPushButton#logoutButton:hover {
                text-decoration: underline;
            }
        """)
        self.logout_button.clicked.connect(self.logout)
        sidebar_layout.addWidget(self.logout_button)

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
        
    def set_user(self, user_id, username):
        """Set the current user after login"""
        self.current_user_id = user_id
        self.current_username = username
        self.user_info_label.setText(f"User: {username}")
        
        # Update any components that need user info
        if hasattr(self, 'history_page') and self.history_page:
            self.history_page.load_user_data(user_id)
        
        # Also pass user_id to meditation and focus pages if they need it
        if hasattr(self, 'meditation_page') and self.meditation_page:
            self.meditation_page.user_id = user_id
        
        if hasattr(self, 'focus_page') and self.focus_page:
            self.focus_page.user_id = user_id
        
    def logout(self):
        """Handle user logout"""
        from backend.database_manager import set_remember_token
        
        # Clear remember token
        if self.current_user_id:
            set_remember_token(self.current_user_id, False)
            
        # Clear token file
        token_file = os.path.join('app_data', 'auth_token.json')
        if os.path.exists(token_file):
            try:
                os.remove(token_file)
            except Exception as e:
                print(f"Error removing token file: {e}")
                
        # Use the stored reference to the ModernWindow wrapper
        if hasattr(self, 'modern_window') and self.modern_window:
            print("Hiding modern window wrapper")
            self.modern_window.hide()
        else:
            print("No modern_window reference, trying parent()")
            parent_window = self.parent()
            if parent_window:
                parent_window.hide()
            else:
                self.hide()
        
        # Show login window
        self.login_window.show()
        
        # Reset user data
        self.current_user_id = None
        self.current_username = None
        self.user_info_label.setText("")

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
                # self.statusBar().showMessage(f"{page_name} page loaded")
            else:
                print(f"Warning: No page mapped for button '{page_name}'")

    def update_active_button(self, active_button):
        """ Ensures the clicked button is visually marked as active (checked) """
        # The autoExclusive property handles the visual check state,
        # but we might add more styling logic here if needed in the future.
        active_button.setChecked(True) # Ensure it's checked programmatically too


# --- Main Execution Block ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Apply qtmodern style
    qtmodern.styles.dark(app)  # Explicitly set dark theme
    
    # Create login window first
    login_widget = LoginWidget()
    login_window = qtmodern.windows.ModernWindow(login_widget)
    
    # Create main app window but don't show it yet
    main_window = NeuroAppMainWindow()
    modern_window = qtmodern.windows.ModernWindow(main_window)
    
    # Store references to both windows
    main_window.login_window = login_window
    main_window.login_widget = login_widget
    main_window.modern_window = modern_window  # Store reference to its own wrapper
    
    # Connect login signal to show main window
    def on_login_successful(user_id, username):
        login_window.hide()
        main_window.set_user(user_id, username)
        modern_window.show()
    
    login_widget.login_successful.connect(on_login_successful)
    
    # Create a flag to track whether we've shown any windows
    window_shown = False
    
    # First check for auto-login
    if login_widget.check_remembered_login():
        try:
            # Read token from file
            import json
            from pathlib import Path
            token_file = Path('./app_data/auth_token.json')

            from backend.database_manager import get_user_by_token
            
            with open(token_file, 'r') as f:
                data = json.load(f)
                token = data.get('token')
                
                if token:
                    user = get_user_by_token(token)
                    if user:
                        print(f"Auto-login succeeded, directly calling handler")
                        # Directly call the handler to show main window
                        on_login_successful(user['user_id'], user['username'])
                        window_shown = True
        except Exception as e:
            print(f"Error during auto-login: {e}")
    
    # Only show login window if no window has been shown yet
    if not window_shown:
        print("Manual login required, showing login window")
        login_window.show()
    
    # Add specific styling
    app.setStyleSheet("""
        #main_widget {
            background-color: #17121C;
        }
        #sidebar {
            background-color: #2a2a2a; /* Slightly different background for sidebar */
        }
        /* Add other global styles if needed */
    """)

    sys.exit(app.exec_())