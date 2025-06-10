from PyQt5 import QtWidgets, QtCore, QtGui
import os
from pathlib import Path
import json

# Import database functions
from backend.database_manager import authenticate_user, register_user, set_remember_token, get_user_by_token

class LoginWidget(QtWidgets.QWidget):
    login_successful = QtCore.pyqtSignal(int, str)  # user_id, username
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NeuroFlow - Login")
        self.setMinimumSize(400, 500)
        self.initUI()
        self.app_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.token_file = self.app_dir / 'app_data' / 'auth_token.json'
        print(f"Token file path set to: {self.token_file}")
        print("Application starting - checking for remembered login...")
        self.check_remembered_login()
        
        
    def initUI(self):
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Create app logo/image
        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap("./assets/logo.png")
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
        else:
            # Fallback if image not found
            logo_label.setText("NeuroFlow")
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            logo_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2ecc71;")
        
        # Stacked widget to switch between login and register
        self.stacked_widget = QtWidgets.QStackedWidget()
        
        # --- Login Page ---
        login_widget = QtWidgets.QWidget()
        login_layout = QtWidgets.QVBoxLayout(login_widget)
        login_layout.setContentsMargins(0, 0, 0, 0)
        
        # Username field
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.username_input.setMinimumHeight(40)
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #444;
                background-color: #333;
            }
            QLineEdit:focus {
                border: 1px solid #5A4275;
            }
        """)
        
        # Password field
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setMinimumHeight(40)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #444;
                background-color: #333;
            }
            QLineEdit:focus {
                border: 1px solid #5A4275;
            }
        """)
        
        # Remember me checkbox
        self.remember_checkbox = QtWidgets.QCheckBox("Stay logged in")
        self.remember_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ccc;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        
        # Login button
        self.login_button = QtWidgets.QPushButton("Log In")
        self.login_button.setMinimumHeight(40)
        self.login_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #6F72B3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #6F72B3;
                border: 1px solid #5A4275;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        
        # Register link
        register_link = QtWidgets.QPushButton("Don't have an account? Register here")
        register_link.setCursor(QtCore.Qt.PointingHandCursor)
        register_link.setFlat(True)
        register_link.setStyleSheet("""
            QPushButton {
                text-decoration: none;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                color:#ccc;
                text-decoration: underline;
            }
        """)
        
        # Error message label
        self.login_error_label = QtWidgets.QLabel("")
        self.login_error_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.login_error_label.setAlignment(QtCore.Qt.AlignCenter)
        self.login_error_label.setWordWrap(True)
        self.login_error_label.setVisible(False)
        
        # Add widgets to login layout
        login_layout.addWidget(self.username_input)
        login_layout.addWidget(self.password_input)
        login_layout.addWidget(self.remember_checkbox)
        login_layout.addWidget(self.login_button)
        login_layout.addWidget(self.login_error_label)
        login_layout.addStretch(1)
        login_layout.addWidget(register_link, alignment=QtCore.Qt.AlignCenter)
        
        # --- Register Page ---
        register_widget = QtWidgets.QWidget()
        register_layout = QtWidgets.QVBoxLayout(register_widget)
        register_layout.setContentsMargins(0, 0, 0, 0)
        
        # Register title
        register_title = QtWidgets.QLabel("Create Account")
        register_title.setAlignment(QtCore.Qt.AlignCenter)
        register_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        
        # Username field
        self.reg_username_input = QtWidgets.QLineEdit()
        self.reg_username_input.setPlaceholderText("Username")
        self.reg_username_input.setMinimumHeight(40)
        self.reg_username_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #444;
                background-color: #333;
            }
            QLineEdit:focus {
                border: 1px solid #5A4275;
            }
        """)
        
        # Password field
        self.reg_password_input = QtWidgets.QLineEdit()
        self.reg_password_input.setPlaceholderText("Password")
        self.reg_password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.reg_password_input.setMinimumHeight(40)
        self.reg_password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #444;
                background-color: #333;
            }
            QLineEdit:focus {
                border: 1px solid #5A4275;
            }
        """)
        
        # Confirm password field
        self.reg_confirm_password = QtWidgets.QLineEdit()
        self.reg_confirm_password.setPlaceholderText("Confirm Password")
        self.reg_confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.reg_confirm_password.setMinimumHeight(40)
        self.reg_confirm_password.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #444;
                background-color: #333;
            }
            QLineEdit:focus {
                border: 1px solid #5A4275;
            }
        """)
        
        # Register button
        self.register_button = QtWidgets.QPushButton("Create Account")
        self.register_button.setMinimumHeight(40)
        self.register_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #6F72B3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #6F72B3;
                border: 1px solid #5A4275;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        
        # Login link
        login_link = QtWidgets.QPushButton("Already have an account? Log in here")
        login_link.setCursor(QtCore.Qt.PointingHandCursor)
        login_link.setFlat(True)
        login_link.setStyleSheet("""
            QPushButton {
                text-decoration: none;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                text-decoration: underline;
                color: #ccc;
            }
        """)
        
        # Error message label
        self.register_error_label = QtWidgets.QLabel("")
        self.register_error_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.register_error_label.setAlignment(QtCore.Qt.AlignCenter)
        self.register_error_label.setWordWrap(True)
        self.register_error_label.setVisible(False)
        
        # Add widgets to register layout
        register_layout.addWidget(register_title)
        register_layout.addWidget(self.reg_username_input)
        register_layout.addWidget(self.reg_password_input)
        register_layout.addWidget(self.reg_confirm_password)
        register_layout.addWidget(self.register_button)
        register_layout.addWidget(self.register_error_label)
        register_layout.addStretch(1)
        register_layout.addWidget(login_link, alignment=QtCore.Qt.AlignCenter)
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(login_widget)
        self.stacked_widget.addWidget(register_widget)
        
        # Connect signals
        self.login_button.clicked.connect(self.handle_login)
        self.password_input.returnPressed.connect(self.handle_login)
        register_link.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        self.register_button.clicked.connect(self.handle_register)
        login_link.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        # Add to main layout
        main_layout.addWidget(logo_label)
        main_layout.addWidget(self.stacked_widget)
        
    def handle_login(self):
        """Process login attempt"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        remember = self.remember_checkbox.isChecked()
        
        if not username or not password:
            self.show_login_error("Please enter both username and password.")
            return
        
        success, result = authenticate_user(username, password)
        
        if success:
            user_id = result
            # Set remember token if requested
            if remember:
                token = set_remember_token(user_id, True)
                if token:
                    print(f"Generated remember token for {username}: {token[:10]}...")
                    self.save_remember_token(token)
                    
                    # Verify token was saved correctly
                    saved_user = get_user_by_token(token)
                    if saved_user:
                        print(f"Token verification successful for user: {saved_user['username']}")
                    else:
                        print("Warning: Token was not correctly saved in database")
            else:
                # Clear any existing token
                print(f"Not remembering user, clearing any existing tokens")
                set_remember_token(user_id, False)
                self.clear_remember_token()
                
            # Emit signal to notify successful login
            self.login_successful.emit(user_id, username)
        else:
            self.show_login_error(result)
    
    def handle_register(self):
        """Process registration attempt"""
        username = self.reg_username_input.text().strip()
        password = self.reg_password_input.text()
        confirm_password = self.reg_confirm_password.text()
        
        # Validate inputs
        if not username or not password:
            self.show_register_error("Please fill in all fields.")
            return
            
        if password != confirm_password:
            self.show_register_error("Passwords do not match.")
            return
            
        if len(password) < 6:
            self.show_register_error("Password must be at least 6 characters.")
            return
        
        # Attempt registration
        success, result = register_user(username, password)
        
        if success:
            user_id = result
            # Auto-login after registration
            self.stacked_widget.setCurrentIndex(0)
            self.username_input.setText(username)
            self.password_input.setText(password)
            self.show_login_error("Registration successful! You can now log in.", error=False)
        else:
            self.show_register_error(result)
    
    def show_login_error(self, message, error=True):
        """Display login error or success message"""
        self.login_error_label.setText(message)
        if error:
            self.login_error_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            self.login_error_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        self.login_error_label.setVisible(True)
    
    def show_register_error(self, message):
        """Display registration error"""
        self.register_error_label.setText(message)
        self.register_error_label.setVisible(True)
    
    def save_remember_token(self, token):
        """Save the remember token to a file"""
        token_dir = self.app_dir / 'app_data'
        token_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.token_file, 'w') as f:
                json.dump({'token': token}, f)
            print(f"Token saved successfully to: {self.token_file}")
        except Exception as e:
            print(f"Error saving token: {e}")

    def clear_remember_token(self):
        """Clear the saved remember token"""
        if self.token_file.exists():
            try:
                self.token_file.unlink()
                print(f"Token file deleted: {self.token_file}")
            except Exception as e:
                print(f"Error deleting token file: {e}")
    
    def check_remembered_login(self):
        """Check if there's a saved token and attempt auto-login"""
        token_file = Path('./app_data/auth_token.json')
        print(f"Checking for remembered login at: {token_file.absolute()}")
        
        if not token_file.exists():
            print("No token file found - user needs to log in manually")
            return False
                
        try:
            with open(token_file, 'r') as f:
                data = json.load(f)
                token = data.get('token')
                
                if token:
                    print(f"Token found in file: {token[:10]}...")
                    user = get_user_by_token(token)
                    
                    if user:
                        print(f"Auto-login successful for: {user['username']}")
                        # Set the flag for auto-login success
                        self.auto_login_succeeded = True
                        # Emit signal
                        self.login_successful.emit(user['user_id'], user['username'])
                        return True
                    else:
                        print("Token verification failed - no matching user in database")
                else:
                    print("Token file exists but contains no token")
        except Exception as e:
            print(f"Error checking remembered login: {e}")
            
        return False