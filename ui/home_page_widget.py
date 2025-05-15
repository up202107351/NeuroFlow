import os
from PyQt5 import QtWidgets, QtCore, QtGui
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