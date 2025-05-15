import os
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
import subprocess # To launch the backend script
import matplotlib
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib

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