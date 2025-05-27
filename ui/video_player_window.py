from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
import os
# import time # Not used

class VideoPlayerWindow(QtWidgets.QMainWindow):
    session_stopped = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_scene = "neutral"
        self.current_level = 0.5
        self.session_type = None
        self.is_closing_initiated = False
        self.using_placeholder = False
        self.initUI()
        
        # Add a timer for smoother UI updates during calibration
        self.ui_update_timer = QtCore.QTimer(self)
        self.ui_update_timer.timeout.connect(self.process_events)
        self.ui_update_timer.setInterval(100)  # Update every 100ms

    def initUI(self):
        self.setWindowTitle("Neurofeedback Session")
        self.setMinimumSize(800, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        self.stacked_widget = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stacked_widget, 1) # Video/placeholder area takes most space

        # Video player
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.media_player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.stacked_widget.addWidget(self.video_widget)

        # Placeholder widget
        self.placeholder_widget = QtWidgets.QWidget()
        placeholder_main_layout = QtWidgets.QVBoxLayout(self.placeholder_widget) # Layout for the placeholder_widget
        placeholder_main_layout.setAlignment(QtCore.Qt.AlignCenter) # Center content within placeholder_widget

        placeholder_container = QtWidgets.QWidget() # Container for sizing
        placeholder_container.setFixedSize(600, 350)
        placeholder_container_layout = QtWidgets.QVBoxLayout(placeholder_container)
        placeholder_container_layout.setContentsMargins(0,0,0,0)

        self.placeholder_label = QtWidgets.QLabel("Video not available.\nUsing visual feedback instead.")
        self.placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 16pt; color: white;")
        self.placeholder_label.setWordWrap(True)
        placeholder_container_layout.addWidget(self.placeholder_label)
        
        placeholder_main_layout.addWidget(placeholder_container) # Add sized container to placeholder_widget's layout
        self.stacked_widget.addWidget(self.placeholder_widget)


        # Effect overlay - make it a child of stacked_widget
        self.effect_overlay = QtWidgets.QLabel(self.stacked_widget) # Child of stacked_widget
        self.effect_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.effect_overlay.hide() # Start hidden, show when effects are applied

        # Status bar
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setFixedHeight(30) # Give it a fixed height
        self.status_label.setStyleSheet("font-size: 12pt; padding: 5px; background-color: rgba(30, 30, 30, 150); color: white;")
        main_layout.addWidget(self.status_label, 0)

        # Calibration progress bar
        self.calibration_progress_bar = QtWidgets.QProgressBar()
        self.calibration_progress_bar.setRange(0, 100)
        self.calibration_progress_bar.setValue(0)
        self.calibration_progress_bar.setFixedHeight(20)
        self.calibration_progress_bar.setStyleSheet("QProgressBar { text-align: center; } QProgressBar::chunk { background-color: #3498db; }")
        self.calibration_progress_bar.hide()
        main_layout.addWidget(self.calibration_progress_bar, 0)

        # Control panel
        control_panel_widget = QtWidgets.QWidget() # Use a widget for control panel styling
        control_panel_widget.setFixedHeight(50)
        control_layout = QtWidgets.QHBoxLayout(control_panel_widget)
        control_layout.setContentsMargins(10,0,10,0)

        self.btn_stop = QtWidgets.QPushButton("Stop Session")
        self.btn_stop.setStyleSheet("font-size: 11pt; padding: 8px 15px;")
        self.btn_stop.clicked.connect(self.stop_session_button_clicked)
        control_layout.addStretch(1)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch(1)
        main_layout.addWidget(control_panel_widget, 0)

        os.makedirs(os.path.join(os.getcwd(), "assets", "videos"), exist_ok=True)

        # No explicit resizeEvent needed for overlay if it's part of layout or sized to parent
        # self.stacked_widget.installEventFilter(self) # Alternative for sizing overlay

    # def eventFilter(self, obj, event):
    #     if obj == self.stacked_widget and event.type() == QtCore.QEvent.Resize:
    #         self.effect_overlay.setGeometry(self.stacked_widget.rect())
    #     return super().eventFilter(obj, event)
    
    def resizeEvent(self, event):
        """Resize overlay to match stacked_widget."""
        # The effect_overlay is a child of stacked_widget.
        # If stacked_widget uses a layout, the overlay should be part of that layout to auto-resize.
        # Or, manually resize it to fill stacked_widget.
        if hasattr(self, 'effect_overlay') and hasattr(self, 'stacked_widget'):
             self.effect_overlay.setGeometry(self.stacked_widget.rect())
        super().resizeEvent(event)
    
    def start_ui_updates(self):
        """Start the timer to keep UI responsive during intensive operations"""
        self.ui_update_timer.start()
    
    def stop_ui_updates(self):
        """Stop the UI update timer"""
        if self.ui_update_timer.isActive():
            self.ui_update_timer.stop()
    
    def set_status(self, status_text):
        """Set status text with forced UI update"""
        self.status_label.setText(status_text)
        print(f"VideoPlayer Status: {status_text}")
        # Use direct processEvents for immediate feedback
        QtWidgets.QApplication.processEvents()
    
    def show_calibration_progress(self, progress_value):
        """Update calibration progress with forced UI update"""
        if not self.calibration_progress_bar.isVisible():
            self.calibration_progress_bar.show()
        
        # Set the value without calling processEvents directly
        self.calibration_progress_bar.setValue(int(progress_value))
        
        # Only update the status occasionally to reduce UI overhead
        if progress_value % 10 == 0 or progress_value == 100:
            self.set_status(f"Calibrating EEG: {progress_value}% complete")

    def process_events(self):
        """Process pending events to keep UI responsive"""
        try:
            # Process a limited number of events to prevent getting stuck
            for _ in range(10):  # Process up to 10 events per timer tick
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 5)
        except Exception as e:
            print(f"Error in process_events: {e}")
    
    def showEvent(self, event):
        """Start UI update timer when window is shown"""
        super().showEvent(event)
        self.is_closing_initiated = False
        self.btn_stop.setEnabled(True)
        self.btn_stop.setText("Stop Session")
        self.effect_overlay.setGeometry(self.stacked_widget.rect())
        self.effect_overlay.raise_()
        self.effect_overlay.show()
        self.start_ui_updates()  # Start the UI update timer
    
    def closeEvent(self, event):
        """Stop UI update timer before closing"""
        self.stop_ui_updates()  # Stop the timer first
        self.media_player.stop()
        
        if not self.is_closing_initiated:
            self.is_closing_initiated = True
            self.session_stopped.emit()
        
        event.accept()
        super().closeEvent(event)

    def set_scene(self, scene_name):
        if scene_name == self.current_scene and not self.using_placeholder: # Force update for placeholder
            return

        self.current_scene = scene_name
        color_map_relaxation = {
            "very_tense": "rgba(180, 50, 50, 0.7)", "tense": "rgba(180, 90, 50, 0.6)",
            "less_relaxed": "rgba(180, 140, 50, 0.5)", "neutral": "rgba(100, 100, 100, 0.4)",
            "slightly_relaxed": "rgba(50, 140, 180, 0.5)", "moderately_relaxed": "rgba(50, 120, 200, 0.6)",
            "strongly_relaxed": "rgba(50, 100, 220, 0.7)", "deeply_relaxed": "rgba(70, 70, 240, 0.8)"
        }
        desc_map_relaxation = {
            "very_tense": "Stressed", "tense": "Alert", "less_relaxed": "Slightly alert",
            "neutral": "Neutral", "slightly_relaxed": "Peaceful", "moderately_relaxed": "Serene",
            "strongly_relaxed": "Deeply peaceful", "deeply_relaxed": "Transcendent"
        }
        color_map_focus = { # Example
            "very_distracted": "rgba(180, 50, 50, 0.7)", "neutral": "rgba(100, 100, 100, 0.4)",
            "deeply_focused": "rgba(20, 200, 20, 0.8)"
        }
        desc_map_focus = { # Example
             "very_distracted": "Chaotic", "neutral": "Balanced", "deeply_focused": "Pure focus"
        }

        if self.using_placeholder:
            if self.session_type == "RELAXATION":
                bg_color = color_map_relaxation.get(scene_name, "rgba(0,0,0,0.5)")
                desc = desc_map_relaxation.get(scene_name, "Unknown State")
            elif self.session_type == "FOCUS":
                bg_color = color_map_focus.get(scene_name, "rgba(0,0,0,0.5)")
                desc = desc_map_focus.get(scene_name, "Unknown State")
            else:
                bg_color = "rgba(50,50,50,0.5)"
                desc = "N/A"

            self.placeholder_widget.setStyleSheet(f"QWidget {{ background-color: {bg_color}; border-radius: 10px; }}") # Added border-radius
            self.placeholder_label.setText(f"Current State: {scene_name.replace('_', ' ').title()}\n({desc})")
        else:
            # Here you would typically load different video segments or apply shader effects
            # For now, we print. The effect_overlay handles continuous feedback.
            # print(f"Video Scene Changed (simulation): {scene_name}")
            pass


    def set_relaxation_level(self, level): # level 0.0 to 1.0
        self.current_level = level
        alpha = int(min(150, max(0, (1.0 - level) * 150))) # More relaxed = less overlay intensity
        blue_intensity = int(min(255, max(0, level * 180 + 50))) # More relaxed = more blue, ensure some base blue
        # Update overlay, ensure it's visible and raised
        self.effect_overlay.setStyleSheet(f"background-color: rgba(0, {blue_intensity}, 255, {alpha});")
        self.effect_overlay.show()
        self.effect_overlay.raise_()


    def set_focus_level(self, level): # level 0.0 to 1.0
        self.current_level = level
        alpha = int(min(150, max(0, (1.0 - level) * 150)))
        green_intensity = int(min(255, max(0, level * 180 + 50)))
        self.effect_overlay.setStyleSheet(f"background-color: rgba(0, {green_intensity}, 100, {alpha});")
        self.effect_overlay.show()
        self.effect_overlay.raise_()

    def start_relaxation_video(self):
        self.session_type = "RELAXATION"
        video_path = os.path.join(os.getcwd(), "assets", "videos", "relaxation_base.mp4")

        if os.path.exists(video_path):
            self.using_placeholder = False
            self.stacked_widget.setCurrentWidget(self.video_widget)
            media_content = QtCore.QUrl.fromLocalFile(video_path)
            self.media_player.setMedia(QtMultimedia.QMediaContent(media_content))
            self.media_player.play()
            self.set_status("Relaxation session active")
        else:
            print(f"Video file not found: {video_path}. Using placeholder.")
            self.using_placeholder = True
            self.stacked_widget.setCurrentWidget(self.placeholder_widget)
            self.set_scene("neutral") # Set initial placeholder scene
            self.set_status("Relaxation: Video not found, using visual feedback.")
        
        QtWidgets.QApplication.processEvents() # Ensure UI updates after stacked widget change

    # Removed duplicate start_relaxation_video

    def show_calibration_progress(self, progress_value):
        if not self.calibration_progress_bar.isVisible():
            self.calibration_progress_bar.show()
        self.calibration_progress_bar.setValue(int(progress_value))
        QtWidgets.QApplication.processEvents() # Keep UI responsive during updates

        # Parent handles when calibration is actually complete via on_calibration_status
        # if progress_value >= 100:
        #     QtCore.QTimer.singleShot(500, self._finish_calibration_display) # Short delay for user to see 100%

    def hide_calibration_progress_bar(self):
        """Hides the progress bar and updates status."""
        self.calibration_progress_bar.hide()
        # self.set_status("Calibration complete! Session starting...") # Parent sets this status

    # def _finish_calibration_display(self): # Renamed from _finish_calibration
    #     self.calibration_progress_bar.hide()
    #     self.set_status("Calibration complete! Session starting...") # Or parent sets this

    def stop_session_button_clicked(self):
        print("VideoPlayerWindow: Stop session button clicked.")
        if self.is_closing_initiated:
            print("VideoPlayerWindow: Closing already in progress.")
            return

        self.is_closing_initiated = True
        self.media_player.stop()
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stopping...")
        QtWidgets.QApplication.processEvents() # Update button text

        # Emit signal that parent (MeditationPageWidget) will catch to do its cleanup
        print("VideoPlayerWindow: Emitting session_stopped signal.")
        self.session_stopped.emit()
        
        # Close this window after a short delay, allowing parent to process signal
        QtCore.QTimer.singleShot(200, self.close)
