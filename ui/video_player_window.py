from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import time
import cv2  # OpenCV for video playback
from ui.signal_quality_widget import SignalQualityWidget

class RelaxationCircle(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0.0
        self.setFixedSize(100, 100)
        self.setStyleSheet("background-color: transparent;")
        self.animation = QtCore.QPropertyAnimation(self, b"level_anim")
        self.animation.setDuration(500)  # Half-second animation
        self.animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        
    def setLevel(self, value, level=None):
        """Set level with animation"""
        self.animation.stop()
        self.animation.setStartValue(self.level)
        self.animation.setEndValue(value)
        self.animation.start()

        if level is not None:
            self.current_level_hint = level
        
    def get_level_anim(self):
        return self.level
        
    def set_level_anim(self, value):
        self.level = value
        self.update()
        
    level_anim = QtCore.pyqtProperty(float, get_level_anim, set_level_anim)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Use level hint for more accurate colors if available
        if hasattr(self, 'current_level_hint'):
            color = self._get_color_for_level(self.current_level_hint)
        else:
            # Fallback to your existing logic
            color = self._get_color_for_value(self.level)
        
        # Draw background circle (gray)
        pen = QtGui.QPen(QtCore.Qt.transparent)
        painter.setPen(pen)
        
        # Background circle
        background_brush = QtGui.QBrush(QtGui.QColor(70, 70, 70, 180))
        painter.setBrush(background_brush)
        painter.drawEllipse(5, 5, 90, 90)
            
        # Fill circle based on relaxation level
        filled_brush = QtGui.QBrush(color)
        painter.setBrush(filled_brush)
        
        # Use angle to draw an arc - 0 degrees is at 3 o'clock, move clockwise
        # -90 degrees starts at 12 o'clock position
        span_angle = int(-360 * self.level)
        painter.drawPie(5, 5, 90, 90, -90 * 16, span_angle * 16)
        
        # Draw center circle with level text
        center_brush = QtGui.QBrush(QtGui.QColor(40, 40, 40, 220))
        painter.setBrush(center_brush)
        painter.drawEllipse(30, 30, 40, 40)
        
        # Draw text percentage
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        painter.drawText(QtCore.QRect(30, 30, 40, 40), 
                         QtCore.Qt.AlignCenter, 
                         f"{int(self.level * 100)}%")

    def _get_color_for_level(self, level):
        """More precise colors based on discrete levels"""
        level_colors = {
            -3: QtGui.QColor(200, 50, 50, 200),   # Deep red
            -2: QtGui.QColor(180, 80, 50, 200),   # Red-orange  
            -1: QtGui.QColor(180, 140, 50, 200),  # Orange
            0:  QtGui.QColor(100, 150, 200, 200), # Blue (neutral)
            1:  QtGui.QColor(120, 180, 100, 200), # Light green
            2:  QtGui.QColor(80, 180, 120, 200),  # Green
            3:  QtGui.QColor(50, 180, 140, 200),  # Deep green
            4:  QtGui.QColor(50, 200, 160, 200)   # Brilliant green
        }
        return level_colors.get(level, QtGui.QColor(180, 180, 50, 200))
    
    def _get_color_for_value(self, value):
        """Fallback color method based on continuous value (0.0-1.0)"""
        if self.session_type == "FOCUS":
            # Focus colors: red (distracted) -> blue (focused)
            if value < 0.3:
                # Distracted: red to orange
                red = 220
                green = int(20 + (value / 0.3) * 49)  # 20 to 69
                blue = 60
            elif value < 0.6:
                # Neutral: orange to blue
                progress = (value - 0.3) / 0.3
                red = int(255 - progress * 155)  # 255 to 100
                green = int(69 + progress * 80)   # 69 to 149
                blue = int(0 + progress * 237)    # 0 to 237
            else:
                # Focused: blue shades
                progress = (value - 0.6) / 0.4
                red = int(100 - progress * 100)  # 100 to 0
                green = int(149 - progress * 149) # 149 to 0
                blue = int(237 - progress * 98)   # 237 to 139
        else:
            # Relaxation colors: red (tense) -> green (relaxed)
            if value < 0.3:
                # Tense: red shades
                red = int(200 - value * 20)  # 200 to 180
                green = int(50 + value * 90)  # 50 to 140
                blue = 50
            elif value < 0.6:
                # Neutral: orange to light green
                progress = (value - 0.3) / 0.3
                red = int(180 - progress * 80)  # 180 to 100
                green = int(140 + progress * 10) # 140 to 150
                blue = int(50 + progress * 150)  # 50 to 200
            else:
                # Relaxed: green shades
                progress = (value - 0.6) / 0.4
                red = int(100 - progress * 50)   # 100 to 50
                green = int(150 + progress * 50) # 150 to 200
                blue = int(200 - progress * 40)  # 200 to 160
        
        return QtGui.QColor(red, green, blue, 200)
    
class VideoPlayerWindow(QtWidgets.QMainWindow):
    session_stopped = QtCore.pyqtSignal()
    recalibration_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_scene = "neutral"
        self.current_level = 0.5
        self.session_type = None
        self.is_closing_initiated = False
        self.using_placeholder = False
        self.current_video_file = None
        
        # Video paths - relative to assets/videos
        self.video_files = {
            "low": "forest.mp4",
            "medium": "beach.mp4",
            "high": "waterfall.mp4",
            "thunder": "forest-thunder.mp4"
        }
        
        # OpenCV video variables
        self.cap = None              # Current video capture
        self.next_cap = None         # Next video capture (for transitions)
        self.target_video = None     # Target video for transition
        self.transition_alpha = 0.0  # Transition progress (0.0 to 1.0)
        self.in_transition = False   # Whether we're currently transitioning
        self.frame_rate = 30
        self.playback_rate = 1.0
        self.video_timer = None
        self.blur_amount = 0.0       # Current blur amount
        
        # Video processing parameters
        self.brightness = 1.0        # Brightness multiplier (1.0 is normal)
        self.saturation = 1.0        # Saturation multiplier (1.0 is normal)
        
        # Smooth calibration progress variables
        self.smooth_calibration_value = 0
        self.target_calibration_value = 0
        self.calibration_start_time = None
        
        # Signal quality widget reference
        self.signal_quality_widget = None
        
        self.initUI()
        
        # Add a timer for smoother UI updates during calibration
        self.ui_update_timer = QtCore.QTimer(self)
        self.ui_update_timer.timeout.connect(self.process_events)
        self.ui_update_timer.setInterval(100)  # Update every 100ms
        
        # Add timer for smooth calibration progress
        self.calibration_timer = QtCore.QTimer(self)
        self.calibration_timer.timeout.connect(self.update_smooth_calibration)
        self.calibration_timer.setInterval(50)  # Update frequently for smooth animation
        
        # Add timer for video transitions and effects
        self.video_effect_timer = QtCore.QTimer(self)
        self.video_effect_timer.timeout.connect(self.update_video_effects)
        self.video_effect_timer.setInterval(200)  # 5 times per second

    def initUI(self):
        self.setWindowTitle("Neurofeedback Session")
        self.setMinimumSize(1200, 700)  # Increased size to accommodate signal quality widget

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for immersive feel

        # Create main horizontal layout for video + signal quality panel
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Create main container for video
        self.main_container = QtWidgets.QWidget()
        self.main_container_layout = QtWidgets.QVBoxLayout(self.main_container)
        self.main_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add stacked widget for videos
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.stacked_widget.setStyleSheet("background-color: black;")
        self.main_container_layout.addWidget(self.stacked_widget, 1)
        
        # OpenCV video display widget
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 360)
        self.stacked_widget.addWidget(self.video_label)

        # Placeholder widget (fallback for videos)
        self.placeholder_widget = QtWidgets.QWidget()
        placeholder_layout = QtWidgets.QVBoxLayout(self.placeholder_widget)
        placeholder_layout.setAlignment(QtCore.Qt.AlignCenter)
        
        self.placeholder_label = QtWidgets.QLabel("Nature Visualization")
        self.placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 16pt; color: white;")
        placeholder_layout.addWidget(self.placeholder_label)
        
        # Add an image to the placeholder
        self.placeholder_image = QtWidgets.QLabel()
        self.placeholder_image.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder_image.setMinimumSize(640, 360)
        placeholder_layout.addWidget(self.placeholder_image)
        
        # Try to load a placeholder image
        placeholder_path = os.path.join(os.getcwd(), "assets", "relax.jpg")
        if os.path.exists(placeholder_path):
            pixmap = QtGui.QPixmap(placeholder_path)
            self.placeholder_image.setPixmap(pixmap.scaled(640, 360, QtCore.Qt.KeepAspectRatio))
        
        self.stacked_widget.addWidget(self.placeholder_widget)

        # Add relaxation circle overlay in top-right corner of video area
        self.circle_container = QtWidgets.QWidget(self.main_container)
        self.circle_container.setFixedSize(120, 120)
        self.circle_container.setStyleSheet("background-color: transparent;")
        
        circle_layout = QtWidgets.QVBoxLayout(self.circle_container)
        circle_layout.setContentsMargins(10, 10, 10, 10)
        
        self.relaxation_circle = RelaxationCircle()
        circle_layout.addWidget(self.relaxation_circle)
        circle_layout.setAlignment(QtCore.Qt.AlignCenter)

        # Add main video container to content layout (takes most space)
        content_layout.addWidget(self.main_container, 2)  # 2/3 of the space

        # Create signal quality panel (initially hidden, shown during calibration)
        self.signal_quality_panel = QtWidgets.QWidget()
        self.signal_quality_panel.setFixedWidth(350)  # Fixed width sidebar
        self.signal_quality_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 240);
                border-left: 2px solid #555;
            }
        """)
        self.signal_quality_panel.hide()  # Initially hidden
        
        # Signal quality widget will be added here later
        self.signal_quality_layout = QtWidgets.QVBoxLayout(self.signal_quality_panel)
        self.signal_quality_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add placeholder for signal quality widget
        quality_placeholder = QtWidgets.QLabel("Signal Quality Monitor\nwill appear here during calibration")
        quality_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        quality_placeholder.setStyleSheet("color: #ccc; font-style: italic;")
        self.signal_quality_layout.addWidget(quality_placeholder)
        
        # Add signal quality panel to content layout
        content_layout.addWidget(self.signal_quality_panel, 1)  # 1/3 of the space

        # Add content layout to main layout
        main_layout.addLayout(content_layout, 1)  # Takes most of the space

        # Status bar (used during calibration) - now spans full width
        self.status_bar = QtWidgets.QWidget()
        status_bar_layout = QtWidgets.QHBoxLayout(self.status_bar)
        status_bar_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QtWidgets.QLabel("Calibrating...")
        self.status_label.setStyleSheet("color: white; font-size: 12pt;")
        status_bar_layout.addWidget(self.status_label)
        
        self.calibration_progress_bar = QtWidgets.QProgressBar()
        self.calibration_progress_bar.setRange(0, 100)
        self.calibration_progress_bar.setValue(0)
        self.calibration_progress_bar.setFixedHeight(15)
        self.calibration_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 7px;
                background: rgba(40, 40, 40, 180);
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 7px;
            }
        """)
        status_bar_layout.addWidget(self.calibration_progress_bar, 1)
        
        main_layout.addWidget(self.status_bar)

        # Control panel at bottom
        control_panel = QtWidgets.QWidget()
        control_panel.setFixedHeight(50)
        control_panel.setStyleSheet("background-color: rgba(20, 20, 20, 180);")
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        
        self.btn_stop = QtWidgets.QPushButton("End Session")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                font-size: 11pt;
                padding: 8px 20px;
                background-color: #444;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_session_button_clicked)
        control_layout.addStretch(1)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch(1)
        
        main_layout.addWidget(control_panel)

        # Make sure videos directory exists
        os.makedirs(os.path.join(os.getcwd(), "assets", "videos"), exist_ok=True)

    def add_signal_quality_widget(self, signal_quality_widget):
        """Add the signal quality widget to the panel"""
        if self.signal_quality_widget:
            # Remove existing widget
            self.signal_quality_layout.removeWidget(self.signal_quality_widget)
            self.signal_quality_widget.setParent(None)
        
        # Clear the placeholder
        for i in reversed(range(self.signal_quality_layout.count())):
            child = self.signal_quality_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add the new signal quality widget
        self.signal_quality_widget = signal_quality_widget
        self.signal_quality_layout.addWidget(self.signal_quality_widget)

        signal_quality_widget.recalibrate_requested.connect(self.handle_recalibration_request)
        
        print("Signal quality widget added to video player window")


    def show_signal_quality_panel(self):
        """Show the signal quality panel during calibration"""
        if self.signal_quality_widget:
            self.signal_quality_panel.show()
            print("Signal quality panel shown")

    def hide_signal_quality_panel(self):
        """Hide the signal quality panel after calibration"""
        self.signal_quality_panel.hide()
        print("Signal quality panel hidden")

    def resizeEvent(self, event):
        """Handle resize events to reposition overlay elements"""
        super().resizeEvent(event)
        
        # Position relaxation circle in top-right corner of video area with padding
        if hasattr(self, 'circle_container') and hasattr(self, 'main_container'):
            padding = 20
            video_width = self.main_container.width()
            self.circle_container.move(
                video_width - self.circle_container.width() - padding, 
                padding
            )
    
    def start_ui_updates(self):
        """Start the timer to keep UI responsive during intensive operations"""
        self.ui_update_timer.start()
    
    def stop_ui_updates(self):
        """Stop the UI update timer"""
        if self.ui_update_timer.isActive():
            self.ui_update_timer.stop()
    
    def handle_recalibration_request(self):
        """Handle recalibration request from signal quality widget"""
        # Emit a signal back to the meditation/focus page to restart calibration
        self.recalibration_requested.emit()
            
    def update_smooth_calibration(self):
        """Update calibration progress smoothly"""
        if self.target_calibration_value <= self.smooth_calibration_value:
            return
            
        # Calculate elapsed time since start
        if not self.calibration_start_time:
            self.calibration_start_time = time.time()
            
        elapsed = time.time() - self.calibration_start_time
        total_duration = 20.0  # Assume calibration takes about 20 seconds
        
        # Create a smooth progress that moves independently of backend updates
        # but converges to actual values when they arrive
        progress = min(100, elapsed / total_duration * 100)
        
        # If backend sent a higher value, use that
        progress = max(progress, self.target_calibration_value)
        
        # Smooth transition to target
        self.smooth_calibration_value = min(progress, self.smooth_calibration_value + 0.5)
        
        # Update the UI
        self.calibration_progress_bar.setValue(int(self.smooth_calibration_value))
        
        # Update status text periodically
        if int(self.smooth_calibration_value) % 10 == 0 or int(self.smooth_calibration_value) >= 100:
            self.set_status(f"Calibrating: {int(self.smooth_calibration_value)}%")
    
    def set_status(self, status_text):
        """Set status text with forced UI update"""
        self.status_label.setText(status_text)
        print(f"VideoPlayer Status: {status_text}")
        # Use direct processEvents for immediate feedback
        QtWidgets.QApplication.processEvents()
    
    def show_calibration_progress(self, progress_value):
        """Update calibration progress with smooth animation"""
        if not self.status_bar.isVisible():
            self.status_bar.show()
            
        # Show signal quality panel during calibration
        self.show_signal_quality_panel()
            
        # Start calibration timer if not running
        if not self.calibration_timer.isActive():
            self.calibration_timer.start()
            self.calibration_start_time = time.time()
        
        # Update target value from backend (actual progress)
        self.target_calibration_value = float(progress_value)
        
        # Smooth updates handled by update_smooth_calibration timer

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
        self.btn_stop.setText("End Session")
        
        # Position circle in corner
        self.resizeEvent(None)
        
        # Start timers
        self.start_ui_updates()
        self.video_effect_timer.start()
    
    def closeEvent(self, event):
        """Stop UI update timer before closing"""
        self.stop_ui_updates()  # Stop the UI timer
        
        # Stop all timers
        if self.video_effect_timer.isActive():
            self.video_effect_timer.stop()
        if self.calibration_timer.isActive():
            self.calibration_timer.stop()
        if self.video_timer and self.video_timer.isActive():
            self.video_timer.stop()
            
        # Release OpenCV video capture
        self.stop_video()
        
        if not self.is_closing_initiated:
            self.is_closing_initiated = True
            self.session_stopped.emit()
        
        event.accept()
        super().closeEvent(event)

    # ... [All your existing video-related methods remain the same] ...
    # I'm keeping them for brevity, but they don't need changes

    def start_relaxation_video(self):
        """Start the relaxation video session"""
        self.session_type = "RELAXATION"
        
        # Hide signal quality panel once calibration is complete
        self.hide_signal_quality_panel()
        
        # Try to load video files
        video_found = False
        
        # Try each video file until one works
        for category, filename in self.video_files.items():
            video_path = os.path.join(os.getcwd(), "assets", "videos", filename)
            if os.path.exists(video_path):
                print(f"Found video file: {video_path}")
                self.load_video_file(filename)
                video_found = True
                break
                
        if not video_found:
            print("No video files found. Using placeholder.")
            self.switch_to_placeholder()
            
        # Initialize circle to 50%
        self.relaxation_circle.setLevel(0.5)
        
        # Ensure visibility of elements
        self.circle_container.raise_()  # Ensure circle is on top
        self.status_bar.hide()  # Hide status after calibration
        
        # Initialize video effect parameters
        self.blur_amount = 0.0
        self.playback_rate = 1.0
        self.brightness = 1.0
        self.saturation = 1.0
        
        # Start video effects timer
        self.video_effect_timer.start()
        
        QtWidgets.QApplication.processEvents()

    def set_focus_level(self, level, level_hint=None):
        """Update focus level visualization (same as relaxation but different messaging)"""
        self.current_level = level
        
        # Update circle visualization with animation - same mechanism
        self.relaxation_circle.setLevel(level, level_hint)
        
        # Update placeholder if using it
        if self.using_placeholder:
            self.update_placeholder_for_focus_level(level)

    def start_focus_video(self):
        """Start the focus video session with focus-specific messaging"""
        self.session_type = "FOCUS"
        
        # Hide signal quality panel once calibration is complete
        self.hide_signal_quality_panel()
        
        # Try to load video files (same as relaxation)
        video_found = False
        for category, filename in self.video_files.items():
            video_path = os.path.join(os.getcwd(), "assets", "videos", filename)
            if os.path.exists(video_path):
                print(f"Found video file: {video_path}")
                self.load_video_file(filename)
                video_found = True
                break
                
        if not video_found:
            print("No video files found. Using placeholder.")
            self.switch_to_placeholder()
            
        # Initialize circle to 50% (same as relaxation)
        self.relaxation_circle.setLevel(0.5)
        
        # Update status for focus
        self.status_bar.hide()
        self.circle_container.raise_()
        
        # Initialize video effect parameters
        self.blur_amount = 0.0
        self.playback_rate = 1.0
        self.brightness = 1.0
        self.saturation = 1.0
        
        self.video_effect_timer.start()
        QtWidgets.QApplication.processEvents()

    def update_placeholder_for_focus_level(self, level):
        """Update placeholder visualization for focus levels"""
        if not self.using_placeholder:
            return
            
        # Focus-themed colors and states
        if level < 0.3:
            bg_color = "rgba(220, 20, 60, 0.7)"  # Crimson for distracted
            state = "Distracted"
        elif level < 0.6:
            bg_color = "rgba(75, 0, 130, 0.5)"   # Indigo for neutral focus  
            state = "Focusing"
        else:
            bg_color = "rgba(25, 25, 112, 0.7)"  # MidnightBlue for deep focus
            state = "Deeply Focused"
            
        self.placeholder_widget.setStyleSheet(f"QWidget {{ background-color: {bg_color}; }}")
        self.placeholder_label.setText(f"Focus State: {state}")

    def hide_calibration_progress_bar(self):
        """Hides the progress bar and status bar"""
        if self.calibration_timer.isActive():
            self.calibration_timer.stop()
        self.status_bar.hide()
        
        # Also hide the signal quality panel when calibration is complete
        self.hide_signal_quality_panel()

    # ... [Rest of your existing methods remain exactly the same] ...
    
    def set_scene(self, scene_name):
        """Set scene based on relaxation state"""
        if scene_name == self.current_scene and not self.using_placeholder:
            return

        self.current_scene = scene_name
        
        # Map scene names to video categories
        video_mapping = {
            "very_tense": "low",
            "tense": "low",
            "less_relaxed": "low",
            "neutral": "medium",
            "slightly_relaxed": "medium",
            "moderately_relaxed": "medium",
            "strongly_relaxed": "high",
            "deeply_relaxed": "high",
            # Focus states (for focus sessions)
            "very_distracted": "low",
            "distracted": "low", 
            "less_focused": "low",
            "slightly_focused": "medium",
            "moderately_focused": "medium",
            "strongly_focused": "high",
            "deeply_focused": "high"
        }
        
        # Special case - use thunder video at random 10% chance when tense/distracted
        if scene_name in ["very_tense", "tense", "very_distracted", "distracted"] and np.random.random() < 0.3:
            video_category = "thunder"
        else:
            video_category = video_mapping.get(scene_name, "medium")
        
        if not self.using_placeholder:
            # Check if we need to switch video
            target_video = self.video_files.get(video_category)
            if target_video != self.current_video_file:
                self.start_video_transition(target_video)

    def start_video_transition(self, target_video_file):
        """Begin smooth transition to a new video"""
        if target_video_file == self.current_video_file or self.in_transition:
            return
            
        target_path = os.path.join(os.getcwd(), "assets", "videos", target_video_file)
        if not os.path.exists(target_path):
            print(f"Target video not found: {target_path}")
            return
            
        print(f"Starting transition to: {target_video_file}")
        
        # Open the target video file
        self.next_cap = cv2.VideoCapture(target_path)
        if not self.next_cap.isOpened():
            print(f"Failed to open target video: {target_path}")
            self.next_cap = None
            return
            
        # Store target info
        self.target_video = target_video_file
        self.transition_alpha = 0.0
        self.in_transition = True
        
        print("Transition started")

    def load_video_file(self, filename, transition=False):
        """Load a specific video file using OpenCV"""
        if not filename:
            return
            
        video_path = os.path.join(os.getcwd(), "assets", "videos", filename)
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            self.switch_to_placeholder()
            return
            
        print(f"Loading video: {video_path}")
        
        if not transition:
            # Stop any existing video playback
            self.stop_video()
            
        # Open the new video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            if not transition:
                self.switch_to_placeholder()
            return None
            
        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate <= 0:
            frame_rate = 30  # Default to 30fps if not detected
            
        if not transition:
            # Set as current video
            self.cap = cap
            self.frame_rate = frame_rate
            self.current_video_file = filename
            
            # Create and start the video timer
            if not self.video_timer:
                self.video_timer = QtCore.QTimer(self)
                self.video_timer.timeout.connect(self.update_frame)
                
            # Start playback
            interval = int(1000.0 / (self.frame_rate * self.playback_rate))
            self.video_timer.start(max(10, interval))  # Ensure at least 10ms interval
            
            # Show the video widget
            self.stacked_widget.setCurrentWidget(self.video_label)
            self.using_placeholder = False
            
            return self.cap
        else:
            # Return for transition use
            return cap

    def stop_video(self):
        """Stop video playback and release resources"""
        if self.video_timer and self.video_timer.isActive():
            self.video_timer.stop()
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        if self.next_cap is not None:
            self.next_cap.release()
            self.next_cap = None
            
        self.in_transition = False
        self.transition_alpha = 0.0

    def update_frame(self):
        """Update video frame using OpenCV"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        # Read current video frame
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video, loop back to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to loop video")
                self.switch_to_placeholder()
                return
        
        # Handle transition if needed
        if self.in_transition and self.next_cap and self.next_cap.isOpened():
            ret_next, next_frame = self.next_cap.read()
            
            if not ret_next:
                # Loop next video if it reached the end
                self.next_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_next, next_frame = self.next_cap.read()
                
            if ret_next:
                # Resize next frame to match current frame's dimensions
                next_frame = cv2.resize(next_frame, (frame.shape[1], frame.shape[0]))
                
                # Blend frames using transition alpha
                frame = cv2.addWeighted(frame, 1.0 - self.transition_alpha, 
                                        next_frame, self.transition_alpha, 0)
                
                # Update transition progress
                self.transition_alpha += 0.02  # Adjust for faster/slower transitions
                
                # Check if transition is complete
                if self.transition_alpha >= 1.0:
                    print("Transition complete")
                    # Swap to the new video
                    self.cap.release()
                    self.cap = self.next_cap
                    self.next_cap = None
                    self.current_video_file = self.target_video
                    self.target_video = None
                    self.in_transition = False
                    self.transition_alpha = 0.0
        
        # Apply video effects
        frame = self.apply_video_effects(frame)
        
        # Convert frame from BGR to RGB for Qt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from the frame
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Scale the image to fit the widget while maintaining aspect ratio
        pixmap = QtGui.QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                              QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        # Update the label
        self.video_label.setPixmap(pixmap)
    
    def apply_video_effects(self, frame):
        """Apply visual effects to video frame based on relaxation level"""
        if frame is None:
            return frame
            
        # Apply blur effect if enabled
        if self.blur_amount > 0:
            blur_radius = int(self.blur_amount * 15)  # Scale to reasonable blur values
            if blur_radius > 0:
                frame = cv2.GaussianBlur(frame, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Adjust brightness and saturation if needed
        if self.brightness != 1.0 or self.saturation != 1.0:
            # Convert to HSV for easier brightness/saturation adjustment
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Adjust V channel for brightness
            hsv[:,:,2] = hsv[:,:,2] * self.brightness
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            
            # Adjust S channel for saturation
            hsv[:,:,1] = hsv[:,:,1] * self.saturation
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            
            # Convert back to BGR
            adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return adjusted
            
        return frame
    
    def switch_to_placeholder(self):
        """Switch to placeholder visualization when videos fail"""
        self.stop_video()
        self.using_placeholder = True
        self.stacked_widget.setCurrentWidget(self.placeholder_widget)
        
        # Update placeholder text based on relaxation level
        self.update_placeholder_for_level(self.current_level)
        
        print("Switched to placeholder visualization")
        
    def update_placeholder_for_level(self, level):
        """Update placeholder visualization based on relaxation level"""
        if not self.using_placeholder:
            return
            
        # Update background color
        if level < 0.3:
            bg_color = "rgba(180, 50, 50, 0.7)"  # Red for low relaxation
            state = "Tense"
        elif level < 0.6:
            bg_color = "rgba(180, 180, 50, 0.5)"  # Yellow for medium
            state = "Calm"
        else:
            bg_color = "rgba(50, 180, 120, 0.7)"  # Green for high
            state = "Deeply Relaxed"
            
        self.placeholder_widget.setStyleSheet(f"QWidget {{ background-color: {bg_color}; }}")
        self.placeholder_label.setText(f"Relaxation State: {state}")

    def update_video_effects(self):
        """Update video effects based on current relaxation level"""
        if not hasattr(self, 'current_level'):
            return
            
        if self.using_placeholder:
            self.update_placeholder_for_level(self.current_level)
            return
            
        # Update playback rate based on relaxation
        target_rate = 1.0
        if self.current_level < 0.3:
            target_rate = 1.4  # Faster for low relaxation
        elif self.current_level > 0.7:
            target_rate = 0.8  # Slower for high relaxation
            
        # Update blur amount based on relaxation (more blur when tense)
        target_blur = max(0, min(1.0, (1.0 - self.current_level) * 0.7))
        
        # Update saturation based on relaxation (more saturated when relaxed)
        target_saturation = 0.8 + (self.current_level * 0.6)  # Range from 0.8 to 1.4
        
        # Smooth transitions for effects
        self.playback_rate = self.playback_rate + (target_rate - self.playback_rate) * 0.1
        self.blur_amount = self.blur_amount + (target_blur - self.blur_amount) * 0.1
        self.saturation = self.saturation + (target_saturation - self.saturation) * 0.1
        
        # Update video timer interval for playback speed
        if self.video_timer and self.video_timer.isActive() and self.frame_rate > 0:
            interval = int(1000.0 / (self.frame_rate * self.playback_rate))
            self.video_timer.setInterval(max(10, interval))

    def set_relaxation_level(self, level):
        """Update relaxation level and related effects"""
        self.current_level = level
        
        # Update circle visualization with animation
        self.relaxation_circle.setLevel(level)
        
        # Scene selection based on relaxation level
        if level < 0.2:
            scene = "very_tense"
        elif level < 0.3:
            scene = "tense"
        elif level < 0.4:
            scene = "less_relaxed"
        elif level < 0.5:
            scene = "neutral"
        elif level < 0.6:
            scene = "slightly_relaxed"
        elif level < 0.75:
            scene = "moderately_relaxed"
        elif level < 0.9:
            scene = "strongly_relaxed"
        else:
            scene = "deeply_relaxed"
            
        # Update scene if needed
        if scene != self.current_scene:
            self.set_scene(scene)
            
        # If using placeholder, update it
        if self.using_placeholder:
            self.update_placeholder_for_level(level)

    def set_focus_level(self, level):
        """Support for focus visualization"""
        self.current_level = level
        
        # Update circle visualization with animation
        self.relaxation_circle.setLevel(level)
        
        # Scene selection based on focus level (different mapping than relaxation)
        if level < 0.2:
            scene = "very_distracted"
        elif level < 0.3:
            scene = "distracted"
        elif level < 0.4:
            scene = "less_focused"
        elif level < 0.5:
            scene = "neutral"
        elif level < 0.6:
            scene = "slightly_focused"
        elif level < 0.75:
            scene = "moderately_focused"
        elif level < 0.9:
            scene = "strongly_focused"
        else:
            scene = "deeply_focused"
            
        # Update scene if needed
        if scene != self.current_scene:
            self.set_scene(scene)
            
        # If using placeholder, update it
        if self.using_placeholder:
            self.update_placeholder_for_level(level)

    def stop_session_button_clicked(self):
        """Handle stop button click"""
        print("VideoPlayerWindow: Stop session button clicked.")
        if self.is_closing_initiated:
            print("VideoPlayerWindow: Closing already in progress.")
            return

        self.is_closing_initiated = True
        self.stop_video()
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Ending Session...")
        QtWidgets.QApplication.processEvents()

        # Emit signal that parent (MeditationPageWidget) will catch to do its cleanup
        print("VideoPlayerWindow: Emitting session_stopped signal.")
        self.session_stopped.emit()
        
        # Close this window after a short delay, allowing parent to process signal
        QtCore.QTimer.singleShot(200, self.close)