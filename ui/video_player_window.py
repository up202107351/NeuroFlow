from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
import os
import time

class VideoPlayerWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_scene = "neutral"
        self.current_level = 0.5  # 0.0 to 1.0
        self.session_type = None  # "RELAXATION" or "FOCUS"
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Neurofeedback Session")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Video player
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.media_player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14pt; padding: 5px; background-color: rgba(0, 0, 0, 100); color: white;")
        
        # Calibration progress bar
        self.calibration_progress_bar = QtWidgets.QProgressBar()
        self.calibration_progress_bar.setRange(0, 100)
        self.calibration_progress_bar.setValue(0)
        self.calibration_progress_bar.setStyleSheet("QProgressBar { text-align: center; }")
        self.calibration_progress_bar.hide()
        
        # Control panel
        control_layout = QtWidgets.QHBoxLayout()
        
        self.btn_stop = QtWidgets.QPushButton("Stop Session")
        self.btn_stop.clicked.connect(self.stop_session)
        
        control_layout.addWidget(self.btn_stop)
        
        # Add widgets to main layout
        main_layout.addWidget(self.video_widget)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.calibration_progress_bar)
        main_layout.addLayout(control_layout)
        
        # Create a placeholder for visual effects
        self.effect_overlay = QtWidgets.QLabel(self.video_widget)
        self.effect_overlay.setGeometry(0, 0, 800, 600)
        self.effect_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.effect_overlay.lower()
        
    def set_status(self, status_text):
        """Set the status text"""
        self.status_label.setText(status_text)
        
    def set_scene(self, scene_name):
        """Set the current scene based on mental state"""
        if scene_name == self.current_scene:
            return  # No change needed
            
        self.current_scene = scene_name
        
        # In a real app, you'd transition to different video segments
        # Here we'll just simulate with descriptions
        
        if self.session_type == "RELAXATION":
            scene_descriptions = {
                "very_tense": "Stressed scene with fast movement and high contrast",
                "tense": "Alert scene with moderate movement",
                "less_relaxed": "Slightly alert scene with gentle movement",
                "neutral": "Neutral calming scene",
                "slightly_relaxed": "Peaceful scene with gentle elements",
                "moderately_relaxed": "Serene natural scene with calming elements",
                "strongly_relaxed": "Deeply peaceful scene with slow-moving elements",
                "deeply_relaxed": "Transcendent scene with ethereal qualities"
            }
            
            # This would actually load a video segment in a real app
            # self.load_video_for_scene(scene_name)
            print(f"Changing relaxation video scene to: {scene_name} - {scene_descriptions.get(scene_name, '')}")
            
        elif self.session_type == "FOCUS":
            scene_descriptions = {
                "very_distracted": "Chaotic scene with distracting elements",
                "distracted": "Busy scene with multiple elements",
                "less_focused": "Scene with mild distractions",
                "neutral": "Balanced scene with moderate detail",
                "slightly_focused": "Clear scene with defined elements",
                "moderately_focused": "Precise scene with strong central focus",
                "strongly_focused": "Intensely focused scene with minimal distraction",
                "deeply_focused": "Pure focus scene with perfect clarity"
            }
            
            # This would actually load a video segment in a real app
            # self.load_video_for_scene(scene_name)
            print(f"Changing focus video scene to: {scene_name} - {scene_descriptions.get(scene_name, '')}")
    
    def set_relaxation_level(self, level):
        """Fine-tune video parameters based on relaxation level (0.0 to 1.0)"""
        self.current_level = level
        
        # In a real app, you'd adjust video effects based on the level
        # Examples:
        # - Adjust brightness/contrast
        # - Control particle effects
        # - Change audio volume or frequency
        
        # Apply a visual effect overlay as a simple example
        alpha = int(min(200, max(0, (1.0 - level) * 150)))  # More relaxed = less overlay
        blue_intensity = int(min(255, max(0, level * 255)))  # More relaxed = more blue
        
        # Create a blue gradient overlay with transparency based on relaxation level
        self.effect_overlay.setStyleSheet(
            f"background-color: rgba(0, {blue_intensity}, 255, {alpha});"
        )
        
    def set_focus_level(self, level):
        """Fine-tune video parameters based on focus level (0.0 to 1.0)"""
        self.current_level = level
        
        # Similar to relaxation, but with different visual effects
        alpha = int(min(200, max(0, (1.0 - level) * 150)))  # More focused = less overlay
        green_intensity = int(min(255, max(0, level * 255)))  # More focused = more green
        
        # Create a green gradient overlay with transparency based on focus level
        self.effect_overlay.setStyleSheet(
            f"background-color: rgba(0, {green_intensity}, 100, {alpha});"
        )
    
    def start_relaxation_video(self):
        """Start the relaxation video"""
        self.session_type = "RELAXATION"
        
        # Path to your relaxation video
        video_path = os.path.join(os.getcwd(), "assets", "videos", "relaxation_base.mp4")
        
        if os.path.exists(video_path):
            media_content = QtCore.QUrl.fromLocalFile(video_path)
            self.media_player.setMedia(QtMultimedia.QMediaContent(media_content))
            self.media_player.play()
            self.set_status("Relaxation session started")
        else:
            self.set_status(f"Video not found: {video_path}")
            print(f"Error: Video file not found at {video_path}")
    
    def start_focus_video(self):
        """Start the focus video"""
        self.session_type = "FOCUS"
        
        # Path to your focus video
        video_path = os.path.join(os.getcwd(), "assets", "videos", "focus_base.mp4")
        
        if os.path.exists(video_path):
            media_content = QtCore.QUrl.fromLocalFile(video_path)
            self.media_player.setMedia(QtMultimedia.QMediaContent(media_content))
            self.media_player.play()
            self.set_status("Focus session started")
        else:
            self.set_status(f"Video not found: {video_path}")
            print(f"Error: Video file not found at {video_path}")
    
    def stop_session(self):
        """Stop the current session"""
        self.media_player.stop()
        
        if self.parent:
            self.parent.stop_video_session()
        
        self.close()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.media_player.stop()
        event.accept()