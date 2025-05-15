import os
from PyQt5 import QtWidgets, QtCore, QtMultimedia, QtMultimediaWidgets

class VideoPlayerWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Feedback Session")
        self.setGeometry(200, 200, 800, 600) # Made it a bit wider for status
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # --- Video Player Setup ---
        self.player = QtMultimedia.QMediaPlayer(self, QtMultimedia.QMediaPlayer.VideoSurface)
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.main_layout.addWidget(self.video_widget, stretch=1) # Video takes most space
        self.player.setVideoOutput(self.video_widget)

        # --- Status and Info Area ---
        info_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft)
        self.ab_ratio_label = QtWidgets.QLabel("A/B Ratio: --")
        self.ab_ratio_label.setAlignment(QtCore.Qt.AlignRight)
        info_layout.addWidget(self.status_label)
        info_layout.addStretch(1)
        info_layout.addWidget(self.ab_ratio_label)
        self.main_layout.addLayout(info_layout)

        # --- Video Paths ---
        # IMPORTANT: Update these paths to your actual video files!
        self.videos = {
            "forest": "videos/forest.mp4",
            "beach": "videos/beach.mp4",
            "waterfall": "videos/waterfall.mp4",
            "thunder": "videos/forest-thunder.mp4"
        }
        self.current_video_key = None
        self.default_calm_video_key = "forest" # Start with this one

        # --- State Variables for Feedback Logic ---
        self.current_ab_ratio = None
        self.previous_ab_ratio = None
        self.target_ab_ratio_baseline = None # This would be set after calibration from main app/backend
        self.ratio_smoothing_factor = 0.3 # For exponential moving average of ratio
        self.smoothed_ab_ratio = None

        # Thresholds for video changes (these need tuning!)
        # These are relative to the baseline A/B ratio if available, or absolute if not.
        self.STRESS_RATIO_THRESHOLD_FACTOR = 0.8 # If ratio drops to 80% of baseline (or an absolute low value)
        self.RELAX_RATIO_THRESHOLD_FACTOR = 1.1 # If ratio is 10% above baseline (or an absolute high value)
        self.NEUTRAL_LOWER_FACTOR = 0.9
        self.NEUTRAL_UPPER_FACTOR = 1.09


        # For blur effect (requires QGraphicsBlurEffect)
        self.blur_effect = QtWidgets.QGraphicsBlurEffect(self)
        self.blur_effect.setBlurRadius(0) # Initially no blur
        self.video_widget.setGraphicsEffect(self.blur_effect) # Apply effect to video widget

        self.load_video(self.default_calm_video_key) # Load initial video

    def load_video(self, video_key):
        if video_key in self.videos and os.path.exists(self.videos[video_key]):
            if self.current_video_key != video_key:
                print(f"VideoPlayer: Loading video '{video_key}' ({self.videos[video_key]})")
                self.player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.videos[video_key])))
                self.player.play()
                self.current_video_key = video_key
            elif self.player.state() != QtMultimedia.QMediaPlayer.PlayingState:
                self.player.play() # Resume if same video and was paused/stopped
        else:
            print(f"VideoPlayer: Video key '{video_key}' not found or path invalid: {self.videos.get(video_key)}")
            self.status_label.setText(f"Error: Video '{video_key}' not found.")

    def set_status(self, message):
        self.status_label.setText(message)

    def set_ab_ratio_baseline(self, baseline_ratio):
        """Called by parent widget after calibration with the baseline A/B ratio."""
        self.target_ab_ratio_baseline = baseline_ratio
        self.smoothed_ab_ratio = baseline_ratio # Initialize smoothed ratio with baseline
        print(f"VideoPlayer: A/B Ratio Baseline set to: {baseline_ratio:.2f}")


    def update_based_on_prediction(self, prediction_data_dict):
        """
        prediction_data_dict should contain:
        "prediction_label": "Relaxed", "Not Relaxed", etc.
        "current_ab_ratio": float
        (and other metrics if needed)
        """
        prediction_label = prediction_data_dict.get("prediction_label", "Unknown")
        self.current_ab_ratio = prediction_data_dict.get("current_ab_ratio") # Raw A/B ratio from backend

        if self.current_ab_ratio is None:
            self.ab_ratio_label.setText("A/B Ratio: --")
            self.set_status(f"State: {prediction_label} (No A/B ratio)")
            return # Not enough info for nuanced feedback

        # --- Apply Smoothing to Alpha/Beta Ratio (Optional but Recommended) ---
        if self.smoothed_ab_ratio is None:
            self.smoothed_ab_ratio = self.current_ab_ratio
        else:
            self.smoothed_ab_ratio = (self.ratio_smoothing_factor * self.current_ab_ratio) + \
                                     ((1 - self.ratio_smoothing_factor) * self.smoothed_ab_ratio)
        self.ab_ratio_label.setText(f"A/B Ratio: {self.smoothed_ab_ratio:.2f}")

        # --- Determine Target Video and Effects ---
        # Use baseline if available, otherwise use absolute ratio interpretation (less ideal)
        baseline_to_compare = self.target_ab_ratio_baseline if self.target_ab_ratio_baseline is not None else 1.0 # Default baseline if not set

        # Define thresholds based on the baseline
        stress_threshold = baseline_to_compare * self.STRESS_RATIO_THRESHOLD_FACTOR
        relax_threshold = baseline_to_compare * self.RELAX_RATIO_THRESHOLD_FACTOR
        neutral_lower_bound = baseline_to_compare * self.NEUTRAL_LOWER_FACTOR
        neutral_upper_bound = baseline_to_compare * self.NEUTRAL_UPPER_FACTOR


        new_video_key = self.current_video_key # Assume no change initially
        blur_radius = 0 # Default no blur

        if self.smoothed_ab_ratio < stress_threshold:
            self.set_status(f"State: Losing Relaxation (A/B: {self.smoothed_ab_ratio:.2f})")
            new_video_key = "thunder"
            # Make blur more intense the further below the stress threshold
            # Max blur (e.g., 15) when ratio is very low (e.g., 0.5 * stress_threshold)
            # This mapping needs tuning.
            blur_intensity_factor = max(0, (stress_threshold - self.smoothed_ab_ratio) / (stress_threshold * 0.5)) # Normalize how far below
            blur_radius = min(15, blur_intensity_factor * 15)

        elif self.smoothed_ab_ratio > relax_threshold:
            self.set_status(f"State: Relaxed (A/B: {self.smoothed_ab_ratio:.2f})")
            # Choose a calm video, maybe cycle through them or pick one
            new_video_key = self.default_calm_video_key # Or "beach", "waterfall"
            blur_radius = 0 # No blur when relaxed

        elif neutral_lower_bound <= self.smoothed_ab_ratio <= neutral_upper_bound :
             self.set_status(f"State: Neutral (A/B: {self.smoothed_ab_ratio:.2f})")
             new_video_key = self.default_calm_video_key # Or another neutral video
             # Slight blur if near the lower end of neutral
             if self.smoothed_ab_ratio < (neutral_lower_bound + (neutral_upper_bound - neutral_lower_bound) * 0.33): # Lower third of neutral
                 blur_radius = 3
             else:
                 blur_radius = 0
        else: # In between stress and neutral, or neutral and relax
            self.set_status(f"State: Transitioning (A/B: {self.smoothed_ab_ratio:.2f})")
            if self.smoothed_ab_ratio < neutral_lower_bound : # Between stress and neutral
                new_video_key = self.default_calm_video_key # Keep calm video but with blur
                blur_intensity_factor = max(0, (neutral_lower_bound - self.smoothed_ab_ratio) / (neutral_lower_bound - stress_threshold + 1e-6))
                blur_radius = min(10, 2 + blur_intensity_factor * 8) # Blur between 2 and 10
            else: # Between neutral and relaxed (usually okay with calm video, no blur)
                 new_video_key = self.default_calm_video_key
                 blur_radius = 0


        # --- Apply Changes ---
        self.load_video(new_video_key) # Loads and plays if different or stopped
        self.blur_effect.setBlurRadius(blur_radius)

        self.previous_ab_ratio = self.current_ab_ratio # Store for next comparison if needed


    def closeEvent(self, event):
        if self.player.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.player.stop()
        print("VideoPlayerWindow closing.")
        # Inform parent to stop session if this window closing means session end
        if self.parent() and hasattr(self.parent(), 'stop_video_session_from_player_close'):
             self.parent().stop_video_session_from_player_close()
        event.accept()