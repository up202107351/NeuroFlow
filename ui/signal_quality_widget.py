#!/usr/bin/env python3
"""
Signal Quality Widget - UI for displaying EEG signal quality metrics

This widget provides real-time feedback about EEG signal quality during calibration,
including movement detection, band power validation, and electrode contact assessment.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import time
from backend.signal_quality_validator import SignalQualityMetrics

class SignalQualityWidget(QtWidgets.QWidget):
    """Widget for displaying real-time signal quality metrics"""
    
    # Signals
    recalibrate_requested = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_metrics = None
        self.last_update_time = 0
        self.setup_ui()
        
        # Update timer
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(500)  # Update every 500ms
        
    def setup_ui(self):
        """Set up the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        title_label = QtWidgets.QLabel("Signal Quality Monitor")
        title_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Overall quality indicator
        self.overall_quality_frame = QtWidgets.QFrame()
        self.overall_quality_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.overall_quality_frame.setMinimumHeight(60)
        
        overall_layout = QtWidgets.QVBoxLayout(self.overall_quality_frame)
        
        self.overall_status_label = QtWidgets.QLabel("Checking signal quality...")
        self.overall_status_label.setAlignment(Qt.AlignCenter)
        self.overall_status_label.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Medium))
        overall_layout.addWidget(self.overall_status_label)
        
        self.overall_progress_bar = QtWidgets.QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(50)
        self.overall_progress_bar.setFormat("Overall Quality: %p%")
        self.overall_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #666;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
        """)
        overall_layout.addWidget(self.overall_progress_bar)
        
        layout.addWidget(self.overall_quality_frame)
        
        # Detailed metrics
        metrics_group = QtWidgets.QGroupBox("Detailed Metrics")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)
        
        # Movement indicator
        metrics_layout.addWidget(QtWidgets.QLabel("Head Movement:"), 0, 0)
        self.movement_bar = QtWidgets.QProgressBar()
        self.movement_bar.setRange(0, 100)
        self.movement_bar.setValue(50)
        self.movement_bar.setFormat("%p%")
        self.movement_bar.setStyleSheet(self._get_progress_bar_style("#27ae60"))  # Green
        metrics_layout.addWidget(self.movement_bar, 0, 1)
        
        # Band power quality
        metrics_layout.addWidget(QtWidgets.QLabel("Signal Strength:"), 1, 0)
        self.band_power_bar = QtWidgets.QProgressBar()
        self.band_power_bar.setRange(0, 100)
        self.band_power_bar.setValue(50)
        self.band_power_bar.setFormat("%p%")
        self.band_power_bar.setStyleSheet(self._get_progress_bar_style("#3498db"))  # Blue
        metrics_layout.addWidget(self.band_power_bar, 1, 1)
        
        # Electrode contact
        metrics_layout.addWidget(QtWidgets.QLabel("Electrode Contact:"), 2, 0)
        self.contact_bar = QtWidgets.QProgressBar()
        self.contact_bar.setRange(0, 100)
        self.contact_bar.setValue(50)
        self.contact_bar.setFormat("%p%")
        self.contact_bar.setStyleSheet(self._get_progress_bar_style("#9b59b6"))  # Purple
        metrics_layout.addWidget(self.contact_bar, 2, 1)
        
        layout.addWidget(metrics_group)
        
        # Recommendations panel
        self.recommendations_group = QtWidgets.QGroupBox("Recommendations")
        recommendations_layout = QtWidgets.QVBoxLayout(self.recommendations_group)
        
        self.recommendations_label = QtWidgets.QLabel("Signal quality assessment in progress...")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.setMinimumHeight(60)
        self.recommendations_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        recommendations_layout.addWidget(self.recommendations_label)
        
        layout.addWidget(self.recommendations_group)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.recalibrate_button = QtWidgets.QPushButton("Recalibrate")
        self.recalibrate_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.recalibrate_button.clicked.connect(self.recalibrate_requested.emit)
        self.recalibrate_button.hide()  # Initially hidden
        button_layout.addWidget(self.recalibrate_button)
        
        button_layout.addStretch()
        
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        button_layout.addWidget(self.help_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
    def _get_progress_bar_style(self, color):
        """Get progress bar stylesheet with specified color"""
        return f"""
            QProgressBar {{
                border: 1px solid #666;
                border-radius: 3px;
                text-align: center;
                height: 18px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """
    
    #!/usr/bin/env python3
"""
Signal Quality Widget - UI for displaying EEG signal quality metrics

This widget provides real-time feedback about EEG signal quality during calibration,
including movement detection, band power validation, and electrode contact assessment.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import time
from backend.signal_quality_validator import SignalQualityMetrics

class SignalQualityWidget(QtWidgets.QWidget):
    """Widget for displaying real-time signal quality metrics"""
    
    # Signals
    recalibrate_requested = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_metrics = None
        self.last_update_time = 0
        self.setup_ui()
        
        # Update timer
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(500)  # Update every 500ms
        
    def setup_ui(self):
        """Set up the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        title_label = QtWidgets.QLabel("Signal Quality Monitor")
        title_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Overall quality indicator
        self.overall_quality_frame = QtWidgets.QFrame()
        self.overall_quality_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.overall_quality_frame.setMinimumHeight(60)
        
        overall_layout = QtWidgets.QVBoxLayout(self.overall_quality_frame)
        
        self.overall_status_label = QtWidgets.QLabel("Checking signal quality...")
        self.overall_status_label.setAlignment(Qt.AlignCenter)
        self.overall_status_label.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Medium))
        overall_layout.addWidget(self.overall_status_label)
        
        self.overall_progress_bar = QtWidgets.QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(50)
        self.overall_progress_bar.setFormat("Overall Quality: %p%")
        self.overall_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #666;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
        """)
        overall_layout.addWidget(self.overall_progress_bar)
        
        layout.addWidget(self.overall_quality_frame)
        
        # Detailed metrics
        metrics_group = QtWidgets.QGroupBox("Detailed Metrics")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)
        
        # Movement indicator
        metrics_layout.addWidget(QtWidgets.QLabel("Head Movement:"), 0, 0)
        self.movement_bar = QtWidgets.QProgressBar()
        self.movement_bar.setRange(0, 100)
        self.movement_bar.setValue(50)
        self.movement_bar.setFormat("%p%")
        self.movement_bar.setStyleSheet(self._get_progress_bar_style("#27ae60"))  # Green
        metrics_layout.addWidget(self.movement_bar, 0, 1)
        
        # Band power quality
        metrics_layout.addWidget(QtWidgets.QLabel("Signal Strength:"), 1, 0)
        self.band_power_bar = QtWidgets.QProgressBar()
        self.band_power_bar.setRange(0, 100)
        self.band_power_bar.setValue(50)
        self.band_power_bar.setFormat("%p%")
        self.band_power_bar.setStyleSheet(self._get_progress_bar_style("#3498db"))  # Blue
        metrics_layout.addWidget(self.band_power_bar, 1, 1)
        
        # Electrode contact
        metrics_layout.addWidget(QtWidgets.QLabel("Electrode Contact:"), 2, 0)
        self.contact_bar = QtWidgets.QProgressBar()
        self.contact_bar.setRange(0, 100)
        self.contact_bar.setValue(50)
        self.contact_bar.setFormat("%p%")
        self.contact_bar.setStyleSheet(self._get_progress_bar_style("#9b59b6"))  # Purple
        metrics_layout.addWidget(self.contact_bar, 2, 1)
        
        layout.addWidget(metrics_group)
        
        # Recommendations panel
        self.recommendations_group = QtWidgets.QGroupBox("Recommendations")
        recommendations_layout = QtWidgets.QVBoxLayout(self.recommendations_group)
        
        self.recommendations_label = QtWidgets.QLabel("Signal quality assessment in progress...")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.setMinimumHeight(60)
        self.recommendations_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        recommendations_layout.addWidget(self.recommendations_label)
        
        layout.addWidget(self.recommendations_group)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.recalibrate_button = QtWidgets.QPushButton("Recalibrate")
        self.recalibrate_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.recalibrate_button.clicked.connect(self.recalibrate_requested.emit)
        self.recalibrate_button.hide()  # Initially hidden
        button_layout.addWidget(self.recalibrate_button)
        
        button_layout.addStretch()
        
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        button_layout.addWidget(self.help_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
    def _get_progress_bar_style(self, color):
        """Get progress bar stylesheet with specified color"""
        return f"""
            QProgressBar {{
                border: 1px solid #666;
                border-radius: 3px;
                text-align: center;
                height: 18px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """
    
    def update_metrics(self, metrics: SignalQualityMetrics):
        """Update the display with new signal quality metrics"""
        self.current_metrics = metrics
        self.last_update_time = time.time()
        
    def update_display(self):
        """Update the visual display based on current metrics"""
        if not self.current_metrics:
            return
        
        metrics = self.current_metrics
        
        # Update overall quality
        overall_percent = int(metrics.overall_score * 100)
        self.overall_progress_bar.setValue(overall_percent)
        
        # Update progress bar color based on quality
        if metrics.overall_score >= 0.8:
            color = "#27ae60"  # Green - excellent
            self.overall_status_label.setText("Excellent signal quality!")
        elif metrics.overall_score >= 0.6:
            color = "#f39c12"  # Orange - good
            self.overall_status_label.setText("Good signal quality")
        elif metrics.overall_score >= 0.4:
            color = "#e67e22"  # Dark orange - fair
            self.overall_status_label.setText("Fair signal quality")
        else:
            color = "#e74c3c"  # Red - poor
            self.overall_status_label.setText("Poor signal quality")
        
        self.overall_progress_bar.setStyleSheet(self._get_progress_bar_style(color))
        
        # Update detailed metrics
        self.movement_bar.setValue(int(metrics.movement_score * 100))
        self.band_power_bar.setValue(int(metrics.band_power_score * 100))
        self.contact_bar.setValue(int(metrics.electrode_contact_score * 100))
        
        # Update movement bar color
        if metrics.movement_score >= 0.8:
            movement_color = "#27ae60"  # Green
        elif metrics.movement_score >= 0.6:
            movement_color = "#f39c12"  # Orange
        else:
            movement_color = "#e74c3c"  # Red
        self.movement_bar.setStyleSheet(self._get_progress_bar_style(movement_color))
        
        # Update band power bar color
        if metrics.band_power_score >= 0.8:
            band_color = "#27ae60"  # Green
        elif metrics.band_power_score >= 0.6:
            band_color = "#f39c12"  # Orange
        else:
            band_color = "#e74c3c"  # Red
        self.band_power_bar.setStyleSheet(self._get_progress_bar_style(band_color))
        
        # Update contact bar color
        if metrics.electrode_contact_score >= 0.8:
            contact_color = "#27ae60"  # Green
        elif metrics.electrode_contact_score >= 0.6:
            contact_color = "#f39c12"  # Orange
        else:
            contact_color = "#e74c3c"  # Red
        self.contact_bar.setStyleSheet(self._get_progress_bar_style(contact_color))
        
        # Update recommendations
        if metrics.recommendations:
            recommendations_text = "\n".join([f"• {rec}" for rec in metrics.recommendations])
        else:
            recommendations_text = "No specific recommendations at this time."
        
        self.recommendations_label.setText(recommendations_text)
        
        # Show/hide recalibrate button based on quality
        if metrics.overall_score < 0.4:
            self.recalibrate_button.show()
        else:
            self.recalibrate_button.hide()
        
        # Update frame color based on quality
        if metrics.overall_score >= 0.6:
            frame_color = "#d5e8d4"  # Light green
        elif metrics.overall_score >= 0.4:
            frame_color = "#fff2cc"  # Light yellow
        else:
            frame_color = "#f8cecc"  # Light red
        
        self.overall_quality_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {frame_color};
                border: 1px solid #ccc;
                border-radius: 5px;
            }}
        """)

    def update_metrics(self, metrics_dict):
        """Update metrics from backend data"""
        try:
            # Convert dict to metrics object format your widget expects
            from backend.signal_quality_validator import SignalQualityMetrics
            
            # Create metrics object from dict
            metrics = SignalQualityMetrics(
                movement_score=metrics_dict.get('movement_score', 0.5),
                band_power_score=metrics_dict.get('band_power_score', 0.5),
                electrode_contact_score=metrics_dict.get('electrode_contact_score', 0.5),
                overall_score=metrics_dict.get('overall_score', 0.5),
                quality_level=metrics_dict.get('quality_level', 'Unknown'),
                recommendations=metrics_dict.get('recommendations', [])
            )
            
            # Update the widget with new metrics
            self.update_quality_display(metrics)
            
        except Exception as e:
            print(f"Error updating signal quality widget: {e}")
            # Fallback to direct update if metrics object creation fails
            self._update_from_dict(metrics_dict)

    def _update_from_dict(self, metrics_dict):
        """Direct update from dictionary (fallback method)"""
        # Update UI elements directly from the dictionary
        overall_score = metrics_dict.get('overall_score', 0.5)
        quality_level = metrics_dict.get('quality_level', 'Unknown')
        
        # Update your widget's displays
        if hasattr(self, 'overall_score_label'):
            self.overall_score_label.setText(f"Overall: {overall_score:.1%}")
        
        if hasattr(self, 'quality_level_label'):
            self.quality_level_label.setText(f"Quality: {quality_level}")
        
        # Update other UI elements as needed
        self._update_recommendation_display(metrics_dict.get('recommendations', []))
    
    def show_help(self):
        """Show help dialog with signal quality information"""
        help_text = """
        <h3>Signal Quality Monitor Help</h3>
        
        <p><b>Overall Quality:</b> Combined score of all signal quality metrics.</p>
        
        <p><b>Head Movement:</b> Measures head movement using the accelerometer.
        Higher scores indicate less movement, which is better for EEG recording.</p>
        
        <p><b>Signal Strength:</b> Evaluates EEG band power levels to ensure they're
        within normal ranges. Low scores may indicate poor electrode contact.</p>
        
        <p><b>Electrode Contact:</b> Assesses the stability of the EEG signal to
        detect loose or disconnected electrodes.</p>
        
        <h4>Quality Levels:</h4>
        <ul>
        <li><b>Excellent (80-100%):</b> Optimal signal quality</li>
        <li><b>Good (60-79%):</b> Acceptable for most applications</li>
        <li><b>Fair (40-59%):</b> May affect accuracy</li>
        <li><b>Poor (0-39%):</b> Adjustment needed</li>
        </ul>
        
        <h4>Improving Signal Quality:</h4>
        <ul>
        <li>Ensure the headband is snug but comfortable</li>
        <li>Clean the electrode contacts if needed</li>
        <li>Sit still and relaxed</li>
        <li>Ensure good skin contact with all electrodes</li>
        </ul>
        """
        
        QtWidgets.QMessageBox.information(self, "Signal Quality Help", help_text)
    
    def reset(self):
        """Reset the widget to initial state"""
        self.current_metrics = None
        self.overall_progress_bar.setValue(50)
        self.movement_bar.setValue(50)
        self.band_power_bar.setValue(50)
        self.contact_bar.setValue(50)
        self.overall_status_label.setText("Checking signal quality...")
        self.recommendations_label.setText("Signal quality assessment in progress...")
        self.recalibrate_button.hide()
        
        # Reset frame color
        self.overall_quality_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        
    def update_display(self):
        """Update the visual display based on current metrics"""
        if not self.current_metrics:
            return
        
        metrics = self.current_metrics
        
        # Update overall quality
        overall_percent = int(metrics.overall_score * 100)
        self.overall_progress_bar.setValue(overall_percent)
        
        # Update progress bar color based on quality
        if metrics.overall_score >= 0.8:
            color = "#27ae60"  # Green - excellent
            self.overall_status_label.setText("Excellent signal quality!")
        elif metrics.overall_score >= 0.6:
            color = "#f39c12"  # Orange - good
            self.overall_status_label.setText("Good signal quality")
        elif metrics.overall_score >= 0.4:
            color = "#e67e22"  # Dark orange - fair
            self.overall_status_label.setText("Fair signal quality")
        else:
            color = "#e74c3c"  # Red - poor
            self.overall_status_label.setText("Poor signal quality")
        
        self.overall_progress_bar.setStyleSheet(self._get_progress_bar_style(color))
        
        # Update detailed metrics
        self.movement_bar.setValue(int(metrics.movement_score * 100))
        self.band_power_bar.setValue(int(metrics.band_power_score * 100))
        self.contact_bar.setValue(int(metrics.electrode_contact_score * 100))
        
        # Update movement bar color
        if metrics.movement_score >= 0.8:
            movement_color = "#27ae60"  # Green
        elif metrics.movement_score >= 0.6:
            movement_color = "#f39c12"  # Orange
        else:
            movement_color = "#e74c3c"  # Red
        self.movement_bar.setStyleSheet(self._get_progress_bar_style(movement_color))
        
        # Update band power bar color
        if metrics.band_power_score >= 0.8:
            band_color = "#27ae60"  # Green
        elif metrics.band_power_score >= 0.6:
            band_color = "#f39c12"  # Orange
        else:
            band_color = "#e74c3c"  # Red
        self.band_power_bar.setStyleSheet(self._get_progress_bar_style(band_color))
        
        # Update contact bar color
        if metrics.electrode_contact_score >= 0.8:
            contact_color = "#27ae60"  # Green
        elif metrics.electrode_contact_score >= 0.6:
            contact_color = "#f39c12"  # Orange
        else:
            contact_color = "#e74c3c"  # Red
        self.contact_bar.setStyleSheet(self._get_progress_bar_style(contact_color))
        
        # Update recommendations
        if metrics.recommendations:
            recommendations_text = "\n".join([f"• {rec}" for rec in metrics.recommendations])
        else:
            recommendations_text = "No specific recommendations at this time."
        
        self.recommendations_label.setText(recommendations_text)
        
        # Show/hide recalibrate button based on quality
        if metrics.overall_score < 0.4:
            self.recalibrate_button.show()
        else:
            self.recalibrate_button.hide()
        
        # Update frame color based on quality
        if metrics.overall_score >= 0.6:
            frame_color = "#d5e8d4"  # Light green
        elif metrics.overall_score >= 0.4:
            frame_color = "#fff2cc"  # Light yellow
        else:
            frame_color = "#f8cecc"  # Light red
        
        self.overall_quality_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {frame_color};
                border: 1px solid #ccc;
                border-radius: 5px;
            }}
        """)

    
    def show_help(self):
        """Show help dialog with signal quality information"""
        help_text = """
        <h3>Signal Quality Monitor Help</h3>
        
        <p><b>Overall Quality:</b> Combined score of all signal quality metrics.</p>
        
        <p><b>Head Movement:</b> Measures head movement using the accelerometer.
        Higher scores indicate less movement, which is better for EEG recording.</p>
        
        <p><b>Signal Strength:</b> Evaluates EEG band power levels to ensure they're
        within normal ranges. Low scores may indicate poor electrode contact.</p>
        
        <p><b>Electrode Contact:</b> Assesses the stability of the EEG signal to
        detect loose or disconnected electrodes.</p>
        
        <h4>Quality Levels:</h4>
        <ul>
        <li><b>Excellent (80-100%):</b> Optimal signal quality</li>
        <li><b>Good (60-79%):</b> Acceptable for most applications</li>
        <li><b>Fair (40-59%):</b> May affect accuracy</li>
        <li><b>Poor (0-39%):</b> Adjustment needed</li>
        </ul>
        
        <h4>Improving Signal Quality:</h4>
        <ul>
        <li>Ensure the headband is snug but comfortable</li>
        <li>Clean the electrode contacts if needed</li>
        <li>Sit still and relaxed</li>
        <li>Ensure good skin contact with all electrodes</li>
        </ul>
        """
        
        QtWidgets.QMessageBox.information(self, "Signal Quality Help", help_text)
    
    def reset(self):
        """Reset the widget to initial state"""
        self.current_metrics = None
        self.overall_progress_bar.setValue(50)
        self.movement_bar.setValue(50)
        self.band_power_bar.setValue(50)
        self.contact_bar.setValue(50)
        self.overall_status_label.setText("Checking signal quality...")
        self.recommendations_label.setText("Signal quality assessment in progress...")
        self.recalibrate_button.hide()
        
        # Reset frame color
        self.overall_quality_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)