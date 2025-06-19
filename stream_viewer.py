#!/usr/bin/env python3
"""
Real-time LSL EEG Viewer - Fixed Plotting Version
"""

import sys
import numpy as np
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
import time
from collections import deque
from datetime import datetime

class RealTimeEEGViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # LSL connection
        self.inlet = None
        self.is_connected = False
        
        # Data storage - use simpler approach
        self.max_points = 2000  # Maximum points to keep
        self.channel_data = [[] for _ in range(4)]  # Store as lists
        self.time_data = []
        self.sampling_rate = 256.0
        
        # Channel info
        self.n_channels = 4
        self.channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
        self.channel_colors = ["#6F72B3", "#9370DB", "#B3B6E6", "#33366B"]
        
        # Display settings
        self.display_seconds = 5.0
        self.update_interval = 100
        
        # Statistics
        self.samples_received = 0
        self.last_update_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        self.data_chunks_received = 0
        
        self.initUI()
        self.setup_animation()
        
    def initUI(self):
        self.setWindowTitle('Real-time LSL EEG Viewer - Fixed')
        self.setGeometry(100, 100, 1200, 800)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.connect_button = QtWidgets.QPushButton("Connect to LSL")
        self.connect_button.clicked.connect(self.toggle_connection)
        controls_layout.addWidget(self.connect_button)
        
        controls_layout.addWidget(QtWidgets.QLabel("Display time (s):"))
        self.time_spinbox = QtWidgets.QDoubleSpinBox()
        self.time_spinbox.setRange(1.0, 30.0)
        self.time_spinbox.setValue(self.display_seconds)
        self.time_spinbox.valueChanged.connect(self.update_display_time)
        controls_layout.addWidget(self.time_spinbox)
        
        self.status_label = QtWidgets.QLabel("Not connected")
        self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        self.stats_label = QtWidgets.QLabel("Samples: 0 | FPS: 0")
        controls_layout.addWidget(self.stats_label)
        
        self.debug_button = QtWidgets.QPushButton("Print Debug Info")
        self.debug_button.clicked.connect(self.print_debug_info)
        controls_layout.addWidget(self.debug_button)
        
        # Clear data button for testing
        self.clear_button = QtWidgets.QPushButton("Clear Data")
        self.clear_button.clicked.connect(self.clear_data)
        controls_layout.addWidget(self.clear_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Info panel
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setStyleSheet("background-color: #2d2d2d; color: #ccc; font-family: monospace; font-size: 9pt;")
        layout.addWidget(self.info_text)
        
        # Plot area
        self.figure = Figure(figsize=(12, 8), dpi=100, facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setup_plot()
        
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                color: #ccc;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        
    def setup_plot(self):
        """Initialize the matplotlib plot"""
        self.figure.clear()
        plt.style.use('dark_background')
        
        # Create 4 subplots
        self.axes = []
        for i in range(4):
            ax = self.figure.add_subplot(4, 1, i+1)
            self.axes.append(ax)
            
            # Style
            ch_idx = 3 - i  # Reverse order
            ax.set_ylabel(f'{self.channel_names[ch_idx]}\n(μV)', color='#ccc', fontsize=11, 
                         rotation=0, ha='right', va='center', labelpad=35)
            ax.tick_params(colors='#ccc', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.3, color='#555')
            ax.set_facecolor('#2d2d2d')
            
            for spine in ax.spines.values():
                spine.set_color('#555')
                
            if i == 3:  # Last subplot
                ax.set_xlabel('Time (seconds)', color='#ccc', fontsize=11)
            else:
                ax.tick_params(labelbottom=False)
        
        self.figure.suptitle('Real-time EEG Stream', color='#ccc', fontsize=14, weight='bold')
        
        # Initialize plot lines
        self.plot_lines = []
        for i, ax in enumerate(self.axes):
            ch_idx = 3 - i
            line, = ax.plot([], [], color=self.channel_colors[ch_idx], linewidth=1.2)
            self.plot_lines.append(line)
            
            # Set initial limits
            ax.set_xlim(0, self.display_seconds)
            ax.set_ylim(-100, 100)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def setup_animation(self):
        """Setup the animation for real-time updates"""
        self.animation = FuncAnimation(
            self.figure, 
            self.update_plot, 
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )
        
    def toggle_connection(self):
        if not self.is_connected:
            self.connect_to_lsl()
        else:
            self.disconnect_from_lsl()
            
    def connect_to_lsl(self):
        try:
            self.info_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Looking for LSL EEG streams...")
            
            streams = pylsl.resolve_byprop('type', 'EEG', 1, timeout=5.0)
            
            if not streams:
                self.info_text.append("❌ No EEG streams found!")
                return False
                
            stream_info = streams[0]
            self.inlet = pylsl.StreamInlet(stream_info, max_chunklen=128)
            
            info = self.inlet.info()
            reported_sr = info.nominal_srate()
            self.sampling_rate = reported_sr if reported_sr > 0 else 256.0
            self.n_channels = info.channel_count()
            
            device_name = info.name()
            
            self.info_text.append(f"✅ Connected to: {device_name}")
            self.info_text.append(f"📊 Channels: {self.n_channels}, Sample rate: {self.sampling_rate:.1f} Hz")
            
            # Clear data
            self.clear_data()
            
            self.is_connected = True
            self.connect_button.setText("Disconnect")
            self.status_label.setText("Connected - Streaming")
            self.status_label.setStyleSheet("color: #51cf66; font-weight: bold;")
            
            self.samples_received = 0
            self.data_chunks_received = 0
            self.last_update_time = time.time()
            
            return True
            
        except Exception as e:
            self.info_text.append(f"❌ Connection error: {e}")
            return False
            
    def disconnect_from_lsl(self):
        self.is_connected = False
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            
        self.connect_button.setText("Connect to LSL")
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        
    def update_display_time(self, value):
        self.display_seconds = value
        
    def clear_data(self):
        """Clear all stored data"""
        self.channel_data = [[] for _ in range(4)]
        self.time_data = []
        self.info_text.append("🧹 Data cleared")
        
    def print_debug_info(self):
        self.info_text.append(f"\n=== DEBUG INFO ===")
        self.info_text.append(f"Connected: {self.is_connected}")
        self.info_text.append(f"Samples received: {self.samples_received}")
        self.info_text.append(f"Data chunks: {self.data_chunks_received}")
        self.info_text.append(f"Data lengths: {[len(ch) for ch in self.channel_data]}")
        self.info_text.append(f"Time data length: {len(self.time_data)}")
        
        if self.time_data:
            self.info_text.append(f"Time range: {self.time_data[0]:.3f} to {self.time_data[-1]:.3f}")
            
        for i in range(4):
            if self.channel_data[i]:
                data = self.channel_data[i][-100:]  # Last 100 samples
                self.info_text.append(f"  {self.channel_names[i]}: {len(self.channel_data[i])} samples, range [{np.min(data):.2f}, {np.max(data):.2f}]")
        self.info_text.append("==================\n")
        
    def update_plot(self, frame):
        """Simplified update plot method"""
        if not self.is_connected or not self.inlet:
            return self.plot_lines
            
        try:
            # Pull data from LSL
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=256)
            
            if chunk and len(chunk) > 0:
                self.data_chunks_received += 1
                current_time = time.time()
                
                # Convert to numpy array: [samples, channels]
                chunk = np.array(chunk, dtype=np.float64)
                n_samples, n_channels = chunk.shape
                
                # Create timestamps if not provided
                if not timestamps or len(timestamps) != n_samples:
                    # Use elapsed time since connection
                    base_time = len(self.time_data) / self.sampling_rate
                    timestamps = [base_time + i/self.sampling_rate for i in range(n_samples)]
                else:
                    # Convert LSL timestamps to elapsed time
                    if self.time_data:
                        base_offset = self.time_data[0] if self.time_data else 0
                        timestamps = [ts - base_offset for ts in timestamps]
                    else:
                        start_time = timestamps[0]
                        timestamps = [ts - start_time for ts in timestamps]
                
                # Add data to our storage
                for sample_idx in range(n_samples):
                    self.time_data.append(timestamps[sample_idx])
                    for ch in range(min(4, n_channels)):  # Only take first 4 channels
                        self.channel_data[ch].append(chunk[sample_idx, ch])
                
                # Limit data size
                if len(self.time_data) > self.max_points:
                    excess = len(self.time_data) - self.max_points
                    self.time_data = self.time_data[excess:]
                    for ch in range(4):
                        self.channel_data[ch] = self.channel_data[ch][excess:]
                
                self.samples_received += n_samples
                
                # Update stats
                if current_time - self.last_update_time >= 1.0:
                    self.fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_update_time = current_time
                else:
                    self.fps_counter += 1
                
                self.stats_label.setText(f"Samples: {self.samples_received:,} | Chunks: {self.data_chunks_received} | FPS: {self.fps}")
                
                # Debug output occasionally
                if self.data_chunks_received % 50 == 0:
                    self.info_text.append(f"Chunk {self.data_chunks_received}: {n_samples} samples x {n_channels} channels")
                    if self.time_data:
                        self.info_text.append(f"  Time range now: {self.time_data[0]:.2f} to {self.time_data[-1]:.2f} seconds")
            
            # Update plots - SIMPLIFIED APPROACH
            if len(self.time_data) > 10:
                # Get the display window
                latest_time = self.time_data[-1]
                start_time = latest_time - self.display_seconds
                
                # Find indices for the display window
                start_idx = 0
                for i, t in enumerate(self.time_data):
                    if t >= start_time:
                        start_idx = i
                        break
                
                # Get data for display window
                display_times = self.time_data[start_idx:]
                
                if len(display_times) > 1:
                    # Update each channel
                    for i, line in enumerate(self.plot_lines):
                        ch_idx = 3 - i  # Reverse order
                        if ch_idx < len(self.channel_data):
                            display_data = self.channel_data[ch_idx][start_idx:]
                            if len(display_data) == len(display_times):
                                line.set_data(display_times, display_data)
                    
                    # Update axis limits
                    x_min, x_max = display_times[0], display_times[-1]
                    for i, ax in enumerate(self.axes):
                        ax.set_xlim(x_min, x_max)
                        
                        # Auto-scale y based on visible data
                        ch_idx = 3 - i
                        if ch_idx < len(self.channel_data):
                            display_data = self.channel_data[ch_idx][start_idx:]
                            if display_data:
                                y_min, y_max = min(display_data), max(display_data)
                                y_margin = max(1, (y_max - y_min) * 0.1)
                                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
        except Exception as e:
            self.info_text.append(f"❌ Update error: {e}")
            import traceback
            traceback.print_exc()
            
        return self.plot_lines

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    viewer = RealTimeEEGViewer()
    viewer.show()
    
    viewer.info_text.append("=== Real-time LSL EEG Viewer - Fixed Version ===")
    viewer.info_text.append("1. Start your LSL EEG stream")
    viewer.info_text.append("2. Click 'Connect to LSL' to start viewing")
    viewer.info_text.append("3. Use 'Print Debug Info' and 'Clear Data' for testing")
    viewer.info_text.append("=====================================================\n")
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()