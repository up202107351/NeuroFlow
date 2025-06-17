from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json

# Import your database manager
from backend import database_manager as db

# Define a purple color palette for NeuroFlow
NEUROFLOW_COLORS = ["#6F72B3", "#9370DB", "#B3B6E6", "#33366B", "#551F41", "#5A4275"]
NEUROFLOW_CMAP = sns.color_palette(NEUROFLOW_COLORS)

class HistoryPageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, main_app_window_ref=None):
        super().__init__(parent)
        self.main_app = main_app_window_ref
        self.user_id = None  # Will be set when user logs in
        self.sessions_data = []  # Store loaded sessions
        self.selected_session_id = None
        self.username = None  # Store username for display
        
        # Set up seaborn with NeuroFlow theme
        sns.set(style="darkgrid")
        sns.set_palette(NEUROFLOW_COLORS)
        
        self.initUI()
        
    def initUI(self):
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Page title with welcome message
        self.title_label = QtWidgets.QLabel("Your Meditation & Focus History")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(self.title_label)
        
        # Dashboard summary widgets
        self.dashboard_widget = QtWidgets.QWidget()
        dashboard_layout = QtWidgets.QHBoxLayout(self.dashboard_widget)
        
        # Create stat cards
        self.total_sessions_card = self.create_stat_card("Total Sessions", "0")
        self.total_duration_card = self.create_stat_card("Total Minutes", "0")
        self.avg_score_card = self.create_stat_card("Avg. Score", "0%")
        self.streak_card = self.create_stat_card("Current Streak", "0 days")
        
        dashboard_layout.addWidget(self.total_sessions_card)
        dashboard_layout.addWidget(self.total_duration_card)
        dashboard_layout.addWidget(self.avg_score_card)
        dashboard_layout.addWidget(self.streak_card)
        
        layout.addWidget(self.dashboard_widget)
        
        # Weekly trend graph
        self.weekly_trend_widget = QtWidgets.QGroupBox("Your Weekly Activity")
        self.weekly_trend_widget.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        weekly_layout = QtWidgets.QVBoxLayout(self.weekly_trend_widget)
        
        self.weekly_figure = Figure(figsize=(8, 3), dpi=100, facecolor='#2d2d2d')
        self.weekly_canvas = FigureCanvas(self.weekly_figure)
        weekly_layout.addWidget(self.weekly_canvas)
        
        layout.addWidget(self.weekly_trend_widget)
        
        # Split layout for sessions list and details
        split_layout = QtWidgets.QHBoxLayout()
        
        # --- Sessions List Panel ---
        sessions_panel = QtWidgets.QWidget()
        sessions_panel.setMaximumWidth(350)
        sessions_layout = QtWidgets.QVBoxLayout(sessions_panel)
        
        # Sessions list label
        sessions_label = QtWidgets.QLabel("Your Sessions")
        sessions_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        sessions_layout.addWidget(sessions_label)
        
        # Filter controls
        filter_layout = QtWidgets.QHBoxLayout()
        
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(["All Sessions", "Meditation Only", "Focus Only"])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        
        self.time_filter_combo = QtWidgets.QComboBox()
        self.time_filter_combo.addItems(["All Time", "Last 7 Days", "Last 30 Days"])
        self.time_filter_combo.currentIndexChanged.connect(self.apply_filter)
        
        filter_layout.addWidget(QtWidgets.QLabel("Type:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addWidget(QtWidgets.QLabel("Period:"))
        filter_layout.addWidget(self.time_filter_combo)
        sessions_layout.addLayout(filter_layout)
        
        # Sessions list
        self.sessions_list = QtWidgets.QListWidget()
        self.sessions_list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
            }
        """)
        self.sessions_list.itemClicked.connect(self.load_session_details)
        sessions_layout.addWidget(self.sessions_list)
        
        # No data message
        self.no_data_label = QtWidgets.QLabel("No sessions found. Start meditating or focusing to see your history.")
        self.no_data_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
        self.no_data_label.setAlignment(QtCore.Qt.AlignCenter)
        self.no_data_label.setWordWrap(True)
        sessions_layout.addWidget(self.no_data_label)
        self.no_data_label.hide()  # Hide initially, show if no sessions
        
        # --- Session Details Panel ---
        details_panel = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details_panel)
        
        # Session details header
        self.session_details_header = QtWidgets.QLabel("Session Details")
        self.session_details_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        details_layout.addWidget(self.session_details_header)
        
        # Session details grid - display outside of tabs as requested
        self.session_info_widget = QtWidgets.QWidget()
        details_grid = QtWidgets.QGridLayout(self.session_info_widget)
        details_grid.setColumnStretch(1, 1)
        details_grid.setVerticalSpacing(5)
        
        # Style for labels
        field_style = "color: #aaa;"
        value_style = "color: white; font-weight: bold;"
        
        type_field = QtWidgets.QLabel("Type:")
        type_field.setStyleSheet(field_style)
        self.session_type_label = QtWidgets.QLabel("-")
        self.session_type_label.setStyleSheet(value_style)
        
        date_field = QtWidgets.QLabel("Date:")
        date_field.setStyleSheet(field_style)
        self.session_date_label = QtWidgets.QLabel("-")
        self.session_date_label.setStyleSheet(value_style)
        
        duration_field = QtWidgets.QLabel("Duration:")
        duration_field.setStyleSheet(field_style)
        self.session_duration_label = QtWidgets.QLabel("-")
        self.session_duration_label.setStyleSheet(value_style)
        
        score_field = QtWidgets.QLabel("Score:")
        score_field.setStyleSheet(field_style)
        self.session_score_label = QtWidgets.QLabel("-")
        self.session_score_label.setStyleSheet(value_style)
        
        details_grid.addWidget(type_field, 0, 0)
        details_grid.addWidget(self.session_type_label, 0, 1)
        details_grid.addWidget(date_field, 1, 0)
        details_grid.addWidget(self.session_date_label, 1, 1)
        details_grid.addWidget(duration_field, 2, 0)
        details_grid.addWidget(self.session_duration_label, 2, 1)
        details_grid.addWidget(score_field, 3, 0)
        details_grid.addWidget(self.session_score_label, 3, 1)
        
        details_layout.addWidget(self.session_info_widget)
        
        # Tabs for graphs - only include Session Progress and Brain Waves
        self.graph_tabs = QtWidgets.QTabWidget()
        self.graph_tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #444;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #333;
                color: #aaa;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background-color: #444;
                color: white;
            }
        """)
        
        # Session Progress tab
        progress_tab = QtWidgets.QWidget()
        progress_layout = QtWidgets.QVBoxLayout(progress_tab)
        progress_layout.setContentsMargins(10, 15, 10, 10)
        
        # Create matplotlib figure for progress
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        progress_layout.addWidget(self.canvas)
        
        # Brain Wave Activity tab
        brainwave_tab = QtWidgets.QWidget()
        brainwave_layout = QtWidgets.QVBoxLayout(brainwave_tab)
        brainwave_layout.setContentsMargins(10, 15, 10, 10)
        
        self.brainwave_figure = Figure(figsize=(5, 3), dpi=100, facecolor='#2d2d2d')
        self.brainwave_canvas = FigureCanvas(self.brainwave_figure)
        brainwave_layout.addWidget(self.brainwave_canvas)
        
        # Add tabs
        self.graph_tabs.addTab(progress_tab, "Session Progress")
        self.graph_tabs.addTab(brainwave_tab, "Brain Waves")
        
        details_layout.addWidget(self.graph_tabs)
        
        # Select session prompt
        self.select_session_label = QtWidgets.QLabel("Select a session from the list to view details")
        self.select_session_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
        self.select_session_label.setAlignment(QtCore.Qt.AlignCenter)
        details_layout.addWidget(self.select_session_label)
        
        # Add panels to split layout
        split_layout.addWidget(sessions_panel)
        split_layout.addWidget(details_panel, 1)  # Give details panel more space
        
        layout.addLayout(split_layout)
        
        # Initialize with empty/hidden details
        self.session_info_widget.hide()
        self.graph_tabs.hide()
    
    def create_stat_card(self, title, value):
        """Create a styled stat card widget"""
        card = QtWidgets.QWidget()
        card.setObjectName("statCard")
        card.setStyleSheet("""
            QWidget#statCard {
                background-color: #3a3a3a;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setAlignment(QtCore.Qt.AlignCenter)
        
        value_label = QtWidgets.QLabel(value)
        value_label.setAlignment(QtCore.Qt.AlignCenter)
        value_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {NEUROFLOW_COLORS[2]};")
        
        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 12px; color: #aaa;")
        
        card_layout.addWidget(value_label)
        card_layout.addWidget(title_label)
        
        # Store the value label so we can update it
        card.value_label = value_label
        
        return card
        
    def load_user_data(self, user_id, username=None):
        """Load session data for the current user"""
        self.user_id = user_id
        self.username = username
        
        if username:
            self.title_label.setText(f"Welcome, {username} - Your Meditation & Focus History")
        
        self.refresh_sessions_list()
        self.update_dashboard_summary()
        self.update_weekly_trend()
        
    def refresh_sessions_list(self):
        """Refresh the sessions list with filtered data"""
        if self.user_id is None:
            # No user logged in yet
            self.sessions_list.clear()
            self.no_data_label.show()
            return
            
        try:
            # Get all sessions for this user
            self.sessions_data = db.get_all_sessions_summary(self.user_id)
            self.apply_filter()  # Apply any active filters
            
            # Update dashboard stats
            self.update_dashboard_summary()
            self.update_weekly_trend()
            
        except Exception as e:
            print(f"Error loading history data: {e}")
            self.sessions_data = []
            self.sessions_list.clear()
            self.no_data_label.show()
            
    def update_dashboard_summary(self):
        """Update the dashboard summary statistics"""
        if not self.sessions_data:
            # No data, set defaults
            self.total_sessions_card.value_label.setText("0")
            self.total_duration_card.value_label.setText("0")
            self.avg_score_card.value_label.setText("0%")
            self.streak_card.value_label.setText("0 days")
            return
            
        # Calculate total sessions
        total_sessions = len(self.sessions_data)
        self.total_sessions_card.value_label.setText(str(total_sessions))
        
        # Calculate total duration (in minutes)
        total_duration = sum(session['duration_seconds'] or 0 for session in self.sessions_data) / 60
        self.total_duration_card.value_label.setText(f"{int(total_duration)}")
        
        # Calculate average score - safely handle sqlite3.Row objects
        scores = []
        for session in self.sessions_data:
            # Safely access percent_on_target - handle both dict and sqlite3.Row
            try:
                # For dict-like objects
                score = session['percent_on_target']
            except (TypeError, IndexError):
                # For sqlite3.Row, try accessing by column name
                try:
                    score = session['percent_on_target']
                except:
                    score = None
                    
            if score is not None:
                scores.append(score)
                
        avg_score = sum(scores) / len(scores) if scores else 0
        self.avg_score_card.value_label.setText(f"{avg_score:.1f}%")
        
        # Calculate streak (consecutive days with sessions)
        streak = self.calculate_streak()
        self.streak_card.value_label.setText(f"{streak} days")
        
    def calculate_streak(self):
        """Calculate the current streak (consecutive days with sessions)"""
        if not self.sessions_data:
            return 0
            
        # Get all session dates
        session_dates = []
        for session in self.sessions_data:
            date_str = session['start_time']
            date = datetime.fromisoformat(date_str).date()
            session_dates.append(date)
            
        # Sort dates and get unique dates (one session per day counts)
        unique_dates = sorted(set(session_dates), reverse=True)
        
        if not unique_dates:
            return 0
            
        # Check if there's a session today
        today = datetime.now().date()
        streak = 0
        
        # If no session today, start checking from yesterday
        check_date = today if unique_dates[0] == today else today - timedelta(days=1)
        
        # Count streak
        for date in unique_dates:
            if date == check_date or date == check_date - timedelta(days=1):
                streak += 1
                check_date = date
            else:
                break
                
        return streak
        
    def update_weekly_trend(self):
        """Update the weekly trend graph"""
        self.weekly_figure.clear()
        ax = self.weekly_figure.add_subplot(111)
        
        if not self.sessions_data:
            ax.text(0.5, 0.5, "No data available for weekly trend", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='#888')
            ax.set_axis_off()
            self.weekly_canvas.draw()
            return
            
        # Prepare data for last 7 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=6)  # 7 days including today
        
        # Create date range
        date_range = [start_date + timedelta(days=i) for i in range(7)]
        
        # Count sessions per day
        meditation_counts = [0] * 7
        focus_counts = [0] * 7
        
        for session in self.sessions_data:
            session_date = datetime.fromisoformat(session['start_time']).date()
            if start_date <= session_date <= end_date:
                day_index = (session_date - start_date).days
                if 0 <= day_index < 7:
                    if 'Meditation' in session['session_type']:
                        meditation_counts[day_index] += 1
                    elif 'Focus' in session['session_type']:
                        focus_counts[day_index] += 1
        
        # Format x-axis labels
        x_labels = [d.strftime('%a') for d in date_range]
        x = np.arange(len(x_labels))
        
        # Plot stacked bars
        bar_width = 0.6
        p1 = ax.bar(x, meditation_counts, bar_width, label='Meditation', color=NEUROFLOW_COLORS[0])
        p2 = ax.bar(x, focus_counts, bar_width, bottom=meditation_counts, label='Focus', color=NEUROFLOW_COLORS[1])
        
        # Add labels and legend
        ax.set_ylabel('Sessions')
        ax.set_title('Your Activity This Week')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        
        # Style the graph for dark theme
        ax.set_facecolor('#2d2d2d')
        for spine in ax.spines.values():
            spine.set_color('#555')
        
        ax.tick_params(colors='#ccc')
        ax.title.set_color('#ccc')
        ax.yaxis.label.set_color('#ccc')
        ax.legend(facecolor='#2d2d2d', edgecolor='#555', labelcolor='#ccc')
        
        # Highlight today
        today_index = (end_date - start_date).days
        ax.get_xticklabels()[today_index].set_color(NEUROFLOW_COLORS[0])
        ax.get_xticklabels()[today_index].set_fontweight('bold')
        
        self.weekly_figure.tight_layout()
        self.weekly_canvas.draw()
            
    def apply_filter(self):
        """Apply filters to the sessions list"""
        self.sessions_list.clear()
        
        if not self.sessions_data:
            self.no_data_label.show()
            return
            
        # Get filter settings
        type_filter = self.filter_combo.currentText()
        time_filter = self.time_filter_combo.currentText()
        
        # Apply filters
        filtered_data = []
        current_date = datetime.now()
        
        for session in self.sessions_data:
            # Type filter
            if type_filter == "Meditation Only" and 'Meditation' not in session['session_type']:
                continue
            if type_filter == "Focus Only" and 'Focus' not in session['session_type']:
                continue
                
            # Time filter
            session_date = datetime.fromisoformat(session['start_time'])
            if time_filter == "Last 7 Days" and (current_date - session_date).days > 7:
                continue
            if time_filter == "Last 30 Days" and (current_date - session_date).days > 30:
                continue
                
            filtered_data.append(session)
            
        # Update UI based on filtered results
        if not filtered_data:
            self.no_data_label.show()
        else:
            self.no_data_label.hide()
            
            # Add sessions to list
            for session in filtered_data:
                session_date = datetime.fromisoformat(session['start_time']).strftime("%b %d, %Y - %I:%M %p")
                session_type = session['session_type']
                
                # Calculate score for display - safely handle sqlite3.Row
                try:
                    score = session['percent_on_target']
                except (TypeError, KeyError, IndexError):
                    # For sqlite3.Row, be more careful
                    try:
                        score = session['percent_on_target']
                    except:
                        score = None
                
                score_text = f"{score:.1f}%" if score is not None else "N/A"
                
                # Format list item
                item = QtWidgets.QListWidgetItem()
                item.setData(QtCore.Qt.UserRole, session['session_id'])
                
                # Create custom widget for list item
                item_widget = QtWidgets.QWidget()
                item_layout = QtWidgets.QVBoxLayout(item_widget)
                
                type_label = QtWidgets.QLabel(session_type)
                type_label.setStyleSheet("font-weight: bold;")
                
                date_label = QtWidgets.QLabel(session_date)
                date_label.setStyleSheet("font-size: 10px; color: #aaa;")
                
                score_label = QtWidgets.QLabel(f"Score: {score_text}")
                score_label.setStyleSheet(f"font-size: 10px; color: {NEUROFLOW_COLORS[2]};")
                
                item_layout.addWidget(type_label)
                item_layout.addWidget(date_label)
                item_layout.addWidget(score_label)
                
                # Set size for the item
                item.setSizeHint(item_widget.sizeHint())
                
                self.sessions_list.addItem(item)
                self.sessions_list.setItemWidget(item, item_widget)
                
    def load_session_details(self, item):
        """Load details for the selected session"""
        self.selected_session_id = item.data(QtCore.Qt.UserRole)
        
        # Find selected session in data
        selected_session = None
        for session in self.sessions_data:
            if session['session_id'] == self.selected_session_id:
                selected_session = session
                break
                
        if not selected_session:
            return
            
        # Update details panel
        self.session_type_label.setText(selected_session['session_type'])
        
        # Format date
        session_date = datetime.fromisoformat(selected_session['start_time'])
        self.session_date_label.setText(session_date.strftime("%B %d, %Y at %I:%M %p"))
        
        # Format duration
        duration_seconds = selected_session['duration_seconds'] or 0
        minutes, seconds = divmod(duration_seconds, 60)
        self.session_duration_label.setText(f"{minutes} min {seconds} sec")
        
        # Format score - safely access
        try:
            percent_on_target = selected_session['percent_on_target']
        except (TypeError, KeyError, IndexError):
            # For sqlite3.Row, be more careful
            try:
                percent_on_target = selected_session['percent_on_target']
            except:
                percent_on_target = None
                
        if percent_on_target is not None:
            self.session_score_label.setText(f"{percent_on_target:.1f}% on target")
        else:
            self.session_score_label.setText("Not available")
            
        # Show details and hide prompt
        self.select_session_label.hide()
        self.session_info_widget.show()
        self.graph_tabs.show()
        
        # Generate graphs
        self.generate_session_graph(self.selected_session_id)
        self.generate_brainwave_graph(self.selected_session_id)
        
        # Switch to the Session Progress tab by default
        self.graph_tabs.setCurrentIndex(0)
        
    def generate_session_graph(self, session_id):
        """Generate clean 4-channel EEG data visualization"""
        try:
            # Get EEG data from new batch format
            eeg_data = db.get_session_eeg_data(session_id)
            
            if not eeg_data:
                # No EEG data available
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No EEG data available for this session", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='#888')
                ax.set_axis_off()
                self.canvas.draw()
                return
                
            # Extract EEG data
            timestamps = eeg_data['timestamps']
            channel_data = [
                eeg_data['channel_0'],  # TP9
                eeg_data['channel_1'],  # AF7  
                eeg_data['channel_2'],  # AF8
                eeg_data['channel_3']   # TP10
            ]
            sampling_rate = eeg_data.get('sampling_rate', 256.0)
            
            if not timestamps or len(timestamps) == 0:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No EEG data points found", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='#888')
                ax.set_axis_off()
                self.canvas.draw()
                return
            
            # Convert timestamps to seconds elapsed
            start_time = timestamps[0]
            time_seconds = [(ts - start_time) for ts in timestamps]
            
            # Create the plot
            self.figure.clear()
            plt.style.use('dark_background')
            self.figure.patch.set_facecolor('#2d2d2d')
            
            # Channel names and colors
            channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
            channel_colors = [NEUROFLOW_COLORS[0], NEUROFLOW_COLORS[1], 
                            NEUROFLOW_COLORS[2], NEUROFLOW_COLORS[3]]
            
            # Create 4 subplots with manual positioning to avoid tight_layout issues
            fig = self.figure
            
            # Manual subplot positioning [left, bottom, width, height]
            subplot_height = 0.18  # Height of each subplot
            subplot_width = 0.75   # Width of subplots
            left_margin = 0.15     # Space for y-labels
            bottom_start = 0.08    # Bottom margin
            vertical_spacing = 0.02  # Space between subplots
            
            axes = []
            for i in range(4):
                # Calculate position (bottom subplot first, then stack upward)
                bottom = bottom_start + i * (subplot_height + vertical_spacing)
                ax = fig.add_axes([left_margin, bottom, subplot_width, subplot_height])
                axes.append(ax)
                
                # Plot the channel data
                ax.plot(time_seconds, channel_data[3-i], color=channel_colors[3-i], 
                    linewidth=0.8, alpha=0.9)
                
                # Style each subplot
                ax.set_ylabel(f'{channel_names[3-i]}\n(μV)', color='#ccc', fontsize=10, 
                            rotation=0, ha='right', va='center')
                ax.tick_params(colors='#ccc', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.2, color='#555')
                
                # Set background
                ax.set_facecolor('#2d2d2d')
                for spine in ax.spines.values():
                    spine.set_color('#555')
                
                # Only show x-axis label on bottom subplot
                if i == 0:  # Bottom subplot
                    ax.set_xlabel('Time (seconds)', color='#ccc', fontsize=10)
                else:
                    ax.tick_params(labelbottom=False)
                    
                # Share x-axis with bottom subplot for consistent scaling
                if i > 0:
                    ax.sharex(axes[0])
            
            # Add title manually positioned
            fig.text(0.5, 0.95, f'Session Raw EEG Data', 
                    ha='center', va='top', color='#ccc', fontsize=12, weight='bold')
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error generating EEG graph: {e}")
            import traceback
            traceback.print_exc()
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error loading EEG data: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color='#e74c3c')
            ax.set_axis_off()
            self.canvas.draw()
    
    def generate_brainwave_graph(self, session_id):
        """Generate brain wave activity visualization using real data"""
        self.brainwave_figure.clear()
        ax = self.brainwave_figure.add_subplot(111)
        
        try:
            # Get band power data from new batch format
            band_data = db.get_session_band_data(session_id)
            
            if not band_data:
                ax.text(0.5, 0.5, "No brainwave data available for this session", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='#888')
                ax.set_axis_off()
                self.brainwave_canvas.draw()
                return
            
            # Extract data
            timestamps = band_data['timestamps']
            alpha_data = band_data['alpha']
            beta_data = band_data['beta']
            theta_data = band_data['theta']
            
            if not timestamps or len(timestamps) == 0:
                ax.text(0.5, 0.5, "No brainwave data points found", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='#888')
                ax.set_axis_off()
                self.brainwave_canvas.draw()
                return
            
            # Convert timestamps to minutes elapsed
            start_time = timestamps[0]
            minutes_elapsed = [(ts - start_time) / 60 for ts in timestamps]
            
            # Plot actual brainwave data
            ax.plot(minutes_elapsed, alpha_data, color=NEUROFLOW_COLORS[0], 
                linewidth=2, label='Alpha (8-13 Hz) - Relaxation')
            ax.plot(minutes_elapsed, beta_data, color=NEUROFLOW_COLORS[1], 
                linewidth=2, label='Beta (13-30 Hz) - Focus')
            ax.plot(minutes_elapsed, theta_data, color=NEUROFLOW_COLORS[2], 
                linewidth=2, label='Theta (4-8 Hz) - Deep State')
            
            # Style the plot
            ax.set_facecolor('#2d2d2d')
            ax.set_xlabel('Time (minutes)', color='#ccc')
            ax.set_ylabel('Power (μV²)', color='#ccc')
            ax.set_title('Brainwave Band Powers Over Time', color='#ccc')
            ax.tick_params(colors='#ccc')
            ax.grid(True, linestyle='--', alpha=0.3, color='#555')
            
            # Add legend
            ax.legend(facecolor='#333', edgecolor='#555', labelcolor='#ccc')
            
            # Add average values as text
            avg_alpha = np.mean(alpha_data) if alpha_data else 0
            avg_beta = np.mean(beta_data) if beta_data else 0
            avg_theta = np.mean(theta_data) if theta_data else 0
            
            stats_text = (
                f"Session Averages:\n"
                f"Alpha: {avg_alpha:.2f} μV²\n"
                f"Beta: {avg_beta:.2f} μV²\n"
                f"Theta: {avg_theta:.2f} μV²"
            )
            
            ax.text(0.98, 0.02, stats_text, 
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes, fontsize=9, color='#ccc',
                    bbox=dict(facecolor='#333', alpha=0.8, boxstyle='round'))
                    
            self.brainwave_figure.tight_layout()
            self.brainwave_canvas.draw()
            
        except Exception as e:
            print(f"Error generating brainwave graph: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error loading brainwave data: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='#e74c3c')
            ax.set_axis_off()
            self.brainwave_canvas.draw()