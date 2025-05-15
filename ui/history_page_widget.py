from PyQt5 import QtWidgets, QtCore, QtGui
import subprocess # To launch the backend script 
import backend.database_manager as db_manager 
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg') # Important: Use Qt5 backend for Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd 

class HistoryPageWidget(QtWidgets.QWidget):
    def __init__(self, parent = None, main_app_window_ref = None):
        super().__init__(parent)
        self.main_app_window = main_app_window_ref
        self.initUI()

    def initUI(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QtWidgets.QLabel("Session History")
        title_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # --- Controls (e.g., refresh button, date filters - optional for now) ---
        self.btn_refresh = QtWidgets.QPushButton("Refresh History")
        self.btn_refresh.clicked.connect(self.load_history_data)
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addStretch()
        control_layout.addWidget(self.btn_refresh)
        main_layout.addLayout(control_layout)


        # --- Session List (Table View) ---
        self.sessions_table = QtWidgets.QTableWidget()
        self.sessions_table.setColumnCount(5) # session_id, type, start_time, duration, % on target
        self.sessions_table.setHorizontalHeaderLabels(["ID", "Type", "Start Time", "Duration (s)", "% On Target"])
        self.sessions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sessions_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sessions_table.itemSelectionChanged.connect(self.on_session_selected)
        main_layout.addWidget(self.sessions_table)

        # --- Details Area (for selected session's graph) ---
        self.details_area_label = QtWidgets.QLabel("Select a session to view details.")
        self.details_area_label.setAlignment(QtCore.Qt.AlignCenter)

        # Matplotlib Canvas
        self.figure = Figure(figsize=(5, 3), dpi=100) # Smaller figure for embedding
        self.canvas = FigureCanvas(self.figure)
        # Initially hide canvas until data is loaded
        self.canvas.setVisible(False)


        details_layout = QtWidgets.QVBoxLayout()
        details_layout.addWidget(self.details_area_label)
        details_layout.addWidget(self.canvas) # Add canvas here

        main_layout.addLayout(details_layout)

        self.load_history_data() # Load data when page is created


    def load_history_data(self):
        self.sessions_table.setRowCount(0) # Clear existing rows
        self.canvas.setVisible(False) # Hide graph
        self.details_area_label.setText("Select a session to view details.")
        self.details_area_label.setVisible(True)

        try:
            sessions_data = db_manager.get_all_sessions_summary()
            self.sessions_table.setRowCount(len(sessions_data))

            for row_idx, session_row in enumerate(sessions_data):
                self.sessions_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(session_row['session_id'])))
                self.sessions_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(session_row['session_type']))
                start_time_dt = datetime.fromisoformat(session_row['start_time'])
                self.sessions_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(start_time_dt.strftime("%Y-%m-%d %H:%M")))
                self.sessions_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(session_row['duration_seconds'])))
                percent_on_target = f"{session_row['percent_on_target']:.1f}%" if session_row['percent_on_target'] is not None else "N/A"
                self.sessions_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(percent_on_target))
            self.sessions_table.resizeColumnsToContents()
        except Exception as e:
            print(f"Error loading history data: {e}")
            QtWidgets.QMessageBox.critical(self, "DB Error", f"Could not load session history: {e}")

    def on_session_selected(self):
        selected_items = self.sessions_table.selectedItems()
        if not selected_items:
            self.canvas.setVisible(False)
            self.details_area_label.setText("Select a session to view details.")
            self.details_area_label.setVisible(True)
            return

        selected_row = selected_items[0].row()
        session_id_item = self.sessions_table.item(selected_row, 0)
        if not session_id_item: return

        try:
            session_id = int(session_id_item.text())
            session_details = db_manager.get_session_details(session_id)

            if not session_details:
                self.details_area_label.setText(f"No detailed metrics found for session {session_id}.")
                self.details_area_label.setVisible(True)
                self.canvas.setVisible(False)
                return

            self.plot_session_details(session_details)
            self.details_area_label.setVisible(False) # Hide placeholder text
            self.canvas.setVisible(True) # Show graph

        except Exception as e:
            print(f"Error loading/plotting session details: {e}")
            self.details_area_label.setText(f"Error loading details: {e}")
            self.details_area_label.setVisible(True)
            self.canvas.setVisible(False)


    def plot_session_details(self, details_data):
        """Plots session metrics (e.g., prediction over time)."""
        self.figure.clear() # Clear previous plot
        ax = self.figure.add_subplot(111)

        if not details_data:
            ax.text(0.5, 0.5, 'No data to display for this session.',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()
            return

        # Convert to DataFrame for easier plotting with Seaborn/Matplotlib
        df = pd.DataFrame(details_data, columns=['timestamp', 'prediction_label', 'is_on_target', 'raw_score'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

        # Example plot: prediction_label over time (as a categorical plot or step plot)
        # For simplicity, let's map predictions to numerical values for a line plot
        # This needs to be adapted based on what 'prediction_label' contains
        prediction_map = {"Relaxed": 1, "Neutral": 0, "Agitated/Focused": -1, "Focused": -1, "Unknown": 0} # Example
        df['prediction_value'] = df['prediction_label'].map(prediction_map).fillna(0)

        # sns.lineplot(x='timestamp', y='prediction_value', data=df, ax=ax, marker='o', hue='is_on_target') # Hue by on_target
        ax.plot(df['timestamp'], df['prediction_value'], marker='.', linestyle='-')
        ax.set_yticks(list(prediction_map.values()))
        ax.set_yticklabels(list(prediction_map.keys()))


        ax.set_title("Session State Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Detected State")
        self.figure.autofmt_xdate() # Auto format date on x-axis
        ax.grid(True, linestyle='--', alpha=0.7)

        self.canvas.draw() # Redraw the canvas

    def update_button_tooltips(self, is_lsl_connected_from_main_app):
        # History page doesn't typically have LSL-dependent start buttons
        pass

    def clean_up_session(self): # Usually not needed for history page
        pass