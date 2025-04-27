import time
import os
from pathlib import Path
import sys
import warnings
import signal # To handle Ctrl+C gracefully
import csv
from datetime import datetime

# Data Handling
import numpy as np

# LSL Streaming
import pylsl

# Filtering (Optional, but good for visualization/basic cleaning)
from brainflow.data_filter import DataFilter, FilterTypes

# Visualization and Control
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- Configuration ---

# LSL Stream Configuration
LSL_STREAM_NAME = 'BlueMuse' # Name of the LSL stream broadcast by BlueMuse
# LSL_STREAM_TYPE = 'EEG' # Alternatively, use type if name is unstable
LSL_CHUNK_MAX = 10 # Max samples to pull per iteration

# Recording Parameters (MUST match LSL output)
ASSUMED_SAMPLE_RATE = 256.0 # Hz (Verify in BlueMuse LSL settings)
PLOT_WINDOW_DURATION_SECONDS = 5 # How many seconds to display in EEG plot
BUFFER_DURATION_SECONDS = 10 # How much data to keep in memory for plotting

# Channel indices from LSL stream to use (e.g., TP9, AF7, AF8, TP10)
# Ensure these are 0-based indices corresponding to the LSL stream's channels.
EEG_CHANNEL_INDICES = [0, 1, 2, 3] # Adjust if your BlueMuse setup differs
NUM_CHANNELS_USED = len(EEG_CHANNEL_INDICES)
# Channel names for the CSV header (Optional but helpful)
CHANNEL_NAMES = [f"Ch{i}" for i in EEG_CHANNEL_INDICES] # Or specific names like ['TP9', 'AF7', 'AF8', 'TP10']

# Directory where the recorded data will be saved
SAVE_DIR = Path('./eeg_recordings') # Creates a folder in the script's directory
SAVE_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

# Filtering parameters (Optional: For visualization/basic cleaning during recording)
# Set CUTOFFS to None or 0 to disable respective filters
FILTER_ORDER = 4
NOTCH_CENTER_FREQ = 50 # Hz (Adjust to 60 Hz for US/Canada)
NOTCH_BANDWIDTH = 2.0 # Hz
BAND_PASS_LOW_CUTOFF = 0.5 # Hz (Set to None or 0 to disable low-cut)
BAND_PASS_HIGH_CUTOFF = 50 # Hz (Set to None or 0 to disable high-cut)

# Visualization settings
UPDATE_INTERVAL_MS = 40 # Update plot roughly every 40ms (~25 FPS)

# --- Automatically calculated ---
SAMPLES_PER_PLOT_WINDOW = int(ASSUMED_SAMPLE_RATE * PLOT_WINDOW_DURATION_SECONDS)
BUFFER_SAMPLES = int(ASSUMED_SAMPLE_RATE * BUFFER_DURATION_SECONDS)
FS_INT = int(ASSUMED_SAMPLE_RATE) # Integer sample rate for filters

# --- Global Variables ---
running = True # Flag to signal exit
lsl_inlet = None
data_buffer = None
timestamps = None
buffer_samples_filled = 0
current_state = "Idle" # Can be 'Idle', 'Relaxing', 'Focusing', 'Baseline'
csv_writer = None
output_file = None
start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = SAVE_DIR / f"eeg_recording_{start_time_str}.csv"

# --- LSL Connection Function (Slightly modified for clarity) ---
def connect_lsl(stream_name=LSL_STREAM_NAME, stream_type='EEG', timeout=5):
    """Attempts to connect to the LSL stream."""
    print(f"Looking for LSL stream '{stream_name}' (type: {stream_type})...")
    try:
        # streams = pylsl.resolve_byprop('name', stream_name, 1, timeout=timeout)
        streams = pylsl.resolve_byprop('type', stream_type, 1, timeout=timeout) # Often more reliable
        if not streams:
            raise ConnectionError(f"Could not find LSL stream named '{stream_name}' (type: {stream_type}).")

        print("LSL stream found. Connecting...")
        inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX)
        stream_info = inlet.info()
        actual_sample_rate = stream_info.nominal_srate()
        num_lsl_channels = stream_info.channel_count()

        print(f"Connected to '{stream_info.name()}'")
        print(f"  Actual Sample Rate: {actual_sample_rate:.2f} Hz")
        print(f"  Number of Channels: {num_lsl_channels}")

        # Validation
        if abs(actual_sample_rate - ASSUMED_SAMPLE_RATE) > 1.0:
             warnings.warn(f"LSL rate ({actual_sample_rate:.2f} Hz) differs significantly from assumed rate ({ASSUMED_SAMPLE_RATE:.2f} Hz).", UserWarning)
        if num_lsl_channels < max(EEG_CHANNEL_INDICES) + 1:
             raise ValueError(f"LSL stream has only {num_lsl_channels} channels, script requires indices up to {max(EEG_CHANNEL_INDICES)}.")

        print(f"Using channels: {EEG_CHANNEL_INDICES}")
        return inlet, actual_sample_rate, None # Success

    except Exception as e:
        error_message = f"ERROR connecting to LSL: {e}"
        print(error_message)
        return None, None, error_message # Failure


# --- Main Recording and Visualization Function ---
def run_realtime_recording():
    """
    Connects to LSL, visualizes EEG, and records data segments with labels.
    """
    global running, lsl_inlet, data_buffer, timestamps, buffer_samples_filled
    global current_state, csv_writer, output_file, output_filename

    print("--- Starting Real-time EEG Recording ---")
    print(f"Data will be saved to: {output_filename}")

    # --- 1. Attempt LSL Connection ---
    lsl_inlet, actual_fs, connection_error_msg = connect_lsl()
    lsl_connected = lsl_inlet is not None

    # --- 2. Setup PyQtGraph Application ---
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Real-time EEG Recording", size=(1000, 700))

    # --- 3. Create Plot Areas ---
    # EEG Plots (Left Side)
    eeg_plots = []
    eeg_curves = []
    plot_layout = win.addLayout(row=0, col=0, rowspan=5) # Span multiple rows
    for i in range(NUM_CHANNELS_USED):
        p = plot_layout.addPlot(row=i, col=0)
        p.setYRange(-80, 80) # Adjust Y range as needed
        p.setLabel('left', f"Ch {EEG_CHANNEL_INDICES[i]}", units='uV')
        p.showGrid(x=True, y=True, alpha=0.3)
        if i < NUM_CHANNELS_USED - 1:
            p.hideAxis('bottom')
        else:
            p.setLabel('bottom', "Time (s)")
        curve = p.plot(pen=pg.mkPen(color=pg.intColor(i, hues=NUM_CHANNELS_USED), width=1))
        eeg_plots.append(p)
        eeg_curves.append(curve)

    # --- 4. Create Control Panel (Right Side) ---
    control_layout = win.addLayout(row=0, col=1) # Place next to plots

    # Instruction/Status Label
    status_label = QtWidgets.QLabel("Press Start...")
    status_label.setAlignment(QtCore.Qt.AlignCenter)
    status_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
    proxy_status = QtWidgets.QGraphicsProxyWidget()
    proxy_status.setWidget(status_label)
    control_layout.addItem(proxy_status, row=0, col=0)

    # Recording Buttons
    button_layout = QtWidgets.QWidget()
    button_grid = QtWidgets.QGridLayout(button_layout)

    btn_relax = QtWidgets.QPushButton("Start Relaxing")
    btn_focus = QtWidgets.QPushButton("Start Focusing")
    btn_baseline = QtWidgets.QPushButton("Start Baseline")
    btn_stop = QtWidgets.QPushButton("Stop Recording")
    btn_stop.setStyleSheet("background-color: #FF8C8C;") # Make stop red-ish

    button_grid.addWidget(btn_relax, 0, 0)
    button_grid.addWidget(btn_focus, 1, 0)
    button_grid.addWidget(btn_baseline, 2, 0)
    button_grid.addWidget(btn_stop, 3, 0)

    proxy_buttons = QtWidgets.QGraphicsProxyWidget()
    proxy_buttons.setWidget(button_layout)
    control_layout.addItem(proxy_buttons, row=1, col=0)

    # Filename Label (optional display)
    filename_label = QtWidgets.QLabel(f"Saving to: {output_filename.name}")
    filename_label.setAlignment(QtCore.Qt.AlignCenter)
    filename_label.setStyleSheet("font-size: 10pt;")
    proxy_filename = QtWidgets.QGraphicsProxyWidget()
    proxy_filename.setWidget(filename_label)
    control_layout.addItem(proxy_filename, row=2, col=0)


    # --- Show the window ---
    win.show()

    # --- 5. Initialize Buffers and State (Only if connected) ---
    if lsl_connected:
        data_buffer = np.zeros((NUM_CHANNELS_USED, BUFFER_SAMPLES))
        timestamps = np.zeros(BUFFER_SAMPLES)
        buffer_samples_filled = 0
        status_label.setText("Connected. Press Start...")
        status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green;")
        print("Buffers initialized.")
    else:
        data_buffer = None
        timestamps = None
        buffer_samples_filled = -1 # Signify not ready/connected
        status_label.setText(f"LSL Connection Failed:\n{connection_error_msg}")
        status_label.setStyleSheet("font-size: 12pt; color: red;")
        # Disable buttons if not connected
        btn_relax.setEnabled(False)
        btn_focus.setEnabled(False)
        btn_baseline.setEnabled(False)
        btn_stop.setEnabled(False)
        print("LSL not connected. Buffers not initialized.")


    # --- 6. Define Button Actions ---
    def start_recording(label):
        global current_state, csv_writer, output_file
        if not lsl_connected:
             print("Cannot start recording: LSL not connected.")
             return
        if current_state != "Idle":
            print(f"Already recording '{current_state}'. Stop first.")
            # Optional: show a warning dialog
            return

        print(f"Starting recording: {label}")
        current_state = label
        status_label.setText(f"RECORDING: {label}")
        status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: red;")

        # Open CSV file and write header if it's new or doesn't exist
        file_exists = output_filename.exists()
        try:
            # Use 'a+' to append if exists, create if not. newline='' is important for csv
            output_file = open(output_filename, 'a+', newline='')
            csv_writer = csv.writer(output_file)
            # Write header only if the file is newly created (or empty)
            if not file_exists or output_file.tell() == 0:
                 header = ['Timestamp'] + CHANNEL_NAMES + ['Label']
                 csv_writer.writerow(header)
                 print("CSV header written.")
        except Exception as e:
             print(f"ERROR opening/writing header to {output_filename}: {e}")
             status_label.setText(f"File Error: {e}")
             status_label.setStyleSheet("font-size: 12pt; color: red;")
             current_state = "Idle" # Reset state
             if output_file:
                 output_file.close()
                 output_file = None
             csv_writer = None


    def stop_recording():
        global current_state, csv_writer, output_file
        if current_state == "Idle":
            print("Not currently recording.")
            return

        print(f"Stopping recording: {current_state}")
        last_state = current_state
        current_state = "Idle"
        status_label.setText(f"Stopped {last_state}. Press Start...")
        status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green;")

        # Close the CSV file
        if output_file:
            try:
                output_file.close()
                print(f"Data saved to {output_filename}")
            except Exception as e:
                print(f"Error closing file: {e}")
        output_file = None
        csv_writer = None

    # Connect buttons to actions
    btn_relax.clicked.connect(lambda: start_recording("Relaxing"))
    btn_focus.clicked.connect(lambda: start_recording("Focusing"))
    btn_baseline.clicked.connect(lambda: start_recording("Baseline"))
    btn_stop.clicked.connect(stop_recording)


    # --- 7. Update Function ---
    def update():
        global data_buffer, timestamps, buffer_samples_filled
        global running, current_state, csv_writer

        if not running:
            if current_state != "Idle": # Ensure data is saved if recording during shutdown
                stop_recording()
            if lsl_connected and lsl_inlet:
                try:
                    lsl_inlet.close_stream()
                    print("LSL stream closed.")
                except Exception as e:
                    print(f"Error closing LSL stream: {e}")
            app.quit() # Exit Qt application
            return

        # Only process LSL if connected
        if lsl_connected:
            # --- Pull Data from LSL ---
            try:
                chunk, ts = lsl_inlet.pull_chunk(timeout=0.0, max_samples=LSL_CHUNK_MAX)
            except Exception as e:
                 print(f"Error pulling LSL chunk: {e}. Stopping.")
                 running = False # Signal exit on LSL error
                 status_label.setText(f"LSL Error: {e}")
                 status_label.setStyleSheet("font-size: 12pt; color: red;")
                 # Disable buttons
                 btn_relax.setEnabled(False)
                 btn_focus.setEnabled(False)
                 btn_baseline.setEnabled(False)
                 btn_stop.setEnabled(False)
                 return # Skip rest of update

            new_samples_count = len(ts)

            if new_samples_count > 0:
                # Convert chunk to numpy array and select channels
                chunk_np = np.array(chunk)[:, EEG_CHANNEL_INDICES].T # Transpose for (channels x samples)
                chunk_np = chunk_np.astype(np.float64)

                # --- Optional: Filter incoming chunk for visualization/basic clean ---
                # Apply filters *before* adding to buffer if desired
                try:
                    for i in range(NUM_CHANNELS_USED):
                        # Notch Filter (Powerline Noise)
                        if NOTCH_CENTER_FREQ and NOTCH_BANDWIDTH:
                            DataFilter.perform_bandstop(chunk_np[i], FS_INT, NOTCH_CENTER_FREQ, NOTCH_BANDWIDTH, FILTER_ORDER, FilterTypes.BUTTERWORTH, 0)
                        # Bandpass Filter
                        if BAND_PASS_LOW_CUTOFF and BAND_PASS_HIGH_CUTOFF:
                             DataFilter.perform_bandpass(chunk_np[i], FS_INT, BAND_PASS_LOW_CUTOFF, BAND_PASS_HIGH_CUTOFF, FILTER_ORDER, FilterTypes.BUTTERWORTH, 0)
                except Exception as filter_e:
                    # Don't stop recording for filter errors, just log them
                    # print(f"Warning: Filtering error on chunk: {filter_e}")
                    pass

                # --- Update Plotting Buffers ---
                # Roll buffer, add new data at the end
                data_buffer = np.roll(data_buffer, -new_samples_count, axis=1)
                timestamps = np.roll(timestamps, -new_samples_count)
                data_buffer[:, -new_samples_count:] = chunk_np
                timestamps[-new_samples_count:] = ts
                buffer_samples_filled = min(BUFFER_SAMPLES, buffer_samples_filled + new_samples_count)

                # --- Write to CSV if Recording ---
                if current_state != "Idle" and csv_writer and output_file:
                    try:
                        for sample_idx in range(new_samples_count):
                            timestamp = ts[sample_idx]
                            sample_data = chunk_np[:, sample_idx].tolist() # Data for this sample
                            row = [timestamp] + sample_data + [current_state]
                            csv_writer.writerow(row)
                    except Exception as e:
                        print(f"ERROR writing data to CSV: {e}. Stopping recording.")
                        stop_recording() # Stop recording on write error
                        status_label.setText(f"CSV Write Error: {e}")
                        status_label.setStyleSheet("font-size: 12pt; color: red;")


            # --- Update EEG Plot ---
            # Use only the most recent samples for plotting
            plot_start_idx = max(0, buffer_samples_filled - SAMPLES_PER_PLOT_WINDOW)
            plot_end_idx = buffer_samples_filled
            # Correct slicing: Need to get data from the *end* of the buffer
            valid_buffer_samples = data_buffer[:, -buffer_samples_filled:]
            valid_timestamps = timestamps[-buffer_samples_filled:]

            plot_data = valid_buffer_samples[:, plot_start_idx:plot_end_idx]
            plot_times = valid_timestamps[plot_start_idx:plot_end_idx]

            # Ensure there's actually data to plot
            if plot_times.size > 0 and plot_times[-1] > 0: # Check if last timestamp is valid
                 plot_start_time = plot_times[0] if plot_times[0] > 0 else (plot_times[-1] - PLOT_WINDOW_DURATION_SECONDS) # Estimate start if buffer not full
                 plot_end_time = plot_times[-1]

                 for i in range(NUM_CHANNELS_USED):
                     # Plot only valid data points (where timestamp > 0)
                     # This handles the initial buffer filling phase
                     valid_indices = plot_times > 0
                     eeg_curves[i].setData(x=plot_times[valid_indices], y=plot_data[i, valid_indices])
                     eeg_plots[i].setXRange(plot_start_time, plot_end_time, padding=0)


    # --- 8. Setup Timer and Run ---
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UPDATE_INTERVAL_MS) # Start timer

    print("Starting visualization and recording interface...")
    print("Press the buttons to start/stop recording segments.")
    print("Press Ctrl+C in the console to exit cleanly.")

    # --- Handle Ctrl+C Gracefully ---
    def signal_handler(sig, frame):
        global running
        if running:
            print("\nCtrl+C detected. Stopping...")
            running = False # Signal the update loop to stop

    signal.signal(signal.SIGINT, signal_handler)

    # --- Start Qt Event Loop ---
    print("Starting Qt event loop...")
    exit_code = app.exec_()
    print(f"Qt event loop finished with exit code {exit_code}.")

    # Final check to ensure file is closed if window was closed manually
    if output_file and not output_file.closed:
        print("Closing output file due to application exit.")
        output_file.close()


# --- Main Execution ---
if __name__ == "__main__":
    print("="*50)
    print(" EEG Real-time Data Recording ".center(50, "="))
    print("="*50)

    # Optional: Add argument parsing here later if needed for config files etc.

    run_realtime_recording()

    print("\nScript finished.")