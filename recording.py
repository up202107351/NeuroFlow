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
# Important: Install pylsl: pip install pylsl
import pylsl

# Filtering (Optional, requires brainflow: pip install brainflow)
# You can comment this out if you don't have/want brainflow just for filtering
try:
    from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("Warning: brainflow library not found. Filtering will be disabled.")

# Visualization and Control
# Important: Install PyQt5 and pyqtgraph: pip install PyQt5 pyqtgraph
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- Configuration ---

# LSL Stream Configuration (Targeting BlueMuse Output)
# LSL_STREAM_NAME = 'BlueMuse' # Name might vary, type is usually more stable
LSL_STREAM_TYPE = 'EEG'      # Standard type for EEG streams
LSL_RESOLVE_TIMEOUT = 5      # Seconds to wait for stream discovery
LSL_CHUNK_MAX = 10           # Max samples to pull per iteration (small for low latency viz)

# Recording Parameters (VERIFY these match your BlueMuse LSL stream settings)
ASSUMED_SAMPLE_RATE = 256.0  # Hz (Common for Muse 2/S, CHECK YOUR SOURCE)
PLOT_WINDOW_DURATION_SECONDS = 5 # How many seconds to display in EEG plot
BUFFER_DURATION_SECONDS = 10   # How much data to keep in memory for plotting/filtering

# Channel indices from LSL stream to use (0-based index).
# For Muse 2/S via BlueMuse/MuseLSL, EEG channels are typically the first 4.
EEG_CHANNEL_INDICES = [0, 1, 2, 3] # Default for TP9, AF7, AF8, TP10 from LSL
NUM_CHANNELS_USED = len(EEG_CHANNEL_INDICES)

# Channel names for the CSV header and plot labels (MUST match order of EEG_CHANNEL_INDICES)
CHANNEL_NAMES = ['TP9', 'AF7', 'AF8', 'TP10'] # Common Muse names corresponding to indices [0,1,2,3]
if len(CHANNEL_NAMES) != NUM_CHANNELS_USED:
    raise ValueError("Length of CHANNEL_NAMES must match length of EEG_CHANNEL_INDICES")

# Directory where the recorded data will be saved
SAVE_DIR = Path('./eeg_recordings_lsl') # Creates a folder in the script's directory
SAVE_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

# Filtering parameters (Optional: For visualization/basic cleaning during recording)
# Only active if brainflow library is installed. Set to None/0 to disable specific filters.
ENABLE_FILTERING = True and BRAINFLOW_AVAILABLE # Master switch for filtering
FILTER_ORDER = 5               # Filter order (Butterworth)
NOTCH_CENTER_FREQ = 60.0       # Hz (Adjust to 50 Hz for Europe/Asia, 60Hz for N. America)
NOTCH_BANDWIDTH = 4.0        # Hz (Wider might be needed for BrainFlow's Butterworth notch)
BAND_PASS_LOW_CUTOFF = 1.0     # Hz (High-pass cutoff, set to 0 or None to disable)
BAND_PASS_HIGH_CUTOFF = 45.0   # Hz (Low-pass cutoff, set to 0 or None to disable)

# Visualization settings
UPDATE_INTERVAL_MS = 40 # Update plot roughly every 40ms (~25 FPS)
Y_RANGE_EEG = [-80, 80] # Plot Y-axis range in uV (adjust based on signal amplitude)

# --- Automatically calculated ---
# We will try to get the actual sample rate from LSL, but use ASSUMED_SAMPLE_RATE for initial buffer sizing
SAMPLES_PER_PLOT_WINDOW = int(ASSUMED_SAMPLE_RATE * PLOT_WINDOW_DURATION_SECONDS)
BUFFER_SAMPLES = int(ASSUMED_SAMPLE_RATE * BUFFER_DURATION_SECONDS)
FS_INT = int(ASSUMED_SAMPLE_RATE) # Integer sample rate for filters (will update if possible)

# --- Global Variables ---
running = True # Flag to control main loop and signal exit
lsl_inlet = None
actual_fs = ASSUMED_SAMPLE_RATE # Will be updated from LSL stream info if possible
data_buffer = None
timestamp_buffer = None
buffer_samples_filled = 0
current_state = "Idle" # Can be 'Idle', 'Relaxing', 'Focusing', 'Baseline'
csv_writer = None
output_file = None
start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = SAVE_DIR / f"eeg_recording_{start_time_str}.csv"
app = None # Hold the QApplication instance

# --- LSL Connection Function ---
def connect_lsl(stream_type=LSL_STREAM_TYPE, stream_name=None, timeout=LSL_RESOLVE_TIMEOUT):
    """Attempts to connect to the LSL stream by type or name."""
    global actual_fs, FS_INT, SAMPLES_PER_PLOT_WINDOW, BUFFER_SAMPLES # Allow updating globals

    print(f"Looking for LSL stream (Type: '{stream_type}', Name: '{stream_name}')...")
    streams = []
    if stream_type:
        try:
            streams = pylsl.resolve_byprop('type', stream_type, 1, timeout=timeout)
        except Exception as e:
            print(f"Error resolving by type: {e}")
            streams = [] # Ensure streams is empty list on error

    if not streams and stream_name: # If not found by type, try by name
        print(f"Stream type '{stream_type}' not found, trying name '{stream_name}'...")
        try:
            streams = pylsl.resolve_byprop('name', stream_name, 1, timeout=timeout)
        except Exception as e:
            print(f"Error resolving by name: {e}")
            streams = []

    if not streams:
        errmsg = f"Could not find LSL stream (Type: '{stream_type}', Name: '{stream_name}').\n" \
                 f"Ensure BlueMuse (or other LSL source) is running and streaming."
        print(errmsg)
        return None, None, errmsg # Failure

    print("LSL stream found. Connecting...")
    try:
        inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX)
        stream_info = inlet.info()
        lsl_actual_sample_rate = stream_info.nominal_srate()
        num_lsl_channels = stream_info.channel_count()
        lsl_name = stream_info.name()
        lsl_type = stream_info.type()

        print(f"Connected to '{lsl_name}' (Type: {lsl_type})")
        print(f"  Actual Sample Rate from LSL: {lsl_actual_sample_rate:.2f} Hz")
        print(f"  Number of Channels in LSL stream: {num_lsl_channels}")

        # --- Update sample rate based on LSL info ---
        if lsl_actual_sample_rate > 0:
            if abs(lsl_actual_sample_rate - ASSUMED_SAMPLE_RATE) > 1.0:
                warnings.warn(
                    f"LSL sample rate ({lsl_actual_sample_rate:.2f} Hz) differs significantly "
                    f"from assumed rate ({ASSUMED_SAMPLE_RATE:.2f} Hz). Using LSL rate.", UserWarning
                )
            actual_fs = lsl_actual_sample_rate
            FS_INT = int(actual_fs)
            # Recalculate buffer sizes based on actual rate
            SAMPLES_PER_PLOT_WINDOW = int(actual_fs * PLOT_WINDOW_DURATION_SECONDS)
            BUFFER_SAMPLES = int(actual_fs * BUFFER_DURATION_SECONDS)
            print(f"  Updated plot window to {SAMPLES_PER_PLOT_WINDOW} samples.")
            print(f"  Updated buffer size to {BUFFER_SAMPLES} samples.")
        else:
            print(f"  Warning: LSL nominal rate is 0. Using assumed rate: {ASSUMED_SAMPLE_RATE:.2f} Hz")
            actual_fs = ASSUMED_SAMPLE_RATE
            FS_INT = int(actual_fs)
            # Use initially calculated buffer sizes

        # Validation
        if num_lsl_channels < max(EEG_CHANNEL_INDICES) + 1:
             raise ValueError(f"LSL stream has only {num_lsl_channels} channels, "
                              f"script requires indices up to {max(EEG_CHANNEL_INDICES)}.")

        print(f"Using channels (0-based indices): {EEG_CHANNEL_INDICES}")
        print(f"Corresponding channel names: {CHANNEL_NAMES}")
        return inlet, actual_fs, None # Success

    except Exception as e:
        error_message = f"ERROR connecting to or processing LSL stream info: {e}"
        print(error_message)
        return None, None, error_message # Failure


# --- Main Recording and Visualization Function ---
def run_realtime_recording():
    """
    Connects to LSL, visualizes EEG, and records data segments with labels.
    """
    global running, lsl_inlet, data_buffer, timestamp_buffer, buffer_samples_filled
    global current_state, csv_writer, output_file, output_filename, app

    print("--- Starting Real-time EEG Recording via LSL ---")
    print(f"Looking for LSL stream from BlueMuse or similar...")
    print(f"Data will be saved to: {output_filename}")

    # --- 1. Attempt LSL Connection ---
    # Try connecting by type first, then by a potential name if type fails
    lsl_inlet, actual_fs_conn, connection_error_msg = connect_lsl(stream_type=LSL_STREAM_TYPE, stream_name='BlueMuse')
    lsl_connected = lsl_inlet is not None

    # --- 2. Setup PyQtGraph Application ---
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Real-time EEG Recording (LSL)", size=(1000, 700))
    win.setBackground('k') # Black background

    # --- 3. Create Plot Areas ---
    eeg_plots = []
    eeg_curves = []
    plot_layout = win.addLayout(row=0, col=0, rowspan=NUM_CHANNELS_USED + 1) # Span rows + 1 for potential title
    # Optional Title Item
    # title_item = plot_layout.addLabel("Real-time EEG", row=0, col=0, size='14pt', bold=True, color='k')
    # plot_layout.nextRow() # Move to next row for plots

    for i in range(NUM_CHANNELS_USED):
        p = plot_layout.addPlot(row=i+1, col=0) # Start plots from row 1 if title is used
        p.setYRange(Y_RANGE_EEG[0], Y_RANGE_EEG[1])
        p.setLabel('left', CHANNEL_NAMES[i], units='uV', color='k', **{'font-size':'10pt'})
        p.getAxis('left').setTextPen('k')
        p.getAxis('bottom').setTextPen('k')
        p.showGrid(x=True, y=True, alpha=0.3)
        if i < NUM_CHANNELS_USED - 1:
            p.hideAxis('bottom')
        else:
            p.setLabel('bottom', "Time (s)", color='k', **{'font-size':'10pt'})
        curve = p.plot(pen=pg.mkPen(color=pg.intColor(i, hues=NUM_CHANNELS_USED, sat=200), width=1.5))
        eeg_plots.append(p)
        eeg_curves.append(curve)
        # Link X axes for synchronized scrolling/zooming
        if i > 0:
            p.setXLink(eeg_plots[0])

    # --- 4. Create Control Panel (Right Side) ---
    control_layout = win.addLayout(row=0, col=1, rowspan=NUM_CHANNELS_USED + 1)

    # Instruction/Status Label
    status_label = QtWidgets.QLabel("Connecting...")
    status_label.setAlignment(QtCore.Qt.AlignCenter)
    status_label.setWordWrap(True)
    status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: black; padding: 10px;")
    proxy_status = QtWidgets.QGraphicsProxyWidget()
    proxy_status.setWidget(status_label)
    control_layout.addItem(proxy_status, row=0, col=0) # Top row

    # Recording Buttons
    button_layout = QtWidgets.QWidget()
    button_grid = QtWidgets.QGridLayout(button_layout)
    button_grid.setSpacing(10)

    btn_relax = QtWidgets.QPushButton("Start Relaxing")
    btn_focus = QtWidgets.QPushButton("Start Focusing")
    btn_baseline = QtWidgets.QPushButton("Start Baseline")
    btn_stop = QtWidgets.QPushButton("Stop Recording")

    button_style = "QPushButton { font-size: 12pt; padding: 10px; } " \
                   "QPushButton:!enabled { background-color: #d3d3d3; }"
    btn_relax.setStyleSheet(button_style)
    btn_focus.setStyleSheet(button_style)
    btn_baseline.setStyleSheet(button_style)
    btn_stop.setStyleSheet(button_style + "QPushButton { background-color: #FF8C8C; color: black; }") # Make stop red-ish

    button_grid.addWidget(btn_relax, 0, 0)
    button_grid.addWidget(btn_focus, 1, 0)
    button_grid.addWidget(btn_baseline, 2, 0)
    button_grid.addWidget(btn_stop, 4, 0) # Add space before stop button
    #button_grid.setRowStretch(3, 1) # Add stretchable space

    proxy_buttons = QtWidgets.QGraphicsProxyWidget()
    proxy_buttons.setWidget(button_layout)
    control_layout.addItem(proxy_buttons, row=1, col=0) # Second row

    # Filename Label
    filename_label = QtWidgets.QLabel(f"Saving to:\n{output_filename.name}")
    filename_label.setAlignment(QtCore.Qt.AlignCenter)
    filename_label.setWordWrap(True)
    filename_label.setStyleSheet("font-size: 10pt; color: grey; margin-top: 15px;")
    proxy_filename = QtWidgets.QGraphicsProxyWidget()
    proxy_filename.setWidget(filename_label)
    control_layout.addItem(proxy_filename, row=2, col=0) # Below buttons

    #control_layout.setRowStretch(3, 1) # Push controls towards top

    # Set column widths (give more space to plot)
    win.ci.layout.setColumnStretchFactor(0, 3) # Plot area takes 3/4 width
    win.ci.layout.setColumnStretchFactor(1, 1) # Control panel takes 1/4 width

    # --- Show the window ---
    win.show()
    win.setWindowTitle(f"EEG Recording LSL - {output_filename.name}")

    # --- 5. Initialize Buffers and State (Only if connected) ---
    if lsl_connected:
        # Use BUFFER_SAMPLES calculated based on actual_fs if possible
        data_buffer = np.zeros((NUM_CHANNELS_USED, BUFFER_SAMPLES))
        timestamp_buffer = np.zeros(BUFFER_SAMPLES)
        buffer_samples_filled = 0
        status_label.setText("Connected to LSL.\nReady to Record.")
        status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green; padding: 10px;")
        print("Buffers initialized based on detected or assumed sample rate.")
        btn_relax.setEnabled(True)
        btn_focus.setEnabled(True)
        btn_baseline.setEnabled(True)
        btn_stop.setEnabled(False) # Stop disabled initially
    else:
        data_buffer = None
        timestamp_buffer = None
        buffer_samples_filled = -1 # Signify not ready/connected
        status_label.setText(f"LSL Connection Failed:\n{connection_error_msg}")
        status_label.setStyleSheet("font-size: 12pt; color: red; padding: 10px;")
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
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Warning)
            msgBox.setText(f"Already recording '{current_state}'.\nPlease stop the current recording before starting a new one.")
            msgBox.setWindowTitle("Recording Active")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msgBox.exec_()
            return

        print(f"Starting recording: {label}")
        current_state = label
        status_label.setText(f"RECORDING:\n{label}")
        status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: red; padding: 10px; background-color: #FFEEF0;")
        # Disable start buttons, enable stop
        btn_relax.setEnabled(False)
        btn_focus.setEnabled(False)
        btn_baseline.setEnabled(False)
        btn_stop.setEnabled(True)

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
             status_label.setText(f"File Error:\n{e}")
             status_label.setStyleSheet("font-size: 12pt; color: red; padding: 10px;")
             current_state = "Idle" # Reset state
             if output_file:
                 output_file.close()
                 output_file = None
             csv_writer = None
             # Re-enable start buttons, disable stop
             btn_relax.setEnabled(True); btn_focus.setEnabled(True); btn_baseline.setEnabled(True)
             btn_stop.setEnabled(False)


    def stop_recording():
        global current_state, csv_writer, output_file
        if current_state == "Idle":
            print("Not currently recording.")
            return

        print(f"Stopping recording: {current_state}")
        last_state = current_state
        current_state = "Idle"
        status_label.setText(f"Stopped {last_state}.\nReady to Record.")
        status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green; padding: 10px;")
        # Re-enable start buttons, disable stop
        btn_relax.setEnabled(True)
        btn_focus.setEnabled(True)
        btn_baseline.setEnabled(True)
        btn_stop.setEnabled(False)

        # Close the CSV file
        if output_file:
            try:
                output_file.flush() # Ensure all buffered data is written
                os.fsync(output_file.fileno()) # Force write to disk
                output_file.close()
                print(f"Data segment saved to {output_filename}")
            except Exception as e:
                print(f"Error closing file: {e}")
        output_file = None
        csv_writer = None

    # Connect buttons to actions
    btn_relax.clicked.connect(lambda: start_recording("Relaxing"))
    btn_focus.clicked.connect(lambda: start_recording("Focusing"))
    btn_baseline.clicked.connect(lambda: start_recording("Baseline"))
    btn_stop.clicked.connect(stop_recording)


    # --- 7. Update Function (Called by Timer) ---
    def update():
        global data_buffer, timestamp_buffer, buffer_samples_filled
        global running, current_state, csv_writer

        if not running: # Check exit flag
            # Cleanup is handled by signal_handler or cleanup_on_close
            return

        # Only process LSL if connected
        if not lsl_connected or lsl_inlet is None:
            return # Should not happen if buttons are disabled, but good practice

        new_samples_count = 0
        try:
            # --- Pull Data from LSL ---
            chunk, ts = lsl_inlet.pull_chunk(timeout=0.0, max_samples=LSL_CHUNK_MAX)
            new_samples_count = len(ts)

        except pylsl.LostError as e:
            print(f"LSL connection lost: {e}. Stopping...")
            running = False # Signal exit
            status_label.setText(f"LSL Connection Lost!\n{e}")
            status_label.setStyleSheet("font-size: 12pt; color: red; padding: 10px;")
            # Disable all buttons
            btn_relax.setEnabled(False); btn_focus.setEnabled(False)
            btn_baseline.setEnabled(False); btn_stop.setEnabled(False)
            # Attempt cleanup immediately
            cleanup()
            return
        except Exception as e:
             print(f"Error pulling LSL chunk: {e}. Stopping.")
             running = False # Signal exit
             status_label.setText(f"LSL Pull Error:\n{e}")
             status_label.setStyleSheet("font-size: 12pt; color: red; padding: 10px;")
             # Disable all buttons
             btn_relax.setEnabled(False); btn_focus.setEnabled(False)
             btn_baseline.setEnabled(False); btn_stop.setEnabled(False)
             # Attempt cleanup immediately
             cleanup()
             return

        if new_samples_count > 0:
            # Convert chunk to numpy array and select configured channels
            # Make sure chunk is treated as list of lists/tuples before converting
            chunk_list = [list(s) for s in chunk]
            chunk_np_all = np.array(chunk_list, dtype=np.float64).T # Transpose for (channels x samples)

            # Select only the desired EEG channels
            chunk_np = chunk_np_all[EEG_CHANNEL_INDICES, :]

            # --- Optional: Filter incoming chunk ---
            # Apply filters *before* adding to buffer if desired for visualization
            # Note: Data saved to CSV will also be filtered if enabled here.
            # If you want to save RAW data, pull raw data first, save it, then filter a copy for plotting.
            if ENABLE_FILTERING and BRAINFLOW_AVAILABLE:
                try:
                    for i in range(NUM_CHANNELS_USED):
                        # Bandpass Filter (apply first)
                        if BAND_PASS_LOW_CUTOFF is not None and BAND_PASS_HIGH_CUTOFF is not None and BAND_PASS_LOW_CUTOFF > 0 and BAND_PASS_HIGH_CUTOFF > 0 :
                             DataFilter.perform_bandpass(chunk_np[i, :], FS_INT, BAND_PASS_LOW_CUTOFF, BAND_PASS_HIGH_CUTOFF, FILTER_ORDER, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                        # Notch Filter (Powerline Noise)
                        if NOTCH_CENTER_FREQ is not None and NOTCH_CENTER_FREQ > 0:
                            DataFilter.perform_bandstop(chunk_np[i, :], FS_INT, NOTCH_CENTER_FREQ, NOTCH_BANDWIDTH, FILTER_ORDER, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                except Exception as filter_e:
                    # Don't stop recording for filter errors, just log them once
                    if not hasattr(update, 'filter_error_logged'): # Log only once
                         print(f"Warning: Filtering error on chunk (will not log again): {filter_e}")
                         update.filter_error_logged = True
                    # Continue with potentially unfiltered or partially filtered data


            # --- Update Plotting/Recording Buffers ---
            # Roll buffers, add new data at the end
            data_buffer = np.roll(data_buffer, -new_samples_count, axis=1)
            timestamp_buffer = np.roll(timestamp_buffer, -new_samples_count)
            data_buffer[:, -new_samples_count:] = chunk_np
            timestamp_buffer[-new_samples_count:] = ts
            buffer_samples_filled = min(BUFFER_SAMPLES, buffer_samples_filled + new_samples_count)

            # --- Write to CSV if Recording ---
            if current_state != "Idle" and csv_writer and output_file:
                try:
                    for sample_idx in range(new_samples_count):
                        timestamp = ts[sample_idx]
                        # Data for this sample from the (potentially filtered) chunk
                        sample_data = chunk_np[:, sample_idx].tolist()
                        row = [timestamp] + sample_data + [current_state]
                        csv_writer.writerow(row)
                except Exception as e:
                    print(f"ERROR writing data to CSV: {e}. Stopping recording.")
                    stop_recording() # Stop current recording segment on write error
                    status_label.setText(f"CSV Write Error:\n{e}")
                    status_label.setStyleSheet("font-size: 12pt; color: red; padding: 10px;")


        # --- Update EEG Plot ---
        # Use only the most recent samples from the buffer for plotting
        plot_start_idx = max(0, buffer_samples_filled - SAMPLES_PER_PLOT_WINDOW)
        plot_end_idx = buffer_samples_filled

        # Get valid data from the *end* of the buffer
        valid_buffer_eeg = data_buffer[:, -buffer_samples_filled:]
        valid_buffer_ts = timestamp_buffer[-buffer_samples_filled:]

        plot_data = valid_buffer_eeg[:, plot_start_idx:plot_end_idx]
        plot_times = valid_buffer_ts[plot_start_idx:plot_end_idx]

        # Ensure there's actually data to plot and timestamps are valid (>0)
        if plot_times.size > 0 and plot_times[-1] > 0:
             # Use actual timestamps for X-axis range if available
             # Estimate start time if buffer isn't full yet based on window duration
             plot_start_time = plot_times[0] if plot_times[0] > 0 else (plot_times[-1] - PLOT_WINDOW_DURATION_SECONDS)
             plot_end_time = plot_times[-1]

             for i in range(NUM_CHANNELS_USED):
                 # Only plot where timestamp is valid (greater than 0)
                 valid_indices = plot_times > 0
                 if np.any(valid_indices): # Check if there are any valid points
                    eeg_curves[i].setData(x=plot_times[valid_indices], y=plot_data[i, valid_indices])
                    # Update range only if needed, prevents jitter
                    # current_range = eeg_plots[i].getViewBox().viewRange()[0]
                    # if abs(current_range[0] - plot_start_time) > 0.1 or abs(current_range[1] - plot_end_time) > 0.1:
                    eeg_plots[i].setXRange(plot_start_time, plot_end_time, padding=0.01)


    # --- 8. Cleanup Function ---
    def cleanup():
        global running, lsl_inlet, output_file, app
        print("Initiating cleanup...")
        running = False # Ensure update loop stops

        if current_state != "Idle": # Ensure data is saved if recording during shutdown
            print("Stopping active recording before exit...")
            stop_recording()

        if lsl_inlet:
            try:
                print("Closing LSL stream...")
                lsl_inlet.close_stream()
                print("LSL stream closed.")
            except Exception as e:
                print(f"Error closing LSL stream: {e}")
            lsl_inlet = None # Prevent trying to close again

        # File is closed by stop_recording, but double-check
        if output_file and not output_file.closed:
             print("Warning: Output file was still open during cleanup. Closing.")
             try:
                 output_file.close()
             except Exception as e:
                 print(f"Error during final file close: {e}")

        if app:
            print("Quitting Qt application...")
            app.quit()
        print("Cleanup finished.")


    # --- 9. Setup Timer and Run ---
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UPDATE_INTERVAL_MS) # Start timer

    print("Starting visualization and recording interface...")
    print(f"Using LSL stream type: {LSL_STREAM_TYPE}")
    print(f"Assumed/Detected Sample Rate: {actual_fs:.2f} Hz")
    print(f"Saving data segments to: {output_filename}")
    print("Press the buttons to start/stop recording segments.")
    print("Press Ctrl+C in the console OR close the window to exit cleanly.")

    # --- Handle Ctrl+C Gracefully ---
    def signal_handler(sig, frame):
        global running
        if running: # Prevent double execution if called rapidly
            print("\nCtrl+C detected. Stopping...")
            cleanup() # Call the main cleanup function

    signal.signal(signal.SIGINT, signal_handler)

    # --- Handle Window Close Gracefully ---
    def cleanup_on_close(event):
        global running
        if running: # Prevent double execution
            print("Window close detected. Stopping...")
            cleanup() # Call the main cleanup function
        event.accept() # Allow window to close

    win.closeEvent = cleanup_on_close # Override close event

    # --- Start Qt Event Loop ---
    print("Starting Qt event loop...")
    exit_code = app.exec_()
    print(f"Qt event loop finished with exit code {exit_code}.")


# --- Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print(" EEG Real-time Data Recording & Visualization via LSL ".center(60, "="))
    print("="*60)
    print(f"Using PyLSL version: {pylsl.__version__}")
    print(f"Using PyQtGraph version: {pg.__version__}")
    print(f"BrainFlow filtering available: {BRAINFLOW_AVAILABLE}")
    print(f"Script PID: {os.getpid()}")
    print("-" * 60)

    # Add configuration check maybe?
    print("Configuration:")
    print(f"  LSL Stream Type: {LSL_STREAM_TYPE}")
    print(f"  LSL Channel Indices: {EEG_CHANNEL_INDICES}")
    print(f"  Channel Names: {CHANNEL_NAMES}")
    print(f"  Sample Rate (Assumed/Detected): {actual_fs:.2f} Hz")
    print(f"  Save Directory: {SAVE_DIR}")
    print(f"  Filtering Enabled: {ENABLE_FILTERING and BRAINFLOW_AVAILABLE}")
    if ENABLE_FILTERING and BRAINFLOW_AVAILABLE:
        print(f"    Notch: {NOTCH_CENTER_FREQ} Hz +/- {NOTCH_BANDWIDTH/2.0} Hz")
        print(f"    Bandpass: {BAND_PASS_LOW_CUTOFF} Hz - {BAND_PASS_HIGH_CUTOFF} Hz")
    print("-" * 60)
    print("Ensure BlueMuse (or another LSL provider) is running and streaming.")
    print("-" * 60)

    run_realtime_recording()

    print("\nScript finished.")