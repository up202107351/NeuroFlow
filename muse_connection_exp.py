import time
import os
from pathlib import Path
import sys
import warnings
import joblib
import signal # To handle Ctrl+C gracefully

# Data Handling
import numpy as np
import pandas as pd

# LSL Streaming
import pylsl

# Filtering (from BrainFlow)
from brainflow.data_filter import DataFilter, FilterTypes

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# >> If you saved an SVM model, uncomment the next line and the relevant loading/prediction code <<
# from sklearn.svm import SVC

# Visualization
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- Configuration ---

# LSL Stream Configuration
LSL_STREAM_NAME = 'BlueMuse' # Name of the LSL stream broadcast by BlueMuse (Check BlueMuse settings)
# LSL_STREAM_TYPE = 'EEG' # Alternatively, use type if name is unstable
LSL_CHUNK_MAX = 10 # Max samples to pull per iteration (affects responsiveness)

# !! CRITICAL: Use the SAME settings assumed during training !!
ASSUMED_SAMPLE_RATE = 256.0 # Hz (MUST MATCH BlueMuse LSL output and training)
SEGMENT_DURATION_SECONDS = 5 # Duration for analysis/prediction
PLOT_WINDOW_DURATION_SECONDS = 3 # How many seconds to display in EEG plot
BUFFER_DURATION_SECONDS = 10 # How much data to keep in memory (should be >= SEGMENT_DURATION)

# --- Automatically calculated ---
SAMPLES_PER_SEGMENT = int(ASSUMED_SAMPLE_RATE * SEGMENT_DURATION_SECONDS)
SAMPLES_PER_PLOT_WINDOW = int(ASSUMED_SAMPLE_RATE * PLOT_WINDOW_DURATION_SECONDS)
BUFFER_SAMPLES = int(ASSUMED_SAMPLE_RATE * BUFFER_DURATION_SECONDS)

# Channel indices from LSL stream to use (MUST MATCH TRAINING - BlueMuse usually TP9, AF7, AF8, TP10)
# Ensure these are 0-based indices corresponding to the LSL stream's channels.
EEG_CHANNEL_INDICES = [0, 1, 2, 3]
NUM_CHANNELS_USED = len(EEG_CHANNEL_INDICES)

# Directory where the trained models/objects were saved
LOAD_DIR = Path(r'C:\Users\Utilizador\OneDrive\Documentos\GitHub\NeuroFlow\saved_models')

# Filtering parameters (MUST MATCH TRAINING SCRIPT)
FILTER_ORDER = 4
BAND_PASS_LOW_CUTOFF = 0.1 # Hz
BAND_PASS_HIGH_CUTOFF = 50 # Hz

# Brainwave Band Definitions (Hz) (MUST MATCH TRAINING SCRIPT)
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)
GAMMA_BAND = (30, 50)

# Visualization settings
UPDATE_INTERVAL_MS = 40 # Update plot roughly every 40ms (~25 FPS)
# How often to run classification (in ms). Less frequent than plotting.
CLASSIFICATION_INTERVAL_MS = 1000 # Classify every 1 second

# Global variable to hold the final histogram window (to prevent garbage collection)
final_hist_window = None
# Global flag to signal exit
running = True

# --- Helper Functions (BrainwaveBins - same as before) ---
def BrainwaveBins(channel_data, sample_rate):
    """Calculates average power in standard EEG bands for a single channel."""
    # (Code is identical to the previous scripts - kept for brevity)
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0: return [0.0] * 5
    if channel_data is None or len(channel_data) == 0: return [0.0] * 5
    n_samples = len(channel_data)
    # Need enough samples for FFT resolution, especially for low frequencies
    min_samples = int(sample_rate / DELTA_BAND[0]) # e.g. 256 / 0.5 = 512 for Delta
    if n_samples < min_samples :
        # print(f"Warning: Not enough samples ({n_samples}) for reliable low freq bins (need ~{min_samples}).")
        # Pad with zeros? Or just return zeros? Returning zeros is safer.
        return [0.0] * 5
    try:
        fft_vals = np.fft.fft(channel_data)
        fft_freq = np.fft.fftfreq(n_samples, d=1.0/sample_rate)
        pos_freq_indices = np.where(fft_freq > 0)[0]
        if len(pos_freq_indices) == 0: return [0.0] * 5
        freqs = fft_freq[pos_freq_indices]
        fft_vals_pos = fft_vals[pos_freq_indices]
        psd = (np.abs(fft_vals_pos)**2) / n_samples # Basic power estimate
        bands = {'Delta': DELTA_BAND, 'Theta': THETA_BAND, 'Alpha': ALPHA_BAND, 'Beta': BETA_BAND, 'Gamma': GAMMA_BAND}
        band_powers = {name: 0.0 for name in bands}
        band_counts = {name: 0 for name in bands}
        for freq, power in zip(freqs, psd):
            for name, (low, high) in bands.items():
                if low <= freq < high:
                    band_powers[name] += power
                    band_counts[name] += 1
                    break
        avg_powers = []
        for name in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
              avg_powers.append(band_powers[name] / band_counts[name] if band_counts[name] > 0 else 0.0)
        return avg_powers
    except Exception as e:
        # print(f"Error in Bins: {e}") # Debug if needed
        return [0.0] * 5

# --- Function to Show Final Histogram (Can be adapted if needed) ---
# This might not be as relevant in real-time, but kept for potential use
# def show_final_histogram(full_eeg_data, sample_rate): ... (Keep if desired)

# --- LSL Connection Function (Modified slightly to just return status) ---
def connect_lsl(stream_name=LSL_STREAM_NAME, stream_type='EEG', timeout=5):
    print(f"Looking for LSL stream '{stream_name}' (type: {stream_type})...")
    # Increase resolve time slightly if network is slow
    # streams = pylsl.resolve_byprop('name', stream_name, 1, timeout=timeout)
    streams = pylsl.resolve_byprop('type', stream_type, 1, timeout=timeout) # Alternative

    if not streams:
        error_message = f"ERROR: Could not find LSL stream named '{stream_name}'.\nMake sure BlueMuse is running and streaming."
        print(error_message) # Keep console message
        return None, None, error_message # Return None for inlet/rate and the error message

    print("LSL stream found. Connecting...")
    try:
        inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX)
        stream_info = inlet.info()
        actual_sample_rate = stream_info.nominal_srate()
        num_lsl_channels = stream_info.channel_count()

        print(f"Connected to '{stream_info.name()}'")
        print(f"  Actual Sample Rate: {actual_sample_rate:.2f} Hz")
        print(f"  Number of Channels: {num_lsl_channels}")

        # --- Add validation checks here ---
        if abs(actual_sample_rate - ASSUMED_SAMPLE_RATE) > 1.0:
             warnings.warn(f"LSL stream rate ({actual_sample_rate:.2f} Hz) differs significantly from assumed rate ({ASSUMED_SAMPLE_RATE:.2f} Hz).", UserWarning)
        if num_lsl_channels < max(EEG_CHANNEL_INDICES) + 1:
             error_message = f"ERROR: LSL stream has only {num_lsl_channels} channels, script requires indices up to {max(EEG_CHANNEL_INDICES)}."
             print(error_message)
             inlet.close_stream()
             return None, None, error_message

        # If all checks pass
        return inlet, actual_sample_rate, None # Return inlet/rate and None for error message

    except Exception as e:
        error_message = f"ERROR: Failed to create LSL inlet: {e}"
        print(error_message)
        return None, None, error_message


# --- Main Visualization and Prediction Function (Modified) ---
def run_realtime_prediction(model_dir):
    """
    Connects to LSL, visualizes EEG, and predicts state in real-time.
    Shows window even if LSL connection fails.
    """
    global running, final_hist_window
    print("--- Starting Real-time Prediction ---")
    fs_int = int(ASSUMED_SAMPLE_RATE)
    feature_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # --- 1. Load Saved Models and Objects ---
    # This happens first, if models are missing, we exit early.
    print(f"Loading models and preprocessing objects from: {model_dir}")
    try:
        # --- Choose ONE model ---
        #model = joblib.load(model_dir / 'knn_model.joblib')
        #model_name = "KNN"
        model = joblib.load(model_dir / 'logreg_model.joblib')
        model_name = "LogReg"
        # model = joblib.load(model_dir / 'svm_model.joblib')
        # model_name = "SVM"
        # --- End Model Choice ---

        scaler = joblib.load(model_dir / 'scaler.joblib')
        label_encoder = joblib.load(model_dir / 'label_encoder.joblib')
        print(f"{model_name} Model, Scaler, and Label Encoder loaded.")
        print(f"Classes known by model: {label_encoder.classes_}")
    except Exception as e:
        # Use more informative error printing here as well
        print(f"ERROR: Could not load required files from '{model_dir}'.")
        print(f"       Exception Type: {type(e)}")
        print(f"       Exception Message: {e}")
        print("Ensure model file (e.g., 'knn_model.joblib'), 'scaler.joblib', and 'label_encoder.joblib' exist and are compatible.")
        # Exit here if models cannot be loaded, as the rest of the app is useless
        return

    # --- 2. Attempt LSL Connection ---
    inlet, actual_fs, connection_error_msg = connect_lsl()
    lsl_connected = inlet is not None # Flag to indicate connection status

    # --- 3. Setup PyQtGraph Application (Always run this) ---
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Real-time EEG Prediction", size=(1000, 700))

    # --- 4. Create Plot Areas (Always create them) ---
    # EEG Plots (Left Side)
    eeg_plots = []
    eeg_curves = []
    plot_layout = win.addLayout(row=0, col=0)
    # Use NUM_CHANNELS_USED which is derived from EEG_CHANNEL_INDICES
    for i in range(NUM_CHANNELS_USED):
        p = plot_layout.addPlot(row=i, col=0)
        p.setYRange(-500, 200) # Adjust Y range
        p.setLabel('left', f"Ch {EEG_CHANNEL_INDICES[i]}", units='uV')
        p.showGrid(x=True, y=True, alpha=0.3)
        if i < NUM_CHANNELS_USED - 1:
            p.hideAxis('bottom')
        else:
            p.setLabel('bottom', "Time (s)")
        curve = p.plot(pen=pg.mkPen(color=pg.intColor(i, hues=NUM_CHANNELS_USED), width=1))
        eeg_plots.append(p)
        eeg_curves.append(curve)

    # Histogram and Prediction (Right Side)
    right_layout = win.addLayout(row=0, col=1)
    # Prediction Text Label - Initialize based on connection status
    pred_label = right_layout.addLabel("Initializing...", row=0, col=0, size='14pt', bold=True, color='white')
    if not lsl_connected:
        # Set specific error text if connection failed
        pred_label.setText("Couldn't connect to LSL stream")
        pred_label.opts['color'] = 'red' # Make error messages red
    else:
        pred_label.setText("Connecting...") # Initial message if connected

    # Histogram Plot
    hist_plot = right_layout.addPlot(row=1, col=0)
    hist_plot.setTitle("Latest 5s Band Power")
    hist_plot.setLabel('left', "Avg Power (relative)")
    hist_plot.setLabel('bottom', "Frequency Band")
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    x_ticks = [list(enumerate(band_names))]
    hist_plot.getAxis('bottom').setTicks(x_ticks)
    bar_graph = pg.BarGraphItem(x=range(len(band_names)), height=[0]*len(band_names), width=0.6, brush='lightblue')
    hist_plot.addItem(bar_graph)
    hist_plot.setYRange(0, 50) # Example initial range

    # --- Show the window ---
    win.show() # Show window regardless of connection status

    # --- 5. Initialize Buffers (Only if connected) ---
    if lsl_connected:
        data_buffer = np.zeros((NUM_CHANNELS_USED, BUFFER_SAMPLES))
        timestamps = np.zeros(BUFFER_SAMPLES)
        buffer_samples_filled = 0
        print("Buffers initialized.")
    else:
        # Set to None if not connected to avoid errors in update function
        data_buffer = None
        timestamps = None
        buffer_samples_filled = -1 # Use -1 to signify not ready/not connected
        print("LSL not connected. Buffers not initialized.")


    # --- 6. Update Function and Timers ---
    last_classification_time = time.time()
    last_prediction_str = "N/A"

    def update():
        nonlocal data_buffer, timestamps, buffer_samples_filled, last_classification_time, last_prediction_str
        global running, final_hist_window

        if not running: # Check flag to stop updates
             if lsl_connected and inlet: # Check if inlet exists before trying to close
                 try:
                     inlet.close_stream()
                     print("LSL stream closed.")
                 except Exception as e:
                     print(f"Error closing LSL stream: {e}")
             app.quit() # Exit Qt application
             return

        # --- Only process LSL if connected ---
        if lsl_connected:
            # --- Pull Data from LSL ---
            chunk, ts = inlet.pull_chunk(timeout=0.0, max_samples=LSL_CHUNK_MAX)
            new_samples_count = len(ts)

            if new_samples_count > 0:
                # Convert chunk to numpy array and select channels
                chunk_np = np.array(chunk)[:, EEG_CHANNEL_INDICES].T
                chunk_np = chunk_np.astype(np.float64)

                # --- Optional: Filter incoming chunk ---
                for i in range(NUM_CHANNELS_USED):
                    try:
                         DataFilter.perform_bandpass(chunk_np[i], fs_int, start_freq=BAND_PASS_LOW_CUTOFF, stop_freq=BAND_PASS_HIGH_CUTOFF, order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH, ripple=0)
                         DataFilter.perform_bandstop(chunk_np[i], fs_int, start_freq=49, stop_freq=51, order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH, ripple=0)
                    except Exception: pass # Ignore filtering errors on small chunks

                # --- Update Buffers ---
                data_buffer = np.roll(data_buffer, -new_samples_count, axis=1)
                timestamps = np.roll(timestamps, -new_samples_count)
                data_buffer[:, -new_samples_count:] = chunk_np
                timestamps[-new_samples_count:] = ts
                buffer_samples_filled = min(BUFFER_SAMPLES, buffer_samples_filled + new_samples_count)

            # --- Update EEG Plot ---
            plot_start_idx = max(0, BUFFER_SAMPLES - SAMPLES_PER_PLOT_WINDOW)
            plot_data = data_buffer[:, plot_start_idx:]
            plot_times = timestamps[plot_start_idx:]
            valid_plot_times = plot_times[plot_times > 0]
            if len(valid_plot_times) > 1:
                 plot_start_time = valid_plot_times[0]
                 plot_end_time = valid_plot_times[-1]
                 for i in range(NUM_CHANNELS_USED):
                     valid_data = plot_data[i, plot_times > 0]
                     eeg_curves[i].setData(x=valid_plot_times, y=valid_data)
                     eeg_plots[i].setXRange(plot_start_time, plot_end_time, padding=0)

            # --- Classification (Run periodically) ---
            current_time = time.time()
            if current_time - last_classification_time >= (CLASSIFICATION_INTERVAL_MS / 1000.0):
                last_classification_time = current_time

                if buffer_samples_filled >= SAMPLES_PER_SEGMENT:
                    segment_data = data_buffer[:, -SAMPLES_PER_SEGMENT:]
                    # --- Optional: Filter 5s segment if not filtering chunks ---
                    # ... filtering logic ...

                    # Extract Features
                    all_channel_bins = []
                    valid_channel_bins = 0
                    for chan_idx in range(NUM_CHANNELS_USED):
                         channel_bins = BrainwaveBins(segment_data[chan_idx, :].copy(), ASSUMED_SAMPLE_RATE)
                         if sum(channel_bins) > 1e-9:
                              all_channel_bins.append(channel_bins)
                              valid_channel_bins += 1

                    if valid_channel_bins > NUM_CHANNELS_USED / 2 and all_channel_bins:
                         features_list = np.mean(all_channel_bins, axis=0).tolist()
                         features_df = pd.DataFrame([features_list], columns=feature_names)

                         # Update Histogram
                         bar_graph.setOpts(height=features_list)
                         # Optional: hist_plot.enableAutoRange(axis='y', enable=True)

                         # Scale and Predict
                         try:
                              scaled_features = scaler.transform(features_df)
                              pred_idx = model.predict(scaled_features)[0]
                              last_prediction_str = label_encoder.inverse_transform([pred_idx])[0]
                         except Exception as pred_e:
                              print(f"Prediction error: {pred_e}")
                              last_prediction_str = 'Error'
                    else:
                         # Not enough valid channels or features
                         # last_prediction_str = 'No Features' # Keep last prediction?
                         pass
                else:
                     # Not enough data in buffer yet
                     last_prediction_str = f"Buffering... ({buffer_samples_filled}/{SAMPLES_PER_SEGMENT})"

                # Update prediction label text only if connected
                pred_label.setText(f"{model_name}: {last_prediction_str}")
                # Reset color if it was red previously
                if pred_label.opts['color'] == 'red':
                     pred_label.opts['color'] = 'white' # Set back to default

        # --- End of "if lsl_connected:" block ---
        else:
            # If LSL was never connected, the error message remains in pred_label
            pass # Do nothing in the update loop


    # --- 7. Setup Timers and Run ---
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UPDATE_INTERVAL_MS) # Start timer even if not connected

    print("Starting visualization... Press Ctrl+C in the console to exit.")

    # --- Handle Ctrl+C Gracefully ---
    def signal_handler(sig, frame):
        global running
        if running: # Prevent multiple calls
             print("\nCtrl+C detected. Stopping...")
             running = False
             # Don't quit app here, let the update loop handle it
    signal.signal(signal.SIGINT, signal_handler)

    # --- Start Qt Event Loop ---
    print("Starting Qt event loop...")
    app_instance = QtWidgets.QApplication.instance()
    exit_code = app_instance.exec_()
    print(f"Qt event loop finished with exit code {exit_code}.")


# --- Main Execution (No changes needed here) ---
if __name__ == "__main__":
    print("="*50)
    print(" EEG Real-time Classification ".center(50, "="))
    print("="*50)

    if not LOAD_DIR.is_dir():
        print(f"Error: Saved model directory not found at {LOAD_DIR}")
        sys.exit(1)

    run_realtime_prediction(LOAD_DIR)

    print("\nScript finished.")