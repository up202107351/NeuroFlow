import time
import os
from pathlib import Path
import sys
import warnings
import joblib

# Data Handling
import numpy as np
import pandas as pd

# Filtering (from BrainFlow)
from brainflow.data_filter import DataFilter, FilterTypes

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Visualization
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- Configuration ---

# !! REQUIRED: Path to the NEW CSV file to visualize/classify !!
NEW_DATA_CSV_PATH = Path('Brainwaves_Relaxed_1.csv') # <--- CHANGE THIS

# !! CRITICAL: Use the SAME settings assumed during training !!
ASSUMED_SAMPLE_RATE = 256.0 # Hz
SEGMENT_DURATION_SECONDS = 5 # Duration for analysis/prediction
PLOT_WINDOW_DURATION_SECONDS = 3 # How many seconds to display in EEG plot
# --- Automatically calculated ---
SAMPLES_PER_SEGMENT = int(ASSUMED_SAMPLE_RATE * SEGMENT_DURATION_SECONDS)
SAMPLES_PER_PLOT_WINDOW = int(ASSUMED_SAMPLE_RATE * PLOT_WINDOW_DURATION_SECONDS)

# Channel indices from CSV (MUST MATCH TRAINING SCRIPT)
EEG_CHANNEL_INDICES = [0, 1, 2, 3]

# Directory where the trained models/objects were saved
LOAD_DIR = Path('saved_models')

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
SAMPLES_PER_UPDATE = int(ASSUMED_SAMPLE_RATE * (UPDATE_INTERVAL_MS / 1000.0))

# Global variable to hold the final histogram window (to prevent garbage collection)
final_hist_window = None

# --- Helper Functions (BrainwaveBins - same as before) ---
def BrainwaveBins(channel_data, sample_rate):
    """Calculates average power in standard EEG bands for a single channel."""
    # (Code is identical to the previous scripts - kept for brevity)
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0: return [0.0] * 5
    if channel_data is None or len(channel_data) == 0: return [0.0] * 5
    n_samples = len(channel_data)
    if n_samples < sample_rate / 2: return [0.0] * 5
    try:
        # Suppress potential FFT warnings for very short segments if needed
        # warnings.simplefilter("ignore", np.ComplexWarning)
        fft_vals = np.fft.fft(channel_data)
        # warnings.simplefilter("default", np.ComplexWarning) # Restore warnings
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

# --- Function to Show Final Histogram ---
def show_final_histogram(full_eeg_data, sample_rate):
    """Calculates overall band powers and displays them in a new window."""
    global final_hist_window # Keep reference to the window

    print("\nCalculating overall average band powers for the entire recording...")
    n_channels = full_eeg_data.shape[0]
    all_channel_bins = []
    valid_channels = 0

    for chan_idx in range(n_channels):
        # Calculate bins for the entire duration of this channel
        channel_bins = BrainwaveBins(full_eeg_data[chan_idx, :], sample_rate)
        if sum(channel_bins) > 1e-9: # Check if calculation was successful
            all_channel_bins.append(channel_bins)
            valid_channels += 1

    if valid_channels < n_channels / 2 or not all_channel_bins:
        print("Could not calculate valid features for enough channels over the full duration.")
        return

    # Average features across channels
    overall_avg_powers = np.mean(all_channel_bins, axis=0).tolist()
    print("Overall Average Powers (Delta, Theta, Alpha, Beta, Gamma):")
    print([f"{p:.2f}" for p in overall_avg_powers])

    # Create the histogram window
    final_hist_window = pg.GraphicsLayoutWidget(title="Overall Band Power Histogram", size=(600, 400))
    hist_plot = final_hist_window.addPlot(title="Average Band Power")
    hist_plot.setLabel('left', "Avg Power (relative units)")
    hist_plot.setLabel('bottom', "Frequency Band")
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    x_ticks = [list(enumerate(band_names))]
    hist_plot.getAxis('bottom').setTicks(x_ticks)
    bar_graph = pg.BarGraphItem(x=range(len(band_names)), height=overall_avg_powers, width=0.6, brush='lightgreen')
    hist_plot.addItem(bar_graph)
    # Auto-adjust Y range based on calculated powers
    max_power = max(overall_avg_powers) if overall_avg_powers else 1
    hist_plot.setYRange(0, max_power * 1.1) # Add 10% margin
    final_hist_window.show()
    print("Final histogram window displayed.")

# --- Main Visualization and Prediction Function ---

def visualize_and_predict(csv_filepath, model_dir):
    """
    Loads EEG data, visualizes it, and predicts using saved models.
    """
    print("--- Starting Visualization and Prediction ---")
    fs_int = int(ASSUMED_SAMPLE_RATE)
    feature_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] # For DataFrame columns

    # --- 1. Load Saved Models and Objects ---
    print(f"Loading models and preprocessing objects from: {model_dir}")
    try:
        knn_model = joblib.load(model_dir / 'knn_model.joblib')
        logreg_model = joblib.load(model_dir / 'logreg_model.joblib')
        scaler = joblib.load(model_dir / 'scaler.joblib')
        label_encoder = joblib.load(model_dir / 'label_encoder.joblib')
        print("Models, Scaler, and Label Encoder loaded.")
    except Exception as e:
        print(f"ERROR: Could not load required files from '{model_dir}'. {e}")
        return # Exit if models can't be loaded

    # --- 2. Load EEG Data ---
    print(f"Loading EEG data from: {csv_filepath}")
    if not csv_filepath.exists():
        print(f"ERROR: Input CSV file not found: {csv_filepath}")
        return
    try:
        eeg_df = pd.read_csv(csv_filepath, header=None)
        raw_eeg_data = eeg_df.to_numpy()
        print(f"Raw data shape: {raw_eeg_data.shape}")
        if raw_eeg_data.shape[1] == 5:
            raw_eeg_data = raw_eeg_data[1:len(raw_eeg_data)-1, 1:5]
        raw_eeg_data = raw_eeg_data.astype(np.float64) # Convert to float64 for BrainFlow compatibility 
        if raw_eeg_data.shape[1] < len(EEG_CHANNEL_INDICES):
             print(f"ERROR: CSV has fewer columns ({raw_eeg_data.shape[1]}) than expected ({len(EEG_CHANNEL_INDICES)}).")
             return
        # Transpose to (channels x samples)
        eeg_data = raw_eeg_data[:, EEG_CHANNEL_INDICES].T
        n_channels, n_total_samples = eeg_data.shape
        print(f"Loaded data (Channels x Samples): {eeg_data.shape}")
        # Pre-filter the entire dataset (optional, can also filter segments live)
        print("Pre-filtering entire dataset...")
        for i in range(n_channels):
            DataFilter.perform_bandpass(
                eeg_data[i], fs_int,
                start_freq= BAND_PASS_LOW_CUTOFF,
                stop_freq= BAND_PASS_HIGH_CUTOFF,
                order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH,
                ripple=0
            )
            DataFilter.perform_bandstop(
                eeg_data[i], fs_int,
                start_freq=49, stop_freq=51,
                order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH,
                ripple=0
            )
        print("Filtering complete.")

    except Exception as e:
        print(f"ERROR loading or processing CSV file: {e}")
        return

    # --- 3. Setup PyQtGraph Application ---
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="EEG Visualization and Prediction", size=(1000, 700))
    win.show()

    # --- 4. Create Plot Areas ---
    # EEG Plots (Left Side)
    eeg_plots = []
    eeg_curves = []
    plot_layout = win.addLayout(row=0, col=0) # Layout for EEG plots
    for i in range(n_channels):
        p = plot_layout.addPlot(row=i, col=0)
        p.setYRange(-80, 80) # Adjust Y range based on your data's expected uV values
        p.setLabel('left', f"Ch {EEG_CHANNEL_INDICES[i]}", units='uV')
        p.showGrid(x=True, y=True, alpha=0.3)
        if i < n_channels - 1: # Hide x-axis for all but bottom plot
              p.hideAxis('bottom')
        curve = p.plot(pen=pg.mkPen(color=pg.intColor(i, hues=n_channels), width=1))
        eeg_plots.append(p)
        eeg_curves.append(curve)

    # Histogram and Prediction (Right Side)
    right_layout = win.addLayout(row=0, col=1)
    # Prediction Text Label
    pred_label = right_layout.addLabel("Waiting for data...", row=0, col=0, size='14pt', bold=True, color='white')
    # Histogram Plot
    hist_plot = right_layout.addPlot(row=1, col=0)
    hist_plot.setTitle("Band Power")
    hist_plot.setLabel('left', "Avg Power (uV^2 / Hz ? - units arbitrary)") # Power units depend on FFT scaling
    hist_plot.setLabel('bottom', "Frequency Band")
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    x_ticks = [list(enumerate(band_names))] # Ticks format for AxisItem
    hist_plot.getAxis('bottom').setTicks(x_ticks)
    # Use BarGraphItem for histogram
    bar_graph = pg.BarGraphItem(x=range(len(band_names)), height=[0]*len(band_names), width=0.6, brush='lightblue')
    hist_plot.addItem(bar_graph)
    hist_plot.setYRange(0, 1) # Start with range 0-1, will auto-range later if needed

    # --- 5. Update Function and Timer ---
    current_sample_index = SAMPLES_PER_PLOT_WINDOW # Start plotting after the first window
    last_prediction = {'knn': 'N/A', 'logreg': 'N/A'} # Store last prediction

    def update():
        nonlocal current_sample_index, last_prediction

        # Check if playback is finished
        if current_sample_index >= n_total_samples:
            timer.stop()
            pred_label.setText("Finished")
            print("Playback finished.")
            show_final_histogram(eeg_data, ASSUMED_SAMPLE_RATE) # Pass full data
            # Update label again after histogram shown
            pred_label.setText(f"Finished\nKNN: {last_prediction['knn']}\nLogReg: {last_prediction['logreg']}")
            return

        # --- Update EEG Plot ---
        start_plot = max(0, current_sample_index - SAMPLES_PER_PLOT_WINDOW)
        end_plot = current_sample_index
        plot_data = eeg_data[:, start_plot:end_plot]
        time_axis = np.arange(start_plot, end_plot) / ASSUMED_SAMPLE_RATE # Time in seconds

        for i in range(n_channels):
            # Adjust x-axis dynamically or set fixed range? Let's shift the data view.
            eeg_curves[i].setData(x=time_axis, y=plot_data[i])
            # Keep X-axis somewhat fixed relative to current time
            eeg_plots[i].setXRange(time_axis[0], time_axis[-1] if len(time_axis)>1 else time_axis[0]+0.1)

        # --- Update Features, Histogram, and Prediction ---
        # Analyze the segment ENDING at the current sample index
        start_segment = max(0, current_sample_index - SAMPLES_PER_SEGMENT)
        end_segment = current_sample_index

        # Only calculate features/predict if we have a full segment's worth of data available *before* the current point
        if (end_segment - start_segment) >= SAMPLES_PER_SEGMENT:
            analysis_segment = eeg_data[:, start_segment:end_segment] # No need to copy if pre-filtered

            # Extract Features (Average across channels)
            all_channel_bins = []
            valid_channel_bins = 0
            for chan_idx in range(n_channels):
                channel_bins = BrainwaveBins(analysis_segment[chan_idx, :], ASSUMED_SAMPLE_RATE)
                if sum(channel_bins) > 1e-9:
                    all_channel_bins.append(channel_bins)
                    valid_channel_bins += 1

            if valid_channel_bins > n_channels / 2 and all_channel_bins:
                features_list = np.mean(all_channel_bins, axis=0).tolist()
                features_df = pd.DataFrame([features_list], columns=feature_names)

                # Update Histogram
                current_max_power = max(features_list) if any(f > 0 for f in features_list) else 1
                bar_graph.setOpts(height=features_list)
                # Adjust histogram Y range dynamically (optional)
                # hist_plot.setYRange(0, current_max_power * 1.1) # Add 10% margin

                # Scale and Predict
                try:
                    scaled_features = scaler.transform(features_df)
                    pred_idx_knn = knn_model.predict(scaled_features)[0]
                    pred_label_knn = label_encoder.inverse_transform([pred_idx_knn])[0]

                    pred_idx_logreg = logreg_model.predict(scaled_features)[0]
                    pred_label_logreg = label_encoder.inverse_transform([pred_idx_logreg])[0]

                    last_prediction = {'knn': pred_label_knn, 'logreg': pred_label_logreg}

                except Exception as pred_e:
                    print(f"Prediction error at sample {current_sample_index}: {pred_e}")
                    last_prediction = {'knn': 'Error', 'logreg': 'Error'}
            else:
                 # If features couldn't be extracted, keep last prediction or show error
                 pass # Keep showing the previous prediction/histogram for smoothness

        # Update prediction label
        pred_label.setText(f"KNN: {last_prediction['knn']}\nLogReg: {last_prediction['logreg']}")

        # --- Advance Playback ---
        current_sample_index += SAMPLES_PER_UPDATE

    # Create and start timer
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UPDATE_INTERVAL_MS)

    # --- 6. Start Qt Event Loop ---
    print("Starting visualization. Close the window to exit.")
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
    print("Visualization stopped.")


# --- Main Execution ---

if __name__ == "__main__":
    print("="*50)
    print(" EEG Vis + Prediction using Saved Classifiers ".center(50, "="))
    print("="*50)

    # Check if required files/dirs exist before starting
    if not NEW_DATA_CSV_PATH.exists():
        print(f"Error: Input data file not found at {NEW_DATA_CSV_PATH}")
        sys.exit(1)
    if not LOAD_DIR.is_dir():
        print(f"Error: Saved model directory not found at {LOAD_DIR}")
        sys.exit(1)

    # Run visualization and prediction
    visualize_and_predict(NEW_DATA_CSV_PATH, LOAD_DIR)

    print("\nScript finished.")