# Combined LSL EEG Prediction Script

import time
import os
from pathlib import Path
import sys

# LSL and Data Handling
import pylsl
import numpy as np
import pandas as pd

# Filtering (from BrainFlow)
from brainflow.data_filter import DataFilter, FilterTypes

# Machine Learning / KAN
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kan import * 

# Optional Plotting Dependencies (Keep if using run_live_plot_lsl)
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# --- Configuration ---
# !! REQUIRED: Set the correct LSL stream details !!
LSL_STREAM_TYPE = 'EEG'
LSL_STREAM_NAME = '' # Optional: Specify name if multiple EEG streams exist (e.g., 'Muse')

# !! REQUIRED: Set indices of EEG channels in your LSL stream !!
# Example for Muse S (TP9, AF7, AF8, TP10) might be [0, 1, 2, 3] or [1, 2, 3, 4]
# CHECK YOUR LSL STREAM SOURCE (BlueMuse/museLSL documentation or LSL viewer)
EEG_CHANNEL_INDICES = [0, 1, 2, 3] # <--- *** MODIFY THIS LIST ***

# Data collection duration for prediction
LSL_READ_DURATION_SECONDS = 6 # How many seconds of data to analyze for one prediction

# Paths (relative to the script location)
MASTER_DATA_PATH = Path('static/data/master_df1.csv')
MODEL_CHECKPOINT_PATH = Path('mood_model/model/0.2') # Directory containing model files

# Filtering parameters
LOW_PASS_CUTOFF = 50 # Hz (More common for cognitive EEG than 125)
FILTER_ORDER = 4
BAND_PASS_LOW_CUTOFF = 1 # Hz
BAND_PASS_HIGH_CUTOFF = 50 # Hz

# Brainwave Band Definitions (Hz)
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)
GAMMA_BAND = (30, 50) # Capped at 50Hz to reduce noise


# --- LSL Data Acquisition ---

def read_lsl(duration_seconds=LSL_READ_DURATION_SECONDS, stream_type=LSL_STREAM_TYPE, stream_name=LSL_STREAM_NAME):
    """
    Reads data from an LSL stream for a specified duration.

    Args:
        duration_seconds: How long to collect data.
        stream_type: The LSL stream type (e.g., 'EEG').
        stream_name: Optional specific name of the LSL stream.

    Returns:
        tuple: (numpy.ndarray, float) containing (data, sample_rate)
               data shape is (channels x samples).
               Returns (None, None) if an error occurs.
    """
    print(f"Looking for an LSL stream of type '{stream_type}' (name: '{stream_name or 'any'}')...")
    streams = pylsl.resolve_stream(stream_type, stream_name, timeout=5) # 5 sec timeout

    if not streams:
        print(f"\n--- ERROR ---")
        print(f"No LSL stream found for type '{stream_type}' (name: '{stream_name or 'any'}').")
        print(f"Make sure your EEG device (BlueMuse/museLSL) is streaming.")
        print(f"---------------\n")
        return None, None

    inlet = pylsl.StreamInlet(streams[0])
    stream_info = inlet.info()
    sample_rate = stream_info.nominal_srate()
    n_channels_total = stream_info.channel_count()
    print(f"\nConnected to LSL stream '{stream_info.name()}' ({stream_info.type()})")
    print(f"Total Channels in Stream: {n_channels_total}")

    if sample_rate <= 0:
        print("Nominal sample rate is 0, attempting to read from description...")
        try:
            srate_desc = stream_info.desc().child("nominal_srate").child_value()
            sample_rate = float(srate_desc)
            print(f"Using sample rate from description: {sample_rate:.2f} Hz")
        except (TypeError, ValueError):
             print("Warning: Could not determine valid sample rate. Assuming 256 Hz.")
             sample_rate = 256.0 # Default assumption

    if not EEG_CHANNEL_INDICES:
         print("\n--- ERROR ---")
         print("EEG_CHANNEL_INDICES list is empty. Please configure it.")
         print("---------------\n")
         inlet.close_stream()
         return None, None

    if max(EEG_CHANNEL_INDICES) >= n_channels_total:
        print("\n--- ERROR ---")
        print(f"Invalid EEG_CHANNEL_INDICES: {EEG_CHANNEL_INDICES}.")
        print(f"Max index ({max(EEG_CHANNEL_INDICES)}) >= total channels ({n_channels_total}).")
        print("Check your LSL stream and the EEG_CHANNEL_INDICES configuration.")
        print("---------------\n")
        inlet.close_stream()
        return None, None

    print(f"Selected EEG Channel Indices: {EEG_CHANNEL_INDICES}")
    print(f"Nominal Sample Rate: {sample_rate:.2f} Hz")

    all_data = []
    start_time = time.time()
    print(f"\nCollecting data for ~{duration_seconds} seconds...")

    samples_to_collect = int(sample_rate * duration_seconds)
    # Add buffer in case pulls are slow
    max_samples_per_pull = max(128, int(sample_rate)) # Pull up to 1 sec chunks

    collected_samples = 0
    while time.time() - start_time < (duration_seconds + 2): # Allow 2 extra secs timeout
        # Check if we have enough samples already, avoids overly long reads if SR is high
        if collected_samples >= samples_to_collect and duration_seconds > 0:
             break

        chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=max_samples_per_pull)
        if chunk:
            all_data.extend(chunk)
            collected_samples = len(all_data)
        elif time.time() - start_time > duration_seconds : # Break if timeout exceeded after duration
            if not all_data: # No data at all after waiting
                print("Warning: LSL timeout with no data received.")
            break # Exit loop if time is up

    inlet.close_stream()
    print(f"Finished collecting data. Total samples acquired: {len(all_data)}")

    if not all_data:
        print("\n--- ERROR ---")
        print("No data collected from LSL stream.")
        print("---------------\n")
        return None, None

    # Convert to NumPy array (samples x channels_total)
    np_data = np.array(all_data)
    print(f"Raw data shape: {np_data.shape}")


    # Select only the configured EEG channels and transpose to (channels x samples)
    try:
        eeg_data = np_data[:, EEG_CHANNEL_INDICES].T
        print(f"Selected EEG data shape (channels x samples): {eeg_data.shape}")
    except IndexError:
        print("\n--- ERROR ---")
        print(f"IndexError selecting EEG channels using indices {EEG_CHANNEL_INDICES}.")
        print(f"Check LSL stream structure and configuration.")
        print(f"Available data shape was: {np_data.shape}")
        print("---------------\n")
        return None, None

    # --- Apply Filtering ---
    print("Applying filters...")
    fs = int(sample_rate) # Ensure integer sample rate for DataFilter
    for i in range(eeg_data.shape[0]):
        try:
            # Option 1: Low-pass only (like original)
            # DataFilter.perform_lowpass(eeg_data[i], fs, LOW_PASS_CUTOFF, FILTER_ORDER, FilterTypes.BUTTERWORTH, 0)

            # Option 2: Band-pass (often better for removing DC offset and high freq noise)
             DataFilter.perform_bandpass(
                eeg_data[i], fs,
                center_freq=(BAND_PASS_HIGH_CUTOFF + BAND_PASS_LOW_CUTOFF) / 2.0,
                bandwidth=(BAND_PASS_HIGH_CUTOFF - BAND_PASS_LOW_CUTOFF),
                order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH, ripple=0
             )

            # Optional: Notch filter for powerline noise (50Hz or 60Hz)
            # powerline_freq = 50 # Or 60
            # DataFilter.perform_bandstop(eeg_data[i], fs, powerline_freq, bandwidth=2, order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH, ripple=0)

        except Exception as filter_e:
             print(f"\n--- WARNING ---")
             print(f"Could not apply filter to channel {i}: {filter_e}")
             print(f"Check filter parameters and data quality.")
             print(f"---------------\n")
             # Continue without filtering this channel, or return None if filtering is critical
             # return None, None
    print("Filtering complete.")

    # Make sure sample rate is float for calculations later if needed
    return eeg_data, float(sample_rate)


# --- Feature Extraction ---

def BrainwaveBins(channel_data, sample_rate):
    """
    Calculates average power in standard EEG bands for a single channel.

    Args:
        channel_data (numpy.ndarray): 1D array of EEG data for one channel.
        sample_rate (float): The sampling rate of the data.

    Returns:
        list: Average power values for [Delta, Theta, Alpha, Beta, Gamma].
              Returns list of zeros if calculation fails.
    """
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        print(f"Error: Invalid sample_rate provided to BrainwaveBins: {sample_rate}")
        return [0.0] * 5
    if channel_data is None or len(channel_data) == 0:
        print("Error: Empty channel data provided to BrainwaveBins.")
        return [0.0] * 5

    n_samples = len(channel_data)
    if n_samples < sample_rate: # Need at least 1 second of data for meaningful FFT
        print(f"Warning: Short data segment ({n_samples} samples) for FFT analysis.")
        # return [0.0] * 5 # Option: return zeros for very short segments

    try:
        # --- FFT Calculation ---
        fft_vals = np.fft.fft(channel_data)
        fft_freq = np.fft.fftfreq(n_samples, d=1.0/sample_rate)

        # Get positive frequencies and corresponding FFT values
        pos_freq_indices = np.where(fft_freq > 0)[0]
        freqs = fft_freq[pos_freq_indices]
        fft_vals_pos = fft_vals[pos_freq_indices]

        # Calculate Power Spectral Density (PSD) - magnitude squared
        # Optional: Normalize by number of samples for better comparison across lengths
        psd = (np.abs(fft_vals_pos)**2) / n_samples

        # --- Binning ---
        bands = {
            'Delta': DELTA_BAND,
            'Theta': THETA_BAND,
            'Alpha': ALPHA_BAND,
            'Beta': BETA_BAND,
            'Gamma': GAMMA_BAND
        }
        band_powers = {name: 0.0 for name in bands}
        band_counts = {name: 0 for name in bands}

        for freq, power in zip(freqs, psd):
            for name, (low, high) in bands.items():
                if low <= freq < high:
                    band_powers[name] += power
                    band_counts[name] += 1
                    break # Move to next frequency once assigned to a band

        # Calculate average power per bin (handle empty bins)
        avg_powers = []
        for name in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']: # Ensure order
             avg_powers.append(band_powers[name] / band_counts[name] if band_counts[name] > 0 else 0.0)

        return avg_powers

    except Exception as e:
        print(f"\n--- ERROR during BrainwaveBins calculation ---")
        print(f"{e}")
        print(f"Input data length: {len(channel_data)}, Sample rate: {sample_rate}")
        print(f"Returning zeros.")
        print(f"--------------------------------------------\n")
        return [0.0] * 5


# --- Data Formatting ---

def wave_to_df(bin_list):
  """Converts a list of band powers to a Pandas DataFrame."""
  wave_data = {
    "Delta" : bin_list[0],
    "Theta" : bin_list[1],
    "Alpha" : bin_list[2],
    "Beta" :  bin_list[3],
    "Gamma" : bin_list[4]
  }
  df = pd.DataFrame([wave_data]) # Use [] to create single row DataFrame
  return df


# --- Prediction Function ---

def predict_emotion_from_lsl():
    """
    Reads LSL data, processes it, and predicts emotion using the KAN model.

    Returns:
        str: The predicted emotion label (e.g., "Happy", "Sad", "Stressed")
             or "Unknown" if prediction fails.
    """
    print("\n--- Starting Emotion Prediction ---")

    # --- 1. Load Preprocessing Tools (Fitted on Master Data) ---
    print("Loading master data for preprocessing context...")
    if not MASTER_DATA_PATH.exists():
        print(f"\n--- ERROR ---")
        print(f"Master data file not found: {MASTER_DATA_PATH}")
        print(f"This file is required to set up the data scaler and label encoder.")
        print(f"Make sure it's in the 'static/data/' directory relative to the script.")
        print(f"---------------\n")
        return "Unknown"

    try:
        master_data = pd.read_csv(MASTER_DATA_PATH)
        y = master_data['Emotion']
        X = master_data.drop('Emotion', axis=1)

        # Ensure feature names match expected bands
        expected_features = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        if not all(feature in X.columns for feature in expected_features):
             print(f"\n--- ERROR ---")
             print(f"Master data columns ({list(X.columns)}) do not match expected features ({expected_features}).")
             print(f"Check the '{MASTER_DATA_PATH}' file.")
             print(f"---------------\n")
             return "Unknown"
        # Ensure scaler uses the correct order
        X = X[expected_features]


        scaler = StandardScaler()
        scaler.fit(X) # Fit scaler ONLY on master data

        le = LabelEncoder()
        le.fit(y) # Fit encoder ONLY on master data labels
        print("Preprocessing tools initialized.")

    except Exception as e:
        print(f"\n--- ERROR loading or processing master data ---")
        print(f"{e}")
        print(f"Check the format and content of '{MASTER_DATA_PATH}'.")
        print(f"--------------------------------------------------\n")
        return "Unknown"

    # --- 2. Acquire and Process Live LSL Data ---
    lsl_eeg_data, sample_rate = read_lsl()

    if lsl_eeg_data is None or sample_rate is None:
        print("Prediction aborted: Failed to acquire valid LSL data.")
        return "Unknown"

    if lsl_eeg_data.shape[0] == 0:
        print("Prediction aborted: No EEG channels selected or data is empty.")
        return "Unknown"

    print("\nCalculating brainwave features...")
    # --- Feature Extraction Strategy ---
    # Option A: Use features from the first selected EEG channel (like original implicit behavior)
    print(f"Using features from the first selected EEG channel (index {EEG_CHANNEL_INDICES[0]} in LSL stream).")
    features_list = BrainwaveBins(lsl_eeg_data[0, :], sample_rate)

    # Option B: Average features across all selected EEG channels (potentially more robust)
    # all_bins = []
    # for i in range(lsl_eeg_data.shape[0]):
    #     channel_bins = BrainwaveBins(lsl_eeg_data[i, :], sample_rate)
    #     all_bins.append(channel_bins)
    # if not all_bins:
    #     print("Error: Could not calculate bins for any channel.")
    #     return "Unknown"
    # features_list = np.mean(all_bins, axis=0).tolist()
    # print(f"Using averaged features across {lsl_eeg_data.shape[0]} channels.")
    # ------------------------------------

    print(f"Extracted Features (Avg Power): {features_list}")
    pred_df = wave_to_df(features_list)

    # --- 3. Scale Live Data ---
    try:
        # Ensure DataFrame columns are in the same order as scaler expects
        pred_df = pred_df[expected_features]
        pred_scaled = scaler.transform(pred_df)
        print(f"Scaled Features: {pred_scaled.flatten().tolist()}")
    except Exception as e:
         print(f"\n--- ERROR scaling live data ---")
         print(f"{e}")
         print(f"Check if live features ({list(pred_df.columns)}) match master data features.")
         print(f"--------------------------------\n")
         return "Unknown"

    # --- 4. Load KAN Model ---
    print("Loading KAN model...")
    if not MODEL_CHECKPOINT_PATH.is_dir():
        print(f"\n--- ERROR ---")
        print(f"KAN model checkpoint directory not found: {MODEL_CHECKPOINT_PATH}")
        print(f"Make sure the 'mood_model/model/0.2/' directory exists and contains model files.")
        print(f"---------------\n")
        return "Unknown"

    try:
        # Determine input and output dimensions from scaler and encoder
        input_dim = scaler.n_features_in_
        output_dim = len(le.classes_)

        # Initialize a KAN model structure - dimensions must match the saved one.
        # These might need adjustment based on how the original model was defined.
        # Common KAN parameters: width (list of layer widths), grid points, spline order.
        # Example: model = KAN(width=[input_dim, 10, output_dim], grid=5, k=3)
        # Since we're loading a checkpoint, we might just need the structure.
        # Let's assume a simple structure if loadckpt handles it internally.
        # The specific `kan.KAN.loadckpt` might require an instantiated model first,
        # or it might load the architecture too. Check KAN library docs if needed.
        # Let's *assume* `loadckpt` works like PyTorch's `load_state_dict` where
        # you need the model class defined first.
        # *** This part might require knowing the exact architecture used for training ***
        # Try loading directly first, if it fails, define the architecture.
        model = KAN.loadckpt(MODEL_CHECKPOINT_PATH, map_location=torch.device('cpu')) # Use CPU
        model.eval() # Set model to evaluation mode
        print("KAN model loaded successfully.")

    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Model file(s) not found inside '{MODEL_CHECKPOINT_PATH}'.")
        print(f"Check the contents of the directory.")
        print(f"---------------\n")
        return "Unknown"
    except Exception as e:
        print(f"\n--- ERROR loading KAN model ---")
        print(f"{e}")
        print(f"Ensure the correct KAN library is installed and the checkpoint is compatible.")
        # print(f"You might need to define the KAN architecture before loading.")
        print(f"--------------------------------\n")
        return "Unknown"

    # --- 5. Make Prediction ---
    print("Making prediction...")
    try:
        # Convert scaled data to tensor
        input_tensor = torch.tensor(pred_scaled, dtype=torch.float32)

        # Get model output (logits)
        with torch.no_grad(): # Disable gradient calculation for inference
            logits = model(input_tensor)

        # Get predicted class index
        predicted_class_idx = torch.argmax(logits, dim=1).item()

        # Decode index to emotion label
        predicted_emotion = le.inverse_transform([predicted_class_idx])[0]
        print(f"Raw Prediction: {predicted_emotion}")

        # Apply specific mapping if needed
        if predicted_emotion == "Angry":
            predicted_emotion = "Stressed"
            print("Mapping 'Angry' to 'Stressed'")

    except Exception as e:
        print(f"\n--- ERROR during prediction step ---")
        print(f"{e}")
        print(f"Check model compatibility and input tensor shape.")
        print(f"-------------------------------------\n")
        return "Unknown"

    print("--- Prediction Complete ---")
    return str(predicted_emotion)


# --- Optional: Live Plotting Function (Adapted for LSL) ---

def run_live_plot_lsl(plot_duration=10):
    """
    Stream EEG data from LSL and plot in real-time using PyQtGraph.
    Requires PyQt5 and pyqtgraph.
    """
    print("\n--- Starting Live LSL Plot ---")
    print("NOTE: This plots raw (or lightly filtered) data, not features or predictions.")
    print("Close the plot window to stop.")

    print(f"Looking for LSL stream '{LSL_STREAM_TYPE}'...")
    streams = pylsl.resolve_stream(LSL_STREAM_TYPE, LSL_STREAM_NAME, timeout=5)
    if not streams:
        print("Error: LSL stream not found for plotting.")
        return

    inlet = pylsl.StreamInlet(streams[0])
    stream_info = inlet.info()
    sample_rate = int(stream_info.nominal_srate())
    if sample_rate <= 0: sample_rate = 256 # Fallback
    n_channels_total = stream_info.channel_count()

    if not EEG_CHANNEL_INDICES or max(EEG_CHANNEL_INDICES) >= n_channels_total:
        print("Error: Invalid EEG_CHANNEL_INDICES for plotting.")
        inlet.close_stream()
        return

    eeg_channels_count = len(EEG_CHANNEL_INDICES)
    plot_window_samples = sample_rate * 3 # Plot ~3 seconds of data
    data_buffer = np.zeros((eeg_channels_count, plot_window_samples))

    print(f"Connected to '{stream_info.name()}'. Plotting {eeg_channels_count} channels.")
    print(f"Sample Rate: {sample_rate} Hz")

    app = QtWidgets.QApplication.instance() # Check if QApplication already exists
    if not app: # Create QApplication if it doesnt exist or is not running
        app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Real-Time LSL EEG Data")
    win.resize(800, 600)
    win.show()

    plots = []
    curves = []
    for i in range(eeg_channels_count):
        plot = win.addPlot(row=i, col=0)
        # Adjust YRange based on typical EEG microvolt values
        plot.setYRange(-80, 80)
        plot.setLabel("left", f"LSL Ch Idx {EEG_CHANNEL_INDICES[i]}", units="uV") # Show LSL index
        plot.showGrid(x=True, y=True)
        # Use different colors for channels
        curve = plot.plot(pen=pg.mkPen(color=pg.intColor(i, hues=eeg_channels_count), width=1))
        plots.append(plot)
        curves.append(curve)

    last_update_time = time.time()

    def update():
        nonlocal data_buffer, last_update_time
        samples_per_pull = int(sample_rate * 0.05) # Try to pull ~50ms chunks
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=samples_per_pull) # Non-blocking pull

        if chunk:
            np_chunk = np.array(chunk)
            # Select EEG channels and transpose: (channels x samples)
            try:
                 eeg_chunk = np_chunk[:, EEG_CHANNEL_INDICES].T
            except IndexError:
                 print("Plotting Error: Index mismatch during update.")
                 return # Skip update if indices are wrong

            n_new_samples = eeg_chunk.shape[1]

            # Apply quick filter (optional, raw might be okay for viz)
            for chan_idx in range(eeg_channels_count):
                  DataFilter.perform_bandpass(eeg_chunk[chan_idx], sample_rate, 25.5, 49, 4, FilterTypes.BUTTERWORTH, 0)

            # Update buffer (shift left, add new data)
            if n_new_samples < plot_window_samples:
                data_buffer[:, :-n_new_samples] = data_buffer[:, n_new_samples:]
                data_buffer[:, -n_new_samples:] = eeg_chunk
            else: # If chunk is larger than buffer, just take the latest part
                data_buffer = eeg_chunk[:, -plot_window_samples:]

        # Limit update rate to ~25 Hz to avoid overwhelming CPU/GUI
        current_time = time.time()
        if current_time - last_update_time > 0.04: # ~25 Hz
            for idx in range(eeg_channels_count):
                curves[idx].setData(data_buffer[idx])
            last_update_time = current_time
            app.processEvents() # Crucial for responsiveness

    # Use QTimer for periodic updates
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(20) # Check for new data every 20ms

    print("Plot window opened. Close the window to exit plotting.")
    # Start Qt event loop
    # Make sure this runs correctly. If app already exists, exec_() might not be needed this way.
    if hasattr(app, 'exec_'):
        app.exec_()
    else:
        # Fallback or alternative for existing app context might be needed
        print("QApplication execution might need manual handling if nested.")
        while win.isVisible(): # Keep updating while window is open
            app.processEvents()
            time.sleep(0.01) # Prevent busy-loop


    # Cleanup after plot window is closed
    print("Plot window closed. Stopping LSL inlet.")
    timer.stop()
    try:
        inlet.close_stream()
    except Exception as e:
        print(f"Error closing LSL stream: {e}")
    print("--- Live LSL Plot Finished ---")


# --- Main Execution ---

if __name__ == "__main__":
    # --- Option 1: Run Prediction Once ---
    print("Running emotion prediction...")
    try:
        predicted_mood = predict_emotion_from_lsl()
        print(f"\n=============================")
        print(f"Predicted Mood: {predicted_mood}")
        print(f"=============================")
    except Exception as e:
        print(f"\n--- An unexpected error occurred during main execution ---")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print(f"-----------------------------------------------------------\n")

    # --- Option 2: Run Live Plotting (Uncomment to use) ---
    print("\nStarting live plotting...")
    print("Prediction will run *after* the plot window is closed.")
    run_live_plot_lsl(plot_duration=30) # Plot for 30 seconds
    # Optionally run prediction again after plotting
    print("\nRunning prediction after plotting finished...")
    try:
        predicted_mood_after_plot = predict_emotion_from_lsl()
        print(f"\n=============================")
        print(f"Predicted Mood (after plot): {predicted_mood_after_plot}")
        print(f"=============================")
    except Exception as e:
        print(f"\n--- An unexpected error occurred during post-plot prediction ---")
        print(f"{e}")
        print(f"-----------------------------------------------------------------\n")

    print("\nScript finished.")