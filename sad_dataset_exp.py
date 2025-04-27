import time
import os
from pathlib import Path
import sys

# Data Handling
import numpy as np
import pandas as pd

# Filtering (from BrainFlow)
from brainflow.data_filter import DataFilter, FilterTypes

# Machine Learning / KAN
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kan import KAN # Assuming 'KAN-pytorch' is the library used

# --- Configuration ---

# !! REQUIRED: Set the correct path to your CSV file !!
CSV_DATA_PATH = Path('Brainwaves_Relaxed_1.csv') # Assumes it's in the same directory

# !! CRITICAL ASSUMPTION: Set the sample rate of your recording !!
ASSUMED_SAMPLE_RATE = 256.0 # Hz (Common for Muse 2 / Muse S) - CHANGE IF DIFFERENT

# Segmentation parameters
SEGMENT_DURATION_SECONDS = 5 # How many seconds of data per classification window
# Calculate samples per segment based on assumed rate
SAMPLES_PER_SEGMENT = int(ASSUMED_SAMPLE_RATE * SEGMENT_DURATION_SECONDS)

# Channel indices to use from the CSV (assuming 4 columns = 4 EEG channels)
EEG_CHANNEL_INDICES = [0, 1, 2, 3]

# Paths (relative to the script location)
MASTER_DATA_PATH = Path('static/data/master_df1.csv') # For scaler/encoder context
MODEL_CHECKPOINT_PATH = Path('mood_model/model/0.2') # KAN model checkpoint dir

# Filtering parameters (same as LSL script)
FILTER_ORDER = 4
BAND_PASS_LOW_CUTOFF = 1 # Hz
BAND_PASS_HIGH_CUTOFF = 50 # Hz

# Brainwave Band Definitions (Hz)
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)
GAMMA_BAND = (30, 50) # Capped at 50Hz to reduce noise

# --- Helper Functions (BrainwaveBins, wave_to_df - same as before) ---

def BrainwaveBins(channel_data, sample_rate):
    """
    Calculates average power in standard EEG bands for a single channel.
    (Identical to the function in the LSL script)
    """
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        print(f"Error: Invalid sample_rate provided to BrainwaveBins: {sample_rate}")
        return [0.0] * 5
    if channel_data is None or len(channel_data) == 0:
        print("Error: Empty channel data provided to BrainwaveBins.")
        return [0.0] * 5

    n_samples = len(channel_data)
    if n_samples < sample_rate:
        print(f"Warning: Short data segment ({n_samples} samples < {sample_rate} Hz) for FFT analysis.")
        # Decide how to handle: return zeros or attempt calculation
        # return [0.0] * 5 # Option 1: Return zeros
        if n_samples < 10: return [0.0] * 5 # Definitely too short

    try:
        fft_vals = np.fft.fft(channel_data)
        fft_freq = np.fft.fftfreq(n_samples, d=1.0/sample_rate)

        pos_freq_indices = np.where(fft_freq > 0)[0]
        if len(pos_freq_indices) == 0: # Handle case with insufficient samples for positive freqs
             print(f"Warning: No positive frequencies found (n_samples={n_samples}). Cannot compute bins.")
             return [0.0] * 5
        freqs = fft_freq[pos_freq_indices]
        fft_vals_pos = fft_vals[pos_freq_indices]

        # Power Spectral Density (PSD)
        psd = (np.abs(fft_vals_pos)**2) / n_samples

        bands = {
            'Delta': DELTA_BAND, 'Theta': THETA_BAND, 'Alpha': ALPHA_BAND,
            'Beta': BETA_BAND, 'Gamma': GAMMA_BAND
        }
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
        print(f"\n--- ERROR during BrainwaveBins calculation ---")
        print(f"{e}")
        print(f"Input data length: {len(channel_data)}, Sample rate: {sample_rate}")
        print(f"Returning zeros.")
        print(f"--------------------------------------------\n")
        return [0.0] * 5

def wave_to_df(bin_list):
  """Converts a list of band powers to a Pandas DataFrame."""
  wave_data = {
    "Delta" : bin_list[0], "Theta" : bin_list[1], "Alpha" : bin_list[2],
    "Beta" :  bin_list[3], "Gamma" : bin_list[4]
  }
  df = pd.DataFrame([wave_data]) # Use [] to create single row DataFrame
  return df

# --- Main Classification Function ---

def classify_eeg_file(csv_filepath, master_data_path, model_checkpoint_path):
    """
    Loads EEG data from a CSV, segments it, and classifies each segment.

    Args:
        csv_filepath (Path): Path to the input EEG CSV file.
        master_data_path (Path): Path to the master data CSV for preprocessing context.
        model_checkpoint_path (Path): Path to the KAN model checkpoint directory.

    Returns:
        list: A list of predicted emotion strings for each segment.
              Returns an empty list if errors occur during setup.
    """
    print("--- Starting EEG File Classification ---")
    predictions = []

    # --- 1. Load EEG Data ---
    print(f"Loading EEG data from: {csv_filepath}")
    if not csv_filepath.exists():
        print(f"\n--- ERROR: Input CSV file not found: {csv_filepath} ---\n")
        return []
    try:
        # Read CSV, assuming no header and 4 columns
        eeg_df = pd.read_csv(csv_filepath, header=None, names=['Ch0', 'Ch1', 'Ch2', 'Ch3'])
        # Convert to numpy array (samples x channels)
        raw_eeg_data = eeg_df.to_numpy()
        # Transpose to (channels x samples)
        eeg_data = raw_eeg_data.T
        n_channels, n_total_samples = eeg_data.shape
        print(f"Loaded data shape (Channels x Samples): {eeg_data.shape}")
        print(f"Total duration (estimated): {n_total_samples / ASSUMED_SAMPLE_RATE:.2f} seconds")

        if n_channels != len(EEG_CHANNEL_INDICES):
            print(f"\n--- WARNING ---")
            print(f"CSV file has {n_channels} columns, but expected {len(EEG_CHANNEL_INDICES)} based on EEG_CHANNEL_INDICES.")
            print(f"Using the first {len(EEG_CHANNEL_INDICES)} channels.")
            print(f"---------------\n")
            eeg_data = eeg_data[EEG_CHANNEL_INDICES, :] # Select configured channels

        # Check for potential issues (like the -1000 values)
        if (eeg_data < -900).any():
             print("\n--- WARNING: Data contains very low values (e.g., -1000). ---")
             print("This might indicate clipping or artifacts, which could affect results.")
             print("--------------------------------------------------------------\n")

    except Exception as e:
        print(f"\n--- ERROR loading or processing CSV file ---")
        print(f"{e}")
        print(f"Check the file format (should be plain CSV, no headers).")
        print(f"---------------------------------------------\n")
        return []

    # --- 2. Load Preprocessing Tools & Model (Done Once) ---
    print("\nLoading preprocessing tools and KAN model...")
    if not master_data_path.exists():
        print(f"\n--- ERROR: Master data file not found: {master_data_path} ---\n")
        return []
    if not model_checkpoint_path.is_dir():
        print(f"\n--- ERROR: KAN model directory not found: {model_checkpoint_path} ---\n")
        return []

    try:
        # Load master data for scaler/encoder
        master_data = pd.read_csv(master_data_path)
        y = master_data['Emotion']
        X = master_data.drop('Emotion', axis=1)
        expected_features = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        if not all(feature in X.columns for feature in expected_features):
             print(f"\n--- ERROR: Master data columns mismatch. Check '{master_data_path}'. ---\n")
             return []
        X = X[expected_features] # Ensure correct order

        scaler = StandardScaler().fit(X)
        le = LabelEncoder().fit(y)

        # Load KAN model
        model = KAN.loadckpt(model_checkpoint_path)
        model.eval() # Set to evaluation mode
        print("Preprocessing tools and model loaded successfully.")

    except Exception as e:
        print(f"\n--- ERROR loading master data or KAN model ---")
        print(f"{e}")
        print(f"-------------------------------------------------\n")
        return []

    # --- 3. Process Data in Segments ---
    print(f"\nProcessing data in {SEGMENT_DURATION_SECONDS}-second segments (approx. {SAMPLES_PER_SEGMENT} samples each)...")
    num_segments = n_total_samples // SAMPLES_PER_SEGMENT
    print(f"Total full segments found: {num_segments}")

    if num_segments == 0:
        print("\n--- WARNING: Data is shorter than one segment duration. Cannot classify. ---")
        print(f"Needed {SAMPLES_PER_SEGMENT} samples, but only have {n_total_samples}.\n")
        return []

    for i in range(num_segments):
        print(f"\n--- Processing Segment {i+1}/{num_segments} ---")
        start_sample = i * SAMPLES_PER_SEGMENT
        end_sample = start_sample + SAMPLES_PER_SEGMENT
        segment_data = eeg_data[:, start_sample:end_sample].copy() # Use copy to avoid modifying original

        # --- 3a. Filter Segment ---
        print("Applying filters...")
        fs_int = int(ASSUMED_SAMPLE_RATE)
        for chan_idx in range(segment_data.shape[0]):
            try:
                DataFilter.perform_bandpass(
                    segment_data[chan_idx], fs_int,
                    start_freq= BAND_PASS_LOW_CUTOFF,
                    stop_freq= BAND_PASS_HIGH_CUTOFF,
                    order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH,
                    ripple=0
                )
            except Exception as filter_e:
                 print(f"Warning: Could not filter channel {chan_idx} in segment {i+1}: {filter_e}")
                 # Decide how to handle: skip segment? continue with unfiltered?
                 # For now, continue with potentially unfiltered data for that channel

        # --- 3b. Extract Features (Average across channels) ---
        print("Calculating brainwave features...")
        all_channel_bins = []
        for chan_idx in range(segment_data.shape[0]):
            channel_bins = BrainwaveBins(segment_data[chan_idx, :], ASSUMED_SAMPLE_RATE)
            all_channel_bins.append(channel_bins)

        if not all_channel_bins or len(all_channel_bins) != segment_data.shape[0]:
            print(f"Error: Could not calculate features for all channels in segment {i+1}. Skipping segment.")
            predictions.append("Error_Features")
            continue

        # Average features across channels
        features_list = np.mean(all_channel_bins, axis=0).tolist()
        print(f"Averaged Features (Delta,Theta,Alpha,Beta,Gamma): {[f'{x:.2f}' for x in features_list]}")
        features_df = wave_to_df(features_list)

        # --- 3c. Scale and Predict ---
        try:
            # Ensure columns are in correct order for scaler
            features_df = features_df[expected_features]
            scaled_features = scaler.transform(features_df)
            # print(f"Scaled Features: {[f'{x:.3f}' for x in scaled_features.flatten()]}")

            # Convert to tensor
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

            # Predict
            with torch.no_grad():
                logits = model(input_tensor)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted_emotion = le.inverse_transform([predicted_idx])[0]

            # Map Angry -> Stressed if needed
            final_prediction = "Stressed" if predicted_emotion == "Angry" else predicted_emotion
            print(f"Segment {i+1} Prediction: {final_prediction}")
            predictions.append(final_prediction)

        except Exception as e:
            print(f"\n--- ERROR during scaling or prediction for segment {i+1} ---")
            print(f"{e}")
            print(f"-----------------------------------------------------------\n")
            predictions.append("Error_Prediction") # Add placeholder for error

    print("\n--- Classification Finished ---")
    return predictions

# --- Main Execution ---

if __name__ == "__main__":
    # Check if required files/dirs exist before starting
    if not CSV_DATA_PATH.exists():
        print(f"Error: Input data file not found at {CSV_DATA_PATH}")
        sys.exit(1)
    if not MASTER_DATA_PATH.exists():
        print(f"Error: Master data file not found at {MASTER_DATA_PATH}")
        sys.exit(1)
    if not MODEL_CHECKPOINT_PATH.is_dir():
        print(f"Error: Model checkpoint directory not found at {MODEL_CHECKPOINT_PATH}")
        sys.exit(1)
    if SAMPLES_PER_SEGMENT <= 0:
         print(f"Error: Calculated samples per segment is zero or negative. Check sample rate and duration.")
         sys.exit(1)


    # Run the classification
    results = classify_eeg_file(CSV_DATA_PATH, MASTER_DATA_PATH, MODEL_CHECKPOINT_PATH)

    # Print the results
    if results:
        print("\n--- Final Predictions per Segment ---")
        for i, prediction in enumerate(results):
            start_time = i * SEGMENT_DURATION_SECONDS
            end_time = start_time + SEGMENT_DURATION_SECONDS
            print(f"Segment {i+1} ({start_time:.1f}s - {end_time:.1f}s approx.): {prediction}")

        # Optional: Calculate summary statistics
        print("\n--- Summary ---")
        prediction_counts = pd.Series(results).value_counts()
        print(prediction_counts)
    else:
        print("\nNo predictions were generated due to errors during setup or processing.")

    print("\nScript finished.")