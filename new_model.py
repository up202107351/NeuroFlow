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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # <-- Import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---

# !! REQUIRED: Set the path to the base directory containing emotion subfolders !!
DATASET_BASE_DIR = Path('emotions_dataset')

# !! CRITICAL ASSUMPTION: Set the sample rate of your recordings !!
ASSUMED_SAMPLE_RATE = 256.0 # Hz (Common for Muse 2 / Muse S) - CHANGE IF DIFFERENT

# Segmentation parameters
SEGMENT_DURATION_SECONDS = 5 # How many seconds of data per classification window
SAMPLES_PER_SEGMENT = int(ASSUMED_SAMPLE_RATE * SEGMENT_DURATION_SECONDS)

# Channel indices to use from the CSV (assuming 4 columns = 4 EEG channels)
EEG_CHANNEL_INDICES = [0, 1, 2, 3] # TP9, AF7, AF8, TP10 (typical order)

# Filtering parameters
FILTER_ORDER = 4
BAND_PASS_LOW_CUTOFF = 0.1 # Hz
BAND_PASS_HIGH_CUTOFF = 50 # Hz

# Brainwave Band Definitions (Hz)
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)
GAMMA_BAND = (30, 50) # Capped at 50Hz to reduce noise

# Directory to save models and objects
SAVE_DIR = Path('saved_models')

# --- Helper Functions (BrainwaveBins - same as before) ---

def BrainwaveBins(channel_data, sample_rate):
    """Calculates average power in standard EEG bands for a single channel."""
    # (Code is identical to previous versions - kept for brevity)
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0: return [0.0] * 5
    if channel_data is None or len(channel_data) == 0: return [0.0] * 5
    n_samples = len(channel_data)
    if n_samples < sample_rate / 2: return [0.0] * 5
    try:
        fft_vals = np.fft.fft(channel_data); fft_freq = np.fft.fftfreq(n_samples, d=1.0/sample_rate)
        pos_freq_indices = np.where(fft_freq > 0)[0]
        if len(pos_freq_indices) == 0: return [0.0] * 5
        freqs = fft_freq[pos_freq_indices]; fft_vals_pos = fft_vals[pos_freq_indices]
        psd = (np.abs(fft_vals_pos)**2) / n_samples
        bands = {'Delta': DELTA_BAND, 'Theta': THETA_BAND, 'Alpha': ALPHA_BAND, 'Beta': BETA_BAND, 'Gamma': GAMMA_BAND}
        band_powers = {name: 0.0 for name in bands}; band_counts = {name: 0 for name in bands}
        for freq, power in zip(freqs, psd):
            for name, (low, high) in bands.items():
                if low <= freq < high: band_powers[name] += power; band_counts[name] += 1; break
        avg_powers = []
        for name in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
             avg_powers.append(band_powers[name] / band_counts[name] if band_counts[name] > 0 else 0.0)
        return avg_powers
    except Exception as e: return [0.0] * 5

# --- Feature Extraction (extract_features_from_datasets - same as before) ---

def extract_features_from_datasets(base_dir):
    """Scans subdirectories for CSV files, extracts features and labels for each segment."""
    # (Code is identical to previous versions - kept for brevity)
    all_features = []; all_labels = []
    fs_int = int(ASSUMED_SAMPLE_RATE)
    if not base_dir.is_dir(): print(f"Error: Base directory not found: {base_dir}"); return None, None
    print(f"Scanning for CSV files in subdirectories of: {base_dir}")
    found_files = list(base_dir.rglob('*.csv'))
    if not found_files: print(f"Error: No CSV files found in subdirectories of {base_dir}"); return None, None
    print(f"Found {len(found_files)} CSV files. Processing...")
    for csv_filepath in found_files:
        try:
            emotion_label = csv_filepath.parent.name
            print(f"\nProcessing: {csv_filepath.name} (Label: {emotion_label})")
            eeg_df = pd.read_csv(csv_filepath, header=None)
            raw_eeg_data = eeg_df.to_numpy()
            if raw_eeg_data.shape[1] < len(EEG_CHANNEL_INDICES): print(f"  Warning: Skipping {csv_filepath.name} - Fewer columns ({raw_eeg_data.shape[1]}) than expected ({len(EEG_CHANNEL_INDICES)})."); continue
            eeg_data = raw_eeg_data[:, EEG_CHANNEL_INDICES].T
            n_channels, n_total_samples = eeg_data.shape
            num_segments = n_total_samples // SAMPLES_PER_SEGMENT
            if num_segments == 0: print(f"  Warning: Skipping {csv_filepath.name} - Data ({n_total_samples} samples) is shorter than one segment ({SAMPLES_PER_SEGMENT} samples)."); continue
            print(f"  Extracting features from {num_segments} segments...")
            for i in range(num_segments):
                start_sample = i * SAMPLES_PER_SEGMENT; end_sample = start_sample + SAMPLES_PER_SEGMENT
                segment_data = eeg_data[:, start_sample:end_sample].copy()
                for chan_idx in range(segment_data.shape[0]):
                    try:
                        # Corrected bandpass call for BrainFlow >= 5.0
                        # DataFilter.perform_bandpass(segment_data[chan_idx], fs_int, (BAND_PASS_HIGH_CUTOFF + BAND_PASS_LOW_CUTOFF) / 2.0, (BAND_PASS_HIGH_CUTOFF - BAND_PASS_LOW_CUTOFF), FILTER_ORDER, FilterTypes.BUTTERWORTH, 0)
                        # Using start/stop freq arguments (check your BrainFlow version documentation if unsure)
                        DataFilter.perform_bandpass(
                            segment_data[chan_idx], fs_int,
                            start_freq=BAND_PASS_LOW_CUTOFF, # Use keyword args for clarity
                            stop_freq=BAND_PASS_HIGH_CUTOFF,  # Use keyword args for clarity
                            order=FILTER_ORDER, filter_type=FilterTypes.BUTTERWORTH, ripple=0
                         )
                        DataFilter.perform_bandstop(
                            segment_data[chan_idx], fs_int,
                            start_freq=49.0, stop_freq=51.0, order=FILTER_ORDER,
                            filter_type=FilterTypes.BUTTERWORTH, ripple=0
                        )
                    except Exception as filter_e: # Catch potential errors
                         # print(f"  Warning: Filter failed on chan {chan_idx} seg {i+1}: {filter_e}")
                         pass # Process unfiltered channel
                all_channel_bins = []; valid_channel_bins = 0
                for chan_idx in range(segment_data.shape[0]):
                    channel_bins = BrainwaveBins(segment_data[chan_idx, :], ASSUMED_SAMPLE_RATE)
                    if sum(channel_bins) > 1e-9: all_channel_bins.append(channel_bins); valid_channel_bins += 1
                if valid_channel_bins < segment_data.shape[0] / 2 : continue
                if all_channel_bins: features_list = np.mean(all_channel_bins, axis=0).tolist(); all_features.append(features_list); all_labels.append(emotion_label)
        except pd.errors.EmptyDataError: print(f"  Warning: Skipping empty file {csv_filepath.name}")
        except Exception as e: print(f"  Error processing file {csv_filepath.name}: {e}"); continue
    if not all_features or not all_labels: print("\nError: No valid features were extracted from any file."); return None, None
    feature_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    X = pd.DataFrame(all_features, columns=feature_names); y = pd.Series(all_labels, name="Emotion")
    print(f"\nFeature extraction complete. Total segments processed: {len(X)}")
    print("Class distribution in extracted data:"); print(y.value_counts())
    return X, y

# --- Model Training, Evaluation, and Saving ---

def train_and_evaluate(X, y):
    """
    Splits data, scales features, trains KNN, LogReg, SVM and evaluates/saves.
    """
    if X is None or y is None: print("Cannot train models: No data provided."); return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\nLabels encoded. Classes: {le.classes_}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
        print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    except ValueError as e: print(f"\nError during train/test split: {e}. Check class distribution."); return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")

    # --- KNN Classifier ---
    print("\n--- Training K-Nearest Neighbors (KNN) ---")
    knn_neighbors = 5
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    print(f"\n--- KNN (k={knn_neighbors}) Evaluation ---")
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Accuracy: {accuracy_knn:.4f}")
    print("Classification Report:"); print(classification_report(y_test, y_pred_knn, target_names=le.classes_, zero_division=0))

    # --- Logistic Regression Classifier ---
    print("\n--- Training Logistic Regression ---")
    log_reg = LogisticRegression(random_state=42, max_iter=1000, multi_class='auto') # Added multi_class='auto' for clarity
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    print("\n--- Logistic Regression Evaluation ---")
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Accuracy: {accuracy_log_reg:.4f}")
    print("Classification Report:"); print(classification_report(y_test, y_pred_log_reg, target_names=le.classes_, zero_division=0))

    # --- Support Vector Machine (SVM) Classifier --- <--- NEW SECTION
    print("\n--- Training Support Vector Machine (SVM) ---")
    # Common parameters: C (regularization), kernel ('rbf', 'linear', 'poly'), gamma (for 'rbf')
    # probability=True allows predict_proba but slows training
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    print("\n--- SVM Evaluation ---")
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Accuracy: {accuracy_svm:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_, zero_division=0))
    # --- End of SVM Section ---

    # --- Save Models and Objects ---
    print("\n--- Saving Models and Preprocessing Objects ---")
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        knn_save_path = SAVE_DIR / 'knn_model.joblib'
        logreg_save_path = SAVE_DIR / 'logreg_model.joblib'
        svm_save_path = SAVE_DIR / 'svm_model.joblib' # <-- Added SVM save path
        scaler_save_path = SAVE_DIR / 'scaler.joblib'
        le_save_path = SAVE_DIR / 'label_encoder.joblib'

        joblib.dump(knn, knn_save_path)
        joblib.dump(log_reg, logreg_save_path)
        joblib.dump(svm, svm_save_path) # <-- Save SVM model
        joblib.dump(scaler, scaler_save_path)
        joblib.dump(le, le_save_path)

        print(f"KNN model saved to: {knn_save_path}")
        print(f"Logistic Regression model saved to: {logreg_save_path}")
        print(f"SVM model saved to: {svm_save_path}") # <-- Added print statement
        print(f"Scaler saved to: {scaler_save_path}")
        print(f"Label Encoder saved to: {le_save_path}")
    except Exception as e:
        print(f"\n--- ERROR saving models ---: {e}\n")


# --- Main Execution ---

if __name__ == "__main__":
    print("="*50)
    print(" Starting EEG Emotion Classification Training ".center(50, "="))
    print("="*50)
    print(f"IMPORTANT: Assuming EEG sample rate is {ASSUMED_SAMPLE_RATE} Hz.")
    print(f"Processing segments of {SEGMENT_DURATION_SECONDS} seconds.")
    print("="*50)

    # 1. Extract Features
    X_features, y_labels = extract_features_from_datasets(DATASET_BASE_DIR)

    # 2. Train, Evaluate, and Save
    if X_features is not None and y_labels is not None:
        if len(X_features) > 10 and len(y_labels.unique()) > 1:
             train_and_evaluate(X_features, y_labels)
        else:
             print("\nNot enough data segments or classes extracted for training.")
             if len(X_features) <= 10: print(f"  Reason: Only {len(X_features)} segments found.")
             if len(y_labels.unique()) <= 1: print(f"  Reason: Only {len(y_labels.unique())} unique class found.")
    else:
        print("\nFeature extraction failed. Cannot proceed.")

    print("\nScript finished.")