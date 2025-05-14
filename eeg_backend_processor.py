# eeg_backend_processor.py
import time
import os
from pathlib import Path
import warnings
import signal
import numpy as np
import pylsl
import zmq # For ZeroMQ
import json # For sending structured data

try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowFunctions
    from brainflow.board_shim import BoardShim # Not for direct use here, but DataFilter might need some constants
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("Backend: Warning: brainflow library not found. PSD calculation will be disabled.")

# --- Configuration (from your original script, adapt as needed) ---
LSL_STREAM_TYPE = 'EEG'
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX = 20 # Can be a bit larger for backend processing
ASSUMED_SAMPLE_RATE = 256.0
EEG_CHANNEL_INDICES = [0, 1, 2, 3]
NUM_CHANNELS_USED = len(EEG_CHANNEL_INDICES)
ENABLE_FILTERING = True and BRAINFLOW_AVAILABLE
FILTER_ORDER = 5
NOTCH_CENTER_FREQ = 50.0 # Adjust
NOTCH_BANDWIDTH = 4.0
BAND_PASS_LOW_CUTOFF = 1.0
BAND_PASS_HIGH_CUTOFF = 45.0

# ZeroMQ Configuration
ZMQ_PUB_ADDRESS = "tcp://*:5556" # Publisher binds to this address

# Classification settings
CLASSIFICATION_BUFFER_SECONDS = 2 # How much data to use for one classification
CLASSIFICATION_INTERVAL_SECONDS = 0.5 # How often to try to classify

# Band definitions
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Alpha/Beta Ratio Threshold for Relaxation (NEEDS CALIBRATION/EXPERIMENTATION)
# Higher ratio (more alpha than beta) -> more relaxed
# This is a CRITICAL parameter to tune.
RELAXATION_THRESHOLD_AB_RATIO = 3.0 # Example: Alpha power >= 3 * Beta power

# --- Global for graceful shutdown ---
running = True

def signal_handler(sig, frame):
    global running
    print("Backend: SIGINT received, shutting down...")
    running = False

class EEGProcessor:
    def __init__(self, config):
        self.config = config
        self.lsl_inlet = None
        self.actual_fs = self.config.get('ASSUMED_SAMPLE_RATE', 256.0)
        self.fs_int = int(self.actual_fs)
        self.classification_buffer_samples = int(self.actual_fs * self.config['CLASSIFICATION_BUFFER_SECONDS'])
        self.eeg_data_buffer = np.empty((self.config['NUM_CHANNELS_USED'], 0))
        # Window for PSD must be <= data_window length
        # Welch's method segments the data, so nperseg should be appropriate
        # A common choice is a power of 2, e.g., same as sampling rate for 1s resolution
        TARGET_FREQ_RESOLUTION = 0.5 # Hz
        self.welch_nperseg = int(self.actual_fs / TARGET_FREQ_RESOLUTION) # e.g., 256 / 0.5 = 512 samples
        
        self.nfft = DataFilter.get_next_power_of_two(self.welch_nperseg) # NFFT points
        # Calculate samples needed per segment for this resolution
        self.welch_overlap = self.welch_nperseg // 2 # 50% overlap is common

        self.classifier = self.load_classifier_model() # Placeholder

        # ZMQ Setup
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(self.config['ZMQ_PUB_ADDRESS'])
        print(f"Backend: Publisher bound to {self.config['ZMQ_PUB_ADDRESS']}")

    def load_classifier_model(self):
        # TODO: Load your actual classifier model
        print("Backend: Classifier loading placeholder.")
        # Example:
        # from joblib import load
        # try:
        #     model = load('path/to/your/model.joblib')
        #     print("Backend: Classifier model loaded.")
        #     return model
        # except Exception as e:
        #     print(f"Backend: Error loading classifier: {e}")
        #     return None
        return "DUMMY_MODEL" # Placeholder

    def connect_lsl(self):
        print(f"Backend: Looking for LSL stream (Type: '{self.config['LSL_STREAM_TYPE']}')...")
        try:
            streams = pylsl.resolve_byprop('type', self.config['LSL_STREAM_TYPE'], 1,
                                           timeout=self.config['LSL_RESOLVE_TIMEOUT'])
            if not streams:
                print("Backend: LSL stream not found.")
                return False
            self.lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=self.config['LSL_CHUNK_MAX'])
            stream_info = self.lsl_inlet.info()
            srate = stream_info.nominal_srate()
            if srate > 0 and abs(srate - self.actual_fs) > 1e-3: # If different enough
                print(f"Backend: Updating sampling rate from {self.actual_fs:.2f} to {srate:.2f} Hz")
                self.actual_fs = srate
                self.fs_int = int(self.actual_fs)
                self.welch_nperseg = int(self.actual_fs*2) # Or a different choice based on new fs_int
                # RECALCULATE PSD PARAMS
                self.nfft = DataFilter.get_next_power_of_two(self.welch_nperseg)
                self.welch_overlap = self.welch_nperseg // 2
                # Also update classification_buffer_samples
                self.classification_buffer_samples = int(self.actual_fs * self.config['CLASSIFICATION_BUFFER_SECONDS'])
            print(f"Backend: Connected to LSL: {stream_info.name()} @ {self.actual_fs:.2f} Hz")
            return True
        except Exception as e:
            print(f"Backend: LSL connection error: {e}")
            return False

    def process_chunk(self, chunk_np):
        if self.config['ENABLE_FILTERING'] and BRAINFLOW_AVAILABLE:
            for i in range(chunk_np.shape[0]):
                if self.config.get('BAND_PASS_LOW_CUTOFF') and self.config.get('BAND_PASS_HIGH_CUTOFF'):
                    DataFilter.perform_bandpass(chunk_np[i, :], self.fs_int, self.config['BAND_PASS_LOW_CUTOFF'],
                                                self.config['BAND_PASS_HIGH_CUTOFF'], self.config['FILTER_ORDER'],
                                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                if self.config.get('NOTCH_CENTER_FREQ'):
                    DataFilter.perform_bandstop(chunk_np[i, :], self.fs_int, self.config['NOTCH_CENTER_FREQ'],
                                               self.config['NOTCH_BANDWIDTH'], self.config['FILTER_ORDER'],
                                               FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        return chunk_np

    def extract_features_and_classify(self, data_window):
        # data_window shape: (n_channels, n_samples)
        # Ensure brainflow is available and data is sufficient
        if not BRAINFLOW_AVAILABLE:
            print("Backend: BrainFlow not available for PSD calculation.")
            return "Error: BrainFlow Missing"
        if data_window.shape[1] < self.welch_nperseg: # Need at least one segment for Welch
            # print(f"Backend: Not enough data for Welch PSD. Have {data_window.shape[1]}, need {self.welch_nperseg}")
            return None # Not enough data yet

        avg_ab_ratios_per_channel = []

        for i in range(data_window.shape[0]): # Iterate over each channel
            channel_data = data_window[i, :]

            # It's good practice to detrend before PSD
            DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value) # Or LINEAR

            # Calculate PSD using Welch method
            # psd_data: [amplitudes, frequencies]
            try:
                psd_data = DataFilter.get_psd_welch(
                    data=channel_data,
                    nfft=self.nfft,
                    nperseg=self.welch_nperseg,
                    noverlap=self.welch_overlap,
                    sampling_rate=int(self.actual_fs), # Ensure this is the correct, potentially updated, sampling rate
                    window=WindowFunctions.HANNING.value # Hanning is a good general window
                )
            except Exception as e: # BrainFlow can raise various errors if data is problematic
                print(f"Backend: Error calculating PSD for channel {i}: {e}")
                continue # Skip this channel if PSD fails

            # Find average power in Alpha band
            alpha_power = DataFilter.get_band_power(
                psd=psd_data,
                freq_start=ALPHA_BAND[0],
                freq_end=ALPHA_BAND[1]
            )

            # Find average power in Beta band
            beta_power = DataFilter.get_band_power(
                psd=psd_data,
                freq_start=BETA_BAND[0],
                freq_end=BETA_BAND[1]
            )

            if beta_power > 1e-10: # Avoid division by zero or very small numbers
                ab_ratio = alpha_power / beta_power
                avg_ab_ratios_per_channel.append(ab_ratio)
            else:
                # Handle cases where beta power is zero or too low
                # Could append a very high ratio, or skip, or set to a default
                avg_ab_ratios_per_channel.append(0) # Or some other indicator

        if not avg_ab_ratios_per_channel:
            print("Backend: Could not calculate A/B ratio for any channel.")
            return "Error: PSD Failed"

        # Combine ratios from channels (e.g., average)
        # You might want to select specific channels known for good alpha/beta activity
        # For example, if EEG_CHANNEL_INDICES maps to [TP9, AF7, AF8, TP10],
        # parietal (TP9, TP10) might be better for alpha.
        # For now, we average all selected channels' ratios.
        overall_ab_ratio = np.mean(avg_ab_ratios_per_channel)
        # print(f"Backend: Overall Alpha/Beta Ratio: {overall_ab_ratio:.2f}")

        # Apply threshold for classification
        if overall_ab_ratio >= self.config.get('RELAXATION_THRESHOLD_AB_RATIO', RELAXATION_THRESHOLD_AB_RATIO):
            prediction = "Relaxed"
        else:
            prediction = "Not Relaxed" # Or "Engaged", "Neutral", etc.

        # For database and UI, you might want to send the ratio too
        # self.publisher.send_json({"prediction": prediction, "ab_ratio": overall_ab_ratio})
        # For now, just returning the label as per the existing structure
        return prediction

    def run(self):
        global running
        if not self.connect_lsl():
            print("Backend: Failed to connect to LSL. Exiting.")
            return

        last_classification_time = time.time()

        while running:
            try:
                chunk, ts = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=self.config['LSL_CHUNK_MAX'])
                if chunk:
                    chunk_np = np.array(chunk, dtype=np.float64).T
                    chunk_np = chunk_np[self.config['EEG_CHANNEL_INDICES'], :]
                    processed_chunk = self.process_chunk(chunk_np.copy())

                    self.eeg_data_buffer = np.append(self.eeg_data_buffer, processed_chunk, axis=1)
                    max_buffer_len = self.classification_buffer_samples + int(self.actual_fs * 0.5)
                    if self.eeg_data_buffer.shape[1] > max_buffer_len:
                        self.eeg_data_buffer = self.eeg_data_buffer[:, -max_buffer_len:]

                    current_time = time.time()
                    if self.eeg_data_buffer.shape[1] >= self.classification_buffer_samples and \
                       (current_time - last_classification_time >= self.config['CLASSIFICATION_INTERVAL_SECONDS']):
                        data_for_classification = self.eeg_data_buffer[:, -self.classification_buffer_samples:]
                        prediction = self.extract_features_and_classify(data_for_classification)
                        if prediction:
                            message = {"timestamp": time.time(), "prediction": prediction}
                            self.publisher.send_json(message) # Send as JSON
                            # print(f"Backend: Published: {message}")
                        last_classification_time = current_time
                else:
                    # If LSL stream is slow or quiet, this allows the loop to check `running` flag
                    time.sleep(0.005)


            except pylsl.LostError:
                print("Backend: LSL connection lost. Attempting to reconnect...")
                self.lsl_inlet.close_stream() # Close broken inlet
                time.sleep(2) # Wait before retrying
                if not self.connect_lsl(): # Try to reconnect
                    print("Backend: Reconnect failed. Exiting.")
                    running = False # Stop if reconnect fails
            except Exception as e:
                print(f"Backend: Error in run loop: {e}")
                time.sleep(0.1) # Brief pause

        print("Backend: EEGProcessor run loop finished.")
        if self.lsl_inlet:
            self.lsl_inlet.close_stream()
        self.publisher.close()
        self.context.term()
        print("Backend: ZMQ resources cleaned up.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler) # Handle Ctrl+C

    config = {
        'LSL_STREAM_TYPE': LSL_STREAM_TYPE,
        'LSL_RESOLVE_TIMEOUT': LSL_RESOLVE_TIMEOUT,
        'LSL_CHUNK_MAX': LSL_CHUNK_MAX,
        'ASSUMED_SAMPLE_RATE': ASSUMED_SAMPLE_RATE,
        'EEG_CHANNEL_INDICES': EEG_CHANNEL_INDICES,
        'NUM_CHANNELS_USED': NUM_CHANNELS_USED,
        'ENABLE_FILTERING': ENABLE_FILTERING,
        'FILTER_ORDER': FILTER_ORDER,
        'NOTCH_CENTER_FREQ': NOTCH_CENTER_FREQ,
        'NOTCH_BANDWIDTH': NOTCH_BANDWIDTH,
        'BAND_PASS_LOW_CUTOFF': BAND_PASS_LOW_CUTOFF,
        'BAND_PASS_HIGH_CUTOFF': BAND_PASS_HIGH_CUTOFF,
        'ZMQ_PUB_ADDRESS': ZMQ_PUB_ADDRESS,
        'CLASSIFICATION_BUFFER_SECONDS': CLASSIFICATION_BUFFER_SECONDS,
        'CLASSIFICATION_INTERVAL_SECONDS': CLASSIFICATION_INTERVAL_SECONDS,
        'RELAXATION_THRESHOLD_AB_RATIO': RELAXATION_THRESHOLD_AB_RATIO,
    }
    processor = EEGProcessor(config)
    processor.run()
    print("Backend: Application shut down.")