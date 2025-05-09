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
    from brainflow.data_filter import DataFilter, FilterTypes
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("Warning: brainflow library not found. Filtering will be disabled.")

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
            if srate > 0:
                self.actual_fs = srate
                self.fs_int = int(self.actual_fs)
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
        # TODO: Implement your actual feature extraction and classification
        if self.classifier is None or data_window.shape[1] < self.classification_buffer_samples:
            return None # Not enough data or no classifier

        # Placeholder classification logic
        activity_level = np.mean(np.std(data_window, axis=1))
        if activity_level < 5:
            prediction = "Relaxed"
        elif activity_level < 15:
            prediction = "Neutral"
        else:
            prediction = "Agitated/Focused"
        # print(f"Backend: Classified as {prediction} (activity: {activity_level:.2f})")
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
    }
    processor = EEGProcessor(config)
    processor.run()
    print("Backend: Application shut down.")