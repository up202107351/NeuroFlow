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
CLASSIFICATION_BUFFER_SECONDS = 4 # How much data to use for one classification
CLASSIFICATION_INTERVAL_SECONDS = 1 # How often to try to classify

CALIBRATION_DURATION_SECONDS = 60 # Duration of baseline recording
# Thresholds for detecting change from baseline (these still need tuning)
RELAXATION_ALPHA_INCREASE_FACTOR = 1.2 # e.g., 20% increase in Alpha over baseline
RELAXATION_AB_RATIO_INCREASE_FACTOR = 1.15 # e.g., 15% increase in A/B ratio

FOCUS_BETA_THETA_RATIO_INCREASE_FACTOR = 1.2
FOCUS_BETA_INCREASE_FACTOR = 1.15
FOCUS_ALPHA_DECREASE_FACTOR = 0.8 # e.g., Alpha drops to 80% of baseline

# Band definitions
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0) # Could split into low/high beta (e.g., 13-20, 20-30)

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

        # Calibration settings
        self.baseline_metrics = {} # To store calculated baseline values (e.g., {'alpha': 10, 'beta': 5, ...})
        self.is_calibrated = False
        self.calibration_data_accumulator = [] # List to store band powers during calibration

        # ZMQ Setup
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(self.config['ZMQ_PUB_ADDRESS'])
        print(f"Backend: Publisher bound to {self.config['ZMQ_PUB_ADDRESS']}")

    def calculate_band_powers(self, data_window_channel):
        """Calculates Theta, Alpha, Beta powers for a single channel window."""
        if not BRAINFLOW_AVAILABLE or data_window_channel.shape[0] < self.welch_nperseg:
            return None

        DataFilter.detrend(data_window_channel, DetrendOperations.CONSTANT.value)
        try:
            psd_data = DataFilter.get_psd_welch(data_window_channel, self.nfft, self.welch_nperseg,
                                                self.welch_overlap, int(self.actual_fs), WindowFunctions.HANNING.value)
        except Exception:
            return None # PSD calculation failed

        theta_power = DataFilter.get_band_power(psd_data, THETA_BAND[0], THETA_BAND[1])
        alpha_power = DataFilter.get_band_power(psd_data, ALPHA_BAND[0], ALPHA_BAND[1])
        beta_power = DataFilter.get_band_power(psd_data, BETA_BAND[0], BETA_BAND[1])
        return {"theta": theta_power, "alpha": alpha_power, "beta": beta_power}
    
    def perform_calibration(self):
        """Collects data for calibration_duration and calculates baseline metrics."""
        print(f"Backend: Starting {self.config['CALIBRATION_DURATION_SECONDS']}s calibration...")
        # UI should be showing "Calibrating..."
        # Inform UI that calibration started (e.g., via a ZMQ message)
        self.publisher.send_json({"status": "calibration_started", "duration": self.config['CALIBRATION_DURATION_SECONDS']})


        calibration_start_time = time.time()
        all_channel_powers_during_calibration = [] # List of dicts [{ch0_powers}, {ch1_powers}, ...] per time window

        temp_buffer = np.empty((self.config['NUM_CHANNELS_USED'], 0))
        samples_needed_for_psd_window = self.welch_nperseg # Use a window size similar to classification

        while time.time() - calibration_start_time < self.config['CALIBRATION_DURATION_SECONDS']:
            if not running: break # Allow shutdown during calibration
            chunk, _ = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=self.config['LSL_CHUNK_MAX'])
            if chunk:
                chunk_np = np.array(chunk, dtype=np.float64).T
                chunk_np = chunk_np[self.config['EEG_CHANNEL_INDICES'], :]
                processed_chunk = self.process_chunk(chunk_np.copy()) # Apply filters

                temp_buffer = np.append(temp_buffer, processed_chunk, axis=1)

                while temp_buffer.shape[1] >= samples_needed_for_psd_window:
                    current_psd_window = temp_buffer[:, :samples_needed_for_psd_window]
                    # Slide window by half its length for overlap, or by its full length for non-overlapping
                    temp_buffer = temp_buffer[:, samples_needed_for_psd_window // 2:] # 50% overlap

                    window_channel_powers = [] # Powers for all channels for this single time window
                    for ch_idx in range(current_psd_window.shape[0]):
                        powers = self.calculate_band_powers(current_psd_window[ch_idx, :])
                        if powers:
                            window_channel_powers.append(powers)
                    if window_channel_powers and len(window_channel_powers) == self.config['NUM_CHANNELS_USED']:
                        all_channel_powers_during_calibration.append(window_channel_powers)
            else:
                time.sleep(0.01)

        if not all_channel_powers_during_calibration:
            print("Backend: No data collected during calibration. Calibration failed.")
            self.publisher.send_json({"status": "calibration_failed", "reason": "No data"})
            self.is_calibrated = False
            return

        # Calculate average powers across all collected windows and channels
        # This creates a list of N dicts, where N is num_channels. Each dict has avg theta, alpha, beta.
        avg_powers_per_channel_list = []
        for ch_idx in range(self.config['NUM_CHANNELS_USED']):
            ch_theta = np.mean([win[ch_idx]['theta'] for win in all_channel_powers_during_calibration if len(win) > ch_idx])
            ch_alpha = np.mean([win[ch_idx]['alpha'] for win in all_channel_powers_during_calibration if len(win) > ch_idx])
            ch_beta  = np.mean([win[ch_idx]['beta']  for win in all_channel_powers_during_calibration if len(win) > ch_idx])
            avg_powers_per_channel_list.append({'theta': ch_theta, 'alpha': ch_alpha, 'beta': ch_beta})

        # For overall baseline, you might average across specific channels or all
        # Example: average of all channels for each band
        self.baseline_metrics['avg_theta'] = np.mean([p['theta'] for p in avg_powers_per_channel_list])
        self.baseline_metrics['avg_alpha'] = np.mean([p['alpha'] for p in avg_powers_per_channel_list])
        self.baseline_metrics['avg_beta']  = np.mean([p['beta']  for p in avg_powers_per_channel_list])

        if self.baseline_metrics['avg_beta'] > 1e-9:
            self.baseline_metrics['ab_ratio'] = self.baseline_metrics['avg_alpha'] / self.baseline_metrics['avg_beta']
        else:
            self.baseline_metrics['ab_ratio'] = 0 # Or some large number

        if self.baseline_metrics['avg_theta'] > 1e-9:
            self.baseline_metrics['bt_ratio'] = self.baseline_metrics['avg_beta'] / self.baseline_metrics['avg_theta']
        else:
            self.baseline_metrics['bt_ratio'] = 0

        self.is_calibrated = True
        print(f"Backend: Calibration complete. Baselines: {self.baseline_metrics}")
        self.publisher.send_json({"status": "calibration_complete", "baselines": self.baseline_metrics})

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
        if not self.is_calibrated:
            return "Not Calibrated"
        if not BRAINFLOW_AVAILABLE:
            return "Error: BrainFlow Missing"
        if data_window.shape[1] < self.welch_nperseg:
            return None

        current_channel_powers_list = []
        for i in range(data_window.shape[0]):
            powers = self.calculate_band_powers(data_window[i, :])
            if powers:
                current_channel_powers_list.append(powers)

        if not current_channel_powers_list or len(current_channel_powers_list) != data_window.shape[0]:
            return "Error: PSD Failed Current"

        # Calculate current average band powers (similar to baseline calculation, but for current window)
        current_avg_theta = np.mean([p['theta'] for p in current_channel_powers_list])
        current_avg_alpha = np.mean([p['alpha'] for p in current_channel_powers_list])
        current_avg_beta  = np.mean([p['beta']  for p in current_channel_powers_list])

        current_ab_ratio = current_avg_alpha / current_avg_beta if current_avg_beta > 1e-9 else 0
        current_bt_ratio = current_avg_beta / current_avg_theta if current_avg_theta > 1e-9 else 0

        # --- Classification based on change from baseline ---
        # This logic needs to know the GOAL (Relaxation or Focus)
        # For now, let's assume a generic output, UI can interpret based on session type
        # Or, the UI needs to tell the backend the current session goal.

        # Relaxation indicators
        alpha_increased = current_avg_alpha > self.baseline_metrics.get('avg_alpha', 0) * self.config.get('RELAXATION_ALPHA_INCREASE_FACTOR', 1.2)
        ab_ratio_increased = current_ab_ratio > self.baseline_metrics.get('ab_ratio', 0) * self.config.get('RELAXATION_AB_RATIO_INCREASE_FACTOR', 1.15)

        # Focus indicators
        bt_ratio_increased = current_bt_ratio > self.baseline_metrics.get('bt_ratio', 0) * self.config.get('FOCUS_BETA_THETA_RATIO_INCREASE_FACTOR', 1.2)
        beta_increased = current_avg_beta > self.baseline_metrics.get('avg_beta', 0) * self.config.get('FOCUS_BETA_INCREASE_FACTOR', 1.15)
        alpha_decreased = current_avg_alpha < self.baseline_metrics.get('avg_alpha', float('inf')) * self.config.get('FOCUS_ALPHA_DECREASE_FACTOR', 0.8)


        # Example: Determine primary state (this logic can be more sophisticated)
        # The backend could send all these flags, or make a primary decision.
        # This needs to be tied to the *type* of session (Meditation vs Focus)
        # For simplicity, let's assume a 'general' state for now.
        # This should be refined based on the active session type (Relaxation or Focus)
        # that the UI would ideally tell the backend.

        prediction = "Neutral" # Default
        if alpha_increased and ab_ratio_increased: # Strong relaxation indicators
            prediction = "Relaxed"
        elif bt_ratio_increased and beta_increased and alpha_decreased: # Strong focus indicators
            prediction = "Focused"
        elif bt_ratio_increased and beta_increased: # Moderate focus
            prediction = "Likely Focused"
        elif alpha_increased : # Moderate relaxation
            prediction = "Likely Relaxed"


        # Return a richer dictionary with current values and flags
        return {
            "prediction_label": prediction,
            "current_alpha": current_avg_alpha,
            "current_beta": current_avg_beta,
            "current_theta": current_avg_theta,
            "current_ab_ratio": current_ab_ratio,
            "current_bt_ratio": current_bt_ratio,
            "flags": {
                "alpha_increased": alpha_increased,
                "ab_ratio_increased": ab_ratio_increased,
                "bt_ratio_increased": bt_ratio_increased,
                "beta_increased": beta_increased,
                "alpha_decreased": alpha_decreased
            }
        }
    
    def run(self):
        global running
        if not self.connect_lsl():
            print("Backend: Failed to connect to LSL. Exiting.")
            return

        # --- Perform Calibration ---
        # In a real app, UI would trigger this, or it happens based on session type
        # For now, let's assume it always calibrates at the start of this backend process
        self.perform_calibration()
        if not self.is_calibrated:
            print("Backend: Calibration failed. Cannot proceed with classification.")
            # Loop here sending "Not Calibrated" or exit
            while running:
                self.publisher.send_json({"status": "error", "prediction": "Not Calibrated"}) # Changed "prediction" to "message" for clarity
                time.sleep(1)
                # Check LSL stream
                try:
                    # Use a non-blocking pull or short timeout to just check stream presence
                    if not self.lsl_inlet.pull_sample(timeout=0.001)[0]: # pull_sample is often better for just checking
                        print("Backend: LSL stream lost while waiting in uncalibrated state.")
                        running = False # Set global running to false to exit outer loop too
                        break
                except pylsl.LostError: # Catch LostError specifically
                    print("Backend: LSL stream lost (LostError) while waiting in uncalibrated state.")
                    running = False
                    break
                except Exception as e_lsl_check: # Catch other potential errors during the check
                    print(f"Backend: Error checking LSL stream in uncalibrated state: {e_lsl_check}")
                    # Decide if this should also cause an exit
                    time.sleep(0.5) # Avoid busy loop on repeated errors
            self.lsl_inlet.close_stream()
            self.publisher.close()
            self.context.term()
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

                        # Call the classification method
                        classification_result = self.extract_features_and_classify(data_for_classification)

                        if classification_result is not None: # Only publish if we have a result (dict or string)
                            message = {"timestamp": time.time()} # Base message with timestamp

                            if isinstance(classification_result, dict):
                                # Successful classification, result is a dictionary of metrics
                                message["type"] = "prediction"
                                message.update(classification_result) # Merge the classification dict
                                # Example: message will be {"timestamp": ..., "type": "prediction", "prediction_label": "Relaxed", "current_alpha": ...}
                            elif isinstance(classification_result, str):
                                # It's an informational message or an error string from classification
                                message["type"] = "info" # Or "error" depending on string content
                                message["message"] = classification_result
                                # Example: message will be {"timestamp": ..., "type": "info", "message": "Not Calibrated"}
                            else:
                                # Should not happen if extract_features_and_classify adheres to dict/str/None return
                                print(f"Backend: Unexpected classification result type: {type(classification_result)}")
                                continue # Skip publishing this odd result

                            self.publisher.send_json(message)
                            # print(f"Backend: Published: {message}") # Uncomment for debugging

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
        'CALIBRATION_DURATION_SECONDS': CALIBRATION_DURATION_SECONDS,
        'RELAXATION_ALPHA_INCREASE_FACTOR': RELAXATION_ALPHA_INCREASE_FACTOR,
        'RELAXATION_AB_RATIO_INCREASE_FACTOR': RELAXATION_AB_RATIO_INCREASE_FACTOR,
        'FOCUS_BETA_THETA_RATIO_INCREASE_FACTOR': FOCUS_BETA_THETA_RATIO_INCREASE_FACTOR,
        'FOCUS_BETA_INCREASE_FACTOR': FOCUS_BETA_INCREASE_FACTOR,
        'FOCUS_ALPHA_DECREASE_FACTOR': FOCUS_ALPHA_DECREASE_FACTOR,
    }
    processor = EEGProcessor(config)
    processor.run()
    print("Backend: Application shut down.")