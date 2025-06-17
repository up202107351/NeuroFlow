#!/usr/bin/env python3
"""
EEG Processing Worker - Qt Threading-based EEG Processing

Updated to handle both single-stream (BlueMuse) and multi-stream (LSL simulator) scenarios.
"""

import time
import numpy as np
import pylsl
import logging
from datetime import datetime
from scipy.signal import butter, filtfilt, welch
from PyQt5 import QtCore
from backend.signal_quality_validator import SignalQualityValidator
from backend import database_manager as db_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EEG_Worker')

try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logger.error("BrainFlow library not found. Please install it (pip install brainflow).")

# --- Configuration ---
LSL_STREAM_TYPE = 'EEG'
LSL_ACCELEROMETER_STREAM_TYPE = 'Accelerometer'
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX_PULL = 128

EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10 for Muse
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

# Accelerometer channel indices for single-stream setup (BlueMuse)
ACC_CHANNEL_INDICES_SINGLE_STREAM = [9, 10, 11]

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0
PSD_WINDOW_SECONDS = 6.0

DEFAULT_SAMPLING_RATE = 256.0

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Session types
SESSION_TYPE_RELAX = "RELAXATION"
SESSION_TYPE_FOCUS = "FOCUS"

# Design Butterworth bandpass filter (0.5 - 30 Hz)
filter_order = 4
lowcut = 0.5
highcut = 30.0

class EEGProcessingWorker(QtCore.QObject):
    """
    EEG Processing Worker that runs in a separate thread.
    Handles all EEG processing, data accumulation, and database saving.
    Now supports both single-stream and multi-stream LSL setups.
    """
    
    # Signals for UI communication
    connection_status_changed = QtCore.pyqtSignal(str, str)  # status, message
    calibration_progress = QtCore.pyqtSignal(float)  # 0.0 to 1.0
    calibration_status_changed = QtCore.pyqtSignal(str, dict)  # status, data
    new_prediction = QtCore.pyqtSignal(dict)  # prediction data for UI feedback only
    signal_quality_update = QtCore.pyqtSignal(dict)  # real-time quality metrics
    error_occurred = QtCore.pyqtSignal(str)  # error message
    session_saved = QtCore.pyqtSignal(int, dict)  # session_id, summary_stats
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Connection state
        self.lsl_inlet = None
        self.lsl_accelerometer_inlet = None  # NEW: Separate accelerometer inlet
        self.is_multi_stream = False  # NEW: Track if we're using multi-stream setup
        self.sampling_rate = DEFAULT_SAMPLING_RATE
        self.running = True
        
        # Session state
        self.is_calibrated = False
        self.is_calibrating = False
        self.current_session_type = None
        self.current_session_id = None
        self.session_start_time = None
        
        # EEG processing state
        self.baseline_metrics = None
        self.recent_metrics_history = []
        self.previous_states = []
        self.channel_weights = np.ones(NUM_EEG_CHANNELS) / NUM_EEG_CHANNELS
        
        # Buffer management
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        self.buffer_timestamps = []
        
        # Signal quality validator
        self.signal_quality_validator = SignalQualityValidator()
        
        # Filter coefficients
        self.b = None
        self.a = None
        self.nfft = DataFilter.get_nearest_power_of_two(int(DEFAULT_SAMPLING_RATE * PSD_WINDOW_SECONDS))
        self.welch_overlap_samples = self.nfft // 2
        
        # Processing timers
        self.processing_timer = QtCore.QTimer()
        self.processing_timer.timeout.connect(self._process_eeg_data)
        
        # SESSION DATA ACCUMULATION
        self.session_predictions = []
        self.session_on_target = []
        self.session_timestamps = []
        self.session_confidence_scores = []
        
        # Band power data accumulation
        self.session_band_data = {
            "alpha": [],
            "beta": [],
            "theta": [],
            "ab_ratio": [],
            "bt_ratio": [],
            "timestamps": []
        }
        
        # EEG data accumulation (downsampled for storage)
        self.session_eeg_data = {
            "channel_0": [],
            "channel_1": [],
            "channel_2": [],
            "channel_3": [],
            "timestamps": []
        }
        
        # Prediction smoothing and stabilization
        self.prediction_smoothing_window = 5
        self.min_confidence_for_change = 0.6
        self.state_change_threshold = 1.0
        self.last_stable_prediction = None
        self.stable_prediction_count = 0
        self.min_stable_count = 3
        
        # Prediction history for smoothing
        self.prediction_history = []
        
        # Current prediction state
        self.current_prediction = {
            'state': 'Unknown', 'level': 0, 'confidence': 0.0, 'smooth_value': 0.5
        }
        
        # State tracking with momentum
        self.state_momentum = 0.75
        self.state_velocity = 0.0
        self.level_momentum = 0.8
        self.level_velocity = 0.0
        
        # State thresholds
        self.current_thresholds = {
            'alpha_incr': [1.10, 1.25, 1.50],
            'alpha_decr': [0.90, 0.80, 0.65],
            'beta_incr': [1.10, 1.20, 1.40],
            'beta_decr': [0.90, 0.80],
            'theta_incr': [1.15, 1.30],
            'theta_decr': [0.85, 0.70],
            'ab_ratio_incr': [1.10, 1.20, 1.40],
            'ab_ratio_decr': [0.90, 0.75],
            'bt_ratio_incr': [1.15, 1.30, 1.60],
            'bt_ratio_decr': [0.85, 0.70]
        }
    
    @QtCore.pyqtSlot()
    def initialize(self):
        """Initialize the worker (called when moved to thread)"""
        self.running = True
        logger.info("EEG Processing Worker initialized")
        
    @QtCore.pyqtSlot()
    def connect_to_lsl(self):
        """Connect to the LSL stream(s) - Updated to handle multi-stream"""
        logger.info("Looking for LSL streams...")
        
        try:
            self.connection_status_changed.emit("CONNECTING", "Looking for EEG stream...")
            
            # First, try to find EEG stream
            eeg_streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
            if not eeg_streams:
                error_msg = "EEG LSL stream not found."
                logger.error(error_msg)
                self.connection_status_changed.emit("ERROR", error_msg)
                self.error_occurred.emit(error_msg)
                return False
                
            self.lsl_inlet = pylsl.StreamInlet(eeg_streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
            info = self.lsl_inlet.info()
            lsl_sr = info.nominal_srate()
            self.sampling_rate = lsl_sr if lsl_sr > 0 else DEFAULT_SAMPLING_RATE
            
            # Update filter coefficients
            self._update_filter_coefficients()
            
            device_name = info.name()
            logger.info(f"Connected to EEG stream '{device_name}' @ {self.sampling_rate:.2f} Hz")
            
            # Check if EEG stream has accelerometer data (single-stream setup)
            if info.channel_count() > max(ACC_CHANNEL_INDICES_SINGLE_STREAM):
                logger.info("Single-stream setup detected (EEG stream includes accelerometer data)")
                self.is_multi_stream = False
            else:
                logger.info("Multi-stream setup detected, looking for separate accelerometer stream...")
                self.is_multi_stream = True
                
                # Try to find separate accelerometer stream
                acc_streams = pylsl.resolve_byprop('type', LSL_ACCELEROMETER_STREAM_TYPE, 1, timeout=2.0)
                if acc_streams:
                    self.lsl_accelerometer_inlet = pylsl.StreamInlet(acc_streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
                    acc_info = self.lsl_accelerometer_inlet.info()
                    logger.info(f"Connected to accelerometer stream '{acc_info.name()}' @ {acc_info.nominal_srate():.2f} Hz")
                else:
                    logger.warning("No accelerometer stream found - signal quality assessment will be limited")
            
            self.connection_status_changed.emit("CONNECTED", f"Connected to {device_name}")
            
            # Verify EEG stream has sufficient channels
            if info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
                error_msg = f"EEG stream has insufficient channels. Need at least {np.max(EEG_CHANNEL_INDICES) + 1}, got {info.channel_count()}"
                logger.error(error_msg)
                self.connection_status_changed.emit("ERROR", error_msg)
                self.error_occurred.emit(error_msg)
                return False
                
            return True
            
        except Exception as e:
            error_msg = f"Error connecting to LSL: {e}"
            logger.error(error_msg)
            self.connection_status_changed.emit("ERROR", error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def _get_accelerometer_data(self):
        """Get accelerometer data from appropriate source"""
        if self.is_multi_stream and self.lsl_accelerometer_inlet:
            # Multi-stream: get from separate accelerometer stream
            try:
                acc_chunk, acc_timestamps = self.lsl_accelerometer_inlet.pull_chunk(timeout=0.1, max_samples=10)
                if acc_chunk:
                    acc_chunk_np = np.array(acc_chunk, dtype=np.float64).T
                    if acc_chunk_np.shape[1] > 0:
                        return acc_chunk_np[:, -1]  # Return latest sample
            except Exception as e:
                # Non-critical error - accelerometer data is optional
                pass
        return None
    
    def _extract_accelerometer_from_eeg_chunk(self, chunk_np):
        """Extract accelerometer data from EEG chunk (single-stream setup)"""
        if not self.is_multi_stream and chunk_np.shape[0] > max(ACC_CHANNEL_INDICES_SINGLE_STREAM):
            try:
                acc_chunk = chunk_np[ACC_CHANNEL_INDICES_SINGLE_STREAM, :]
                if acc_chunk.shape[1] > 0:
                    return acc_chunk[:, -1]  # Return latest sample
            except Exception as e:
                # Non-critical error
                pass
        return None
    
    def _update_filter_coefficients(self):
        """Update filter coefficients based on current sampling rate"""
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(filter_order, [low, high], btype='band', analog=False)
        self.nfft = DataFilter.get_nearest_power_of_two(int(self.sampling_rate * PSD_WINDOW_SECONDS))
        self.welch_overlap_samples = self.nfft // 2
    
    @QtCore.pyqtSlot(str, int)
    def start_session(self, session_type, session_id):
        """Start a new session with session ID from page widget"""
        if session_type not in [SESSION_TYPE_RELAX, SESSION_TYPE_FOCUS]:
            error_msg = f"Invalid session type: {session_type}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return
            
        logger.info(f"Starting {session_type} session with ID {session_id}")
        
        # Reset state
        self.is_calibrated = False
        self.is_calibrating = False
        self.running = True
        self.current_session_type = session_type
        self.current_session_id = session_id
        self.session_start_time = None
        self.recent_metrics_history = []
        self.previous_states = []
        self.signal_quality_validator.reset()
        
        # Reset prediction smoothing state
        self.prediction_history = []
        self.last_stable_prediction = None
        self.stable_prediction_count = 0
        self.current_prediction = {
            'state': 'Unknown', 'level': 0, 'confidence': 0.0, 'smooth_value': 0.5
        }
        
        # Clear session data
        self._clear_session_data()
        
        # Connect to LSL if not already connected
        if not self.lsl_inlet:
            if not self.connect_to_lsl():
                error_msg = "Failed to connect to LSL. Session aborted."
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return
        
        # Start calibration
        QtCore.QTimer.singleShot(100, self._start_calibration)

    def _start_calibration(self):
        """Start the calibration process"""
        if not self.current_session_type:
            self.error_occurred.emit("Cannot start calibration: No active session")
            return
        
        # Ensure LSL connection first
        if not self.lsl_inlet:
            print("EEG Worker: No LSL connection, attempting to connect...")
            self.connection_status_changed.emit("CONNECTING", "Connecting to EEG device...")
            
            if not self.connect_to_lsl():
                error_msg = "Failed to connect to LSL stream. Please check your Muse connection."
                print(f"EEG Worker: {error_msg}")
                self.connection_status_changed.emit("ERROR", error_msg)
                self.calibration_status_changed.emit("FAILED", {
                    "error_message": error_msg,
                    "timestamp": time.time()
                })
                return
        
        # Test LSL connection
        try:
            print("EEG Worker: Testing LSL connection...")
            test_chunk, _ = self.lsl_inlet.pull_chunk(timeout=2.0, max_samples=10)
            if not test_chunk:
                error_msg = "LSL stream is not providing data. Please check your Muse device."
                print(f"EEG Worker: {error_msg}")
                self.connection_status_changed.emit("ERROR", error_msg)
                self.calibration_status_changed.emit("FAILED", {
                    "error_message": error_msg,
                    "timestamp": time.time()
                })
                return
            else:
                print(f"EEG Worker: LSL connection test successful, got {len(test_chunk)} samples")
        except Exception as e:
            error_msg = f"LSL connection test failed: {e}"
            print(f"EEG Worker: {error_msg}")
            self.connection_status_changed.emit("ERROR", error_msg)
            self.calibration_status_changed.emit("FAILED", {
                "error_message": error_msg,
                "timestamp": time.time()
            })
            return
        
        logger.info(f"Starting calibration for {self.current_session_type}")
        
        self.is_calibrating = True
        self.is_calibrated = False
        
        # Clear buffer
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        
        # Emit calibration start
        self.calibration_status_changed.emit("STARTED", {
            "session_type": self.current_session_type,
            "timestamp": time.time()
        })
        
        # Add a small delay to ensure UI is ready
        QtCore.QTimer.singleShot(500, self._perform_calibration)
    
    @QtCore.pyqtSlot()
    def stop_session(self):
        """Stop the current session and save all data to database"""
        logger.info("Stopping session and saving data to database")
        
        # Stop processing
        if self.processing_timer.isActive():
            self.processing_timer.stop()
        
        # Save all session data to database
        if self.current_session_id and self._has_session_data():
            try:
                summary_stats = self._save_session_to_database()
                self.session_saved.emit(self.current_session_id, summary_stats)
                print(f"EEG Worker: Successfully saved session {self.current_session_id} to database")
            except Exception as e:
                error_msg = f"Error saving session data to database: {e}"
                print(f"EEG Worker: {error_msg}")
                self.error_occurred.emit(error_msg)
        else:
            print("EEG Worker: No session data to save")
        
        # Reset state
        self.current_session_type = None
        self.current_session_id = None
        self.session_start_time = None
        self.is_calibrated = False
        self.is_calibrating = False
        self.running = False
        
        # Clear buffers
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        self._clear_session_data()
        
        self.calibration_status_changed.emit("STOPPED", {})
    
    @QtCore.pyqtSlot()
    def recalibrate(self):
        """Restart calibration process"""
        logger.info("Recalibrating EEG baseline")
        
        if not self.current_session_type:
            self.error_occurred.emit("No active session to recalibrate")
            return
        
        # Reset calibration state
        self.is_calibrated = False
        self.is_calibrating = False
        self.signal_quality_validator.reset()
        self.recent_metrics_history = []
        self.previous_states = []
        
        # Reset prediction smoothing state
        self.prediction_history = []
        self.last_stable_prediction = None
        self.stable_prediction_count = 0
        
        # Clear buffer
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        
        # Restart calibration
        QtCore.QTimer.singleShot(100, self._start_calibration)   
    
    def _perform_calibration(self):
        """Perform calibration with signal quality monitoring - Updated for multi-stream"""
        print(f"EEG Worker: _perform_calibration starting - running: {self.running}, is_calibrating: {self.is_calibrating}")
    
        if not self.running:
            print("EEG Worker: Cannot start calibration - worker is not running!")
            self.calibration_status_changed.emit("FAILED", {
                "error_message": "Worker not in running state",
                "timestamp": time.time()
            })
            return
        
        calibration_start_time = time.time()
        calibration_metrics_list = []
        quality_pause_time = 0
        max_quality_pause = 30.0
        
        # Debug counters
        total_chunks_received = 0
        total_samples_received = 0
        chunks_with_no_data = 0
        chunks_processed = 0
        chunks_failed_processing = 0
        last_quality_update = 0
        
        logger.info("Calibration data collection started")
        print(f"EEG Worker: Starting calibration data collection for {CALIBRATION_DURATION_SECONDS} seconds")
        
        while (time.time() - calibration_start_time < CALIBRATION_DURATION_SECONDS and 
            self.is_calibrating and self.running):
            
            current_time = time.time()
            elapsed_time = current_time - calibration_start_time
                      
            # Get EEG data chunk
            try:
                chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
                total_chunks_received += 1
            except Exception as e:
                logger.error(f"Error pulling LSL chunk during calibration: {e}")
                print(f"EEG Worker: LSL pull error: {e}")
                self.calibration_status_changed.emit("FAILED", {
                    "error_message": f"LSL data error during calibration: {e}",
                    "timestamp": time.time()
                })
                self.is_calibrating = False
                return
            
            if not chunk:
                chunks_with_no_data += 1
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.01)
                continue
                
            # Process EEG chunk
            chunk_np = np.array(chunk, dtype=np.float64).T
            total_samples_received += chunk_np.shape[1]
            
            # Check if we have enough channels for EEG
            if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
                print(f"EEG Worker: Not enough channels in chunk: {chunk_np.shape[0]} <= {max(EEG_CHANNEL_INDICES)}")
                chunks_failed_processing += 1
                continue
                
            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
            
            # Get accelerometer data from appropriate source
            acc_sample = None
            if self.is_multi_stream:
                # Multi-stream: get from separate accelerometer stream
                acc_sample = self._get_accelerometer_data()
            else:
                # Single-stream: extract from EEG chunk
                acc_sample = self._extract_accelerometer_from_eeg_chunk(chunk_np)
            
            # Add accelerometer data to signal quality validator if available
            if acc_sample is not None:
                self.signal_quality_validator.add_accelerometer_data(acc_sample)
            
            # Add to EEG buffer
            self.eeg_buffer = np.append(self.eeg_buffer, eeg_chunk, axis=1)
            
            # Process if we have enough data
            if self.eeg_buffer.shape[1] >= self.nfft:
                chunks_processed += 1
                
                eeg_window = self.eeg_buffer[:, -self.nfft:]
                
                try:
                    filtered_window = self._filter_eeg_data(eeg_window)
                    metrics = self._calculate_band_powers(filtered_window)
                    
                    if metrics:
                        # Add to signal quality validator
                        self.signal_quality_validator.add_band_power_data(metrics)
                        self.signal_quality_validator.add_raw_eeg_data(eeg_window)
                        
                        # Update signal quality
                        if current_time - last_quality_update >= 1.0:
                            quality_metrics = self.signal_quality_validator.assess_overall_quality()
                            print(f"EEG Worker: Quality score: {quality_metrics.overall_score} ({quality_metrics.quality_level})")
                            
                            # Emit signal quality update for UI
                            quality_data = {
                                "movement_score": quality_metrics.movement_score,
                                "band_power_score": quality_metrics.band_power_score,
                                "electrode_contact_score": quality_metrics.electrode_contact_score,
                                "overall_score": quality_metrics.overall_score,
                                "quality_level": quality_metrics.quality_level,
                                "recommendations": quality_metrics.recommendations,
                                "timestamp": current_time
                            }
                            self.signal_quality_update.emit(quality_data)
                            last_quality_update = current_time
                            
                            # Check if we should pause calibration
                            if quality_metrics.overall_score < 0.4:
                                quality_pause_time += 1.0
                                print(f"EEG Worker: Poor quality, pause time: {quality_pause_time}")
                                
                                self.calibration_status_changed.emit("PAUSED", {
                                    "reason": "Poor signal quality",
                                    "signal_quality": quality_data,
                                    "timestamp": current_time
                                })
                                
                                if quality_pause_time > max_quality_pause:
                                    print(f"EEG Worker: Quality timeout after {quality_pause_time}s")
                                    self.calibration_status_changed.emit("FAILED", {
                                        "error_message": "Signal quality timeout - please adjust headband and try again",
                                        "timestamp": time.time()
                                    })
                                    self.is_calibrating = False
                                    return
                                    
                                QtCore.QCoreApplication.processEvents()
                                time.sleep(0.1)
                                continue
                            else:
                                quality_pause_time = 0
                        
                        # Add to calibration data
                        calibration_metrics_list.append(metrics)
                        
                        # Update progress
                        progress = min(1.0, (time.time() - calibration_start_time) / CALIBRATION_DURATION_SECONDS)
                        self.calibration_progress.emit(progress)
                        if int(progress * 100) % 10 == 0:
                            print(f"EEG Worker: Progress: {progress:.1%}")
                    else:
                        print(f"EEG Worker: Metrics calculation returned None")
                        
                except Exception as e:
                    print(f"EEG Worker: Error processing EEG window: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Prevent excessive buffer growth
            max_buffer_size = int(self.sampling_rate * 10)
            if self.eeg_buffer.shape[1] > max_buffer_size:
                self.eeg_buffer = self.eeg_buffer[:, -max_buffer_size:]
            
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.001)
        
        # Calculate baseline from calibration data
        if calibration_metrics_list and self.is_calibrating:
            print(f"EEG Worker: Creating baseline from {len(calibration_metrics_list)} metrics")
            
            self.baseline_metrics = {
                'alpha': np.mean([m['alpha'] for m in calibration_metrics_list]),
                'beta': np.mean([m['beta'] for m in calibration_metrics_list]),
                'theta': np.mean([m['theta'] for m in calibration_metrics_list])
            }
            self.baseline_metrics['ab_ratio'] = (
                self.baseline_metrics['alpha'] / self.baseline_metrics['beta'] 
                if self.baseline_metrics['beta'] > 1e-9 else 0
            )
            self.baseline_metrics['bt_ratio'] = (
                self.baseline_metrics['beta'] / self.baseline_metrics['theta'] 
                if self.baseline_metrics['theta'] > 1e-9 else 0
            )
            
            print(f"EEG Worker: Baseline metrics: {self.baseline_metrics}")
            
            self.is_calibrated = True
            self.is_calibrating = False
            self.session_start_time = time.time()
            
            # Get final signal quality
            final_quality = self.signal_quality_validator.assess_overall_quality()
            
            self.calibration_status_changed.emit("COMPLETED", {
                "session_type": self.current_session_type,
                "baseline": self.baseline_metrics,
                "signal_quality": {
                    "overall_score": final_quality.overall_score,
                    "quality_level": final_quality.quality_level,
                    "recommendations": final_quality.recommendations
                },
                "samples_collected": len(calibration_metrics_list),
                "timestamp": time.time()
            })
            
            logger.info(f"Calibration complete: Alpha={self.baseline_metrics['alpha']:.2f}, "
                    f"Beta={self.baseline_metrics['beta']:.2f}")
            
            # Start real-time processing
            QtCore.QMetaObject.invokeMethod(
                self.processing_timer, 
                "start", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, int(ANALYSIS_WINDOW_SECONDS * 1000))
            )
            print(f"EEG Worker: Started processing timer with {int(ANALYSIS_WINDOW_SECONDS * 1000)}ms interval")
            
        else:
            print(f"EEG Worker: Calibration failed - metrics list length: {len(calibration_metrics_list)}")
            self.is_calibrating = False
            self.calibration_status_changed.emit("FAILED", {
                "error_message": f"No valid EEG data collected during calibration.",
                "timestamp": time.time()
            })
    
    def _process_eeg_data(self):
        """Process EEG data for real-time feedback and accumulate session data - Updated for multi-stream"""
        if not self.is_calibrated or not self.lsl_inlet:
            return
        
        try:
            # Get EEG data chunk
            chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
            
            if not chunk:
                return
                
            # Process EEG chunk
            chunk_np = np.array(chunk, dtype=np.float64).T
            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
            
            # Get accelerometer data from appropriate source
            acc_sample = None
            if self.is_multi_stream:
                # Multi-stream: get from separate accelerometer stream
                acc_sample = self._get_accelerometer_data()
            else:
                # Single-stream: extract from EEG chunk
                acc_sample = self._extract_accelerometer_from_eeg_chunk(chunk_np)
            
            # Add accelerometer data to signal quality validator if available
            if acc_sample is not None:
                self.signal_quality_validator.add_accelerometer_data(acc_sample)
            
            # Add to buffer
            self.eeg_buffer = np.append(self.eeg_buffer, eeg_chunk, axis=1)
            
            # Process if we have enough data
            if self.eeg_buffer.shape[1] >= self.nfft:
                eeg_window = self.eeg_buffer[:, -self.nfft:]
                filtered_window = self._filter_eeg_data(eeg_window)
                current_metrics = self._calculate_band_powers(filtered_window)
                
                if current_metrics:
                    # Add to signal quality validator
                    self.signal_quality_validator.add_band_power_data(current_metrics)
                    self.signal_quality_validator.add_raw_eeg_data(eeg_window)
                    
                    # Get signal quality assessment
                    signal_quality_metrics = self.signal_quality_validator.assess_overall_quality()
                    
                    # Update history
                    self.recent_metrics_history.append(current_metrics)
                    if len(self.recent_metrics_history) > 15:
                        self.recent_metrics_history.pop(0)
                    
                    # Classify mental state
                    classification = self._classify_mental_state_enhanced(current_metrics)
                    
                    # Store current prediction
                    self.current_prediction = classification
                    
                    # ACCUMULATE ALL SESSION DATA
                    self._accumulate_session_data(classification, current_metrics, eeg_window)
                    
                    # Prepare prediction data for UI feedback only
                    prediction_data = {
                        "message_type": "PREDICTION",
                        "timestamp": time.time(),
                        "session_type": self.current_session_type,
                        "classification": classification,
                        "smooth_value": classification.get('smooth_value', 0.5),
                        "metrics": {
                            "alpha": round(current_metrics['alpha'], 3),
                            "beta": round(current_metrics['beta'], 3),
                            "theta": round(current_metrics['theta'], 3),
                            "ab_ratio": round(current_metrics['ab_ratio'], 3),
                            "bt_ratio": round(current_metrics['bt_ratio'], 3)
                        },
                        "signal_quality": {
                            "accelerometer": acc_sample.tolist() if acc_sample is not None else [0, 0, 0],
                            "band_powers": current_metrics,
                            "quality_metrics": {
                                "movement_score": signal_quality_metrics.movement_score,
                                "band_power_score": signal_quality_metrics.band_power_score,
                                "electrode_contact_score": signal_quality_metrics.electrode_contact_score,
                                "overall_score": signal_quality_metrics.overall_score,
                                "quality_level": signal_quality_metrics.quality_level,
                                "recommendations": signal_quality_metrics.recommendations
                            },
                            "timestamp": time.time()
                        }
                    }
                    
                    # Emit prediction for UI feedback only
                    self.new_prediction.emit(prediction_data)
            
            # Prevent excessive buffer growth
            max_buffer_size = int(self.sampling_rate * 10)
            if self.eeg_buffer.shape[1] > max_buffer_size:
                self.eeg_buffer = self.eeg_buffer[:, -max_buffer_size:]
                
        except Exception as e:
            logger.error(f"Error in EEG processing: {e}")
            self.error_occurred.emit(f"EEG processing error: {e}")
    
    def _calculate_is_on_target(self, level):
        """Calculate if current state is 'on target' based on session type and level"""
        if self.current_session_type == SESSION_TYPE_RELAX:
            # For relaxation, positive levels (relaxed states) are on target
            return level > 0
        elif self.current_session_type == SESSION_TYPE_FOCUS:
            # For focus, positive levels (focused states) are on target
            return level > 0
        else:
            # Default case
            return level > 0
    
    def _has_session_data(self):
        """Check if we have any session data to save"""
        return (len(self.session_predictions) > 0 or 
                len(self.session_band_data["alpha"]) > 0 or 
                len(self.session_eeg_data["channel_0"]) > 0)
    
    def _save_session_to_database(self):
        """Save all accumulated session data to database and return summary stats"""
        if not self.current_session_id:
            raise Exception("No session ID available for saving")
        
        summary_stats = {
            "total_predictions": len(self.session_predictions),
            "on_target_count": sum(self.session_on_target),
            "percent_on_target": 0.0,
            "band_data_points": len(self.session_band_data["alpha"]),
            "eeg_data_points": len(self.session_eeg_data["channel_0"])
        }
        
        # Calculate percentage on target
        if summary_stats["total_predictions"] > 0:
            summary_stats["percent_on_target"] = (summary_stats["on_target_count"] / summary_stats["total_predictions"]) * 100.0
        
        try:
            # Save session metrics (predictions, on_target, timestamps)
            if self.session_predictions:
                success = db_manager.save_session_metrics_batch(
                    self.current_session_id,
                    self.session_predictions,
                    self.session_on_target,
                    self.session_timestamps
                )
                if not success:
                    raise Exception("Failed to save session metrics")
            
            # Save band data
            if self.session_band_data["alpha"]:
                success = db_manager.save_session_band_data_batch(
                    self.current_session_id,
                    self.session_band_data
                )
                if not success:
                    raise Exception("Failed to save band data")
            
            # Save EEG data
            if self.session_eeg_data["channel_0"]:
                success = db_manager.save_session_eeg_data_batch(
                    self.current_session_id,
                    self.session_eeg_data,
                    self.sampling_rate
                )
                if not success:
                    raise Exception("Failed to save EEG data")
            
            # End the session and create summary
            db_manager.end_session(self.current_session_id)
            
            return summary_stats
            
        except Exception as e:
            error_msg = f"Database save error: {e}"
            raise Exception(error_msg)

    def _accumulate_session_data(self, classification, current_metrics, eeg_window):
        """Accumulate all session data for later database saving"""
        current_time = time.time()
        
        # Calculate is_on_target based on session type and level
        level = classification.get('level', 0)
        is_on_target = self._calculate_is_on_target(level)
        
        # Store prediction data
        self.session_predictions.append(classification.get('state', 'Unknown'))
        self.session_on_target.append(is_on_target)
        self.session_timestamps.append(current_time)
        self.session_confidence_scores.append(classification.get('confidence', 0.0))
        
        # Store band data
        self.session_band_data["alpha"].append(round(current_metrics['alpha'], 3))
        self.session_band_data["beta"].append(round(current_metrics['beta'], 3))
        self.session_band_data["theta"].append(round(current_metrics['theta'], 3))
        self.session_band_data["ab_ratio"].append(round(current_metrics['ab_ratio'], 3))
        self.session_band_data["bt_ratio"].append(round(current_metrics['bt_ratio'], 3))
        self.session_band_data["timestamps"].append(current_time)
        
        # Store EEG data (downsampled for storage efficiency)
        if eeg_window.shape[1] >= self.nfft:
            eeg_window = self._filter_eeg_data(eeg_window)
            # Downsample by taking every 4th sample for storage
            downsample_factor = 4
            downsampled_indices = np.arange(0, eeg_window.shape[1], downsample_factor)
            eeg_downsampled = eeg_window[:, downsampled_indices]
            
            # Store each sample
            for i in range(eeg_downsampled.shape[1]):
                self.session_eeg_data["channel_0"].append(float(eeg_downsampled[0, i]))
                self.session_eeg_data["channel_1"].append(float(eeg_downsampled[1, i]))
                self.session_eeg_data["channel_2"].append(float(eeg_downsampled[2, i]))
                self.session_eeg_data["channel_3"].append(float(eeg_downsampled[3, i]))
                self.session_eeg_data["timestamps"].append(current_time + (i * downsample_factor / self.sampling_rate))
    
    def _clear_session_data(self):
        """Clear all session data storage"""
        self.session_predictions = []
        self.session_on_target = []
        self.session_timestamps = []
        self.session_confidence_scores = []
        
        self.session_band_data = {
            "alpha": [],
            "beta": [],
            "theta": [],
            "ab_ratio": [],
            "bt_ratio": [],
            "timestamps": []
        }
        
        self.session_eeg_data = {
            "channel_0": [],
            "channel_1": [],
            "channel_2": [],
            "channel_3": [],
            "timestamps": []
        }
        
        print("EEG Worker: Cleared all session data arrays")
    
    # [Keep all the existing helper methods unchanged]
    
    def _filter_eeg_data(self, eeg_data):
        """Apply bandpass filter to EEG data"""
        min_samples = 3 * filter_order + 1
        
        if eeg_data.shape[1] < min_samples:
            return eeg_data
            
        eeg_filtered = np.zeros_like(eeg_data)
        for i in range(NUM_EEG_CHANNELS):
            eeg_filtered[i] = filtfilt(self.b, self.a, eeg_data[i])
        return eeg_filtered
    
    def _calculate_band_powers(self, eeg_segment):
        """Calculate band powers for EEG segment"""
        if eeg_segment.shape[1] < self.nfft:
            return None
            
        # Apply artifact rejection
        artifact_mask = self._improved_artifact_rejection(eeg_segment)
        if np.sum(artifact_mask) < 0.7 * eeg_segment.shape[1]:
            return None
        
        # Calculate band powers for each channel
        metrics_list = []
        for ch_idx in range(NUM_EEG_CHANNELS):
            ch_data = eeg_segment[ch_idx, artifact_mask].copy() if np.any(artifact_mask) else eeg_segment[ch_idx].copy()
            
            if len(ch_data) < self.nfft:
                pad_length = self.nfft - len(ch_data)
                ch_data = np.pad(ch_data, (0, pad_length), mode='reflect')
                
            DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
            
            try:
                psd = DataFilter.get_psd_welch(
                    ch_data, 
                    self.nfft, 
                    self.welch_overlap_samples, 
                    int(self.sampling_rate), 
                    WindowOperations.HANNING.value
                )
                
                metrics_list.append({
                    'theta': DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1]),
                    'alpha': DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1]),
                    'beta': DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])
                })
            except Exception as e:
                logger.error(f"PSD calculation failed: {e}")
                return None
                
        if len(metrics_list) != NUM_EEG_CHANNELS:
            return None
            
        # Calculate weighted average
        avg_metrics = {
            'theta': np.sum([m['theta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
            'alpha': np.sum([m['alpha'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
            'beta': np.sum([m['beta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)])
        }
        
        avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
        avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
        
        return avg_metrics
    
    def _improved_artifact_rejection(self, eeg_data):
        """Artifact rejection for EEG data"""
        channel_thresholds = [150, 100, 100, 150]
        amplitude_mask = ~np.any(np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1), axis=0)
        
        diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
        if eeg_data.shape[1] > 1:
            diff_thresholds = [50, 30, 30, 50]
            diff_mask = ~np.any(
                np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
                np.array(diff_thresholds).reshape(-1, 1), 
                axis=0
            )
        
        return amplitude_mask & diff_mask
    
    def _classify_mental_state_enhanced(self, current_metrics):
        """Enhanced classify mental state with smoothing and stability"""
        if not self.baseline_metrics or not current_metrics:
            return {
                "state": "Unknown", "level": 0, "confidence": 0.0,
                "value": 0.5, "smooth_value": 0.5, "state_key": "unknown"
            }
        
        # Calculate ratios relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha'] if self.baseline_metrics['alpha'] > 0 else 1
        beta_ratio = current_metrics['beta'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 0 else 1
        theta_ratio = current_metrics['theta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 0 else 1
        
        # Session-specific classification
        if self.current_session_type == SESSION_TYPE_RELAX:
            # For relaxation: higher alpha is better, lower beta/theta is better
            if alpha_ratio > 1.4 and beta_ratio < 0.8:
                raw_state, raw_level = "Deeply Relaxed", 3
                confidence = min(0.95, 0.5 + (alpha_ratio - 1.4) * 0.3)
            elif alpha_ratio > 1.2:
                raw_state, raw_level = "Relaxed", 2
                confidence = min(0.9, 0.5 + (alpha_ratio - 1.2) * 0.5)
            elif alpha_ratio > 1.05:
                raw_state, raw_level = "Slightly Relaxed", 1
                confidence = min(0.8, 0.4 + (alpha_ratio - 1.05) * 2.0)
            elif alpha_ratio < 0.8 and beta_ratio > 1.2:
                raw_state, raw_level = "Tense", -2
                confidence = min(0.9, 0.5 + (1.2 - alpha_ratio) * 0.5)
            elif alpha_ratio < 0.9:
                raw_state, raw_level = "Slightly Tense", -1
                confidence = min(0.8, 0.4 + (0.9 - alpha_ratio) * 2.0)
            else:
                raw_state, raw_level = "Neutral", 0
                confidence = 0.6
            
            value = (raw_level + 3) / 6.0
            
        elif self.current_session_type == SESSION_TYPE_FOCUS:
            # For focus: higher beta/theta ratio is better
            bt_ratio = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] if self.baseline_metrics['bt_ratio'] > 0 else 1
            if bt_ratio > 1.4 and beta_ratio > 1.2:
                raw_state, raw_level = "Highly Focused", 3
                confidence = min(0.95, 0.5 + (bt_ratio - 1.4) * 0.3)
            elif bt_ratio > 1.2:
                raw_state, raw_level = "Focused", 2
                confidence = min(0.9, 0.5 + (bt_ratio - 1.2) * 0.5)
            elif bt_ratio > 1.05:
                raw_state, raw_level = "Slightly Focused", 1
                confidence = min(0.8, 0.4 + (bt_ratio - 1.05) * 2.0)
            elif bt_ratio < 0.8 or theta_ratio > 1.3:
                raw_state, raw_level = "Distracted", -2
                confidence = min(0.9, 0.5 + (0.8 - bt_ratio) * 0.5)
            elif bt_ratio < 0.9:
                raw_state, raw_level = "Slightly Distracted", -1
                confidence = min(0.8, 0.4 + (0.9 - bt_ratio) * 2.0)
            else:
                raw_state, raw_level = "Neutral", 0
                confidence = 0.6
            
            value = (raw_level + 3) / 6.0
        else:
            raw_state, raw_level = "Unknown", 0
            confidence = 0.0
            value = 0.5
        
        # Apply smoothing and stability logic
        smoothed_prediction = self._apply_prediction_smoothing(raw_state, raw_level, confidence, value)
        
        return smoothed_prediction
    
    def _apply_prediction_smoothing(self, raw_state, raw_level, confidence, raw_value):
        """Apply smoothing and stability requirements to predictions"""
        # Get previous smoothed value
        if self.prediction_history:
            prev_smooth_value = self.prediction_history[-1]['smooth_value']
            prev_state = self.prediction_history[-1]['state']
            prev_level = self.prediction_history[-1]['level']
        else:
            prev_smooth_value = 0.5
            prev_state = "Unknown"
            prev_level = 0
        
        # Apply exponential smoothing
        smoothing_factor = 0.3 if confidence > 0.8 else 0.1  # Faster smoothing for high confidence
        smooth_value = smoothing_factor * raw_value + (1 - smoothing_factor) * prev_smooth_value
        
        # Determine if we should change the displayed state
        level_change = abs(raw_level - prev_level)
        
        # Check stability requirements
        if (confidence >= self.min_confidence_for_change and 
            level_change >= self.state_change_threshold):
            
            # Check if this is consistent with recent predictions
            if self.last_stable_prediction == raw_state:
                self.stable_prediction_count += 1
            else:
                self.last_stable_prediction = raw_state
                self.stable_prediction_count = 1
            
            # Only change state if we have enough consistent predictions
            if self.stable_prediction_count >= self.min_stable_count:
                final_state = raw_state
                final_level = raw_level
                final_confidence = confidence
            else:
                # Keep previous state but update smooth_value
                final_state = prev_state
                final_level = prev_level
                final_confidence = confidence * 0.7  # Reduce confidence during transition
        else:
            # Not confident enough or change too small - keep previous state
            final_state = prev_state
            final_level = prev_level
            final_confidence = confidence * 0.8
        
        prediction = {
            "state": final_state,
            "state_key": final_state.lower().replace(' ', '_'),
            "level": final_level,
            "confidence": final_confidence,
            "value": raw_value,
            "smooth_value": smooth_value,
            "raw_state": raw_state,
            "raw_level": raw_level,
            "raw_confidence": confidence
        }
        
        # Add to prediction history
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.prediction_smoothing_window:
            self.prediction_history.pop(0)
        
        return prediction
    
    @QtCore.pyqtSlot()
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up EEG worker")
        
        self.running = False
        
        if self.processing_timer.isActive():
            self.processing_timer.stop()
        
        if self.lsl_inlet:
            self.lsl_inlet.close_stream()
            self.lsl_inlet = None
        
        if self.lsl_accelerometer_inlet:
            self.lsl_accelerometer_inlet.close_stream()
            self.lsl_accelerometer_inlet = None