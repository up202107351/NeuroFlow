#!/usr/bin/env python3
"""
EEG Processing Worker - Qt Threading-based EEG Processing

This worker runs in a separate thread and handles:
1. LSL connection to EEG device
2. Baseline calibration with signal quality monitoring
3. Real-time EEG processing and state classification
4. Direct Qt signal emission for UI updates
"""

import time
import numpy as np
import pylsl
import logging
from datetime import datetime
from scipy.signal import butter, filtfilt, welch
from PyQt5 import QtCore
from backend.signal_quality_validator import SignalQualityValidator

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
LSL_RESOLVE_TIMEOUT = 5
LSL_CHUNK_MAX_PULL = 128

EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10 for Muse
NUM_EEG_CHANNELS = len(EEG_CHANNEL_INDICES)

CALIBRATION_DURATION_SECONDS = 20.0
ANALYSIS_WINDOW_SECONDS = 1.0
PSD_WINDOW_SECONDS = 6.0

DEFAULT_SAMPLING_RATE = 256.0

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Design Butterworth bandpass filter (0.5 - 30 Hz)
filter_order = 4
lowcut = 0.5
highcut = 30.0

class EEGProcessingWorker(QtCore.QObject):
    """
    EEG Processing Worker that runs in a separate thread.
    Handles all EEG processing and emits Qt signals for UI updates.
    """
    
    # Signals for UI communication
    connection_status_changed = QtCore.pyqtSignal(str, str)  # status, message
    calibration_progress = QtCore.pyqtSignal(float)  # 0.0 to 1.0
    calibration_status_changed = QtCore.pyqtSignal(str, dict)  # status, data
    new_prediction = QtCore.pyqtSignal(dict)  # prediction data with signal quality
    signal_quality_update = QtCore.pyqtSignal(dict)  # real-time quality metrics
    error_occurred = QtCore.pyqtSignal(str)  # error message
    session_data_ready = QtCore.pyqtSignal(dict)  # session data for saving
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Connection state
        self.lsl_inlet = None
        self.sampling_rate = DEFAULT_SAMPLING_RATE
        self.running = True
        
        # Session state
        self.is_calibrated = False
        self.is_calibrating = False
        self.current_session_type = None
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
        
        # Filter coefficients (will be updated when sampling rate is known)
        self.b = None
        self.a = None
        self.nfft = DataFilter.get_nearest_power_of_two(int(DEFAULT_SAMPLING_RATE * PSD_WINDOW_SECONDS))
        self.welch_overlap_samples = self.nfft // 2
        
        # Processing timers
        self.processing_timer = QtCore.QTimer()
        self.processing_timer.timeout.connect(self._process_eeg_data)
        
        # Session data collection
        self.session_band_data = {
            "alpha": [],
            "beta": [],
            "theta": [],
            "ab_ratio": [],
            "bt_ratio": []
        }
        self.session_eeg_data = []
        self.session_timestamps = []
        
        # State tracking
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
        
        self.state_momentum = 0.75
        self.state_velocity = 0.0
        self.level_momentum = 0.8
        self.level_velocity = 0.0
    
    @QtCore.pyqtSlot()
    def initialize(self):
        """Initialize the worker (called when moved to thread)"""
        self.running = True
        logger.info("EEG Processing Worker initialized")
        
    @QtCore.pyqtSlot()
    def connect_to_lsl(self):
        """Connect to the LSL stream with detailed channel debugging"""
        logger.info(f"Looking for LSL stream (Type: '{LSL_STREAM_TYPE}')...")
        
        try:
            self.connection_status_changed.emit("CONNECTING", "Looking for EEG stream...")
            
            streams = pylsl.resolve_byprop('type', LSL_STREAM_TYPE, 1, timeout=LSL_RESOLVE_TIMEOUT)
            if not streams:
                error_msg = "LSL stream not found."
                logger.error(error_msg)
                self.connection_status_changed.emit("ERROR", error_msg)
                self.error_occurred.emit(error_msg)
                return False
                
            self.lsl_inlet = pylsl.StreamInlet(streams[0], max_chunklen=LSL_CHUNK_MAX_PULL)
            info = self.lsl_inlet.info()
            lsl_sr = info.nominal_srate()
            self.sampling_rate = lsl_sr if lsl_sr > 0 else DEFAULT_SAMPLING_RATE
            
            # DEBUG: Print detailed channel information
            print(f"EEG Worker: LSL Stream Info:")
            print(f"  Name: {info.name()}")
            print(f"  Type: {info.type()}")
            print(f"  Channel count: {info.channel_count()}")
            print(f"  Sampling rate: {self.sampling_rate}")
            print(f"  Expected EEG channels: {EEG_CHANNEL_INDICES}")
            print(f"  Expected ACC channels: [9, 10, 11]")
            
            # Try to get channel labels if available
            try:
                ch = info.desc().child("channels").child("channel")
                for i in range(info.channel_count()):
                    label = ch.child_value("label")
                    print(f"  Channel {i}: {label}")
                    ch = ch.next_sibling()
            except:
                print("  Channel labels not available")
            
            # Update filter coefficients and processing parameters
            self._update_filter_coefficients()
            
            device_name = info.name()
            logger.info(f"Connected to '{device_name}' @ {self.sampling_rate:.2f} Hz")
            
            self.connection_status_changed.emit("CONNECTED", f"Connected to {device_name}")
            
            if info.channel_count() < np.max(EEG_CHANNEL_INDICES) + 1:
                error_msg = f"LSL stream has insufficient channels. Need at least {np.max(EEG_CHANNEL_INDICES) + 1}, got {info.channel_count()}"
                logger.error(error_msg)
                print(f"EEG Worker: {error_msg}")
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
    
    def _update_filter_coefficients(self):
        """Update filter coefficients based on current sampling rate"""
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(filter_order, [low, high], btype='band', analog=False)
        self.nfft = DataFilter.get_nearest_power_of_two(int(self.sampling_rate * PSD_WINDOW_SECONDS))
        self.welch_overlap_samples = self.nfft // 2
    
    @QtCore.pyqtSlot(str)
    def start_session(self, session_type):
        """Start a new session of the specified type"""
        if session_type not in ["RELAXATION", "FOCUS"]:
            error_msg = f"Invalid session type: {session_type}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return
            
        logger.info(f"Starting {session_type} session")
        
        # Reset state
        self.is_calibrated = False
        self.is_calibrating = False
        self.running = True
        self.current_session_type = session_type
        self.session_start_time = None
        self.recent_metrics_history = []
        self.previous_states = []
        self.signal_quality_validator.reset()
        
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
        """Start the calibration process with better error handling"""
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
        
        # Test LSL connection by trying to pull some data
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
        """Stop the current session"""
        logger.info("Stopping session")
        
        # Stop processing
        if self.processing_timer.isActive():
            self.processing_timer.stop()
        
        # Emit session data for saving
        if self.current_session_type and self.session_band_data["alpha"]:
            session_data = {
                "session_type": self.current_session_type,
                "band_data": self.session_band_data.copy(),
                "eeg_data": self.session_eeg_data.copy(),
                "timestamps": self.session_timestamps.copy(),
                "sampling_rate": self.sampling_rate
            }
            self.session_data_ready.emit(session_data)
        
        # Reset state
        self.current_session_type = None
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
        
        # Clear buffer
        self.eeg_buffer = np.array([]).reshape(NUM_EEG_CHANNELS, 0)
        
        # Restart calibration
        QtCore.QTimer.singleShot(100, self._start_calibration)   
    
    def _perform_calibration(self):
        """Perform calibration with signal quality monitoring and detailed debugging"""
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
                      
            # Get data chunk
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
                
            # Process chunk
            chunk_np = np.array(chunk, dtype=np.float64).T
            total_samples_received += chunk_np.shape[1]
            
            # Check if we have enough channels
            if chunk_np.shape[0] <= max(EEG_CHANNEL_INDICES):
                print(f"EEG Worker: Not enough channels in chunk: {chunk_np.shape[0]} <= {max(EEG_CHANNEL_INDICES)}")
                chunks_failed_processing += 1
                continue
                
            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
            
            # Extract accelerometer data if available
            ACC_CHANNEL_INDICES = [9, 10, 11]
            if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
                try:
                    acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                    if acc_chunk.shape[1] > 0:
                        latest_acc_sample = acc_chunk[:, -1]
                        self.signal_quality_validator.add_accelerometer_data(latest_acc_sample)
                except Exception as e:
                    print(f"EEG Worker: Could not extract accelerometer data: {e}")
            else:
                print(f"EEG Worker: No accelerometer channels available (need >{max(ACC_CHANNEL_INDICES)}, have {chunk_np.shape[0]})")
            
            # Add to buffer
            old_buffer_size = self.eeg_buffer.shape[1]
            self.eeg_buffer = np.append(self.eeg_buffer, eeg_chunk, axis=1)
            new_buffer_size = self.eeg_buffer.shape[1]
            
            # Process if we have enough data
            if self.eeg_buffer.shape[1] >= self.nfft:
                chunks_processed += 1
                
                eeg_window = self.eeg_buffer[:, -self.nfft:]
                # print(f"EEG Worker: Processing window shape: {eeg_window.shape}")
                
                try:
                    filtered_window = self._filter_eeg_data(eeg_window)
                    #print(f"EEG Worker: Filtered window shape: {filtered_window.shape}")
                    
                    metrics = self._calculate_band_powers(filtered_window)
                    #print(f"EEG Worker: Calculated metrics: {metrics}")
                    
                    if metrics:
                        # Add to signal quality validator
                        self.signal_quality_validator.add_band_power_data(metrics)
                        self.signal_quality_validator.add_raw_eeg_data(eeg_window)
                        
                        # UPDATE SIGNAL QUALITY MORE FREQUENTLY (every 1 second)
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
                        if int(progress * 100) % 10 == 0:  # Print every 10%
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
                old_size = self.eeg_buffer.shape[1]
                self.eeg_buffer = self.eeg_buffer[:, -max_buffer_size:]
            
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.001)
        
        # Print final debug info
        print(f"\nEEG Worker: Calibration loop finished")
        print(f"  - Total duration: {time.time() - calibration_start_time:.1f}s")
        print(f"  - Chunks received: {total_chunks_received}")
        print(f"  - Chunks with no data: {chunks_with_no_data}")
        print(f"  - Chunks processed: {chunks_processed}")
        print(f"  - Chunks failed processing: {chunks_failed_processing}")
        print(f"  - Total samples received: {total_samples_received}")
        print(f"  - Final buffer size: {self.eeg_buffer.shape[1]}")
        print(f"  - Metrics collected: {len(calibration_metrics_list)}")
        print(f"  - Is calibrating: {self.is_calibrating}")
        print(f"  - Running: {self.running}")
        
        # Calculate baseline from calibration data
        if calibration_metrics_list and self.is_calibrating:
            print(f"EEG Worker: Creating baseline from {len(calibration_metrics_list)} metrics")
            
            # Debug: Print some metrics
            for i, m in enumerate(calibration_metrics_list[:3]):  # Print first 3
                print(f"  Metric {i}: alpha={m['alpha']:.3f}, beta={m['beta']:.3f}, theta={m['theta']:.3f}")
            
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
            print(f"EEG Worker: Calibration failed - metrics list length: {len(calibration_metrics_list)}, is_calibrating: {self.is_calibrating}")
            self.is_calibrating = False
            self.calibration_status_changed.emit("FAILED", {
                "error_message": f"No valid EEG data collected during calibration. "
                            f"Collected {len(calibration_metrics_list)} metrics from {total_chunks_received} chunks.",
                "debug_info": {
                    "chunks_received": total_chunks_received,
                    "chunks_no_data": chunks_with_no_data,
                    "chunks_processed": chunks_processed,
                    "chunks_failed": chunks_failed_processing,
                    "total_samples": total_samples_received,
                    "buffer_size": self.eeg_buffer.shape[1],
                    "nfft_required": self.nfft
                },
                "timestamp": time.time()
            })
    
    def _process_eeg_data(self):
        """Process EEG data for real-time feedback"""
        if not self.is_calibrated or not self.lsl_inlet:
            return
        
        try:
            # Get data chunk
            chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=0.1, max_samples=LSL_CHUNK_MAX_PULL)
            
            if not chunk:
                return
                
            # Process chunk
            chunk_np = np.array(chunk, dtype=np.float64).T
            eeg_chunk = chunk_np[EEG_CHANNEL_INDICES, :]
            
            # Extract accelerometer data
            ACC_CHANNEL_INDICES = [9, 10, 11]
            if chunk_np.shape[0] > max(ACC_CHANNEL_INDICES):
                try:
                    acc_chunk = chunk_np[ACC_CHANNEL_INDICES, :]
                    if acc_chunk.shape[1] > 0:
                        latest_acc_sample = acc_chunk[:, -1]
                        self.signal_quality_validator.add_accelerometer_data(latest_acc_sample)
                except Exception as e:
                    pass  # Non-critical error
            
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
                    if len(self.recent_metrics_history) > 15:  # MAX_HISTORY_SIZE
                        self.recent_metrics_history.pop(0)
                    
                    # Classify mental state
                    classification = self._classify_mental_state(current_metrics)
                    
                    # Prepare prediction data
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
                            "accelerometer": self.signal_quality_validator.accelerometer_history[-1]['data'].tolist() 
                                           if self.signal_quality_validator.accelerometer_history else [0, 0, 0],
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
                    
                    # Emit prediction
                    self.new_prediction.emit(prediction_data)
                    
                    # Store data for session
                    self._store_session_data(current_metrics, eeg_window)
            
            # Prevent excessive buffer growth
            max_buffer_size = int(self.sampling_rate * 10)
            if self.eeg_buffer.shape[1] > max_buffer_size:
                self.eeg_buffer = self.eeg_buffer[:, -max_buffer_size:]
                
        except Exception as e:
            logger.error(f"Error in EEG processing: {e}")
            self.error_occurred.emit(f"EEG processing error: {e}")
    
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
    
    def _classify_mental_state(self, current_metrics):
        """Classify mental state based on current metrics"""
        if not self.baseline_metrics or not current_metrics:
            return {
                "state": "Calibrating",
                "level": 0,
                "confidence": "N/A",
                "value": 0.5,
                "smooth_value": 0.5,
                "state_key": "calibrating"
            }
        
        # Get state probabilities
        state_probs = self._calculate_state_probabilities(current_metrics)
        
        # Determine most probable state
        most_probable_state = max(state_probs.items(), key=lambda x: x[1])
        state_name, prob_value = most_probable_state
        
        level = 0
        
        if self.current_session_type == "RELAXATION":
            relaxation_value = min(1.0, max(0.0, state_probs['relaxed']))
            
            # Calculate alert signal
            ab_ratio_decrease = self.baseline_metrics['ab_ratio'] / current_metrics['ab_ratio'] - 1.0 if current_metrics['ab_ratio'] > 0 else 0
            beta_increase = current_metrics['beta'] / self.baseline_metrics['beta'] - 1.0
            alpha_decrease = 1.0 - current_metrics['alpha'] / self.baseline_metrics['alpha']
            
            alert_signal = (0.4 * ab_ratio_decrease + 0.3 * beta_increase + 0.3 * alpha_decrease) * 4.0
            
            # Determine relaxation levels
            if (ab_ratio_decrease > 0.1 and (beta_increase > 0.1 or alpha_decrease > 0.1)):
                if alert_signal > 1.5:
                    level = -3
                    state_name = "tense"
                elif alert_signal > 1.0:
                    level = -2
                    state_name = "alert"
                else:
                    level = -1
                    state_name = "less_relaxed"
            elif relaxation_value > 0.3:
                alpha_increase = current_metrics['alpha'] / self.baseline_metrics['alpha'] - 1.0
                ab_ratio_increase = current_metrics['ab_ratio'] / self.baseline_metrics['ab_ratio'] - 1.0 if self.baseline_metrics['ab_ratio'] > 0 else 0
                relax_signal = (0.5 * alpha_increase + 0.5 * ab_ratio_increase) * 5.0
                
                if relax_signal > 0.5:
                    level = 4
                    state_name = "deeply_relaxed"
                elif relax_signal > 0.25:
                    level = 3
                    state_name = "strongly_relaxed"
                elif relax_signal > 0.1:
                    level = 2
                    state_name = "moderately_relaxed"
                else:
                    level = 1
                    state_name = "slightly_relaxed"
            else:
                level = 0
                state_name = "neutral"
            
            value = (level + 3) / 7.0
            
        elif self.current_session_type == "FOCUS":
            # Similar logic for focus
            focus_value = min(1.0, max(0.0, state_probs['focused']))
            value = (level + 3) / 7.0
        else:
            value = prob_value
        
        # Determine confidence
        if prob_value > 0.8:
            confidence = "high"
        elif prob_value > 0.65:
            confidence = "medium"
        elif prob_value > 0.55:
            confidence = "low"
        else:
            confidence = "very_low"
        
        # Apply temporal smoothing
        smooth_value = value
        
        self.previous_states.append((state_name, value, level))
        if len(self.previous_states) > 5:  # MAX_STATE_HISTORY
            self.previous_states.pop(0)
        
        if len(self.previous_states) > 1:
            current_value = value
            recent_values = [s[1] for s in self.previous_states]
            prev_value = recent_values[-2]
            
            target_velocity = current_value - prev_value
            self.state_velocity = (self.state_velocity * self.state_momentum + 
                                 target_velocity * (1 - self.state_momentum))
            smooth_value = prev_value + self.state_velocity
            smooth_value = min(1.0, max(0.0, smooth_value))
        
        # Map state names to display names
        state_display_map = {
            'deeply_relaxed': "Deeply Relaxed",
            'strongly_relaxed': "Strongly Relaxed",
            'moderately_relaxed': "Moderately Relaxed",
            'slightly_relaxed': "Slightly Relaxed",
            'neutral': "Neutral",
            'less_relaxed': "Less Relaxed",
            'alert': "Alert",
            'tense': "Tense",
            'calibrating': "Calibrating"
        }
        
        display_state = state_display_map.get(state_name, state_name.title())
        
        return {
            "state": display_state,
            "state_key": state_name,
            "level": level,
            "confidence": confidence,
            "value": round(value, 3),
            "smooth_value": round(smooth_value, 3)
        }
    
    def _calculate_state_probabilities(self, current_metrics):
        """Calculate probabilities for different mental states"""
        if not self.baseline_metrics:
            return {
                'relaxed': 0.5,
                'focused': 0.5,
                'drowsy': 0.2,
                'internal_focus': 0.3,
                'neutral': 0.7
            }
        
        def sigmoid(x, k=5):
            return 1 / (1 + np.exp(-k * x))
        
        # Calculate normalized differences from baseline
        alpha_diff = current_metrics['alpha'] / self.baseline_metrics['alpha'] - 1.0
        beta_diff = current_metrics['beta'] / self.baseline_metrics['beta'] - 1.0
        theta_diff = current_metrics['theta'] / self.baseline_metrics['theta'] - 1.0
        ab_ratio_diff = current_metrics['ab_ratio'] / self.baseline_metrics['ab_ratio'] - 1.0
        bt_ratio_diff = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] - 1.0
        
        # Calculate state probabilities
        relaxation_signal = 0.4 * alpha_diff + 0.6 * ab_ratio_diff
        relaxed_prob = sigmoid(relaxation_signal, k=3)
        
        alpha_inverse_diff = 1.0 - (current_metrics['alpha'] / self.baseline_metrics['alpha'])
        focus_signal = 0.3 * beta_diff + 0.5 * bt_ratio_diff + 0.2 * max(0, alpha_inverse_diff)
        focused_prob = sigmoid(focus_signal, k=3)
        
        drowsy_signal = 0.6 * theta_diff - 0.4 * beta_diff
        drowsy_prob = sigmoid(drowsy_signal, k=3)
        
        internal_focus_signal = 0.5 * beta_diff + 0.5 * alpha_diff
        internal_focus_prob = sigmoid(internal_focus_signal, k=3)
        
        baseline_closeness = -2.0 * (abs(alpha_diff) + abs(beta_diff) + abs(theta_diff))
        neutral_prob = sigmoid(baseline_closeness, k=5)
        
        return {
            'relaxed': relaxed_prob,
            'focused': focused_prob,
            'drowsy': drowsy_prob,
            'internal_focus': internal_focus_prob,
            'neutral': neutral_prob
        }
    
    def _store_session_data(self, current_metrics, eeg_window):
        """Store session data for later saving"""
        # Store band data
        for band in ["alpha", "beta", "theta", "ab_ratio", "bt_ratio"]:
            self.session_band_data[band].append(round(current_metrics[band], 3))
        
        # Store timestamps
        self.session_timestamps.append(time.time())
        
        # Store EEG data (downsampled)
        if eeg_window.shape[1] >= self.nfft:
            samples_to_store = int(self.sampling_rate * 1)  # 1 second
            data_to_store = eeg_window[:, -samples_to_store::4]  # Downsample by 4
            self.session_eeg_data.append(data_to_store.tolist())
    
    def _clear_session_data(self):
        """Clear session data storage"""
        self.session_band_data = {
            "alpha": [],
            "beta": [],
            "theta": [],
            "ab_ratio": [],
            "bt_ratio": []
        }
        self.session_eeg_data = []
        self.session_timestamps = []
    
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