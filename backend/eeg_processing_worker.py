#!/usr/bin/env python3
"""
EEG Processing Worker - Qt Threading-based EEG Processing with Hybrid Classifier

Updated to handle both single-stream (BlueMuse) and multi-stream (LSL simulator) scenarios.
Integrates a hybrid ML-based classifier with traditional band power level detection.
"""

import time
import numpy as np
import pylsl
import logging
import joblib
from datetime import datetime
from scipy.signal import butter, filtfilt, welch
from PyQt5 import QtCore
from backend.signal_quality_validator import SignalQualityValidator
from backend import database_manager as db_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='neuroflow_debug.log',  # Logs will go to this file
    filemode='w'  # 'w' to overwrite, 'a' to append
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

# Default model path - adjust as needed
DEFAULT_MODEL_PATH = "dataset_results/gradient_boosting_model.joblib"

class HybridEEGClassifier:
    """
    Hybrid classifier that combines ML model prediction with band-power level detection
    """
    
    def __init__(self, model_path=None, sampling_rate=DEFAULT_SAMPLING_RATE):
        self.model = self._load_classifier_model(model_path)
        self.baseline_metrics = None

        self.sampling_rate = sampling_rate
        
        self.feature_names = [
            # Basic band powers
            'alpha_power', 'beta_power', 'theta_power', 'gamma_power',
            
            # Region-specific features
            'frontal_alpha', 'frontal_beta', 'frontal_theta', 'frontal_gamma',
            'temporal_alpha', 'temporal_beta', 'temporal_theta', 'temporal_gamma',
            
            # Hemisphere-specific features
            'left_alpha', 'left_beta', 'left_theta', 'left_gamma',
            'right_alpha', 'right_beta', 'right_theta', 'right_gamma',
            
            # Neuroanatomically-informed ratios
            'temporal_alpha_beta', 'temporal_alpha_theta',
            'frontal_theta_alpha', 'frontal_theta_beta', 'frontal_beta_alpha', 'frontal_beta_theta',
            
            # Asymmetry features
            'frontal_alpha_asymmetry', 'temporal_alpha_asymmetry', 'beta_asymmetry',
            
            # Cross-regional comparisons
            'fronto_temporal_alpha', 'fronto_temporal_theta', 'fronto_temporal_beta',
            
            # Composite indices
            'relaxation_index', 'concentration_index', 'cognitive_load_index',
            
            # Overall band powers and ratios
            'total_power', 'alpha_beta_ratio', 'beta_theta_ratio', 'theta_alpha_ratio',
            
            # Spectral features
            'spectral_centroid', 'spectral_edge_95', 'spectral_spread',
            'dominant_frequency', 'spectral_skewness', 'spectral_kurtosis',
            
            # Temporal features
            'temporal_mean', 'temporal_std', 'temporal_var',
            'temporal_skewness', 'temporal_kurtosis', 'temporal_rms',
            
            # Delta features
            'delta_alpha_power', 'delta_beta_power', 'delta_theta_power', 'delta_gamma_power',
            'delta_alpha_beta_ratio', 'delta_beta_theta_ratio', 'delta_spectral_centroid',
            'rel_delta_alpha_power', 'rel_delta_beta_power', 'rel_delta_theta_power', 'rel_delta_gamma_power',
            'rel_delta_alpha_beta_ratio', 'rel_delta_beta_theta_ratio', 'rel_delta_spectral_centroid',
            
            # Moving average features
            'ma3_alpha_beta_ratio', 'ma3_beta_theta_ratio', 'ma3_alpha_power', 'ma3_beta_power',
            'ma5_alpha_beta_ratio', 'ma5_beta_theta_ratio', 'std5_alpha_beta_ratio', 'std5_beta_theta_ratio',
        ]
        
        # Try to extract feature names from model if available
        if hasattr(self.model, 'feature_names_'):
            self.feature_names = self.model.feature_names_
        
        # Define state levels by category
        self.relaxation_levels = ["Slightly Relaxed", "Relaxed", "Deeply Relaxed"]
        self.concentration_levels = ["Slightly Focused", "Focused", "Highly Focused"]
        self.neutral_levels = ["Neutral"]
        self.tension_levels = ["Slightly Tense", "Tense"]
        self.distraction_levels = ["Slightly Distracted", "Distracted"]
        
        # Thresholds for level determination within each class
        self.level_thresholds = {
            # For relaxation levels, based on alpha increases 
            "relaxed": {
                "alpha_ratio": [1.05, 1.2, 1.4],  # Thresholds for each level
                "beta_ratio_max": [0.9, 0.8, 0.8]  # Maximum beta ratio allowed for each level
            },
            # For concentration levels, based on beta/theta increases
            "concentrating": {
                "beta_ratio": [1.05, 1.2, 1.4],
                "bt_ratio": [1.05, 1.2, 1.4]
            },
            # For tension/distraction (negative states)
            "negative": {
                "alpha_ratio": [0.9, 0.8],  # Below these thresholds = tension levels
                "bt_ratio": [0.9, 0.8]  # Below these thresholds = distraction levels
            }
        }
    
    def _load_classifier_model(self, model_path):
        """Load the trained classifier model"""
        try:
            if model_path:
                logger.info(f"Loading classifier model from {model_path}")
                return joblib.load(model_path)
            else:
                logger.warning("No model path provided, classifier will be disabled")
                return None
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            return None
        
    def _extract_classifier_features(self, metrics):
        """
        Extract the complete feature set for classifier prediction
        """
        try:
            # Check if we have the raw EEG window data
            if 'eeg_window' in metrics:
                # Extract the complete feature set from raw data
                full_features = self._extract_full_features(metrics['eeg_window'])
                if full_features:
                    # Convert the feature dictionary to an array in the right order
                    feature_list = []
                    for feature_name in self.feature_names:
                        if feature_name in full_features:
                            feature_list.append(full_features[feature_name])
                        else:
                            feature_list.append(0.0)  # Default value if feature is missing
                    return np.array(feature_list)
            
            # Fallback: try to construct feature vector from existing metrics
            # This is more likely to fail or be inaccurate
            logger.warning("Raw EEG window not available - using limited feature set")
            feature_vector = np.zeros(73)  # Assuming 73 features expected
            
            # Map the metrics we have to their positions
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in metrics:
                    feature_vector[i] = metrics[feature_name]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting classifier features: {e}")
            raise
    
    def classify_mental_state(self, current_metrics, use_classifier=True,  session_type=None):
        """
        Hybrid classification with detailed logging to debug issues
        """
        if not self.baseline_metrics:
            logger.warning("No baseline metrics available - can't classify")
            return {"state": "Unknown", "level": 0, "confidence": 0.0, "smooth_value": 0.5}
        
        # Calculate band power ratios relative to baseline
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha'] if self.baseline_metrics['alpha'] > 0 else 1
        beta_ratio = current_metrics['beta'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 0 else 1
        theta_ratio = current_metrics['theta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 0 else 1
        bt_ratio = beta_ratio / theta_ratio if theta_ratio > 0 else 1
        ab_ratio = alpha_ratio / beta_ratio if beta_ratio > 0 else 1
        
        # DEBUG: Log the current ratios
        logger.debug(f"METRICS - alpha: {current_metrics['alpha']:.2f}, beta: {current_metrics['beta']:.2f}, theta: {current_metrics['theta']:.2f}")
        logger.debug(f"BASELINE - alpha: {self.baseline_metrics['alpha']:.2f}, beta: {self.baseline_metrics['beta']:.2f}, theta: {self.baseline_metrics['theta']:.2f}")
        logger.debug(f"RATIOS - alpha_ratio: {alpha_ratio:.2f}, beta_ratio: {beta_ratio:.2f}, theta_ratio: {theta_ratio:.2f}, bt_ratio: {bt_ratio:.2f}")
        
        # STEP 1: Try ML classifier with debugging
        predicted_class = "unknown"
        classifier_confidence = 0.0
        
        try:
            if use_classifier and self.model:
                # Extract features needed for classifier
                features = self._extract_classifier_features(current_metrics)
                
                # DEBUG: Log features
                logger.debug(f"FEATURES - Shape: {features.shape}, Max: {np.max(features):.2f}, Min: {np.min(features):.2f}, Mean: {np.mean(features):.2f}")
                
                # Get prediction and probability
                raw_prediction = self.model.predict([features])[0]
                class_probabilities = self.model.predict_proba([features])[0]
                
                # DEBUG: Log raw prediction and probabilities 
                classes = self.model.classes_ if hasattr(self.model, 'classes_') else ["unknown", "concentrating", "relaxed", "neutral"]
                prob_dict = {cls: prob for cls, prob in zip(classes, class_probabilities)}
                logger.debug(f"RAW PREDICTION: {raw_prediction} with probabilities: {prob_dict}")
                
                predicted_class = raw_prediction
                classifier_confidence = max(class_probabilities)
                
            else:
                # Fallback to simple classification
                predicted_class = self._simple_classify(alpha_ratio, beta_ratio, theta_ratio, bt_ratio)
                classifier_confidence = 0.7  # Default confidence for simple method
                logger.debug(f"SIMPLE CLASSIFICATION used (no model): {predicted_class}")
        
        except Exception as e:
            # If ML classification fails, fall back to simple method
            logger.error(f"Error in classifier prediction: {e}")
            predicted_class = self._simple_classify(alpha_ratio, beta_ratio, theta_ratio, bt_ratio)
            classifier_confidence = 0.65  # Lower confidence for fallback
            logger.debug(f"SIMPLE CLASSIFICATION used (after error): {predicted_class}")
        
        # STEP 2: Determine specific level based on band power ratios and predicted class
        state, level, confidence, smooth_value = self._determine_level(
            predicted_class, 
            alpha_ratio, 
            beta_ratio, 
            theta_ratio, 
            bt_ratio,
            ab_ratio,
            classifier_confidence, 
            session_type
        )
        
        # DEBUG: Log final classification results
        logger.debug(f"FINAL CLASSIFICATION: State={state}, Level={level}, Confidence={confidence:.2f}, Value={smooth_value:.2f}")
        
        return {
            "state": state,
            "level": level,
            "confidence": confidence,
            "smooth_value": smooth_value,
            "predicted_class": predicted_class,
            "classifier_confidence": classifier_confidence,
            "alpha_ratio": alpha_ratio,
            "beta_ratio": beta_ratio,
            "theta_ratio": theta_ratio,
            "bt_ratio": bt_ratio
        }
    


    def _extract_full_features(self, window_data):
        """
        Extract complete feature set to match the classifier - Updated to match training exactly
        
        Args:
            window_data: Shape (4, samples) - [TP9, AF7, AF8, TP10]
        """
        features = {}
        
        try:
            # Channel mapping for neuroanatomical analysis (same as training)
            temporal_channels = [0, 3]  # TP9, TP10
            frontal_channels = [1, 2]   # AF7, AF8
            left_channels = [0, 1]      # TP9, AF7
            right_channels = [2, 3]     # AF8, TP10
            
            # === 1. CALCULATE CHANNEL-SPECIFIC BAND POWERS (MATCH TRAINING) ===
            channel_band_powers = self._calculate_channel_band_powers_training_style(window_data)
            if not channel_band_powers:
                return None
            
            # === 2. OVERALL BAND POWERS (for backward compatibility) ===
            total_powers = {}
            for band in ['theta', 'alpha', 'beta', 'gamma']:
                total_powers[band] = np.mean([channel_band_powers[ch][band] for ch in range(4)])
                features[f'{band}_power'] = total_powers[band]
            
            # Total power
            features['total_power'] = sum(total_powers.values())
            
            # Classic ratios
            features['alpha_beta_ratio'] = total_powers['alpha'] / total_powers['beta'] if total_powers['beta'] > 0 else 0
            features['beta_theta_ratio'] = total_powers['beta'] / total_powers['theta'] if total_powers['theta'] > 0 else 0
            features['theta_alpha_ratio'] = total_powers['theta'] / total_powers['alpha'] if total_powers['alpha'] > 0 else 0
            
            # === 3. REGION-SPECIFIC FEATURES ===
            
            # Temporal region powers
            temporal_powers = {}
            for band in ['theta', 'alpha', 'beta', 'gamma']:
                temporal_powers[band] = np.mean([channel_band_powers[ch][band] for ch in temporal_channels])
                features[f'temporal_{band}'] = temporal_powers[band]
            
            # Frontal region powers
            frontal_powers = {}
            for band in ['theta', 'alpha', 'beta', 'gamma']:
                frontal_powers[band] = np.mean([channel_band_powers[ch][band] for ch in frontal_channels])
                features[f'frontal_{band}'] = frontal_powers[band]
            
            # === 4. HEMISPHERE-SPECIFIC FEATURES ===
            
            # Left hemisphere
            left_powers = {}
            for band in ['theta', 'alpha', 'beta', 'gamma']:
                left_powers[band] = np.mean([channel_band_powers[ch][band] for ch in left_channels])
                features[f'left_{band}'] = left_powers[band]
            
            # Right hemisphere  
            right_powers = {}
            for band in ['theta', 'alpha', 'beta', 'gamma']:
                right_powers[band] = np.mean([channel_band_powers[ch][band] for ch in right_channels])
                features[f'right_{band}'] = right_powers[band]
            
            # === 5. NEUROANATOMICALLY-INFORMED RATIOS ===
            
            # Alpha ratios (temporal regions more sensitive)
            features['temporal_alpha_beta'] = temporal_powers['alpha'] / temporal_powers['beta'] if temporal_powers['beta'] > 0 else 0
            features['temporal_alpha_theta'] = temporal_powers['alpha'] / temporal_powers['theta'] if temporal_powers['theta'] > 0 else 0
            
            # Theta ratios (frontal regions for cognitive load)
            features['frontal_theta_alpha'] = frontal_powers['theta'] / frontal_powers['alpha'] if frontal_powers['alpha'] > 0 else 0
            features['frontal_theta_beta'] = frontal_powers['theta'] / frontal_powers['beta'] if frontal_powers['beta'] > 0 else 0
            
            # Beta ratios (frontal regions for concentration)
            features['frontal_beta_alpha'] = frontal_powers['beta'] / frontal_powers['alpha'] if frontal_powers['alpha'] > 0 else 0
            features['frontal_beta_theta'] = frontal_powers['beta'] / frontal_powers['theta'] if frontal_powers['theta'] > 0 else 0
            
            # === 6. ASYMMETRY FEATURES ===
            
            # Frontal alpha asymmetry (approach/withdrawal)
            frontal_left_alpha = channel_band_powers[1]['alpha']  # AF7
            frontal_right_alpha = channel_band_powers[2]['alpha']  # AF8
            features['frontal_alpha_asymmetry'] = (frontal_right_alpha - frontal_left_alpha) / (frontal_right_alpha + frontal_left_alpha) if (frontal_right_alpha + frontal_left_alpha) > 0 else 0
            
            # Temporal alpha asymmetry
            temporal_left_alpha = channel_band_powers[0]['alpha']  # TP9
            temporal_right_alpha = channel_band_powers[3]['alpha']  # TP10
            features['temporal_alpha_asymmetry'] = (temporal_right_alpha - temporal_left_alpha) / (temporal_right_alpha + temporal_left_alpha) if (temporal_right_alpha + temporal_left_alpha) > 0 else 0
            
            # Beta asymmetry (concentration lateralization)
            features['beta_asymmetry'] = (right_powers['beta'] - left_powers['beta']) / (right_powers['beta'] + left_powers['beta']) if (right_powers['beta'] + left_powers['beta']) > 0 else 0
            
            # === 7. CROSS-REGIONAL COMPARISONS ===
            
            # Frontal-Temporal ratios (attention vs relaxation)
            features['fronto_temporal_alpha'] = frontal_powers['alpha'] / temporal_powers['alpha'] if temporal_powers['alpha'] > 0 else 0
            features['fronto_temporal_theta'] = frontal_powers['theta'] / temporal_powers['theta'] if temporal_powers['theta'] > 0 else 0
            features['fronto_temporal_beta'] = frontal_powers['beta'] / temporal_powers['beta'] if temporal_powers['beta'] > 0 else 0
            
            # === 8. COMPOSITE INDICES ===
            
            # Relaxation index: High temporal alpha, low frontal beta/theta
            relaxation_numerator = temporal_powers['alpha']
            relaxation_denominator = frontal_powers['beta'] + frontal_powers['theta']
            features['relaxation_index'] = relaxation_numerator / relaxation_denominator if relaxation_denominator > 0 else 0
            
            # Concentration index: High frontal beta/theta, low temporal alpha
            concentration_numerator = frontal_powers['beta'] + frontal_powers['theta']
            concentration_denominator = temporal_powers['alpha']
            features['concentration_index'] = concentration_numerator / concentration_denominator if concentration_denominator > 0 else 0
            
            # Cognitive load index: Frontal theta/alpha ratio
            features['cognitive_load_index'] = frontal_powers['theta'] / frontal_powers['alpha'] if frontal_powers['alpha'] > 0 else 0
            
            # === 9. SPECTRAL FEATURES (MATCH TRAINING) ===
            spectral_features = self._calculate_spectral_features_training_style(window_data)
            features.update(spectral_features)
            
            # === 10. TEMPORAL FEATURES (MATCH TRAINING) ===
            temporal_features = self._calculate_temporal_features_training_style(window_data)
            features.update(temporal_features)
            
            # === 11. DELTA FEATURES (Use history if available) ===
            delta_features = self._calculate_delta_features(features)
            features.update(delta_features)
            
            # === 12. MOVING AVERAGE FEATURES (Use history if available) ===
            ma_features = self._calculate_moving_average_features(features)
            features.update(ma_features)
            
            return features
                
        except Exception as e:
            logger.error(f"Error extracting full features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_channel_band_powers_training_style(self, window_data):
        """Calculate band powers for each channel using training-style method"""
        
        try:
            channel_powers = []
            fs = self.sampling_rate
            
            # Use same parameters as training
            nperseg = min(256, window_data.shape[1])
            noverlap = nperseg // 2
            
            for ch_idx in range(window_data.shape[0]):
                channel_data = window_data[ch_idx, :]
                
                # Calculate PSD using Welch method (SAME AS TRAINING)
                from scipy.signal import welch
                freqs, psd = welch(
                    channel_data,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window='hann',
                    detrend='constant'
                )
                
                # Calculate band powers
                ch_band_powers = {}
                bands = {
                    'theta': (4, 8),
                    'alpha': (8, 13), 
                    'beta': (13, 30),
                    'gamma': (30, 50)
                }
                
                for band_name, (low_freq, high_freq) in bands.items():
                    freq_mask = (freqs >= low_freq) & (freqs < high_freq)
                    if np.any(freq_mask):
                        band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                        ch_band_powers[band_name] = band_power
                    else:
                        ch_band_powers[band_name] = 0
                
                channel_powers.append(ch_band_powers)
            
            return channel_powers
                
        except Exception as e:
            logger.error(f"Error calculating channel band powers: {e}")
            return None

    def _calculate_spectral_features_training_style(self, window_data):
        """Calculate spectral features exactly like training"""
        
        features = {}
        
        try:
            from scipy.signal import welch
            from scipy.stats import skew, kurtosis
            
            all_psds = []
            fs = self.sampling_rate
            nperseg = min(256, window_data.shape[1])
            noverlap = nperseg // 2
            
            for ch_idx in range(window_data.shape[0]):
                channel_data = window_data[ch_idx, :]
                
                freqs, psd = welch(
                    channel_data,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window='hann',
                    detrend='constant'
                )
                
                all_psds.append(psd)
            
            # Average PSD across channels
            avg_psd = np.mean(all_psds, axis=0)
            
            # Spectral features (SAME AS TRAINING)
            features['spectral_centroid'] = np.sum(freqs * avg_psd) / np.sum(avg_psd) if np.sum(avg_psd) > 0 else 0
            
            cumsum_psd = np.cumsum(avg_psd)
            total_power = cumsum_psd[-1]
            if total_power > 0:
                edge_95_idx = np.where(cumsum_psd >= 0.95 * total_power)[0]
                features['spectral_edge_95'] = freqs[edge_95_idx[0]] if len(edge_95_idx) > 0 else freqs[-1]
            else:
                features['spectral_edge_95'] = 0
            
            centroid = features['spectral_centroid']
            if np.sum(avg_psd) > 0:
                features['spectral_spread'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * avg_psd) / np.sum(avg_psd))
            else:
                features['spectral_spread'] = 0
            
            dominant_freq_idx = np.argmax(avg_psd)
            features['dominant_frequency'] = freqs[dominant_freq_idx]
            
            features['spectral_skewness'] = skew(avg_psd)
            features['spectral_kurtosis'] = kurtosis(avg_psd)
            
        except Exception as e:
            logger.error(f"Error calculating spectral features: {e}")
            for feat_name in ['spectral_centroid', 'spectral_edge_95', 'spectral_spread', 
                            'dominant_frequency', 'spectral_skewness', 'spectral_kurtosis']:
                features[feat_name] = 0
        
        return features

    def _calculate_temporal_features_training_style(self, window_data):
        """Calculate temporal/statistical features exactly like training"""
        
        features = {}
        
        try:
            from scipy.stats import skew, kurtosis
            channel_features = []
            
            for ch_idx in range(window_data.shape[0]):
                channel_data = window_data[ch_idx, :]
                
                ch_features = {}
                ch_features['mean'] = np.mean(channel_data)
                ch_features['std'] = np.std(channel_data)
                ch_features['var'] = np.var(channel_data)
                ch_features['skewness'] = skew(channel_data)
                ch_features['kurtosis'] = kurtosis(channel_data)
                
                # RMS amplitude
                ch_features['rms'] = np.sqrt(np.mean(channel_data**2))
                
                channel_features.append(ch_features)
            
            # Average across channels (SAME AS TRAINING)
            for feat_name in channel_features[0].keys():
                features[f'temporal_{feat_name}'] = np.mean([ch[feat_name] for ch in channel_features])
                
        except Exception as e:
            logger.error(f"Error calculating temporal features: {e}")
            default_features = [
                'temporal_mean', 'temporal_std', 'temporal_var', 'temporal_skewness', 
                'temporal_kurtosis', 'temporal_rms'
            ]
            for feat_name in default_features:
                features[feat_name] = 0
        
        return features

    def _calculate_delta_features(self, current_features):
        """Calculate delta features using history"""
        delta_features = {}
        
        # If we have history, calculate deltas
        if hasattr(self, 'feature_history') and len(self.feature_history) > 0:
            prev_features = self.feature_history[-1]
            
            key_features = ['alpha_power', 'beta_power', 'theta_power', 'gamma_power',
                        'alpha_beta_ratio', 'beta_theta_ratio', 'spectral_centroid']
            
            for feat in key_features:
                if feat in current_features and feat in prev_features:
                    delta_val = current_features[feat] - prev_features[feat]
                    delta_features[f'delta_{feat}'] = delta_val
                    
                    if prev_features[feat] != 0:
                        delta_features[f'rel_delta_{feat}'] = delta_val / prev_features[feat]
                    else:
                        delta_features[f'rel_delta_{feat}'] = 0
                else:
                    delta_features[f'delta_{feat}'] = 0
                    delta_features[f'rel_delta_{feat}'] = 0
        else:
            # No history, set to zero
            key_features = ['alpha_power', 'beta_power', 'theta_power', 'gamma_power',
                        'alpha_beta_ratio', 'beta_theta_ratio', 'spectral_centroid']
            for feat in key_features:
                delta_features[f'delta_{feat}'] = 0
                delta_features[f'rel_delta_{feat}'] = 0
        
        return delta_features

    def _calculate_moving_average_features(self, current_features):
        """Calculate moving average features using history"""
        ma_features = {}
        
        # Initialize feature history if not exists
        if not hasattr(self, 'feature_history'):
            self.feature_history = []
        
        # Add current features to history
        self.feature_history.append(current_features.copy())
        
        # Limit history size
        if len(self.feature_history) > 10:
            self.feature_history.pop(0)
        
        # Calculate moving averages
        if len(self.feature_history) >= 3:
            recent_3 = self.feature_history[-3:]
            for feat in ['alpha_beta_ratio', 'beta_theta_ratio', 'alpha_power', 'beta_power']:
                values = [f.get(feat, 0) for f in recent_3]
                ma_features[f'ma3_{feat}'] = np.mean(values)
        else:
            for feat in ['alpha_beta_ratio', 'beta_theta_ratio', 'alpha_power', 'beta_power']:
                ma_features[f'ma3_{feat}'] = current_features.get(feat, 0)
        
        # 5-window moving averages
        if len(self.feature_history) >= 5:
            recent_5 = self.feature_history[-5:]
            for feat in ['alpha_beta_ratio', 'beta_theta_ratio']:
                values = [f.get(feat, 0) for f in recent_5]
                ma_features[f'ma5_{feat}'] = np.mean(values)
                ma_features[f'std5_{feat}'] = np.std(values)
        else:
            for feat in ['alpha_beta_ratio', 'beta_theta_ratio']:
                ma_features[f'ma5_{feat}'] = ma_features.get(f'ma3_{feat}', 0)
                ma_features[f'std5_{feat}'] = 0
        
        return ma_features
    
    
    def _simple_classify(self, alpha_ratio, beta_ratio, theta_ratio, bt_ratio):
        """Simple classification method as fallback"""
        if alpha_ratio > 1.1 and beta_ratio < 0.9:
            return "relaxed"
        elif beta_ratio > 1.1 and bt_ratio > 1.05:
            return "concentrating"
        else:
            return "neutral"
    
    def _determine_level(self, predicted_class, alpha_ratio, beta_ratio, theta_ratio, bt_ratio, ab_ratio, classifier_confidence, session_type=None):
        """
        Determine specific level based on predicted class and band power ratios relative to baseline.
        Uses classifier confidence to adjust thresholds and adapt state determination.
        
        Args:
            predicted_class: The class predicted by the ML model (e.g., "relaxed", "concentrating", etc.)
            alpha_ratio: Current alpha power relative to baseline
            beta_ratio: Current beta power relative to baseline  
            theta_ratio: Current theta power relative to baseline
            bt_ratio: Beta/theta ratio relative to baseline
            ab_ratio: Alpha/beta ratio relative to baseline
            classifier_confidence: Confidence score from the classifier
            session_type: Type of session (RELAXATION or FOCUS)
            
        Returns:
            Tuple of (state, level, confidence, smooth_value)
        """
        # Default values
        state = "Neutral"
        level = 0
        confidence = classifier_confidence
        smooth_value = 0.5
        
        # Determine if we're in a relaxation or focus session
        is_relaxation = session_type == SESSION_TYPE_RELAX if session_type else None
        is_focus = session_type == SESSION_TYPE_FOCUS if session_type else None
        
        # Weight our decisions by classifier confidence
        # Low confidence = more conservative level determination
        confidence_weight = min(1.0, classifier_confidence)
        
        # Level thresholds adjusted by confidence
        # Lower confidence = stricter thresholds for positive states
        threshold_adj = (1 - confidence_weight) * 0.2
        
        alpha_high = 1.4 + threshold_adj
        alpha_med = 1.2 + threshold_adj
        alpha_low = 1.05 + threshold_adj
        
        beta_high = 1.4 + threshold_adj
        beta_med = 1.2 + threshold_adj
        beta_low = 1.1 + threshold_adj
        
        bt_high = 1.4 + threshold_adj
        bt_med = 1.2 + threshold_adj
        bt_low = 1.05 + threshold_adj
        
        # Negative state thresholds (less affected by confidence)
        alpha_low_thresh = 0.8 - threshold_adj * 0.5
        alpha_very_low = 0.5 - threshold_adj * 0.5
        beta_high_thresh = 1.2 - threshold_adj * 0.5
        beta_very_high = 1.5 - threshold_adj * 0.5
        
        # FIRST: Honor the classifier's prediction for the main state
        if predicted_class == "relaxed" or predicted_class == 2:  # Relaxed state
            # For relaxation sessions, use relaxation terminology
            if is_relaxation or is_relaxation is None:
                if alpha_ratio > alpha_high:
                    state = "Deeply Relaxed"
                    level = 3
                    confidence = min(0.95, classifier_confidence + 0.1 * confidence_weight)
                    smooth_value = 0.9
                elif alpha_ratio > alpha_med:
                    state = "Relaxed"
                    level = 2
                    confidence = min(0.9, classifier_confidence + 0.05 * confidence_weight)
                    smooth_value = 0.75
                elif alpha_ratio > alpha_low:
                    state = "Slightly Relaxed"
                    level = 1
                    confidence = min(0.85, classifier_confidence)
                    smooth_value = 0.6
                else:
                    state = "Slightly Relaxed"
                    level = 1
                    confidence = min(0.8, classifier_confidence - 0.05 * (1 - confidence_weight))
                    smooth_value = 0.55
            # For focus sessions, relaxation is off-target
            elif is_focus:
                if alpha_ratio > alpha_high:
                    state = "Too Relaxed"
                    level = -2
                    confidence = min(0.9, classifier_confidence)
                    smooth_value = 0.2
                elif alpha_ratio > alpha_med:
                    state = "Relaxed"
                    level = -1
                    confidence = min(0.85, classifier_confidence)
                    smooth_value = 0.35
                else:
                    state = "Slightly Relaxed"
                    level = -1
                    confidence = min(0.8, classifier_confidence)
                    smooth_value = 0.4
                    
        elif predicted_class == "concentrating" or predicted_class == 1:  # Concentrated/focused state
            # For focus sessions, use focus terminology
            if is_focus or is_relaxation is None:
                if beta_ratio > beta_high and bt_ratio > bt_high:
                    state = "Highly Focused"
                    level = 3
                    confidence = min(0.95, classifier_confidence + 0.1 * confidence_weight)
                    smooth_value = 0.9
                elif beta_ratio > beta_med and bt_ratio > bt_med:
                    state = "Focused"
                    level = 2
                    confidence = min(0.9, classifier_confidence + 0.05 * confidence_weight)
                    smooth_value = 0.75
                elif beta_ratio > beta_low and bt_ratio > bt_low:
                    state = "Slightly Focused"
                    level = 1
                    confidence = min(0.85, classifier_confidence)
                    smooth_value = 0.6
                else:
                    state = "Attentive"
                    level = 1
                    confidence = min(0.8, classifier_confidence - 0.05 * (1 - confidence_weight))
                    smooth_value = 0.55
            # For relaxation sessions, concentration could be tense/alert
            elif is_relaxation:
                if beta_ratio > beta_high and bt_ratio > bt_high:
                    state = "Alert"
                    level = -2
                    confidence = min(0.9, classifier_confidence)
                    smooth_value = 0.2
                elif beta_ratio > beta_med:
                    state = "Slightly Alert"
                    level = -1
                    confidence = min(0.85, classifier_confidence)
                    smooth_value = 0.35
                else:
                    state = "Attentive"
                    level = 0
                    confidence = min(0.8, classifier_confidence)
                    smooth_value = 0.45
                    
        elif predicted_class == "neutral" or predicted_class == 0:  # Neutral state
            # Neutral is neutral regardless of session type
            state = "Neutral"
            level = 0
            confidence = classifier_confidence
            smooth_value = 0.5
            
        # SECOND: Check for negative states based on ratios - these can override classifier predictions
        # when patterns are very strong, but are weighted by classifier confidence
        
        # Very low alpha + high beta could indicate tension regardless of classifier
        if alpha_ratio < alpha_very_low and beta_ratio > beta_very_high:
            override_confidence = 0.9 - (confidence_weight * 0.2)  # Lower override confidence if classifier is confident
            
            if override_confidence > classifier_confidence:  # Only override if more confident than classifier
                if is_relaxation:
                    state = "Tense"
                    level = -3
                    confidence = override_confidence
                    smooth_value = 0.1
                elif is_focus:
                    state = "Overthinking"
                    level = -2
                    confidence = override_confidence
                    smooth_value = 0.25
                else:
                    state = "Tense"
                    level = -2
                    confidence = override_confidence
                    smooth_value = 0.2
                
        # High theta + low beta could indicate distraction/drowsiness
        elif theta_ratio > 1.5 and beta_ratio < 0.8:
            override_confidence = 0.85 - (confidence_weight * 0.2)
            
            if override_confidence > classifier_confidence:
                if is_relaxation:
                    state = "Drowsy"
                    level = 1  # Actually good for relaxation
                    confidence = override_confidence
                    smooth_value = 0.65
                elif is_focus:
                    state = "Distracted"
                    level = -2
                    confidence = override_confidence
                    smooth_value = 0.25
                else:
                    state = "Drowsy"
                    level = -1
                    confidence = override_confidence
                    smooth_value = 0.4
                    
        # High alpha + low beta could indicate deep relaxation
        elif alpha_ratio > 1.6 and beta_ratio < 0.7:
            override_confidence = 0.9 - (confidence_weight * 0.2)
            
            if override_confidence > classifier_confidence:
                if is_relaxation:
                    state = "Deeply Relaxed"
                    level = 3
                    confidence = override_confidence
                    smooth_value = 0.9
                elif is_focus:
                    state = "Too Relaxed" 
                    level = -2
                    confidence = override_confidence
                    smooth_value = 0.2
                else:
                    state = "Deeply Relaxed"
                    level = 2
                    confidence = override_confidence
                    smooth_value = 0.8
        
        # Debug output
        logger.debug(f"SESSION TYPE: {session_type}, PREDICTION: {predicted_class}, CONFIDENCE: {classifier_confidence:.2f}")
        logger.debug(f"RATIOS - alpha: {alpha_ratio:.2f}, beta: {beta_ratio:.2f}, theta: {theta_ratio:.2f}, bt: {bt_ratio:.2f}")
        logger.debug(f"DETERMINED: {state} (level={level}), confidence={confidence:.2f}, value={smooth_value:.2f}")
        
        return state, level, confidence, smooth_value
    
    def set_baseline(self, baseline_metrics):
        """Set baseline metrics for relative comparisons"""
        self.baseline_metrics = baseline_metrics


class EEGProcessingWorker(QtCore.QObject):
    """
    EEG Processing Worker that runs in a separate thread.
    Handles all EEG processing, data accumulation, and database saving.
    Now supports both single-stream and multi-stream LSL setups.
    Integrated with hybrid ML-based and band power classification.
    """
    
    # Signals for UI communication
    connection_status_changed = QtCore.pyqtSignal(str, str)  # status, message
    calibration_progress = QtCore.pyqtSignal(float)  # 0.0 to 1.0
    calibration_status_changed = QtCore.pyqtSignal(str, dict)  # status, data
    new_prediction = QtCore.pyqtSignal(dict)  # prediction data for UI feedback only
    signal_quality_update = QtCore.pyqtSignal(dict)  # real-time quality metrics
    error_occurred = QtCore.pyqtSignal(str)  # error message
    session_saved = QtCore.pyqtSignal(int, dict)  # session_id, summary_stats
    
    def __init__(self, parent=None, model_path=DEFAULT_MODEL_PATH):
        super().__init__(parent)
        
        # Initialize Hybrid Classifier
        self.hybrid_classifier = HybridEEGClassifier(model_path)
        
        # Connection state
        self.lsl_inlet = None
        self.lsl_accelerometer_inlet = None  # Separate accelerometer inlet
        self.is_multi_stream = False  # Track if we're using multi-stream setup
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

        self.set_testing_mode(True)
        
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
        
    def set_testing_mode(self, enabled=True):
        """Enable/disable testing mode with reduced stability requirements"""
        if enabled:
            # Remember original values
            self._original_min_confidence = self.min_confidence_for_change
            self._original_state_change_threshold = self.state_change_threshold  
            self._original_min_stable_count = self.min_stable_count
            
            # Set more permissive values for testing
            self.min_confidence_for_change = 0.4  # Lower confidence threshold
            self.state_change_threshold = 0.5     # Lower level change threshold
            self.min_stable_count = 1             # Only need 1 prediction to change state
            
            logger.info("TESTING MODE ENABLED - Reduced stability requirements")
        else:
            # Restore original values if available
            if hasattr(self, '_original_min_confidence'):
                self.min_confidence_for_change = self._original_min_confidence
                self.state_change_threshold = self._original_state_change_threshold
                self.min_stable_count = self._original_min_stable_count
                logger.info("TESTING MODE DISABLED - Restored normal stability requirements")
    
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
        
        # Bandpass filter (0.5 - 30 Hz)
        low = lowcut / nyq
        high = highcut / nyq
        self.b_bandpass, self.a_bandpass = butter(filter_order, [low, high], btype='band', analog=False)
        
        # Notch filter (50 Hz)
        notch_freq = 50.0  # Power line frequency (50 Hz in Europe/many countries, 60 Hz in US)
        notch_width = 2.0  # Width of the notch
        
        # Create notch filter
        w0 = notch_freq / nyq
        quality_factor = notch_freq / notch_width
        if 0 < w0 < 1:
            from scipy.signal import iirnotch
            self.b_notch, self.a_notch = iirnotch(w0, quality_factor)
            self.use_notch = True
        else:
            self.use_notch = False
            
        # Update FFT parameters
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
            
            # Set baseline in hybrid classifier
            self.hybrid_classifier.set_baseline(self.baseline_metrics)
            
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
        """Process EEG data for real-time feedback and accumulate session data - Updated for hybrid classifier"""
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
                    current_metrics['eeg_window'] = filtered_window
                    # Add to signal quality validator
                    self.signal_quality_validator.add_band_power_data(current_metrics)
                    self.signal_quality_validator.add_raw_eeg_data(eeg_window)
                    
                    # Get signal quality assessment
                    signal_quality_metrics = self.signal_quality_validator.assess_overall_quality()
                    
                    # Update history
                    self.recent_metrics_history.append(current_metrics)
                    if len(self.recent_metrics_history) > 15:
                        self.recent_metrics_history.pop(0)
                    
                    # Use hybrid classifier instead of simple classification
                    hybrid_classification = self.hybrid_classifier.classify_mental_state(
                        current_metrics, 
                        use_classifier=True,
                        session_type=self.current_session_type
                    )
                    
                    # Apply prediction smoothing
                    smoothed_classification = self._apply_prediction_smoothing(
                        hybrid_classification["state"], 
                        hybrid_classification["level"],
                        hybrid_classification["confidence"],
                        hybrid_classification["smooth_value"]
                    )
                    
                    # Store current prediction
                    self.current_prediction = smoothed_classification
                    
                    # ACCUMULATE ALL SESSION DATA
                    self._accumulate_session_data(smoothed_classification, current_metrics, filtered_window)
                    
                    # Prepare prediction data for UI feedback only
                    prediction_data = {
                        "message_type": "PREDICTION",
                        "timestamp": time.time(),
                        "session_type": self.current_session_type,
                        "classification": smoothed_classification,
                        "smooth_value": smoothed_classification.get('smooth_value', 0.5),
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

    def _accumulate_session_data(self, classification, current_metrics, filtered_window):
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
        
        # Store FILTERED EEG data
        if filtered_window.shape[1] >= self.nfft:
            # The window is now already filtered, so we can proceed directly.
            
            # Proper downsampling with anti-aliasing
            downsample_factor = 4
            if filtered_window.shape[1] >= downsample_factor:
                # Apply anti-aliasing filter before downsampling
                from scipy.signal import decimate
                try:
                    # Decimate properly handles anti-aliasing
                    downsampled_eeg = np.zeros((4, filtered_window.shape[1] // downsample_factor))
                    for ch in range(4):
                        downsampled_eeg[ch, :] = decimate(filtered_window[ch, :], downsample_factor, ftype='iir')
                except:
                    # Fallback to simple method if decimate fails
                    downsampled_indices = np.arange(0, filtered_window.shape[1], downsample_factor)
                    downsampled_eeg = filtered_window[:, downsampled_indices]
            else:
                downsampled_eeg = filtered_window
            
            # Store each sample with proper timing
            sample_interval = downsample_factor / self.sampling_rate
            for i in range(downsampled_eeg.shape[1]):
                self.session_eeg_data["channel_0"].append(float(downsampled_eeg[0, i]))
                self.session_eeg_data["channel_1"].append(float(downsampled_eeg[1, i]))
                self.session_eeg_data["channel_2"].append(float(downsampled_eeg[2, i]))
                self.session_eeg_data["channel_3"].append(float(downsampled_eeg[3, i]))
                self.session_eeg_data["timestamps"].append(current_time + (i * sample_interval))
        
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
    
    def _filter_eeg_data(self, eeg_data):
        """Apply bandpass and notch filters to EEG data"""
        min_samples = 3 * filter_order + 1
        
        if eeg_data.shape[1] < min_samples:
            return eeg_data
            
        eeg_filtered = np.zeros_like(eeg_data)
        for i in range(NUM_EEG_CHANNELS):
            channel_data = eeg_data[i]
            
            # Apply bandpass filter
            filtered_signal = filtfilt(self.b_bandpass, self.a_bandpass, channel_data)
            
            # Apply notch filter if enabled
            if hasattr(self, 'use_notch') and self.use_notch:
                filtered_signal = filtfilt(self.b_notch, self.a_notch, filtered_signal)
            
            eeg_filtered[i] = filtered_signal
            
        return eeg_filtered
    
    def _calculate_band_powers(self, eeg_segment):
        """Calculate band powers for EEG segment - Robust version with better error handling"""
        if eeg_segment.shape[1] < 64:  # Need minimum samples
            logger.debug(f"Insufficient samples for band power calculation: {eeg_segment.shape[1]}")
            return None
            
        try:
            # Apply more lenient artifact rejection
            artifact_mask = self._lenient_artifact_rejection(eeg_segment)
            if np.sum(artifact_mask) < 0.5 * eeg_segment.shape[1]:  # More lenient threshold
                logger.debug("Too many artifacts detected, skipping window")
                return None
            
            # Calculate band powers for each channel using scipy (matches training)
            metrics_list = []
            for ch_idx in range(NUM_EEG_CHANNELS):
                ch_data = eeg_segment[ch_idx, artifact_mask].copy() if np.any(artifact_mask) else eeg_segment[ch_idx].copy()
                
                # Ensure minimum length
                if len(ch_data) < 64:
                    ch_data = np.pad(ch_data, (0, 64 - len(ch_data)), mode='constant', constant_values=0)
                
                try:
                    # Primary method: Use scipy.signal.welch (matches training)
                    from scipy.signal import welch
                    
                    nperseg = min(256, len(ch_data))
                    noverlap = nperseg // 2
                    
                    freqs, psd = welch(
                        ch_data,
                        fs=self.sampling_rate,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        window='hann',
                        detrend='constant'
                    )
                    
                    # Calculate band powers
                    ch_metrics = {}
                    bands = {
                        'theta': THETA_BAND,
                        'alpha': ALPHA_BAND,
                        'beta': BETA_BAND
                    }
                    
                    for band_name, (low_freq, high_freq) in bands.items():
                        freq_mask = (freqs >= low_freq) & (freqs < high_freq)
                        if np.any(freq_mask):
                            band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                            ch_metrics[band_name] = max(0.001, band_power)  # Avoid zero division
                        else:
                            ch_metrics[band_name] = 0.001
                    
                    metrics_list.append(ch_metrics)
                    
                except Exception as e:
                    logger.warning(f"Scipy method failed for channel {ch_idx}: {e}, trying BrainFlow fallback")
                    
                    # Fallback: Try BrainFlow method
                    try:
                        if BRAINFLOW_AVAILABLE:
                            # Simple detrend
                            ch_data_detrended = ch_data - np.mean(ch_data)
                            
                            psd = DataFilter.get_psd_welch(
                                ch_data_detrended.astype(np.float64), 
                                min(256, len(ch_data_detrended)), 
                                min(128, len(ch_data_detrended) // 2), 
                                int(self.sampling_rate), 
                                WindowOperations.HANNING.value
                            )
                            
                            ch_metrics = {
                                'theta': max(0.001, DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1])),
                                'alpha': max(0.001, DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1])),
                                'beta': max(0.001, DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1]))
                            }
                            
                            metrics_list.append(ch_metrics)
                            
                        else:
                            logger.error(f"Both scipy and BrainFlow failed for channel {ch_idx}")
                            return None
                            
                    except Exception as e2:
                        logger.error(f"BrainFlow fallback also failed for channel {ch_idx}: {e2}")
                        return None
                        
            if len(metrics_list) != NUM_EEG_CHANNELS:
                logger.error(f"Expected {NUM_EEG_CHANNELS} channel metrics, got {len(metrics_list)}")
                return None
                
            # Calculate weighted average
            avg_metrics = {
                'theta': np.sum([m['theta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
                'alpha': np.sum([m['alpha'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
                'beta': np.sum([m['beta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)])
            }
            
            # Calculate ratios with safe division
            avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 0.001 else 0
            avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 0.001 else 0
            
            logger.debug(f"Successfully calculated band powers: alpha={avg_metrics['alpha']:.3f}, beta={avg_metrics['beta']:.3f}, theta={avg_metrics['theta']:.3f}")
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Band power calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _lenient_artifact_rejection(self, eeg_data):
        """More lenient artifact rejection"""
        # Use more permissive thresholds
        channel_thresholds = [300, 200, 200, 300]  # Doubled from original
        amplitude_mask = ~np.any(np.abs(eeg_data) > np.array(channel_thresholds).reshape(-1, 1), axis=0)
        
        diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
        if eeg_data.shape[1] > 1:
            diff_thresholds = [100, 60, 60, 100]  # Doubled from original
            diff_mask = ~np.any(
                np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
                np.array(diff_thresholds).reshape(-1, 1), 
                axis=0
            )
        
        return amplitude_mask & diff_mask
    
    def _apply_prediction_smoothing(self, raw_state, raw_level, confidence, raw_value):
        """Apply smoothing and stability requirements with momentum for realistic brain-like transitions"""
        # Get previous smoothed value
        if self.prediction_history:
            prev_smooth_value = self.prediction_history[-1]['smooth_value']
            prev_state = self.prediction_history[-1]['state']
            prev_level = self.prediction_history[-1]['level']
        else:
            prev_smooth_value = 0.5
            prev_state = "Unknown"
            prev_level = 0
        
        # Apply exponential smoothing with momentum (physics-based)
        # This creates more realistic, gradual transitions
        
        # 1. For the smooth_value (continuous measure)
        target_value = raw_value
        current_value = prev_smooth_value
        
        # Physics-based smoothing with momentum
        # Calculate "force" based on distance to target
        force = (target_value - current_value) * (1.0 - self.state_momentum)
        
        # Update velocity with momentum
        self.state_velocity = self.state_velocity * self.state_momentum + force
        
        # Apply velocity to position (smooth_value)
        smooth_value = current_value + self.state_velocity
        
        # Clamp to valid range
        smooth_value = max(0.0, min(1.0, smooth_value))
        
        # 2. For level transitions, need discrete but smooth behavior
        level_diff = raw_level - prev_level
        
        # Only allow level changes when confidence is sufficient and change is significant
        if abs(level_diff) >= self.state_change_threshold and confidence >= self.min_confidence_for_change:
            # Update level velocity using momentum
            self.level_velocity = self.level_velocity * self.level_momentum + level_diff * (1.0 - self.level_momentum)
            
            # Only change level when velocity exceeds threshold
            if abs(self.level_velocity) >= 0.5:
                # Direction of change
                level_step = 1 if self.level_velocity > 0 else -1
                
                # Apply step change to level (constrained)
                final_level = prev_level + level_step
                
                # Reset velocity partially after change
                self.level_velocity *= 0.5
                
                # If this is a new state type altogether, check for consistency
                if raw_state != prev_state:
                    if self.last_stable_prediction == raw_state:
                        self.stable_prediction_count += 1
                    else:
                        self.last_stable_prediction = raw_state
                        self.stable_prediction_count = 1
                    
                    # Only change state type if we have consistent predictions
                    if self.stable_prediction_count >= self.min_stable_count:
                        final_state = raw_state
                        final_confidence = confidence
                    else:
                        final_state = prev_state
                        final_confidence = confidence * 0.7
                else:
                    # Same state type, just different level
                    final_state = raw_state
                    final_confidence = confidence
            else:
                # Not enough momentum for level change
                final_state = prev_state
                final_level = prev_level
                final_confidence = confidence * 0.8
        else:
            # No significant change
            final_state = prev_state
            final_level = prev_level
            final_confidence = confidence * 0.8
        
        # Create final prediction
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