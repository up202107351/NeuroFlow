
"""
Windowed Binary Classifier Evaluation - Matching EEG Processing Worker

This script properly emulates the real-time windowed processing approach
used in the EEG processing worker, with calibration phase and rolling predictions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from scipy import signal
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("Warning: BrainFlow not available, using scipy for PSD calculations")

class WindowedEEGClassifier:
    """
    Windowed EEG classifier that exactly matches EEG processing worker approach
    """
    
    def __init__(self, fs: float = 256.0):
        self.fs = fs
        
        # EXACT same configuration as EEG processing worker
        self.EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # TP9, AF7, AF8, TP10
        self.NUM_EEG_CHANNELS = len(self.EEG_CHANNEL_INDICES)
        
        # EXACT same timing parameters
        self.CALIBRATION_DURATION_SECONDS = 20.0
        self.ANALYSIS_WINDOW_SECONDS = 1.0
        self.PSD_WINDOW_SECONDS = 6.0
        
        # EXACT same frequency bands
        self.THETA_BAND = (4.0, 8.0)
        self.ALPHA_BAND = (8.0, 13.0)
        self.BETA_BAND = (13.0, 30.0)
        
        # EXACT same PSD parameters
        self.nfft = DataFilter.get_nearest_power_of_two(int(self.fs * self.PSD_WINDOW_SECONDS)) if BRAINFLOW_AVAILABLE else int(self.fs * self.PSD_WINDOW_SECONDS)
        self.welch_overlap_samples = self.nfft // 2
        
        # EXACT same channel weights and thresholds
        self.channel_weights = np.ones(self.NUM_EEG_CHANNELS) / self.NUM_EEG_CHANNELS
        self.channel_thresholds = [150, 100, 100, 150]  # TP9, AF7, AF8, TP10
        self.diff_thresholds = [50, 30, 30, 50]
        
        # EXACT same filtering
        self.setup_filters()
        
        # Classification state (same as worker)
        self.baseline_metrics = None
        self.is_calibrated = False
        
    def setup_filters(self):
        """EXACT same filter setup as EEG processing worker"""
        filter_order = 4
        lowcut = 0.5
        highcut = 30.0
        
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = signal.butter(filter_order, [low, high], btype='band', analog=False)
    
    def load_session_data(self, data_path: str) -> Dict:
        """Load complete session data (not just the 6000:8000 window)"""
        print(f"Loading session data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        sessions = {}
        
        # Look for all files
        all_csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'subject' in f]
        print(f"Found {len(all_csv_files)} CSV files")
        
        for file in all_csv_files:
            file_path = os.path.join(data_path, file)
            print(f"Loading: {file}")
            
            try:
                df = pd.read_csv(file_path)
                print(f"  Raw shape: {df.shape}")
                
                # Extract EEG channels (columns 1-4: TP9, AF7, AF8, TP10)
                eeg_data = df.iloc[:, [1, 2, 3, 4]].values  # Full session data
                eeg_data = eeg_data.T  # Shape: (4 channels, time_samples)
                
                print(f"  EEG shape: {eeg_data.shape}")
                print(f"  Duration: {eeg_data.shape[1] / self.fs:.1f} seconds")
                print(f"  Value range: {eeg_data.min():.2f} to {eeg_data.max():.2f}")
                
                # Parse filename
                parts = file.replace('.csv', '').split('-')
                subject = parts[0].replace('subject', '')
                condition = parts[1]
                trial = parts[2] if len(parts) > 2 else '1'
                
                key = (subject, condition, trial)
                sessions[key] = {
                    'eeg_data': eeg_data,
                    'filename': file,
                    'duration_seconds': eeg_data.shape[1] / self.fs
                }
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        return sessions
    
    def filter_eeg_data(self, eeg_data: np.ndarray) -> np.ndarray:
        """EXACT same filtering as EEG processing worker"""
        min_samples = 3 * 4 + 1
        
        if eeg_data.shape[1] < min_samples:
            return eeg_data
            
        eeg_filtered = np.zeros_like(eeg_data)
        for i in range(self.NUM_EEG_CHANNELS):
            eeg_filtered[i] = signal.filtfilt(self.b, self.a, eeg_data[i])
        return eeg_filtered
    
    def improved_artifact_rejection(self, eeg_data: np.ndarray) -> np.ndarray:
        """EXACT same artifact rejection as EEG processing worker"""
        amplitude_mask = ~np.any(
            np.abs(eeg_data) > np.array(self.channel_thresholds).reshape(-1, 1), 
            axis=0
        )
        
        diff_mask = np.ones(eeg_data.shape[1], dtype=bool)
        if eeg_data.shape[1] > 1:
            diff_mask = ~np.any(
                np.abs(np.diff(eeg_data, axis=1, prepend=eeg_data[:, :1])) > 
                np.array(self.diff_thresholds).reshape(-1, 1), 
                axis=0
            )
        
        return amplitude_mask & diff_mask
    
    def calculate_band_powers(self, eeg_segment: np.ndarray) -> Optional[Dict]:
        """EXACT same band power calculation as EEG processing worker"""
        if eeg_segment.shape[1] < self.nfft:
            return None
            
        # Apply artifact rejection
        artifact_mask = self.improved_artifact_rejection(eeg_segment)
        if np.sum(artifact_mask) < 0.7 * eeg_segment.shape[1]:
            return None
        
        # Calculate band powers for each channel
        metrics_list = []
        for ch_idx in range(self.NUM_EEG_CHANNELS):
            ch_data = eeg_segment[ch_idx, artifact_mask].copy() if np.any(artifact_mask) else eeg_segment[ch_idx].copy()
            
            if len(ch_data) < self.nfft:
                pad_length = self.nfft - len(ch_data)
                ch_data = np.pad(ch_data, (0, pad_length), mode='reflect')
            
            if BRAINFLOW_AVAILABLE:
                try:
                    DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
                    
                    psd = DataFilter.get_psd_welch(
                        ch_data, 
                        self.nfft, 
                        self.welch_overlap_samples, 
                        int(self.fs), 
                        WindowOperations.HANNING.value
                    )
                    
                    metrics_list.append({
                        'theta': DataFilter.get_band_power(psd, self.THETA_BAND[0], self.THETA_BAND[1]),
                        'alpha': DataFilter.get_band_power(psd, self.ALPHA_BAND[0], self.ALPHA_BAND[1]),
                        'beta': DataFilter.get_band_power(psd, self.BETA_BAND[0], self.BETA_BAND[1])
                    })
                except Exception as e:
                    return None
            else:
                # Fallback to scipy
                try:
                    freqs, psd = signal.welch(ch_data, fs=self.fs, nperseg=min(len(ch_data), self.nfft))
                    
                    def get_band_power(freqs, psd, low_freq, high_freq):
                        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        if not np.any(freq_mask):
                            return 0.0
                        return np.trapz(psd[freq_mask], freqs[freq_mask])
                    
                    metrics_list.append({
                        'theta': get_band_power(freqs, psd, *self.THETA_BAND),
                        'alpha': get_band_power(freqs, psd, *self.ALPHA_BAND),
                        'beta': get_band_power(freqs, psd, *self.BETA_BAND)
                    })
                except Exception as e:
                    return None
                    
        if len(metrics_list) != self.NUM_EEG_CHANNELS:
            return None
            
        # Calculate weighted average (EXACT same as worker)
        avg_metrics = {
            'theta': np.sum([m['theta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
            'alpha': np.sum([m['alpha'] * self.channel_weights[i] for i, m in enumerate(metrics_list)]),
            'beta': np.sum([m['beta'] * self.channel_weights[i] for i, m in enumerate(metrics_list)])
        }
        
        avg_metrics['ab_ratio'] = avg_metrics['alpha'] / avg_metrics['beta'] if avg_metrics['beta'] > 1e-9 else 0
        avg_metrics['bt_ratio'] = avg_metrics['beta'] / avg_metrics['theta'] if avg_metrics['theta'] > 1e-9 else 0
        
        return avg_metrics
    
    def calibrate_from_session(self, eeg_data: np.ndarray) -> bool:
        """
        Calibrate baseline from the first 20 seconds of a session
        (exactly like EEG processing worker calibration)
        """
        print("Calibrating baseline from session data...")
        
        # Use first 20 seconds for calibration
        calibration_samples = int(self.CALIBRATION_DURATION_SECONDS * self.fs)
        if eeg_data.shape[1] < calibration_samples:
            print(f"Insufficient data for calibration: {eeg_data.shape[1]} samples < {calibration_samples}")
            return False
        
        calibration_data = eeg_data[:, :calibration_samples]
        
        # Apply filtering
        filtered_data = self.filter_eeg_data(calibration_data)
        
        # Collect metrics from overlapping windows (like the worker does)
        calibration_metrics_list = []
        
        window_samples = self.nfft
        step_samples = int(self.ANALYSIS_WINDOW_SECONDS * self.fs)  # 1-second steps
        
        for start_idx in range(0, filtered_data.shape[1] - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_data = filtered_data[:, start_idx:end_idx]
            
            metrics = self.calculate_band_powers(window_data)
            if metrics:
                calibration_metrics_list.append(metrics)
        
        if len(calibration_metrics_list) == 0:
            print("No valid calibration windows found")
            return False
        
        # Calculate baseline (same as worker)
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
        
        self.is_calibrated = True
        
        print(f"Baseline calibrated from {len(calibration_metrics_list)} windows:")
        for key, value in self.baseline_metrics.items():
            print(f"  {key}: {value:.6f}")
        
        return True
    
    def classify_mental_state_binary(self, current_metrics: Dict, session_type: str) -> Dict:
        """EXACT same classification as EEG processing worker, simplified to binary"""
        
        if not self.baseline_metrics or not current_metrics:
            return {"binary_prediction": "neutral", "confidence": 0.0, "level": 0}
        
        # Calculate ratios relative to baseline (EXACT same as worker)
        alpha_ratio = current_metrics['alpha'] / self.baseline_metrics['alpha'] if self.baseline_metrics['alpha'] > 0 else 1
        beta_ratio = current_metrics['beta'] / self.baseline_metrics['beta'] if self.baseline_metrics['beta'] > 0 else 1
        theta_ratio = current_metrics['theta'] / self.baseline_metrics['theta'] if self.baseline_metrics['theta'] > 0 else 1
        
        # Session-specific classification (EXACT same logic as worker)
        if session_type == "RELAXATION":
            if alpha_ratio > 1.4 and beta_ratio < 0.8:
                raw_level, confidence = 3, min(0.95, 0.5 + (alpha_ratio - 1.4) * 0.3)
            elif alpha_ratio > 1.2:
                raw_level, confidence = 2, min(0.9, 0.5 + (alpha_ratio - 1.2) * 0.5)
            elif alpha_ratio > 1.05:
                raw_level, confidence = 1, min(0.8, 0.4 + (alpha_ratio - 1.05) * 2.0)
            elif alpha_ratio < 0.8 and beta_ratio > 1.2:
                raw_level, confidence = -2, min(0.9, 0.5 + (1.2 - alpha_ratio) * 0.5)
            elif alpha_ratio < 0.9:
                raw_level, confidence = -1, min(0.8, 0.4 + (0.9 - alpha_ratio) * 2.0)
            else:
                raw_level, confidence = 0, 0.6
            
            binary_prediction = "relaxed" if raw_level > 0 else "not_relaxed"
            
        elif session_type == "FOCUS":
            bt_ratio = current_metrics['bt_ratio'] / self.baseline_metrics['bt_ratio'] if self.baseline_metrics['bt_ratio'] > 0 else 1
            
            if bt_ratio > 1.4 and beta_ratio > 1.2:
                raw_level, confidence = 3, min(0.95, 0.5 + (bt_ratio - 1.4) * 0.3)
            elif bt_ratio > 1.2:
                raw_level, confidence = 2, min(0.9, 0.5 + (bt_ratio - 1.2) * 0.5)
            elif bt_ratio > 1.05:
                raw_level, confidence = 1, min(0.8, 0.4 + (bt_ratio - 1.05) * 2.0)
            elif bt_ratio < 0.8 or theta_ratio > 1.3:
                raw_level, confidence = -2, min(0.9, 0.5 + (0.8 - bt_ratio) * 0.5)
            elif bt_ratio < 0.9:
                raw_level, confidence = -1, min(0.8, 0.4 + (0.9 - bt_ratio) * 2.0)
            else:
                raw_level, confidence = 0, 0.6
            
            binary_prediction = "focused" if raw_level > 0 else "not_focused"
            
        else:
            binary_prediction = "neutral"
            confidence = 0.0
            raw_level = 0
        
        return {
            "binary_prediction": binary_prediction,
            "confidence": confidence,
            "level": raw_level,
            "ratios": {
                "alpha_ratio": alpha_ratio,
                "beta_ratio": beta_ratio,
                "theta_ratio": theta_ratio
            }
        }
    
    def process_session_windowed(self, eeg_data: np.ndarray, session_type: str, 
                                start_offset_seconds: float = 20.0) -> Dict:
        """
        Process session with windowed approach (exactly like EEG processing worker)
        
        Args:
            eeg_data: Full session EEG data (4 channels x samples)
            session_type: "RELAXATION" or "FOCUS"
            start_offset_seconds: Start processing after this many seconds (skip calibration period)
        """
        
        if not self.is_calibrated:
            print("Error: Must calibrate baseline first")
            return {}
        
        # Apply filtering to full session
        filtered_data = self.filter_eeg_data(eeg_data)
        
        # Calculate processing parameters
        window_samples = self.nfft  # 6 seconds of data for each prediction
        step_samples = int(self.ANALYSIS_WINDOW_SECONDS * self.fs)  # 1-second steps
        start_sample = int(start_offset_seconds * self.fs)  # Skip calibration period
        
        predictions = []
        confidences = []
        levels = []
        timestamps = []
        window_metrics = []
        
        print(f"Processing session: {filtered_data.shape[1]} samples, {filtered_data.shape[1]/self.fs:.1f} seconds")
        print(f"Starting at {start_offset_seconds}s, window={window_samples} samples, step={step_samples} samples")
        
        # Process in 1-second steps (like real-time processing)
        for start_idx in range(start_sample, filtered_data.shape[1] - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_data = filtered_data[:, start_idx:end_idx]
            timestamp = start_idx / self.fs
            
            # Calculate band powers for this window
            metrics = self.calculate_band_powers(window_data)
            
            if metrics:
                # Make classification
                classification = self.classify_mental_state_binary(metrics, session_type)
                
                predictions.append(classification['binary_prediction'])
                confidences.append(classification['confidence'])
                levels.append(classification['level'])
                timestamps.append(timestamp)
                window_metrics.append(metrics)
            else:
                # Handle artifact windows
                predictions.append("artifact")
                confidences.append(0.0)
                levels.append(0)
                timestamps.append(timestamp)
                window_metrics.append(None)
        
        # Calculate session summary
        valid_predictions = [p for p in predictions if p != "artifact"]
        valid_levels = [l for l, p in zip(levels, predictions) if p != "artifact"]
        
        if len(valid_predictions) == 0:
            session_result = "unknown"
            session_confidence = 0.0
            percent_on_target = 0.0
        else:
            # Determine session result based on majority of predictions
            if session_type == "RELAXATION":
                on_target_count = sum(1 for p in valid_predictions if p == "relaxed")
                session_result = "relaxed" if on_target_count > len(valid_predictions) / 2 else "not_relaxed"
            elif session_type == "FOCUS":
                on_target_count = sum(1 for p in valid_predictions if p == "focused")
                session_result = "focused" if on_target_count > len(valid_predictions) / 2 else "not_focused"
            else:
                on_target_count = 0
                session_result = "unknown"
            
            percent_on_target = (on_target_count / len(valid_predictions)) * 100
            session_confidence = np.mean([c for c, p in zip(confidences, predictions) if p != "artifact"])
        
        return {
            'session_result': session_result,
            'session_confidence': session_confidence,
            'percent_on_target': percent_on_target,
            'total_predictions': len(predictions),
            'valid_predictions': len(valid_predictions),
            'artifact_windows': len(predictions) - len(valid_predictions),
            'predictions': predictions,
            'confidences': confidences,
            'levels': levels,
            'timestamps': timestamps,
            'window_metrics': window_metrics
        }
    
    def evaluate_windowed_approach(self, sessions: Dict, baseline_strategy: str = "neutral") -> Dict:
        """
        Evaluate the windowed approach across all sessions
        """
        
        print(f"\nEvaluating windowed approach with {baseline_strategy} baseline strategy...")
        
        results = []
        
        # Group sessions by subject and condition
        subjects = set([key[0] for key in sessions.keys()])
        conditions = set([key[1] for key in sessions.keys()])
        
        print(f"Subjects: {subjects}")
        print(f"Conditions: {conditions}")
        
        # Determine session types available
        if 'relaxed' in conditions and 'neutral' in conditions:
            session_type = "RELAXATION"
            print("Using RELAXATION session type")
        elif 'concentrating' in conditions and 'neutral' in conditions:
            session_type = "FOCUS"  
            print("Using FOCUS session type")
        else:
            print("Cannot determine appropriate session types for binary classification")
            return {}
        
        # Process each subject
        for subject in subjects:
            print(f"\nProcessing subject {subject}...")
            
            # Get neutral session for this subject (for baseline calibration)
            neutral_sessions = [(key, data) for key, data in sessions.items() 
                              if key[0] == subject and key[1] == 'neutral']
            
            if not neutral_sessions:
                print(f"  No neutral session found for subject {subject}")
                continue
            
            # Use first neutral session for calibration
            neutral_key, neutral_session = neutral_sessions[0]
            neutral_data = neutral_session['eeg_data']
            
            if not self.calibrate_from_session(neutral_data):
                print(f"  Calibration failed for subject {subject}")
                continue
            
            # Process all sessions for this subject
            subject_sessions = [(key, data) for key, data in sessions.items() if key[0] == subject]
            
            for key, session_data in subject_sessions:
                subject, condition, trial = key
                eeg_data = session_data['eeg_data']
                
                print(f"  Processing {condition} trial {trial}...")
                
                # Determine true label for binary classification
                if session_type == "RELAXATION":
                    true_label = "relaxed" if condition == "relaxed" else "not_relaxed"
                elif session_type == "FOCUS":
                    true_label = "focused" if condition == "concentrating" else "not_focused"
                else:
                    true_label = "unknown"
                
                # Process session with windowed approach
                session_result = self.process_session_windowed(eeg_data, session_type)
                
                if session_result:
                    result = {
                        'subject': subject,
                        'condition': condition,
                        'trial': trial,
                        'true_label': true_label,
                        'predicted_label': session_result['session_result'],
                        'session_confidence': session_result['session_confidence'],
                        'percent_on_target': session_result['percent_on_target'],
                        'total_predictions': session_result['total_predictions'],
                        'valid_predictions': session_result['valid_predictions'],
                        'artifact_windows': session_result['artifact_windows']
                    }
                    
                    results.append(result)
                    
                    print(f"    True: {true_label}, Predicted: {session_result['session_result']}")
                    print(f"    Confidence: {session_result['session_confidence']:.3f}")
                    print(f"    Valid predictions: {session_result['valid_predictions']}/{session_result['total_predictions']}")
                    print(f"    Percent on target: {session_result['percent_on_target']:.1f}%")
        
        # Calculate overall performance
        if results:
            results_df = pd.DataFrame(results)
            
            # Calculate accuracy
            accuracy = accuracy_score(results_df['true_label'], results_df['predicted_label'])
            
            # Calculate other metrics
            avg_confidence = results_df['session_confidence'].mean()
            avg_on_target = results_df['percent_on_target'].mean()
            avg_valid_predictions = results_df['valid_predictions'].mean()
            total_artifact_windows = results_df['artifact_windows'].sum()
            
            # Generate classification report
            class_report = classification_report(
                results_df['true_label'], 
                results_df['predicted_label'], 
                zero_division=0
            )
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(
                results_df['true_label'], 
                results_df['predicted_label']
            )
            
            evaluation_results = {
                'session_type': session_type,
                'baseline_strategy': baseline_strategy,
                'overall_accuracy': accuracy,
                'average_confidence': avg_confidence,
                'average_percent_on_target': avg_on_target,
                'average_valid_predictions': avg_valid_predictions,
                'total_artifact_windows': total_artifact_windows,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'detailed_results': results_df
            }
            
            return evaluation_results
        else:
            print("No valid results obtained")
            return {}
    
    def print_evaluation_results(self, results: Dict):
        """Print comprehensive evaluation results"""
        
        print("\n" + "="*80)
        print("WINDOWED EEG CLASSIFIER EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nSession Type: {results['session_type']}")
        print(f"Baseline Strategy: {results['baseline_strategy']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Average Confidence: {results['average_confidence']:.3f}")
        print(f"Average Percent On Target: {results['average_percent_on_target']:.1f}%")
        print(f"Average Valid Predictions per Session: {results['average_valid_predictions']:.1f}")
        print(f"Total Artifact Windows: {results['total_artifact_windows']}")
        
        print(f"\nClassification Report:")
        print(results['classification_report'])
        
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        # Per-subject analysis
        detailed_df = results['detailed_results']
        
        print(f"\nPer-Subject Performance:")
        for subject in sorted(detailed_df['subject'].unique()):
            subject_data = detailed_df[detailed_df['subject'] == subject]
            subject_accuracy = accuracy_score(subject_data['true_label'], subject_data['predicted_label'])
            avg_confidence = subject_data['session_confidence'].mean()
            avg_on_target = subject_data['percent_on_target'].mean()
            
            print(f"  Subject {subject.upper()}:")
            print(f"    Accuracy: {subject_accuracy:.3f}")
            print(f"    Avg Confidence: {avg_confidence:.3f}")
            print(f"    Avg On Target: {avg_on_target:.1f}%")
        
        # Per-condition analysis
        print(f"\nPer-Condition Performance:")
        for condition in sorted(detailed_df['condition'].unique()):
            condition_data = detailed_df[detailed_df['condition'] == condition]
            condition_accuracy = accuracy_score(condition_data['true_label'], condition_data['predicted_label'])
            avg_confidence = condition_data['session_confidence'].mean()
            avg_on_target = condition_data['percent_on_target'].mean()
            
            print(f"  {condition.capitalize()}:")
            print(f"    Accuracy: {condition_accuracy:.3f}")
            print(f"    Avg Confidence: {avg_confidence:.3f}")
            print(f"    Avg On Target: {avg_on_target:.1f}%")


def main():
    """Main evaluation function"""
    
    print("Windowed EEG Classifier Evaluation")
    print("Exactly matching EEG Processing Worker approach")
    print("=" * 60)
    
    # Initialize classifier
    classifier = WindowedEEGClassifier()
    
    # Set data path
    data_path = r"C:\Users\berna\OneDrive\Documentos\4A 2S\Neuro\Projeto Antigo\Dataset"
    
    print(f"Data path: {data_path}")
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        return
    
    try:
        # Load full session data (not just 6000:8000 window)
        sessions = classifier.load_session_data(data_path)
        
        if not sessions:
            print("No sessions loaded successfully")
            return
        
        print(f"\nLoaded {len(sessions)} sessions")
        
        # Show session summary
        subjects = set([key[0] for key in sessions.keys()])
        conditions = set([key[1] for key in sessions.keys()])
        
        print(f"Subjects: {sorted(subjects)}")
        print(f"Conditions: {sorted(conditions)}")
        
        for subject in sorted(subjects):
            subject_sessions = [key for key in sessions.keys() if key[0] == subject]
            print(f"Subject {subject}: {len(subject_sessions)} sessions")
        
        # Evaluate windowed approach
        print(f"\n" + "="*60)
        print("STARTING WINDOWED EVALUATION")
        print("="*60)
        
        results = classifier.evaluate_windowed_approach(sessions, baseline_strategy="neutral")
        
        if results:
            classifier.print_evaluation_results(results)
            
            # Save detailed results
            results['detailed_results'].to_csv('windowed_classifier_results.csv', index=False)
            print(f"\nDetailed results saved to windowed_classifier_results.csv")
            
            # Save summary
            with open('windowed_classifier_summary.txt', 'w') as f:
                f.write("Windowed EEG Classifier Evaluation Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Session Type: {results['session_type']}\n")
                f.write(f"Overall Accuracy: {results['overall_accuracy']:.3f}\n")
                f.write(f"Average Confidence: {results['average_confidence']:.3f}\n")
                f.write(f"Average Percent On Target: {results['average_percent_on_target']:.1f}%\n")
                f.write(f"\nClassification Report:\n")
                f.write(results['classification_report'])
            
            print(f"Summary saved to windowed_classifier_summary.txt")
        else:
            print("Evaluation failed - no results obtained")
        
        print(f"\nEvaluation completed!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()