#!/usr/bin/env python3
"""
Signal Quality Validator for Muse EEG Data

This module validates EEG signal quality using accelerometer data and band power analysis
to ensure optimal headband placement and signal integrity during calibration.
"""

import numpy as np
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

logger = logging.getLogger(__name__)

@dataclass
class SignalQualityMetrics:
    """Container for signal quality metrics"""
    movement_score: float  # 0-1, higher is better (less movement)
    band_power_score: float  # 0-1, higher is better
    electrode_contact_score: float  # 0-1, higher is better
    overall_score: float  # 0-1, overall quality
    quality_level: str  # "excellent", "good", "fair", "poor"
    recommendations: List[str]  # List of user recommendations

class SignalQualityValidator:
    """Validates EEG signal quality using multiple metrics"""
    
    def __init__(self, 
                 movement_threshold: float = 0.5,
                 history_size: int = 50,
                 update_interval: float = 1.0):
        """
        Initialize the signal quality validator
        
        Args:
            movement_threshold: Maximum acceptable accelerometer variance
            history_size: Number of samples to keep in history
            update_interval: Time between quality assessments (seconds)
        """
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        self.update_interval = update_interval
        
        # Band power ranges (μV²) - typical values for different states
        self.band_power_ranges = {
            'alpha': (1.0, 25.0),    # 8-12 Hz - relaxation/attention
            'beta': (0.5, 20.0),     # 13-30 Hz - active thinking
            'theta': (2.0, 30.0),    # 4-8 Hz - drowsiness/creativity
            'gamma': (0.1, 10.0),    # 30+ Hz - high-level cognition
        }
        
        # Signal artifact detection thresholds
        self.artifact_thresholds = {
            'max_power': 100.0,      # Maximum reasonable band power
            'min_power': 0.01,       # Minimum power (disconnected electrode)
            'power_variation': 5.0,   # Maximum acceptable power variation
        }
        
        # History buffers
        self.accelerometer_history = deque(maxlen=history_size)
        self.band_power_history = deque(maxlen=history_size)
        self.eeg_raw_history = deque(maxlen=history_size)
        
        # Quality tracking
        self.last_assessment_time = 0
        self.quality_history = deque(maxlen=20)  # Store last 20 assessments
        
        # State tracking
        self.is_calibrating = False
        self.calibration_start_time = None
        self.stable_signal_duration = 0
        self.required_stable_duration = 10.0  # seconds of good signal needed
        
    def add_accelerometer_data(self, acc_data: np.ndarray) -> None:
        """Add accelerometer data sample (x, y, z accelerations)"""
        if acc_data.shape[-1] == 3:  # Ensure we have x, y, z
            self.accelerometer_history.append({
                'timestamp': time.time(),
                'data': acc_data.copy(),
                'magnitude': np.linalg.norm(acc_data)
            })
    
    def add_band_power_data(self, band_powers: Dict[str, float]) -> None:
        """Add band power data sample"""
        self.band_power_history.append({
            'timestamp': time.time(),
            'powers': band_powers.copy()
        })
    
    def add_raw_eeg_data(self, eeg_data: np.ndarray) -> None:
        """Add raw EEG data for additional analysis"""
        self.eeg_raw_history.append({
            'timestamp': time.time(),
            'data': eeg_data.copy()
        })
    
    def calculate_movement_score(self) -> Tuple[float, List[str]]:
        """Calculate movement quality score from accelerometer data"""
        if len(self.accelerometer_history) < 5:
            return 0.5, ["Collecting movement data..."]
        
        recommendations = []
        
        # Get recent accelerometer data
        recent_data = list(self.accelerometer_history)[-10:]  # Last 10 samples
        
        # Calculate movement variance for each axis
        x_data = [sample['data'][0] for sample in recent_data]
        y_data = [sample['data'][1] for sample in recent_data]
        z_data = [sample['data'][2] for sample in recent_data]
        
        x_variance = np.var(x_data)
        y_variance = np.var(y_data)
        z_variance = np.var(z_data)
        
        total_variance = x_variance + y_variance + z_variance
        
        # Calculate movement score (inverse of variance, clamped to 0-1)
        movement_score = max(0.0, min(1.0, 1.0 - (total_variance / self.movement_threshold)))
        
        # Generate recommendations based on movement level
        if total_variance > self.movement_threshold * 2:
            recommendations.append("Please sit still - too much head movement detected")
        elif total_variance > self.movement_threshold:
            recommendations.append("Try to minimize head movement")
        elif movement_score > 0.8:
            recommendations.append("Good - minimal movement detected")
        
        return movement_score, recommendations
    
    def calculate_band_power_score(self) -> Tuple[float, List[str]]:
        """Calculate band power quality score"""
        if len(self.band_power_history) < 3:
            return 0.5, ["Analyzing EEG signal quality..."]
        
        recommendations = []
        recent_powers = list(self.band_power_history)[-5:]  # Last 5 samples
        
        # Calculate average band powers
        avg_powers = {}
        for band in self.band_power_ranges.keys():
            powers = [sample['powers'].get(band, 0) for sample in recent_powers if band in sample['powers']]
            avg_powers[band] = np.mean(powers) if powers else 0
        
        # Score each band
        band_scores = {}
        for band, (min_val, max_val) in self.band_power_ranges.items():
            power = avg_powers.get(band, 0)
            
            if power < self.artifact_thresholds['min_power']:
                band_scores[band] = 0.0
                recommendations.append(f"Poor {band} signal - check electrode contact")
            elif power > self.artifact_thresholds['max_power']:
                band_scores[band] = 0.2
                recommendations.append(f"Excessive {band} power - possible artifact")
            elif min_val <= power <= max_val:
                # Power is in good range
                band_scores[band] = 1.0
            else:
                # Power is outside normal range but not extreme
                band_scores[band] = 0.6
                if power < min_val:
                    recommendations.append(f"Low {band} power - adjust headband")
                else:
                    recommendations.append(f"High {band} power - relax and sit still")
        
        # Calculate overall band power score
        if band_scores:
            overall_score = np.mean(list(band_scores.values()))
        else:
            overall_score = 0.0
            recommendations.append("No valid EEG data received")
        
        return overall_score, recommendations
    
    def calculate_electrode_contact_score(self) -> Tuple[float, List[str]]:
        """Calculate electrode contact quality based on signal characteristics"""
        if len(self.band_power_history) < 5:
            return 0.5, ["Checking electrode contact..."]
        
        recommendations = []
        recent_powers = list(self.band_power_history)[-10:]
        
        # Check for signal dropouts (sudden power drops)
        dropout_count = 0
        power_variations = []
        
        for i in range(1, len(recent_powers)):
            prev_sample = recent_powers[i-1]['powers']
            curr_sample = recent_powers[i]['powers']
            
            for band in ['alpha', 'beta', 'theta']:
                if band in prev_sample and band in curr_sample:
                    prev_power = prev_sample[band]
                    curr_power = curr_sample[band]
                    
                    if prev_power > 0.1:  # Avoid division by very small numbers
                        variation = abs(curr_power - prev_power) / prev_power
                        power_variations.append(variation)
                        
                        if variation > self.artifact_thresholds['power_variation']:
                            dropout_count += 1
        
        # Calculate contact score
        if power_variations:
            avg_variation = np.mean(power_variations)
            contact_score = max(0.0, min(1.0, 1.0 - (avg_variation / 2.0)))
        else:
            contact_score = 0.0
        
        # Generate recommendations
        if dropout_count > 3:
            recommendations.append("Poor electrode contact - adjust headband position")
            contact_score *= 0.5
        elif dropout_count > 1:
            recommendations.append("Unstable signal - check headband tightness")
        elif contact_score > 0.8:
            recommendations.append("Good electrode contact")
        
        return contact_score, recommendations
    
    def assess_overall_quality(self) -> SignalQualityMetrics:
        """Perform comprehensive signal quality assessment"""
        current_time = time.time()
        
        # Skip if not enough time has passed
        if current_time - self.last_assessment_time < self.update_interval:
            if self.quality_history:
                return self.quality_history[-1]
            else:
                return SignalQualityMetrics(0.5, 0.5, 0.5, 0.5, "unknown", ["Initializing..."])
        
        self.last_assessment_time = current_time
        
        # Calculate individual scores
        movement_score, movement_recs = self.calculate_movement_score()
        band_power_score, band_power_recs = self.calculate_band_power_score()
        contact_score, contact_recs = self.calculate_electrode_contact_score()
        
        # Calculate weighted overall score
        weights = {
            'movement': 0.3,
            'band_power': 0.4,
            'contact': 0.3
        }
        
        overall_score = (
            weights['movement'] * movement_score +
            weights['band_power'] * band_power_score +
            weights['contact'] * contact_score
        )
        
        # Determine quality level
        if overall_score >= 0.8:
            quality_level = "excellent"
        elif overall_score >= 0.6:
            quality_level = "good"
        elif overall_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        # Combine recommendations
        all_recommendations = movement_recs + band_power_recs + contact_recs
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        # Create metrics object
        metrics = SignalQualityMetrics(
            movement_score=movement_score,
            band_power_score=band_power_score,
            electrode_contact_score=contact_score,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=unique_recommendations[:3]  # Limit to 3 most important
        )
        
        # Store in history
        self.quality_history.append(metrics)
        
        # Update stable signal tracking
        if overall_score >= 0.6:  # Good enough for calibration
            if self.stable_signal_duration == 0:
                self.stable_signal_start = current_time
            self.stable_signal_duration = current_time - self.stable_signal_start
        else:
            self.stable_signal_duration = 0
        
        return metrics
    
    def is_ready_for_calibration(self) -> bool:
        """Check if signal quality is good enough to start/continue calibration"""
        if not self.quality_history:
            return False
        
        latest_quality = self.quality_history[-1]
        return (latest_quality.overall_score >= 0.6 and 
                self.stable_signal_duration >= 3.0)  # Need 3 seconds of stable signal
    
    def should_pause_calibration(self) -> bool:
        """Check if calibration should be paused due to poor signal quality"""
        if not self.quality_history:
            return True
        
        latest_quality = self.quality_history[-1]
        return latest_quality.overall_score < 0.4
    
    def reset(self):
        """Reset all history and state"""
        self.accelerometer_history.clear()
        self.band_power_history.clear()
        self.eeg_raw_history.clear()
        self.quality_history.clear()
        self.stable_signal_duration = 0
        self.last_assessment_time = 0
        self.is_calibrating = False
        self.calibration_start_time = None
    
    def get_calibration_readiness(self) -> Dict[str, any]:
        """Get detailed calibration readiness information"""
        if not self.quality_history:
            return {
                'ready': False,
                'progress': 0.0,
                'time_remaining': self.required_stable_duration,
                'status': 'Initializing signal quality assessment...'
            }
        
        latest_quality = self.quality_history[-1]
        ready = self.is_ready_for_calibration()
        progress = min(1.0, self.stable_signal_duration / self.required_stable_duration)
        time_remaining = max(0, self.required_stable_duration - self.stable_signal_duration)
        
        if ready:
            status = "Signal quality good - ready for calibration"
        elif latest_quality.overall_score < 0.4:
            status = f"Poor signal quality - {latest_quality.recommendations[0] if latest_quality.recommendations else 'adjust headband'}"
        else:
            status = f"Signal stabilizing... {time_remaining:.1f}s remaining"
        
        return {
            'ready': ready,
            'progress': progress,
            'time_remaining': time_remaining,
            'status': status,
            'quality_metrics': latest_quality
        }