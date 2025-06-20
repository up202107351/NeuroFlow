#!/usr/bin/env python3
"""
LSL EEG Simulator with Automatic Repeat

This script loads EEG data from a .npy or .csv file and continuously streams it through LSL,
automatically repeating the signal when it reaches the end.

Usage:
    python lsl_eeg_simulator.py --file live_session_data/eeg_data_raw.npy [--speed 1.0]
    python lsl_eeg_simulator.py --file dataset/subjectb-concentrating-1.csv
"""

import numpy as np
import pandas as pd
import pylsl
import time
import argparse
import threading
import os
import sys
from datetime import datetime

class LSLEEGSimulator:
    def __init__(self, data_file, speed_factor=1.0, device_name="MuseSimulator"):
        """
        Initialize the LSL EEG Simulator
        
        Args:
            data_file: Path to the .npy or .csv file containing EEG data
            speed_factor: Playback speed (1.0 is real-time)
            device_name: Name of the simulated device
        """
        self.data_file = data_file
        self.speed_factor = speed_factor
        self.device_name = device_name
        self.running = False
        self.thread = None
        
        # Muse configuration (will be updated based on data)
        self.n_channels = None
        self.sample_rate = 256.0  # Default Muse sampling rate
        self.data = None
        self.timestamps = None
        self.channel_names = []
        self.outlet = None
        
    def load_data(self):
        """Load EEG data from the .npy or .csv file"""
        try:
            print(f"Loading EEG data from: {self.data_file}")
            
            # Check file extension to determine loading method
            file_ext = os.path.splitext(self.data_file)[1].lower()
            
            if file_ext == '.npy':
                return self._load_npy_data()
            elif file_ext == '.csv':
                return self._load_csv_data()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Use .npy or .csv files.")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _load_npy_data(self):
        """Load data from a .npy file"""
        data = np.load(self.data_file)
        
        # Check data shape and format
        if len(data.shape) == 2:
            # If data is in shape [channels, samples]
            if data.shape[0] <= 64:  # Assuming channels < samples
                self.data = data
                self.n_channels = data.shape[0]
            # If data is in shape [samples, channels]
            else:
                self.data = data.T
                self.n_channels = data.shape[1]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}. Expected 2D array.")
        
        print(f"Loaded NPY data with {self.n_channels} channels and {self.data.shape[1]} samples")
        
        # Generate timestamps if not available (evenly spaced at sample_rate)
        self.timestamps = np.linspace(0, self.data.shape[1]/self.sample_rate, self.data.shape[1])
        
        # Create generic channel names
        self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]
        
        # Estimate sample rate from data length if 120 seconds (2 minutes)
        est_sample_rate = self.data.shape[1] / 120
        if abs(est_sample_rate - 256.0) < 10.0:  # Close to Muse's 256 Hz
            self.sample_rate = 256.0
        elif abs(est_sample_rate - 500.0) < 10.0:  # Some devices use 500 Hz
            self.sample_rate = 500.0
        else:
            self.sample_rate = est_sample_rate
            
        print(f"Using sample rate: {self.sample_rate} Hz")
        return True
    
    def _load_csv_data(self):
        """Load data from a .csv file with timestamps in first column"""
        try:
            # Read CSV file with pandas
            df = pd.read_csv(self.data_file)
            
            # Check if we have a timestamp column
            if 'timestamps' in df.columns:
                timestamp_col = 'timestamps'
            elif 'timestamp' in df.columns:
                timestamp_col = 'timestamp'
            else:
                timestamp_col = df.columns[0]  # Assume first column is timestamps
            
            # Extract timestamps and convert to float if needed
            timestamps = df[timestamp_col].values
            if not np.issubdtype(timestamps.dtype, np.number):
                timestamps = pd.to_numeric(timestamps, errors='coerce').values
            
            # Remove timestamp column from data
            data_df = df.drop(columns=[timestamp_col])
            
            # Get channel names from headers
            self.channel_names = list(data_df.columns)
            
            # Extract EEG data
            self.data = data_df.values.T  # Transpose to [channels, samples] format
            self.n_channels = self.data.shape[0]
            
            # Store timestamps
            self.timestamps = timestamps
            
            # Calculate sample rate from timestamps
            if len(timestamps) > 1:
                # Calculate sample rate from median time difference
                time_diffs = np.diff(timestamps)
                median_diff = np.median(time_diffs)
                if median_diff > 0:
                    self.sample_rate = 1.0 / median_diff
                    print(f"Calculated sample rate from timestamps: {self.sample_rate:.1f} Hz")
            
            print(f"Loaded CSV data with {self.n_channels} channels and {self.data.shape[1]} samples")
            print(f"Channel names: {self.channel_names}")
            
            return True
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_outlet(self):
        """Create an LSL outlet with appropriate metadata"""
        if self.data is None or self.n_channels is None:
            print("Error: Data not loaded")
            return False
        
        stream_info = pylsl.StreamInfo(
            name=self.device_name,
            type='EEG',
            channel_count=self.n_channels,
            nominal_srate=self.sample_rate,
            channel_format='float32',
            source_id='musesim123'
        )
        channels = stream_info.desc().append_child("channels")

        # Use loaded channel names if available, otherwise use defaults
        if not self.channel_names or len(self.channel_names) != self.n_channels:
            # Default Muse channel names
            if self.n_channels == 4:
                self.channel_names = ["TP9", "AF7", "AF8", "TP10"]
            elif self.n_channels == 5:
                self.channel_names = ["TP9", "AF7", "AF8", "TP10", "Right AUX"]
            else:
                self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]

        # Add channel info to LSL stream info
        for i, ch_name in enumerate(self.channel_names):
            unit = "microvolts"
            ch_type = "EEG"
            if "ACC" in ch_name:
                unit = "g"
                ch_type = "Accelerometer"
            channels.append_child("channel") \
                .append_child_value("label", ch_name) \
                .append_child_value("unit", unit) \
                .append_child_value("type", ch_type)

        self.outlet = pylsl.StreamOutlet(stream_info)
        print(f"Created LSL outlet: {self.device_name} (type: EEG, {self.n_channels} channels @ {self.sample_rate:.1f} Hz)")
        return True
    
    def stream_data(self):
        """Stream the data through the LSL outlet with automatic repeating"""
        if self.data is None or self.outlet is None:
            print("Error: Data or outlet not initialized")
            return
        
        sample_interval = 1.0 / self.sample_rate
        chunk_size = max(1, int(self.sample_rate / 32))  # Approx. 32 chunks per second
        
        n_samples = self.data.shape[1]
        sample_idx = 0
        
        print(f"Starting LSL stream with {n_samples} samples (chunk size: {chunk_size})")
        loop_count = 0
        
        # Create a small overlap between repeats to ensure smooth transition
        overlap_samples = int(self.sample_rate * 0.1)  # 100ms overlap
        
        # Prepare the first chunk
        first_chunk = self.data[:, 0:min(chunk_size, n_samples)].T
        last_chunk = self.data[:, max(0, n_samples - overlap_samples):n_samples].T
        
        while self.running:
            # Determine size of current chunk (smaller at the end of the data)
            remaining = n_samples - sample_idx
            current_chunk_size = min(chunk_size, remaining)
            
            if current_chunk_size <= 0:
                # We've reached the end of the data, loop back to beginning
                print(f"End of data reached, repeating from beginning (loop #{loop_count+1})")
                sample_idx = 0
                loop_count += 1
                
                # Create a smooth transition by overlaying end and beginning
                # This helps prevent jarring artifacts at loop points
                if overlap_samples > 0 and n_samples > overlap_samples * 2:
                    # Create a transition chunk that blends the end and start
                    transition_chunk = np.zeros((overlap_samples, self.n_channels))
                    
                    # Linear crossfade weights
                    fade_out = np.linspace(1, 0, overlap_samples).reshape(-1, 1)
                    fade_in = np.linspace(0, 1, overlap_samples).reshape(-1, 1)
                    
                    # Last few samples from previous loop
                    end_chunk = last_chunk[:overlap_samples]
                    
                    # First few samples from new loop
                    start_chunk = first_chunk[:overlap_samples]
                    
                    # Crossfade between end and start
                    if len(end_chunk) == overlap_samples and len(start_chunk) == overlap_samples:
                        transition_chunk = end_chunk * fade_out + start_chunk * fade_in
                        
                        # Push the transition chunk
                        self.outlet.push_chunk(transition_chunk.tolist())
                        
                        # Skip the first part we've already sent in the transition
                        sample_idx = overlap_samples
                
                # Get a new chunk size for the start of the data
                current_chunk_size = min(chunk_size, n_samples - sample_idx)
            
            # Extract and push the chunk
            chunk = self.data[:, sample_idx:sample_idx+current_chunk_size].T
            self.outlet.push_chunk(chunk.tolist())
            
            # Update sample index
            sample_idx += current_chunk_size
            
            # Sleep for the appropriate interval (accounting for speed factor)
            sleep_time = (current_chunk_size * sample_interval) / self.speed_factor
            time.sleep(sleep_time)
    
    def start(self):
        """Start the LSL streaming in a background thread"""
        if self.running:
            print("Simulator already running")
            return False
        
        if not self.load_data():
            return False
        
        if not self.create_outlet():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self.stream_data)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"Simulator started at {datetime.now().strftime('%H:%M:%S')} - Signal will automatically repeat")
        return True
    
    def stop(self):
        """Stop the LSL streaming"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        print("Simulator stopped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stream EEG data from a .npy or .csv file via LSL with automatic repeating')
    parser.add_argument('--file', type=str, required=True, help='Path to the .npy or .csv file')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (1.0 is real-time)')
    parser.add_argument('--name', type=str, default='MuseSimulator', help='Device name')
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1
    
    simulator = LSLEEGSimulator(
        data_file=args.file,
        speed_factor=args.speed,
        device_name=args.name
    )
    
    try:
        if simulator.start():
            print("Press Ctrl+C to stop the simulator")
            while simulator.running:
                time.sleep(0.1)
        else:
            print("Failed to start simulator")
            return 1
    except KeyboardInterrupt:
        print("\nStopping simulator...")
    finally:
        simulator.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())