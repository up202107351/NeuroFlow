#!/usr/bin/env python3
"""
Simulate all Muse streams from a saved .npz session file.
Usage: python lsl_multistream_simulator.py --file muse_test_data/RELAXATION_20250613_110840.npz
"""

import numpy as np
import pylsl
import time
import argparse
import threading

# Stream config
STREAMS = {
    "EEG": {"key": "eeg_raw", "rate": 256, "type": "EEG"},
    "Accelerometer": {"key": "accelerometer_data", "rate": 52, "type": "Accelerometer", "fallback_channels": 3},
    "Gyroscope": {"key": "gyroscope_data", "rate": 52, "type": "Gyroscope", "fallback_channels": 3},
    "PPG": {"key": "ppg_data", "rate": 64, "type": "PPG", "fallback_channels": 1},
}

def create_outlet(name, stype, n_channels, srate):
    info = pylsl.StreamInfo(name, stype, n_channels, srate, 'float32', name + "_sim")
    return pylsl.StreamOutlet(info)

def generate_synthetic_data(stream_name, n_channels, n_samples, rate):
    """Generate synthetic sensor data when real data is not available"""
    if stream_name == "Accelerometer":
        # Generate realistic accelerometer data (slight variations around gravity)
        base_accel = np.array([0.0, 0.0, 9.81])  # Gravity in Z
        data = np.random.normal(0, 0.1, (n_channels, n_samples)) + base_accel.reshape(-1, 1)
        # Add some head movement simulation
        t = np.linspace(0, n_samples/rate, n_samples)
        data[0] += 0.2 * np.sin(2 * np.pi * 0.1 * t)  # Slow head nod
        data[1] += 0.1 * np.sin(2 * np.pi * 0.05 * t)  # Very slow side movement
        return data
    elif stream_name == "Gyroscope":
        # Generate realistic gyroscope data (small rotational movements)
        data = np.random.normal(0, 0.02, (n_channels, n_samples))
        t = np.linspace(0, n_samples/rate, n_samples)
        data[0] += 0.01 * np.sin(2 * np.pi * 0.08 * t)  # Small pitch rotation
        return data
    elif stream_name == "PPG":
        # Generate realistic PPG data (heart rate ~70 bpm)
        t = np.linspace(0, n_samples/rate, n_samples)
        heart_rate = 70 / 60  # beats per second
        ppg_signal = np.sin(2 * np.pi * heart_rate * t) + 0.1 * np.random.normal(0, 1, n_samples)
        return ppg_signal.reshape(n_channels, -1)
    else:
        # Generic synthetic data
        return np.random.normal(0, 1, (n_channels, n_samples))

def push_stream(outlet, data, rate, stop_flag, stream_name):
    """Push data to LSL stream with proper error handling"""
    n_channels, n_samples = data.shape
    chunk_size = max(1, int(rate / 16))
    
    print(f"Starting stream '{stream_name}' with {n_channels} channels, {n_samples} samples at {rate} Hz")
    
    while not stop_flag.is_set():
        for i in range(0, n_samples, chunk_size):
            if stop_flag.is_set():
                break
                
            chunk = data[:, i:i+chunk_size].T
            try:
                outlet.push_chunk(chunk.tolist())
                time.sleep(chunk.shape[0] / rate)
            except Exception as e:
                print(f"Error pushing data for stream '{stream_name}': {e}")
                break
                
        # Loop the data
        if not stop_flag.is_set():
            print(f"Stream '{stream_name}' looping...")
        
def main(npz_file):
    try:
        npz = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file {npz_file}: {e}")
        return
        
    print(f"Available keys in NPZ file: {list(npz.keys())}")
    
    threads = []
    stop_flag = threading.Event()
    active_streams = []
    
    # Get EEG data properties for reference (assuming EEG is always present)
    eeg_data = None
    if "eeg_raw" in npz:
        eeg_data = npz["eeg_raw"]
        if len(eeg_data.shape) == 1:
            eeg_data = eeg_data.reshape(1, -1)
        elif eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T
    
    for stream_name, conf in STREAMS.items():
        arr = None
        data_source = "recorded"
        
        if conf["key"] in npz:
            arr = npz[conf["key"]]
            if arr.size == 0:
                print(f"Warning: {stream_name} data is empty, generating synthetic data")
                arr = None
        
        # If no data available, generate synthetic data
        if arr is None or arr.size == 0:
            if eeg_data is not None:
                # Base synthetic data duration on EEG data
                duration_seconds = eeg_data.shape[1] / 256  # Assume EEG is 256 Hz
                n_samples = int(duration_seconds * conf["rate"])
                n_channels = conf.get("fallback_channels", 1)
                arr = generate_synthetic_data(stream_name, n_channels, n_samples, conf["rate"])
                data_source = "synthetic"
                print(f"Generated synthetic {stream_name} data: {arr.shape}")
            else:
                print(f"Skipping {stream_name}: no EEG reference data available")
                continue
                
        # Ensure proper shape [channels, samples]
        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
        elif arr.shape[0] > arr.shape[1]:
            print(f"Transposing {stream_name} data from {arr.shape} to {arr.T.shape}")
            arr = arr.T
            
        print(f"{stream_name}: {arr.shape}, {conf['rate']} Hz ({data_source} data)")
        
        try:
            outlet = create_outlet("Muse-" + stream_name, conf["type"], arr.shape[0], conf["rate"])
            t = threading.Thread(
                target=push_stream, 
                args=(outlet, arr, conf["rate"], stop_flag, stream_name)
            )
            t.daemon = True
            t.start()
            threads.append(t)
            active_streams.append(f"{stream_name} ({data_source})")
        except Exception as e:
            print(f"Error creating outlet for {stream_name}: {e}")
    
    if not active_streams:
        print("No streams could be started. Check your data file.")
        return
        
    print(f"Started {len(active_streams)} streams: {', '.join(active_streams)}")
    print("Press Ctrl+C to stop simulator")
    
    try:
        while True:
            time.sleep(1)
            alive_threads = [t for t in threads if t.is_alive()]
            if len(alive_threads) < len(threads):
                print(f"Warning: Some streams may have stopped. {len(alive_threads)}/{len(threads)} still running.")
    except KeyboardInterrupt:
        print("\nStopping all streams...")
        stop_flag.set()
        
        for t in threads:
            t.join(timeout=2.0)
            if t.is_alive():
                print(f"Warning: Thread did not stop gracefully")
                
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .npz session file")
    parser.add_argument("--synthetic", action="store_true", 
                        help="Generate synthetic data for missing streams")
    args = parser.parse_args()
    main(args.file)