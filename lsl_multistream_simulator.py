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
    "Accelerometer": {"key": "accelerometer_data", "rate": 52, "type": "Accelerometer"},
    "Gyroscope": {"key": "gyroscope_data", "rate": 52, "type": "Gyroscope"},
    "PPG": {"key": "ppg_data", "rate": 64, "type": "PPG"},
}

def create_outlet(name, stype, n_channels, srate):
    info = pylsl.StreamInfo(name, stype, n_channels, srate, 'float32', name + "_sim")
    return pylsl.StreamOutlet(info)

def push_stream(outlet, data, rate, stop_flag):
    n_channels, n_samples = data.shape
    chunk_size = max(1, int(rate / 16))
    while not stop_flag.is_set():
        for i in range(0, n_samples, chunk_size):
            chunk = data[:, i:i+chunk_size].T
            outlet.push_chunk(chunk.tolist())
            time.sleep(chunk.shape[0] / rate)
        # Loop
        print(f"Stream {outlet.name()} looping...")
        
def main(npz_file):
    npz = np.load(npz_file, allow_pickle=True)
    threads = []
    stop_flag = threading.Event()
    for stream_name, conf in STREAMS.items():
        if conf["key"] in npz:
            arr = npz[conf["key"]]
            if arr.size == 0:
                continue
            if arr.shape[0] > arr.shape[1]:
                arr = arr[:,:]  # [channels, samples]
            print(f"{stream_name}: {arr.shape}, {conf['rate']} Hz")
            outlet = create_outlet("Muse-" + stream_name, conf["type"], arr.shape[0], conf["rate"])
            t = threading.Thread(target=push_stream, args=(outlet, arr, conf["rate"], stop_flag))
            t.start()
            threads.append(t)
    
    print("Press Ctrl+C to stop simulator")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping all streams...")
        stop_flag.set()
        for t in threads:
            t.join()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .npz session file")
    args = parser.parse_args()
    main(args.file)