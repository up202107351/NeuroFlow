#!/usr/bin/env python3
"""
Script to add realistic accelerometer data to existing EEG recordings
with the correct channel layout expected by the NeuroFlow app.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def generate_realistic_accelerometer_data(num_samples, sampling_rate=256, high_movement_duration=30):
    """
    Generate realistic accelerometer data that starts with high movement
    (to trigger calibration pause) then settles to normal levels.
    """
    
    # Time array
    time_array = np.arange(num_samples) / sampling_rate
    
    # Initialize accelerometer data arrays
    acc_x = np.zeros(num_samples)
    acc_y = np.zeros(num_samples)
    acc_z = np.zeros(num_samples)
    
    # Calculate transition point
    high_movement_samples = int(high_movement_duration * sampling_rate)
    
    print(f"Generating accelerometer data:")
    print(f"  Total duration: {num_samples/sampling_rate:.1f} seconds")
    print(f"  High movement period: {high_movement_duration} seconds")
    print(f"  Transition at sample: {high_movement_samples}")
    
    for i in range(num_samples):
        current_time = time_array[i]
        
        if i < high_movement_samples:
            # High movement period (first 30 seconds) - this should trigger calibration pause
            
            # Base movement with larger amplitude
            base_movement_scale = 0.8  # Higher baseline movement
            
            # Add periodic head movements (nodding, shaking)
            nodding_freq = 0.5  # Hz - slow nodding
            shaking_freq = 1.2  # Hz - head shaking
            
            # Simulate head nodding (primarily Y-axis)
            nodding_component = 1.5 * np.sin(2 * np.pi * nodding_freq * current_time)
            
            # Simulate head shaking (primarily X-axis)  
            shaking_component = 1.2 * np.sin(2 * np.pi * shaking_freq * current_time)
            
            # Add random jerky movements
            jerk_amplitude = 0.8
            random_jerk_x = jerk_amplitude * (np.random.random() - 0.5)
            random_jerk_y = jerk_amplitude * (np.random.random() - 0.5)
            random_jerk_z = jerk_amplitude * (np.random.random() - 0.5)
            
            # Combine movements
            acc_x[i] = shaking_component + random_jerk_x + base_movement_scale * (np.random.random() - 0.5)
            acc_y[i] = nodding_component + random_jerk_y + base_movement_scale * (np.random.random() - 0.5)
            acc_z[i] = 9.81 + 0.3 * np.sin(2 * np.pi * 0.3 * current_time) + random_jerk_z  # Gravity + small movements
            
        else:
            # Normal/settled period - good signal quality
            
            # Much smaller baseline movement
            base_movement_scale = 0.1
            
            # Very subtle breathing-related movement
            breathing_freq = 0.25  # Hz - breathing rate
            breathing_amplitude = 0.05
            
            breathing_component_x = breathing_amplitude * np.sin(2 * np.pi * breathing_freq * current_time)
            breathing_component_y = breathing_amplitude * np.cos(2 * np.pi * breathing_freq * current_time)
            
            # Minimal random noise
            noise_amplitude = 0.02
            noise_x = noise_amplitude * (np.random.random() - 0.5)
            noise_y = noise_amplitude * (np.random.random() - 0.5)
            noise_z = noise_amplitude * (np.random.random() - 0.5)
            
            # Very stable accelerometer readings
            acc_x[i] = breathing_component_x + noise_x
            acc_y[i] = breathing_component_y + noise_y
            acc_z[i] = 9.81 + noise_z  # Gravity with minimal variation
    
    # Stack into (3, num_samples) array
    acc_data = np.stack([acc_x, acc_y, acc_z], axis=0)
    
    return acc_data

def add_accelerometer_to_eeg_file(file_path, output_path=None):
    """
    Add accelerometer data to existing EEG file with the correct Muse channel layout.
    
    Expected app channel layout:
    - Channel 0-3: EEG data (TP9, AF7, AF8, TP10)
    - Channels 4-8: Other Muse channels (AUX, etc.) - we'll pad with zeros
    - Channels 9-11: Accelerometer (X, Y, Z)
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Load existing data
    print(f"Loading existing EEG data from: {file_path}")
    existing_data = np.load(file_path)
    
    print(f"Original data shape: {existing_data.shape}")
    print(f"Data type: {existing_data.dtype}")
    
    # Analyze the structure
    if len(existing_data.shape) == 2:
        num_rows, num_samples = existing_data.shape
        print(f"Detected format: {num_rows} rows x {num_samples} samples")
        
        # Determine the structure
        if num_rows == 5:  # Timestamps + 4 EEG channels
            print("Format: [Timestamps, EEG_0, EEG_1, EEG_2, EEG_3]")
            timestamps = existing_data[0:1, :]  # Row 0: timestamps
            eeg_data = existing_data[1:5, :]     # Rows 1-4: EEG channels
            has_timestamps = True
        elif num_rows == 4:  # Just 4 EEG channels
            print("Format: [EEG_0, EEG_1, EEG_2, EEG_3]")
            timestamps = None
            eeg_data = existing_data  # All rows are EEG
            has_timestamps = False
        else:
            print(f"Unexpected number of rows: {num_rows}")
            return False
            
    else:
        print(f"Unexpected data shape: {existing_data.shape}")
        return False
    
    # Generate accelerometer data
    print(f"Generating accelerometer data for {num_samples} samples...")
    acc_data = generate_realistic_accelerometer_data(num_samples)
    print(f"Generated accelerometer data shape: {acc_data.shape}")
    
    # Create the full Muse channel layout expected by the app
    # Standard Muse LSL stream has 12 channels total:
    # 0-3: EEG (TP9, AF7, AF8, TP10)
    # 4-8: AUX channels (usually zeros or other sensor data)
    # 9-11: Accelerometer (X, Y, Z)
    
    print("Creating full Muse channel layout...")
    
    if has_timestamps:
        # Final structure: [Timestamps, EEG_0-3, AUX_4-8, ACC_9-11]
        num_aux_channels = 5  # Channels 4, 5, 6, 7, 8
        aux_data = np.zeros((num_aux_channels, num_samples))  # Pad with zeros
        
        # Combine all data
        full_data = np.vstack([
            timestamps,           # Row 0: Timestamps
            eeg_data,            # Rows 1-4: EEG channels (TP9, AF7, AF8, TP10)
            aux_data,            # Rows 5-9: AUX channels (padded with zeros)
            acc_data             # Rows 10-12: Accelerometer (X, Y, Z)
        ])
        
        channel_labels = (
            ["Timestamps"] + 
            [f"EEG_{i} ({'TP9,AF7,AF8,TP10'.split(',')[i]})" for i in range(4)] +
            [f"AUX_{i+4}" for i in range(5)] +
            ["ACC_X", "ACC_Y", "ACC_Z"]
        )
        
        print("Channel mapping for app:")
        print("  EEG_CHANNEL_INDICES = [1, 2, 3, 4]  # Rows 1-4 (after timestamps)")
        print("  ACC_CHANNEL_INDICES = [10, 11, 12]  # Rows 10-12")
        
    else:
        # Final structure: [EEG_0-3, AUX_4-8, ACC_9-11] (no timestamps)
        num_aux_channels = 5  # Channels 4, 5, 6, 7, 8
        aux_data = np.zeros((num_aux_channels, num_samples))  # Pad with zeros
        
        # Combine all data
        full_data = np.vstack([
            eeg_data,            # Rows 0-3: EEG channels (TP9, AF7, AF8, TP10)
            aux_data,            # Rows 4-8: AUX channels (padded with zeros)
            acc_data             # Rows 9-11: Accelerometer (X, Y, Z)
        ])
        
        channel_labels = (
            [f"EEG_{i} ({'TP9,AF7,AF8,TP10'.split(',')[i]})" for i in range(4)] +
            [f"AUX_{i+4}" for i in range(5)] +
            ["ACC_X", "ACC_Y", "ACC_Z"]
        )
        
        print("Channel mapping for app:")
        print("  EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # Rows 0-3")
        print("  ACC_CHANNEL_INDICES = [9, 10, 11]   # Rows 9-11")
    
    print(f"Final data shape: {full_data.shape}")
    print(f"Channel layout: {channel_labels}")
    
    # Save the enhanced data
    if output_path is None:
        output_path = file_path  # Overwrite original
    
    print(f"Saving enhanced data to: {output_path}")
    np.save(output_path, full_data)
    
    # Create a backup of original if overwriting
    if output_path == file_path:
        backup_path = file_path.replace('.npy', '_backup.npy')
        print(f"Creating backup of original data: {backup_path}")
        np.save(backup_path, existing_data)
    
    # Create a summary file
    summary_path = output_path.replace('.npy', '_channel_info.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Enhanced EEG File Channel Information\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original file: {file_path}\n")
        f.write(f"Enhanced file: {output_path}\n\n")
        f.write(f"Data shape: {full_data.shape}\n")
        f.write(f"Sampling rate: 256 Hz (assumed)\n")
        f.write(f"Duration: {num_samples/256:.1f} seconds\n\n")
        f.write("Channel Layout:\n")
        for i, label in enumerate(channel_labels):
            f.write(f"  Row {i:2d}: {label}\n")
        f.write(f"\nFor your app configuration:\n")
        if has_timestamps:
            f.write(f"EEG_CHANNEL_INDICES = [1, 2, 3, 4]  # Rows 1-4 (after timestamps)\n")
            f.write(f"ACC_CHANNEL_INDICES = [10, 11, 12]  # Rows 10-12\n")
        else:
            f.write(f"EEG_CHANNEL_INDICES = [0, 1, 2, 3]  # Rows 0-3\n")
            f.write(f"ACC_CHANNEL_INDICES = [9, 10, 11]   # Rows 9-11\n")
    
    print(f"Channel information saved to: {summary_path}")
    
    return True

def visualize_accelerometer_data(file_path, duration_to_plot=60):
    """
    Visualize the accelerometer data to verify it looks realistic.
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Load data
    data = np.load(file_path)
    
    # Determine if there are timestamps
    if data.shape[0] >= 13:  # Has timestamps
        acc_indices = [10, 11, 12]  # Accelerometer at rows 10-12
    else:  # No timestamps
        acc_indices = [9, 10, 11]   # Accelerometer at rows 9-11
    
    acc_data = data[acc_indices, :]  # Extract accelerometer data
    
    # Estimate sampling rate
    sampling_rate = 256
    samples_to_plot = min(duration_to_plot * sampling_rate, acc_data.shape[1])
    
    # Create time array
    time_array = np.arange(samples_to_plot) / sampling_rate
    
    # Plot accelerometer data
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Generated Accelerometer Data for NeuroFlow Signal Quality Testing', fontsize=14)
    
    labels = ['X-axis (Head Shaking)', 'Y-axis (Head Nodding)', 'Z-axis (Vertical/Gravity)']
    colors = ['red', 'green', 'blue']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time_array, acc_data[i, :samples_to_plot], color=color, linewidth=0.8)
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at 30 seconds to show transition
        ax.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='Movement settles')
        ax.legend()
    
    axes[-1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate movement statistics
    high_movement_samples = 30 * sampling_rate
    
    # Calculate RMS for movement assessment
    high_period_rms = np.sqrt(np.mean(acc_data[:2, :high_movement_samples]**2))
    normal_period_rms = np.sqrt(np.mean(acc_data[:2, high_movement_samples:]**2))
    
    print(f"\nMovement Statistics:")
    print(f"High movement period (0-30s) RMS: {high_period_rms:.3f} m/s²")
    print(f"Normal period (30s+) RMS: {normal_period_rms:.3f} m/s²")
    print(f"Movement reduction ratio: {high_period_rms/normal_period_rms:.1f}x")

def main():
    """Main function to process the EEG file"""
    
    # File path
    file_path = "live_session_data/eeg_data_raw.npy"
    
    print("=" * 70)
    print("NeuroFlow EEG File Accelerometer Data Enhancement Script")
    print("=" * 70)
    print(f"Target file: {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\nError: File not found at {file_path}")
        print("Please check the file path and ensure the file exists.")
        
        # Try to find similar files
        directory = os.path.dirname(file_path) or "."
        if os.path.exists(directory):
            print(f"\nFiles in {directory}:")
            for f in os.listdir(directory):
                if f.endswith('.npy'):
                    print(f"  - {f}")
        return
    
    # Show current file info
    existing_data = np.load(file_path)
    print(f"\nCurrent file info:")
    print(f"  Shape: {existing_data.shape}")
    print(f"  Duration: ~{existing_data.shape[1]/256:.1f} seconds (assuming 256 Hz)")
    
    # Ask for confirmation
    print(f"\nThis will enhance the file to match NeuroFlow's expected channel layout:")
    print(f"  - Preserve existing EEG data")
    print(f"  - Add padding channels for full Muse compatibility")
    print(f"  - Add realistic accelerometer data at correct indices")
    print(f"  - Create backup of original file")
    
    response = input(f"\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Process the file
    success = add_accelerometer_to_eeg_file(file_path)
    
    if success:
        print(f"\n✅ Successfully enhanced {file_path} with proper NeuroFlow channel layout!")
        print("\nThe enhanced file includes:")
        print("- Original EEG data preserved")
        print("- Proper Muse channel layout with padding")
        print("- Realistic accelerometer data at correct indices")
        print("- High movement period (0-30 seconds) - will trigger calibration pause")
        print("- Normal movement period (30+ seconds) - good signal quality")
        
        # Ask if user wants to visualize
        viz_response = input("\nWould you like to visualize the generated accelerometer data? (y/n): ")
        if viz_response.lower() == 'y':
            visualize_accelerometer_data(file_path)
            
    else:
        print("❌ Failed to enhance the file. Please check the error messages above.")

if __name__ == "__main__":
    main()