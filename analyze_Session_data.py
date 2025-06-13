#!/usr/bin/env python3
"""
Muse Session Data Analyzer

This script loads and analyzes saved session data from the Muse EEG GUI,
providing detailed visualizations and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from datetime import datetime
import seaborn as sns
from pathlib import Path

def load_session_data(filepath):
    """Load session data from NPZ file"""
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_session(data_file):
    """Analyze a complete session"""
    print(f"Analyzing session: {data_file}")
    
    # Load data
    data = load_session_data(data_file)
    if data is None:
        return
    
    # Extract basic info
    session_type = str(data['session_type']) if 'session_type' in data else 'Unknown'
    sampling_rate = float(data['sampling_rate']) if 'sampling_rate' in data else 256
    baseline_metrics = data['baseline_metrics'].item() if 'baseline_metrics' in data else None
    
    print(f"Session Type: {session_type}")
    print(f"Sampling Rate: {sampling_rate} Hz")
    
    if baseline_metrics:
        print("\nBaseline Metrics:")
        for key, value in baseline_metrics.items():
            print(f"  {key}: {value:.3f}")
    
    # Analyze band powers
    if 'band_powers' in data and len(data['band_powers']) > 0:
        print(f"\nBand Power Data Points: {len(data['band_powers'])}")
        analyze_band_powers(data['band_powers'], baseline_metrics)
    
    # Analyze predictions
    if 'predictions' in data and len(data['predictions']) > 0:
        print(f"Prediction Data Points: {len(data['predictions'])}")
        analyze_predictions(data['predictions'])
    
    # Analyze state changes
    if 'state_changes' in data and len(data['state_changes']) > 0:
        print(f"State Changes: {len(data['state_changes'])}")
        analyze_state_changes(data['state_changes'])
    
    # Analyze events
    if 'events' in data and len(data['events']) > 0:
        print(f"Events: {len(data['events'])}")
        analyze_events(data['events'])
    
    # Create visualizations
    create_visualizations(data, data_file)

def analyze_band_powers(band_powers, baseline_metrics):
    """Analyze band power data"""
    # Convert to DataFrame
    df_data = []
    for entry in band_powers:
        df_data.append({
            'timestamp': entry['timestamp'],
            'alpha': entry['alpha'],
            'beta': entry['beta'],
            'theta': entry['theta']
        })
    
    df = pd.DataFrame(df_data)
    
    print("\nBand Power Statistics:")
    print(df[['alpha', 'beta', 'theta']].describe())
    
    # Calculate relative to baseline
    if baseline_metrics:
        print("\nRelative to Baseline (mean ratio):")
        for band in ['alpha', 'beta', 'theta']:
            if band in baseline_metrics:
                baseline_val = baseline_metrics[band]
                mean_ratio = df[band].mean() / baseline_val
                std_ratio = df[band].std() / baseline_val
                print(f"  {band}: {mean_ratio:.3f}x ± {std_ratio:.3f} baseline")

def analyze_predictions(predictions):
    """Analyze prediction data"""
    # Extract prediction info
    states = []
    levels = []
    confidences = []
    timestamps = []
    
    for pred in predictions:
        pred_data = pred['prediction']
        states.append(pred_data['state'])
        levels.append(pred_data['level'])
        confidences.append(pred_data['confidence'])
        timestamps.append(pred['timestamp'])
    
    print("\nPrediction Statistics:")
    print(f"  Unique states: {set(states)}")
    print(f"  Level range: {min(levels)} to {max(levels)}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Mean level: {np.mean(levels):.2f}")
    
    # State distribution
    from collections import Counter
    state_counts = Counter(states)
    print("\nState Distribution:")
    for state, count in state_counts.most_common():
        percentage = count / len(states) * 100
        print(f"  {state}: {count} ({percentage:.1f}%)")

def analyze_state_changes(state_changes):
    """Analyze user-reported state changes"""
    print("\nUser State Changes:")
    start_time = state_changes[0]['time'] if state_changes else 0
    
    for i, change in enumerate(state_changes):
        time_rel = change['time'] - start_time
        print(f"  {time_rel:6.1f}s: {change['from_state']} → {change['to_state']}")

def analyze_events(events):
    """Analyze marked events"""
    print("\nMarked Events:")
    event_counts = {}
    start_time = events[0]['time'] if events else 0
    
    for event in events:
        label = event['label']
        event_counts[label] = event_counts.get(label, 0) + 1
        time_rel = event['time'] - start_time
        print(f"  {time_rel:6.1f}s: {label}")
    
    print("\nEvent Summary:")
    for label, count in event_counts.items():
        print(f"  {label}: {count}")

def create_visualizations(data, data_file):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(16, 14))
    
    # Plot 1: Band Powers Over Time
    if 'band_powers' in data and len(data['band_powers']) > 0:
        ax1 = plt.subplot(3, 3, 1)
        
        timestamps = [entry['timestamp'] for entry in data['band_powers']]
        alphas = [entry['alpha'] for entry in data['band_powers']]
        betas = [entry['beta'] for entry in data['band_powers']]
        thetas = [entry['theta'] for entry in data['band_powers']]
        
        # Convert to relative time
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        plt.plot(rel_times, alphas, 'b-', label='Alpha', linewidth=2)
        plt.plot(rel_times, betas, 'r-', label='Beta', linewidth=2)
        plt.plot(rel_times, thetas, 'g-', label='Theta', linewidth=2)
        
        # Add baseline lines
        if 'baseline_metrics' in data:
            baseline = data['baseline_metrics'].item()
            plt.axhline(baseline['alpha'], color='blue', linestyle='--', alpha=0.5, label='Alpha baseline')
            plt.axhline(baseline['beta'], color='red', linestyle='--', alpha=0.5, label='Beta baseline')
            plt.axhline(baseline['theta'], color='green', linestyle='--', alpha=0.5, label='Theta baseline')
        
        # Add state change markers
        if 'state_changes' in data and len(data['state_changes']) > 0:
            for change in data['state_changes']:
                change_time = change['time'] - timestamps[0]
                plt.axvline(change_time, color='black', linestyle=':', alpha=0.7)
                plt.text(change_time, plt.gca().get_ylim()[1]*0.9, change['to_state'], 
                        rotation=90, ha='right', va='top', fontsize=8)
        
        plt.title('Band Powers Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Levels Over Time
    if 'predictions' in data and len(data['predictions']) > 0:
        ax2 = plt.subplot(3, 3, 2)
        
        pred_timestamps = [entry['timestamp'] for entry in data['predictions']]
        pred_levels = [entry['prediction']['level'] for entry in data['predictions']]
        pred_confidences = [entry['prediction']['confidence'] for entry in data['predictions']]
        
        start_time = pred_timestamps[0]
        rel_times = [(t - start_time) for t in pred_timestamps]
        
        plt.plot(rel_times, pred_levels, 'k-', linewidth=2, label='Prediction Level')
        plt.fill_between(rel_times, pred_levels, alpha=0.3)
        
        # Add state change markers
        if 'state_changes' in data and len(data['state_changes']) > 0:
            for change in data['state_changes']:
                change_time = change['time'] - pred_timestamps[0]
                plt.axvline(change_time, color='red', linestyle=':', alpha=0.7)
        
        plt.title('Mental State Predictions')
        plt.xlabel('Time (s)')
        plt.ylabel('Prediction Level')
        plt.grid(True, alpha=0.3)
        plt.ylim(-4.5, 4.5)
        
        # Add level labels
        level_labels = {4: 'Very High', 2: 'High', 0: 'Neutral', -2: 'Low', -4: 'Very Low'}
        for level, label in level_labels.items():
            plt.axhline(level, color='gray', linestyle='-', alpha=0.2)
            plt.text(max(rel_times)*0.02, level, label, fontsize=8, alpha=0.7)
    
    # Plot 3: State Distribution Pie Chart
    if 'predictions' in data and len(data['predictions']) > 0:
        ax3 = plt.subplot(3, 3, 3)
        
        states = [entry['prediction']['state'] for entry in data['predictions']]
        from collections import Counter
        state_counts = Counter(states)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(state_counts)))
        plt.pie(state_counts.values(), labels=state_counts.keys(), autopct='%1.1f%%', colors=colors)
        plt.title('Predicted State Distribution')
    
    # Plot 4: Confidence Over Time
    if 'predictions' in data and len(data['predictions']) > 0:
        ax4 = plt.subplot(3, 3, 4)
        
        plt.plot(rel_times, pred_confidences, 'orange', linewidth=2)
        plt.axhline(np.mean(pred_confidences), color='orange', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(pred_confidences):.2f}')
        
        plt.title('Prediction Confidence')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    
    # Plot 5: Band Power Ratios
    if 'band_powers' in data and len(data['band_powers']) > 0:
        ax5 = plt.subplot(3, 3, 5)
        
        ab_ratios = [entry['alpha'] / entry['beta'] if entry['beta'] > 0 else 0 for entry in data['band_powers']]
        bt_ratios = [entry['beta'] / entry['theta'] if entry['theta'] > 0 else 0 for entry in data['band_powers']]
        
        plt.plot(rel_times, ab_ratios, 'purple', label='Alpha/Beta', linewidth=2)
        plt.plot(rel_times, bt_ratios, 'brown', label='Beta/Theta', linewidth=2)
        
        # Add baseline ratios
        if 'baseline_metrics' in data:
            baseline = data['baseline_metrics'].item()
            if 'ab_ratio' in baseline:
                plt.axhline(baseline['ab_ratio'], color='purple', linestyle='--', alpha=0.5)
            if 'bt_ratio' in baseline:
                plt.axhline(baseline['bt_ratio'], color='brown', linestyle='--', alpha=0.5)
        
        plt.title('Band Power Ratios')
        plt.xlabel('Time (s)')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Raw EEG Overview (if available)
    if 'eeg_raw' in data and data['eeg_raw'].size > 0:
        ax6 = plt.subplot(3, 3, 6)
        
        eeg_raw = data['eeg_raw']
        sampling_rate = float(data['sampling_rate']) if 'sampling_rate' in data else 256
        
        # Show first 10 seconds of each channel
        samples_to_show = min(int(sampling_rate * 10), eeg_raw.shape[1])
        time_axis = np.arange(samples_to_show) / sampling_rate
        
        channel_names = data['channel_names'] if 'channel_names' in data else [f'Ch{i+1}' for i in range(eeg_raw.shape[0])]
        
        for i in range(min(4, eeg_raw.shape[0])):
            offset = i * 200  # Larger offset for better visualization
            plt.plot(time_axis, eeg_raw[i, :samples_to_show] + offset, label=channel_names[i])
        
        plt.title('Raw EEG (First 10s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude + Offset (μV)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 7: Band Power Correlation Matrix
    if 'band_powers' in data and len(data['band_powers']) > 0:
        ax7 = plt.subplot(3, 3, 7)
        
        # Create correlation matrix
        df_bands = pd.DataFrame({
            'Alpha': [entry['alpha'] for entry in data['band_powers']],
            'Beta': [entry['beta'] for entry in data['band_powers']],
            'Theta': [entry['theta'] for entry in data['band_powers']]
        })
        
        corr_matrix = df_bands.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Band Power Correlations')
    
    # Plot 8: Spectral Power Distribution
    if 'band_powers' in data and len(data['band_powers']) > 0:
        ax8 = plt.subplot(3, 3, 8)
        
        # Box plot of band powers
        band_data = {
            'Alpha': [entry['alpha'] for entry in data['band_powers']],
            'Beta': [entry['beta'] for entry in data['band_powers']],
            'Theta': [entry['theta'] for entry in data['band_powers']]
        }
        
        plt.boxplot(band_data.values(), labels=band_data.keys())
        plt.title('Band Power Distributions')
        plt.ylabel('Power')
        plt.grid(True, alpha=0.3)
        
        # Add baseline markers
        if 'baseline_metrics' in data:
            baseline = data['baseline_metrics'].item()
            for i, band in enumerate(['alpha', 'beta', 'theta']):
                if band in baseline:
                    plt.plot(i+1, baseline[band], 'ro', markersize=8, label='Baseline' if i == 0 else '')
        plt.legend()
    
    # Plot 9: Session Timeline
    ax9 = plt.subplot(3, 3, 9)
    
    # Create timeline visualization
    timeline_data = []
    
    # Add predictions
    if 'predictions' in data and len(data['predictions']) > 0:
        for entry in data['predictions']:
            timeline_data.append({
                'time': entry['timestamp'],
                'type': 'Prediction',
                'value': entry['prediction']['level'],
                'label': entry['prediction']['state']
            })
    
    # Add state changes
    if 'state_changes' in data and len(data['state_changes']) > 0:
        for change in data['state_changes']:
            timeline_data.append({
                'time': change['time'],
                'type': 'User State',
                'value': 0,
                'label': change['to_state']
            })
    
    # Add events
    if 'events' in data and len(data['events']) > 0:
        for event in data['events']:
            timeline_data.append({
                'time': event['time'],
                'type': 'Event',
                'value': -3,
                'label': event['label']
            })
    
    if timeline_data:
        # Sort by time
        timeline_data.sort(key=lambda x: x['time'])
        start_time = timeline_data[0]['time']
        
        # Plot different types
        for item in timeline_data:
            rel_time = item['time'] - start_time
            if item['type'] == 'Prediction':
                plt.scatter(rel_time, item['value'], c='blue', alpha=0.6, s=20)
            elif item['type'] == 'User State':
                plt.scatter(rel_time, item['value'], c='red', alpha=0.8, s=50, marker='^')
            elif item['type'] == 'Event':
                plt.scatter(rel_time, item['value'], c='green', alpha=0.8, s=30, marker='s')
        
        plt.title('Session Timeline')
        plt.xlabel('Time (s)')
        plt.ylabel('Level / Type')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Predictions'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='User States'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8, label='Events')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    base_name = Path(data_file).stem
    output_path = f"{base_name}_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_path}")
    
    # Show the plot
    plt.show()

def compare_sessions(data_files):
    """Compare multiple sessions"""
    print(f"\nComparing {len(data_files)} sessions:")
    
    comparison_data = []
    
    for file_path in data_files:
        data = load_session_data(file_path)
        if data is None:
            continue
        
        session_info = {
            'filename': os.path.basename(file_path),
            'session_type': str(data.get('session_type', 'Unknown')),
            'duration': 0,
            'num_predictions': 0,
            'num_state_changes': 0,
            'num_events': 0,
            'mean_alpha': 0,
            'mean_beta': 0,
            'mean_theta': 0,
            'mean_prediction_level': 0,
            'mean_confidence': 0
        }
        
        # Calculate metrics
        if 'band_powers' in data and len(data['band_powers']) > 0:
            timestamps = [entry['timestamp'] for entry in data['band_powers']]
            session_info['duration'] = timestamps[-1] - timestamps[0]
            session_info['mean_alpha'] = np.mean([entry['alpha'] for entry in data['band_powers']])
            session_info['mean_beta'] = np.mean([entry['beta'] for entry in data['band_powers']])
            session_info['mean_theta'] = np.mean([entry['theta'] for entry in data['band_powers']])
        
        if 'predictions' in data:
            session_info['num_predictions'] = len(data['predictions'])
            if len(data['predictions']) > 0:
                session_info['mean_prediction_level'] = np.mean([entry['prediction']['level'] for entry in data['predictions']])
                session_info['mean_confidence'] = np.mean([entry['prediction']['confidence'] for entry in data['predictions']])
        
        if 'state_changes' in data:
            session_info['num_state_changes'] = len(data['state_changes'])
        
        if 'events' in data:
            session_info['num_events'] = len(data['events'])
        
        comparison_data.append(session_info)
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\nSession Comparison:")
    print(df_comparison.to_string(index=False))
    
    # Save comparison to CSV
    df_comparison.to_csv('session_comparison.csv', index=False)
    print("\nComparison saved to: session_comparison.csv")
    
    return df_comparison

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Analyze Muse EEG session data')
    parser.add_argument('files', nargs='+', help='Session data files (.npz)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple sessions')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Change to output directory
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Process files
    if args.compare and len(args.files) > 1:
        compare_sessions(args.files)
    else:
        for file_path in args.files:
            print("="*80)
            analyze_session(file_path)
            print("="*80)

if __name__ == "__main__":
    # If no command line arguments, look for NPZ files in current directory
    import sys
    if len(sys.argv) == 1:
        npz_files = list(Path('.').glob('*.npz'))
        if npz_files:
            print(f"Found {len(npz_files)} NPZ files in current directory:")
            for i, file in enumerate(npz_files):
                print(f"  {i+1}. {file}")
            
            choice = input("\nEnter file number to analyze (or 'all' for all files): ")
            
            if choice.lower() == 'all':
                for file in npz_files:
                    print("="*80)
                    analyze_session(str(file))
                    print("="*80)
            else:
                try:
                    file_idx = int(choice) - 1
                    if 0 <= file_idx < len(npz_files):
                        analyze_session(str(npz_files[file_idx]))
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Invalid input")
        else:
            print("No NPZ files found in current directory.")
            print("Usage: python analyze_session_data.py <file.npz> [<file2.npz> ...]")
    else:
        main()