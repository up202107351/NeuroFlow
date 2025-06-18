
"""
Frequency Content Analysis for Research Dataset - Full Files Version

Analyzes the frequency content of ENTIRE sessions from the research dataset.
Uses complete files instead of just 6000:8000 range.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, welch, iirnotch
import os
import glob
from typing import Dict, List, Tuple, Optional

def frequency_analysis_research_dataset():
    """
    Comprehensive frequency analysis of the research dataset
    Using ENTIRE files, filter entire sessions once, then analyze windows
    """
    
    # Data path for research dataset
    data_path = r"C:\Users\berna\OneDrive\Documentos\4A 2S\Neuro\Projeto Antigo\Dataset"
    
    print(f"Analyzing research dataset from: {data_path}")
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'subject' in f]
    print(f"Found {len(csv_files)} CSV files")
    
    # Parameters
    fs = 256.0
    nyq = fs / 2.0
    
    # Setup filters (matching your original preprocessing)
    b_bpf, a_bpf = butter(4, [0.5, 50.0], btype="band", fs=fs)
    w0 = 50.0 / nyq
    if 0 < w0 < 1:
        b_notch, a_notch = iirnotch(w0, Q=30)
    
    # Band definitions (matching EEG processing worker)
    band_defs = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    
    colors = {
        'theta': 'blue', 
        'alpha': 'green', 
        'beta': 'orange', 
        'gamma': 'red'
    }
    
    # EEG channel names (correct format for research dataset)
    chans = ["TP9", "AF7", "AF8", "TP10"]
    
    # Storage for all results
    all_results = {}
    
    # Process each file
    for file in csv_files:
        file_path = os.path.join(data_path, file)
        print(f"\nAnalyzing: {file}")
        
        try:
            # Parse filename
            parts = file.replace('.csv', '').split('-')
            subject = parts[0].replace('subject', '')
            condition = parts[1]
            trial = parts[2] if len(parts) > 2 else '1'
            
            # Load ENTIRE file
            df = pd.read_csv(file_path)
            print(f"  Raw shape: {df.shape}")
            
            # Check minimum file size
            min_samples = 1000  # Minimum 4 seconds of data
            if len(df) < min_samples:
                print(f"  ✗ File too short ({len(df)} samples < {min_samples} minimum)")
                continue
            
            # Extract EEG data from ENTIRE file (columns 1-4: TP9, AF7, AF8, TP10)
            eeg_data = df.iloc[:, 1:5]  # USE ENTIRE FILE
            eeg_data.columns = chans
            
            total_duration_sec = len(eeg_data) / fs
            total_duration_min = total_duration_sec / 60
            print(f"  EEG data: {len(eeg_data)} samples ({total_duration_sec:.1f}s = {total_duration_min:.1f}min)")
            print(f"  Value range: {eeg_data.min().min():.2f} to {eeg_data.max().max():.2f}")
            
            # FILTER ENTIRE SESSION ONCE (per channel)
            filtered_data = {}
            print(f"  Filtering entire session...")
            
            for ch in chans:
                raw_data = eeg_data[ch].values
                
                # Handle extreme outliers once for entire session
                q75, q25 = np.percentile(raw_data, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 3 * iqr
                upper_bound = q75 + 3 * iqr
                clean_data = np.clip(raw_data, lower_bound, upper_bound)
                
                # Detrend entire session
                detrended_data = detrend(clean_data)
                
                # Filter entire session
                filtered_signal = filtfilt(b_bpf, a_bpf, detrended_data)
                if 0 < w0 < 1:
                    filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
                
                filtered_data[ch] = filtered_signal
                print(f"    {ch}: {raw_data.min():.1f} to {raw_data.max():.1f} → {filtered_signal.min():.1f} to {filtered_signal.max():.1f}")
            
            # Now analyze windows from the already-filtered data
            window_sec = 5  # 5-second windows for longer sessions
            window_samples = int(window_sec * fs)
            step_samples = window_samples // 2  # 50% overlap
            
            num_windows = (len(eeg_data) - window_samples) // step_samples + 1
            print(f"  Analyzing {num_windows} windows of {window_sec} seconds each")
            
            # Storage for this session
            time_points = []
            band_powers = {band: {ch: [] for ch in chans} for band in band_defs.keys()}
            avg_band_powers = {band: [] for band in band_defs.keys()}
            ratios = {'theta_alpha': [], 'beta_alpha': [], 'alpha_beta': [], 'beta_theta': []}
            dominant_freqs = []
            dominant_bands = []
            psd_matrices = []  # Store PSDs for later analysis
            
            # Process each window (NO FILTERING HERE - use pre-filtered data)
            for window_idx in range(num_windows):
                start_sample = window_idx * step_samples
                end_sample = start_sample + window_samples
                
                if end_sample > len(eeg_data):
                    break
                
                time_sec = start_sample / fs
                time_min = time_sec / 60
                
                # Extract window from pre-filtered data
                window_powers = {band: [] for band in band_defs.keys()}
                window_psds = []
                
                for ch in chans:
                    # Use already filtered data - NO ADDITIONAL PROCESSING
                    window_data = filtered_data[ch][start_sample:end_sample]
                    
                    # Calculate PSD directly from filtered window
                    f, psd = welch(window_data, fs=fs, nperseg=min(len(window_data), int(fs*2)), 
                                  noverlap=int(fs), window="hann", detrend=None)  # No detrending
                    
                    window_psds.append(psd)
                    
                    # Calculate band powers for this channel
                    for band, (fl, fh) in band_defs.items():
                        band_mask = (f >= fl) & (f < fh)
                        if np.any(band_mask):
                            power = np.trapz(psd[band_mask], f[band_mask])
                            band_powers[band][ch].append(power)
                            window_powers[band].append(power)
                        else:
                            band_powers[band][ch].append(0)
                            window_powers[band].append(0)
                
                # Calculate average across channels for this window
                for band in band_defs.keys():
                    avg_power = np.mean(window_powers[band]) if window_powers[band] else 0
                    avg_band_powers[band].append(avg_power)
                
                # Calculate ratios
                alpha_power = avg_band_powers['alpha'][-1]
                beta_power = avg_band_powers['beta'][-1]
                theta_power = avg_band_powers['theta'][-1]
                
                if alpha_power > 0:
                    ratios['theta_alpha'].append(theta_power / alpha_power)
                    ratios['beta_alpha'].append(beta_power / alpha_power)
                    ratios['alpha_beta'].append(alpha_power / beta_power if beta_power > 0 else 0)
                else:
                    ratios['theta_alpha'].append(0)
                    ratios['beta_alpha'].append(0)
                    ratios['alpha_beta'].append(0)
                
                if theta_power > 0:
                    ratios['beta_theta'].append(beta_power / theta_power)
                else:
                    ratios['beta_theta'].append(0)
                
                # Find dominant frequency (average across channels)
                avg_psd = np.mean(window_psds, axis=0)
                dominant_freq_idx = np.argmax(avg_psd)
                dominant_freq = f[dominant_freq_idx]
                dominant_freqs.append(dominant_freq)
                
                # Determine dominant band
                for band, (fl, fh) in band_defs.items():
                    if fl <= dominant_freq < fh:
                        dominant_bands.append(band)
                        break
                else:
                    dominant_bands.append('other')
                
                time_points.append(time_min)  # Store in minutes for better readability
                psd_matrices.append(avg_psd)
            
            # Store results for this session
            session_key = f"{subject}_{condition}_{trial}"
            all_results[session_key] = {
                'subject': subject,
                'condition': condition,
                'trial': trial,
                'time_points': time_points,
                'band_powers': avg_band_powers,
                'channel_powers': band_powers,
                'ratios': ratios,
                'dominant_bands': dominant_bands,
                'dominant_freqs': dominant_freqs,
                'psd_matrices': psd_matrices,
                'frequencies': f,
                'filtered_data': filtered_data,  # Store filtered session data
                'file_info': {
                    'filename': file,
                    'total_samples': len(eeg_data),
                    'duration_sec': total_duration_sec,
                    'duration_min': total_duration_min,
                    'num_windows': len(time_points),
                    'window_size_sec': window_sec
                }
            }
            
            print(f"  ✓ Processed {len(time_points)} windows successfully ({total_duration_min:.1f} minutes)")
            
        except Exception as e:
            print(f"  ✗ Error processing {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comprehensive analysis plots
    generate_comprehensive_plots(all_results, band_defs, colors)
    
    # Generate detailed statistics
    generate_detailed_statistics(all_results, band_defs)
    
    return all_results

def generate_comprehensive_plots(all_results: Dict, band_defs: Dict, colors: Dict):
    """Generate comprehensive plots for all sessions"""
    
    print(f"\nGenerating comprehensive plots...")
    
    # Organize data by subject and condition
    subjects = set([r['subject'] for r in all_results.values()])
    conditions = set([r['condition'] for r in all_results.values()])
    
    print(f"Subjects: {sorted(subjects)}")
    print(f"Conditions: {sorted(conditions)}")
    
    # 1. Session overview plots for each subject-condition combination
    for subject in sorted(subjects):
        subject_sessions = {k: v for k, v in all_results.items() if v['subject'] == subject}
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'Subject {subject.upper()} - Complete Session Analysis', fontsize=16)
        
        # Plot 1: Band powers over time for all conditions
        ax = axes[0, 0]
        
        condition_linestyles = {'neutral': '-', 'relaxed': '--', 'concentrating': ':'}
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            
            for session_key, session in condition_sessions.items():
                trial_label = f"{condition} T{session['trial']}"
                
                for band in band_defs.keys():
                    ax.plot(session['time_points'], session['band_powers'][band], 
                           color=colors[band], 
                           linestyle=condition_linestyles.get(condition, '-'),
                           alpha=0.8, linewidth=1.5,
                           label=f'{band}-{condition}' if session_key == list(condition_sessions.keys())[0] else "")
        
        ax.set_title('Band Powers Over Time (All Conditions)')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Power (log scale)')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Alpha/Beta ratio evolution
        ax = axes[0, 1]
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            
            for session_key, session in condition_sessions.items():
                duration = session['file_info']['duration_min']
                ax.plot(session['time_points'], session['ratios']['alpha_beta'], 
                       linestyle=condition_linestyles.get(condition, '-'),
                       linewidth=2, alpha=0.8,
                       label=f'{condition} ({duration:.1f}min)' if session_key == list(condition_sessions.keys())[0] else "")
        
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
        ax.set_title('Alpha/Beta Ratio Evolution')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Alpha/Beta Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Session duration comparison
        ax = axes[1, 0]
        
        session_durations = {}
        session_counts = {}
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            durations = [s['file_info']['duration_min'] for s in condition_sessions.values()]
            
            if durations:
                session_durations[condition] = durations
                session_counts[condition] = len(durations)
        
        # Bar plot of durations
        x_pos = range(len(session_durations))
        for i, (condition, durations) in enumerate(session_durations.items()):
            for j, duration in enumerate(durations):
                ax.bar(i + j*0.3, duration, width=0.25, alpha=0.8, 
                      label=f'{condition} T{j+1}')
        
        ax.set_title('Session Durations')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Duration (minutes)')
        ax.set_xticks(range(len(session_durations)))
        ax.set_xticklabels(session_durations.keys())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Average frequency spectra by condition
        ax = axes[1, 1]
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            
            all_psds = []
            total_duration = 0
            
            for session in condition_sessions.values():
                all_psds.extend(session['psd_matrices'])
                total_duration += session['file_info']['duration_min']
            
            if all_psds:
                avg_psd = np.mean(all_psds, axis=0)
                frequencies = session['frequencies']  # Same for all sessions
                
                ax.semilogy(frequencies, avg_psd, 
                           linestyle=condition_linestyles.get(condition, '-'),
                           linewidth=3, label=f'{condition} ({total_duration:.1f}min total)')
        
        ax.set_xlim(0, 30)
        ax.set_title('Average Frequency Spectra by Condition')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (µV²/Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add band boundaries
        for band, (fl, fh) in band_defs.items():
            if fh <= 30:
                ax.axvspan(fl, fh, alpha=0.15, color=colors[band])
                ax.text((fl+fh)/2, ax.get_ylim()[0]*2, band, rotation=90, 
                       ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Dominant frequency distribution over time
        ax = axes[2, 0]
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            
            for session_key, session in condition_sessions.items():
                duration = session['file_info']['duration_min']
                # Color points by dominant band
                colors_mapped = [colors.get(band, 'gray') for band in session['dominant_bands']]
                
                scatter = ax.scatter(session['time_points'], session['dominant_freqs'], 
                                   c=colors_mapped, s=30, alpha=0.7,
                                   label=f'{condition} ({duration:.1f}min)' if session_key == list(condition_sessions.keys())[0] else "")
        
        ax.set_title('Dominant Frequency Over Time')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add band boundaries
        for band, (fl, fh) in band_defs.items():
            ax.axhspan(fl, fh, alpha=0.1, color=colors[band])
            ax.text(ax.get_xlim()[1]*0.95, (fl+fh)/2, band, fontsize=8, va='center')
        
        # Plot 6: Session summary statistics
        ax = axes[2, 1]
        
        # Create summary table
        summary_data = []
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            
            for session in condition_sessions.values():
                mean_ab_ratio = np.mean(session['ratios']['alpha_beta'])
                mean_bt_ratio = np.mean(session['ratios']['beta_theta'])
                duration = session['file_info']['duration_min']
                
                summary_data.append([
                    f"{condition} T{session['trial']}", 
                    f"{duration:.1f}min",
                    f"{mean_ab_ratio:.3f}",
                    f"{mean_bt_ratio:.3f}"
                ])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Session', 'Duration', 'α/β Ratio', 'β/θ Ratio'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax.axis('off')
        ax.set_title('Session Summary Statistics')
        
        plt.tight_layout()
        filename = f'subject_{subject}_complete_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")
    
    # 2. Cross-subject comparison with session durations
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Cross-Subject Analysis - Complete Sessions', fontsize=16)
    
    # Plot 1: Session durations by subject and condition
    ax = axes[0, 0]
    
    subjects_list = sorted(subjects)
    conditions_list = sorted(conditions)
    
    x = np.arange(len(subjects_list))
    width = 0.25
    
    for i, condition in enumerate(conditions_list):
        durations_by_subject = []
        
        for subject in subjects_list:
            subject_sessions = [s for s in all_results.values() 
                              if s['subject'] == subject and s['condition'] == condition]
            avg_duration = np.mean([s['file_info']['duration_min'] for s in subject_sessions]) if subject_sessions else 0
            durations_by_subject.append(avg_duration)
        
        ax.bar(x + i*width, durations_by_subject, width, label=condition, alpha=0.8)
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Average Duration (minutes)')
    ax.set_title('Average Session Duration by Subject and Condition')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Subject {s.upper()}' for s in subjects_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Alpha/Beta ratio by condition (session-level averages)
    ax = axes[0, 1]
    
    ab_ratio_data = {condition: [] for condition in conditions_list}
    
    for session in all_results.values():
        condition = session['condition']
        session_avg_ab = np.mean(session['ratios']['alpha_beta'])
        ab_ratio_data[condition].append(session_avg_ab)
    
    # Box plot
    box_data = [ab_ratio_data[condition] for condition in conditions_list]
    bp = ax.boxplot(box_data, labels=conditions_list, patch_artist=True)
    
    # Color the boxes
    condition_colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], condition_colors[:len(conditions_list)]):
        patch.set_facecolor(color)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
    ax.set_title('Alpha/Beta Ratio Distribution by Condition (Session Averages)')
    ax.set_ylabel('Alpha/Beta Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total data duration by subject
    ax = axes[0, 2]
    
    total_durations = {}
    condition_durations = {condition: {} for condition in conditions_list}
    
    for subject in subjects_list:
        subject_sessions = [s for s in all_results.values() if s['subject'] == subject]
        total_duration = sum([s['file_info']['duration_min'] for s in subject_sessions])
        total_durations[subject] = total_duration
        
        for condition in conditions_list:
            condition_sessions = [s for s in subject_sessions if s['condition'] == condition]
            condition_duration = sum([s['file_info']['duration_min'] for s in condition_sessions])
            condition_durations[condition][subject] = condition_duration
    
    # Stacked bar chart
    bottom = np.zeros(len(subjects_list))
    
    for i, condition in enumerate(conditions_list):
        durations = [condition_durations[condition].get(subject, 0) for subject in subjects_list]
        ax.bar(subjects_list, durations, bottom=bottom, label=condition, alpha=0.8)
        bottom += durations
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Total Duration (minutes)')
    ax.set_title('Total Data Duration by Subject and Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Time evolution patterns
    ax = axes[1, 0]
    
    # Show how alpha/beta ratio changes over normalized time across all sessions
    for condition in conditions_list:
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        
        all_normalized_times = []
        all_ab_ratios = []
        
        for session in condition_sessions:
            duration = session['file_info']['duration_min']
            normalized_times = [t / duration for t in session['time_points']]  # Normalize to 0-1
            
            all_normalized_times.extend(normalized_times)
            all_ab_ratios.extend(session['ratios']['alpha_beta'])
        
        if all_normalized_times:
            # Bin the data to show trends
            bins = np.linspace(0, 1, 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = []
            
            for i in range(len(bins)-1):
                mask = (np.array(all_normalized_times) >= bins[i]) & (np.array(all_normalized_times) < bins[i+1])
                if np.any(mask):
                    bin_means.append(np.mean(np.array(all_ab_ratios)[mask]))
                else:
                    bin_means.append(np.nan)
            
            ax.plot(bin_centers, bin_means, 'o-', linewidth=2, label=condition, markersize=4)
    
    ax.set_xlabel('Normalized Time (0=start, 1=end)')
    ax.set_ylabel('Alpha/Beta Ratio')
    ax.set_title('Alpha/Beta Ratio Evolution (Normalized Time)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    # Plot 5: Dominant band percentages
    ax = axes[1, 1]
    
    condition_band_percentages = {}
    
    for condition in conditions_list:
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        
        all_dominant_bands = []
        for session in condition_sessions:
            all_dominant_bands.extend(session['dominant_bands'])
        
        band_counts = {band: all_dominant_bands.count(band) for band in band_defs.keys()}
        total_windows = len(all_dominant_bands)
        
        if total_windows > 0:
            band_percentages = {band: count/total_windows*100 for band, count in band_counts.items()}
            condition_band_percentages[condition] = band_percentages
    
    # Stacked bar chart
    x_pos = range(len(conditions_list))
    bottom = np.zeros(len(conditions_list))
    
    for band in band_defs.keys():
        percentages = [condition_band_percentages.get(condition, {}).get(band, 0) for condition in conditions_list]
        ax.bar(x_pos, percentages, bottom=bottom, label=band, color=colors[band], alpha=0.8)
        bottom += percentages
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('Percentage of Windows')
    ax.set_title('Dominant Band Distribution by Condition')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions_list)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Data quality overview
    ax = axes[1, 2]
    
    # Summary statistics table
    summary_stats = []
    
    for condition in conditions_list:
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        
        total_sessions = len(condition_sessions)
        total_duration = sum([s['file_info']['duration_min'] for s in condition_sessions])
        avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
        total_windows = sum([s['file_info']['num_windows'] for s in condition_sessions])
        
        # Average alpha/beta ratio
        all_ab_ratios = []
        for session in condition_sessions:
            all_ab_ratios.extend(session['ratios']['alpha_beta'])
        avg_ab_ratio = np.mean(all_ab_ratios) if all_ab_ratios else 0
        
        summary_stats.append([
            condition,
            f"{total_sessions}",
            f"{total_duration:.1f}",
            f"{avg_duration:.1f}",
            f"{total_windows}",
            f"{avg_ab_ratio:.3f}"
        ])
    
    # Create summary table
    table = ax.table(cellText=summary_stats,
                    colLabels=['Condition', 'Sessions', 'Total\nDuration\n(min)', 'Avg\nDuration\n(min)', 'Total\nWindows', 'Avg α/β\nRatio'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    ax.axis('off')
    ax.set_title('Data Quality Summary')
    
    plt.tight_layout()
    plt.savefig('complete_sessions_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved complete_sessions_analysis.png")

def generate_detailed_statistics(all_results: Dict, band_defs: Dict):
    """Generate detailed statistical analysis for complete sessions"""
    
    print(f"\n{'='*80}")
    print("COMPLETE SESSION FREQUENCY ANALYSIS STATISTICS")
    print(f"{'='*80}")
    
    # Organize data
    subjects = set([r['subject'] for r in all_results.values()])
    conditions = set([r['condition'] for r in all_results.values()])
    
    print(f"\nDataset Summary:")
    print(f"  Subjects: {len(subjects)} ({sorted(subjects)})")
    print(f"  Conditions: {len(conditions)} ({sorted(conditions)})")
    print(f"  Total sessions: {len(all_results)}")
    
    # Session duration analysis
    print(f"\nSession Duration Analysis:")
    
    total_duration = 0
    duration_by_condition = {condition: [] for condition in conditions}
    
    for session in all_results.values():
        duration = session['file_info']['duration_min']
        total_duration += duration
        duration_by_condition[session['condition']].append(duration)
    
    print(f"  Total data: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")
    
    for condition in sorted(conditions):
        durations = duration_by_condition[condition]
        if durations:
            total_cond = sum(durations)
            avg_cond = np.mean(durations)
            std_cond = np.std(durations)
            min_cond = min(durations)
            max_cond = max(durations)
            
            print(f"  {condition}:")
            print(f"    Sessions: {len(durations)}")
            print(f"    Total: {total_cond:.1f} min ({total_cond/60:.1f} hours)")
            print(f"    Average: {avg_cond:.1f} ± {std_cond:.1f} min")
            print(f"    Range: {min_cond:.1f} - {max_cond:.1f} min")
    
    # Session-level band power statistics
    print(f"\nSession-Level Band Power Statistics:")
    print(f"  (Each session contributes one data point - average of all windows)")
    
    session_stats = {}
    
    for condition in sorted(conditions):
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        print(f"\n{condition.upper()}:")
        print(f"  Sessions: {len(condition_sessions)}")
        
        session_stats[condition] = {}
        
        # Calculate session-level averages for band powers
        for band in band_defs.keys():
            session_averages = []
            for session in condition_sessions:
                session_avg = np.mean(session['band_powers'][band])
                session_averages.append(session_avg)
            
            if session_averages:
                mean_power = np.mean(session_averages)
                std_power = np.std(session_averages)
                session_stats[condition][f'{band}_power'] = (mean_power, std_power)
                print(f"    {band:5s} power: {mean_power:8.3f} ± {std_power:6.3f}")
        
        # Calculate session-level ratio averages
        for ratio_name in ['alpha_beta', 'beta_theta', 'theta_alpha']:
            session_averages = []
            for session in condition_sessions:
                session_avg = np.mean(session['ratios'][ratio_name])
                session_averages.append(session_avg)
            
            if session_averages:
                mean_ratio = np.mean(session_averages)
                std_ratio = np.std(session_averages)
                session_stats[condition][f'{ratio_name}_ratio'] = (mean_ratio, std_ratio)
                print(f"    {ratio_name:11s}: {mean_ratio:6.3f} ± {std_ratio:5.3f}")
    
    # Per-subject analysis with complete session data
    print(f"\nPer-Subject Analysis (Complete Sessions):")
    for subject in sorted(subjects):
        subject_sessions = [s for s in all_results.values() if s['subject'] == subject]
        
        # Calculate total data for this subject
        total_subject_duration = sum([s['file_info']['duration_min'] for s in subject_sessions])
        
        print(f"\nSUBJECT {subject.upper()} (Total: {total_subject_duration:.1f} minutes):")
        
        for condition in sorted(conditions):
            condition_sessions = [s for s in subject_sessions if s['condition'] == condition]
            if not condition_sessions:
                continue
            
            # Calculate condition totals for this subject
            condition_duration = sum([s['file_info']['duration_min'] for s in condition_sessions])
            
            print(f"  {condition} ({condition_duration:.1f} min):")
            
            # Alpha/Beta ratio (key metric for classification)
            ab_ratios = []
            for session in condition_sessions:
                session_avg_ab = np.mean(session['ratios']['alpha_beta'])
                ab_ratios.append(session_avg_ab)
            
            if ab_ratios:
                mean_ab = np.mean(ab_ratios)
                std_ab = np.std(ab_ratios) if len(ab_ratios) > 1 else 0
                print(f"    Alpha/Beta ratio: {mean_ab:.3f} ± {std_ab:.3f}")
                
                # Classification hint based on alpha/beta ratio
                if mean_ab > 1.2:
                    hint = "HIGH (relaxed-like)"
                elif mean_ab < 0.8:
                    hint = "LOW (focused-like)"
                else:
                    hint = "NEUTRAL"
                print(f"    Classification hint: {hint}")
                
                # Show session-by-session variation if multiple sessions
                if len(ab_ratios) > 1:
                    for i, session_ab in enumerate(ab_ratios):
                        session = condition_sessions[i]
                        duration = session['file_info']['duration_min']
                        print(f"      Trial {session['trial']}: {session_ab:.3f} ({duration:.1f}min)")
    
    # Statistical significance testing (session-level)
    print(f"\nStatistical Significance Testing (Session-Level):")
    
    try:
        from scipy.stats import ttest_ind, f_oneway
        
        # Test Alpha/Beta ratio differences
        print(f"\nALPHA/BETA RATIO COMPARISONS:")
        
        condition_ab_data = {}
        for condition in conditions:
            condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
            
            session_averages = []
            for session in condition_sessions:
                session_avg = np.mean(session['ratios']['alpha_beta'])
                session_averages.append(session_avg)
            
            condition_ab_data[condition] = session_averages
        
        # Pairwise t-tests
        condition_list = list(condition_ab_data.keys())
        for i in range(len(condition_list)):
            for j in range(i+1, len(condition_list)):
                cond1, cond2 = condition_list[i], condition_list[j]
                
                if len(condition_ab_data[cond1]) > 0 and len(condition_ab_data[cond2]) > 0:
                    t_stat, p_val = ttest_ind(condition_ab_data[cond1], condition_ab_data[cond2])
                    significance = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                    
                    mean1 = np.mean(condition_ab_data[cond1])
                    mean2 = np.mean(condition_ab_data[cond2])
                    
                    print(f"  {cond1} vs {cond2}:")
                    print(f"    Means: {mean1:.3f} vs {mean2:.3f}")
                    print(f"    t-test: p={p_val:.4f} ({significance})")
                    print(f"    Effect size: {abs(mean1-mean2):.3f}")
        
        # ANOVA test if more than 2 conditions
        if len(condition_list) > 2:
            all_session_data = [condition_ab_data[cond] for cond in condition_list if len(condition_ab_data[cond]) > 0]
            if len(all_session_data) > 2:
                f_stat, p_val_anova = f_oneway(*all_session_data)
                significance = "SIGNIFICANT" if p_val_anova < 0.05 else "not significant"
                print(f"\nOverall ANOVA test:")
                print(f"  F-statistic: {f_stat:.3f}")
                print(f"  p-value: {p_val_anova:.4f} ({significance})")
        
    except ImportError:
        print("  scipy.stats not available for significance testing")
    
    # Recommendations for EEG classifier based on complete sessions
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR EEG CLASSIFIER (COMPLETE SESSIONS)")
    print(f"{'='*80}")
    
    # Calculate baseline from neutral condition
    if 'neutral' in conditions:
        neutral_sessions = [s for s in all_results.values() if s['condition'] == 'neutral']
        
        neutral_ab_ratios = []
        for session in neutral_sessions:
            session_avg_ab = np.mean(session['ratios']['alpha_beta'])
            neutral_ab_ratios.append(session_avg_ab)
        
        if neutral_ab_ratios:
            baseline_ab_ratio = np.mean(neutral_ab_ratios)
            baseline_std = np.std(neutral_ab_ratios)
            
            print(f"\nBaseline Calibration (from {len(neutral_sessions)} neutral sessions):")
            print(f"  Alpha/Beta ratio: {baseline_ab_ratio:.3f} ± {baseline_std:.3f}")
            print(f"  Coefficient of variation: {baseline_std/baseline_ab_ratio*100:.1f}%")
            
            # Calculate total neutral data duration
            total_neutral_duration = sum([s['file_info']['duration_min'] for s in neutral_sessions])
            print(f"  Total neutral data: {total_neutral_duration:.1f} minutes")
            
            print(f"\nSuggested Thresholds (based on complete sessions):")
            print(f"  Conservative approach (mean ± 1 SD):")
            print(f"    Relaxed detection: > {baseline_ab_ratio + baseline_std:.3f}")
            print(f"    Focused detection: < {baseline_ab_ratio - baseline_std:.3f}")
            
            print(f"  Moderate approach (±15% from baseline):")
            print(f"    Relaxed detection: > {baseline_ab_ratio * 1.15:.3f}")
            print(f"    Focused detection: < {baseline_ab_ratio * 0.85:.3f}")
            
            print(f"  Current EEG worker thresholds:")
            print(f"    Relaxed (α/β > baseline*1.1): {baseline_ab_ratio * 1.10:.3f}")
            print(f"    Focused (α/β < baseline*0.9): {baseline_ab_ratio * 0.90:.3f}")
            
            # Evaluate current thresholds with complete session data
            if 'relaxed' in conditions:
                relaxed_sessions = [s for s in all_results.values() if s['condition'] == 'relaxed']
                relaxed_ab_ratios = []
                for session in relaxed_sessions:
                    session_avg_ab = np.mean(session['ratios']['alpha_beta'])
                    relaxed_ab_ratios.append(session_avg_ab)
                
                if relaxed_ab_ratios:
                    relaxed_mean = np.mean(relaxed_ab_ratios)
                    current_threshold = baseline_ab_ratio * 1.10
                    
                    detection_rate = sum(1 for r in relaxed_ab_ratios if r > current_threshold) / len(relaxed_ab_ratios) * 100
                    
                    total_relaxed_duration = sum([s['file_info']['duration_min'] for s in relaxed_sessions])
                    
                    print(f"\nCurrent Threshold Performance (Relaxed Detection):")
                    print(f"  Relaxed sessions: {len(relaxed_sessions)} ({total_relaxed_duration:.1f} min total)")
                    print(f"  Relaxed condition mean: {relaxed_mean:.3f}")
                    print(f"  Current threshold: {current_threshold:.3f}")
                    print(f"  Detection rate: {detection_rate:.1f}%")
            
            if 'concentrating' in conditions:
                concentrating_sessions = [s for s in all_results.values() if s['condition'] == 'concentrating']
                concentrating_ab_ratios = []
                for session in concentrating_sessions:
                    session_avg_ab = np.mean(session['ratios']['alpha_beta'])
                    concentrating_ab_ratios.append(session_avg_ab)
                
                if concentrating_ab_ratios:
                    concentrating_mean = np.mean(concentrating_ab_ratios)
                    current_threshold = baseline_ab_ratio * 0.90
                    
                    detection_rate = sum(1 for r in concentrating_ab_ratios if r < current_threshold) / len(concentrating_ab_ratios) * 100
                    
                    total_concentrating_duration = sum([s['file_info']['duration_min'] for s in concentrating_sessions])
                    
                    print(f"\nCurrent Threshold Performance (Focus Detection):")
                    print(f"  Concentrating sessions: {len(concentrating_sessions)} ({total_concentrating_duration:.1f} min total)")
                    print(f"  Concentrating condition mean: {concentrating_mean:.3f}")
                    print(f"  Current threshold: {current_threshold:.3f}")
                    print(f"  Detection rate: {detection_rate:.1f}%")
    
    # Save comprehensive results
    save_comprehensive_results(all_results, band_defs)

def save_comprehensive_results(all_results: Dict, band_defs: Dict):
    """Save comprehensive results to CSV files"""
    
    # 1. Session-level summary
    session_summary = []
    
    for session_key, session in all_results.items():
        row = {
            'session_key': session_key,
            'subject': session['subject'],
            'condition': session['condition'],
            'trial': session['trial'],
            'filename': session['file_info']['filename'],
            'duration_min': session['file_info']['duration_min'],
            'total_samples': session['file_info']['total_samples'],
            'num_windows': session['file_info']['num_windows'],
            'window_size_sec': session['file_info']['window_size_sec']
        }
        
        # Add session-averaged band powers
        for band in band_defs.keys():
            row[f'{band}_power_mean'] = np.mean(session['band_powers'][band])
            row[f'{band}_power_std'] = np.std(session['band_powers'][band])
            row[f'{band}_power_min'] = np.min(session['band_powers'][band])
            row[f'{band}_power_max'] = np.max(session['band_powers'][band])
        
        # Add session-averaged ratios
        for ratio_name in session['ratios'].keys():
            row[f'{ratio_name}_mean'] = np.mean(session['ratios'][ratio_name])
            row[f'{ratio_name}_std'] = np.std(session['ratios'][ratio_name])
            row[f'{ratio_name}_min'] = np.min(session['ratios'][ratio_name])
            row[f'{ratio_name}_max'] = np.max(session['ratios'][ratio_name])
        
        # Add dominant frequency stats
        row['dominant_freq_mean'] = np.mean(session['dominant_freqs'])
        row['dominant_freq_std'] = np.std(session['dominant_freqs'])
        
        # Add dominant band percentages
        for band in band_defs.keys():
            count = session['dominant_bands'].count(band)
            percentage = count / len(session['dominant_bands']) * 100 if session['dominant_bands'] else 0
            row[f'dominant_{band}_percent'] = percentage
        
        session_summary.append(row)
    
    session_df = pd.DataFrame(session_summary)
    session_df.to_csv('complete_sessions_summary.csv', index=False)
    print(f"\nSession summary saved to 'complete_sessions_summary.csv'")
    
    # 2. Window-level detailed data
    window_data = []
    for session_key, session in all_results.items():
        for i in range(len(session['time_points'])):
            row = {
                'session_key': session_key,
                'subject': session['subject'],
                'condition': session['condition'],
                'trial': session['trial'],
                'window_index': i,
                'time_minutes': session['time_points'][i]
            }
            
            # Add band powers for this window
            for band in band_defs.keys():
                row[f'{band}_power'] = session['band_powers'][band][i]
            
            # Add ratios for this window
            for ratio_name in session['ratios'].keys():
                row[f'{ratio_name}'] = session['ratios'][ratio_name][i]
            
            # Add dominant info
            row['dominant_frequency'] = session['dominant_freqs'][i]
            row['dominant_band'] = session['dominant_bands'][i]
            
            window_data.append(row)
    
    window_df = pd.DataFrame(window_data)
    window_df.to_csv('complete_sessions_windows.csv', index=False)
    print(f"Window-level data saved to 'complete_sessions_windows.csv'")
    
    # 3. Condition summary statistics
    conditions = set([r['condition'] for r in all_results.values()])
    condition_summary = []
    
    for condition in sorted(conditions):
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        
        row = {
            'condition': condition,
            'num_sessions': len(condition_sessions),
            'total_duration_min': sum([s['file_info']['duration_min'] for s in condition_sessions]),
            'avg_duration_min': np.mean([s['file_info']['duration_min'] for s in condition_sessions]),
            'total_windows': sum([s['file_info']['num_windows'] for s in condition_sessions])
        }
        
        # Add average band powers across all sessions
        for band in band_defs.keys():
            all_powers = []
            for session in condition_sessions:
                all_powers.extend(session['band_powers'][band])
            
            if all_powers:
                row[f'{band}_power_mean'] = np.mean(all_powers)
                row[f'{band}_power_std'] = np.std(all_powers)
        
        # Add average ratios across all sessions
        for ratio_name in ['alpha_beta', 'beta_theta', 'theta_alpha']:
            all_ratios = []
            for session in condition_sessions:
                all_ratios.extend(session['ratios'][ratio_name])
            
            if all_ratios:
                row[f'{ratio_name}_mean'] = np.mean(all_ratios)
                row[f'{ratio_name}_std'] = np.std(all_ratios)
        
        condition_summary.append(row)
    
    condition_df = pd.DataFrame(condition_summary)
    condition_df.to_csv('complete_sessions_condition_summary.csv', index=False)
    print(f"Condition summary saved to 'complete_sessions_condition_summary.csv'")

def main():
    """Main analysis function"""
    
    print("Research Dataset Frequency Analysis - Complete Files Version")
    print("Analyzing ENTIRE session files (not just 6000:8000 range)")
    print("=" * 70)
    
    try:
        results = frequency_analysis_research_dataset()
        
        if results:
            print(f"\nAnalysis completed successfully!")
            print(f"Results for {len(results)} complete sessions saved.")
            
            # Calculate total data processed
            total_duration = sum([s['file_info']['duration_min'] for s in results.values()])
            total_hours = total_duration / 60
            
            print(f"\nData Summary:")
            print(f"  Total duration processed: {total_duration:.1f} minutes ({total_hours:.1f} hours)")
            print(f"  Average session duration: {total_duration/len(results):.1f} minutes")
            
            print(f"\nGenerated files:")
            print(f"  - subject_*_complete_analysis.png (individual subject plots)")
            print(f"  - complete_sessions_analysis.png (cross-subject comparison)")
            print(f"  - complete_sessions_summary.csv (session-level statistics)")
            print(f"  - complete_sessions_windows.csv (window-level data)")
            print(f"  - complete_sessions_condition_summary.csv (condition statistics)")
        else:
            print("No results obtained - check data path and file formats")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()