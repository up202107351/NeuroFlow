
"""
Frequency Content Analysis for Research Dataset - Corrected Version

Analyzes the frequency content of sessions from the research dataset.
Filters entire sessions once, then analyzes windows from the filtered data.
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
    Filter entire sessions once, then analyze windows
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
            
            # Load data
            df = pd.read_csv(file_path)
            print(f"  Raw shape: {df.shape}")
            
            # Extract EEG data (columns 1-4: TP9, AF7, AF8, TP10)
            # Use the 6000:8000 range like in your original analysis
            eeg_data = df.iloc[6000:8000, 1:5]
            eeg_data.columns = chans
            
            total_duration_sec = len(eeg_data) / fs
            print(f"  EEG segment: {len(eeg_data)} samples ({total_duration_sec:.1f} seconds)")
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
            window_sec = 2  # 2-second windows
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
                
                # Extract window from pre-filtered data
                window_powers = {band: [] for band in band_defs.keys()}
                window_psds = []
                
                for ch in chans:
                    # Use already filtered data - NO ADDITIONAL PROCESSING
                    window_data = filtered_data[ch][start_sample:end_sample]
                    
                    # Calculate PSD directly from filtered window
                    f, psd = welch(window_data, fs=fs, nperseg=min(len(window_data), int(fs)), 
                                  noverlap=int(fs*0.5), window="hann", detrend=None)  # No detrending
                    
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
                
                time_points.append(time_sec)
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
                    'num_windows': len(time_points)
                }
            }
            
            print(f"  ✓ Processed {len(time_points)} windows successfully")
            
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
        
        for condition in sorted(conditions):
            condition_sessions = {k: v for k, v in subject_sessions.items() if v['condition'] == condition}
            
            if not condition_sessions:
                continue
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Subject {subject.upper()} - {condition.upper()} - Detailed Analysis', fontsize=14)
            
            # Plot 1: Raw vs Filtered signals (first 2 seconds)
            ax = axes[0, 0]
            session = list(condition_sessions.values())[0]  # Take first session
            
            # Show first 2 seconds of data
            time_samples = np.arange(512) / 256.0  # 2 seconds
            
            for i, ch in enumerate(['TP9', 'AF7', 'AF8', 'TP10']):
                if ch in session['filtered_data']:
                    filtered_signal = session['filtered_data'][ch][:512]
                    ax.plot(time_samples, filtered_signal + i*50, label=ch, alpha=0.8)
            
            ax.set_title('Filtered EEG Signals (First 2s)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude (µV) + offset')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Band powers over time for all trials
            ax = axes[0, 1]
            
            for session_key, session in condition_sessions.items():
                trial_label = f"Trial {session['trial']}"
                
                for band in band_defs.keys():
                    ax.plot(session['time_points'], session['band_powers'][band], 
                           color=colors[band], alpha=0.7, linewidth=1.5,
                           label=f'{band}' if session_key == list(condition_sessions.keys())[0] else "")
            
            ax.set_title('Band Powers Over Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Power (log scale)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Alpha/Beta and Beta/Theta ratios
            ax = axes[0, 2]
            
            for session_key, session in condition_sessions.items():
                trial_label = f"Trial {session['trial']}"
                
                ax.plot(session['time_points'], session['ratios']['alpha_beta'], 
                       'g-', alpha=0.8, linewidth=2, 
                       label='Alpha/Beta' if session_key == list(condition_sessions.keys())[0] else "")
                ax.plot(session['time_points'], session['ratios']['beta_theta'], 
                       'b-', alpha=0.8, linewidth=2,
                       label='Beta/Theta' if session_key == list(condition_sessions.keys())[0] else "")
            
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
            ax.set_title('Key Ratios Over Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Average frequency spectrum for this condition
            ax = axes[1, 0]
            
            # Collect all PSDs for this condition
            all_psds = []
            for session in condition_sessions.values():
                all_psds.extend(session['psd_matrices'])
            
            if all_psds:
                avg_psd = np.mean(all_psds, axis=0)
                std_psd = np.std(all_psds, axis=0)
                frequencies = session['frequencies']  # Same for all sessions
                
                ax.semilogy(frequencies, avg_psd, 'k-', linewidth=2, label='Mean')
                ax.fill_between(frequencies, avg_psd - std_psd, avg_psd + std_psd, 
                               alpha=0.3, label='±1 SD')
            
            ax.set_xlim(0, 30)
            ax.set_title(f'Average Frequency Spectrum - {condition}')
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
            
            # Plot 5: Dominant frequency distribution
            ax = axes[1, 1]
            
            all_dominant_freqs = []
            all_dominant_bands = []
            
            for session in condition_sessions.values():
                all_dominant_freqs.extend(session['dominant_freqs'])
                all_dominant_bands.extend(session['dominant_bands'])
            
            if all_dominant_freqs:
                # Histogram of dominant frequencies
                ax.hist(all_dominant_freqs, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(all_dominant_freqs), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(all_dominant_freqs):.1f} Hz')
            
            ax.set_title('Dominant Frequency Distribution')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add band boundaries
            for band, (fl, fh) in band_defs.items():
                ax.axvspan(fl, fh, alpha=0.1, color=colors[band])
            
            # Plot 6: Band dominance pie chart
            ax = axes[1, 2]
            
            if all_dominant_bands:
                band_counts = {}
                for band in all_dominant_bands:
                    band_counts[band] = band_counts.get(band, 0) + 1
                
                # Only show bands that actually appear
                appearing_bands = [band for band in band_defs.keys() if band in band_counts]
                appearing_counts = [band_counts.get(band, 0) for band in appearing_bands]
                appearing_colors = [colors[band] for band in appearing_bands]
                
                if appearing_counts:
                    wedges, texts, autotexts = ax.pie(appearing_counts, labels=appearing_bands, 
                                                     colors=appearing_colors, autopct='%1.1f%%', 
                                                     startangle=90)
                    
                    # Make percentage text more readable
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
            
            ax.set_title('Dominant Band Distribution')
            
            plt.tight_layout()
            filename = f'subject_{subject}_{condition}_detailed_analysis.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {filename}")
    
    # 2. Cross-subject, cross-condition comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Cross-Subject and Cross-Condition Comparison', fontsize=16)
    
    # Plot 1: Average band powers by condition (all subjects)
    ax = axes[0, 0]
    
    condition_means = {}
    condition_stds = {}
    
    for condition in sorted(conditions):
        condition_means[condition] = {}
        condition_stds[condition] = {}
        
        for band in band_defs.keys():
            all_powers = []
            for session in all_results.values():
                if session['condition'] == condition:
                    all_powers.extend(session['band_powers'][band])
            
            if all_powers:
                condition_means[condition][band] = np.mean(all_powers)
                condition_stds[condition][band] = np.std(all_powers)
            else:
                condition_means[condition][band] = 0
                condition_stds[condition][band] = 0
    
    # Bar plot
    x = np.arange(len(band_defs))
    width = 0.25
    
    for i, condition in enumerate(sorted(conditions)):
        means = [condition_means[condition][band] for band in band_defs.keys()]
        stds = [condition_stds[condition][band] for band in band_defs.keys()]
        
        ax.bar(x + i*width, means, width, yerr=stds, label=condition, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Average Power (log scale)')
    ax.set_yscale('log')
    ax.set_title('Average Band Powers by Condition')
    ax.set_xticks(x + width)
    ax.set_xticklabels(band_defs.keys())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Alpha/Beta ratio comparison
    ax = axes[0, 1]
    
    ab_ratio_data = {condition: [] for condition in conditions}
    
    for session in all_results.values():
        condition = session['condition']
        ab_ratio_data[condition].extend(session['ratios']['alpha_beta'])
    
    # Box plot
    box_data = [ab_ratio_data[condition] for condition in sorted(conditions)]
    bp = ax.boxplot(box_data, labels=sorted(conditions), patch_artist=True)
    
    # Color the boxes
    condition_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], condition_colors[:len(conditions)]):
        patch.set_facecolor(color)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
    ax.set_title('Alpha/Beta Ratio Distribution by Condition')
    ax.set_ylabel('Alpha/Beta Ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Beta/Theta ratio comparison
    ax = axes[0, 2]
    
    bt_ratio_data = {condition: [] for condition in conditions}
    
    for session in all_results.values():
        condition = session['condition']
        bt_ratio_data[condition].extend(session['ratios']['beta_theta'])
    
    # Box plot
    box_data = [bt_ratio_data[condition] for condition in sorted(conditions)]
    bp = ax.boxplot(box_data, labels=sorted(conditions), patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], condition_colors[:len(conditions)]):
        patch.set_facecolor(color)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
    ax.set_title('Beta/Theta Ratio Distribution by Condition')
    ax.set_ylabel('Beta/Theta Ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Subject variability in Alpha/Beta ratio
    ax = axes[1, 0]
    
    for subject in sorted(subjects):
        subject_ab_means = []
        subject_conditions = []
        
        for condition in sorted(conditions):
            subject_sessions = [s for s in all_results.values() 
                              if s['subject'] == subject and s['condition'] == condition]
            
            if subject_sessions:
                all_ab_ratios = []
                for session in subject_sessions:
                    all_ab_ratios.extend(session['ratios']['alpha_beta'])
                
                if all_ab_ratios:
                    subject_ab_means.append(np.mean(all_ab_ratios))
                    subject_conditions.append(condition)
        
        if subject_ab_means:
            ax.plot(subject_conditions, subject_ab_means, 'o-', 
                   label=f'Subject {subject.upper()}', linewidth=2, markersize=8)
    
    ax.set_title('Subject Variability in Alpha/Beta Ratio')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Mean Alpha/Beta Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Dominant frequency by condition
    ax = axes[1, 1]
    
    dom_freq_data = {condition: [] for condition in conditions}
    
    for session in all_results.values():
        condition = session['condition']
        dom_freq_data[condition].extend(session['dominant_freqs'])
    
    # Box plot
    box_data = [dom_freq_data[condition] for condition in sorted(conditions)]
    bp = ax.boxplot(box_data, labels=sorted(conditions), patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], condition_colors[:len(conditions)]):
        patch.set_facecolor(color)
    
    ax.set_title('Dominant Frequency Distribution by Condition')
    ax.set_ylabel('Dominant Frequency (Hz)')
    ax.grid(True, alpha=0.3)
    
    # Add band boundaries
    for band, (fl, fh) in band_defs.items():
        ax.axhspan(fl, fh, alpha=0.1, color=colors[band])
        ax.text(len(conditions) + 0.2, (fl+fh)/2, band, fontsize=8, va='center')
    
    # Plot 6: Overall frequency spectra comparison
    ax = axes[1, 2]
    
    for condition in sorted(conditions):
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        
        all_psds = []
        for session in condition_sessions:
            all_psds.extend(session['psd_matrices'])
        
        if all_psds:
            avg_psd = np.mean(all_psds, axis=0)
            frequencies = session['frequencies']  # Same for all sessions
            
            ax.semilogy(frequencies, avg_psd, label=condition, linewidth=3)
    
    ax.set_xlim(0, 30)
    ax.set_title('Average Frequency Spectra by Condition')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add band boundaries
    for band, (fl, fh) in band_defs.items():
        if fh <= 30:
            ax.axvspan(fl, fh, alpha=0.1, color=colors[band])
    
    plt.tight_layout()
    plt.savefig('comprehensive_frequency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comprehensive_frequency_analysis.png")

def generate_detailed_statistics(all_results: Dict, band_defs: Dict):
    """Generate detailed statistical analysis"""
    
    print(f"\n{'='*80}")
    print("DETAILED FREQUENCY ANALYSIS STATISTICS")
    print(f"{'='*80}")
    
    # Organize data
    subjects = set([r['subject'] for r in all_results.values()])
    conditions = set([r['condition'] for r in all_results.values()])
    
    print(f"\nDataset Summary:")
    print(f"  Subjects: {len(subjects)} ({sorted(subjects)})")
    print(f"  Conditions: {len(conditions)} ({sorted(conditions)})")
    print(f"  Total sessions: {len(all_results)}")
    
    # Calculate session-level statistics (not window-level)
    print(f"\nSession-Level Statistics:")
    print(f"  (Each session contributes one data point - average of all windows)")
    
    session_stats = {}
    
    for condition in sorted(conditions):
        condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
        print(f"\n{condition.upper()}:")
        print(f"  Sessions: {len(condition_sessions)}")
        
        session_stats[condition] = {}
        
        # Calculate session-level averages
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
    
    # Per-subject analysis
    print(f"\nPer-Subject Analysis:")
    for subject in sorted(subjects):
        subject_sessions = [s for s in all_results.values() if s['subject'] == subject]
        print(f"\nSUBJECT {subject.upper()}:")
        
        for condition in sorted(conditions):
            condition_sessions = [s for s in subject_sessions if s['condition'] == condition]
            if not condition_sessions:
                continue
                
            print(f"  {condition}:")
            
            # Alpha/Beta ratio (key metric)
            ab_ratios = []
            for session in condition_sessions:
                session_avg_ab = np.mean(session['ratios']['alpha_beta'])
                ab_ratios.append(session_avg_ab)
            
            if ab_ratios:
                mean_ab = np.mean(ab_ratios)
                print(f"    Alpha/Beta ratio: {mean_ab:.3f}")
                
                # Classification hint based on alpha/beta ratio
                if mean_ab > 1.2:
                    hint = "HIGH (relaxed-like)"
                elif mean_ab < 0.8:
                    hint = "LOW (focused-like)"
                else:
                    hint = "NEUTRAL"
                print(f"    Classification hint: {hint}")
    
    # Statistical significance testing
    print(f"\nStatistical Significance Testing:")
    print(f"  (Comparing session-level averages)")
    
    try:
        from scipy.stats import ttest_ind, f_oneway
        
        # Test if there are significant differences between conditions
        for metric in ['alpha_beta_ratio', 'beta_theta_ratio']:
            print(f"\n{metric.upper().replace('_', '/')}:")
            
            condition_data = {}
            for condition in conditions:
                condition_sessions = [s for s in all_results.values() if s['condition'] == condition]
                
                session_averages = []
                for session in condition_sessions:
                    if metric == 'alpha_beta_ratio':
                        session_avg = np.mean(session['ratios']['alpha_beta'])
                    else:
                        session_avg = np.mean(session['ratios']['beta_theta'])
                    session_averages.append(session_avg)
                
                condition_data[condition] = session_averages
            
            # Pairwise t-tests
            condition_list = list(condition_data.keys())
            for i in range(len(condition_list)):
                for j in range(i+1, len(condition_list)):
                    cond1, cond2 = condition_list[i], condition_list[j]
                    
                    if len(condition_data[cond1]) > 0 and len(condition_data[cond2]) > 0:
                        t_stat, p_val = ttest_ind(condition_data[cond1], condition_data[cond2])
                        significance = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                        
                        mean1 = np.mean(condition_data[cond1])
                        mean2 = np.mean(condition_data[cond2])
                        
                        print(f"    {cond1} vs {cond2}:")
                        print(f"      Means: {mean1:.3f} vs {mean2:.3f}")
                        print(f"      t-test: p={p_val:.4f} ({significance})")
        
    except ImportError:
        print("    scipy.stats not available for significance testing")
    
    # Recommendations for EEG classifier
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR EEG CLASSIFIER")
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
            
            print(f"\nBaseline Calibration (from neutral sessions):")
            print(f"  Alpha/Beta ratio: {baseline_ab_ratio:.3f} ± {baseline_std:.3f}")
            print(f"  Coefficient of variation: {baseline_std/baseline_ab_ratio*100:.1f}%")
            
            print(f"\nSuggested Thresholds:")
            print(f"  Conservative approach:")
            print(f"    Relaxed detection: > {baseline_ab_ratio + baseline_std:.3f}")
            print(f"    Focused detection: < {baseline_ab_ratio - baseline_std:.3f}")
            
            print(f"  Moderate approach:")
            print(f"    Relaxed detection: > {baseline_ab_ratio * 1.15:.3f}")
            print(f"    Focused detection: < {baseline_ab_ratio * 0.85:.3f}")
            
            print(f"  Current EEG worker thresholds:")
            print(f"    Relaxed (α/β > baseline*1.1): {baseline_ab_ratio * 1.10:.3f}")
            print(f"    Focused (α/β < baseline*0.9): {baseline_ab_ratio * 0.90:.3f}")
            
            # Evaluate current thresholds
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
                    
                    print(f"\nCurrent Threshold Performance:")
                    print(f"  Relaxed condition mean: {relaxed_mean:.3f}")
                    print(f"  Current threshold: {current_threshold:.3f}")
                    print(f"  Detection rate: {detection_rate:.1f}%")
    
    # Save detailed session-level results
    session_level_data = []
    
    for session_key, session in all_results.items():
        row = {
            'session_key': session_key,
            'subject': session['subject'],
            'condition': session['condition'],
            'trial': session['trial'],
            'duration_sec': session['file_info']['duration_sec'],
            'num_windows': session['file_info']['num_windows']
        }
        
        # Add session-averaged band powers
        for band in band_defs.keys():
            row[f'{band}_power_mean'] = np.mean(session['band_powers'][band])
            row[f'{band}_power_std'] = np.std(session['band_powers'][band])
        
        # Add session-averaged ratios
        for ratio_name in session['ratios'].keys():
            row[f'{ratio_name}_mean'] = np.mean(session['ratios'][ratio_name])
            row[f'{ratio_name}_std'] = np.std(session['ratios'][ratio_name])
        
        # Add dominant frequency stats
        row['dominant_freq_mean'] = np.mean(session['dominant_freqs'])
        row['dominant_freq_std'] = np.std(session['dominant_freqs'])
        
        session_level_data.append(row)
    
    # Save to CSV
    session_df = pd.DataFrame(session_level_data)
    session_df.to_csv('frequency_analysis_session_level.csv', index=False)
    print(f"\nSession-level results saved to 'frequency_analysis_session_level.csv'")
    
    # Save window-level data for detailed analysis
    window_level_data = []
    for session_key, session in all_results.items():
        for i in range(len(session['time_points'])):
            row = {
                'session_key': session_key,
                'subject': session['subject'],
                'condition': session['condition'],
                'trial': session['trial'],
                'window_index': i,
                'time_point': session['time_points'][i]
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
            
            window_level_data.append(row)
    
    window_df = pd.DataFrame(window_level_data)
    window_df.to_csv('frequency_analysis_window_level.csv', index=False)
    print(f"Window-level results saved to 'frequency_analysis_window_level.csv'")

def main():
    """Main analysis function"""
    
    print("Research Dataset Frequency Analysis - Corrected Version")
    print("Filter entire sessions once, then analyze windows")
    print("=" * 60)
    
    try:
        results = frequency_analysis_research_dataset()
        
        if results:
            print(f"\nAnalysis completed successfully!")
            print(f"Results for {len(results)} sessions saved.")
            print(f"\nGenerated files:")
            print(f"  - subject_*_*_detailed_analysis.png (individual session plots)")
            print(f"  - comprehensive_frequency_analysis.png (comparison plots)")
            print(f"  - frequency_analysis_session_level.csv (session-averaged data)")
            print(f"  - frequency_analysis_window_level.csv (window-by-window data)")
        else:
            print("No results obtained - check data path and file formats")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()