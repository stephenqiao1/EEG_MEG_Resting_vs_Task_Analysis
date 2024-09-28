import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_multitaper
from collections import defaultdict
from mne.stats import permutation_cluster_test
import json

# Set the base path to the dataset directory
base_data_path = 'dataset'

# List of subject numbers to process
subjects = [1, 2, 3]  # Update this list with your subject numbers

# Run information
runs = {
    'Rest Open': 1,
    'Rest Closed': 2,
    'Task': 3
}

# Frequency bands for analysis
frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Define all conditions explicitly
all_conditions = [
    'Rest Open',
    'Rest Closed',
    'Task Rest',
    'Left Fist',
    'Right Fist'
]


# Function to load EDF files
def load_edf(subject_number, run_number):
    """
    Load an EDF file for a given subject and run number.
    
    Parameters:
    - subject_number: int, subject identifier
    - run_number: int, run identifier

    Returns:
    - raw: Raw EDF data object
    """
    data_path = os.path.join(base_data_path, f'S{subject_number:03d}')
    file_name = f'S{subject_number:03d}R{run_number:02d}.edf'
    file_path = os.path.join(data_path, file_name)
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto')
    return raw

# Function to clean up channel names
def clean_channel_names(raw):
    """
    Clean and standardize channel names to match MNE-Python conventions.
    
    Parameters:
    - raw: Raw data object
    """
    new_names = {}
    for name in raw.ch_names:
        # Remove periods and whitespace
        clean_name = name.replace('.', '').replace(' ', '')
        # Make all letters uppercase
        clean_name = clean_name.upper()
        # Special handling for 'FP1', 'FP2', 'FPZ'
        if clean_name.startswith('FP'):
            clean_name = 'Fp' + clean_name[2:]
        # Handle 'Z' in channel names (e.g., 'CZ' -> 'Cz')
        elif 'Z' in clean_name:
            idx = clean_name.find('Z')
            clean_name = clean_name[:idx] + 'z' + clean_name[idx+1:]
        # No change needed for other channels
        new_names[name] = clean_name
    raw.rename_channels(new_names)
        
# Function to preprocess raw data
def preprocess_raw(raw, bad_channels):
    """
    Preprocess raw data by setting montage, filtering, and interpolating bad channels.

    Parameters:
    - raw: Raw data object
    - bad_channels: list of str, names of bad channels
    """
    # Clean channel names
    clean_channel_names(raw)
    
    # Set montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, match_case=False)
    
    # Filter data
    raw.filter(l_freq=1., h_freq=40.)
    
    # Mark and interpolate bad channels
    raw.info['bads'].extend(bad_channels)
    raw.interpolate_bads(reset_bads=True)
    
# Function to perform ICA and remove EOG artifacts
def perform_ica(raw, n_components=25, eog_channel='Fp1'):
    """
    Perform ICA on raw data to identify and remove EOG artifacts.

    Parameters:
    - raw: Raw data object
    - n_components: int, number of ICA components
    - eog_channel: str, channel used as surrogate EOG channel

    Returns:
    - raw: Raw data object with ICA applied
    """
    ica = ICA(n_components=n_components, method='fastica', random_state=97)
    ica.fit(raw, picks='eeg')
    eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channel)
    ica.exclude.extend(eog_inds)
    raw = ica.apply(raw)
    return raw

# Function to create epochs from raw data
def create_epochs(raw, tmin, tmax, baseline, event_id_filter=None):
    """
    Create epochs from raw data based on annotations.

    Parameters:
    - raw: Raw data object
    - tmin: float, start time before event
    - tmax: float, end time after event
    - baseline: tuple or None, baseline correction
    - event_id_filter: list or None, events to include

    Returns:
    - epochs: Epochs object 
    """
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True
    )
    if event_id_filter is not None:
        epochs = epochs[event_id_filter]
    return epochs

# Function to compute PSD for epochs
def compute_psd_epochs(epochs, fmin, fmax):
    """
    Compute Power Spectral Density (PSD) for given epochs and frequency band.

    Parameters:
    - epochs: Epochs object
    - fmin: float, minimum frequency
    - fmax: float, maximum frequency

    Returns:
    - psds: ndarray, PSD values
    - freqs: ndarray, frequency values
    """
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = data.shape
    psds = []
    for epoch in data:
        psd, freqs = psd_array_multitaper(
            epoch, sfreq=sfreq, fmin=fmin, fmax=fmax,
            adaptive=True, normalization='full', verbose=0
        )
        psds.append(psd)
    psds = np.array(psds)
    psds = 10 * np.log10(psds)  # Convert power to decibels
    return psds, freqs

# Function to compute band power
def compute_band_power(psds):
    """
    Compute band power by summing PSD values across frequencies.

    Parameters:
    - psds: ndarray, PSD values

    Returns:
    - band_power: ndarray, band power values
    """
    band_power = np.sum(psds, axis=2)
    return band_power

# Function for visual inspection to identify bad channels
def inspect_raw_data(raw):
    """
    Launch an interactive plot to inspect raw EEG data and mark bad channels.

    Parameters:
    - raw: Raw data object
    """
    raw.plot(
        scalings='auto',  # Adjust scaling automatically
        n_channels=30,    # Number of channels to display at once
        block=True        # Block execution until the plot window is closed
    )
    # After closing the plot, raw.info['bads'] will contain the marked bad channels

# Functions to save and load bad channels
def save_bad_channels(subject_number, bad_channels):
    """
    Save bad channels to a JSON file for a given subject.

    Parameters:
    - subject_number: int, subject identifier
    - bad_channels: dict, bad channels for each condition
    """
    file_path = os.path.join('bad_channels', f'S{subject_number:03d}_bad_channels.json')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(bad_channels, f)

def load_bad_channels(subject_number):
    """
    Load bad channels from a JSON file for a given subject.

    Parameters:
    - subject_number: int, subject identifier

    Returns:
    - bad_channels: dict, bad channels for each condition
    """
    file_path = os.path.join('bad_channels', f'S{subject_number:03d}_bad_channels.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            bad_channels = json.load(f)
        return bad_channels
    else:
        return {}

# Initialize data structures for storing results across subjects
all_band_power_data = defaultdict(lambda: defaultdict(list))
info = None  # To store the info structure from the first subject

# Loop over subjects
for subject_number in subjects:
    print(f'Processing Subject {subject_number}')
    # Load raw data
    raw_data = {}
    try:
        raw_data['Rest Open'] = load_edf(subject_number, runs['Rest Open'])
        raw_data['Rest Closed'] = load_edf(subject_number, runs['Rest Closed'])
        raw_data['Task'] = load_edf(subject_number, runs['Task'])
    except FileNotFoundError as e:
        print(f'File not found for subject {subject_number}: {e}')
        continue  # Skip this subject if data is missing

    # Load bad channels if previously saved
    bad_channels = load_bad_channels(subject_number)

    # Preprocess data and perform ICA
    for condition in raw_data:
        # Clean channel names
        clean_channel_names(raw_data[condition])

        # If bad channels are not loaded, perform visual inspection
        if condition not in bad_channels:
            print(f'Inspecting {condition} data for Subject {subject_number}')
            inspect_raw_data(raw_data[condition])
            # Store the bad channels marked during inspection
            bad_channels[condition] = raw_data[condition].info['bads']
            # Save bad channels after inspection
            save_bad_channels(subject_number, bad_channels)
        else:
            # Set the bad channels from the loaded list
            raw_data[condition].info['bads'] = bad_channels[condition]

        # Preprocess raw data
        preprocess_raw(raw_data[condition], bad_channels[condition])

        # Perform ICA
        raw_data[condition] = perform_ica(raw_data[condition])

    # Create epochs
    epochs = {}
    # For resting state, use tmin=0, tmax=4, no baseline correction
    epochs['Rest Open'] = create_epochs(
        raw_data['Rest Open'], tmin=0, tmax=4, baseline=None, event_id_filter='T0'
    )
    epochs['Rest Closed'] = create_epochs(
        raw_data['Rest Closed'], tmin=0, tmax=4, baseline=None, event_id_filter='T0'
    )
    # For task, use tmin=0, tmax=4, no baseline correction
    epochs_task = create_epochs(
        raw_data['Task'], tmin=0, tmax=4, baseline=None
    )
    # Separate task epochs
    epochs['Task Rest'] = epochs_task['T0']
    epochs['Left Fist'] = epochs_task['T1']
    epochs['Right Fist'] = epochs_task['T2']

    # Compute PSD for each condition and frequency band
    psd_data = {}
    for condition in epochs:
        psd_data[condition] = {}
        for band_name, (fmin, fmax) in frequency_bands.items():
            psds, freqs = compute_psd_epochs(epochs[condition], fmin, fmax)
            psd_data[condition][band_name] = {
                'psds': psds,
                'freqs': freqs
            }

    # Compute band power for statistical analysis
    for condition in psd_data:
        for band in psd_data[condition]:
            psds = psd_data[condition][band]['psds']
            band_power = compute_band_power(psds)  # Shape: (n_epochs, n_channels)
            # Average over epochs for this subject
            band_power_mean = np.mean(band_power, axis=0)  # Shape: (n_channels,)
            # Append to the list for this band and condition
            all_band_power_data[band][condition].append(band_power_mean)

    # Store info structure for later use
    if info is None:
        info = epochs['Rest Open'].info

# Convert lists to NumPy arrays
for band in all_band_power_data:
    for condition in all_band_power_data[band]:
        all_band_power_data[band][condition] = np.array(all_band_power_data[band][condition])  # Shape: (n_subjects, n_channels)

# Statistical Analysis and Visualization

# Define condition pairs for comparison
condition_pairs = [
    ('Rest Open', 'Rest Closed'),
    ('Task Rest', 'Rest Closed'),
    ('Left Fist', 'Right Fist')
]

stat_results = {}

# Perform statistical tests for each frequency band and condition pair
for band in frequency_bands:
    stat_results[band] = {}
    for (cond1, cond2) in condition_pairs:
        if cond1 in all_band_power_data[band] and cond2 in all_band_power_data[band]:
            # Extract data: Shape (n_subjects, n_channels)
            data_cond1 = all_band_power_data[band][cond1]
            data_cond2 = all_band_power_data[band][cond2]
            
            # Stack data into a list for permutation_cluster_test
            X = [data_cond1, data_cond2]
            
            # Perform cluster-based permutation test with adjusted parameters
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                X, n_permutations=5000, tail=0, threshold=2.0, n_jobs=1, seed=42
            )
            
            # Identify significant clusters
            significant_cluster_indices = np.where(cluster_p_values < 0.05)[0]
            
            # Store statistical results
            stat_results[band][(cond1, cond2)] = {
                'T_obs': T_obs,
                'clusters': clusters,
                'cluster_p_values': cluster_p_values,
                'significant_clusters': significant_cluster_indices
            }

# Function to plot topographic map of differences
def plot_topomap_difference(info, mean_power_cond1, mean_power_cond2, title):
    """
    Plot the topographic map of band power differences between two conditions.
    
    Parameters:
    - info: MNE Info object containing channel locations
    - mean_power_cond1: ndarray, mean band power for condition 1 (n_channels,)
    - mean_power_cond2: ndarray, mean band power for condition 2 (n_channels,)
    - title: str, title of the plot
    """
    # Compute difference
    power_diff = mean_power_cond1 - mean_power_cond2  # Shape: (n_channels,)
    
    # Create an EvokedArray for plotting
    evoked_diff = mne.EvokedArray(
        power_diff[:, np.newaxis],
        info,
        tmin=0.0
    )
    
    # Plot topomap
    fig, ax = plt.subplots(figsize=(8, 6))
    im, cn = mne.viz.plot_topomap(
        evoked_diff.data[:, 0], info, axes=ax, show=False, cmap='RdBu_r',
        contours=0
    )
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()

# Function to plot bar plots with statistical significance
def plot_bar_with_significance(cond1, mean1, sem1, cond2, mean2, sem2, significant, title):
    """
    Plot bar plots of average band power for two conditions with significance asterisks.
    
    Parameters:
    - cond1: str, name of condition 1
    - mean1: float, mean band power for condition 1
    - sem1: float, SEM for condition 1
    - cond2: str, name of condition 2
    - mean2: float, mean band power for condition 2
    - sem2: float, SEM for condition 2
    - significant: bool, whether the difference is significant
    - title: str, title of the plot
    """
    conditions = [cond1, cond2]
    means = [mean1, mean2]
    sems = [sem1, sem2]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(conditions, means, yerr=sems, capsize=5, color=['skyblue', 'salmon'])
    
    if significant:
        # Add asterisk above the bars
        y_max = max(means) + max(sems) + 0.05 * max(means)
        ax.text(
            0.5, y_max, '*', ha='center', va='bottom', fontsize=20, color='k'
        )
    
    ax.set_ylabel('Average Band Power (dB)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# Prepare mean and SEM data
mean_data = defaultdict(dict)
sem_data = defaultdict(dict)

for band in frequency_bands:
    for condition in all_conditions:
        if condition in all_band_power_data[band]:
            # Compute mean and SEM
            mean = all_band_power_data[band][condition].mean(axis=0).mean()  # Average across channels and subjects
            sem = all_band_power_data[band][condition].mean(axis=0).std() / np.sqrt(len(subjects))
            mean_data[band][condition] = mean
            sem_data[band][condition] = sem

# Plot topographic maps for each frequency band and condition pair
for band in frequency_bands:
    for (cond1, cond2) in condition_pairs:
        if cond1 in all_band_power_data[band] and cond2 in all_band_power_data[band]:
            title = f'Band Power Difference ({band.capitalize()}): {cond1} - {cond2}'
            mean_cond1 = all_band_power_data[band][cond1].mean(axis=0)  # Shape: (n_channels,)
            mean_cond2 = all_band_power_data[band][cond2].mean(axis=0)  # Shape: (n_channels,)
            
            # Check if there are significant clusters
            significant_clusters = stat_results[band][(cond1, cond2)]['significant_clusters']
            if len(significant_clusters) == 0:
                print(f'No significant clusters found for {band} band between {cond1} and {cond2}. Skipping topomap.')
                continue
            
            plot_topomap_difference(
                info,
                mean_power_cond1=mean_cond1,
                mean_power_cond2=mean_cond2,
                title=title
            )

# Plot bar plots with significance
for band in frequency_bands:
    for (cond1, cond2) in condition_pairs:
        if band in stat_results and (cond1, cond2) in stat_results[band]:
            title = f'Average {band.capitalize()} Band Power: {cond1} vs {cond2}'
            mean_cond1 = mean_data[band][cond1]
            sem_cond1 = sem_data[band][cond1]
            mean_cond2 = mean_data[band][cond2]
            sem_cond2 = sem_data[band][cond2]
            
            significant = np.any(stat_results[band][(cond1, cond2)]['cluster_p_values'] < 0.05)
            
            plot_bar_with_significance(
                cond1,
                mean_cond1,
                sem_cond1,
                cond2,
                mean_cond2,
                sem_cond2,
                significant,
                title
            )