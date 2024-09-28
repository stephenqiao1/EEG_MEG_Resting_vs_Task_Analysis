import mne 
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import os
from mne.preprocessing import ICA, create_eog_epochs
from mne.time_frequency import psd_array_multitaper
from collections import defaultdict
from mne.stats import permutation_cluster_test

# Set the path to the dataset directory
data_path = 'dataset/S001'

# Subject and run information
subject_number = 1
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

# Function to load EDF files
def load_edf(subject_number, run_number):
    """
    Load an EDF file for a given subject and run number
    
    Parameters:
    - subject_number; int, subject identifier
    - run_number: int, run identifier

    Returns:
    - raw: Raw EDF data object
    """
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
        # Remove periods
        clean_name = name.replace('.', '')
        # Use uppercase for comparison
        clean_name_upper = clean_name.upper()
        # Adjust case to match montage conventions
        if clean_name_upper == 'FP1':
            clean_name = 'Fp1'
        elif clean_name_upper == 'FP2':
            clean_name = 'Fp2'
        elif clean_name_upper == 'FPZ':
            clean_name = 'Fpz'
        elif 'Z' in clean_name_upper:
            idx = clean_name_upper.find('Z')
            # Capitalize letters before 'Z', use 'z' instead of 'Z', capitalize letters after 'Z'
            clean_name = clean_name_upper[:idx] + 'z' + clean_name_upper[(idx+1):]
        else:
            # Capitalize all letters
            clean_name = clean_name_upper
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
    raw.set_montage(montage)
    
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
    psds = 10 * np.log10(psds) # Convert power to decibels
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

# Load raw data
raw_data = {}
raw_data['Rest Open'] = load_edf(subject_number, runs['Rest Open'])
raw_data['Rest Closed'] = load_edf(subject_number, runs['Rest Closed'])
raw_data['Task'] = load_edf(subject_number, runs['Task'])

# Preprocess data and perform ICA
bad_channels = {
    'Rest Open': ['T8', 'T10'],
    'Rest Closed': ['FT7', 'FT8', 'T7', 'T8', 'T10'],
    'Task': ['Fp1', 'FT8', 'T8', 'T10', 'TP8']
}
for condition in raw_data:
    preprocess_raw(raw_data[condition], bad_channels[condition])
    raw_data[condition]= perform_ica(raw_data[condition])
    
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
        
# Analyze and visualize PSD results
# Initialize band_power_mean_data
band_power_mean_data = {}
# Compute average PSD across epochs
channel_name = 'Cz'
ch_index = epochs['Rest Open'].ch_names.index(channel_name)
for condition in psd_data:
    band_power_mean_data[condition] = {}
    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
    for band in psd_data[condition]:
        psds = psd_data[condition][band]['psds']  # Shape: (n_epochs, n_channels, n_freqs)
        freqs = psd_data[condition][band]['freqs']  # Array of frequency points
        
        # Compute band power and mean across epochs
        band_power = compute_band_power(psds)  # Shape: (n_epochs, n_channels)
        band_power_mean = np.mean(band_power, axis=0)  # Shape: (n_channels,)
        band_power_mean_data[condition][band] = band_power_mean
        
        # Compute mean PSD across epochs
        psd_mean = np.mean(psds, axis=0)  # Shape: (n_channels, n_freqs)
        
        # Plot the PSD at the selected channel
        plt.plot(freqs, psd_mean[ch_index, :], label=band)
    
    plt.title(f'PSD at {channel_name} during {condition}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.legend()
    plt.tight_layout()
    plt.show()
        
# Plot topographic maps
for condition in band_power_mean_data:
    for band in frequency_bands:
        band_power_mean = band_power_mean_data[condition][band]
        evoked = mne.EvokedArray(
            band_power_mean[:, np.newaxis],
            epochs['Rest Open'].info,
            tmin=0.0
        )
        fig = evoked.plot_topomap(
            times=0.0,
            scalings=1,
            time_format='',
            cmap='viridis',
            show=False
        )
        fig.suptitle(f'{band.capitalize()} Band Power - {condition}', fontsize=12)
        plt.show()