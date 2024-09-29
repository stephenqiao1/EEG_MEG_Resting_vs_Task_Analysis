import os
import json
import mne
from mne.preprocessing import ICA

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