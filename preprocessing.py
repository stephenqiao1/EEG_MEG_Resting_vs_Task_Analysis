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
    
    # # For debugging purposes
    # print(eog_inds)
    # ica.plot_components(raw, picks=eog_inds)
    
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
    
    # Print events and event IDs to verify extraction
    print("\nExtracted Events:")
    print(events)
    print("\nEvent IDs:")
    print(event_id)
    
    # Check if any events were found
    if len(events) == 0:
        print("No events found in the raw data.")
        return None
    
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True
    )
    
    # If specific event IDs are to be selected
    if event_id_filter is not None:
        # Ensure that event_id_filter is a list
        if isinstance(event_id_filter, str):
            event_id_filter = [event_id_filter]
        # Check if the specified event IDs exist
        valid_event_ids = [eid for eid in event_id_filter if eid in event_id]
        if not valid_event_ids:
            print(f"No matching event IDs found for filter {event_id_filter}.")
            return None
        # Select epochs with valid event IDs
        epochs = epochs[valid_event_ids]

    # Print the number of epochs created
    print(f"\nNumber of epochs created: {len(epochs)}")
    print(f"Epochs info:\n{epochs}")
    
    return epochs
    
def create_fixed_length_epochs(raw, duration, overlap=0.0):
    """
    Create fixed-length epochs from continuous raw data.

    Parameters:
    - raw: Raw data object
    - duration: float, length of each epoch in seconds
    - overlap: float, overlap between epochs in seconds

    Returns:
    - epochs: Epochs object
    """
    epochs = mne.make_fixed_length_epochs(
        raw, duration=duration, overlap=overlap, preload=True
    )
    print(f"\nNumber of fixed-length epochs created: {len(epochs)}")
    print(f"Epochs info:\n{epochs}")
    return epochs