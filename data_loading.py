import os
import mne

# Set the base path to the dataset directory
base_data_path = 'dataset'

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
    # raw.plot(block=True)
    return raw