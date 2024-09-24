import mne 
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import os
from mne.preprocessing import ICA, create_eog_epochs

# Set the path to the dataset directory
data_path = 'dataset/S001'

# Function to load EDF files
def load_edf(subject_number, run_number):
    file_name = f'S{subject_number:03d}R{run_number:02d}.edf'
    file_path = os.path.join(data_path, file_name)
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto')
    return raw

# Clean up Channel Names
def clean_channel_names(raw):
    new_names = {}
    print("Original channel names:")
    print(raw.ch_names)
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
    print("Cleaned channel names:")
    print(raw.ch_names)

subject_number = 1 

# Separate Analysis for Each Resting State
# Load resting state runs separately 
raw_rest_open = load_edf(subject_number, 1) # Eyes open
raw_rest_closed = load_edf(subject_number, 2) # Eyes closed

# Load Task Runs
# Run 3: Open and close left or right fist
raw_task = load_edf(subject_number, 3)

# Plot raw data
# raw_rest_open.plot(scalings='auto', show=True, block=True) 
# raw_rest_closed.plot(scalings='auto', show=True, block=True)
# raw_task.plot(scalings='auto', show=True, block=True)

# Preprocess each run separately
# Set montage, filter, remove artifacts for raw_rest_open and raw_rest_closed individually

# Eyes open run
montage = mne.channels.make_standard_montage('standard_1005')
    
clean_channel_names(raw_rest_open)
clean_channel_names(raw_rest_closed)
clean_channel_names(raw_task)

raw_rest_open.set_montage(montage)
raw_rest_closed.set_montage(montage)
raw_task.set_montage(montage)

raw_rest_open.filter(l_freq=1., h_freq=40.) # Filtering
raw_rest_closed.filter(l_freq=1., h_freq=40.)
raw_task.filter(l_freq=1., h_freq=40.)

# raw_rest_open.plot(scalings='auto', n_channels=64, duration=10, block=True) # Visually inspect and remove the bad channels
# raw_rest_closed.plot(scalings='auto', n_channels=64, duration=10, block=True)
# raw_task.plot(scalings='auto', n_channels=64, duration=10, block=True)

raw_rest_open.info['bads'].extend(['T8', 'T10'])
raw_rest_open.interpolate_bads(reset_bads=True)

raw_rest_closed.info['bads'].extend(['FT7', 'FT8', 'T7', 'T8', 'T10'])
raw_rest_closed.interpolate_bads(reset_bads=True)

raw_task.info['bads'].extend(['Fp1', 'FT8', 'T8', 'T10', 'TP8'])
raw_task.interpolate_bads(reset_bads=True)

# raw_rest_open.plot(scalings='auto', n_channels=64, duration=10, block=True)
# raw_rest_closed.plot(scalings='auto', n_channels=64, duration=10, block=True)
# raw_task.plot(scalings='auto', n_channels=64, duration=10, block=True)

# Fit ICA
ica = ICA(n_components=25, method='fastica', random_state=97)
ica.fit(raw_rest_open, picks='eeg')

ica_closed = ICA(n_components=25, method='fastica', random_state=97)
ica_closed.fit(raw_rest_closed, picks='eeg')

ica_task = ICA(n_components=25, method='fastica', random_state=97)
ica_task.fit(raw_task, picks='eeg')

# Use 'Fp1' as a surrogate EOG channel
eog_channel = 'Fp1'
eog_inds, eog_scores = ica.find_bads_eog(raw_rest_open, ch_name=eog_channel)
eog_inds_closed, eog_scores_closed = ica_closed.find_bads_eog(raw_rest_closed, ch_name=eog_channel)
eog_inds_task, eog_scores_task = ica_task.find_bads_eog(raw_task, ch_name=eog_channel)

# Mark components for exclusion
ica.exclude.extend(eog_inds)
ica_closed.exclude.extend(eog_inds_closed)
ica_task.exclude.extend(eog_inds_task)

# Visualize components
# ica.plot_scores(eog_scores) # To visualize how strongly each ICA component correlates with the EOG signal.
# ica.plot_components(picks=eog_inds) # To display the spatial topographies (scalp maps) of the ICA components identified as likely representing EOG artifacts.
# ica.plot_properties(raw_rest_open, picks=eog_inds) # To provide a detailed overview of the properties of the specified ICA components in the context of the raw data.

# ica_closed.plot_scores(eog_scores_closed)
# ica_closed.plot_components(picks=eog_inds_closed)
# ica.plot_properties(raw_rest_closed, picks=eog_inds_closed)

# ica_task.plot_scores(eog_scores_task)
# ica_task.plot_components(picks=eog_inds_task)
# ica_task.plot_properties(raw_task, picks=eog_inds_task)

# Apply ICA correction
ica.exclude.append(eog_inds[0]) # Remove ICA component 000
raw_rest_open = ica.apply(raw_rest_open)

ica_closed.exclude.append(eog_inds_closed[0])
raw_rest_closed = ica_closed.apply(raw_rest_closed)

ica_task.exclude.append(eog_inds_task[0])
raw_task = ica_task.apply(raw_task)

# Epoching for eyes open
# Extract events from annotations
events_rest_open, event_id_rest_open = mne.events_from_annotations(raw_rest_open)
events_rest_closed, event_id_rest_closed = mne.events_from_annotations(raw_rest_closed)
events_task, event_id_task = mne.events_from_annotations(raw_task)

print("Event IDs for task run:", event_id_task)

# Define tmin and tmax relative to events
tmin, tmax = 0, 4 # Full duration of each trial
tmin_task, tmax_task = -1, 4 

# Create epochs for resting open state 
epochs_rest_open = mne.Epochs(raw_rest_open, events_rest_open, event_id=event_id_rest_open, tmin=tmin, tmax=tmax, baseline=None, preload=True)
epochs_rest_closed = mne.Epochs(raw_rest_closed, events_rest_closed, event_id=event_id_rest_closed, tmin=tmin, tmax=tmax, baseline=None, preload=True)
epochs_task = mne.Epochs(raw_task, events_task, event_id=event_id_task, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True) # We included a baseline from tmin to 

epochs_rest_open = epochs_rest_open['T0']
epochs_rest_closed = epochs_rest_closed['T0']
epochs_task_rest = epochs_task['T0']
epochs_task_left = epochs_task['T1']
epochs_task_right = epochs_task['T2']


