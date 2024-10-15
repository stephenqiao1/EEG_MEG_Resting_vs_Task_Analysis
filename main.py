import numpy as np
import mne
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from data_loading import load_edf
from preprocessing import (clean_channel_names, preprocess_raw, perform_ica,
                           inspect_raw_data, save_bad_channels, load_bad_channels,
                           create_epochs, create_fixed_length_epochs)
from analysis import compute_psd_epochs, compute_band_power, permutation_cluster_test
from visualization import plot_topomap_difference, plot_bar_with_significance
import matplotlib.pyplot as plt

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

        # Update the info object to include only good channels
        raw_data[condition].pick_types(eeg=True, exclude='bads')

    # Store info structure for later use
    if info is None:
        info = raw_data['Rest Open'].info.copy()

        # Ensure that the montage is set
        if not info['chs'][0]['loc'].any():
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)

    # Create epochs
    epochs = {}
    # For resting state, use fixed-length epochs
    epochs['Rest Open'] = create_fixed_length_epochs(
        raw_data['Rest Open'], duration=4.0, overlap=0.0
    )
    epochs['Rest Closed'] = create_fixed_length_epochs(
        raw_data['Rest Closed'], duration=4.0, overlap=0.0
    )
    # For task, use event-based epochs
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
            print(f"Band power for {condition} in {band} band:")
            print(band_power)
            # Average over epochs for this subject
            band_power_mean = np.mean(band_power, axis=0)  # Shape: (n_channels,)
            # Append to the list for this band and condition
            all_band_power_data[band][condition].append(band_power_mean)

# Convert lists to NumPy arrays
for band in all_band_power_data:
    for condition in all_band_power_data[band]:
        all_band_power_data[band][condition] = np.array(all_band_power_data[band][condition])  # Shape: (n_subjects, n_channels)

        # Debugging
        data = all_band_power_data[band][condition]
        mean_power = data.mean()
        std_power = data.std()
        print(f"Mean band power for {condition} in {band} band: {mean_power}")
        print(f"STD band power for {condition} in {band} band: {std_power}")

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
                X, n_permutations=5000, tail=0, threshold=None, n_jobs=1, seed=42
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

            # Check for NaNs or Infs in mean_cond1 and mean_cond2
            if np.isnan(mean_cond1).any() or np.isinf(mean_cond1).any():
                print(f"mean_cond1 contains invalid values for {band} band between {cond1} and {cond2}.")
                continue  # Skip plotting for this pair
            if np.isnan(mean_cond2).any() or np.isinf(mean_cond2).any():
                print(f"mean_cond2 contains invalid values for {band} band between {cond1} and {cond2}.")
                continue  # Skip plotting for this pair

            # Plot the topomap difference with adjusted color scales
            plot_topomap_difference(
                info,
                mean_power_cond1=mean_cond1,
                mean_power_cond2=mean_cond2,
                title=title,
                cmap='RdBu_r'  # Red-blue colormap reversed for divergent data
            )

# Plot individual subject data
for band in frequency_bands:
    for condition in all_conditions:
        if condition in all_band_power_data[band]:
            # Extract data: Shape (n_subjects, n_channels)
            data = all_band_power_data[band][condition]

            # Compute global vmin and vmax for this band and condition
            vmin = data.min()
            vmax = data.max()

            for subject_idx in range(len(subjects)):
                band_power = data[subject_idx]  # Shape: (n_channels,)
                subject_number = subjects[subject_idx]
                title = f'Subject {subject_number} - {condition} - {band.capitalize()} Band Power'

                # Check for NaNs or Infs in band_power
                if np.isnan(band_power).any() or np.isinf(band_power).any():
                    print(f"Band power contains invalid values for Subject {subject_number}, {condition}, {band} band.")
                    continue  # Skip plotting for this subject

                # Plot topomap for individual subject
                fig, ax = plt.subplots()
                im, cn = mne.viz.plot_topomap(
                    band_power,
                    info,
                    axes=ax,
                    show=False,
                    cmap='viridis',  # Choose an appropriate colormap
                    sensors=True
                )
                fig.colorbar(im, ax=ax)
                ax.set_title(title)
                plt.tight_layout()
                # Save and close the figure inside the loop
                fig.savefig(f'subject_{subject_number}_{condition}_{band}.png')
                plt.close(fig)
