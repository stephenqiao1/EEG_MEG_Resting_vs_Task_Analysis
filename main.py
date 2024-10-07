import numpy as np
import mne
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from data_loading import load_edf
from preprocessing import clean_channel_names, preprocess_raw, perform_ica, inspect_raw_data, save_bad_channels, load_bad_channels, create_epochs, create_fixed_length_epochs
from analysis import compute_psd_epochs, compute_band_power, permutation_cluster_test
from visualization import plot_topomap_difference, plot_bar_with_significance

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

    # Create epochs
    epochs = {}
    # For resting state, use tmin=0, tmax=4, no baseline correction
    epochs['Rest Open'] = create_fixed_length_epochs(
        raw_data['Rest Open'], duration=4.0, overlap=0.0
    )
    epochs['Rest Closed'] = create_fixed_length_epochs(
        raw_data['Rest Closed'], duration=4.0, overlap=0.0
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
            print(f"Band power for {condition} in {band} band:")
            print(band_power)
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

# After statistical tests
# Collect all p-values
all_p_values = []
for band in stat_results:
    for cond_pair in stat_results[band]:
        p_vals = stat_results[band][cond_pair]['cluster_p_values']
        all_p_values.extend(p_vals)

# Apply FDR correction
reject, pvals_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')
print('Applied FDR correction to p-values.')

# Assign corrected p-values back to stat_results
idx = 0
for band in stat_results:
    for cond_pair in stat_results[band]:
        n_clusters = len(stat_results[band][cond_pair]['cluster_p_values'])
        stat_results[band][cond_pair]['cluster_p_values_corrected'] = pvals_corrected[idx:idx+n_clusters]
        stat_results[band][cond_pair]['significant_clusters_corrected'] = np.where(stat_results[band][cond_pair]['cluster_p_values_corrected'] < 0.05)[0]
        idx += n_clusters

# Plot topographic maps for each frequency band and condition pair
for band in frequency_bands:
    for (cond1, cond2) in condition_pairs:
        if cond1 in all_band_power_data[band] and cond2 in all_band_power_data[band]:
            title = f'Band Power Difference ({band.capitalize()}): {cond1} - {cond2}'
            mean_cond1 = all_band_power_data[band][cond1].mean(axis=0)  # Shape: (n_channels,)
            mean_cond2 = all_band_power_data[band][cond2].mean(axis=0)  # Shape: (n_channels,)
            
            # Check if there are significant clusters based on corrected p-values
            significant_clusters = stat_results[band][(cond1, cond2)]['significant_clusters_corrected']
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
            
            # Retrieve corrected p-values
            p_vals_corrected = stat_results[band][(cond1, cond2)]['cluster_p_values_corrected']
            significant = np.any(p_vals_corrected < 0.05)
            
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