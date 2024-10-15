import matplotlib.pyplot as plt
import mne
import numpy as np
def plot_topomap_difference(info, mean_power_cond1, mean_power_cond2, title, cmap='RdBu_r'):
    difference = mean_power_cond1 - mean_power_cond2  # Shape: (n_channels,)

    # Compute vmin and vmax to center the color scale around zero
    max_abs = np.max(np.abs(difference))
    vmin, vmax = -max_abs, max_abs

    # Ensure that the montage is set in info
    if not np.any(info['chs'][0]['loc']):
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

    # Ensure that channels match between difference and info
    picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
    data_channels = [info['ch_names'][i] for i in picks]
    if len(difference) != len(data_channels):
        raise ValueError("Mismatch between number of channels in data and info.")

    # Check for NaN or infinite values
    if np.isnan(difference).any() or np.isinf(difference).any():
        raise ValueError("Data contains NaN or infinite values.")

    # Create the figure
    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(
        difference,
        pos=info,
        ch_type='eeg',
        axes=ax,
        show=False,
        vlim=(vmin, vmax),  # Use vlim instead of vmin and vmax
        cmap=cmap,
        sensors=True
    )
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.show()

    
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
