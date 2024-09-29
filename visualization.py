import matplotlib.pyplot as plt
import mne
import numpy as np

def plot_topomap_difference(info, mean_power_cond1, mean_power_cond2, title, significant_clusters=None):
    """
    Plot the topographic map of band power differences between two conditions,
    optionally highlighting significant clusters.
    """
    power_diff = mean_power_cond1 - mean_power_cond2  # Shape: (n_channels,)
    evoked_diff = mne.EvokedArray(
        power_diff[:, np.newaxis],
        info,
        tmin=0.0
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im, cn = mne.viz.plot_topomap(
        evoked_diff.data[:, 0], info, axes=ax, show=False, cmap='RdBu_r',
        contours=0
    )
    ax.set_title(title, fontsize=12)
    
    if significant_clusters is not None:
        for cluster_idx in significant_clusters:
            cluster = stat_results[band][(cond1, cond2)]['clusters'][cluster_idx]
            mne.viz.plot_topomap(
                np.isin(np.arange(len(info['ch_names'])), cluster).astype(int),
                info, axes=ax, show=False, cmap='Greens',
                alpha=0.5, contours=0
            )
    
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
