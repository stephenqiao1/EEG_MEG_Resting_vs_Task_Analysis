from mne.stats import permutation_cluster_test
from mne.time_frequency import psd_array_multitaper
import numpy as np

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