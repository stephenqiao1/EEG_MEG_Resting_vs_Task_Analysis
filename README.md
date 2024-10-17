# EEG_MEG_Resting_vs_Task_Analysis
## Overview 
The EEG Data Analysis Pipeline is a comprehensive Python-based framework designed to process, analyze, and visualize EEG (Electroencephalography) data across multiple subjects and conditions. Leveraging powerful libraries such as MNE-Python, NumPy, and Matplotlib, this pipeline facilitates efficient data handling, robust preprocessing, advanced signal analysis, and insightful visualizations, enabling researchers and engineers to derive meaningful insights from EEG datasets.

## Features
* **Scalable Processing**: Efficiently handles EEG data from 100+ subjects using parallel processing techniques.
* **Robust Preprocessing**: Automated cleaning, bad channel detection, and artifact removal using Independent Component Analysis (ICA).
* **Flexible Epoching**: Supports both fixed-length and event-based epoch creation tailored to different experimental conditions.
* **Advanced Signal Analysis**: Computes Power Spectral Density (PSD) across standard EEG frequency bands (Delta, Theta, Alpha, Beta, Gamma).
* **Statistical Testing**: Implements permutation cluster tests to identify significant differences between experimental conditions.
* **Comprehensive Visualization**: Generates topographic maps and individual subject plots to visualize spatial patterns of EEG band power.
* **Modular Design**: Organized into distinct modules for data loading, preprocessing, analysis, and visualization, promoting maintainability and extensibility.
* **Error Handling and Logging**: Robust mechanisms to handle missing data and log errors without interrupting the processing pipeline.

## Technologies Used
* **Programming Languages**: Python
* **Libraries**: MNE-Python, Matplotlib, statsmodels
* **Tools**: Git, Jupyter Notebooks
* **Data Formats**: EDF

## Installation
### Prerequisites
* Python 3.8 or higher
* pip
### Step-by-Step Installation
1. **Clone the Repository**
```
git clone https://github.com/yourusername/eeg-data-analysis-pipeline.git
cd eeg-data-analysis-pipeline
```
2. **Create a Virtual Environment (Optional but Recommended)
```
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
3. **Install Dependencies**
```
pip install numpy mne matplotlib statsmodels
```
## Usage
### Configuration
1. **Update Subject List**
   By default, the pipeline is set to process subjects 1 through 3.
   ```
   subjects = [1, 2, 3]  # Update this list with your subject numbers
   ```
2. **Set Run Information**
   Ensure that the runs dictionary correctly maps condition names to their respective run numbers.
   ```
   runs = {
    'Rest Open': 1,
    'Rest Closed': 2,
    'Task': 3
   }
   ```
3. **Define Frequency Bands**
   The pipeline analyzes EEG data across standard frequency bands. Adjust if necessary.
  ```
frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
  ```
4. **Specify Conditions**
   List all experimental conditions to be analyzed.
   ```
   all_conditions = [
    'Rest Open',
    'Rest Closed',
    'Task Rest',
    'Left Fist',
    'Right Fist'
   ]
   ```
### Running the Pipeline
1. **Execute the Main Script**
   ```
   python main.py
   ```
2. **Accessing Results**
   * **Statistical Results**: Stored in the stat_results dictionary
   * **Visualizations**: Generated plots are saved in designated directories (plots/topomaps and plots/individual_subjects).
  
## Data
### Data Requirements
* **Format**: EDF
* **Structure**: Each subject should have EEG recordings for the following conditions:
  - Resting with eyes open (Rest Open)
  - Resting with eyes closed (Rest Closed)
  - Task-related activity (Task)
* **Directory Structure**: Organize data in a consistent directory hierarchy to facilitate automated loading.
### Loading Data
The load_edf function from the data_loading module handles the ingestion of EDF files. Ensure that EDF files are named or organized in a manner that aligns with the function's expectations.

## Pipeline Workflow
### 1. Data Loading
* **Function**: load_edf(subject_number, run_number)
* **Purpose**: Loads raw EEG data for each subject and condition from EDF files.
* **Error Handling**: Skips subjects with missing data files, logging the issue.
### 2. Preprocessing
* **Functions**:
  - clean_channel_names(raw_data)
  - preprocess_raw(raw_data, bad_channels)
  - perform_ica(raw_data)
* **Steps**:
  - **Channel Name Cleaning**: Standardizes EEG channel names for consistency
  - **Bad Channel Detection**: Loads previously identified bad channels or performs visual inspection to mark bad channels.
  - **Artifact Removal**: Uses ICA to remove artifacts like eye blinks and muscle movements.
  - **Channel Selection**: Retains only good EEG channels for further analysis.
### 3. Epoch Creation
* **Functions**:
  - create_fixed_length_epochs(raw_data, duration, overlap)
  - create_epochs(raw_data, tmin, tmax, baseline)
* **Steps**:
  - **Resting State**: Creates fixed-length epochs for resting conditions
  - **Task Conditions**: Creates event-based epochs aligned to specific task-related events (e.g., Left Fist, Right Fist)
### 4. Power Spectral Density (PSD) Computation
* **Function**: compute_psd_epochs(epochs, fmin, fmax)
* **Purpose**: Calculates PSD for each epoch within defined frequency bands.
* **Output**: PSD values and corresponding frequencies for each condition and band.
### 5. Band Power Calculation
* **Function**: compute_band_power(psds)
* **Purpose**: Computes average band power across epochs for statistical analysis.
* **Aggregation**: Stores mean band power for each subject, condition, and frequency band.
### 6. Statistical Analysis
* **Function**: permutation_cluster_test(X, n_permutations, tail, threshold, n_jobs, seed)
* **Purpose**: Performs cluster-based permutation tests to identify significant differences between conditions.
* **Parameters**:
  - X: List containing data arrays for the two conditions being compared
  - n_permutations: Number of permutations for the test (e.g., 5000).
  - n_jobs: Number of CPU cores to use
* **Output**:
  - T_obs: Observed test statistics
  - clusters: Identified clusters of significant differences.
  - cluster_p_values: P-values for each cluster
  - significant_clusters: Indices of clusters with p-values below the significance threshold
### 7. Visualization
* **Topographic Maps**:
  - **Function**: plot_topomap_difference(info, mean_power_cond1, mean_power_cond2, title, cmap)
  - **Purpose**: Visualizes spatial differences in band power between two conditions across the scalp.
* **Individual Subject Plots**:
  - **Function**: Utilizes MNE's plot_topomap to create and save individual subject topographic maps.
* **Organizing Plots**:
  - Plots are saved in structured directories (plots/topomaps and plots/individual_subjects) for easy access and organization.


