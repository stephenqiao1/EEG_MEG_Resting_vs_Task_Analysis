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
1. 
