# -*- coding: utf-8 -*-
"""
===============================================

01_read_and_calculate_size_PSE_correlations

This code

    1. Read DataFrame.
    2. Scatter Plot Loop:
        Loops through structural_columns and 
        behavioral_columns to generate scatter plots.
    3. Correlation and p-value:
        Calculates Pearson correlation coefficients
        and p-values using pearsonr.
    4. Dynamic Plot Layout:
        The number of rows and columns is 
        determined dynamically based on the 
        number of plots.
    5. Annotations:
        Each scatter plot is annotated with the 
        correlation coefficient (r) and p-value (p).
    6. Extra Plot:
        Includes a scatter plot for 
        Landmark_PSE vs. Target_PSE_Laterality.

written by Tara Ghafari
===============================================
"""

import pandas as pd
import os.path as op

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/substr-beh'
# op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/collated')
lat_index_csv = op.join(volume_sheet_dir, 'unified_behavioral_structural_asymmetry_lateralisation_indices_1_45.csv')

# Read the CSV as a DataFrame
data = pd.read_csv(lat_index_csv)

# Define columns for analysis
structural_columns = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
behavioural_columns = ['Landmark_PSE', 'Target_PSE_Laterality', 'Landmark_MS', 'Target_MS_Laterality']

# Initialize plots
num_cols = len(structural_columns)
num_rows = len(behavioural_columns) + 1  # One additional row for Landmark_PSE vs Target_PSE_Laterality
fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), constrained_layout=True)

# Plot structural vs. behavioral columns
for col_idx, structural in enumerate(structural_columns):
    for row_idx, behavioural in enumerate(behavioural_columns):
        x = data[structural]
        y = data[behavioural]

        # Exclude NaN values - manually put Nan in place of outliers on the csv file
        valid_idx = ~(x.isna() | y.isna())
        x = x[valid_idx]
        y = y[valid_idx]

        # Scatter plot
        ax = axs[row_idx, col_idx]
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel(structural)
        ax.set_ylabel(behavioural)

        # Pearson correlation
        correlation, p_value = spearmanr(x, y)
        ax.text(0.05, 0.95, f'r={correlation:.2f}\np={p_value:.2e}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Plot Landmark_PSE vs. Target_PSE_Laterality
x = data['Landmark_PSE']
y = data['Target_PSE_Laterality']

valid_idx = ~(x.isna() | y.isna())
x = x[valid_idx]
y = y[valid_idx]

for col_idx, structural in enumerate(structural_columns):
    ax = axs[-1, col_idx]
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel('Landmark_PSE')
    ax.set_ylabel('Target_PSE_Laterality')

    # Pearson correlation
    correlation, p_value = spearmanr(x, y)
    ax.text(0.05, 0.95, f'r={correlation:.2f}\np={p_value:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Adjust margins and display the plot
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

