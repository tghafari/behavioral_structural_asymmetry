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
    7. Histograms for Landmark_PSE and Target_PSE_Laterality
       with statistical comparison to zero.

written by Tara Ghafari
===============================================
"""

import pandas as pd
import os.path as op
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/behaviour'
# op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/collated')
lat_index_csv = op.join(volume_sheet_dir, 'FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers.csv')

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

        box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, 
                f'r={correlation:.2f}\np={p_value:.3f}', 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=box_props)


# Plot Landmark_PSE vs. Target_PSE_Laterality
x = data['Landmark_PSE']
y = data['Target_PSE_Laterality']

valid_idx = ~(x.isna() | y.isna())
x = x[valid_idx]
y = y[valid_idx]


ax = axs[-1, 0]
ax.scatter(x, y, alpha=0.7)
ax.set_xlabel('Landmark_PSE')
ax.set_ylabel('Target_PSE_Laterality')

# Spearman correlation
correlation, p_value = spearmanr(x, y)
box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, 
        f'r={correlation:.2f}\np={p_value:.3f}', 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=box_props)

# Plot Landmark_PSE vs Landmark_MS
x = data['Landmark_PSE']
y = data['Landmark_MS']

valid_idx = ~(x.isna() | y.isna())
x = x[valid_idx]
y = y[valid_idx]


ax = axs[-1, 1]
ax.scatter(x, y, alpha=0.7)
ax.set_xlabel('Landmark_PSE')
ax.set_ylabel('Landmark_MS')

# Spearman correlation
correlation, p_value = spearmanr(x, y)
box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, 
        f'r={correlation:.2f}\np={p_value:.3f}', 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=box_props)

x = data['Target_PSE_Laterality']
y = data['Target_MS_Laterality']

valid_idx = ~(x.isna() | y.isna())
x = x[valid_idx]
y = y[valid_idx]


ax = axs[-1, 2]
ax.scatter(x, y, alpha=0.7)
ax.set_xlabel('Target_PSE_Laterality')
ax.set_ylabel('Target_MS_Laterality')

# Spearman correlation
correlation, p_value = spearmanr(x, y)
box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, 
        f'r={correlation:.2f}\np={p_value:.3f}', 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=box_props)

# Plot Landmark_MS vs. Target_MS_Laterality
x = data['Landmark_MS']
y = data['Target_MS_Laterality']

valid_idx = ~(x.isna() | y.isna())
x = x[valid_idx]
y = y[valid_idx]


ax = axs[-1, 3]
ax.scatter(x, y, alpha=0.7)
ax.set_xlabel('Landmark_MS')
ax.set_ylabel('Target_MS_Laterality')

# Spearman correlation
correlation, p_value = spearmanr(x, y)
box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, 
        f'r={correlation:.2f}\np={p_value:.3f}', 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=box_props)


[fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]  # remove empty plots
# Adjust margins and display the plot
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()


# ================================================
# Add Histograms for Landmark_PSE and Target_PSE
# ================================================

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Function to compute stats and annotate histograms
def plot_histogram_with_stats(ax, data, title):
    # Remove NaN
    data = data.dropna()

    # Calculate stats
    mean_val = np.mean(data)
    median_val = np.median(data)
    stat, p_value = wilcoxon(data)  # Wilcoxon test for median difference from 0

    # Plot histogram
    ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', alpha=0.7, label='Zero (Null)')
    ax.axvline(mean_val, color='green', linestyle='--', label=f'Mean={mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle='--', label=f'Median={median_val:.2f}')

    # Annotate
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Annotate stats
    ax.text(0.05, 0.95, 
            r'$wilcox\_p = {:.4f}$'.format(p_value), 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot Landmark_PSE
plot_histogram_with_stats(axs[0], data['Landmark_PSE'], 'Landmark_PSE Distribution')

# Plot Target_PSE_Laterality
plot_histogram_with_stats(axs[1], data['Target_PSE_Laterality'], 'Target_PSE_Laterality Distribution')

plt.tight_layout()
plt.show()