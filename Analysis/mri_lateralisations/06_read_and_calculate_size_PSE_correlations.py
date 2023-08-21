# -*- coding: utf-8 -*-
"""
===============================================

06_read_and_calculate_size_PSE_correlations

This code reads lateralisation indices from 
a csv file and correlates them with each subject's
PSE

written by Tara Ghafari
===============================================
"""

import csv
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Read the CSV file
volume_sheet_dir = r'Z:\Projects\Subcortical_Structures\SubStr_and_behavioral_bias\Analysis\MRI_lateralisations\lateralisation_indices'
lat_volume_csv = op.join(volume_sheet_dir, 'lateralisation_volumes.csv')
data = []
with open(lat_volume_csv, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
    
# Exclude subject's with NaN data
remove = [2,5]
keep = [1,3,4,6]
cols = np.setxor1d(np.arange(1,7),remove)

# Extract the columns of lateralisation indices
data = np.array(data)
columns = data[cols, 1:7].astype(float)

# Extract PSE column
target_column = data[1:, 8].astype(float)

# Extract MS column
ms_column = data[cols, 9].astype(float)

# Plot the data and calculate correlations
correlations = []
p_values = []
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
axs = axs.flatten()

for i in range(6):
    x = columns[:, i]
    y = ms_column

    axs[i].scatter(x, y)
    axs[i].set_xlabel(f"LV_ {data[0,i+1]}")
    axs[i].set_ylabel("microsaccade")

    correlation, p_value = pearsonr(x, y)
    correlations.append(correlation)
    p_values.append(p_value)

    axs[i].set_title(f"Pearson Correlation: {correlation:.4f}\n p-value: {p_value:.4f}")
    
plt.tight_layout()
plt.show()

# Print the correlations and p-values
for i, (correlation, p_value) in enumerate(zip(correlations, p_values)):
    print(f"Correlation between {data[0,i+1]} and PSE:: {correlation:.4f} (p-value: {p_value:.4f})")    
    
    
    
    