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
from scipy.stats import (pearsonr, spearmanr)

platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/MRI_lateralisations/lateralisation_indices')
lat_volume_csv = op.join(volume_sheet_dir, 'lateralisation_volumes_1_32.csv')

data = []
with open(lat_volume_csv, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
    
# Exclude subject's with NaN data- if NaNs present in data
# remove = [2,5]
# keep = [1,3,4,6]
# cols = np.setxor1d(np.arange(1,7),remove)
# if no NaNs
cols = range(1001,1032)

# Extract the columns of lateralisation indices
data = np.array(data)
columns = data[cols, 1:32].astype(float)

# what are you plotting? PSE_landmark or microssaccades
plotting = 'MS_target'

# Extract PSE column
PSE_column = data[1:, 8].astype(float)  # these data should be added to the csv file manually before running this script

# Extract MS column
ms_column = data[cols, 9].astype(float) # these data should be added to the csv file manually before running this script

# Plot the data and calculate correlations
correlations = []
p_values = []
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
axs = axs.flatten()

if plotting == 'PSE_landmark':
    y = PSE_column

elif plotting == 'MS_target':
    y = ms_column

for i in range(6):
    x = columns[:, i]

    axs[i].scatter(x, y)
    axs[i].set_xlabel(f"LV_ {data[0,i+1]}")
    axs[i].set_ylabel(plotting)

    correlation, p_value = spearmanr(x, y)  # or pearsonr
    correlations.append(correlation)
    p_values.append(p_value)

    axs[i].set_title(f"Spearman Correlation: {correlation:.4f}\n p-value: {p_value:.4f}")
    
plt.tight_layout()
plt.show()


# Print the correlations and p-values
for i, (correlation, p_value) in enumerate(zip(correlations, p_values)):
    print(f"Correlation between {data[0,i+1]} and PSE:: {correlation:.4f} (p-value: {p_value:.4f})")    
    
    
    
    