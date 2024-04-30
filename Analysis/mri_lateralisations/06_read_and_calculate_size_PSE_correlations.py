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
volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/collated')
lat_volume_csv = op.join(volume_sheet_dir, 'lateralisation_vol-PSE_1_31.csv')

data = []
with open(lat_volume_csv, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
    
# Extract the columns of lateralisation volumes
data = np.array(data)
LV_columns = data[1:-1, 1:8].astype(float)  # last row is removed becuase the values are still missing

# what are you plotting? PSE_landmark or MS_target
plotting = 'PSE_target'


if plotting == 'PSE_landmark':
    PSE_column = data[1:, 8].astype(float)  # these data should be added to the csv file manually before running this script
    y = PSE_column
    
elif plotting == 'MS_target':
    ms_column = data[1:, 9].astype(float) # these data should be added to the csv file manually before running this script
    y = ms_column

elif plotting == 'PSE_target':
    target_column = data[1:-1, 10].astype(float) # these data should be added to the csv file manually before running this script
    y = target_column

# Plot the data and calculate correlations
correlations = []
p_values = []
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
axs = axs.flatten()

for i in range(7):
    x = LV_columns[:, i]

    axs[i].scatter(x, y)
    axs[i].set_xlabel(f"LV_{data[0,i+1]}")
    axs[i].set_ylabel(plotting)

    correlation, p_value = spearmanr(x, y)  # or pearsonr
    correlations.append(correlation)
    p_values.append(p_value)

    axs[i].set_title(f"Spearman Correlation: {correlation:.4f}\n p-value: {p_value:.4f}")
    
plt.tight_layout()
plt.show()


# Print the correlations and p-values
for i, (correlation, p_value) in enumerate(zip(correlations, p_values)):
    print(f"Correlation between {data[0, i+1]} and PSE:: {correlation:.4f} (p-value: {p_value:.4f})")    
    
    
    
    