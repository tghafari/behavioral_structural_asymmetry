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

data = np.array(data)

# what are you plotting? 
plotting = 'PSE_target' # lateralised performance in 'PSE_landmark' or 'MS_target' or 'PSE_target'
LV = 'thomas'  # lateralisation volume os 'substr' or 'thomas'?

if plotting == 'PSE_landmark':
    PSE_column = data[1:, 8].astype(float)  # these data should be added to the csv file manually before running this script
    outlier_idx = [15, 27] # remove outliers from PSE_landmark: 1016,1028
    y = np.delete(PSE_column, outlier_idx)


elif plotting == 'MS_target':
    ms_column = data[1:-1, 9].astype(float) # these data should be added to the csv file manually before running this script
    outlier_idx = [] # remove outliers from MS_target: 
    y = np.delete(ms_column, outlier_idx)
                       

elif plotting == 'PSE_target':
    target_column = data[1:-1, 10].astype(float) # these data should be added to the csv file manually before running this script
    outlier_idx = [2, 3, 6, 7, 15, 16, 17,  # 0-based index
                24, 26, 27, 30] # remove outliers from PSE_target: 1003,1004,1007,1008,1016,1017,1018,1025,1027,1028,1031,1032
    y = np.delete(target_column, outlier_idx)

# Extract the columns of lateralisation volumes and remove outliers
LV_columns_outlier = data[1:-1, 1:8].astype(float)  # last row is removed becuase the values are still missing
LV_columns = np.delete(LV_columns_outlier, outlier_idx, axis=0)

thomas_columns_outlier = data[1:-1, 11:20].astype(float)
thomas_columns = np.delete(thomas_columns_outlier, outlier_idx, axis=0)


# Plot the data and calculate correlations
correlations = []
p_values = []
fig, axs = plt.subplots(3, 4, figsize=(24, 12))
axs = axs.flatten()

for i in range(9):  # calculating correlations for substrs (range(7)) or thomas (range(9))
    if LV == 'substr':
        x = LV_columns[:, i]
    else:
        x = thomas_columns[:, i]

    axs[i].scatter(x, y)
    axs[i].set_xlabel(f"LV_{data[0,i+11]}")  # for substr data[0,i+1] for thomas data[0,i+11]
    axs[i].set_ylabel(plotting)

    correlation, p_value = spearmanr(x, y)  # or pearsonr
    correlations.append(correlation)
    p_values.append(p_value)

    axs[i].set_title(f"Spearman Correlation: {correlation:.4f}\n p-value: {p_value:.4f}")

# Hide any unused subplots
for i in range(9, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


# Print the correlations and p-values
for i, (correlation, p_value) in enumerate(zip(correlations, p_values)):
    print(f"Correlation between {data[0, i+11]} and PSE:: {correlation:.4f} (p-value: {p_value:.4f})")    
