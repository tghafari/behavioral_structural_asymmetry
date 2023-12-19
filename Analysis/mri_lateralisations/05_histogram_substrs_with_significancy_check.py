# -*- coding: utf-8 -*-
"""
===============================================
05. histogram substrs

This code read the lateralized volumes from a 
csv file, plots a histogram for each substr and
checks for significant differences with normal 
distribution

written by Tara Ghafari
==============================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as op


# Load the lateralization index sheet
volume_sheet_dir = r'Z:\Projects\subcortical-structures\SubStr-and-behavioral-bias\results\MRI_lateralisations\lateralisation_indices'
lat_sheet_fname = op.join(volume_sheet_dir, 'lateralisation_volumes.csv')
df = pd.read_csv(lat_sheet_fname)
lateralisation_volume = df.iloc[:,1:8].to_numpy()

colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
p_values = []

fig, axs = plt.subplots(2, 4)
fig.set_figheight(6)
fig.set_figwidth(10)

for his in range(7):       
    # Define plot settings
    ax = axs[his // 4, his % 4]
    ax.set_title(structures[his], fontsize=12, fontname='Calibri')
    ax.set_xlabel('Lateralisation Volume', fontsize=12, fontname='Calibri')
    ax.set_ylabel('# Subjects', fontsize=12, fontname='Calibri')
    ax.axvline(x=0, color='k', linewidth=0.25, linestyle=':')
    
    # Remove nans and plot normalized (z-scored) distributions
    valid_lateralisation_volume = lateralisation_volume[~np.isnan(lateralisation_volume[:, his]), his]
    lateralisation_volume_hist = np.histogram(valid_lateralisation_volume, bins=6, density=False)
    
    # Throw out the outliers
    mean_lateralisation_volume = np.nanmean(valid_lateralisation_volume)
    std_lateralisation_volume = np.nanstd(valid_lateralisation_volume)
    threshold = mean_lateralisation_volume - (2.5 * std_lateralisation_volume)
    valid_lateralisation_volume[:][valid_lateralisation_volume[:] <= threshold] = np.nan
        
    # plot histogram
    x = lateralisation_volume_hist[1]
    y = lateralisation_volume_hist[0]
    ax.bar(x[:-1], y, width=np.diff(x), color=colormap[his])
        
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
