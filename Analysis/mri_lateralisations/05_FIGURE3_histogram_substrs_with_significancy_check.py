# -*- coding: utf-8 -*-
"""
===============================================
05_FIGURE2_histogram_substrs_with_significancy_check.py

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
from scipy import stats
from scipy.stats import shapiro

platform = 'mac'

# Define where to read and write the data
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# # Define where to read and write the data
# volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/MRI_lateralisations/lateralisation_indices')
# lat_sheet_fname = op.join(volume_sheet_dir, 'lateralisation_volumes_1_45.csv')
# df = pd.read_csv(lat_sheet_fname)
# lateralisation_volume = df.iloc[:,1:8].to_numpy()

# Collated data
volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/data/collated')
lat_index_csv = op.join(volume_sheet_dir, 'FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_eye-dominance.csv')

# Save figure in BEAR outage (that's where the latest version of the manuscript is)
save_path = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/landmark-manus/Figures/Figure3_substr_hist'

data_full = pd.read_csv(lat_index_csv)
lateralisation_volume = data_full.iloc[:,3:10].to_numpy()

colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
p_values = []
p_values_shapiro =[]

# null hypothesis (H0) mean value
null_hypothesis_mean = 0.0
medians = []
t_stats = []
t_p_vals = []

# wilcoxon p-vals
null_hypothesis_median = 0.0
wilcox_p_vals = []

fig, axs = plt.subplots(2, 4)
fig.set_figheight(6)
fig.set_figwidth(10)

for his in range(7):       
    # Define plot settings
    ax = axs[his // 4, his % 4]
    ax.set_title(structures[his], fontsize=12)#, fontname='Calibri')
    ax.set_xlabel('Lateralisation Volume', fontsize=12)#, fontname='Calibri')
    ax.set_ylabel('# Subjects', fontsize=12)#, fontname='Calibri')
    ax.axvline(x=0, color='dimgray', linewidth=0.5, linestyle='-')

    # Compute statistics
    median_val = lateralisation_volume[:,his].mean()
    medians.append(median_val)
    
    # Remove nans and plot normalized (z-scored) distributions
    valid_lateralisation_volume = lateralisation_volume[~np.isnan(lateralisation_volume[:, his]), his]
    lateralisation_volume_hist = np.histogram(valid_lateralisation_volume, bins=6, density=False)
    
    # Throw out the outliers
    mean_lateralisation_volume = np.nanmean(valid_lateralisation_volume)
    std_lateralisation_volume = np.nanstd(valid_lateralisation_volume)
    threshold = mean_lateralisation_volume - (2.5 * std_lateralisation_volume)
    valid_lateralisation_volume[:][valid_lateralisation_volume[:] <= threshold] = np.nan
    print(len(valid_lateralisation_volume))

    # Perform the ranksum test
    k2, p = stats.normaltest(valid_lateralisation_volume, nan_policy='omit')
    p_values.append(p)
    stat, shapiro_p = shapiro(valid_lateralisation_volume)
    p_values_shapiro.append(shapiro_p)
    
    # 1 sample t-test for left/right lateralisation
    # t_statistic, t_p_value = stats.ttest_1samp(valid_lateralisation_volume, 
    #                                            null_hypothesis_mean, 
    #                                            nan_policy='omit')
    # t_stats.append(t_statistic)
    # t_p_vals.append(t_p_value)
    # txt_t = r'$1samp\_p = {:.2f}$'.format(t_p_value)

    
    # one sample wilcoxon signed rank (for non normal distributions)
    _, wilcox_p = stats.wilcoxon(valid_lateralisation_volume - null_hypothesis_median,
                                 zero_method='wilcox', 
                                 nan_policy='omit',
                                 correction=False)
    wilcox_p_vals.append(wilcox_p)
  
    # plot histogram
    x = lateralisation_volume_hist[1]
    y = lateralisation_volume_hist[0]
    ax.bar(x[:-1], y, width=np.diff(x), color=colormap[his])
    
    # # plot a normal density function
    # mu, std = stats.norm.fit(valid_lateralisation_volume)
    # pdf = stats.norm.pdf(x, mu, std)
    # txt_norm = r'$p = {:.2f}$'.format(p)
    # ax.plot(x, pdf, 'r-', label='Normal Fit', linewidth=0.5)        
        
    box_props = dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round')
    txt_wilx = f"Wilcoxon p = {wilcox_p:.3f}" if wilcox_p >= 0.001 else "Wilcoxon p < 0.001"

    
    ax.text(0.05, 0.95, 
            txt_wilx,
            transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=box_props,
            style='italic')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)

[fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]  # remove empty plots
plt.tight_layout()
# plt.show()

fig.savefig(f'{save_path}.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}.tiff', format='tiff', dpi=800, bbox_inches='tight')