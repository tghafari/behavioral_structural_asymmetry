
"""
===============================================
FIGURE1_calculate_PSE
Figure one of paper

Purpose:
    This script processes and visualizes Point of Subjective Equality (PSE) values from the Landmark Task.
    Specifically, it flips the sign of all PSE values to align the directional bias,
    bins the values into 0.2° intervals, and creates a publication-ready bar plot.
    It annotates statistical markers including mean, median, and Wilcoxon signed-rank.

What the code does:
    - Flips the sign of PSE values (e.g., +0.5 becomes -0.5).
    - Bins values symmetrically from -0.8° to +0.8° in 0.2° steps.
    - Creates a histogram showing the number of participants per bin.
    - Adds vertical lines for zero, mean, and median.
    - Conducts a Wilcoxon signed-rank test against zero and displays the p-value.
    - Annotates "Leftward Bias" on the negative side and "Rightward Bias" on the positive side.
    - Stylizes the plot for publishing (font sizes, labels, colors, etc.).

Written by Tara Ghafari
tara.ghafari@gmail.com
07/08/2025
===============================================
"""

import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/data/collated')
lat_index_csv = op.join(volume_sheet_dir, 'FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_eye-dominance.csv')
# BEAR outage
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/behaviour'
# Save figure in BEAR outage (that's where the latest version of the manuscript is)
save_path = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/landmark-manus/Figures/Figure2_Landmark_PSE'

# Read and flip the signs of Landmark_PSE values
"""the signs we calculated from the task should be flipped
 because we want negative pse to be on the left side of the histogram
 (more intuitive for leftward bias) and positive on the rightside."""
data_full = pd.read_csv(lat_index_csv)
flipped_data_full = data_full.copy()
flipped_data_full['Landmark_PSE'] = -data_full['Landmark_PSE']
flipped_data_full.to_csv(op.join(volume_sheet_dir, 'FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_flipped.csv'))

flipped_pse = flipped_data_full['Landmark_PSE'].dropna()  # subject 29 is removed

# Compute statistics
mean_val = flipped_pse.mean()
median_val = flipped_pse.median()
stat, p_value = wilcoxon(flipped_pse)

# Define bins and compute histogram
bins = np.linspace(-0.8, 0.8, 17)  # 0.2 width bins from -0.8 to 0.8
bias_data = pd.cut(flipped_pse, bins=bins)
bias_table = bias_data.value_counts().sort_index()
bin_midpoints = (bins[:-1] + bins[1:]) / 2

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(bin_midpoints, bias_table.values, width=0.08,
       color='#a6c6de', edgecolor='#2F4F4F') 

# Axis and tick formatting
ax.set_xlabel('Spatial Bias (Deg. Vis. Ang.)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 1)
ax.set_xticks([-0.8 ,-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels(['-0.8°', '-0.6°', '-0.4°', '-0.2°', '0°', '+0.2°', '+0.4°', '+0.6°', '+0.8°'])
ax.set_ylim(0, max(bias_table.values) + 1)

# Add vertical lines for zero, mean, and median
"""Only plot zero and median"""
ax.axvline(x=0, color='dimgray', linestyle='-', linewidth=2, label='Zero (Null)')
# ax.axvline(x=mean_val, color='orange', linestyle=':', linewidth=2, label=f'Mean = {mean_val:.2f}')
ax.axvline(x=median_val, color='darkorange', linestyle='--', linewidth=2, label=f'Median = {median_val:.2f}')


# # Add bias direction annotations
# ax.text(-0.75, ax.get_ylim()[1]*0.95, 'Leftward Bias',
#         fontsize=11, color='darkgreen', ha='left')
# ax.text(0.75, ax.get_ylim()[1]*0.95, 'Rightward Bias',
#         fontsize=11, color='darkgreen', ha='right')

# Show Wilcoxon p-value
p_text = f"Wilcoxon p = {p_value:.4f}" if p_value >= 0.001 else "Wilcoxon p < 0.001"
ax.text(-0.88, ax.get_ylim()[1]*0.90, p_text,
        bbox=dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round,pad=1'),
        fontsize=10, style='italic')

# Aesthetic adjustments
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Distribution of PSEs', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
# plt.show()

fig.savefig(f'{save_path}.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}.tiff', format='tiff', dpi=800, bbox_inches='tight')