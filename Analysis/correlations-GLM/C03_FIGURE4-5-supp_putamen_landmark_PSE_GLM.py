"""
===============================================
FIGURE4-5-supp_putamen_landmark_PSE_GLM
Figure four, five and supplementary of paper

Purpose:
    this script reads the model results as well as mediation and moderaion
    results from C02_systematic_model_selection.py and plots the results 
    for publication.

    it will plot:
        1a. beta coefficient of the best model (Landmark_PSE ~ Putamen)
        1b. partial correlation plot of Putamen
        2. moderation and mediation analysis of Handedness on Putamen landmark PSE relationship
        3. moderation and mediation analysis of Eye dominance and microsaccade laterality
          on Putamen landmark PSE relationship


Written by Tara Ghafari
tara.ghafari@gmail.com
11/08/2025
===============================================
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import os.path as op

import pingouin as pg
from itertools import combinations, chain
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Define paths
platform = 'mac'
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
# BEAR outage
# volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/behaviour'
volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias')
models_fname = op.join(volume_sheet_dir, 'Results/model-results/FINAL-Landmark_model_results')
lat_index_csv = op.join(volume_sheet_dir, 'data/collated/FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_flipped.csv')
# Save figure in BEAR outage (that's where the latest version of the manuscript is)
save_path = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/landmark-manus/Figures'

# Define dependent, independent and mediator/moderator variables
data_full = pd.read_csv(lat_index_csv)  # only use this for partial regression plots (not coefficients of the best model)

dep_vars = {'Landmark': 'Landmark_PSE'}
dependent_var = 'Landmark'
independent_var = ['Puta']
mediator =  'Eye_Dominance'  # 'Handedness' or 'Landmark_MS' or 'Eye_Dominance'
moderator = mediator

# Read the models (landmark_pse ~ putamen)
results_df = pd.read_csv(f'{models_fname}/{dependent_var}_model_results.csv')
best_model_table = pd.read_csv(f"{models_fname}/{dependent_var}_best_model.csv")  # this is used for coefficients of the best model
best_model_row = results_df.iloc[0]
best_model = best_model_row['Model']
best_predictors = best_model_row['Predictors']

med_df = pd.read_csv(f'{models_fname}/{dependent_var}_mediator-{mediator}.csv')
moderation_df = pd.read_csv(f'{models_fname}/{dependent_var}_moderator-{mediator}.csv')

# -----------------------
# Plotting Figure 4a and 4b:
# -----------------------
# --- Plot beta coefficients ---
# Extract row-wise stats from flat table
coefficients = best_model_table.set_index('Predictor')['coefficients']
std_err = best_model_table.set_index('Predictor')['standard_error']

# Model-level stats (same across all rows)
fvalue = best_model_table['fvalue'].iloc[0]
fp_value = best_model_table['f_pvalue'].iloc[0]
adj_rsq = best_model_table['rsquared_adj'].iloc[0]
aic = best_model_table['aic'].iloc[0]
bic = best_model_table['bic'].iloc[0]

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8))

# Custom bar colors
color_map = {
    'Puta': '#191970',     # Deep blue
    'const': '#C0C0C0'     # Silver
}
bar_colors = [color_map.get(idx, '#999999') for idx in coefficients.index]

coefficients.plot(kind='bar', yerr=std_err, capsize=6, alpha=0.8,
                  color=bar_colors, edgecolor='black', ax=ax)

ax.set_title(f'Beta Coefficients of the Best Model for {dependent_var}', fontsize=14, fontweight='bold')
ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Predictors', fontsize=12, fontweight='bold')
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.tick_params(axis='x', labelrotation=0)

# === Fit statistics box ===
text = (f"AIC: {aic:.2f}\n"
        f"BIC: {bic:.2f}"
        # f"Adjusted R²: {adj_rsq:.3f}\n"
        # f"F = {fvalue:.3f}\n"
        # f"p-value = {fp_value:.3f}"
        )
ax.text(-0.3, 2, text, fontsize=10, color='black',
        bbox=dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round,pad=1'))

# === Coefficient table box ===
coef_table_text = '\n'.join([
    f"{row['Predictor']}: coef = {row['coefficients']:.3f}, SE = {row['standard_error']:.3f}, "
    f"CI = [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]"
    # f"t = {row['t']:.3f}, p = {row['p_values']:.3f}"
    for _, row in best_model_table.iterrows()
])
ax.text(-0.3, 1, coef_table_text, fontsize=10, color='black',
        bbox=dict(facecolor='whitesmoke', alpha=0.85, edgecolor='dimgray', boxstyle='round,pad=1'))

plt.tight_layout()

# === Save in multiple formats ===
fig.savefig(f'{save_path}/Figure4a_Putamen_GLM.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure4a_Putamen_GLM.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure4a_Putamen_GLM.tiff', format='tiff', dpi=800, bbox_inches='tight')
plt.show()

# --- Partial regression plot ---
# === Reconstruct predictors and DV from saved model table ===
predictors = best_model_table['Predictor'].tolist()
predictors = [p for p in predictors if p != 'const']
dv = dep_vars[dependent_var]  # 'Landmark_PSE'

# Drop NA rows used in model
model_data = data_full.dropna(subset=predictors + [dv])

# Define X and y
X = model_data[predictors]
y = model_data[dv]

# === Partial out other predictors (excluding Putamen) ===
# Residual of y ~ other predictors
other_predictors = [p for p in predictors if p != 'Puta']
model_y = sm.OLS(y, sm.add_constant(model_data[other_predictors])).fit()
y_resid = model_y.resid

# Residual of Puta ~ other predictors
model_x = sm.OLS(model_data['Puta'], sm.add_constant(model_data[other_predictors])).fit()
x_resid = model_x.resid

# === Plotting ===
fig, ax = plt.subplots(figsize=(12, 8))

sns.regplot(
    x=x_resid, y=y_resid,
    scatter=True,
    ci=95,
    ax=ax,
    color='#191970',
    scatter_kws={'s': 60, 'edgecolor': 'black', 'linewidths': 0.6},
    line_kws={'color': 'black', 'lw': 1.5}
)

# === Format ===
ax.set_xlabel(r'LV$_{\mathbf{Puta}}$', fontsize=12, fontweight='bold')
ax.set_ylabel('Landmark PSE', fontsize=12, fontweight='bold')
ax.set_title('Partial Regression Plot for Putamen', fontsize=14, fontweight='bold')

sns.despine()
plt.tight_layout()
fig.savefig(f'{save_path}/Figure4b_Putamen_GLM_partregress.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure4b_Putamen_GLM_partregress.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure4b_Putamen_GLM_partregress.tiff', format='tiff', dpi=800, bbox_inches='tight')
plt.show()

# -----------------------
# Plotting Figure 5 and Supp:
# -----------------------
# Mediation & Moderation:
# Convert SE column to NumPy array
y_errors = med_df['SE'].values  # Ensure it's a 1D NumPy array

# --- Mediation Analysis Results ---
fig, ax = plt.subplots(figsize=(12, 8))

# Bar plot: single color, no auto error bars
sns.barplot(
    data=med_df,
    x='Subcortical',
    y='Indirect_Effect',
    errorbar=None,
    color='#4682B4',  # Steel blue
    edgecolor='black'
)  # Disable automatic error bars

# Manually add error bars
plt.errorbar(
    x=range(len(med_df)),
    y=med_df['Indirect_Effect'],
    yerr=y_errors,
    fmt='none',
    capsize=6,
    color='black'
)

plt.title(f'Mediation Analysis: Indirect effects', fontsize=14, fontweight='bold')
plt.xlabel('Subcortical Structure', fontsize=12, fontweight='bold')
plt.ylabel('Indirect Effect', fontsize=12, fontweight='bold')

txt_pval = f'p-value = {med_df['p_value'][0]:.3f}'
box_props = dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round')
plt.text(0.05, 0.05, 
        txt_pval,
        transform=plt.gca().transAxes,
        fontsize=10, 
        verticalalignment='top', 
        bbox=box_props,
        style='italic')

plt.tight_layout()
fig.savefig(f'{save_path}/Figure1bSupp_{mediator}_mediation.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure1bSupp_{mediator}_mediation.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure1bSupp_{mediator}_mediation.tiff', format='tiff', dpi=800, bbox_inches='tight')
plt.show()

# --- Plot moderation effects --
# === Extract components from moderation_df ===
# Make sure index = predictor names
moderation_df = moderation_df.set_index('Unnamed: 0')  # Use predictor names as index

mod_coefficients = moderation_df['coefficients']
mod_std_err = moderation_df['standard_error']
mod_p_values = moderation_df['p_values']
mod_fvalue = moderation_df['fvalue'].iloc[0]
mod_fp_value = moderation_df['f_pvalue'].iloc[0]
mod_rsquared_adj = moderation_df['rsquared_adj'].iloc[0]

# === Custom bar colors ===
color_map = {
    'Puta': '#191970',                          # Deep blue
    f'{mediator}': '#4682B4',                   # Steel blue
    f'Puta_x_{mediator}': '#20B2AA',            # Light sea green
    'const': '#C0C0C0'                          # Silver
}
bar_colors = [color_map.get(idx, '#999999') for idx in moderation_df.index]

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8))

mod_coefficients.plot(
    kind='bar', 
    yerr=mod_std_err, 
    ax=ax, 
    alpha=0.85,
    color=bar_colors, 
    edgecolor='black', 
    capsize=6
)

ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.set_title(f'Moderation Effect', fontsize=14, fontweight='bold')
ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Predictors', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', labelrotation=0)

# === Model fit text box ===
fit_text = (f"Adjusted R²: {mod_rsquared_adj:.3f}\n"
            f"fvalue = {mod_fvalue:.3f}\n"
            f"p-value = {mod_fp_value:.3f}")
ax.text(-0.4, 15, fit_text, fontsize=10, color='black',
        bbox=dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round,pad=1'))

# === Coefficient table box ===
mod_table = pd.DataFrame({
    'coef': mod_coefficients,
    'SE': mod_std_err,
    't': mod_coefficients / mod_std_err,
    'p': mod_p_values,
})
mod_table['CI_lower'] = mod_table['coef'] - 1.96 * mod_table['SE']
mod_table['CI_upper'] = mod_table['coef'] + 1.96 * mod_table['SE']
mod_table_rounded = mod_table.round(3).astype(str)

# Format as text block
coef_table_text = '\n'.join([
    f"{idx}: coef = {row['coef']}, SE = {row['SE']}, CI = [{row['CI_lower']}, {row['CI_upper']}], "
    # f" t = {row['t']}, " 
    f"p-value = {row['p']}"
    for idx, row in mod_table_rounded.iterrows()
])

ax.text(-0.4, 9, coef_table_text, fontsize=10, color='black',
        bbox=dict(facecolor='whitesmoke', alpha=0.85, edgecolor='dimgray', boxstyle='round,pad=1'))

plt.tight_layout()
fig.savefig(f'{save_path}/Figure1aSupp_{mediator}_moderation.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure1aSupp_{mediator}_moderation.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure1aSupp_{mediator}_moderation.tiff', format='tiff', dpi=800, bbox_inches='tight')
plt.show()