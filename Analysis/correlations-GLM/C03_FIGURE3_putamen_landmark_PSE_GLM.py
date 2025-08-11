"""
===============================================
FIGURE345_putamen_landmark_PSE_GLM
Figure three, four and five of paper

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
models_fname = op.join(volume_sheet_dir, 'Results/model-results/FINAL-model_results')
lat_index_csv = op.join(volume_sheet_dir, 'data/collated/FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_eye-dominance.csv')
# Save figure in BEAR outage (that's where the latest version of the manuscript is)
save_path = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/landmark-manus/Figures'

# Step 1: Load the CSV file
data_full = pd.read_csv(lat_index_csv)
print(data_full.head())

# Step 2: Define dependent and independent variables
dep_vars = {'Landmark': 'Landmark_PSE'}
dependent_var = 'Landmark'
independent_var = ['Puta']
mediator =  'Handedness'  # or 'Landmark_MS' or 'Eye_Dominance'
moderator = mediator

# Remove NaNs from dependent variable (but keep rows in the dataset)
data = data_full.dropna(subset=[dep_vars[dependent_var]] + independent_var)

# Read the main model (landmark_pse ~ putamen)
results_df = pd.read_csv(f'{models_fname}/{dependent_var}_model_results.csv')
best_model_row = results_df.iloc[0]
best_model = best_model_row['Model']
best_predictors = best_model_row['Predictors']

# -----------------------
# Plotting Figure 3a and 3b:
# -----------------------
# --- Plot beta coefficients ---
coefficients = best_model.params
std_err = best_model.bse
fvalue = best_model.fvalue
fp_value = best_model.f_pvalue

fig, ax = plt.subplots(figsize=(12, 8))
coefficients.plot(kind='bar', yerr=std_err, alpha=0.8, 
                  color='#191970', edgecolor='black')
plt.title(f'Beta Coefficients of the Best Model for {dependent_var}', fontsize=14, fontweight='bold')
plt.ylabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.xlabel('Predictors', fontsize=12, fontweight='bold')
plt.axhline(0, color='k', linestyle='--', linewidth=1)

# Add text annotations with goodness-of-fit statistics
text = (f"AIC: {best_model.aic:.2f}\n"
        f"BIC: {best_model.bic:.2f}\n"
        f"Adjusted R²: {best_model.rsquared_adj:.3f}\n"
        f"fvalue: {fvalue:.3f}\n"
        f"p-value: {fp_value:.3f}")
plt.text(-0.3, coefficients.min() * 1.1, text, fontsize=12, color='black', 
         bbox=dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round,pad=1'))

# Add coeficient table to this plot
coef_table = best_model.summary2().tables[1]  # This gives the coefficient table
coef_table_rounded = coef_table.round(3).astype(str)

# Format each row into a string
coef_table_text = '\n'.join([
    f"{idx}: coef = {row['Coef.']}, SE = {row['Std.Err.']}, t = {row['t']}, "
    f"p = {row['P>|t|']}, CI = [{row['[0.025']}, {row['0.975]']}]"
    for idx, row in coef_table_rounded.iterrows()
])

# Add coefficient table below the stats box
plt.text(-0.3, coefficients.min() * 1.3, coef_table_text, fontsize=11, color='black',
         bbox=dict(facecolor='whitesmoke', alpha=0.85, edgecolor='dimgray', boxstyle='round,pad=1'))
plt.tight_layout()

fig.savefig(f'{save_path}/Figure3a_Putamen_GLM.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure3a_Putamen_GLM.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure3a_Putamen_GLM.tiff', format='tiff', dpi=800, bbox_inches='tight')
plt.show()

# --- Partial regression plot ---
# === Extract data used in model ===
exog_names = best_model.model.exog_names
endog_name = best_model.model.endog_names

# Create DataFrame
X = pd.DataFrame(best_model.model.exog, columns=exog_names)
y = pd.Series(best_model.model.endog, name=endog_name)
data = pd.concat([y, X], axis=1)

# === Manually compute partial regression residuals ===

# Residual of y ~ other predictors
other_X = X.drop(columns=['const', 'Puta'])
model_y = sm.OLS(y, sm.add_constant(other_X)).fit()
y_resid = model_y.resid

# Residual of Puta ~ other predictors
model_x = sm.OLS(X['Puta'], sm.add_constant(other_X)).fit()
x_resid = model_x.resid

# === Plotting ===
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter with custom style
sns.regplot(
    x=x_resid, y=y_resid,
    scatter=True,
    ci=95,
    ax=ax,
    color='#191970',
    scatter_kws={'s': 60, 'edgecolor': 'black', 'linewidths': 0.6},
    line_kws={'color': 'black', 'lw': 1.5}
)

# === Annotate and format ===
ax.set_xlabel(r'LV$_{\textbf{\textrm{Puta}}}$', fontsize=12, fontweight='bold')
ax.set_ylabel('Landmark PSE', fontsize=12, fontweight='bold')
ax.set_title('Partial Regression Plot for Putamen', fontsize=14, fontweight='bold')

sns.despine()
plt.tight_layout()
fig.savefig(f'{save_path}/Figure3b_Putamen_GLM_partregress.svg', format='svg', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure3b_Putamen_GLM_partregress.png', format='png', dpi=800, bbox_inches='tight')
fig.savefig(f'{save_path}/Figure3b_Putamen_GLM_partregress.tiff', format='tiff', dpi=800, bbox_inches='tight')

# -----------------------
# Plotting Figure 4 and 5:
# -----------------------
# Mediation & Moderation:
med_df = pd.read_csv(f'{models_fname}/{dependent_var}_mediator-{mediator}.csv')
moderation_df = pd.read_csv(f'{models_fname}/{dependent_var}_moderator-{mediator}.csv')

# Convert SE column to NumPy array
y_errors = med_df['SE'].values  # Ensure it's a 1D NumPy array

# --- Mediation Analysis Results ---
plt.figure(figsize=(10, 6))
sns.barplot(data=med_df, x='Subcortical', y='Indirect_Effect', palette="viridis", errorbar=None)  # Disable automatic error bars
# Manually add error bars
plt.errorbar(x=range(len(med_df)), y=med_df['Indirect_Effect'], yerr=y_errors, fmt='none', capsize=5, color='black')
plt.title(f'Mediation Analysis: Indirect Effects {dependent_var}', fontsize=16)
for index, row in med_df.iterrows():
    plt.text(index, row['Indirect_Effect'], f"p = {row['p_value']:.3f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot moderation effects --
mod_coefficients = moderation_df['coefficients']
mod_std_err = moderation_df['standard_error']
mod_fvalue = moderation_df['fvalue']
mod_fp_value = moderation_df['f_pvalue']
mod_rsquared_adj = moderation_df['rsquared_adj']

plt.figure(figsize=(12, 8))
mod_coefficients.plot(kind='bar', yerr=mod_std_err, color='skyblue', alpha=0.8, edgecolor='black')
plt.title(f'Beta Coefficients of moderation in the Best Model for {dependent_var}', fontsize=16)
plt.ylabel('Coefficient Value')
plt.xlabel('Predictors')
plt.axhline(0, color='red', linestyle='--', linewidth=1)

# Add text annotations with goodness-of-fit statistics
text = (f"Adjusted R²: {mod_rsquared_adj:.3f}\n"
        f"fvalue: {mod_fvalue:.3f}\n"
        f"fp-value: {mod_fp_value:.3f}")
plt.text(-1.5, mod_coefficients.min() * 0.8, text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
