"""
===============================================
C02_systematic_model_selection - Mediation and Moderation Analyses
--------------------------------------------------------
This script calculates for each subcortical region the mediation and moderation 
effects of microsaccade laterality ("MicroLat") on two behavioral laterality measures:
   - Target behavior (e.g., 'Target_PSE_Laterality')
   - Landmark behavior (e.g., 'Landmark_PSE')
The script performs the following steps:
1. Loads the data and defines the dependent variables, independent subcortical variables, and the mediator.
2. Visualizes relationships between variables using a pairplot.
3. Checks multicollinearity using Variance Inflation Factor (VIF).
4. For each subcortical region, performs:
   a. Mediation analysis using Pingouin (testing if MicroLat mediates the effect on behavior).
   b. Moderation analysis using an OLS regression model with interaction (testing if MicroLat moderates the effect on behavior).
5. Reports the results and plots the mediation (indirect effects) and moderation (interaction coefficients) results separately for target and landmark behavior.
   
Written by Tara Ghafari
===============================================
"""

# Import necessary libraries
import os.path as op
import pandas as pd
import numpy as np
from itertools import combinations, chain
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pingouin as pg
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def calculate_vif(X):
    """
    Check for multicollinearity using VIF
    Drop variables with high VIF (>5 as a rule of thumb)
    """
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Define paths
platform = 'mac'
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/behaviour'
lat_index_csv = op.join(volume_sheet_dir, 'unified_behavioral_structural_asymmetry_lateralisation_indices_1_45.csv')
pairplot_figname = op.join(volume_sheet_dir, 'pair_plot2')
models_fname = op.join(volume_sheet_dir, 'model_results2')
res_figname = op.join(models_fname, 'residuals2')
qqplot_figname = op.join(models_fname, 'qqplot2')
coefficient_figname = op.join(models_fname, 'beta_coefficients2')
regresplot_figname = op.join(models_fname, 'partial_regression2')

# Step 1: Load the CSV file
data_full = pd.read_csv(lat_index_csv)
print("Data Head:\n", data_full.head())

# Define dependent variables 
dep_vars = {'Target': 'Target_PSE_Laterality', 'Landmark': 'Landmark_PSE'}
# Define subcortical laterality variables (independent variables of interest)
independent_vars = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
# Define the mediator variable (microsaccade laterality)
mediator =  {'Target': 'Target_MS_Laterality', 'Landmark': 'Landmark_MS'}

# Remove rows with NaN in any of the key columns (dependent variables, subcortical variables, mediator)
dependent_var = input(f'which dependent variable to do now? {dep_vars.keys()}\n (Do not add quotation marks!)\n')
data = data_full.dropna(subset=[dep_vars[dependent_var]] + independent_vars + [mediator[dependent_var]])

# Step 2: Visualize relationships between the variables using a pairplot (for one dependent variable, for example)
plot_cols = independent_vars + [mediator[dependent_var]] + [dep_vars[dependent_var]]
g = sns.pairplot(data[plot_cols])
g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
g.fig.suptitle(f'Pairplot of Variables ({dependent_var} Behavior)', y=0.95, fontsize=16)
# Loop through the axes of the pairplot to calculate and annotate Spearman correlations
for i, j in zip(*np.triu_indices_from(g.axes, k=1)):  # Loop over upper triangle
    ax = g.axes[i, j]
    if ax is not None:
        x_var = independent_vars[j] if j < len(independent_vars) else dep_vars[dependent_var]
        y_var = independent_vars[i] if i < len(independent_vars) else dep_vars[dependent_var]
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(data[x_var], data[y_var])
        
        # Annotate correlation on the plot
        ax.text(0.05, 0.95, f"ρ = {spearman_corr:.2f} \np = {spearman_p:.2f}", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
plt.savefig(f'{pairplot_figname}_{dependent_var}.png')
plt.show()

# Step 3: Check for multicollinearity using VIF on the subcortical predictors
X_vif = data[independent_vars]
vif = calculate_vif(X_vif)
print("Variance Inflation Factor (VIF) for subcortical variables:\n", vif)

# Step 4: Mediation and Moderation Analysis for each dependent variable and each subcortical region

# Dictionaries to store results
mediation_results = {'Target': [], 'Landmark': []}
moderation_results = {'Target': [], 'Landmark': []}

for substr in independent_vars:
    # -----------------------
    # Mediation Analysis:
    # -----------------------
    try:
        med_result = pg.mediation_analysis(data=data, x=substr, 
                                            m=mediator[dependent_var], 
                                            y=dep_vars[dependent_var], n_boot=5000)
        print(f'Extracting the indirect effect for {substr} on {dep_vars[dependent_var]}')
        indirect_effect = med_result.loc[med_result['path'] == 'Indirect', 'coef'].values[0]
        indirect_p = med_result.loc[med_result['path'] == 'Indirect', 'pval'].values[0]
    except Exception as e:
        print(f"Mediation analysis error for {substr} on {dep_vars[dependent_var]}: {e}")
        indirect_effect = np.nan
        indirect_p = np.nan
    
    mediation_results[dependent_var].append({
        'Subcortical': substr,
        'Indirect_Effect': indirect_effect,
        'p_value': indirect_p
    })
    
    # -----------------------
    # Moderation Analysis:
    # -----------------------
    # Fit an OLS model with an interaction term: dep_var ~ sub + mediator + sub:mediator
    formula = f"{dep_vars[dependent_var]} ~ {substr} * {mediator[dependent_var]}"
    try:
        mod_model = smf.ols(formula, data=data).fit()
        # The interaction term is typically named "sub:mediator" or "mediator:sub"
        interaction_term = f"{substr}:{mediator[dependent_var]}"
        if interaction_term not in mod_model.params.index:
            interaction_term = f"{mediator[dependent_var]}:{substr}"
        interaction_coef = mod_model.params[interaction_term]
        interaction_p = mod_model.pvalues[interaction_term]
    except Exception as e:
        print(f"Moderation analysis error for {substr} on {dependent_var}: {e}")
        interaction_coef = np.nan
        interaction_p = np.nan
    
    moderation_results[dependent_var].append({
        'Subcortical': substr,
        'Interaction_Coefficient': interaction_coef,
        'p_value': interaction_p
    })

# Convert results to DataFrames for plotting
med_df_target = pd.DataFrame(mediation_results['Target'])
med_df_landmark = pd.DataFrame(mediation_results['Landmark'])

mod_df_target = pd.DataFrame(moderation_results['Target'])
mod_df_landmark = pd.DataFrame(moderation_results['Landmark'])

# Step 5: Plot Mediation Analysis Results
# Bar plot for indirect effects for Target behavior
plt.figure(figsize=(10, 6))
sns.barplot(data=med_df_target, x='Subcortical', y='Indirect_Effect', palette="viridis")
plt.title('Mediation Analysis: Indirect Effects (Target Behavior)', fontsize=16)
for index, row in med_df_target.iterrows():
    plt.text(index, row['Indirect_Effect'], f"p = {row['p_value']:.3f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig("mediation_target.png")
plt.show()

# Bar plot for indirect effects for Landmark behavior
plt.figure(figsize=(10, 6))
sns.barplot(data=med_df_landmark, x='Subcortical', y='Indirect_Effect', palette="viridis")
plt.title('Mediation Analysis: Indirect Effects (Landmark Behavior)', fontsize=16)
for index, row in med_df_landmark.iterrows():
    plt.text(index, row['Indirect_Effect'], f"p = {row['p_value']:.3f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig("mediation_landmark.png")
plt.show()

# Step 6: Plot Moderation Analysis Results
# Bar plot for interaction coefficients for Target behavior
plt.figure(figsize=(10, 6))
sns.barplot(data=mod_df_target, x='Subcortical', y='Interaction_Coefficient', palette="magma")
plt.title('Moderation Analysis: Interaction Coefficients (Target Behavior)', fontsize=16)
for index, row in mod_df_target.iterrows():
    plt.text(index, row['Interaction_Coefficient'], f"p = {row['p_value']:.3f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig("moderation_target.png")
plt.show()

# Bar plot for interaction coefficients for Landmark behavior
plt.figure(figsize=(10, 6))
sns.barplot(data=mod_df_landmark, x='Subcortical', y='Interaction_Coefficient', palette="magma")
plt.title('Moderation Analysis: Interaction Coefficients (Landmark Behavior)', fontsize=16)
for index, row in mod_df_landmark.iterrows():
    plt.text(index, row['Interaction_Coefficient'], f"p = {row['p_value']:.3f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig("moderation_landmark.png")
plt.show()
