"""
===============================================
C02_systematic_model_selection

this code will do:

GLM Analysis with Interaction Terms and Model Selection
--------------------------------------------------------
This script performs the following steps:
1. Loads the data and defines dependent and independent variables.
2. Visualizes relationships between variables using pairplots.
3. Checks multicollinearity using Variance Inflation Factor (VIF).
4. Creates interaction terms between predictors.
5. Automates GLM fitting for all combinations of predictors 
    (main effects and interactions).
6. Selects the best model based on AIC and confirms with BIC, 
    log-likelihood, and adjusted R².
7. Visualizes results, including:
   - Bar plot of beta coefficients with model statistics.
   - Partial regression plots for predictors in the best model.

written by Tara Ghafari
===============================================
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import os.path as op

from itertools import combinations, chain
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress_grid
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_vif(X):
    """
    Check for multicollinearity using VIF
    Drop variables with high VIF (>5 as a rule of thumb)
    """

    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def create_interaction_terms(variables):
    """
    Create all interaction terms of the independent variables,
    considering 2-way to n-way interactions.
    r defines n, can be from 2 to 7

    it makes more sense to only add interactions that are meaningful.
    """
    interactions = []
    for r in range(2, 3 + 1):  # Generate 2-way to n-way interactions
        interactions.extend(['*'.join(comb) for comb in combinations(variables, r)])
    return interactions

# Define paths
platform = 'mac'
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/substr-beh'
# op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/collated')
lat_index_csv = op.join(volume_sheet_dir, 'unified_behavioral_structural_asymmetry_lateralisation_indices_1_45.csv')
pairplot_figname = op.join(volume_sheet_dir, 'pair_plot')
models_fname = op.join(volume_sheet_dir, 'model_results')
res_figname = op.join(volume_sheet_dir, 'residuals')
qqplot_figname = op.join(volume_sheet_dir, 'qqplot')
coefficient_figname = op.join(volume_sheet_dir, 'beta_coefficients')
regresplot_figname = op.join(volume_sheet_dir, 'partial_regression')

report_all_methods = True  # do you want to report best 5 models with all methods?
plotting = False

# Step 1: Load the CSV file
data_full = pd.read_csv(lat_index_csv)

# Check the data
print(data_full.head())

# Step 2: Define dependent and independent variables
dependent_vars = ['Landmark_PSE', 'Target_PSE_Laterality', 'Landmark_MS', 'Target_MS_Laterality']
independent_vars = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']  
dependent_var = input(f'which dependent variable to do now? {dependent_vars}\n (Do not add quotation marks!)\n')

print(f'\nRunning models on {dependent_var}')
# Remove NaNs from dependent variable (but keep rows in the dataset)
data = data_full.dropna(subset=[dependent_var])

# Step 3: Visualize relationships 
g = sns.pairplot(data[independent_vars + [dependent_var]])
g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
g.fig.suptitle(f'Pairplot of Variables Modeling {dependent_var}', y=0.95, fontsize=16)
# Loop through the axes of the pairplot to calculate and annotate Spearman correlations
for i, j in zip(*np.triu_indices_from(g.axes, k=1)):  # Loop over upper triangle
    ax = g.axes[i, j]
    if ax is not None:
        x_var = independent_vars[j] if j < len(independent_vars) else dependent_var
        y_var = independent_vars[i] if i < len(independent_vars) else dependent_var
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(data[x_var], data[y_var])
        
        # Annotate correlation on the plot
        ax.text(0.05, 0.95, f"ρ = {spearman_corr:.2f} \np = {spearman_p:.2f}", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
plt.savefig(f'{pairplot_figname}_{dependent_var}.png')

# Step 4: Check for multicollinearity using VIF
X = data[independent_vars]
vif = calculate_vif(X)
print("Variance Inflation Factor (VIF):\n", vif)

# Step 5: Generate interaction terms
"""don't use the create_interaction_terms,
doesn't make sense to add all possible interaction combinations"""
interaction_terms = ['Caud*Puta', 'Caud*Pall', 'Pall*Puta', 'Caud*Puta*Pall']  # those interactions that make sense to me
all_terms = independent_vars + interaction_terms  # Include main effects and interactions

# Add interaction terms to the DataFrame
for term in interaction_terms:
    variables = term.split('*')
    data[term] = data[variables].prod(axis=1)  # Multiply corresponding columns to create interaction terms

# Step 6: Automate GLM fitting for all combinations of variables
results = []

# Generate all possible subsets of predictors
for i in range(1, len(all_terms) + 1):
    for combo in combinations(all_terms, i):
        # Check if interaction terms have their respective main effects included
        main_effects = set(chain.from_iterable(var.split('*') for var in combo if '*' in var))  # Extract variables in interactions
        if not main_effects.issubset(set(combo)):
            continue  # Skip invalid combinations (interaction terms without their main effects)

        # Prepare the model data
        X = data[list(combo)].copy()
        X = sm.add_constant(X)  # Add intercept
        y = data[dependent_var]

        try:
            model = sm.OLS(y, X).fit()  # Fit the model

            results.append({
                'Predictors': combo,
                'AIC': model.aic,
                'BIC': model.bic,
                'LogLik': model.llf,
                'Adj_R2': model.rsquared_adj,
                'Model': model
            })
        except Exception as e:
            # Skip combinations that fail (e.g., singular matrices)
            print(f"Model failed for predictors: {combo}. Error: {e}")
            continue

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)
results_df.sort_values(by='AIC', inplace=True)  # can change AIC to something else here
results_df.to_csv(f'{models_fname}_{dependent_var}.csv')

# Step 7: Analyze the best model based on AIC
best_model_row = results_df.iloc[0]
best_model = best_model_row['Model']
best_predictors = best_model_row['Predictors']

print("\nBest Model Predictors Based on AIC:", best_predictors)
print("\nBest Model Summary Base on AIC:\n", best_model.summary())

if report_all_methods:
    # Confirm findings with BIC, Log-Likelihood, and R²
    print("\nTop 5 Models by AIC:\n", results_df.nsmallest(5, 'AIC'))
    print("\nTop 5 Models by BIC:\n", results_df.nsmallest(5, 'BIC'))
    print("\nTop 5 Models by Log-Likelihood:\n", results_df.nlargest(5, 'LogLik'))
    print("\nTop 5 Models by Adjusted R²:\n", results_df.nlargest(5, 'Adj_R2'))


if plotting: 
        
    #################################### PLOTTING #############################################
    y_pred = best_model.predict(sm.add_constant(data[list(best_predictors)]))
    residuals = data[dependent_var] - y_pred  # more complicated way of doing residuals=best_model.resid

    # --- Scatter Plot for Residuals ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Predicted Values Modeling {dependent_var}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(f'{res_figname}_{dependent_var}.png')

    # --- Q-Q Plot for Residuals ---
    plt.figure(figsize=(8, 6))
    qqplot(residuals, line='45', fit=True, alpha=0.5, color='blue')
    plt.title(f'Q-Q Plot of Residuals Modeling {dependent_var}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{qqplot_figname}_{dependent_var}.png')  # Save the Q-Q plot

    # --- Plot beta coefficients ---
    coefficients = best_model.params
    std_err = best_model.bse
    fvalue = best_model.fvalue
    fp_value = best_model.f_pvalue

    plt.figure(figsize=(12, 8))
    coefficients.plot(kind='bar', yerr=std_err, color='skyblue', alpha=0.8, edgecolor='black')
    plt.title(f'Beta Coefficients of the Best Model for {dependent_var}', fontsize=16)
    plt.ylabel('Coefficient Value')
    plt.xlabel('Predictors')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    # Add text annotations with goodness-of-fit statistics
    text = (f"AIC: {best_model.aic:.2f}\n"
            f"BIC: {best_model.bic:.2f}\n"
            f"Adjusted R²: {best_model.rsquared_adj:.3f}\n"
            f"fvalue: {fvalue:.3f}\n"
            f"fp-value: {fp_value:.3f}")
    plt.text(-1.5, coefficients.min() * 0.8, text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig(f'{coefficient_figname}_{dependent_var}.png')

    # --- Partial regression plot ---
    plt.figure(figsize=(12, 8))
    plot_partregress_grid(best_model)
    plt.suptitle(f'Partial Regression Plots Modeling {dependent_var}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{regresplot_figname}_{dependent_var}.png')
