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





def moderation_analysis(data, dependent_var, independent_vars, moderator_var):
    """
    Perform moderation analysis by adding interaction terms between independent variables and a moderator.

    Parameters:
    data (pd.DataFrame): The dataset containing the variables.
    dependent_var (str): The name of the dependent variable (Y).
    independent_vars (list): A list of independent variables (X1, X2, ...).
    moderator_var (str): The name of the moderator variable (Z).

    Returns:
    dict: A dictionary containing regression results and coefficients.

    A significant coefficient of the interaction terms indicates a moderation effect, meaning 
    the relationship between the independent variable and the dependent variable 
    varies depending on the moderator's value.
    """
    results = {}
    interaction_terms = []

    # Create interaction terms between each independent variable and the moderator
    for iv in list(independent_vars):
        interaction_term = f'{iv}_x_{moderator_var}'
        data[interaction_term] = data[iv] * data[moderator_var]
        interaction_terms.append(interaction_term)

    # Define the predictors: independent variables, moderator, and interaction terms
    predictors = list(independent_vars) + [moderator_var] + interaction_terms
    X = sm.add_constant(data[predictors])

    # Fit the regression model
    model = sm.OLS(data[dependent_var], X).fit()
    coefficients = model.params

    # Save
    results['model_summary'] = model.summary()
    results['coefficients'] = coefficients
    results['p_values'] = model.pvalues
    results['standard_error'] = model.bse
    results['fvalue'] = model.fvalue
    results['f_pvalue'] = model.f_pvalue
    results['rsquared_adj'] = model.rsquared_adj

    return results


# Define paths
platform = 'mac'
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/behaviour'
# op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/collated')
lat_index_csv = op.join(volume_sheet_dir, 'unified_behavioral_structural_asymmetry_lateralisation_indices_1_45.csv')
pairplot_figname = op.join(volume_sheet_dir, 'pair_plot2')
mediators_fname = op.join(volume_sheet_dir, 'mediators2')
models_fname = op.join(volume_sheet_dir, 'model_results2')
res_figname = op.join(models_fname, 'residuals2')
qqplot_figname = op.join(models_fname, 'qqplot2')
coefficient_figname = op.join(models_fname, 'beta_coefficients2')
regresplot_figname = op.join(models_fname, 'partial_regression2')
mediationplot_figname = op.join(models_fname, 'mediation2')
mod_coefficient_figname = op.join(models_fname, 'moderation2')

report_all_methods = True  # do you want to report best 5 models with all methods?
plotting = True

# Step 1: Load the CSV file
data_full = pd.read_csv(lat_index_csv)

# Check the data
print(data_full.head())

# Step 2: Define dependent and independent variables
dep_vars = {'Target': 'Target_PSE_Laterality', 'Landmark': 'Landmark_PSE'}
independent_vars = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']  
# Define the mediator variable (microsaccade laterality)
mediator =  {'Target': 'Target_MS_Laterality', 'Landmark': 'Landmark_MS'}

# Remove NaNs from dependent variable (but keep rows in the dataset)
dependent_var = input(f'which dependent variable to do now? {dep_vars.keys()}\n (Do not add quotation marks!)\n')
print(f'\nRunning models on {dependent_var}')
data = data_full.dropna(subset=[dep_vars[dependent_var]] + independent_vars + [mediator[dependent_var]])

# Step 3: Visualize relationships 
plot_cols = independent_vars + [mediator[dependent_var]] + [dep_vars[dependent_var]]
g = sns.pairplot(data[plot_cols])
g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
g.fig.suptitle(f'Pairplot of Variables ({dependent_var})', y=0.95, fontsize=16)

# Loop through the axes of the pairplot to calculate and annotate Spearman correlations
for i, j in zip(*np.triu_indices_from(g.axes, k=1)):  # Loop over upper triangle
    ax = g.axes[i, j]
    if ax is not None:
        x_var = plot_cols[j]  # Get column name from plot_cols
        y_var = plot_cols[i]  

        # Check if both variables exist in the DataFrame
        if x_var in data.columns and y_var in data.columns:
            # Calculate Spearman correlation
            spearman_corr, spearman_p = stats.spearmanr(data[x_var], data[y_var], nan_policy='omit')

            # Format p-value for readability
            p_text = f"p < 0.001" if spearman_p < 0.001 else f"p = {spearman_p:.3f}"

            # Annotate correlation on the plot
            ax.text(0.05, 0.95, f"ρ = {spearman_corr:.2f} \n{p_text}", 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle="round", fc="beige", alpha=0.8))
plt.savefig(f'{pairplot_figname}_{dependent_var}.png')
plt.show()

# Step 4: Check for multicollinearity using VIF
X = data[independent_vars + [mediator[dependent_var]]]
vif = calculate_vif(X)
print("Variance Inflation Factor (VIF):\n", vif)

# Step 5: Generate interaction terms
"""don't use the create_interaction_terms,
doesn't make sense to add all possible interaction combinations"""
moderator = mediator[dependent_var]
# those interactions that make sense to me
interaction_terms = ['Caud*Puta', 'Caud*Pall', 
                     'Pall*Puta', 'Thal*Puta',
                     'Caud*Puta*Pall']

all_terms = independent_vars + [moderator] + interaction_terms  # Include main effects and interactions

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
        y = data[dep_vars[dependent_var]]

        try:
            model = sm.OLS(y, X).fit()  # Fit the model

            results.append({
                'Predictors': combo,
                'AIC': model.aic,
                'BIC': model.bic,
                'LogLik': model.llf,
                'Adj_R2': model.rsquared_adj,
                'Model_summary': model.summary(),
                'Model': model,
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
best_model_summary = best_model_row['Model_summary']
best_predictors = best_model_row['Predictors']

print("\nBest Model Predictors Based on AIC:", best_predictors)
print("\nBest Model Summary Base on AIC:\n", best_model_summary)

if report_all_methods:
    # Confirm findings with BIC, Log-Likelihood, and R²
    print("\nTop 5 Models by AIC:\n", results_df.nsmallest(5, 'AIC'))
    print("\nTop 5 Models by BIC:\n", results_df.nsmallest(5, 'BIC'))
    print("\nTop 5 Models by Log-Likelihood:\n", results_df.nlargest(5, 'LogLik'))
    print("\nTop 5 Models by Adjusted R²:\n", results_df.nlargest(5, 'Adj_R2'))


# Step 8: Mediation Analysis for microsaccades on the regresors of best model
mediation_results = {f'{dependent_var}': []}

for regres in best_predictors:
    # -----------------------
    # Mediation Analysis:
    # -----------------------
    try:
        # Pingpuin method
        med_result = pg.mediation_analysis(data=data, x=regres, 
                                            m=mediator[dependent_var], 
                                            y=dep_vars[dependent_var], n_boot=5000)
        print(f'Extracting the indirect effect for {regres} on {dep_vars[dependent_var]}')
        indirect_effect = med_result.loc[med_result['path'] == 'Indirect', 'coef'].values[0]
        indirect_p = med_result.loc[med_result['path'] == 'Indirect', 'pval'].values[0]
        indirect_se = med_result.loc[med_result['path'] == 'Indirect', 'se'].values[0]  # Extract standard error

        # Baron and Kenny's (1986) method:
        results, coefficients = baron_kenny_mediation(data, regres, mediator[dependent_var], dep_vars[dependent_var])

    except Exception as e:
        print(f"Mediation analysis error for {regres} on {dep_vars[dependent_var]}: {e}")
        indirect_effect = np.nan
        indirect_p = np.nan
        indirect_se = np.nan  # Handle missing SE values
    
    mediation_results[dependent_var].append({
        'Subcortical': regres,
        'Indirect_Effect': indirect_effect,
        'p_value': indirect_p,
        'SE': indirect_se,  # Store standard error in results
        'baron_kenny_results': results,
        'baron_kenny_coefs': coefficients,
        
    })

