
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


def baron_kenny_mediation(data, independent_var, mediator_var, dependent_var):
    """
    Conduct mediation analysis following Baron and Kenny's three-step approach.

    The analysis involves:
      Step 1: Regressing the dependent variable (Y) on the independent variable (X)
              to assess the total effect.
      Step 2: Regressing the mediator (Me) on the independent variable (X) to confirm
              that X predicts Me.
      Step 3: Regressing Y on both X and Me to determine the direct effect of X on Y
              and the effect of the mediator.
    
    Parameters:
    data (pd.DataFrame): Dataset containing all relevant variables.
    independent_var (str): Name of the independent variable (X).
    mediator_var (str): Name of the mediator variable (Me).
    dependent_var (str): Name of the dependent variable (Y).

    Returns:
    tuple: A dictionary with regression summaries for each step and a dictionary with
           key coefficients from the models.
    """

    results = {}

    # Step 1: Regress Y on X
    X = sm.add_constant(data[independent_var])
    model1 = sm.OLS(data[dependent_var], X).fit()
    results['Step 1'] = model1.summary()
    beta_11 = model1.params[independent_var]

    # Step 2: Regress Me on X
    model2 = sm.OLS(data[mediator_var], X).fit()
    results['Step 2'] = model2.summary()
    beta_21 = model2.params[independent_var]

    # Step 3: Regress Y on X and Me
    X_Me = sm.add_constant(data[[independent_var, mediator_var]])
    model3 = sm.OLS(data[dependent_var], X_Me).fit()
    results['Step 3'] = model3.summary()
    beta_31 = model3.params[independent_var]
    beta_32 = model3.params[mediator_var]

    # Compile coefficients
    coefficients = {
        'beta_11 (X -> Y)': beta_11,
        'beta_21 (X -> Me)': beta_21,
        'beta_31 (X -> Y | Me)': beta_31,
        'beta_32 (Me -> Y | X)': beta_32
    }

    return results, coefficients

def moderation_analysis(data, dependent_var, independent_vars, moderator_var):
    """
    Perform moderation analysis by testing interaction effects between independent variables and a moderator.

    The function creates interaction terms between each independent variable and the moderator,
    then fits a regression model including main effects and these interactions. A significant
    interaction term suggests that the effect of an independent variable on the dependent variable
    is moderated by the moderator variable.

    Parameters:
    data (pd.DataFrame): Dataset containing all relevant variables.
    dependent_var (str): Name of the dependent variable (Y).
    independent_vars (list): List of independent variable names.
    moderator_var (str): Name of the moderator variable (e.g., microsaccades).

    Returns:
    dict: A dictionary containing the regression model summary, coefficients, p-values, standard errors,
          F-statistic, its p-value, and adjusted R-squared value.
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
# BEAR outage
# volume_sheet_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/behaviour'
volume_sheet_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/data/collated')
lat_index_csv = op.join(volume_sheet_dir, 'FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_eye-dominance.csv')
# Save figure in BEAR outage (that's where the latest version of the manuscript is)
save_path = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/landmark-manus/Figures'

# Step 1: Load the CSV file
data_full = pd.read_csv(lat_index_csv)
print(data_full.head())

# Step 2: Define dependent and independent variables
dep_vars = {'Landmark': 'Landmark_PSE'}
independent_var = ['Puta']

# Remove NaNs from dependent variable (but keep rows in the dataset)
dependent_var = 'Landmark'
data = data_full.dropna(subset=[dep_vars[dependent_var]] + independent_var)

# Step 3: Define the mediator variable (microsaccade laterality)
mediator =  {'Landmark': 'Handedness'}  #'Landmark_MS'} 'Eye_Dominance'}
moderator = mediator[dependent_var]

# Step 4: fit GLM for Putamen and landmark pse
results = []

# Prepare the model data
X = data[list(independent_var)].copy()
X = sm.add_constant(X)  # Add intercept
y = data[dep_vars[dependent_var]]

model = sm.OLS(y, X).fit()  # Fit the model

results.append({
    'Predictors': independent_var,
    'AIC': model.aic,
    'BIC': model.bic,
    'LogLik': model.llf,
    'Adj_R2': model.rsquared_adj,
    'Model_summary': model.summary(),
    'Model': model,
})
# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)
results_df.sort_values(by='AIC', inplace=True)  # can change AIC to something else here

best_model_row = results_df.iloc[0]
best_model = best_model_row['Model']
best_model_summary = best_model_row['Model_summary']
best_predictors = best_model_row['Predictors']

# Step 5: Mediation Analysis for microsaccades on the regresors of best model
"""only remove nans from mediator here, not at the beginning"""
data = data_full.dropna(subset=[dep_vars[dependent_var]] + independent_var + [mediator[dependent_var]])
mediation_results = {f'{dependent_var}': []}

for regres in best_predictors:
    # -----------------------
    # Mediation Analysis:
    # -----------------------
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
    
    mediation_results[dependent_var].append({
        'Subcortical': regres,
        'Indirect_Effect': indirect_effect,
        'p_value': indirect_p,
        'SE': indirect_se,  # Store standard error in results
        'baron_kenny_results': results,
        'baron_kenny_coefs': coefficients,
        
    })

# Convert results to DataFrames for plotting
med_df = pd.DataFrame(mediation_results[dependent_var])

# -----------------------
# Moderation Analysis:
# -----------------------
moderation_results = {f'{dependent_var}': []}
moderation_results[dependent_var] = moderation_analysis(data, dep_vars[dependent_var], best_predictors, mediator[dependent_var])
moderation_df = pd.DataFrame(moderation_results[dependent_var])

# -----------------------
# Plotting:
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


# ---Plot Mediation Analysis Results ---
# Read mediation results
med_df = pd.read_csv

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


# --- Plot moderation effects ---
mod_coefficients = moderation_results[dependent_var]['coefficients']
mod_std_err = moderation_results[dependent_var]['standard_error']
mod_fvalue = moderation_results[dependent_var]['fvalue']
mod_fp_value = moderation_results[dependent_var]['f_pvalue']
mod_rsquared_adj = moderation_results[dependent_var]['rsquared_adj']

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
