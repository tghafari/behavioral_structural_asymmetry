# -*- coding: utf-8 -*-
"""
===============================================
02_selecting_model_with_interactions

This code reads a CSV file containing lateralization 
indices of substructures, PSEs, and MS.
It fits models to predict behavioral 
measures (PSE/MS laterality) using structural measures
and their interactions, then selects the best models 
based on AIC, BIC, log-likelihood, 
and adjusted R-squared.


written by Tara Ghafari
===============================================
"""# -*- coding: utf-8 -*-
"""
================================================

02_selecting_model_with_interactions_and_combinations

This script evaluates all possible combinations of independent variables (structural columns)
and their interactions to predict behavioral measures. It identifies the best models based on
AIC, BIC, log-likelihood, and adjusted R-squared.

Updated by Assistant
================================================
"""

import pandas as pd
import numpy as np
import os.path as op
import itertools
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

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

# Define columns for analysis
structural_columns = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
behavioural_columns = ['Landmark_PSE', 'Target_PSE_Laterality', 'Landmark_MS', 'Target_MS_Laterality']

def E2_ModelSelection(lat_index_csv):
    data = pd.read_csv(lat_index_csv)
    results = {}

    for dependent in behavioural_columns:
        print(f"\nAnalyzing dependent variable: {dependent}")
        
        # Drop NaNs in the dependent variable
        data_filtered = data.dropna(subset=[dependent])
        
        # Prepare data
        y = data_filtered[dependent]
        X = data_filtered[structural_columns]

        # Generate all unique combinations of independent variables
        nReg = [list(itertools.combinations(range(len(structural_columns)), n_regr)) for n_regr in range(1, len(structural_columns) + 1)]

        def preallocate_metrics(nReg):
            return [np.full(len(combs), np.nan) for combs in nReg]

        LME = [[None] * len(combs) for combs in nReg]
        str_lbl = [[None] * len(combs) for combs in nReg]
        AIC, BIC, logLikelihood, Rsqrd_adjst = [preallocate_metrics(nReg) for _ in range(4)]

        # Fit models for all combinations
        for n_regr in range(len(nReg)):
            for j, reg_comb in enumerate(nReg[n_regr]):
                cols = [structural_columns[i] for i in reg_comb]
                interaction_terms = [":".join(comb) 
                                     for r in range(2, min(len(cols), 7)) 
                                     for comb in itertools.combinations(cols, r)]
                formula = f'{dependent} ~ ' + ' + '.join(cols + interaction_terms)

                model = smf.ols(formula, data=data_filtered).fit()
                LME[n_regr][j] = model
                str_lbl[n_regr][j] = cols + interaction_terms
                AIC[n_regr][j] = model.aic
                BIC[n_regr][j] = model.bic
                logLikelihood[n_regr][j] = model.llf
                Rsqrd_adjst[n_regr][j] = model.rsquared_adj

        # Find the best models based on each metric
        def get_best_metric(metric, is_min=True):
            best_values = np.full(len(metric), np.nan)
            best_labels = [None] * len(metric)
            for n_regr in range(len(metric)):
                best_idx = np.nanargmin(metric[n_regr]) if is_min else np.nanargmax(metric[n_regr])
                best_values[n_regr] = metric[n_regr][best_idx]
                best_labels[n_regr] = str_lbl[n_regr][best_idx]
            return best_values, best_labels

        best_AICs, AIC_labels = get_best_metric(AIC)
        best_BICs, BIC_labels = get_best_metric(BIC)
        best_LL, LL_labels = get_best_metric(logLikelihood, is_min=False)
        best_Rsqrd, Rsqrd_labels = get_best_metric(Rsqrd_adjst, is_min=False)

        results[dependent] = {
            'AIC': best_AICs,
            'AIC_labels': AIC_labels,
            'BIC': best_BICs,
            'BIC_labels': BIC_labels,
            'log_likelihood': best_LL,
            'LL_labels': LL_labels,
            'adjusted_R_squared': best_Rsqrd,
            'Rsqrd_labels': Rsqrd_labels,
        }

    return results

def plot_model_summary(model, title, labels):
    """Visualizes the model coefficients and their statistical significance."""
    params = model.params.iloc[1:]  # Exclude the intercept
    errors = model.bse.iloc[1:]
    pvalues = model.pvalues.iloc[1:]

    x = np.arange(len(params))
    plt.figure(figsize=(12, 6))
    plt.bar(x, params, yerr=errors, capsize=5, color='steelblue')
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x, labels, rotation=45, ha='right')

    for i, (param, error, pvalue) in enumerate(zip(params, errors, pvalues)):
        if pvalue < 0.05:
            plt.text(i, param + (error if param > 0 else -error), '*', ha='center', color='red', fontsize=14)

    plt.title(title)
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

# Run the model selection
results = E2_ModelSelection(lat_index_csv)

# Visualize the results
for dependent, result in results.items():
    print(f"\nResults for {dependent}:")
    for metric, values in result.items():
        if metric.endswith("_labels"):
            continue
        print(f"  {metric}: {values}")


# Visualize the results
for dependent, result in results.items():
    print(f"\nBest model for {dependent}:\n{result['model'].summary()}")
    plot_model_summary(result['model'], f"Model Summary: {dependent}", result['model'].params.index[1:])


















import pandas as pd
import numpy as np
import os.path as op
import itertools
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


platform = 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
volume_sheet_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives/collated')
lat_index_csv = op.join(volume_sheet_dir, 'lateralisation_vol-PSE_1_31.csv')

# What are you plotting? 
dependent = 'PSE_landmark'  # lateralised performance in 'PSE_landmark' or 'MS_target' or 'PSE_target'
independent = 'thomas'  # lateralisation volume of 'substr' or 'thomas'?

def E2_ModelSelection(lat_index_csv, dependent, independent):

    data = pd.read_csv(lat_index_csv)

    # which column of the data is the dependent variable? which rows are the outliers for that dependent?
    dependent_mapping = {
        'PSE_landmark': (8, [15, 27]),
        'MS_target': (9, []),
        'PSE_target': (10, [2, 3, 6, 7, 15, 16, 17, 24, 26, 27, 30])
    }
    
    dependent_column, outlier_idx = dependent_mapping[dependent]
    y = data.iloc[1:-1, dependent_column].astype(float).drop(outlier_idx).values  # 1:-1 is to remove header and the last row which is nan now

    independent_mapping = {
        'substr': data.iloc[1:-1, 1:8].astype(float),  # substr 
        'thomas': data.iloc[1:-1, 11:20].astype(float)  # sub regions of thalamus
    }

    X = independent_mapping[independent].drop(outlier_idx).values

    # Create all unique combinations of 'n' numbers, from 1 to 7
    nReg = [list(itertools.combinations(range(1, 8), n_regr)) for n_regr in range(1, 8)]  # shape: 7x21x35x35x21x7x1

    def preallocate_metrics(nReg):
        return [np.full(len(combs), np.nan) for combs in nReg]

    LME = [[None] * len(combs) for combs in nReg]
    str_lbl = [[None] * len(combs) for combs in nReg]
    AIC, BIC, logLikelihood, Rsqrd_adjst = [preallocate_metrics(nReg) for _ in range(4)]

    # Fit models
    for n_regr in range(7):
        for j, reg_comb in enumerate(nReg[n_regr]):
            cols = [i - 1 for i in reg_comb]
            design_matrix = X[:, cols]
            design_df = pd.DataFrame(design_matrix, columns=[f'X{i}' for i in reg_comb])
            design_df['y'] = y

            formula = 'y ~ ' + ' + '.join([f'X{i}' for i in reg_comb])
            model = smf.ols(formula, data=design_df).fit()
            LME[n_regr][j] = model
            str_lbl[n_regr][j] = [data.columns[i] for i in reg_comb]
            AIC[n_regr][j] = model.aic
            BIC[n_regr][j] = model.bic
            logLikelihood[n_regr][j] = model.llf
            Rsqrd_adjst[n_regr][j] = model.rsquared_adj

    # Find best models based on AIC and BIC
    def get_best_metrics(metric):
        best_metrics = np.zeros((7, 2))
        best_labels = [[None] * 7 for _ in range(7)]
        for n_regr in range(7):
            best_metrics[n_regr, 0] = np.nanmin(metric[n_regr])  # the minimum AIC value
            best_metrics[n_regr, 1] = np.nanargmin(metric[n_regr])  # the index of minimum AIC
            best_labels[n_regr][:n_regr+1] = str_lbl[n_regr][int(best_metrics[n_regr, 1])]
        return best_metrics, best_labels

    best_AICs, AICs_best_lbl = get_best_metrics(AIC)
    best_BICs, BICs_best_lbl = get_best_metrics(BIC)

    best_AIC = np.nanargmin(best_AICs[:, 0])
    AIC_best_lbl = AICs_best_lbl[best_AIC]
    AIC_best_lbl = [label for label in AIC_best_lbl if label is not None]  # Remove 'None' values

    best_BIC = np.nanargmin(best_BICs[:, 0])
    BIC_best_lbl = BICs_best_lbl[best_BIC]
    BIC_best_lbl = [label for label in BIC_best_lbl if label is not None]  # Remove 'None' values

    avg_AICs_BICs = np.nanmean([best_AICs[:, 0], best_BICs[:, 0]], axis=0)
    AB_best_n_reg = np.array([  # best number of regressors based on min avg (AIC,BIC)
        best_AIC + 1,  
        best_BIC + 1,
        np.nanargmin(avg_AICs_BICs) + 1  # I don't think we need an average
    ])
    # Find the summary of the best model based on AIC and BIC
    AIC_best_model = LME[best_AIC][int(best_AICs[best_AIC, 1])]
    BIC_best_model = LME[best_BIC][int(best_BICs[best_BIC, 1])]

    # Find best models based on log-likelihood and R-squared
    def get_best_metrics_max(metric):
        best_metrics = np.zeros((7, 2))
        best_labels = [[None] * 7 for _ in range(7)]
        for n_regr in range(7):
            best_metrics[n_regr, 0] = np.nanmax(metric[n_regr])
            best_metrics[n_regr, 1] = np.nanargmax(metric[n_regr])
            best_labels[n_regr][:n_regr+1] = str_lbl[n_regr][int(best_metrics[n_regr, 1])]
        return best_metrics, best_labels

    best_LLs, LLs_best_lbl = get_best_metrics_max(logLikelihood)
    best_Rsqrds, Rsqrds_best_lbl = get_best_metrics_max(Rsqrd_adjst)

    best_LL = np.nanargmax(best_LLs[:, 0])
    LL_best_lbl = LLs_best_lbl[best_LL]
    LL_best_lbl = [label for label in LL_best_lbl if label is not None]  # Remove 'None' values
    best_Rsqrd = np.nanargmax(best_Rsqrds[:, 0])

    Rsqrd_best_lbl = Rsqrds_best_lbl[best_Rsqrd]
    Rsqrd_best_lbl = [label for label in Rsqrd_best_lbl if label is not None]  # Remove 'None' values

    avg_LLs_Rsqrds = np.nanmean([best_LLs[:, 0], best_Rsqrds[:, 0]], axis=0)
    LR_best_n_reg = np.array([    # best number of regressors based on max avg (LL,Rsqrd)
        best_LL + 1,
        best_Rsqrd + 1,
        np.nanargmax(avg_LLs_Rsqrds) + 1  # I don't think we need an average
    ])

    # Find the summary of the best model based on log-likelihood and R-squared
    LL_best_model = LME[best_LL][int(best_LLs[best_LL, 1])]
    Rsqrd_best_model = LME[best_Rsqrd][int(best_Rsqrds[best_Rsqrd, 1])]

    return {
            'best_AIC': best_AIC,
            'AIC_best_lbl': AIC_best_lbl,
            'AIC_best_model': AIC_best_model, 

            'best_BIC': best_BIC,
            'BIC_best_lbl': BIC_best_lbl,
            'BIC_best_model': BIC_best_model,
            'AB_best_n_reg': AB_best_n_reg,

            'best_LL': best_LL,
            'LL_best_lbl': LL_best_lbl,
            'LL_best_model': LL_best_model,

            'best_Rsqrd': best_Rsqrd,
            'Rsqrd_best_lbl': Rsqrd_best_lbl,
            'Rsqrd_best_model': Rsqrd_best_model,
            'LR_best_n_reg': LR_best_n_reg
    }

def plot_model_summary(model, title, labels):
    params = model.params.values[1:]  # exclude the intercept
    errors = model.bse.values[1:]  # exclude the intercept
    pvalues = model.pvalues.values[1:]  # exclude the intercept

    color = ['darkkhaki', 'olive', 'rosybrown', 'indianred', 'darkred', 'firebrick', 'maroon']

    x = range(len(params))
    plt.figure(figsize=(10, 6))
    plt.bar(x, params, yerr=errors, capsize=5, color=color[:len(x)])
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x, labels, rotation=45, ha='right')
    
    for i, (param, error, pvalue) in enumerate(zip(params, errors, pvalues)):
        if pvalue < 0.05:
            if param > 0:
                plt.text(i, param + error + 0.2, '*', ha='center', va='bottom', fontsize=14, color='black')
            else:
                plt.text(i, param - error - 0.2, '*', ha='center', va='top', fontsize=14, color='black')

    fstat_pvalue = model.f_pvalue
    plt.title(f'{title} (F-stat p-value: {fstat_pvalue:.3f})')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

# Example usage
results = E2_ModelSelection(lat_index_csv, dependent, independent)

# Plot the summaries of the best models
plot_model_summary(results['AIC_best_model'], 'AIC Best Model Parameters', results['AIC_best_lbl'])
plot_model_summary(results['BIC_best_model'], 'BIC Best Model Parameters', results['BIC_best_lbl'])
plot_model_summary(results['Rsqrd_best_model'], 'R-squared Adjusted Best Model Parameters', results['Rsqrd_best_lbl'])
plot_model_summary(results['LL_best_model'], 'Log-likelihood Best Model Parameters', results['LL_best_lbl'])



