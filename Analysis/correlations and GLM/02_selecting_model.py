# -*- coding: utf-8 -*-
"""
===============================================

02_selecting_model

This code reads loads a csv file containing
the lateralisation index of substrs, PSEs and MS.
then runs all the possible combination of models
to select which model can predict PSE/MS laterality
better. that's the winning model.

written by Tara Ghafari
===============================================
"""


import pandas as pd
import numpy as np
import os.path as op
import itertools
import statsmodels.formula.api as smf

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
independent = 'substr'  # lateralisation volume of 'substr' or 'thomas'?

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
    best_BIC = np.nanargmin(best_BICs[:, 0])
    BIC_best_lbl = BICs_best_lbl[best_BIC]

    avg_AICs_BICs = np.nanmean([best_AICs[:, 0], best_BICs[:, 0]], axis=0)
    AB_best_n_reg = np.array([  # best number of regressors based on min avg (AIC,BIC)
        best_AIC + 1,  
        best_BIC + 1,
        np.nanargmin(avg_AICs_BICs) + 1  # I don't think we need an average
    ])
    # Find the summary of the best model based on AIC and BIC
    AIC_best_model_summary = LME[best_AIC][int(best_AICs[best_AIC, 1])].summary()
    BIC_best_model_summary = LME[best_BIC][int(best_BICs[best_BIC, 1])].summary()


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
    best_Rsqrd = np.nanargmax(best_Rsqrds[:, 0])
    Rsqrd_best_lbl = Rsqrds_best_lbl[best_LL]

    avg_LLs_Rsqrds = np.nanmean([best_LLs[:, 0], best_Rsqrds[:, 0]], axis=0)
    LR_best_n_reg = np.array([    # best number of regressors based on max avg (LL,Rsqrd)
        best_LL + 1,
        best_Rsqrd + 1,
        np.nanargmax(avg_LLs_Rsqrds) + 1  # I don't think we need an average
    ])

    # Find the summary of the best model based on AIC and BIC
    LL_best_model_summary = LME[best_LL][int(best_LLs[best_LL, 1])].summary()
    Rsqrd_best_model_summary = LME[best_Rsqrd][int(best_Rsqrds[best_Rsqrd, 1])].summary()

    return {
        'best_AICs': best_AICs,
        'best_BICs': best_BICs,
        'AIC_best_lbl': AIC_best_lbl,
        'BIC_best_lbl': BIC_best_lbl,
        'AB_best_n_reg': AB_best_n_reg,
        'AB_best_models': {AIC_best_model_summary, BIC_best_model_summary},
        'best_LLs': best_LLs,
        'best_Rsqrds': best_Rsqrds,
        'LL_best_lbl': LL_best_lbl,
        'Rsqrd_best_lbl': Rsqrd_best_lbl,
        'LR_best_n_reg': LR_best_n_reg,
        'LR_best_models': {LL_best_model_summary, Rsqrd_best_model_summary}
    }

# Example usage
results = E2_ModelSelection(lat_index_csv, dependent, independent)
print(results)


