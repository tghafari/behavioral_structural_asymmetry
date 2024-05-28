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
dependent = 'MS_target'  # lateralised performance in 'PSE_landmark' or 'MS_target' or 'PSE_target'
independent = 'thomas'  # lateralisation volume of 'substr' or 'thomas'?

def E2_ModelSelection(lat_index_csv, dependent, independent):

    data = pd.read_csv(lat_index_csv)

    if dependent == 'PSE_landmark':
        PSE_column = data.iloc[1:-1, 8].astype(float)  # these data should be added to the csv file manually before running this script
        outlier_idx = [15, 27]  # remove outliers from PSE_landmark: 1016, 1028
        y = PSE_column.drop(outlier_idx).values

    elif dependent == 'MS_target':
        ms_column = data.iloc[1:-1, 9].astype(float)  # these data should be added to the csv file manually before running this script
        outlier_idx = []  # remove outliers from MS_target
        y = ms_column.drop(outlier_idx).values

    elif dependent == 'PSE_target':
        target_column = data.iloc[1:-1, 10].astype(float)  # these data should be added to the csv file manually before running this script
        outlier_idx = [2, 3, 6, 7, 15, 16, 17, 24, 26, 27, 30]  # remove outliers from PSE_target: 1003, 1004, 1007, 1008, 1016, 1017, 1018, 1025, 1027, 1028, 1031, 1032
        y = target_column.drop(outlier_idx).values

    # Extract the columns of lateralisation volumes and remove outliers
    LV_columns_outlier = data.iloc[1:-1, 1:8].astype(float)  # last row is removed because the values are still missing
    LV_columns = LV_columns_outlier.drop(outlier_idx).values

    thomas_columns_outlier = data.iloc[1:-1, 11:20].astype(float)
    thomas_columns = thomas_columns_outlier.drop(outlier_idx).values
    
    # Choose the independent variables based on the selection
    if independent == 'substr':
        X = LV_columns
    elif independent == 'thomas':
        X = thomas_columns
    
    # Create all unique combinations of 'n' numbers, from 1 to 7
    nReg = [list(itertools.combinations(range(1, 8), n_regr)) for n_regr in range(1, 8)]
    
    # Preallocation with correct sizes
    LME = [[None]*len(nReg[i]) for i in range(7)]
    str_lbl = [[None]*len(nReg[i]) for i in range(7)]
    AIC = [np.zeros(len(nReg[i])) for i in range(7)]
    BIC = [np.zeros(len(nReg[i])) for i in range(7)]
    logLikelihood = [np.zeros(len(nReg[i])) for i in range(7)]
    Rsqrd_adjst = [np.zeros(len(nReg[i])) for i in range(7)]
    
    # Fit models
    for n_regr in range(1, 8):
        for j, reg_comb in enumerate(nReg[n_regr-1]):
            design_matrix = X[:, [i-1 for i in reg_comb]]
            design_df = pd.DataFrame(design_matrix, columns=[f'X{i}' for i in reg_comb])
            design_df['y'] = y
            
            formula = 'y ~ ' + ' + '.join([f'X{i}' for i in reg_comb])
            
            model = smf.ols(formula, data=design_df).fit()
            LME[n_regr-1][j] = model
            str_lbl[n_regr-1][j] = [data.columns[i] for i in reg_comb]
            AIC[n_regr-1][j] = model.aic
            BIC[n_regr-1][j] = model.bic
            logLikelihood[n_regr-1][j] = model.llf
            Rsqrd_adjst[n_regr-1][j] = model.rsquared_adj
    
    AIC = [np.where(aic == 0, np.nan, aic) for aic in AIC]
    BIC = [np.where(bic == 0, np.nan, bic) for bic in BIC]
    logLikelihood = [np.where(ll == 0, np.nan, ll) for ll in logLikelihood]
    Rsqrd_adjst = [np.where(rsq == 0, np.nan, rsq) for rsq in Rsqrd_adjst]
    
    # Find best models based on AIC and BIC
    best_AICs = np.zeros((7, 2))
    best_BICs = np.zeros((7, 2))
    AIC_best_lbl = [[None]*7 for _ in range(7)]
    BIC_best_lbl = [[None]*7 for _ in range(7)]
    
    for n_regr in range(1, 8):
        best_AICs[n_regr-1, 0] = np.nanmin(AIC[n_regr-1])
        best_AICs[n_regr-1, 1] = np.nanargmin(AIC[n_regr-1])
        best_BICs[n_regr-1, 0] = np.nanmin(BIC[n_regr-1])
        best_BICs[n_regr-1, 1] = np.nanargmin(BIC[n_regr-1])
        AIC_best_lbl[n_regr-1][:n_regr] = str_lbl[n_regr-1][int(best_AICs[n_regr-1, 1])]
        BIC_best_lbl[n_regr-1][:n_regr] = str_lbl[n_regr-1][int(best_BICs[n_regr-1, 1])]
    
    avg_AICs_BICs = np.nanmean([best_AICs[:, 0], best_BICs[:, 0]], axis=0)
    AB_best_n_reg = np.array([
        np.nanargmin(best_AICs[:, 0]) + 1,
        np.nanargmin(best_BICs[:, 0]) + 1,
        np.nanargmin(avg_AICs_BICs) + 1
    ])
    
    # Find best models based on log-likelihood and R-squared
    best_LLs = np.zeros((7, 2))
    best_Rsqrds = np.zeros((7, 2))
    LL_best_lbl = [[None]*7 for _ in range(7)]
    Rsqrd_best_lbl = [[None]*7 for _ in range(7)]
    
    for n_regr in range(1, 8):
        best_LLs[n_regr-1, 0] = np.nanmax(logLikelihood[n_regr-1])
        best_LLs[n_regr-1, 1] = np.nanargmax(logLikelihood[n_regr-1])
        best_Rsqrds[n_regr-1, 0] = np.nanmax(Rsqrd_adjst[n_regr-1])
        best_Rsqrds[n_regr-1, 1] = np.nanargmax(Rsqrd_adjst[n_regr-1])
        LL_best_lbl[n_regr-1][:n_regr] = str_lbl[n_regr-1][int(best_LLs[n_regr-1, 1])]
        Rsqrd_best_lbl[n_regr-1][:n_regr] = str_lbl[n_regr-1][int(best_Rsqrds[n_regr-1, 1])]
    
    avg_LLs_Rsqrds = np.nanmean([best_LLs[:, 0], best_Rsqrds[:, 0]], axis=0)
    LR_best_n_reg = np.array([
        np.nanargmax(best_LLs[:, 0]) + 1,
        np.nanargmax(best_Rsqrds[:, 0]) + 1,
        np.nanargmax(avg_LLs_Rsqrds) + 1
    ])
    
    return {
        'best_AICs': best_AICs,
        'best_BICs': best_BICs,
        'AIC_best_lbl': AIC_best_lbl,
        'BIC_best_lbl': BIC_best_lbl,
        'AB_best_n_reg': AB_best_n_reg,
        'best_LLs': best_LLs,
        'best_Rsqrds': best_Rsqrds,
        'LL_best_lbl': LL_best_lbl,
        'Rsqrd_best_lbl': Rsqrd_best_lbl,
        'LR_best_n_reg': LR_best_n_reg
    }

# Example usage
results = E2_ModelSelection(lat_index_csv, dependent, independent)
print(results)


