"""
===============================================
Script: Cross-validation of Putamen asymmetry predicting Landmark PSE

This script computes:
1. Split-sample cross-validation (50/50 split)
2. Formula-based CV estimates (Pedhazur: adjusted R², regression model CV, correlation model CV)
3. Modern k-fold CV (10-fold) and Leave-One-Out CV (LOOCV)
4. Permutation test for cross-validated R² (empirical p-value)

The permutation test shuffles the dependent variable (Landmark_PSE) 
many times to build a null distribution of CV R². 
The observed CV R² is compared against this distribution to compute 
an empirical p-value.

Input: CSV file containing Landmark_PSE and Puta

written by Tara Ghafari
tara.ghafari@gmail.com
27/08/2025
===============================================
"""

import pandas as pd
import numpy as np
import os.path as op
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import statsmodels.api as sm

# ---------------------------------------------------
# Define paths 
# ---------------------------------------------------
platform = 'mac'
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

volume_sheet_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/SubStr-and-behavioral-bias')
lat_index_csv = op.join(volume_sheet_dir, 'data/collated/FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_flipped.csv')

# Load data
data_full = pd.read_csv(lat_index_csv)
df = data_full.dropna(subset=['Landmark_PSE', 'Puta'])  # remove rows with missing data in these two variables (subject #28)

# Dependent and independent variables
y = df['Landmark_PSE'].values
X = df[['Puta']].values

# ---------------------------
# 1. Split-sample cross-validation
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
split_r2 = r2_score(y_test, y_pred)

# Correlation-based R² (squared Pearson correlation)
r_corr_split = np.corrcoef(y_test, y_pred)[0, 1]

# ---------------------------
# 2. Formula-based cross-validation (Pedhazur)
# ---------------------------
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()
R2 = ols_model.rsquared
N = len(y)
k = X.shape[1]

# Adjusted R²
adj_R2 = 1 - (1 - R2) * ((N - 1) / (N - k - 1))

# Cross-validated R² (regression model, predictors fixed)
R2_cv_regression = 1 - ((N - 1) / N) * ((N + k + 1) / (N - k - 1)) * (1 - R2)

# Cross-validated R² (correlation model, predictors random)
R2_cv_correlation = 1 - ((N - 1) / (N - k - 1)) * ((N - 2) / (N - k - 2)) * ((N + 1) / N) * (1 - R2)

# ---------------------------
# 3. k-fold CV and LOOCV (empirical Q²)
# ---------------------------
# ----- 10-fold CV
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
yhat_kfold = cross_val_predict(LinearRegression(), X, y, cv=kfold)
r2_kfold = r2_score(y, yhat_kfold)

# Correlation-based R² (squared Pearson correlation)
r_corr_kfold = np.corrcoef(y, yhat_kfold)[0, 1]

# ----- LOOCV
loo = LeaveOneOut()
yhat_loo = cross_val_predict(LinearRegression(), X, y, cv=loo)
r2_loo = r2_score(y, yhat_loo)

# Correlation-based R² (squared Pearson correlation)
r_corr_loo = np.corrcoef(y, yhat_loo)[0, 1]

# ---------------------------
# 4. Permutation test for CV R²
# ---------------------------
def permutation_test_cv_r2(X, y, cv, n_permutations=5000, random_state=42):
    """
    Compute empirical p-value for cross-validated R² using permutation testing.
    Parameters
    ----------
    X : array, predictors
    y : array, dependent variable
    cv : cross-validation splitter (e.g., KFold or LeaveOneOut)
    n_permutations : int, number of permutations
    random_state : int, RNG seed
    
    Returns
    -------
    observed_r2 : float, observed CV R²
    p_value : float, proportion of permutations with R² >= observed
    null_dist : array, distribution of permuted CV R²
    """
    rng = np.random.RandomState(random_state)
    # observed CV R²
    yhat = cross_val_predict(LinearRegression(), X, y, cv=cv)
    observed_r2 = r2_score(y, yhat)
    # null distribution
    null_dist = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        yhat_perm = cross_val_predict(LinearRegression(), X, y_perm, cv=cv)
        r2_perm = r2_score(y_perm, yhat_perm)
        null_dist.append(r2_perm)
    null_dist = np.array(null_dist)
    # empirical p-value
    p_value = np.mean(null_dist >= observed_r2)
    return observed_r2, p_value, null_dist

# Run permutation tests
r2_kfold_obs, pval_kfold, null_kfold = permutation_test_cv_r2(X, y, kfold, n_permutations=5000)
r2_loo_obs, pval_loo, null_loo = permutation_test_cv_r2(X, y, loo, n_permutations=5000)

# ---------------------------
# Results summary
# ---------------------------
results = {
    "Observed R²": R2,
    "Adjusted R²": adj_R2,
    "Formula CV R² (Regression)": R2_cv_regression,
    "Formula CV R² (Correlation)": R2_cv_correlation,
    "Split-sample R² (50/50 split)": split_r2,
    "Pearson correlation (y_test vs y_pred)": r_corr_split,
    "10-fold CV R²": r2_kfold,
    "10-fold Pearson correlation": r_corr_kfold,
    "10-fold CV p-value (perm)": pval_kfold,
    "LOOCV R²": r2_loo,
    "LOOCV Pearson correlation": r_corr_loo,
    "LOOCV p-value (perm)": pval_loo
}

results_df = pd.DataFrame([results])
print(results)
