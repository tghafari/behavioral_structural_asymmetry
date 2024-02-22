===============================================
This code will read the data from the landmark task, calculate the PSE for each
subject using the Weibull distribution (Figure 3-A)
Finally, it plots the PSE and bias direction of all subjects in Figure 3-B
Author: S.M.H Ghafari
Email: m8ghafari@gamil.com
==============================================  

import os
import os.path as op
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import weibull_min, kstest
from scipy.optimize import curve_fit

# Obligate pandas to show entire data
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# Define address of resuls and figures
rds_dir = '/Volumes/jenseno-avtemporal-attention'
behavioural_bias_dir = r'Projects/subcortical-structures/SubStr-and-behavioral-bias'
landmark_resutls_dir = op.join(rds_dir, behavioural_bias_dir, 'programming/MATLAB/main-study/landmark-task/Results')
deriv_dir = op.join(rds_dir, behavioural_bias_dir, 'derivatives/landmark/figure3')

subjects = np.arange(1,20) # number of subjects


# Define databinning function for figure A
def DataBin(column):
    return round(math.log(column, 0.8))


# Define Weibull distrbituion parameters
y_scale_guess = 1
y_bias_guess = 0
ppf = 0.5


# Define Weibull distrbituion function
def weibull_min_cdf(x_weibull, shape, loc, scale, y_scale, y_bias):
    y = weibull_min.cdf(x_weibull, shape, loc, scale)
    # y_scaled = (y * y_scale) + y_bias
    y_scaled = (y * y_scale_guess) + y_bias_guess
    return y_scaled


def weibull_min_ppf(ppf, shape, loc, scale, y_scale, y_bias):
    # ppf_unscaled = (ppf - y_bias) / y_scale
    ppf_unscaled = (ppf - y_bias_guess) / y_scale_guess
    return weibull_min.ppf(ppf_unscaled, shape, loc, scale)

# this function plots figure 3A from 'cite sabine's paper'
def Figure3A(fpath, savefig_path):
    Data = pd.read_csv(fpath)
    
    # Data binning
    Data['Bin'] = Data['Shift_Size'].apply(DataBin)
    Rightvalues = Data.loc[Data['Shift_Direction'] == 'Right', 'Bin']
    Rightvaluesmax = Rightvalues.max()+1
    Leftvalues = Data.loc[Data['Shift_Direction'] == 'Left', 'Bin']
    Leftvaluesmax = Leftvalues.max()+1
    Data.loc[Data['Shift_Direction'] == 'Left', 'Bin_Mean'] = Leftvaluesmax - \
        Data.loc[Data['Shift_Direction'] == 'Left', 'Bin']
    Data.loc[Data['Shift_Direction'] == 'Right', 'Bin_Mean'] = Rightvaluesmax - \
        Data.loc[Data['Shift_Direction'] == 'Right', 'Bin']
    Data['Bin_Mean'] = np.where(
        Data['Shift_Direction'] == 'Left', Data['Bin_Mean'] * -1, Data['Bin_Mean'])
    # Define "Biggerright" column:
    Data['Biggerright'] = 0
    Data['Biggerright'] = np.where((Data['Block_Question'] == 'Longer') & (
        Data['Answer'] == 'Right'), Data['Biggerright'] + 1, Data['Biggerright'])
    Data['Biggerright'] = np.where((Data['Block_Question'] == 'Shorter') & (
        Data['Answer'] == 'Left'), Data['Biggerright'] + 1, Data['Biggerright'])
    # Draw table of binned data:
    Table = pd.DataFrame()
    Table['Bin_Size'] = Data.groupby(
        ['Block_Number', 'Bin', 'Bin_Mean', 'Shift_Direction'])['Biggerright'].count()
    Table['Rights'] = Data.groupby(
        ['Block_Number', 'Bin', 'Bin_Mean', 'Shift_Direction'])['Biggerright'].sum()
    Table['Proportion_Reported_Right'] = Data.groupby(
        ['Block_Number', 'Bin', 'Bin_Mean', 'Shift_Direction'])['Biggerright'].mean()
    # Plot scatter plot:
    x = Table.index.get_level_values('Bin_Mean')
    y = Table['Proportion_Reported_Right'].tolist()
    numberofpoints = len(x)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, marker='x', color='red', s=10)
    # Define axis lables:
    plt.xlabel('Horizontal Line Offset (Log of Deg. Vis. Ang.)',
               fontsize='x-large', fontweight=1000)
    plt.ylabel('Proportion Reported Right',
               fontsize='x-large', fontweight=1000)
    # Define axis starting and end points:
    plt.xlim(Leftvaluesmax*-1 - 1, Rightvaluesmax + 1)
    plt.ylim(0, 1.1)
    # Define axis ticks:
    xaxisticks = range(Leftvaluesmax*-1, Rightvaluesmax+1, 1)
    xaxislables = ['-0.8\u00b0']
    for i in range(Leftvaluesmax*-1+1, Rightvaluesmax):
        if i == 0:
            xaxislables.append('0')
        else:
            xaxislables.append('')
    xaxislables.append('+0.8\u00b0')
    plt.xticks(ticks=xaxisticks, labels=xaxislables)
    plt.yticks([0, 0.5, 1])
    # Fit Weibull distribution:
    x_weibull = np.linspace(min(x), max(x), numberofpoints)
    shape_x, loc_x, scale_x = weibull_min.fit(x_weibull)
    fit, temp = curve_fit(weibull_min_cdf, x, y, p0=[
                          shape_x, loc_x, scale_x, y_scale_guess, y_bias_guess], maxfev=10000, check_finite=False)
    shape_x = fit[0]
    loc_x = fit[1]
    scale_x = fit[2]
    y_scale = fit[3]
    y_bias = fit[4]
    cdf_y = weibull_min_cdf(x_weibull, shape_x, loc_x,
                            scale_x, y_scale, y_bias)
    # Define direction of bias:
    PSE_x = weibull_min_ppf(0.5, shape_x, loc_x, scale_x, y_scale, y_bias)
    if PSE_x < 0:
        Bias = 'Lefward Bias'
    elif PSE_x > 0:
        Bias = 'Righward Bias'
    else:
        Bias = 'No Bias'
    # Draw Weibull Curves:
    plt.plot(x_weibull, cdf_y, 'blue', lw=1.3, label='Weibull CDF')
    # Draw "veridical Midponit" line:
    plt.axvline(x=0, color='black', linestyle='--', dashes=(5, 3),
                lw=1.75, label='Veridical Midponit')
    # Draw PSE Vertical and Horizontal Lines:
    plt.axvline(x=PSE_x, color='grey', lw=1, linestyle=':')
    plt.axhline(y=0.5, color='grey', lw=1, linestyle=':', label='PSE')
    # Find the Best Location for Plot Guide Box:
    plt.legend(loc=2, title='PSE={} VA{} ({})'.format(round(PSE_x, 4), chr(176), Bias), title_fontsize='x-large',
               alignment='left', fontsize='large')
    # Goodness of Weibull fit statistics (R-squared):
    Table_r2 = Table.sort_values(by=['Bin_Mean'])
    y_true_r2 = Table_r2['Proportion_Reported_Right'].tolist()
    r2 = r2_score(y_true=y_true_r2, y_pred=cdf_y)
    # Remove top and left frames:
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    return PSE_x, r2

# Plot figure 3-A for all subjects:
bias_list = [] # list of PSEs for figure B
for sub in subjects:
    sub_code = f"sub-S{sub+1000}"
    file_name = f"sub-S{sub+1000}_ses-01_task-Landmark_run-01_logfile.csv"
    savefig_path = op.join(deriv_dir, sub_code + '_figure3A2.png')
    fpath = op.join(landmark_resutls_dir, sub_code, 'ses-01/beh', file_name)
    # plot figure 3A
    PSE_x, r2 = Figure3A(fpath, savefig_path)
    bias_list.append(PSE_x)
    # Define plot(s) title:
    plt.title('Figure 3-A. Subject %s _ r2 = %s' % (sub_code, r2), pad=10, fontsize=10, fontweight=100, loc='left')
    # Full screnn plot:
    plt.tight_layout()
    # Save figure 3-A plot(s):
    plt.savefig(savefig_path, dpi=300)
    
#Figure 3-B. Raw Data:
Bias_Data=pd.DataFrame()
Bias_Data['PSE']=bias_list
# Figure 3-B.
Bias_Data['Bin_Mean'] = Bias_Data['PSE'].apply(DataBinB)
Bias_Table = pd.DataFrame()
Bias_Table['Number_of_Subjets'] = Bias_Data.groupby(['Bin_Mean'])[
    'PSE'].count()
Bias_x = Bias_Table.index.get_level_values('Bin_Mean')
Bias_y = Bias_Table['Number_of_Subjets']
# Plot figure 3-B:
plt.figure(figsize=(8, 8))
plt.bar(Bias_x, Bias_y, width=0.5, color='black')
# Define axis lables:
plt.xlabel('Spatial Bias (Log of Deg. Vis. Ang.)',
           fontsize='x-large', fontweight=1000)
plt.ylabel('Number of Subjets', fontsize='x-large', fontweight=1000)
# Define axis starting and end points
plt.xlim(-12.1, 12.1)
plt.ylim(0, 10)
# Define axis ticks
listofxticks = range(-12, 12, 1)
plt.xticks(ticks=listofxticks)
plt.yticks(np.arange(0, 10, 1))
# Draw "veridical Midponit" line:
plt.axvline(x=0, color='black', linestyle='--', dashes=(5, 3),
            lw=1.75, label='Veridical Midponit')
# Add bias side text:
plt.text(-4, 12, 'LVF Bias', fontsize=18)
plt.text(4, 12, 'RVF Bias', fontsize=18)
# Define plot(s) title:
plt.title('Figure 3-B', pad=15, fontsize=10, fontweight=200, loc='left')
# Remove top and left frames:
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Full screnn plot:
plt.tight_layout()
# Save figure 3-B plot:
savefig_path_3B = op.join(deriv_dir, 'figure3B.png')
plt.savefig(savefig_path_3B, dpi=300)
