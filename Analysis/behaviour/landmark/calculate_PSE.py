
"""
===============================================
This code will read the data from the landmark task, calculate the PSE for each
subject using the Weibull distribution (Figure 3-A)
Finally, it plots the PSE and bias direction of all subjects in Figure 3-B

Author: S.M.H Ghafari
Email: m8ghafari@gamil.com
==============================================  
"""

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

platform = 'mac'

# Define where to read and write the data
if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

behavioural_bias_dir = 'Projects/subcortical-structures/SubStr-and-behavioral-bias'
landmark_resutls_dir = op.join(jenseno_dir, behavioural_bias_dir, 'programming/MATLAB/main-study/landmark-task/Results')
deriv_dir = op.join(jenseno_dir, behavioural_bias_dir, 'derivatives/landmark/figure3-290424')

subjects = np.arange(1,33) # number of subjects


# Define databinning function for figure A
def DataBinner(column):
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

# These below lists are used to list calculated bias of each subject. list will be used to plot figure 3-B:
Left_Bias_list=[]
Right_Bias_list=[]
No_Bias_list=[]

# this function plots figure 3A from 'cite sabine's paper'
def Figure3A(fpath):
    Data = pd.read_csv(fpath)

    # Data binning
    Data['Bin'] = Data['Shift_Size'].apply(DataBinner)
    Rightvalues = Data.loc[Data['Shift_Direction'] == 'Right', 'Bin']
    Rightvaluesmax = Rightvalues.max()+1  # why +1?
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
    # Define axis start and end points:
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
    PSE = weibull_min_ppf(0.5, shape_x, loc_x, scale_x, y_scale, y_bias)
    PSE_floored = math.floor(PSE)
    if PSE_floored < 0:
        Bias = 'Lefward Bias'
        PSE_floored = Leftvaluesmax + PSE_floored
        Left_Bias_list.append(PSE_floored)
    elif PSE_floored > 0:
        Bias = 'Righward Bias'
        PSE_floored = Rightvaluesmax - PSE_floored + 1
        Right_Bias_list.append(PSE_floored)
    else:
        Bias = 'No Bias'
        No_Bias_list.append(PSE_floored)
    # Draw Weibull Curves:
    plt.plot(x_weibull, cdf_y, 'blue', lw=1.3, label='Weibull CDF')
    # Draw "veridical Midponit" line:
    plt.axvline(x=0, color='black', linestyle='--', dashes=(5, 3),
                lw=1.75, label='Veridical Midponit')
    # Draw PSE Vertical and Horizontal Lines:
    plt.axvline(x=PSE, color='grey', lw=1, linestyle=':')
    plt.axhline(y=0.5, color='grey', lw=1, linestyle=':', label='PSE')
    # Find the Best Location for Plot Guide Box:
    if PSE > 0:
        PSE = Rightvaluesmax - PSE
        plt.legend(loc=2, title='PSE={} VA{} ({})'.format(round(0.8**PSE, 4), chr(176), Bias), title_fontsize='x-large',
               alignment='left', fontsize='large')
    else:
        PSE = Leftvaluesmax + PSE
        plt.legend(loc=2, title='PSE=-{} VA{} ({})'.format(round(0.8**PSE, 4), chr(176), Bias), title_fontsize='x-large',
              alignment='left', fontsize='large')
    # Goodness of Weibull fit statistics (R-squared):
    Table_r2 = Table.sort_values(by=['Bin_Mean'])
    y_true_r2 = Table_r2['Proportion_Reported_Right'].tolist()
    r2 = r2_score(y_true=y_true_r2, y_pred=cdf_y)
    # Remove top and left frames:
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    return r2, Left_Bias_list, Right_Bias_list, No_Bias_list

# Plot figure 3-A for all subjects:
for sub in subjects:
    sub_code = f"sub-S{sub+1000}"
    file_name = f"sub-S{sub+1000}_ses-01_task-Landmark_run-01_logfile.csv"
    savefig_path = op.join(deriv_dir, sub_code + '_figure3A2.png')
    fpath = op.join(landmark_resutls_dir, sub_code, 'ses-01/beh', file_name)
    # plot figure 3A
    r2, left_bias_list, right_bias_list, no_bias_list = Figure3A(fpath)
    # Define plot(s) title:
    plt.title('Figure 3-A. Subject %s _ r2 = %s' % (sub_code, r2), pad=10, fontsize=10, fontweight=100, loc='left')
    # Full screnn plot:
    plt.tight_layout()
    # Save figure 3-A plot(s):
    plt.savefig(savefig_path, dpi=300)
    plt.close()
PSE_Data = pd.DataFrame()
PSE_list = right_bias_list + left_bias_list + no_bias_list
PSE_Data['PSE'] = PSE_list

# Calculate mean and standard deviation
mean_PSE = np.mean(PSE_Data['PSE'])
std_PSE = np.std(PSE_Data['PSE'])

# Define boolean mask to identify elements to keep (non outliers)
mask = (PSE_Data['PSE'] >= mean_PSE - 2 * std_PSE) & (PSE_Data['PSE'] <= mean_PSE + 2 * std_PSE)

# Remove outliers outside the range of mean ± 2 * std
outliers = PSE_Data['PSE'][~mask].to_numpy()
PSE_Data = PSE_Data[mask]

# Filter right_list and left_list using the mask
right_list = [x for x in right_bias_list if x not in outliers]
left_list= [x for x in left_bias_list if x not in outliers]

# Divide left and right bias based on PSEs
right_bias_max = max(right_list)
left_bias_max = max(left_list)
bias_list=[]
for r in right_list:
    right = right_bias_max - r + 1
    bias_list.append(right)
for l in left_list:
    left = l - left_bias_max - 1
    bias_list.append(left)
bias_list = bias_list + no_bias_list
Bias_Data = pd.DataFrame()
Bias_Data['PSE_bin'] = bias_list

# Plot figure 3-B:
Bias_Table=pd.DataFrame()
Bias_Table['Number_Subjets'] = Bias_Data.groupby(['PSE_bin'])['PSE_bin'].count()
Bias_x = Bias_Table.index.get_level_values('PSE_bin')
Bias_y = Bias_Table['Number_Subjets']
plt.figure(figsize=(8, 8))
plt.bar(Bias_x, Bias_y, width=0.5, color='black')
# Define axis lables:
plt.xlabel('Spatial Bias (Log of Deg. Vis. Ang.)',
           fontsize='x-large', fontweight=1000)
plt.ylabel('# Subjets', fontsize='x-large', fontweight=1000)
# Define axis start and end points
plt.xlim(left_bias_max*-1-2,right_bias_max+1)
plt.ylim(0,10)
# Define axis ticks
xaxisticks_Bias = np.arange(left_bias_max*-1-1,right_bias_max+1)
xaxislables_Bias = ['-0.8\u00b0']
for i in np.arange(left_bias_max*-1,right_bias_max):
    if i == 0:
        xaxislables_Bias.append('0')
    else:
        xaxislables_Bias.append('')
xaxislables_Bias.append('+0.8\u00b0')
plt.xticks(ticks=xaxisticks_Bias, labels=xaxislables_Bias)
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
