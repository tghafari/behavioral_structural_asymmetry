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

# databining for figure B
def DataBinB(column):
    return math.floor(column)

# Define Weibull distrbituion function:
y_scale_guess = 1
y_bias_guess = 0
ppf = 0.5

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

    Bias_list = [] # list of PSEs for figure B
    Data = pd.read_csv(fpath)
    
    # Data binning
    Data['Bin'] = Data['Shift_Size'].apply(DataBin)
    Data['Shift_Size'] = np.where(Data['Shift_Direction'] == 'Left', Data['Shift_Size'] * -10, Data['Shift_Size'])
    Data['Shift_Size'] = np.where(Data['Shift_Direction'] == 'Right', Data['Shift_Size'] * 10, Data['Shift_Size'])
    Data['Bin_Mean'] = Data['Shift_Size'].apply(DataBin)
    # Define "Biggerright" column:
    Data['Biggerright'] = 0
    Data['Biggerright'] = np.where((Data['Block_Question'] == 'Longer') & (Data['Answer'] == 'Right'),
                                   Data['Biggerright'] + 1, Data['Biggerright'])
    Data['Biggerright'] = np.where((Data['Block_Question'] == 'Shorter') & (Data['Answer'] == 'Left'),
                                   Data['Biggerright'] + 1, Data['Biggerright'])
    # Draw table of binned data:
    Table = pd.DataFrame()
    Table['Bin Size'] = Data.groupby(['Block_Number', 'Bin Mean'])['Biggerright'].count()
    Table['Rigts'] = Data.groupby(['Block_Number', 'Bin Mean'])['Biggerright'].sum()
    Table['Proportion Reported Right'] = Data.groupby(['Block_Number', 'Bin Mean'])['Biggerright'].mean()
    # Plot scatter plot:
    x=Table.index.get_level_values('Bin Mean')
    y=Table['Proportion Reported Right'].tolist()
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, marker='x', color='red', s=10)
    # Define axis lables:
    plt.xlabel('Horizontal Line Offset (Deg. Vis. Ang.)', fontsize='x-large', fontweight=1000)
    plt.ylabel('Proportion Reported Right', fontsize='x-large', fontweight=1000)
    # Define axis starting and end points:
    plt.xlim(-1, 1.1)
    plt.ylim(0, 1.1)
    # Define axis ticks:
    plt.xticks([-1,0,1])
    plt.yticks([0,0.5,1])
    # Fit Weibull distribution:
    shape_x, loc_x, scale_x = weibull_min.fit(x)
    x_weibull = np.linspace(min(x), max(x), 100)
    #pdf_y = weibull_min.pdf(x_weibull, shape_x, loc_x, scale_x)
    cdf_y = weibull_min.cdf(x_weibull, shape_x, loc_x, scale_x)
    # Define direction of bias:
    PSE_x = weibull_min.ppf(0.5, shape_x, loc_x, scale_x)
    Bias_list.append(PSE_x)
    if PSE_x < 0:
        Bias = 'Lefward Bias'
    elif PSE_x > 0:  # possibly PSE_x?
        Bias = 'Righward Bias'
    else:
        Bias = 'No Bias'
    # Draw Weibull Curves:
    plt.plot(x_weibull, cdf_y, 'blue', lw=1, label='Weibull CDF')
    #plt.plot(x_weibull, pdf_y, 'green', lw=1, label='Weibull PDF')
    # Draw "veridical Midponit" line:
    plt.axvline(x=0, color='black', linestyle='--', dashes=(5, 3), lw=1.75, label='Veridical Midponit')
    # Draw PSE Vertical and Horizontal Lines:
    plt.axvline(x=PSE_x, color='grey', lw=1, linestyle=':')
    plt.axhline(y=0.5, color='grey', lw=1, linestyle=':', label='PSE')
    # Find the Best Location for Plot Guide Box:
    plt.legend(loc=2, title='PSE={} VA{} ({})'.format(round(PSE_x,4), chr(176), Bias), title_fontsize='x-large',
               alignment='left', fontsize='large', edgecolor='pink', shadow=True)
    # Print the estimated parameters and the goodness of Weibull fit statistics:
    ####print('k =', shape)
    ###print('lambda =', loc)
    ###print('c=',scale)
    # Remove top and left frames:
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    return Bias_list

# Plot figure 3-A for all subjects:
for sub in subjects:
    print(sub)
    sub_code = f"sub-S{sub+1000}"
    file_name = f"sub-S{sub+1000}_ses-01_task-Landmark_run-01_logfile.csv"
    savefig_path = op.join(deriv_dir, sub_code + '_figure3A.png')
    fpath = op.join(landmark_resutls_dir, sub_code, 'ses-01/beh', file_name)
    # plot figure 3A
    Figure_3_A = Figure3A(fpath, savefig_path)
    # Define plot(s) title:
    plt.title('Figure 3-A. Subject %s' % (sub_code), pad=10, fontsize=10, fontweight=100, loc='left')
    # Full screnn plot:
    plt.tight_layout()
    # Save figure 3-A plot(s):
    plt.savefig(savefig_path, dpi=300)
    
#Figure 3-B. Raw Data:
Bias_Data=pd.DataFrame()
Bias_Data['PSE'] = Bias_list
#Figure 3-B. Data binning:
Bias_Data['Bin Mean']=Bias_Data['PSE'].apply(DataBin)
Bias_Table=pd.DataFrame()
Bias_Table['Number of Subjets']=Bias_Data.groupby(['Bin Mean'])['PSE'].count()
Bias_x=Bias_Table.index.get_level_values('Bin Mean')
Bias_y=Bias_Table['Number of Subjets']
# Plot figure 3-B:
plt.figure(figsize=(8, 8))
plt.bar(Bias_x,Bias_y,width=0.03,color='black')
# Define axis lables:
plt.xlabel('Spatial Bias (Deg. Vis. Ang.)', fontsize='x-large', fontweight=1000)
plt.ylabel('Number of Subjets', fontsize='x-large', fontweight=1000)
# Define axis starting and end points:
plt.xlim(-0.7,0.7)
plt.ylim(0,12.5)
# Define axis ticks:
plt.xticks([-0.6,-0.3,0,0.3,0.6])
plt.yticks(np.arange(0, 12.1, 4))
# Draw "veridical Midponit" line:
plt.axvline(x=0, color='black', linestyle='--', dashes=(5, 3), lw=1.75, label='Veridical Midponit')
# Add bias side text:
plt.text(-0.55,12,'LVF Bias',fontsize=18)
plt.text(0.4,12,'RVF Bias',fontsize=18)
# Define plot(s) title:
plt.title('Figure 3-B',pad=15,fontsize=25,fontweight=100,loc='left')
# Remove top and left frames:
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Full screnn plot:
plt.tight_layout()
# Save figure 3-B plot:
savefig_path_3B = op.join(deriv_dir, 'figure3B.png')
plt.savefig(savefig_path_3B, dpi=300)