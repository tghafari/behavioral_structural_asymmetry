"""
===============================================
01. calculate_attention_bias

this code will read in the data from target
orientation detection task and calculates
each participants bias towards right and left

written by Mohammad Ebrahim Katebi (mekatebi.2000@gmail.com)
adapted by Tara Ghafari
==============================================  
ToDos:
    1) 
Issues:
    1) 
Questions:
    1) 
"""

# import libraries
import os.path as op
import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

# Obligate pandas to show entire data
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# Define Weibull distrbituion parameters
y_scale_guess = 0.5
y_bias_guess = 0.5
ppf = 0.75

# Define Weibull functions
def weibull_min_cdf(x_log, shape, loc, scale, y_scale, y_bias):

    # leave the parameters to be optimized
    y = weibull_min.cdf(x_log, 6, loc, scale)

    # y_scaled = (y * y_scale) + y_bias  # uses parameters from weibull
    # uses fixed parameters - final decision
    y_scaled = (y * y_scale_guess) + y_bias_guess

    return y_scaled


def weibull_min_ppf(ppf, shape, loc, scale, y_scale, y_bias):

    # ppf_unscaled = (ppf - y_bias) / y_scale  # uses parameters from weibull
    # uses fixed parameters - final decision
    ppf_unscaled = (ppf - y_bias_guess) / y_scale_guess

    return weibull_min.ppf(ppf_unscaled, 6, loc, scale)


def Finalysis(fpath):
    """reads in csv file from PTB and calculates
    number of correct trials in right and left attention.
    Returns a table containing those values
    """

    Data = pd.read_csv(fpath)  # address to csv file from PTB

    Data = Data[Data['State'] == 1]

    Contrasts = np.unique(Data['Contrast'])

    # preallocate Results: Contrast, Right Correct Percent, Left Correct Percent, All Correct Percent
    Results = np.stack((Contrasts, np.zeros(np.shape(Contrasts)[0]),
                        np.zeros(np.shape(Contrasts)[0]),
                        np.zeros(np.shape(Contrasts)[0])), axis=1)

    for Contrast in Contrasts:

        Contrast_Data = Data[Data['Contrast'] == Contrast]

        Contrast_Trials_All = np.shape(Contrast_Data)[0]

        Contrast_Attention_Right = Contrast_Data[Contrast_Data['Attention_Direction'] == 'Right']
        Contrast_Trials_Right = np.shape(Contrast_Attention_Right)[0]

        Contrast_Attention_Right_Corrects = Contrast_Attention_Right[((Contrast_Attention_Right['Answer'] == 'LeftShift') &
                                                                      (Contrast_Attention_Right['Target_Oriention'] == 45)) |
                                                                     ((Contrast_Attention_Right['Answer'] == 'RightShift') &
                                                                      (Contrast_Attention_Right['Target_Oriention'] == -45))]

        Contrast_Attention_Right_Correct_Count = np.shape(
            Contrast_Attention_Right_Corrects)[0]
        Contrast_Attention_Right_Correct_Percent = Contrast_Attention_Right_Correct_Count / \
            Contrast_Trials_Right

        Contrast_Attention_Left = Contrast_Data[Contrast_Data['Attention_Direction'] == 'Left']
        Contrast_Trials_Left = np.shape(Contrast_Attention_Left)[0]

        Contrast_Attention_Left_Corrects = Contrast_Attention_Left[((Contrast_Attention_Left['Answer'] == 'LeftShift') &
                                                                    (Contrast_Attention_Left['Target_Oriention'] == 45)) |
                                                                   ((Contrast_Attention_Left['Answer'] == 'RightShift') &
                                                                    (Contrast_Attention_Left['Target_Oriention'] == -45))]

        Contrast_Attention_Left_Correct_Count = np.shape(
            Contrast_Attention_Left_Corrects)[0]
        Contrast_Attention_Left_Correct_Percent = Contrast_Attention_Left_Correct_Count / \
            Contrast_Trials_Left

        for i in range(np.shape(Results)[0]):
            """puts right corrects and left corrects and total corrects in Result matrix"""

            if Results[i][0] == Contrast:

                Results[i][1] = Contrast_Attention_Right_Correct_Percent
                Results[i][2] = Contrast_Attention_Left_Correct_Percent
                Results[i][3] = (Contrast_Attention_Right_Correct_Count +
                                 Contrast_Attention_Left_Correct_Count) / Contrast_Trials_All

    contrast_Table = pd.DataFrame(data=Results, columns=[
        "Contrast", "Right_Correct_Percent", "Left_Correct_Percent", "All_Correct_Percent"])
    contrast_Table = contrast_Table.set_index(['Contrast'])

    return contrast_Table


def check_for_outlier(contrast_Table, sub_code, outliers):
    """this definition finds the outliers
    outliers are those who have 
    max performance below %75: poor performers
    or min performance above %75: the staircase hasn't worked """

    right_correct_percent = contrast_Table['Right_Correct_Percent']
    left_correct_percent = contrast_Table['Left_Correct_Percent']
    
    # Use numpy to find the min and max values
    max_performance_right = np.max(right_correct_percent)
    min_performance_right = np.min(right_correct_percent)
    max_performance_left = np.max(left_correct_percent)
    min_performance_left = np.min(left_correct_percent)

    # Check the conditions and append to outliers if any condition is met
    if (max_performance_right < 0.75 or max_performance_left < 0.75 or 
        min_performance_right > 0.75 or min_performance_left > 0.75):
        outliers.append(sub_code)


def plot_fitted_data(contrast_Table, sub_code):
    """scatter plots corrects with log x-axis"""
    # dfi.export(contrast_Table, op.join(deriv_dir, sub_code + '_contrast_table.png'), dpi=400)  # figures the contrasts with correct responses

    # Plot scatter plot:
    x_log = np.log10(contrast_Table.index)  # contrasts are in log10
    y = contrast_Table['All_Correct_Percent']
    y_Right = contrast_Table['Right_Correct_Percent']
    y_Left = contrast_Table['Left_Correct_Percent']

    plt.figure(figsize=(9, 9))
    plt.scatter(x_log, y, marker='*', color='black', s=25)
    plt.scatter(x_log, y_Right, marker='x', color='red', s=25)
    plt.scatter(x_log, y_Left, marker='+', color='blue', s=25)

    # Define axis lables:
    plt.xlabel('Log10 Contrast',
               fontsize='x-large', fontweight=1000)
    plt.ylabel('% Answered Correct', fontsize='x-large', fontweight=1000)

    # Define axis starting and end points:
    plt.xlim(x_log[0]-1, x_log[-1]+1)
    plt.ylim(0.2, 1.1)

    # plt.xticks(np.linspace(-10, 5, 16))
    plt.yticks([0.25, 0.5, 0.75, 1])
    # plt.show()

    cdf_Plot_x = np.linspace(x_log[0]-1, x_log[-1]+1, 1000)

    # All  ///////////////////////////////////////////////////////////////

    # Fit Weibull distribution:
    shape_All, loc_All, scale_All = weibull_min.fit(x_log)
    fit, Temp = curve_fit(weibull_min_cdf, x_log, y, p0=[
                          shape_All, loc_All, scale_All, y_scale_guess, y_bias_guess], maxfev=1000000, check_finite=False)

    shape_All = fit[0]
    loc_All = fit[1]
    scale_All = fit[2]
    y_scale_All = fit[3]
    y_bias_All = fit[4]

    cdf_All_Plot = weibull_min_cdf(
        cdf_Plot_x, shape_All, loc_All, scale_All, y_scale_All, y_bias_All)

    # Draw Weibull Curves:
    plt.plot(cdf_Plot_x, cdf_All_Plot, 'black', lw=1, label='All Weibull CDF')

    # Define direction of bias:
    PSE_All = weibull_min_ppf(ppf, shape_All, loc_All,
                              scale_All, y_scale_All, y_bias_All)

    # Draw PSE Vertical and Horizontal Lines:
    plt.axvline(x=PSE_All, color='gray', lw=1, linestyle=':')
    plt.axhline(y=ppf, color='gray', lw=1, linestyle=':', label='PSE')

    # Right ///////////////////////////////////////////////////////////////

    # Fit Weibull distribution:
    shape_Right, loc_Right, scale_Right = weibull_min.fit(x_log)
    fit, Temp = curve_fit(weibull_min_cdf, x_log, y_Right, p0=[
                          shape_Right, loc_Right, scale_Right, y_scale_guess, y_bias_guess], maxfev=1000000, check_finite=False)

    shape_Right = fit[0]
    loc_Right = fit[1]
    scale_Right = fit[2]
    y_scale_Right = fit[3]
    y_bias_Right = fit[4]

    cdf_Right = weibull_min_cdf(
        x_log, shape_Right, loc_Right, scale_Right, y_scale_Right, y_bias_Right)
    cdf_Right_Plot = weibull_min_cdf(
        cdf_Plot_x, shape_Right, loc_Right, scale_Right, y_scale_Right, y_bias_Right)

    # Draw Weibull Curves:
    plt.plot(cdf_Plot_x, cdf_Right_Plot, 'red',
             lw=1, label='Right Weibull CDF')

    # Define direction of bias:
    PSE_Right = weibull_min_ppf(
        ppf, shape_Right, loc_Right, scale_Right, y_scale_Right, y_bias_Right)

    # Draw PSE Vertical and Horizontal Lines:
    plt.axvline(x=PSE_Right, color='red', lw=1, linestyle=':')

    # Goodness of Weibull fit statistics (R-squared):
    ss_res_Right = np.sum((y_Right-cdf_Right)**2)
    ss_tot_Right = np.sum((y_Right-np.mean(y_Right))**2)
    r_square_Right = 1-(ss_res_Right/ss_tot_Right)

    # Left ///////////////////////////////////////////////////////////////

    # Fit Weibull distribution:
    shape_Left, loc_Left, scale_Left = weibull_min.fit(x_log)

    # Use non-linear least squares to fit weibull_min_cdf function, to x_log
    fit, Temp = curve_fit(weibull_min_cdf, x_log, y_Left, p0=[
                          shape_Left, loc_Left, scale_Left, y_scale_guess, y_bias_guess], maxfev=100000, check_finite=False)

    shape_Left = fit[0]
    loc_Left = fit[1]
    scale_Left = fit[2]
    y_scale_Left = fit[3]
    y_bias_Left = fit[4]

    cdf_Left = weibull_min_cdf(
        x_log, shape_Left, loc_Left, scale_Left, y_scale_Left, y_bias_Left)
    cdf_Left_Plot = weibull_min_cdf(
        cdf_Plot_x, shape_Left, loc_Left, scale_Left, y_scale_Left, y_bias_Left)

    # Draw Weibull Curves:
    plt.plot(cdf_Plot_x, cdf_Left_Plot, 'blue',
             lw=1, label='Left Weibull CDF')

    # Define direction of bias:
    PSE_Left = weibull_min_ppf(
        ppf, shape_Left, loc_Left, scale_Left, y_scale_Left, y_bias_Left)

    # Draw PSE Vertical and Horizontal Lines:
    plt.axvline(x=PSE_Left, color='blue', lw=1, linestyle=':')

    # Goodness of Weibull fit statistics (R-squared):
    ss_res_Left = np.sum((y_Left-cdf_Left)**2)
    ss_tot_Left = np.sum((y_Left-np.mean(y_Left))**2)
    r_square_Left = 1-(ss_res_Left/ss_tot_Left)

    legend_title = 'Bias= {} Log Contrast'.format(
        round(PSE_Right - PSE_Left, 3))

    if sub_code in outliers:

        legend_title = 'Outlier\n' + legend_title

    # Plot Guide Box:
    plt.legend(loc=5, title=legend_title, title_fontsize='x-large',
               alignment='left', fontsize='medium', edgecolor='pink')

    # Remove top and left frames:
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    return PSE_Right, r_square_Right, PSE_Left, r_square_Left

# ====================================== main body of code ==================================#


# Define address of resuls and figures
rds_dir = '/Volumes/jenseno-avtemporal-attention'
behavioural_bias_dir = 'Projects/subcortical-structures/SubStr-and-behavioral-bias'
target_resutls_dir = op.join(rds_dir, behavioural_bias_dir,
                             'programming/MATLAB/main-study/target-orientation-detection/Results')
deriv_dir = op.join(rds_dir, behavioural_bias_dir,
                    'derivatives/target_orientation')
outliers_list_dir = op.join(deriv_dir, 'outliers')

subjects = np.arange(1, 33)  # number of subjects
sub_ids = []
PSE_lateralisation_indices = []  # PSE lateralisations for all participants
outliers = []  # outlier participants list

for sub in subjects:
    sub_code = f"sub-S{sub+1000}"
    file_name = f"sub-S{sub+1000}_ses-01_task-Orientation_Detection_run-01_logfile.csv"
    fpath = op.join(target_resutls_dir, sub_code, 'ses-01/beh', file_name)
    savefig_path = op.join(deriv_dir, 'figuresB', sub_code +
                           '_contrast_psychometric_plot.png')

    contrast_Table = Finalysis(fpath)

    check_for_outlier(contrast_Table, sub_code, outliers)

    PSE_Right, r_square_Right, PSE_Left, r_square_Left = plot_fitted_data(
        contrast_Table, sub_code)
    PSE_lateralisation_index = (
        abs(PSE_Right) - abs(PSE_Left)) / (abs(PSE_Right) + abs(PSE_Left))
    
    if sub_code not in outliers:
        PSE_lateralisation_indices.append(PSE_lateralisation_index)
        sub_ids.append(sub_code)

    plt.title(f"Subject {sub_code} Right PSE = {round(PSE_Right, 3)}, Right r2 = {round(r_square_Right, 3)}, "
              f"Left PSE = {round(PSE_Left, 3)}, Left r2 = {round(r_square_Left, 3)}", pad=15, fontsize=10, fontweight=200, loc='left')

    plt.tight_layout()   # full screnn plot
    plt.savefig(savefig_path, dpi=300)

lateralisation_indices_df = pd.DataFrame(PSE_lateralisation_indices, columns=['PSE_lat_idx'], index=sub_ids)
lateralisation_indices_df.to_csv(op.join(deriv_dir, 'lateralisation_indices',
                                          'PSE_lateralisation_indices.csv'))
outliers = pd.DataFrame(outliers, columns=['outlier_participants'])
outliers.to_csv(op.join(outliers_list_dir,
                'outlier_participants.csv'), index=False)
