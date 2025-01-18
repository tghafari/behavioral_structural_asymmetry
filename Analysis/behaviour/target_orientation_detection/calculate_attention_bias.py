"""
===============================================
01. calculate_attention_bias

this code will read in the data from target
orientation detection task and calculates
each participants bias towards right and left

written by Mohammad Ebrahim Katebi (mekatebi.2000@gmail.com)
adapted by Tara Ghafari
==============================================  
"""

# Import libraries
import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

# Set up directories
DATA_DIR = r"E:/Target_Data"
OUTPUT_FOLDER_PATH = r"../../../Results/Beh/Target"
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# Obligate pandas to show entire data
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# Define Weibull distribution parameters
y_scale_guess = 0.5
y_bias_guess = 0.5
ppf = 0.75

# Define Weibull functions


def weibull_min_cdf(x_log, shape, loc, scale, y_scale, y_bias):
    """Calculate Weibull CDF with fixed parameters."""
    y = weibull_min.cdf(x_log, 6, loc, scale)
    y_scaled = (y * y_scale_guess) + y_bias_guess
    return y_scaled


def weibull_min_ppf(ppf, shape, loc, scale, y_scale, y_bias):
    """Calculate Weibull PPF with fixed parameters."""
    ppf_unscaled = (ppf - y_bias_guess) / y_scale_guess
    return weibull_min.ppf(ppf_unscaled, 6, loc, scale)


def analyze_data(fpath):
    """Analyze single subject data from CSV file."""
    try:
        data = pd.read_csv(fpath)
        data = data[data['State'] == 1]
        contrasts = np.unique(data['Contrast'])

        # Preallocate Results
        results = np.stack((
            contrasts,
            np.zeros(np.shape(contrasts)[0]),
            np.zeros(np.shape(contrasts)[0]),
            np.zeros(np.shape(contrasts)[0])
        ), axis=1)

        for contrast in contrasts:
            contrast_data = data[data['Contrast'] == contrast]
            contrast_trials_all = len(contrast_data)

            # Process right attention trials
            contrast_attention_right = contrast_data[contrast_data['Attention_Direction'] == 'Right']
            contrast_trials_right = len(contrast_attention_right)

            right_corrects = contrast_attention_right[
                ((contrast_attention_right['Answer'] == 'LeftShift') &
                 (contrast_attention_right['Target_Oriention'] == 45)) |
                ((contrast_attention_right['Answer'] == 'RightShift') &
                 (contrast_attention_right['Target_Oriention'] == -45))
            ]
            right_correct_percent = len(
                right_corrects) / contrast_trials_right if contrast_trials_right > 0 else 0

            # Process left attention trials
            contrast_attention_left = contrast_data[contrast_data['Attention_Direction'] == 'Left']
            contrast_trials_left = len(contrast_attention_left)

            left_corrects = contrast_attention_left[
                ((contrast_attention_left['Answer'] == 'LeftShift') &
                 (contrast_attention_left['Target_Oriention'] == 45)) |
                ((contrast_attention_left['Answer'] == 'RightShift') &
                 (contrast_attention_left['Target_Oriention'] == -45))
            ]
            left_correct_percent = len(
                left_corrects) / contrast_trials_left if contrast_trials_left > 0 else 0

            # Update results
            idx = np.where(results[:, 0] == contrast)[0][0]
            results[idx, 1] = right_correct_percent
            results[idx, 2] = left_correct_percent
            results[idx, 3] = (len(right_corrects) +
                               len(left_corrects)) / contrast_trials_all

        contrast_table = pd.DataFrame(
            data=results,
            columns=["Contrast", "Right_Correct_Percent",
                     "Left_Correct_Percent", "All_Correct_Percent"]
        ).set_index(['Contrast'])

        return contrast_table

    except Exception as e:
        print(f"Error processing file {fpath}: {str(e)}")
        return None


def check_for_outlier(contrast_table, sub_code, outliers):
    """Check if subject performance indicates they should be marked as outlier."""
    if contrast_table is None:
        outliers.append(sub_code)
        return

    right_correct = contrast_table['Right_Correct_Percent']
    left_correct = contrast_table['Left_Correct_Percent']

    max_right = np.max(right_correct)
    min_right = np.min(right_correct)
    max_left = np.max(left_correct)
    min_left = np.min(left_correct)

    if (max_right < 0.75 or max_left < 0.75 or
            min_right > 0.75 or min_left > 0.75):
        outliers.append(sub_code)


def plot_fitted_data(contrast_table, sub_code, outliers):
    """Plot psychometric function and fit Weibull curves."""
    try:
        x_log = np.log10(contrast_table.index)
        y = contrast_table['All_Correct_Percent']
        y_right = contrast_table['Right_Correct_Percent']
        y_left = contrast_table['Left_Correct_Percent']

        fig, ax = plt.subplots(figsize=(9, 9))

        # Plot scatter points
        ax.scatter(x_log, y, marker='*', color='black', s=25, label='All')
        ax.scatter(x_log, y_right, marker='x',
                   color='red', s=25, label='Right')
        ax.scatter(x_log, y_left, marker='+', color='blue', s=25, label='Left')

        # Set up plot parameters
        ax.set_xlabel('Log10 Contrast', fontsize='x-large', fontweight='bold')
        ax.set_ylabel('% Answered Correct',
                      fontsize='x-large', fontweight='bold')
        ax.set_xlim(x_log[0]-1, x_log[-1]+1)
        ax.set_ylim(0.2, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1])

        # Plot Weibull fits
        cdf_plot_x = np.linspace(x_log[0]-1, x_log[-1]+1, 1000)

        # Fit and plot for all data
        pse_all, r2_all = fit_and_plot_weibull(
            ax, x_log, y, cdf_plot_x, 'black', 'All')

        # Fit and plot for right data
        pse_right, r2_right = fit_and_plot_weibull(
            ax, x_log, y_right, cdf_plot_x, 'red', 'Right')

        # Fit and plot for left data
        pse_left, r2_left = fit_and_plot_weibull(
            ax, x_log, y_left, cdf_plot_x, 'blue', 'Left')

        # Add PSE reference line
        ax.axhline(y=ppf, color='gray', lw=1, linestyle=':', label='PSE')

        # Create legend
        legend_title = f'Bias= {round(pse_right - pse_left, 3)} Log Contrast'
        if sub_code in outliers:
            legend_title = 'Outlier\n' + legend_title

        ax.legend(loc='center right', title=legend_title,
                  title_fontsize='x-large', fontsize='medium',
                  edgecolor='pink')

        # Clean up plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        plt.title(f"Subject {sub_code}\nRight PSE = {round(pse_right, 3)}, R² = {round(r2_right, 3)}\n"
                  f"Left PSE = {round(pse_left, 3)}, R² = {round(r2_left, 3)}",
                  pad=15, fontsize=10, fontweight=200, loc='left')

        plt.tight_layout()

        return pse_right, r2_right, pse_left, r2_left

    except Exception as e:
        print(f"Error plotting data for {sub_code}: {str(e)}")
        return None, None, None, None


def fit_and_plot_weibull(ax, x_log, y, cdf_plot_x, color, label):
    """Helper function to fit and plot Weibull curve."""
    shape, loc, scale = weibull_min.fit(x_log)
    fit, _ = curve_fit(weibull_min_cdf, x_log, y,
                       p0=[shape, loc, scale, y_scale_guess, y_bias_guess],
                       maxfev=100000, check_finite=False)

    cdf = weibull_min_cdf(x_log, *fit)
    cdf_plot = weibull_min_cdf(cdf_plot_x, *fit)

    ax.plot(cdf_plot_x, cdf_plot, color, lw=1, label=f'{label} Weibull CDF')

    pse = weibull_min_ppf(ppf, *fit)
    ax.axvline(x=pse, color=color, lw=1, linestyle=':')

    # Calculate R-squared
    ss_res = np.sum((y - cdf)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res/ss_tot)

    return pse, r_squared


def process_all_subjects():
    """Process all subjects in the data directory."""
    all_pses = []
    outliers = []

    for item in os.listdir(DATA_DIR):
        if item.startswith("sub-"):
            sub_dir = op.join(DATA_DIR, item)
            for session in os.listdir(sub_dir):
                if session.startswith("ses-"):
                    ses_dir = op.join(sub_dir, session)
                    beh_dir = op.join(ses_dir, "beh")
                    if op.isdir(beh_dir):
                        for file in os.listdir(beh_dir):
                            if file.endswith("_logfile.csv"):
                                subject_name = item
                                print(f"\nProcessing: {subject_name}")

                                try:
                                    # Process single subject
                                    fpath = op.join(beh_dir, file)
                                    contrast_table = analyze_data(fpath)

                                    if contrast_table is not None:
                                        check_for_outlier(
                                            contrast_table, subject_name, outliers)

                                        # Plot and save results
                                        pses = plot_fitted_data(
                                            contrast_table, subject_name, outliers)
                                        if pses[0] is not None:
                                            all_pses.append(pses)

                                            # Save the plot
                                            plt.savefig(op.join(OUTPUT_FOLDER_PATH,
                                                                f'{subject_name}_psychometric.png'),
                                                        dpi=300)
                                        plt.close()

                                except Exception as e:
                                    print(
                                        f"Error processing {subject_name}: {str(e)}")
                                    continue

    return all_pses, outliers


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    # Process all subjects
    all_pses, outliers = process_all_subjects()

    # Save results
    pse_df = pd.DataFrame(all_pses,
                          columns=['PSE_Right', 'R2_Right', 'PSE_Left', 'R2_Left'])
    pse_df.to_csv(op.join(OUTPUT_FOLDER_PATH, 'PSE_values.csv'), index=False)

    outliers_df = pd.DataFrame(outliers, columns=['outlier_participants'])
    outliers_df.to_csv(
        op.join(OUTPUT_FOLDER_PATH, 'outliers.csv'), index=False)
