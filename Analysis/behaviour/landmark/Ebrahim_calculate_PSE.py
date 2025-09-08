"""
Landmark Behavioral Analysis

This script analyzes landmark behavioral data using binned linear analysis 
with Weibull distribution fitting to determine Point of Subjective Equality (PSE).

Author: Mohammad Ebrahim Katebi
Date: 2025-09-08
"""

import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import r2_score
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

# Plot configuration
rcParams.update({
    'font.size': 11,
    'font.family': "Arial",
    'axes.linewidth': 1,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': True,
    'savefig.bbox': 'tight'
})

# Constants
DATA_DIR = r"../../../Landmark_Data"
OUTPUT_FOLDER_PATH = r"../../../Results/Beh/Landmark"
WEIBULL_SHAPE_FIXED = 7  # Fixed shape parameter for Weibull distribution
Y_SCALE_GUESS = 1.0      # Initial guess for y-scaling
Y_BIAS_GUESS = 0.0       # Initial guess for y-bias
BIN_WIDTH = 0.1          # Bin width for binning

# Create output directory
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)


def weibull_cdf_function(x, shape, location, scale, y_scale, y_bias):
    """Weibull cumulative distribution function with scaling and bias parameters."""
    y = weibull_min.cdf(-x, WEIBULL_SHAPE_FIXED, location, scale)
    return (y * Y_SCALE_GUESS) + Y_BIAS_GUESS


def weibull_ppf_function(percentile, shape, location, scale, y_scale, y_bias):
    """Weibull percent point function (inverse CDF) with unscaling."""
    unscaled_percentile = (percentile - Y_BIAS_GUESS) / Y_SCALE_GUESS
    return -weibull_min.ppf(unscaled_percentile, WEIBULL_SHAPE_FIXED, location, scale)


def bin_data_linear(shift_values, bin_width=BIN_WIDTH):
    """Bin data using linear binning.
    """
    shift_values = np.asarray(shift_values)

    binned = np.where(
        shift_values > 0,
        np.ceil(shift_values / bin_width) * bin_width,
        np.ceil(shift_values / bin_width) * bin_width
    )
    return binned


def prepare_behavioral_data(data):
    """Prepare behavioral data for analysis using binned linear approach."""
    # Create signed shift sizes
    data['shift_size_signed'] = np.where(
        data['Shift_Direction'] == 'Left',
        -data['Shift_Size'],
        data['Shift_Size']
    )

    # Mirror the x-axis
    data['shift_size_signed'] = -data['shift_size_signed']

    # Apply linear binning
    data['shift_bin'] = bin_data_linear(data['shift_size_signed'])

    # Determine if participant reported "bigger on right"
    data['reported_bigger_right'] = (
        ((data['Block_Question'] == 'Longer') & (data['Answer'] == 'Right')) |
        ((data['Block_Question'] == 'Shorter') & (data['Answer'] == 'Left'))
    )

    # Group by bins and calculate summary statistics
    summary_table = data.groupby('shift_bin')['reported_bigger_right'].agg([
        'count', 'sum', 'mean']).reset_index()

    # Rename columns and convert proportion to percentage
    summary_table.columns = ['shift_bin', 'trial_count',
                             'right_responses', 'proportion_right']
    summary_table['percent_right'] = summary_table['proportion_right'] * 100

    return summary_table, 'shift_bin'


def fit_weibull_distribution(x_data, y_data):
    """Fit Weibull distribution to psychometric data."""
    # Create fitting range
    x_fit = np.linspace(min(x_data), max(x_data), len(x_data) * 10)

    # Initial parameter estimation
    shape_init, loc_init, scale_init = weibull_min.fit(x_fit)

    # Fit Weibull CDF to data
    optimal_params, _ = curve_fit(
        weibull_cdf_function,
        x_data,
        y_data,
        p0=[shape_init, loc_init, scale_init, Y_SCALE_GUESS, Y_BIAS_GUESS],
        maxfev=100000,
        check_finite=False
    )

    # Generate fitted curve
    y_fit = weibull_cdf_function(
        x_fit, *optimal_params) * 100  # Convert to percentage

    # Calculate Point of Subjective Equality (50% point)
    pse = weibull_ppf_function(0.5, *optimal_params)

    # Calculate R-squared
    y_pred = weibull_cdf_function(x_data, *optimal_params)
    r2 = r2_score(y_data, y_pred)

    return x_fit, y_fit, pse, r2


def create_figure(x_data, y_data, x_fit, y_fit, pse, r2, subject_name):
    """Create publication-ready psychometric function figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Convert proportion to percentage for plotting
    y_data_percent = y_data * 100

    # Plot data points
    ax.scatter(x_data, y_data_percent, s=60, c='black', marker='o',
               alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)

    # Plot fitted curve
    ax.plot(x_fit, y_fit, color='red', linewidth=2,
            label='Weibull fit', zorder=4)

    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='--',
               linewidth=2, alpha=0.7, label='Midpoint')
    ax.axvline(x=pse, color='blue', linestyle=':', linewidth=1.5, label='PSE')
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5)

    # Formatting
    ax.set_xlabel('Horizontal Line Offset (Deg. Vis. Ang.)',
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Percent Answered Right', fontsize=11, fontweight='bold')
    ax.set_title('Psychometric Function for One Example Participant',
                 fontsize=14, fontweight='bold', pad=20)

    # Set axis limits and ticks
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-5, 105)
    ax.set_xticks(np.arange(-1.0, 1.1, 0.2))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add PSE annotation
    bias_direction = 'Leftward' if pse < 0 else 'Rightward' if pse > 0 else 'No'
    bias_text = f'Bias: {pse:.3f}° {bias_direction}'
    ax.text(0.045, 1, bias_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='oldlace', alpha=0.8,
                      edgecolor='darkgoldenrod', boxstyle='round,pad=0.6'),
            fontsize=11, style='italic')

    # Legend with matching style from reference
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.0, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def analyze_subject_data(file_path):
    """Analyze behavioral data for a single subject."""
    # Load data
    data = pd.read_csv(file_path)

    subject_name = op.basename(file_path).replace('_logfile.csv', '')
    print(f"Analyzing subject: {subject_name}")

    # Prepare data using binned linear approach
    processed_data, x_column = prepare_behavioral_data(data)

    # Extract data for fitting
    x_values = processed_data[x_column].values
    y_values = processed_data['proportion_right'].values

    # Fit Weibull distribution
    x_fit, y_fit, pse, r2 = fit_weibull_distribution(x_values, y_values)

    # Create publication figure
    fig = create_figure(x_values, y_values, x_fit,
                        y_fit, pse, r2, subject_name)

    # Save figure
    figure_path = op.join(OUTPUT_FOLDER_PATH,
                          f"{subject_name}_Psychometric.png")
    fig.savefig(figure_path, dpi=1200, bbox_inches='tight')
    plt.close(fig)

    return pse


def process_all_subjects():
    """Process all subject data files and compile results."""
    pse_values = []
    subject_ids = []
    print("Starting batch analysis...")

    # Iterate through data directory structure
    for subject_folder in os.listdir(DATA_DIR):
        if subject_folder.startswith("sub-"):
            subject_path = op.join(DATA_DIR, subject_folder)

            for session_folder in os.listdir(subject_path):
                if session_folder.startswith("ses-"):
                    session_path = op.join(subject_path, session_folder)
                    behavior_path = op.join(session_path, "beh")

                    if op.isdir(behavior_path):
                        for filename in os.listdir(behavior_path):
                            if filename.endswith("_logfile.csv"):
                                file_path = op.join(behavior_path, filename)
                                try:
                                    pse = analyze_subject_data(file_path)
                                    pse_values.append(pse)
                                    subject_ids.append(
                                        filename.replace('_logfile.csv', ''))
                                except Exception as e:
                                    print(f"Error processing {filename}: {e}")

    # Save results
    results_df = pd.DataFrame({
        'ID': subject_ids,
        'PSE_Binned_Linear': pse_values
    })
    results_path = op.join(OUTPUT_FOLDER_PATH, 'PSE_Results.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\nAnalysis complete. Results saved to: {results_path}")
    print(f"Processed {len(pse_values)} subjects")
    return pse_values


def main():
    """Main analysis function."""
    pse_results = process_all_subjects()
    if pse_results:
        print("\nSummary Statistics:")
        print(f"Number of subjects: {len(pse_results)}")
    return pse_results


if __name__ == "__main__":
    main()
