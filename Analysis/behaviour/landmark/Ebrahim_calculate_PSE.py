"""
Landmark Behavioral Analysis

This script analyzes landmark behavioral data using binned linear analysis 
with Weibull distribution fitting to determine Point of Subjective Equality (PSE).

Author: Mohammad Ebrahim Katebi
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
    'font.size': 10,
    'font.family': 'Arial',
    'axes.linewidth': 1,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,
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
    """
    Weibull cumulative distribution function with scaling and bias parameters.
    """
    y = weibull_min.cdf(x, WEIBULL_SHAPE_FIXED, location, scale)
    return (y * Y_SCALE_GUESS) + Y_BIAS_GUESS


def weibull_ppf_function(percentile, shape, location, scale, y_scale, y_bias):
    """
    Weibull percent point function (inverse CDF) with unscaling.
    """
    unscaled_percentile = (percentile - Y_BIAS_GUESS) / Y_SCALE_GUESS
    return weibull_min.ppf(unscaled_percentile, WEIBULL_SHAPE_FIXED, location, scale)


def bin_data_linear(shift_values, bin_width=BIN_WIDTH):
    """
    Bin data using linear binning approach.
    """
    return np.floor(shift_values / bin_width) * bin_width


def prepare_behavioral_data(data):
    """
    Prepare behavioral data for analysis using binned linear approach.
    """
    # Create signed shift sizes
    data['shift_size_signed'] = np.where(
        data['Shift_Direction'] == 'Left',
        -data['Shift_Size'],
        data['Shift_Size']
    )

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

    # Rename columns
    summary_table.columns = ['shift_bin', 'trial_count',
                             'right_responses', 'proportion_right']

    return summary_table, 'shift_bin'


def fit_weibull_distribution(x_data, y_data):
    """
    Fit Weibull distribution to psychometric data.
    """
    # Create fitting range
    x_fit = np.linspace(min(x_data), max(x_data), len(x_data) * 10)

    # Initial parameter estimation
    shape_init, loc_init, scale_init = weibull_min.fit(x_fit)

    # Fit Weibull CDF to data
    try:
        optimal_params, _ = curve_fit(
            weibull_cdf_function,
            x_data,
            y_data,
            p0=[shape_init, loc_init, scale_init, Y_SCALE_GUESS, Y_BIAS_GUESS],
            maxfev=100000,
            check_finite=False
        )

        # Generate fitted curve
        y_fit = weibull_cdf_function(x_fit, *optimal_params)

        # Calculate Point of Subjective Equality (50% point)
        pse = weibull_ppf_function(0.5, *optimal_params)

        # Calculate R-squared
        y_pred = weibull_cdf_function(x_data, *optimal_params)
        r2 = r2_score(y_data, y_pred)

    except Exception as e:
        print(f"Warning: Fitting failed - {e}")
        y_fit = np.full_like(x_fit, 0.5)
        pse = 0.0
        r2 = 0.0

    return x_fit, y_fit, pse, r2


def create_figure(x_data, y_data, x_fit, y_fit, pse, r2, subject_name):
    """
    Create publication-ready psychometric function figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot data points
    ax.scatter(x_data, y_data, s=60, c='black', marker='o',
               alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)

    # Plot fitted curve
    ax.plot(x_fit, y_fit, color='red', linewidth=2,
            label='Weibull fit', zorder=4)

    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='--',
               linewidth=2, alpha=0.7, label='Midpoint')
    ax.axvline(x=pse, color='blue', linestyle=':', linewidth=2, label=f'PSE')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2)

    # Formatting
    ax.set_xlabel('Horizontal Line Offset (Visual Degrees)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion Answered Right', fontsize=14, fontweight='bold')
    ax.set_title(f'{subject_name}', fontsize=14, fontweight='bold', pad=20)

    # Set axis limits and ticks
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(-1.0, 1.1, 0.2))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # Add PSE annotation
    bias_direction = 'Leftward' if pse > 0 else 'Rightward' if pse < 0 else 'No'
    ax.text(0.02, 0.98, f'Bias = {-1 * pse:.3f}°\n{bias_direction}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.0),
            fontsize=12)

    # Legend and grid
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.0, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def analyze_subject_data(file_path):
    """
    Analyze behavioral data for a single subject.
    """
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
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return pse


def process_all_subjects():
    """
    Process all subject data files and compile results.
    """
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
                                        filename.replace('_logfile.csv', '')
                                    )
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
