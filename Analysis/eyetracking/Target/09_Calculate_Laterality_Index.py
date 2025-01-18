import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up directories
DATA_DIR = "E:/Target_Data"
EYETRACKING_DIR = r"../../../Results/EyeTracking/Target"
OUTPUT_DIR = r"../../../Results/EyeTracking/Target/Laterality_Correlations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_laterality_index(data, cue_direction=None):
    """Calculate laterality index from directional data for specific cue direction."""
    # Filter data by cue direction if specified
    if cue_direction is not None:
        data = data[data['Cue_Direction'] == cue_direction]

    R = data[(data['Direction'] >= 315) | (
        data['Direction'] <= 45)]['Total_Amplitude'].sum()
    L = data[(data['Direction'] >= 135) & (
        data['Direction'] <= 225)]['Total_Amplitude'].sum()
    return (R - L) / (R + L) if (R + L) != 0 else 0


def process_subject_data(file_path):
    """Process individual subject's microsaccade data."""
    data = pd.read_csv(file_path)
    data = data[data['Epoch_Exclusion'] == 0]

    # Apply direction correction
    mask = data['Distance_To_Fixation'] < 0
    data.loc[mask, 'Direction'] = (data.loc[mask, 'Direction'] + 180) % 360

    # Calculate laterality indices for each cue direction
    laterality_right = calculate_laterality_index(data, cue_direction="Right")
    laterality_left = calculate_laterality_index(data, cue_direction="Left")
    laterality_overall = calculate_laterality_index(data)  # Overall laterality

    subject_id = os.path.basename(file_path).split(
        '_')[0].lower().replace('e01s', '').replace('.asc', '')

    return subject_id, laterality_overall, laterality_right, laterality_left


def create_laterality_histogram(df_laterality):
    """Create histogram of laterality indices."""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df_laterality.copy()

    # Reset index before removing outliers
    df_plot = df_plot.reset_index(drop=True)

    # Remove outliers for histogram
    data_clean, n_outliers = remove_outliers(
        df_plot, 'Microsaccade_Laterality_Index')

    sns.histplot(data=data_clean, x='Microsaccade_Laterality_Index',
                 kde=True, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    plt.title('Distribution of Microsaccade Laterality Indices',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Microsaccade Laterality Index', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    mean = data_clean['Microsaccade_Laterality_Index'].mean()
    std = data_clean['Microsaccade_Laterality_Index'].std()
    plt.axvline(x=mean, color='red', linestyle='--', linewidth=1.5)

    t_stat, p_value = stats.ttest_1samp(
        data_clean['Microsaccade_Laterality_Index'], 0)
    sig_text = "Significant" if p_value < 0.05 else "Not Significant"

    stats_text = (f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\n'
                  f'Difference From Zero:\np-value: {p_value:.4f}\n{sig_text}\n'
                  f'Outliers removed: {n_outliers}')

    plt.text(0.97, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'Microsaccade_Laterality_Index_Histogram_NoOutliers.png'),
        dpi=300, bbox_inches='tight')
    plt.close()

    return mean, p_value


def create_comparative_laterality_plot(df_laterality):
    """Create box plot comparing laterality indices across cue directions."""
    plt.figure(figsize=(10, 6))

    # Create a copy and reset index
    df_plot = df_laterality.copy()
    df_plot = df_plot.reset_index(drop=True)

    # Prepare data for plotting
    plot_data = pd.melt(df_plot,
                        value_vars=['Microsaccade_Laterality_Index_Right',
                                    'Microsaccade_Laterality_Index_Left'],
                        var_name='Condition', value_name='Laterality Index')

    # Create box plot
    sns.boxplot(x='Condition', y='Laterality Index', data=plot_data)

    # Perform statistical test between conditions
    t_stat, p_value = stats.ttest_rel(
        df_plot['Microsaccade_Laterality_Index_Right'],
        df_plot['Microsaccade_Laterality_Index_Left']
    )

    plt.title('Laterality Indices by Cue Direction\n' +
              f'Paired t-test p-value: {p_value:.3f}',
              fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'Laterality_By_Cue_Direction.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def load_csv_files(directory):
    """Load all CSV files from directory."""
    return {file: pd.read_csv(os.path.join(directory, file))
            for file in os.listdir(directory)
            if file.endswith('.csv')}


def process_subject_id(subject_id):
    """Process subject ID to consistent format."""
    return str(int(float(subject_id))) if not pd.isna(subject_id) else np.nan


def merge_dataframes(csv_files):
    """Merge multiple dataframes preserving common columns."""
    merged_df = None
    for df in csv_files.values():
        if 'SubID' in df.columns:
            df['SubID'] = df['SubID'].apply(process_subject_id)
            df = df.dropna(subset=['SubID'])
            if merged_df is None:
                merged_df = df.copy()
            else:
                common_columns = list(set(merged_df.columns) & set(df.columns))
                for col in common_columns:
                    if col != 'SubID':
                        merged_df.loc[:, col] = merged_df[col].fillna(df[col])
                new_columns = list(set(df.columns) - set(merged_df.columns))
                for col in new_columns:
                    merged_df.loc[:, col] = df[col]
    return merged_df


def remove_outliers(data, column, n_std=100):
    """Remove outliers beyond n standard deviations."""
    # Reset index to avoid duplicate index issues
    data = data.reset_index(drop=True)

    mean = data[column].mean()
    std = data[column].std()
    outlier_mask = (data[column] - mean).abs() <= n_std * std

    # Return filtered data and number of outliers
    return data[outlier_mask], sum(~outlier_mask)


def create_correlation_plot(ax, data, x_col, y_col, title):
    """Create individual correlation plot with cleaned data and statistics."""
    # Create a copy and reset index
    data = data.copy().reset_index(drop=True)

    # Remove outliers independently for each metric
    data_clean_x, n_outliers_x = remove_outliers(data, x_col)
    data_clean, n_outliers_y = remove_outliers(data_clean_x, y_col)

    total_outliers = len(data) - len(data_clean)

    if len(data_clean) > 0:
        # Calculate correlations
        slope, intercept, r_value, p_value, _ = stats.linregress(
            data_clean[x_col], data_clean[y_col])
        spearman_corr, spearman_p = stats.spearmanr(
            data_clean[x_col], data_clean[y_col])

        # Create plot
        sns.regplot(x=x_col, y=y_col, data=data_clean,
                    ax=ax, scatter_kws={'alpha': 0.6},
                    line_kws={'color': 'red'}, ci=None)

        # Customize plot
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_col.replace("_", " "), fontsize=10)
        ax.set_ylabel(y_col.replace("_", " "), fontsize=10)

        # Add statistics text
        stats_text = (f'Pearson R² = {r_value**2:.3f}\n'
                      f'Pearson P = {p_value:.3f}\n'
                      f'Spearman R = {spearman_corr:.3f}\n'
                      f'Spearman P = {spearman_p:.3f}\n'
                      f'N = {len(data_clean)}\n'
                      f'Outliers removed = {total_outliers}')

        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white',
                          alpha=0.7,
                          edgecolor='gray'))

        return f"{x_col} vs {y_col}: {total_outliers} outliers removed"
    else:
        ax.set_title(f'{title}\nInsufficient data after outlier removal',
                     fontsize=12)
        return f"{x_col} vs {y_col}: insufficient data"


def create_correlation_plots(merged_df, ms_laterality_df, x_columns, y_column, output_dir):
    """Create and save correlation plots for multiple comparisons."""
    # Set up figure
    nrows, ncols = (len(x_columns) + 3) // 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    axes = axes.flatten()
    outlier_info = []

    # Process data based on whether we're using microsaccade data
    if 'Microsaccade_Laterality_Index' in y_column:
        ms_laterality_df_processed = ms_laterality_df.copy()
        ms_laterality_df_processed['SubID'] = ms_laterality_df_processed['Subject_ID']
        base_df = pd.merge(
            merged_df,
            ms_laterality_df_processed[['SubID', y_column]],
            on='SubID',
            how='inner'
        )
    else:
        base_df = merged_df

    # Create correlation plots
    for i, x_col in enumerate(x_columns):
        if x_col in base_df.columns:
            data = base_df[[x_col, y_column]].dropna()
            if len(data) > 0:
                info = create_correlation_plot(
                    axes[i], data, x_col, y_column,
                    f'{x_col.replace("_", " ")}')
                outlier_info.append(info)

    # Remove empty subplots
    for j in range(len(x_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save plot and outlier information
    plot_filename = f'Correlations_{y_column}_NoOutliers.png'
    plt.savefig(os.path.join(output_dir, plot_filename),
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # Save outlier information
    with open(os.path.join(output_dir, f'outlier_info_{y_column}.txt'), 'w') as f:
        f.write('\n'.join(outlier_info))

    return outlier_info


def create_behavior_ms_correlation_plot(merged_df, ms_laterality_df, output_dir):
    """Create correlation plot between behavioral PSE and microsaccade laterality."""
    # Prepare data
    ms_laterality_df_processed = ms_laterality_df.copy()
    ms_laterality_df_processed['SubID'] = ms_laterality_df_processed['Subject_ID']

    # Create separate plots for each laterality measure
    laterality_measures = ['Microsaccade_Laterality_Index',
                           'Microsaccade_Laterality_Index_Right',
                           'Microsaccade_Laterality_Index_Left']

    results = []
    for measure in laterality_measures:
        # Merge behavioral and microsaccade data
        combined_df = pd.merge(
            merged_df[['SubID', 'PSE_Target']],
            ms_laterality_df_processed[['SubID', measure]],
            on='SubID',
            how='inner'
        )

        plt.figure(figsize=(10, 8))

        # Remove outliers
        combined_df_clean_x, n_outliers_x = remove_outliers(
            combined_df, 'PSE_Target')
        combined_df_clean, n_outliers_y = remove_outliers(
            combined_df_clean_x, measure)

        total_outliers = len(combined_df) - len(combined_df_clean)

        # Calculate correlations
        slope, intercept, r_value, p_value, _ = stats.linregress(
            combined_df_clean['PSE_Target'],
            combined_df_clean[measure]
        )
        spearman_corr, spearman_p = stats.spearmanr(
            combined_df_clean['PSE_Target'],
            combined_df_clean[measure]
        )

        # Create plot
        sns.regplot(x='PSE_Target', y=measure,
                    data=combined_df_clean,
                    scatter_kws={'alpha': 0.6},
                    line_kws={'color': 'red'}, ci=None)

        plt.title(f'Target PSE vs {measure}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Target PSE', fontsize=12)
        plt.ylabel(measure, fontsize=12)

        # Add statistics text
        stats_text = (f'Pearson R² = {r_value**2:.3f}\n'
                      f'Pearson P = {p_value:.3f}\n'
                      f'Spearman R = {spearman_corr:.3f}\n'
                      f'Spearman P = {spearman_p:.3f}\n'
                      f'N = {len(combined_df_clean)}\n'
                      f'Outliers removed = {total_outliers}')

        plt.text(0.05, 0.95, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0.8,
                           edgecolor='gray'))

        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(output_dir, f'Behavior_Microsaccade_Correlation_{measure}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        results.append(f"{measure}: {total_outliers} outliers removed")

    return "\n".join(results)


def main():
    # Process microsaccade data
    laterality_indices = []
    for item in os.listdir(EYETRACKING_DIR):
        file_dir = os.path.join(EYETRACKING_DIR, item)
        if os.path.isdir(file_dir):
            for file in os.listdir(file_dir):
                if file.endswith("_Final_MicroSaccade_Data.csv"):
                    file_path = os.path.join(file_dir, file)
                    try:
                        subject_id, laterality_overall, laterality_right, laterality_left = process_subject_data(
                            file_path)
                        laterality_indices.append(
                            (subject_id, laterality_overall, laterality_right, laterality_left))
                    except Exception as e:
                        logging.error(
                            f"Error processing file {file_path}: {str(e)}")

    # Create and save laterality indices
    df_laterality = pd.DataFrame(laterality_indices, columns=[
        'Subject_ID', 'Microsaccade_Laterality_Index',
        'Microsaccade_Laterality_Index_Right', 'Microsaccade_Laterality_Index_Left'])

    # Save raw laterality indices
    df_laterality.to_csv(os.path.join(
        OUTPUT_DIR, 'Laterality_Indices.csv'), index=False)

    # Create comparative visualization
    try:
        create_comparative_laterality_plot(df_laterality)
        logging.info("Created comparative laterality plot")
    except Exception as e:
        logging.error(f"Error creating comparative plot: {str(e)}")

    # Create histograms and get statistics for each condition
    for column in ['Microsaccade_Laterality_Index',
                   'Microsaccade_Laterality_Index_Right',
                   'Microsaccade_Laterality_Index_Left']:
        try:
            # Create a temporary DataFrame with renamed column
            temp_df = df_laterality.copy()
            temp_df = temp_df.rename(
                columns={column: 'Microsaccade_Laterality_Index'})

            mean, p_value = create_laterality_histogram(temp_df)
            logging.info(
                f"{column} - Mean: {mean:.3f}, p-value: {p_value:.3e}")
        except Exception as e:
            logging.error(f"Error processing {column}: {str(e)}")

    try:
        # Load and merge data
        csv_files = load_csv_files(DATA_DIR)
        merged_df = merge_dataframes(csv_files)

        # Define column groups
        structural_columns = [col for col in merged_df.columns if col not in [
            'SubID', 'Handedness', 'PSE_landmark', 'PSE_Target', 'ms_target']]
        behavioral_columns = ['PSE_landmark', 'PSE_Target']

        # Create correlation plots for each laterality measure
        for y_col in ['Microsaccade_Laterality_Index',
                      'Microsaccade_Laterality_Index_Right',
                      'Microsaccade_Laterality_Index_Left',
                      'PSE_Target']:
            outlier_info = create_correlation_plots(
                merged_df, df_laterality, structural_columns, y_col, OUTPUT_DIR)
            logging.info(f"\nCorrelation analysis for {y_col}:")
            for info in outlier_info:
                logging.info(info)

        # Create behavior vs microsaccade correlation plots
        behavior_ms_info = create_behavior_ms_correlation_plot(
            merged_df, df_laterality, OUTPUT_DIR)
        logging.info(
            f"\nBehavior vs Microsaccade correlation analysis:\n{behavior_ms_info}")

    except Exception as e:
        logging.error(f"Error in correlation analysis: {str(e)}")

    logging.info(f"Analysis complete. Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
