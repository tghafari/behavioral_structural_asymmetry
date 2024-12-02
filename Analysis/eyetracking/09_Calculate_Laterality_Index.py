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
DATA_DIR = r"../../Landmark_Data"
EYETRACKING_DIR = r"../../Results/EyeTracking"
OUTPUT_DIR = r"../../Results/EyeTracking/Laterality_Correlations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_laterality_index(data):
    R = data[(data['Direction'] >= 315) | (
        data['Direction'] <= 45)]['Total_Amplitude'].sum()
    L = data[(data['Direction'] >= 135) & (
        data['Direction'] <= 225)]['Total_Amplitude'].sum()
    return (R - L) / (R + L) if (R + L) != 0 else 0


def process_subject_data(file_path):
    data = pd.read_csv(file_path)
    data = data[data['Epoch_Exclusion'] == 0]
    mask = data['Distance_To_Fixation'] < 0
    data.loc[mask, 'Direction'] = (data.loc[mask, 'Direction'] + 180) % 360
    laterality_index = calculate_laterality_index(data)
    subject_id = os.path.basename(file_path).split(
        '_')[0].lower().replace('e01s', '').replace('.asc', '')
    return subject_id, laterality_index


def create_laterality_histogram(df_laterality):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    sns.histplot(data=df_laterality, x='Microsaccade_Laterality_Index',
                 kde=True, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    plt.title('Distribution of Microsaccade Laterality Indices',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Microsaccade Laterality Index', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    mean = df_laterality['Microsaccade_Laterality_Index'].mean()
    std = df_laterality['Microsaccade_Laterality_Index'].std()
    plt.axvline(x=mean, color='red', linestyle='--', linewidth=1.5)

    t_stat, p_value = stats.ttest_1samp(
        df_laterality['Microsaccade_Laterality_Index'], 0)
    sig_text = "Significant" if p_value < 0.05 else "Not Significant"

    stats_text = f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\n'
    stats_text += f'Difference From Zero:\np-value: {p_value:.4f}\n{sig_text}'

    plt.text(0.97, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'Microsaccade_Laterality_Index_Histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return mean, p_value


def load_csv_files(directory):
    return {file: pd.read_csv(os.path.join(directory, file)) for file in os.listdir(directory) if file.endswith('.csv')}


def process_subject_id(subject_id):
    return str(int(float(subject_id))) if not pd.isna(subject_id) else np.nan


def merge_dataframes(csv_files):
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


def create_scatter_plots(merged_df, ms_laterality_df, output_dir):
    ms_laterality_df['SubID'] = ms_laterality_df['Subject_ID'].apply(
        lambda x: x.lower().replace('e01s', '').replace('.asc', '') if isinstance(x, str) else x)
    merged_df['SubID'] = merged_df['SubID'].astype(str)
    ms_laterality_df['SubID'] = ms_laterality_df['SubID'].astype(str)

    logging.info(f"Number of subjects in merged_df: {len(merged_df)}")
    logging.info(
        f"Number of subjects in ms_laterality_df: {len(ms_laterality_df)}")
    logging.info(f"Columns in merged dataframe: {merged_df.columns.tolist()}")

    handedness_col = 'Handedness' if 'Handedness' in merged_df.columns else None
    structural_columns = [col for col in merged_df.columns if col not in [
        'SubID', 'Handedness', 'PSE_landmark', 'PSE_target', 'ms_target']]
    behavioral_columns = ['PSE_landmark', 'PSE_target'] + \
        ([handedness_col] if handedness_col else [])

    # Structural correlations
    nrows, ncols = (len(structural_columns) + 3) // 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    axes = axes.flatten()

    for i, col in enumerate(structural_columns):
        ax = axes[i]
        x = merged_df[['SubID', col]].dropna()
        y = ms_laterality_df[ms_laterality_df['SubID'].isin(x['SubID'])]
        data = pd.merge(
            x, y[['SubID', 'Microsaccade_Laterality_Index']], on='SubID', how='inner')

        logging.info(f"Column {col}: {len(data)} non-null values")

        if len(data) > 0:
            slope, intercept, r_value, p_value, _ = stats.linregress(
                data[col], data['Microsaccade_Laterality_Index'])
            spearman_corr, spearman_p = stats.spearmanr(
                data[col], data['Microsaccade_Laterality_Index'])
            sns.regplot(x=col, y='Microsaccade_Laterality_Index', data=data,
                        ax=ax, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
            ax.set_title(f'{col.replace("_", " ")}',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel(col.replace("_", " "), fontsize=10)
            ax.set_ylabel('Microsaccade Laterality', fontsize=10)
            ax.text(0.05, 0.95, f'Pearson R² = {r_value**2:.3f}\nPearson P = {p_value:.3f}\n'
                    f'Spearman R = {spearman_corr:.3f}\nSpearman P = {spearman_p:.3f}\nN = {len(data)}',
                    transform=ax.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
        else:
            ax.set_title(f'{col.replace("_", " ")} - No data', fontsize=12)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Structural_Correlations.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # Behavioral correlations
    fig, axes = plt.subplots(1, len(behavioral_columns),
                             figsize=(7*len(behavioral_columns), 6))
    if len(behavioral_columns) == 1:
        axes = [axes]

    for i, col in enumerate(behavioral_columns):
        ax = axes[i]
        if col in merged_df.columns:
            x = merged_df[['SubID', col]].dropna()
            y = ms_laterality_df[ms_laterality_df['SubID'].isin(x['SubID'])]
            data = pd.merge(
                x, y[['SubID', 'Microsaccade_Laterality_Index']], on='SubID', how='inner')

            logging.info(f"Column {col}: {len(data)} non-null values")

            if len(data) > 0:
                slope, intercept, r_value, p_value, _ = stats.linregress(
                    data[col], data['Microsaccade_Laterality_Index'])
                spearman_corr, spearman_p = stats.spearmanr(
                    data[col], data['Microsaccade_Laterality_Index'])
                sns.regplot(x=col, y='Microsaccade_Laterality_Index', data=data, ax=ax, scatter_kws={
                            'alpha': 0.6}, line_kws={'color': 'red'})
                ax.set_title(f'{col.replace("_", " ")}',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel(col.replace("_", " "), fontsize=12)
                ax.set_ylabel('Microsaccade Laterality', fontsize=12)
                ax.text(0.05, 0.95, f'Pearson R² = {r_value**2:.3f}\nPearson P = {p_value:.3f}\n'
                        f'Spearman R = {spearman_corr:.3f}\nSpearman P = {spearman_p:.3f}\nN = {len(data)}',
                        transform=ax.transAxes, verticalalignment='top', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
            else:
                ax.set_title(f'{col.replace("_", " ")} - No data', fontsize=14)
        else:
            ax.set_title(
                f'{col.replace("_", " ")} - Column not found', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Behavioral_Correlations.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def main():
    laterality_indices = []
    for item in os.listdir(EYETRACKING_DIR):
        file_dir = os.path.join(EYETRACKING_DIR, item)
        if os.path.isdir(file_dir):
            for file in os.listdir(file_dir):
                if file.endswith("_Final_MicroSaccade_Data.csv"):
                    file_path = os.path.join(file_dir, file)
                    subject_id, microsaccade_laterality = process_subject_data(
                        file_path)
                    laterality_indices.append(
                        (subject_id, microsaccade_laterality))

    df_laterality = pd.DataFrame(laterality_indices, columns=[
                                 'Subject_ID', 'Microsaccade_Laterality_Index'])
    df_laterality.to_csv(os.path.join(
        OUTPUT_DIR, 'Laterality_Indices.csv'), index=False)
    logging.info(
        f"Saved Laterality_Indices.csv with {len(df_laterality)} subjects")
    logging.info(
        f"Subject_ID values in df_laterality: {df_laterality['Subject_ID'].tolist()}")

    mean, p_value = create_laterality_histogram(df_laterality)
    logging.info(
        f"Microsaccade Laterality Index - Mean: {mean:.3f}, p-value: {p_value:.3e}")

    csv_files = load_csv_files(DATA_DIR)
    logging.info(f"Loaded {len(csv_files)} CSV files")

    merged_df = merge_dataframes(csv_files)
    logging.info(f"Merged dataframe shape: {merged_df.shape}")
    logging.info(f"Columns in merged dataframe: {merged_df.columns.tolist()}")

    create_scatter_plots(merged_df, df_laterality, OUTPUT_DIR)

    logging.info(f"Analysis complete. Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
