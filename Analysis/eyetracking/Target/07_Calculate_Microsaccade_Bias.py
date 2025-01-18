import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import circmean, circstd
import copy

# Set up directories
DATA_DIR =  r"../../../Results/EyeTracking/Target"
OUTPUT_FOLDER_PATH =  r"../../../Results/EyeTracking/Target/Microsaccade_Analysis"

os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)


def aggregate_data(data_dir):
    all_data = []
    count = 0

    for item in os.listdir(data_dir):
        if item.startswith("sub-"):
            sub_dir = os.path.join(data_dir, item)
            for file in os.listdir(sub_dir):
                if file.endswith("_Final_MicroSaccade_Data.csv"):
                    file_path = os.path.join(sub_dir, file)
                    data = pd.read_csv(file_path)
                    data = data[data['Epoch_Exclusion'] == 0]
                    all_data.append(data)
                    count += 1

    return pd.concat(all_data, ignore_index=True), count


def create_figure_1(data, file_name):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        f"Microsaccade Analysis - {file_name}", fontsize=22, fontweight='bold', y=1.0)

    # Polar histogram
    ax1 = fig.add_subplot(431, projection='polar')
    direction_rad = np.deg2rad(data['Direction'])
    ax1.hist(direction_rad, bins=32, range=(0, 2*np.pi))
    ax1.set_theta_zero_location('E')
    ax1.set_theta_direction(1)
    ax1.set_title('Distribution of Microsaccade Directions')

    # Polar scatter plot
    ax2 = fig.add_subplot(432, projection='polar')
    palette = sns.color_palette("Set1")
    color_map = {'Left': palette[0], 'Right': palette[1]}
    colors = data['Cue_Direction'].map(color_map)
    ax2.scatter(np.deg2rad(data['Direction']),
                data['Total_Amplitude'], c=colors, alpha=0.5, s=8)
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.set_title('Microsaccades in Polar Coordinates')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=direction, markerfacecolor=color, markersize=10)
                       for direction, color in color_map.items()]
    ax2.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1.25, 1))

    # Mean amplitude by binned direction
    ax3 = fig.add_subplot(433)
    data['Direction_binned'] = pd.cut(
        data['Direction'], bins=8, labels=range(1, 9))
    direction_amplitude = data.groupby('Direction_binned')[
        'Total_Amplitude'].agg(['mean', 'std'])
    direction_amplitude['mean'].plot(
        kind='bar', yerr=direction_amplitude['std'], ax=ax3, capsize=5)
    ax3.set_xlabel('Direction Bin')
    ax3.set_ylabel('Mean Amplitude')
    ax3.set_title('Mean Amplitude by Binned Direction')

    # Mean amplitude by stim direction
    ax4 = fig.add_subplot(434)
    stim_direction_amplitude = data.groupby(
        'Cue_Direction')['Total_Amplitude'].agg(['mean', 'std'])
    stim_direction_amplitude['mean'].plot(
        kind='bar', yerr=stim_direction_amplitude['std'], ax=ax4, capsize=5)
    ax4.set_xlabel('Cue Direction')
    ax4.set_ylabel('Mean Amplitude')
    ax4.set_title('Mean Amplitude by Cue Direction')

    # Distance to fixation in polar coordinates
    ax6 = fig.add_subplot(436, projection='polar')
    sc_pos = ax6.scatter(np.deg2rad(data[data['Distance_To_Fixation'] >= 0]['Direction']),
                         data[data['Distance_To_Fixation']
                              >= 0]['Distance_To_Fixation'],
                         c='blue', alpha=0.5, s=4, label='Positive')
    sc_neg = ax6.scatter(np.deg2rad(data[data['Distance_To_Fixation'] < 0]['Direction']),
                         data[data['Distance_To_Fixation']
                              < 0]['Distance_To_Fixation'],
                         c='red', alpha=0.5, s=4, label='Negative')
    ax6.set_theta_zero_location('E')
    ax6.set_theta_direction(1)
    ax6.set_title('Distance to Fixation in Polar Coordinates')
    ax6.legend()

    # Mean distance to fixation by binned direction
    ax7 = fig.add_subplot(437)
    direction_distance = data.groupby('Direction_binned')[
        'Distance_To_Fixation'].agg(['mean', 'std'])
    direction_distance['mean'].plot(
        kind='bar', yerr=direction_distance['std'], ax=ax7, capsize=5)
    ax7.set_xlabel('Direction Bin')
    ax7.set_ylabel('Mean Distance to Fixation')
    ax7.set_title('Mean Distance to Fixation by Binned Direction')

    # Mean distance to fixation by stim direction
    ax8 = fig.add_subplot(438)
    stim_direction_distance = data.groupby('Cue_Direction')[
        'Distance_To_Fixation'].agg(['mean', 'std'])
    stim_direction_distance['mean'].plot(
        kind='bar', yerr=stim_direction_distance['std'], ax=ax8, capsize=5)
    ax8.set_xlabel('Cue Direction')
    ax8.set_ylabel('Mean Distance to Fixation')
    ax8.set_title('Mean Distance to Fixation by Cue Direction')

    # Absolute distance to fixation in polar coordinates
    ax9 = fig.add_subplot(439, projection='polar')
    sc_pos = ax9.scatter(np.deg2rad(data[data['Distance_To_Fixation'] >= 0]['Direction']),
                         abs(data[data['Distance_To_Fixation'] >= 0]
                             ['Distance_To_Fixation']),
                         c='blue', alpha=0.5, s=4, label='Positive')
    sc_neg = ax9.scatter(np.deg2rad(data[data['Distance_To_Fixation'] < 0]['Direction']),
                         abs(data[data['Distance_To_Fixation'] < 0]
                             ['Distance_To_Fixation']),
                         c='red', alpha=0.5, s=4, label='Negative')
    ax9.set_theta_zero_location('E')
    ax9.set_theta_direction(1)
    ax9.set_title('|Distance to Fixation| in Polar Coordinates')
    ax9.legend()

    plt.tight_layout(pad=4.0)
    return fig


def create_figure_2_cue_direction(data, file_name):
    unique_stim_directions = sorted(data['Cue_Direction'].unique())
    # unique_block_types = ['Shorter', 'Longer']

    fig, axs = plt.subplots(2, 4, figsize=(
        20, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle(f"Microsaccade Analysis by Cue Direction - {file_name}",
                 fontsize=22, fontweight='bold', y=1.05)

    for i, stim_direction in enumerate(unique_stim_directions):
        
            filtered_data = data[(data['Cue_Direction'] == stim_direction)]

            ax = axs[i]
            direction_rad = np.deg2rad(filtered_data['Direction'])
            ax.hist(direction_rad, bins=32, range=(0, 2*np.pi))
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_title(
                f'Direction Distribution\n(Stim: {stim_direction}')

            ax = axs[i]
            ax.scatter(np.deg2rad(filtered_data['Direction']), filtered_data['Total_Amplitude'],
                       alpha=0.5, s=6)
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_title(
                f'Microsaccades\n(Stim: {stim_direction}')

    plt.tight_layout(pad=4.0)
    fig.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)

    return fig


def create_figure_3(data, file_name):
    if 'Distance_To_Fixation' not in data.columns:
        return None

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Microsaccade Analysis with Direction Adjustment - {file_name}", fontsize=18, fontweight='bold')

    adjusted_data = copy.deepcopy(data)
    mask = adjusted_data['Distance_To_Fixation'] < 0
    adjusted_data.loc[mask, 'Direction'] = (
        adjusted_data.loc[mask, 'Direction'] + 180) % 360

    ax1 = fig.add_subplot(221, projection='polar')
    direction_rad = np.deg2rad(data['Direction'])
    ax1.hist(direction_rad, bins=32, range=(0, 2*np.pi))
    ax1.set_theta_zero_location('E')
    ax1.set_theta_direction(1)
    ax1.set_title('Original Distribution of Microsaccade Directions')

    ax2 = fig.add_subplot(222, projection='polar')
    adjusted_direction_rad = np.deg2rad(adjusted_data['Direction'])
    ax2.hist(adjusted_direction_rad, bins=32, range=(0, 2*np.pi))
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.set_title('Adjusted Distribution of Microsaccade Directions')

    ax3 = fig.add_subplot(223, projection='polar')
    sc1 = ax3.scatter(np.deg2rad(data['Direction']), data['Total_Amplitude'],
                      c=data['Distance_To_Fixation'], cmap='bwr', alpha=0.5, s=8)
    ax3.set_theta_zero_location('E')
    ax3.set_theta_direction(1)
    ax3.set_title('Original Microsaccades in Polar Coordinates')
    plt.colorbar(sc1, ax=ax3, label='Distance to Fixation')

    ax4 = fig.add_subplot(224, projection='polar')
    sc2 = ax4.scatter(np.deg2rad(adjusted_data['Direction']), adjusted_data['Total_Amplitude'],
                      c=adjusted_data['Distance_To_Fixation'], cmap='bwr', alpha=0.5, s=8)
    ax4.set_theta_zero_location('E')
    ax4.set_theta_direction(1)
    ax4.set_title('Adjusted Microsaccades in Polar Coordinates')
    plt.colorbar(sc2, ax=ax4, label='Distance to Fixation')

    plt.tight_layout(pad=4.0)
    return fig


def process_file(file_path, output_folder):
    print(f"Processing File: {file_path}")

    data = pd.read_csv(file_path)
    data = data[data['Epoch_Exclusion'] == 0]

    base_name = os.path.splitext(os.path.basename(file_path))[
        0].removesuffix('_Final_MicroSaccade_Data')

    fig1 = create_figure_1(data, base_name)
    output_file_1 = f"{base_name}_Microsaccade_Analysis.png"
    fig1.savefig(os.path.join(output_folder, output_file_1),
                 dpi=300, bbox_inches='tight', pad_inches=0.55)
    plt.close(fig1)

    # fig2_stim_direction = create_figure_2_cue_direction(data, base_name)
    # output_file_2_stim_direction = f"{base_name}_Microsaccade_Analysis_CueDirection.png"
    # fig2_stim_direction.savefig(os.path.join(output_folder, output_file_2_stim_direction),
    #                            dpi=200, bbox_inches='tight', pad_inches=0.55)
    # plt.close(fig2_stim_direction)

    fig3 = create_figure_3(data, base_name)
    if fig3:
        output_file_3 = f"{base_name}_Microsaccade_Analysis_Adjusted.png"
        fig3.savefig(os.path.join(output_folder, output_file_3),
                     dpi=300, bbox_inches='tight', pad_inches=0.55)
        plt.close(fig3)


def main():
    for item in os.listdir(DATA_DIR):
        if item.startswith("sub-"):
            sub_dir = os.path.join(DATA_DIR, item)
            for file in os.listdir(sub_dir):
                if file.endswith("_Final_MicroSaccade_Data.csv"):
                    file_path = os.path.join(sub_dir, file)
                    process_file(file_path, OUTPUT_FOLDER_PATH)

    all_data, count = aggregate_data(DATA_DIR)

    avg_fig1 = create_figure_1(
        all_data, f"Landmark - All_Data - {count} Subject(s)")
    avg_output_file_1 = "All_Microsaccade_Analysis.png"
    avg_fig1.savefig(os.path.join(OUTPUT_FOLDER_PATH, avg_output_file_1),
                     dpi=300, bbox_inches='tight', pad_inches=0.55)
    plt.close(avg_fig1)

    # avg_fig2_stim_direction = create_figure_2_cue_direction(
    #     all_data, f"Landmark - All_Data - {count} Subject(s)")
    # avg_output_file_2_stim_direction = "All_Microsaccade_Analysis_StimDirection_BlockType.png"
    # avg_fig2_stim_direction.savefig(os.path.join(OUTPUT_FOLDER_PATH, avg_output_file_2_stim_direction),
    #                                 dpi=200, bbox_inches='tight', pad_inches=0.55)
    # plt.close(avg_fig2_stim_direction)

    avg_fig3 = create_figure_3(
        all_data, f"Landmark - All_Data - {count} Subject(s)")
    if avg_fig3:
        avg_output_file_3 = "All_Microsaccade_Analysis_Adjusted.png"
        avg_fig3.savefig(os.path.join(OUTPUT_FOLDER_PATH, avg_output_file_3),
                         dpi=300, bbox_inches='tight', pad_inches=0.55)
        plt.close(avg_fig3)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
