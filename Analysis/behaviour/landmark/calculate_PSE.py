import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

# Set up directories
DATA_DIR = r"E:/Landmark_Data"
OUTPUT_FOLDER_PATH = r"../../../Results/Beh/Landmark"
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# Define Weibull distribution parameters and functions
Y_SCALE_GUESS, Y_BIAS_GUESS = 1, 0


def weibull_min_cdf(x_weibull, shape, loc, scale, y_scale, y_bias):
    y = weibull_min.cdf(x_weibull, 7, loc, scale)
    y_scaled = (y * Y_SCALE_GUESS) + Y_BIAS_GUESS
    return y_scaled


def weibull_min_ppf(ppf, shape, loc, scale, y_scale, y_bias):
    ppf_unscaled = (ppf - Y_BIAS_GUESS) / Y_SCALE_GUESS
    return weibull_min.ppf(ppf_unscaled, 7, loc, scale)


def data_binner(x, use_log):
    return np.round(x).astype(int) if use_log else np.floor(x / 0.1).astype(int) * 0.1


def prepare_data(data, bin_data=True, use_log=True):

    data['Shift_Size_Signed'] = np.where(
        data['Shift_Direction'] == 'Left', -data['Shift_Size'], data['Shift_Size'])

    if use_log:

        global max_x_log

        max_x_log = np.round(
            np.max(np.log(np.abs(data['Shift_Size_Signed'])) / np.log(0.8)))

        data['x_log'] = np.sign(data['Shift_Size_Signed']) * \
            (max_x_log -
             np.log(np.abs(data['Shift_Size_Signed'])) / np.log(0.8))
        x_values = 'x_log'
    else:
        x_values = 'Shift_Size_Signed'

    if bin_data:
        data['Bin'] = data[x_values].apply(lambda x: data_binner(x, use_log))
        x_values = 'Bin'

    data['Biggerright'] = ((data['Block_Question'] == 'Longer') & (data['Answer'] == 'Right')) | \
                          ((data['Block_Question'] == 'Shorter')
                           & (data['Answer'] == 'Left'))

    table = data.groupby([x_values])['Biggerright'].agg(
        ['count', 'sum', 'mean']).reset_index()
    table.columns = [x_values, 'Bin_Size',
                     'Rights', 'Proportion_Reported_Right']

    return table, x_values


def fit_weibull(x, y):
    x_weibull = np.linspace(min(x), max(x), len(x))
    shape_x, loc_x, scale_x = weibull_min.fit(x_weibull)
    fit, _ = curve_fit(weibull_min_cdf, x, y, p0=[shape_x, loc_x, scale_x, Y_SCALE_GUESS, Y_BIAS_GUESS],
                       maxfev=100000, check_finite=False)
    cdf_y = weibull_min_cdf(x_weibull, *fit)
    pse = weibull_min_ppf(0.5, *fit)
    r2 = r2_score(y, cdf_y)
    return x_weibull, cdf_y, pse, r2


def plot_analysis(ax, x, y, x_weibull, cdf_y, pse, r2, title, x_label, use_log):
    ax.scatter(x, y, marker='x', color='red', s=10)
    ax.plot(x_weibull, cdf_y, 'blue', lw=1.3, label='Weibull CDF')
    ax.axvline(x=0, color='black', linestyle='--', dashes=(5, 3),
               lw=1.75, label='Veridical Midpoint')
    ax.axvline(x=pse, color='grey', lw=1, linestyle=':')
    ax.axhline(y=0.5, color='grey', lw=1, linestyle=':', label='PSE')

    ax.set_xlabel(x_label, fontsize='small', fontweight='bold')
    ax.set_ylabel('Proportion Reported Right',
                  fontsize='small', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])

    if use_log:

        ax.set_xlim(-max_x_log - 0.1, max_x_log + 0.1)
        xaxisticks = [-max_x_log, 0, max_x_log]
        ax.set_xticks(xaxisticks)

    else:

        ax.set_xlim(-1.1, 1.1)
        xaxisticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xticks(xaxisticks)

    bias = 'Leftward Bias' if pse < 0 else 'Rightward Bias' if pse > 0 else 'No Bias'
    ax.legend(loc=2, title=f'PSE={abs(pse):.3f}° ({bias})\nR2={r2:.3f}',
              title_fontsize='small', fontsize='small')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize='small', fontweight='bold', loc='left')


def plot_staircase(data, output_folder, subject_name):
    unique_lengths = data['Line_Lenght'].unique()
    unique_blocks = data['Block_Number'].unique()

    fig, axs = plt.subplots(len(unique_lengths), len(
        unique_blocks), figsize=(6*len(unique_blocks), 5*len(unique_lengths)))
    fig.suptitle(
        f'Staircase Process - {subject_name}', fontsize=22, fontweight='bold')

    color_scheme = {'Left': 'blue', 'Right': 'red'}
    color_correct = 'limegreen'
    color_incorrect = 'red'
    color_missing = 'gray'

    for length_idx, length in enumerate(unique_lengths):
        for block_idx, block in enumerate(unique_blocks):
            ax = axs[length_idx, block_idx] if len(unique_lengths) > 1 and len(
                unique_blocks) > 1 else axs[max(length_idx, block_idx)]

            block_length_data = data[(data['Line_Lenght'] == length) & (
                data['Block_Number'] == block)]

            if not block_length_data.empty:
                for direction in ['Left', 'Right']:
                    direction_data = block_length_data[block_length_data['Shift_Direction'] == direction]
                    trial_index = range(1, len(direction_data) + 1)

                    ax.plot(trial_index, direction_data['Shift_Size'], color=color_scheme[direction],
                            linewidth=2, alpha=0.7, label=f'{direction} Shift')

                    correct_trials = (
                        ((direction_data['Block_Question'] == 'Longer') & (direction_data['Shift_Direction'] == 'Right') & (direction_data['Answer'] == 'Right')) |
                        ((direction_data['Block_Question'] == 'Longer') & (direction_data['Shift_Direction'] == 'Left') & (direction_data['Answer'] == 'Left')) |
                        ((direction_data['Block_Question'] == 'Shorter') & (direction_data['Shift_Direction'] == 'Right') & (direction_data['Answer'] == 'Left')) |
                        ((direction_data['Block_Question'] == 'Shorter') & (
                            direction_data['Shift_Direction'] == 'Left') & (direction_data['Answer'] == 'Right'))
                    )

                    ax.scatter(np.array(trial_index)[correct_trials], direction_data['Shift_Size'][correct_trials],
                               c=color_correct, s=40, edgecolors='k', zorder=3)

                    incorrect_trials = ~correct_trials & (
                        direction_data['State'] == 1)
                    ax.scatter(np.array(trial_index)[incorrect_trials], direction_data['Shift_Size'][incorrect_trials],
                               c=color_incorrect, s=40, edgecolors='k', zorder=3)

                    missing_trials = direction_data['State'] == 3
                    ax.scatter(np.array(trial_index)[missing_trials], direction_data['Shift_Size'][missing_trials],
                               c=color_missing, s=40, edgecolors='k', zorder=3)

            ax.set_ylim(0, data['Shift_Size'].max() * 1.1)
            ax.set_xlim(0, len(block_length_data) // 2 + 1)
            ax.grid(alpha=0.2)

            if length_idx == len(unique_lengths) - 1:
                ax.set_xlabel('Trial Index', fontweight='bold')
            if block_idx == 0:
                ax.set_ylabel('Shift Size', fontweight='bold')

            ax.set_title(f'Length: {length}, Block: {block}',
                         fontsize=16, fontweight='bold')

    custom_lines = [plt.Line2D([0], [0], color=color_scheme['Left'], lw=2),
                    plt.Line2D([0], [0], color=color_scheme['Right'], lw=2),
                    plt.Line2D([0], [0], color=color_correct,
                               marker='o', linestyle='None'),
                    plt.Line2D([0], [0], color=color_incorrect,
                               marker='o', linestyle='None'),
                    plt.Line2D([0], [0], color=color_missing, marker='o', linestyle='None')]
    fig.legend(custom_lines, ['Left Shift', 'Right Shift', 'Correct', 'Incorrect', 'Missing'],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0))

    plt.tight_layout(pad=4)
    plt.subplots_adjust(top=0.95, bottom=0.05)

    savefig_path = op.join(output_folder, f'{subject_name}_Staircase.png')
    fig.savefig(savefig_path, dpi=200)
    plt.close(fig)


def analyze_subject(fpath):
    data = pd.read_csv(fpath)
    subject_name = op.basename(fpath).removesuffix('_logfile.csv')

    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    pses = []

    for i, (bin_data, use_log) in enumerate([(True, True), (True, False), (False, False), (False, True)]):
        table, x_values = prepare_data(data, bin_data, use_log)
        x = table[x_values]
        y = table['Proportion_Reported_Right']

        x_weibull, cdf_y, pse, r2 = fit_weibull(x, y)

        title = f"{'Binned' if bin_data else 'Unbinned'}, {'Log' if use_log else 'Linear'} Scale"
        x_label = f"Horizontal Line Offset ({'Log of ' if use_log else ''}Deg. Vis. Ang.)"

        plot_analysis(axs[i//2, i % 2], x, y, x_weibull,
                      cdf_y, pse, r2, title, x_label, use_log)

        if use_log:
            pse_raw = np.sign(pse) * (0.8 ** (max_x_log - np.abs(pse)))
            pses.append(pse_raw)
        else:
            pses.append(pse)

    fig.suptitle(
        f'Figure 3-A. Subject {subject_name}', fontsize='large', fontweight='bold')
    plt.tight_layout(pad=2.5)

    savefig_path = op.join(OUTPUT_FOLDER_PATH, f"{subject_name}_figure3A.png")
    fig.savefig(savefig_path, dpi=200)
    plt.close(fig)

    plot_staircase(data, OUTPUT_FOLDER_PATH, subject_name)

    return pses


def process_all_subjects():
    all_pses = []
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
                                print(f"\nProcessing File: {file}")
                                fpath = op.join(beh_dir, file)
                                pses = analyze_subject(fpath)
                                all_pses.append(pses)
    return all_pses


def plot_figure_3b(all_pses):
    pse_data = pd.DataFrame(all_pses, columns=[
                            'Binned_Log', 'Binned_Linear', 'Unbinned_Linear', 'Unbinned_Log'])

    # fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    # fig.suptitle('Figure 3-B: Distribution of PSE Values',
    #              fontsize='x-large', fontweight='bold')

    # for i, column in enumerate(pse_data.columns):
    #     pse_values = pse_data[column]
    #     bins = np.linspace(-0.8, 0.8, 17)

    #     bias_data = pd.cut(pse_values, bins=bins)
    #     bias_table = bias_data.value_counts().sort_index()

    #     ax = axs[i//2, i % 2]
    #     bin_midpoints = (bins[:-1] + bins[1:]) / 2

    #     ax.bar(bin_midpoints, bias_table.values, width=0.08,
    #            color='skyblue', edgecolor='navy')

    #     ax.set_xlabel('Spatial Bias (Deg. Vis. Ang.)',
    #                   fontsize='medium', fontweight='bold')
    #     ax.set_ylabel('Number of Subjects',
    #                   fontsize='medium', fontweight='bold')
    #     ax.set_xlim(-0.8, 0.8)
    #     ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    #     ax.set_xticklabels(['-0.8°', '-0.4°', '0°', '+0.4°', '+0.8°'])
    #     ax.set_ylim(0, max(bias_table.values) + 1)

    #     ax.axvline(x=0, color='red', linestyle='--',
    #                lw=1.5, label='Veridical Midpoint')
    #     ax.text(-0.75, ax.get_ylim()[1]*0.95,
    #             'LVF Bias', fontsize=10, color='darkgreen')
    #     ax.text(0.55, ax.get_ylim()[1]*0.95,
    #             'RVF Bias', fontsize=10, color='darkgreen')

    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.set_title(column.replace('_', ' '),
    #                  fontsize='large', fontweight='bold')

    #     ax.legend(loc='upper left')

    # plt.tight_layout()
    # savefig_path_3b = op.join(OUTPUT_FOLDER_PATH, 'figure3B.png')
    # plt.savefig(savefig_path_3b, dpi=300)
    # plt.close(fig)

    pse_data.to_csv(op.join(OUTPUT_FOLDER_PATH, 'PSE_values.csv'), index=False)


if __name__ == "__main__":
    all_pses = process_all_subjects()
    plot_figure_3b(all_pses)
