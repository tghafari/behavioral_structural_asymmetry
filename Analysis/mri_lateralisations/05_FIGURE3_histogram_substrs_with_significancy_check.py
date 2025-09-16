# -*- coding: utf-8 -*-
"""
===============================================
05_FIGURE3_histogram_substrs_with_significancy_check.py

This script plots lateralisation volume indices 
across participants for 7 subcortical structures
from a *collated* dataset (or any CSV you pass in).
Saves figure as TIFF/PNG/SVG (dpi=800). 
Each subplot shows a Wilcoxon p-value vs 0

written by Tara Ghafari
tara.ghafari@gmail.com
19/08/2025
==============================================
"""


import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ----------------------- Config / Paths ----------------------- #
platform = 'mac'  # 'mac' or 'bluebear'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    fig_output_root = '/rds/projects/j/jenseno-avtemporal-attention/Manuscript/Figures'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    fig_output_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/landmark-manus/Figures'
else:
    raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

# ---- Collated dataset (edit if needed) ----
volume_sheet_dir = op.join(
    jenseno_dir,
    'Projects/subcortical-structures/SubStr-and-behavioral-bias/data/collated'
)
lat_index_csv = op.join(
    volume_sheet_dir,
    'FINAL_unified_behavioral_structural_asymmetry_lateralisation_indices_1_45-nooutliers_flipped.csv'
)

# Output figure basename (without extension)
save_basename = 'Figure3_substr_hist'


# ---------------- Structures & Colors ---------------- #
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
# 7-color palette (your original order):
colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']


# ----------------------- Utilities ----------------------- #
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_figure_all_formats(fig: plt.Figure, out_dir: str, basename: str, dpi: int = 800) -> None:
    ensure_dir(out_dir)
    base = basename.replace(' ', '_')
    fig.savefig(op.join(out_dir, f"{base}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.svg"), format='svg', bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.eps"), format='eps', bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.pdf"), format='pdf', bbox_inches='tight')



# ----------------------- Loading ------------------------ #
def load_collated_dataframe(csv_path: str,
                            structures: list[str],
                            ) -> pd.DataFrame:
    """
    Load the collated lateralisation CSV and (optionally) filter to a subject list.

    Assumptions:
      - CSV contains the 7 structure columns with the given names (case-insensitive),
        OR the structures are in columns 3..9 (0-based; as in your earlier code).

    Parameters
    ----------
    csv_path : str
        Path to collated CSV.
    structures : list[str]
        Expected structure names (7).
    throw_out_outliers:bool 
        do you want to remove outliers? 
        default is False

    Returns
    -------
    df : pd.DataFrame
        DataFrame with at least: ['subjectID'] + structures (7 columns).
    """
    if not op.exists(csv_path):
        raise FileNotFoundError(f"Data CSV not found:\n{csv_path}")

    df = pd.read_csv(csv_path)

    # Map (case-insensitive) structure columns if present;
    # else use positional slice [3:10] like your previous code.
    lower_cols = {c.lower(): c for c in df.columns}
    found = []
    for s in structures:
        key = s.lower()
        if key in lower_cols:
            found.append(lower_cols[key])

    # All structure columns present by name: select them in the requested order
    df_struct = df[['SubID'] + [lower_cols[s.lower()] for s in structures] + ['Landmark_PSE']].copy()
    df_struct.columns = ['SubID'] + structures + ['Landmark_PSE']

   # Remove subjects with missing Landmark_PSE
    if 'Landmark_PSE' not in df_struct.columns:
        raise RuntimeError("Column 'Landmark_PSE' not found in the collated CSV.")
    before_len = len(df_struct)
    df_struct = df_struct.dropna(subset=['Landmark_PSE']).reset_index(drop=True)
    after_len = len(df_struct)
    print(f"Removed {before_len - after_len} subjects with missing Landmark_PSE (final N={after_len}).")

    return df_struct


# ----------------------- Plotting ----------------------- #
def plot_lateralisation_volumes(df: pd.DataFrame,
                                structures: list[str],
                                colormap: list[str],
                                bins: int = 10,
                                title: str = 'Lateralisation Volume of Subcortical Structures (N=44)',
                                throw_out_outliers:bool = False):
    """
    Plot histograms for each subcortical structure, annotate with Wilcoxon p-value vs 0,
    and save at 800 dpi (Arial; title and labels 14; y-label padding fixed; tick params labelsize=8,
    texts and legends fontsize=10).
    """
    # Use Arial globally
    plt.rcParams['font.family'] = 'Arial'

    n_structures = len(structures)
    n_cols = 4
    n_rows = int(np.ceil(n_structures / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6.3, 4))
    axs = axs.flatten()

    # Styled p-value box (your settings)
    box_props = dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round')

    null_hypothesis_median = 0.0

    for idx, structure in enumerate(structures):
        ax = axs[idx]
        lateralisation_volumes = pd.to_numeric(df[structure], errors='coerce').values
        
        # --- optional outlier handling per structure ---
        """we are not doing this for the paper."""
        if throw_out_outliers:
            med = np.nanmedian(lateralisation_volumes)
            sd  = np.nanstd(lateralisation_volumes)
            thr_neg = med - 2.5 * sd
            thr_pos = med + 2.5 * sd

            # build mask of outliers across all structure cells
            mask_out = (lateralisation_volumes <= thr_neg) | (lateralisation_volumes >= thr_pos)
            n_out = int(np.nansum(mask_out))

            # set outlier cells to NaN in the DataFrame
            lateralisation_volumes[mask_out] = np.nan
            df[structure] = lateralisation_volumes  # write back

            print(f"Outlier marking: median={med:.4f}, std={sd:.4f}, "
                f"thresholds=({thr_neg:.4f}, {thr_pos:.4f}), "
                f"cells set to NaN={n_out}")
        
        if lateralisation_volumes.size == 0:
            ax.set_visible(False)
            continue

        # Histogram
        ax.hist(lateralisation_volumes, bins=bins, color=colormap[idx], edgecolor='white')

        # Zero reference line
        ax.axvline(x=0.0, color='dimgray', linewidth=0.8, linestyle='-')

        # Wilcoxon signed-rank test vs 0
        diffs = lateralisation_volumes - null_hypothesis_median
        if np.allclose(diffs, 0.0):
            wilcox_p = 1.0
        else:
            _, wilcox_p = stats.wilcoxon(diffs, zero_method='wilcox', correction=False, alternative='two-sided')

        txt_wilx = f"Wilcoxon p = {wilcox_p:.3f}" if wilcox_p >= 0.001 else "Wilcoxon p < 0.001"
        ax.text(0.05, 0.95,
                txt_wilx,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=box_props,
                style='italic')

        # Axis labels & title (Arial, 14)
        ax.set_title(structure, fontsize=10)
        # Only bottom row gets x-labels in a 2x4 grid: indices 3,4,5,6 show x-label
        if idx in [3, 4, 5, 6]:
            ax.set_xlabel('Lateralisation Volume', fontsize=10)
        # First column of each row gets y-label (indices 0 and 4)
        if idx in [0, 4]:
            ax.set_ylabel('# Subjects', fontsize=10, labelpad=5)

        # Ticks styling
        ax.tick_params(axis='both', which='both', length=0, labelsize=8)
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', alpha=0.25)
        x_symmetric = np.max([np.abs(np.min(lateralisation_volumes)), np.max(lateralisation_volumes)])
        ax.set_xlim(-x_symmetric, x_symmetric)

    # Remove unused axes
    for ax in axs:
        if not ax.has_data():
            fig.delaxes(ax)

    # Title + layout
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.subplots_adjust(
    wspace=0.2,  # horizontal space (smaller = closer)
    hspace=0.3
)
    # Save
    out_dir = op.join(fig_output_root)
    save_figure_all_formats(fig, out_dir, save_basename, dpi=800)
    plt.show()
    plt.close(fig)


# ----------------------- Run ----------------------- #
if __name__ == '__main__':
    # If you also have a subject list to filter by, pass it as subject_list_csv=...
    df_struct = load_collated_dataframe(
        structures=structures,
        csv_path=lat_index_csv,
  )
    plot_lateralisation_volumes(
        df=df_struct,
        structures=structures,
        colormap=colormap,
        bins=8,  
        title='Lateralisation Volume of Subcortical Structures (N=44)'
    )


# from this part keep the ranksum tests etc
# for his in range(7): 
    # Define plot settings 
    # ax = axs[his // 4, his % 4] 
    # ax.set_title(structures[his], fontsize=14, fontname='Arial', fontweight='bold') 
    # ax.set_xlabel('Lateralisation Volume', fontsize=12, fontname='Arial', fontweight='bold') 
    # ax.set_ylabel('# Subjects', fontsize=12, fontname='Arial', fontweight='bold') ax.axvline(x=0, color='dimgray', linewidth=0.5, linestyle='-') 
    # # Compute statistics 
    # median_val = lateralisation_volume[:,his].mean() #TODO:shouldn't this be median instead of mean? medians.append(median_val) 
    # # Remove nans and plot normalized (z-scored) distributions 
    # valid_lateralisation_volume = lateralisation_volume[~np.isnan(lateralisation_volume[:, his]), his] 
    # lateralisation_volume_hist = np.histogram(valid_lateralisation_volume, bins=6, density=False) # Throw out the outliers 
    # mean_lateralisation_volume = np.nanmean(valid_lateralisation_volume) 
    # std_lateralisation_volume = np.nanstd(valid_lateralisation_volume) 
    # threshold = mean_lateralisation_volume - (2.5 * std_lateralisation_volume) 
    # valid_lateralisation_volume[:][valid_lateralisation_volume[:] <= threshold] = np.nan 
    # print(len(valid_lateralisation_volume)) 
    # # Perform the ranksum test 
    # k2, p = stats.normaltest(valid_lateralisation_volume, nan_policy='omit') 
    # p_values.append(p) 
    # stat, shapiro_p = shapiro(valid_lateralisation_volume) 
    # p_values_shapiro.append(shapiro_p) # 1 sample t-test for left/right lateralisation 
    # # t_statistic, t_p_value = stats.ttest_1samp(valid_lateralisation_volume, 
    # # null_hypothesis_mean, # nan_policy='omit') 
    # # t_stats.append(t_statistic) 
    # # t_p_vals.append(t_p_value) 
    # # txt_t = r'$1samp\_p = {:.2f}$'.format(t_p_value) 
