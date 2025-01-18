import os
import os.path as op
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Set up directories
DATA_DIR = r"../../../Results/EyeTracking/Target"
OUTPUT_FOLDER_PATH = r"../../../Results/EyeTracking/Target/Raw_Data_Plot"

os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)


def process_file(sub_dir, file):
    """Process a single file and generate the plot."""
    print(f"Processing file: {file}")

    # Load data
    with open(op.join(sub_dir, file), 'rb') as f:
        saccadeinfo = pickle.load(f)
    with open(op.join(sub_dir, f"{file.removesuffix('_EL_saccadeinfo.pkl')}_EL_noblinks_All.pkl"), 'rb') as f:
        blink_data = pickle.load(f)
    with open(op.join(sub_dir, f"{file.removesuffix('_EL_saccadeinfo.pkl')}_EL_params.pkl"), 'rb') as f:
        params = pickle.load(f)

    gaze_both = [epoch[['LX', 'LY', 'RX', 'RY']]
                 for epoch in blink_data[0][:20]]

    # Create figure
    n_epochs = len(gaze_both)
    n_cols = int(np.ceil(np.sqrt(n_epochs)))
    n_rows = int(np.ceil(n_epochs / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(
        6*n_cols, 5*n_rows), sharex=True, sharey=True)
    axs = axs.flatten()

    for epoch, ax in enumerate(axs):
        if epoch < n_epochs:
            plot_epoch(ax, epoch, gaze_both[epoch],
                       params, blink_data, saccadeinfo[epoch])
        else:
            ax.axis('off')

    # Set common labels
    fig.text(0.5, -0.025, 'Time (seconds)', ha='center', fontsize=18)
    fig.text(-0.02, 0.5, 'Gaze position (degrees from fixation)',
             va='center', rotation='vertical', fontsize=18)

    # Set main title (file name)
    fig.suptitle(file, fontsize=22, y=1.01)

    # Adjust layout and save plot
    plt.tight_layout()
    output_file = f"{file.removesuffix('_EL_saccadeinfo.pkl')}_Raw_Data.png"
    plt.savefig(os.path.join(OUTPUT_FOLDER_PATH, output_file),
                dpi=200, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)

    print(f"Figure saved as: {output_file}")


def plot_epoch(ax, epoch, epoch_gaze, params, blink_data, saccadeinfo):
    """Plot a single epoch."""
    epoch_gaze = np.array(epoch_gaze)

    # Process gaze data
    epoch_gaze[:, [0, 2]] -= params['ScreenResolution'][0] / 2
    epoch_gaze[:, [1, 3]] -= params['ScreenResolution'][1] / 2
    epoch_gaze *= params['cmPerPixel'][0]
    epoch_gaze = np.degrees(np.arctan(epoch_gaze / params['ViewDistance']))

    time = np.arange(len(epoch_gaze)) / 500  # Convert to seconds

    # Plot gaze data
    for i, label in enumerate(['LX', 'LY', 'RX', 'RY']):
        ax.plot(time, epoch_gaze[:, i], label=label)
    ax.axis(ymin=-7, ymax=7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot blinks
    blink_mask = blink_data[2][epoch] == 1
    ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1], where=(
        blink_mask * 2), color='red', alpha=0.1, ec=None)

    # Plot microsaccades
    epoch_microsaccadeinfo = saccadeinfo['Microsaccades']
    if epoch_microsaccadeinfo['both'] is not None and len(epoch_microsaccadeinfo['both']) > 0:
        epoch_microsaccadeinfo_both = pd.DataFrame(
            epoch_microsaccadeinfo['both']).reset_index(drop=True)
        for _, microsaccade in epoch_microsaccadeinfo_both.iterrows():
            start_time = microsaccade['start'] / 500
            end_time = microsaccade['end'] / 500
            ax.axvspan(start_time, end_time, color='blue', alpha=0.1, ec=None)

        direction = f"{epoch_microsaccadeinfo_both['direction'][0]:.2f}"
        total_amplitude = f"{epoch_microsaccadeinfo_both['total_amplitude'][0]:.2f}"
        distance_to_fixation = f"{epoch_microsaccadeinfo_both['distance_to_fixation'][0]:.2f}"
    else:
        direction = total_amplitude = distance_to_fixation = "N/A"

    # Set subplot title
    stim_direction = blink_data[3]['epoch_Direction'][epoch]
    epoch_exclusion = int(blink_data[3]['epoch_Exclusion'][epoch])
    ax.set_title(f'Epoch {epoch+1}\nDir: {direction}, Amp: {total_amplitude}, Dist: {distance_to_fixation}\n'
                 f'Stim_Direction: {stim_direction}',
                 fontsize=10, pad=10)
    ax.legend(loc='upper right', fontsize='small')

    # Add "Excluded" stamp if Epoch_Exclusion is 1
    if epoch_exclusion == 1:
        ax.text(0.5, 0.9, 'Excluded', transform=ax.transAxes, fontsize=12,
                color='red', fontweight='bold', ha='center', va='bottom', alpha=0.8)


def main():
    for item in os.listdir(DATA_DIR):
        if item.startswith("sub-"):
            sub_dir = os.path.join(DATA_DIR, item)
            for file in os.listdir(sub_dir):
                if file.endswith("_EL_saccadeinfo.pkl"):
                    process_file(sub_dir, file)

    print("All files processed.")


if __name__ == "__main__":
    main()
