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
EYETRACKING_DIR = r"../../../Results/EyeTracking/Target"
OUTPUT_DIR = r"../../../Results/EyeTracking/Target/Microsaccades_Per_Epoch"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_microsaccades_per_trial():
    """Analyze the distribution of microsaccades per epoch across all subjects."""
    all_trial_counts = []
    total_epochs = 0
    num_subjects = 0

    # Process each subject's data
    for item in os.listdir(EYETRACKING_DIR):
        file_dir = os.path.join(EYETRACKING_DIR, item)
        if os.path.isdir(file_dir):
            for file in os.listdir(file_dir):
                if file.endswith("_Final_MicroSaccade_Data.csv"):
                    file_path = os.path.join(file_dir, file)
                    try:
                        num_subjects += 1  # Count number of subjects
                        # Load data
                        data = pd.read_csv(file_path)
                        # Only include valid epochs
                        data = data[data['Epoch_Exclusion'] == 0]

                        # Get unique epochs and their microsaccade counts
                        epoch_counts = data['Epoch'].value_counts()

                        # Get all unique epochs to account for epochs with 0 microsaccades
                        max_epoch = int(data['Epoch'].max())
                        all_epochs = pd.Series(
                            0, index=range(1, max_epoch + 1))
                        epoch_counts = epoch_counts.add(
                            all_epochs, fill_value=0)

                        all_trial_counts.extend(epoch_counts.values)
                        total_epochs += len(epoch_counts)

                        logging.info(f"Processed {file}")
                    except Exception as e:
                        logging.error(
                            f"Error processing file {file_path}: {str(e)}")

    return all_trial_counts, total_epochs, num_subjects


def create_microsaccade_histogram(ms_counts, total_epochs, num_subjects):
    """Create histogram showing distribution of microsaccades per epoch."""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({'Microsaccades': ms_counts})

    # Create main histogram with normalized counts
    ax = sns.histplot(data=df,
                      x='Microsaccades',
                      stat='count',
                      discrete=True,
                      color='skyblue',
                      alpha=0.6,
                      edgecolor='black',
                      linewidth=1)

    # Get the histogram data and normalize by number of subjects
    heights = [p.get_height() / num_subjects for p in ax.patches]

    # Update bar heights with normalized values
    for patch, new_height in zip(ax.patches, heights):
        patch.set_height(new_height)

    # Add normalized count labels on top of each bar
    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width()/2., p.get_height(),
                f'{heights[i]:.1f}',
                ha='center', va='bottom')

    # Calculate statistics
    mean = np.mean(ms_counts)
    median = np.median(ms_counts)
    std = np.std(ms_counts)
    zero_ms_epochs = sum(1 for x in ms_counts if x == 0)
    zero_ms_percent = (zero_ms_epochs / len(ms_counts)) * 100

    # Add vertical lines for mean and median
    plt.axvline(x=mean, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(x=median, color='green', linestyle='--',
                linewidth=2, label=f'Median: {median:.2f}')

    # Customize plot
    plt.title('Distribution of Microsaccades per Epoch - Target\n',
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Number of Microsaccades per Epoch', fontsize=14)
    plt.ylabel('Average Number of Epochs per Subject', fontsize=14)

    plt.ylim(-1, 900)

    # Add statistics text box
    stats_text = (f'Subjects: {num_subjects}\n'
                  f'Mean MS per epoch: {mean:.2f}\n'
                  f'Median MS per epoch: {median:.2f}\n'
                  f'Std Dev: {std:.2f}\n'
                  f'Total Epochs: {total_epochs}\n'
                  f'Avg Epochs per Subject: {total_epochs/num_subjects:.1f}')

    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='white',
                       alpha=0.9,
                       edgecolor='gray'))

    # Move legend to a better position (outside the plot)
    plt.legend(bbox_to_anchor=(1.007, 1), loc='upper left',
               fontsize=12, framealpha=0.9)

    # Adjust layout to accommodate the legend
    plt.tight_layout()

    # Save plot with extra space for legend
    plt.savefig(os.path.join(OUTPUT_DIR, 'Microsaccades_Per_Epoch_Distribution_Target.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return mean, median, std, zero_ms_epochs, zero_ms_percent, num_subjects


def main():
    try:
        # Analyze microsaccades per epoch
        logging.info("Starting microsaccade analysis...")
        ms_counts, total_epochs, num_subjects = analyze_microsaccades_per_trial()

        # Create and save histogram
        mean, median, std, zero_ms_epochs, zero_ms_percent, num_subjects = create_microsaccade_histogram(
            ms_counts, total_epochs, num_subjects)

        # Log results
        logging.info(f"Analysis complete. Summary statistics:")
        logging.info(f"Number of subjects: {num_subjects}")
        logging.info(f"Mean microsaccades per epoch: {mean:.2f}")
        logging.info(f"Median microsaccades per epoch: {median:.2f}")
        logging.info(f"Standard deviation: {std:.2f}")
        logging.info(f"Total epochs: {total_epochs}")
        logging.info(
            f"Average epochs per subject: {total_epochs/num_subjects:.1f}")
        logging.info(
            f"Epochs with 0 microsaccades: {zero_ms_epochs} ({zero_ms_percent:.1f}%)")
        logging.info(f"Results saved in {OUTPUT_DIR}")

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
