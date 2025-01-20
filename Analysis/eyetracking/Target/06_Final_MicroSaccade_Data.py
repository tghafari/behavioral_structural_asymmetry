import os
import os.path as op
import pandas as pd
import pickle
import numpy as np
import logging
import re

# Set up logging
logging.basicConfig(
    filename=r"../../../Results/EyeTracking/Target/Eyetracking_Analysis.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define constants
DATA_DIR = r"../../../Results/EyeTracking/Target"
BEH_DATA_DIR = r"E:/Target_Data"
OUTPUT_FOLDER_PATH = DATA_DIR
TIME_WINDOW = 100  # in milliseconds

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
logging.info(f"Output folder: {OUTPUT_FOLDER_PATH}")


def load_behavioral_data(subject_id):
    """Load behavioral data for a given subject."""
    beh_file_pattern = f"sub-{subject_id}/ses-01/beh/sub-{subject_id}_ses-01_task-Landmark_run-01_logfile.csv"
    beh_file_path = os.path.join(BEH_DATA_DIR, beh_file_pattern)

    try:
        beh_data = pd.read_csv(beh_file_path)
        logging.info(f"Loaded behavioral data for subject {subject_id}")
        return beh_data
    except FileNotFoundError:
        logging.error(
            f"Behavioral data file not found for subject {subject_id}")
        return None


def process_eyetracking_data(sub_dir, file, beh_data):
    """Process eyetracking data and merge with behavioral data."""
    logging.info(f"Processing File: {file}")

    # Load saccade and blink data
    with open(op.join(sub_dir, file), 'rb') as f:
        saccadeinfo = pickle.load(f)
    with open(op.join(sub_dir, f"{file.removesuffix('_EL_saccadeinfo.pkl')}_EL_noblinks_All.pkl"), 'rb') as f:
        blink_data = pickle.load(f)

    clean_data = []

    for epoch, epoch_saccadeinfo in enumerate(saccadeinfo):
        epoch_microsaccadeinfo = epoch_saccadeinfo['Microsaccades']

        if epoch_microsaccadeinfo['both'] is not None:
            epoch_microsaccadeinfo_both = pd.DataFrame(
                epoch_microsaccadeinfo['both']).sort_values('start').reset_index(drop=True)
            last_included_time = -np.inf

            for _, microsaccade in epoch_microsaccadeinfo_both.iterrows():
                start_time = microsaccade['start']
                end_time = microsaccade['end']

                if start_time >= last_included_time + TIME_WINDOW:
                    epoch_duration = blink_data[0][epoch].shape[0]
                    if start_time + TIME_WINDOW <= epoch_duration:
                        # Get behavioral data for this epoch
                        beh_epoch_data = beh_data[beh_data['ID'] == epoch +
                                                  1].iloc[0] if beh_data is not None else None

                        # Combine eyetracking and behavioral data
                        combined_data = [
                            epoch + 1,
                            microsaccade['direction'],
                            microsaccade['total_amplitude'],
                            microsaccade['distance_to_fixation'],
                            blink_data[3]['epoch_Direction'][epoch],
                            int(blink_data[3]['epoch_Exclusion'][epoch]),
                            microsaccade['gazeOnset_x'],
                            microsaccade['gazeOnset_y'],
                            microsaccade['gazeOffset_x'],
                            microsaccade['gazeOffset_y'],
                            start_time,
                            end_time]

                        clean_data.append(combined_data)
                        last_included_time = start_time

    # Create DataFrame and save to CSV
    columns = [
        'Epoch', 'Direction', 'Total_Amplitude', 'Distance_To_Fixation', 'Cue_Direction',
        'Epoch_Exclusion', 'GazeOnset_x', 'GazeOnset_y', 'GazeOffset_x', 'GazeOffset_y',
        'Start_Time', 'End_Time'
    ]
    clean_data_df = pd.DataFrame(clean_data, columns=columns)

    output_file_name = f"{file.removesuffix('_EL_saccadeinfo.pkl')}_Final_MicroSaccade_Data.csv"
    output_file_path = os.path.join(sub_dir, output_file_name)
    clean_data_df.to_csv(output_file_path, index=False)
    logging.info(f"Saved processed data to {output_file_path}")


def main():
    for item in os.listdir(DATA_DIR):
        if item.startswith("sub-"):
            sub_dir = os.path.join(DATA_DIR, item)

            # Extract subject ID
            subject_id_match = re.search(r'S\d+', item)

            if subject_id_match:
                subject_id = subject_id_match.group()
                beh_data = load_behavioral_data(subject_id)
            else:
                logging.warning(
                    f"Could not extract subject ID from folder name: {item}")
                beh_data = None

            for file in os.listdir(sub_dir):
                if file.endswith("_EL_saccadeinfo.pkl"):
                    try:
                        process_eyetracking_data(sub_dir, file, beh_data)
                    except Exception as e:
                        logging.error(
                            f"Error processing file {file}: {str(e)}")


if __name__ == "__main__":
    main()
