"""
===============================================
03_epoch_eye_data

This code gets the output of 02_initialise_eyetracking_parameters
to segment the full eye tracking data into epochs.

Written by Tara Ghafari 
Modified by Mohammad Ebrahim Katebi

Adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
===============================================
ToDos:
"""

import numpy as np
import pickle
import os.path as op
import os


def sequence_eye_data(params, eye_data):
    """
    Segments the full eye data into trials.

    Args:
    params (dict): The parameters dictionary, output of initParams.
    eye_data (tuple): The data frames with the eye tracker data, output of the asc parser function.

    Returns:
    tuple: (Epoch_All, EpochInfo)
        Epoch_All: A list of dataframes holding the samples for each trial.
        EpochInfo: A dictionary holding trial information.
    """
    data = eye_data[5]

    stim_onset = np.array(
        eye_data[1].time[eye_data[1].text.str.contains(' Shift')])
    stim_direction = np.array(
        eye_data[1].text[eye_data[1].text.str.contains(' Shift')])
    stim_direction = np.array([direction.split(" Shift")[0]
                              for direction in stim_direction])

    epoch_start = stim_onset
    epoch_end = epoch_start + \
        (params['StimDur'] * 1000) + (params['PostStimOffset'] * 1000)

    epoch_all = [None] * len(stim_onset)
    epoch_info = {
        'epoch_start': epoch_start,
        'epoch_end': epoch_end,
        'epoch_Direction': stim_direction
    }

    print(f'Number of epochs = {len(stim_onset)}')

    for epoch in range(len(stim_onset)):
        epoch_st_idx = data.iloc[:, 0] == epoch_start[epoch]
        epoch_end_idx = data.iloc[:, 0] == epoch_end[epoch]

        if not np.sum(epoch_st_idx) or not np.sum(epoch_end_idx):
            epoch_st_idx = data.iloc[:, 0] == epoch_start[epoch] - 1
            epoch_end_idx = data.iloc[:, 0] == epoch_end[epoch] - 1

        epoch_st_idx = epoch_st_idx[epoch_st_idx].index.values[0]
        epoch_end_idx = epoch_end_idx[epoch_end_idx].index.values[0]

        epoch_all[epoch] = data.iloc[epoch_st_idx:epoch_end_idx,
                                     :].reset_index(drop=True)

    return epoch_all, epoch_info


def remove_global_signal_per_trial(epoch_data, eye_data):
    """
    Removes the global signal baseline from the gaze data for each RX, LX, RY, LY channel
    for each trial separately.

    Args:
    epoch_data (tuple): (Epoch_All, EpochInfo) from sequence_eye_data function.
    eye_data (tuple): The original eye data.

    Returns:
    tuple: (Epoch_All, EpochInfo) with global signal removed.
    """
    epoch_all, epoch_info = epoch_data
    channels = ['RX', 'LX', 'RY', 'LY']

    trial_info = eye_data[0]
    trial_starts = trial_info['tStart'].values
    trial_ends = trial_info['tEnd'].values

    global_signals = {i: {channel: np.nan for channel in channels}
                      for i in range(len(trial_starts))}

    for trial_idx, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        trial_epochs_indices = [i for i, epoch in enumerate(
            epoch_all) if start <= epoch.iloc[0, 0] < end]

        for channel in channels:
            trial_channel_data = np.concatenate(
                [epoch_all[i][channel].dropna().values for i in trial_epochs_indices])

            if len(trial_channel_data) > 0:
                global_signal = np.nanmean(trial_channel_data)
                global_signals[trial_idx][channel] = global_signal

                for i in trial_epochs_indices:
                    epoch_all[i][channel] = epoch_all[i][channel].subtract(
                        global_signal, fill_value=np.nan)
            else:
                print(f"Warning: All data for channel {channel} in trial {trial_idx} is NaN. "
                      f"Skipping baseline removal for this channel in this trial.")

    epoch_info['Global_Signals_Per_Trial'] = global_signals

    return epoch_all, epoch_info


def adjust_to_screen_center(epoch_all, params):
    """
    Adjusts the gaze coordinates to be centered on the screen.

    Args:
    epoch_all (list): List of epoch dataframes.
    params (dict): Parameters dictionary.

    Returns:
    list: Adjusted epoch_all.
    """
    screen_center_x = params['ScreenResolution'][0] / 2
    screen_center_y = params['ScreenResolution'][1] / 2

    for i, epoch in enumerate(epoch_all):
        epoch_all[i]['RX'] = epoch['RX'] + screen_center_x
        epoch_all[i]['LX'] = epoch['LX'] + screen_center_x
        epoch_all[i]['RY'] = epoch['RY'] + screen_center_y
        epoch_all[i]['LY'] = epoch['LY'] + screen_center_y

    return epoch_all


def process_eye_tracking_data(file_path, params_path):
    """
    Processes eye tracking data from a file, removes global signal per trial,
    and adjusts coordinates to screen center.

    Args:
    file_path (str): Path to the eye data file.
    params_path (str): Path to the parameters file.

    Returns:
    tuple: (Epoch_All, EpochInfo) processed data.
    """
    with open(file_path, 'rb') as f:
        eye_data = pickle.load(f)

    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    epoch_data = sequence_eye_data(params, eye_data)
    epoch_all, epoch_info = remove_global_signal_per_trial(
        epoch_data, eye_data)
    epoch_all = adjust_to_screen_center(epoch_all, params)

    return epoch_all, epoch_info


def main():
    data_dir = r"../../Results/EyeTracking"
    os.makedirs(data_dir, exist_ok=True)

    for item in os.listdir(data_dir):
        if item.startswith("sub-"):
            sub_dir = os.path.join(data_dir, item)

            for file in os.listdir(sub_dir):
                if file.endswith("_EL_eyeData.pkl"):
                    print(f"\nProcessing File: {file}")

                    file_path = op.join(sub_dir, file)
                    params_path = op.join(
                        sub_dir, f"{file.removesuffix('_EL_eyeData.pkl')}_EL_params.pkl")

                    epoch_all, epoch_info = process_eye_tracking_data(
                        file_path, params_path)

                    output_file_name = f"{file.removesuffix('_EL_eyeData.pkl')}_EL_epochs.pkl"
                    output_file_path = os.path.join(sub_dir, output_file_name)

                    with open(output_file_path, 'wb') as f:
                        pickle.dump((epoch_all, epoch_info), f)

                    print(f"Processed data saved to: {output_file_path}")


if __name__ == "__main__":
    main()
