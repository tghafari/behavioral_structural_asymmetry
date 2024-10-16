"""
===============================================
02_Initialise_eyetracking_parameters

This code gets the output of parsing info
from asc function to define screen width,
screen height and camera distance and then
calculates parameters for further analysis.

Written by Tara Ghafari 
Adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
===============================================
ToDos:
"""

import os
import os.path as op
import pickle
import numpy as np
import AnalysisHelpers


def init_params(eye_data, participant_code, fs, eye):
    """
    Defines and returns the parameters that will be used for analysis based on the experiment being analyzed.

    Args:
    eye_data (tuple): Output of 01_parsing_info_from_asc that contains messages from eyelink pc
    participant_code (str): Name of the participant whose data is being analyzed
    fs (int): Sampling rate of the eye tracker
    eye (str): Either 'L' or 'R' for left or right eye

    Returns:
    dict: A dictionary holding the parameters that will be used for analysis
    """
    params = {}

    screen_width = round(float(eye_data[1].iloc[2].text[-3:]) / 10)
    screen_height = round(float(eye_data[1].iloc[2].text[-7:-4]) / 10)
    view_distance = round(float(eye_data[1].iloc[3].text[-3:]) / 10)
    screen_resolution = np.array([int(eye_data[1].iloc[1].text[-9:-5]),
                                  int(eye_data[1].iloc[1].text[-4:])])

    params['ScreenWidth'] = screen_width
    params['ScreenHeight'] = screen_height
    params['ViewDistance'] = view_distance
    params['ParticipantName'] = participant_code
    params['SamplingFrequency'] = fs
    params['Eye'] = eye
    params['ScreenResolution'] = screen_resolution
    params['cmPerPixel'] = np.array(
        [screen_width, screen_height]) / params['ScreenResolution']
    params['ScreenCenter'] = params['ScreenResolution'] / 2

    params['EventTypes'] = [['Right Shift', 'Left Shift'],
                            ['Right Response:', 'Left Response', 'Neutral Response']]

    # max shift from fixation circle that we accept (2 deg on each side)
    params['AcceptedShift'] = 2

    # convert visual angles to pixels
    params['FixationWindow'] = AnalysisHelpers.deg2pix(
        view_distance, params['AcceptedShift'], params['cmPerPixel'])
    params['DegreesPerPixel'] = params['AcceptedShift'] / \
        params['FixationWindow']

    # define the time before and after the stimulus presentation over which the analysis is performed
    params['PreStim'] = 1  # in sec, ITI jitters between 1000 and 2000ms
    params['StimDur'] = 0.2
    # in sec, the important duration for microsaccades
    params['PostStimOffset'] = 1.5

    # total number of time points for each trial
    params['TrialTimePts'] = int(
        (params['PreStim'] + params['StimDur'] + params['PostStimOffset']) * fs)

    # parameters relevant for saccade and microsaccade detection
    params['SaccadeDetection'] = {
        'threshold': 1,  # upper cutoff for microsaccades (in degrees)
        'msOverlap': 2,  # number of overlapping points to count as a binocular saccade
        'vfac': 5,  # will be multiplied by E&K criterion to get velocity threshold
        # minimum duration of a microsaccade (in indices or samples)
        'mindur': 3,
    }

    return params


def main():
    data_dir = r"../../Results/EyeTracking"
    output_folder_path = data_dir

    os.makedirs(output_folder_path, exist_ok=True)

    for item in os.listdir(data_dir):
        if item.startswith("sub-"):
            sub_dir = os.path.join(data_dir, item)

            for file in os.listdir(sub_dir):
                if file.endswith("_EL_eyeData.pkl"):
                    print(f"\nProcessing File: {file}")

                    with open(op.join(sub_dir, file), 'rb') as f:
                        eye_data = pickle.load(f)

                    fs = 500  # sampling frequency
                    eye = 'B'

                    participant_code = file.rsplit('_EL_eyeData.pkl', 1)[0]
                    params = init_params(eye_data, participant_code, fs, eye)

                    output_file_name = f"{participant_code}_EL_params.pkl"
                    output_file_path = os.path.join(sub_dir, output_file_name)

                    # save params as pickle file
                    with open(output_file_path, 'wb') as f:
                        pickle.dump(params, f)


if __name__ == "__main__":
    main()
