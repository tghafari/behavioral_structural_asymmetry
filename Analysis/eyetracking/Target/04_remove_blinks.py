"""
===============================================
04_remove_blinks

this code gets the output of 03_epoch_eye_data
and removes the blinks from it.

written by Tara Ghafari 
Modified by Mohammad Ebrahim Katebi

adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
==============================================
ToDos:
    the way this is currently implemented, if a blink starts in a trial/segment and ends in another, it's
    counted twice, this needs to be solved
"""

import copy
import pickle
import os.path as op
import numpy as np
import pandas as pd
from based_noise_blinks_detection import based_noise_blinks_detection
import os

Invalid_Data_Around_Blinks_Padding = 20  # In Samples
Fixation_Criteria_Threshold = 0.1  # Samples Out of Fixation
Fixation_Circle = 2  # In Degrees


def Fixation_Criteria(Blink_Data, params):

    Fixation_Criteria_Blink_Data = copy.deepcopy(Blink_Data)

    GazeBoth = [epoch[['LX', 'LY', 'RX', 'RY']]
                for epoch in Fixation_Criteria_Blink_Data[0]]

    # initialize the Exclusion array
    Fixation_Criteria_Blink_Data[3]["epoch_Exclusion"] = np.ones(
        (len(GazeBoth), 1))

    # loop through the trials
    for epoch in range(0, (len(GazeBoth))):

        # get the trials gaze data and convert it to degrees relative to screen center
        epochGaze = np.array(GazeBoth[epoch])

        # make it relative to screen center
        epochGaze[:, [0, 2]] = epochGaze[:, [0, 2]] - \
            (params['ScreenResolution'][0] / 2)

        # CHANGE: Updated y-axis inversion
        epochGaze[:, [1, 3]] = - \
            (epochGaze[:, [1, 3]] - (params['ScreenResolution'][1] / 2))

        epochGaze = epochGaze * params['cmPerPixel'][0]  # convert to cm
        # convert to degrees
        epochGaze = np.degrees(np.arctan(epochGaze / params['ViewDistance']))

        L_distToFix = np.sqrt(epochGaze[:, 0] ** 2 + epochGaze[:, 1] ** 2)
        R_distToFix = np.sqrt(epochGaze[:, 2] ** 2 + epochGaze[:, 3] ** 2)

        distToFix = np.vstack((L_distToFix, R_distToFix)).T

        Samples_Out_Of_Fixation = np.any(
            distToFix > Fixation_Circle, axis=1)
        Proportion_Samples_Out_Of_Fixation = np.sum(
            Samples_Out_Of_Fixation) / distToFix.shape[0]

        if (Proportion_Samples_Out_Of_Fixation <= Fixation_Criteria_Threshold):

            Fixation_Criteria_Blink_Data[3]["epoch_Exclusion"][epoch] = 0

    return Fixation_Criteria_Blink_Data


def RemoveBlinks(epoch_data, params):
    """
    this function takes in the sequenced TrialData and for each trial, it detects and removes blinks
    the blink detection script and algorithm used here are based on Hershman et al. 2018 (https://osf.io/jyz43/)
    :param epoch_data: a list of dataframes wihere each element corresponds to the data frame of a single trial. this
    should be the output of the sequenceEyeData
    :param params: the parameters dictionary. should be the output of initParams
    :return: TrialDataNoBlinks: an updated version of trial data where the blinks have been removed
    :return: Blinks: a list of dataframes where each entry corresponds to a trial and holds the information about
    the starting time/index and ending time/index of each blink. the number of rows in a dataframe is the number of
    blinks in that trial.
    :return BlinkArray: an array with rows of zeros and ones with ones at the indices of blinks. Each row in
    the array is a trial and the number of columns is the total time points or number of samples associated with a
    trial.
    """

    EpochBoth = epoch_data[0]

    # initialize the no blinks epoch data
    EpochDataNoBlinks = copy.deepcopy(EpochBoth)

    # initialize the list of blinks
    Blinks = [None] * (len(EpochBoth))

    # initialize the blink array
    BlinkArray = np.zeros([EpochBoth[0].shape[0], (len(EpochBoth))])

    # loop through the trials
    for epoch in range(0, (len(EpochBoth))):

        # get the pupil size array
        pupilArray = np.array((EpochDataNoBlinks[epoch].LPupil +
                               EpochDataNoBlinks[epoch].RPupil)/2)

        # replace nan values with zeros
        pupilArray[np.isnan(pupilArray)] = 0

        # detect blinks
        epochBlinks = based_noise_blinks_detection(
            pupilArray, params['SamplingFrequency'])

        # put into the Blinks array
        Blinks[epoch] = pd.DataFrame(
            {'Onset': epochBlinks['blink_onset'], 'Offset': epochBlinks['blink_offset']})

        # loop through the blinks and replace them
        for bOn, bOff in zip(epochBlinks['blink_onset'], epochBlinks['blink_offset']):

            bOn = bOn - 1

            bOn_Padded = np.max([bOn - Invalid_Data_Around_Blinks_Padding,
                                int(EpochDataNoBlinks[epoch].tSample.keys()[0])])
            bOff_Padded = np.min([bOff + Invalid_Data_Around_Blinks_Padding,
                                 int(EpochDataNoBlinks[epoch].tSample.keys()[-1]) + 1])

            # replace with nans
            EpochDataNoBlinks[epoch].iloc[int(
                bOn_Padded):int(bOff_Padded), :] = np.nan

            # add 1s in the blink array where there is a blink
            BlinkArray[int(bOn):int(bOff), epoch] = 1

    return EpochDataNoBlinks, Blinks, BlinkArray.T, epoch_data[1]


data_dir = r"../../../Results/EyeTracking/Target"
os.makedirs(data_dir, exist_ok=True)

for item in os.listdir(data_dir):
    if item.startswith("sub-"):
        sub_dir = os.path.join(data_dir, item)

        for file in os.listdir(sub_dir):
            if file.endswith("_EL_epochs.pkl"):
                print(f"\nProcessing File: {file}")

                with open(op.join(sub_dir, file), 'rb') as f:
                    epoch_data = pickle.load(f)

                with open(op.join(sub_dir, f"{file.removesuffix('_EL_epochs.pkl')}_EL_params.pkl"), 'rb') as f:
                    params = pickle.load(f)

                Blink_Data = RemoveBlinks(epoch_data, params)

                Fixation_Criteria_Blink_Data = Fixation_Criteria(
                    Blink_Data, params)

                Output_file_name = f"{file.removesuffix('_EL_epochs.pkl')}_EL_noblinks_All.pkl"
                Output_file_path = os.path.join(sub_dir, Output_file_name)

                # save epoch_data as json file
                with open(Output_file_path, 'wb') as f:
                    pickle.dump(Fixation_Criteria_Blink_Data, f)
