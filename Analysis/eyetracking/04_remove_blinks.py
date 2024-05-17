# -*- coding: utf-8 -*-
"""
===============================================
04_remove_blinks

this code gets the output of 03_epoch_eye_data
and removes the blinks from it.

written by Tara Ghafari 
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
    
    # use the EpochBoth data from (which includes all the epochs: right and left)
    EpochBoth = epoch_data 
    
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
        epochBlinks = based_noise_blinks_detection(pupilArray, params['SamplingFrequency'])

        # put into the Blinks array
        Blinks[epoch] = pd.DataFrame({'Onset': epochBlinks['blink_onset'], 'Offset': epochBlinks['blink_offset']})

        # loop through the blinks and replace them
        for bOn, bOff in zip(epochBlinks['blink_onset'], epochBlinks['blink_offset']):
            # replace with nans
            EpochDataNoBlinks[epoch].iloc[int(bOn):int(bOff), :] = np.nan

            # add 1s in the blink array where there is a blink
            BlinkArray[int(bOn):int(bOff), epoch] = 1

    return EpochDataNoBlinks, Blinks, BlinkArray.T

platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
deriv_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives')

# load in params and epochs
for sub_code in range(7,33):
    output_fpath = op.join(deriv_dir, 'target_orientation', 'eyetracking')
    output_dir = op.join(output_fpath,'sub-S' + str(1000+sub_code))
    with open(op.join(output_dir, 'EL_params.json'), 'rb') as f:
        params = pickle.load(f)
        
    with open(op.join(output_dir, 'EL_epochs.json'), 'rb') as f:
        epoch_data = pickle.load(f)
    
    
    blink_data_left = RemoveBlinks(epoch_data[1], params)  # epoch_data[2] includes both right and left attention epochs
    blink_data_right = RemoveBlinks(epoch_data[0], params)  # epoch_data[1] includes left attention epochs
                                                            # epoch_data[0] includes right attention epochs
    
    with open(op.join(output_dir, 'EL_noblinks_left.json'), 'wb') as f:
        pickle.dump(blink_data_left, f)
    with open(op.join(output_dir, 'EL_noblinks_right.json'), 'wb') as f:
        pickle.dump(blink_data_right, f)
    
    
    
    