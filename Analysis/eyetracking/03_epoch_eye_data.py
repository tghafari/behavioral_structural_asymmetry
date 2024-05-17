# -*- coding: utf-8 -*-
"""
===============================================
03_epoch_eye_data

this code gets the output of 02_initialise_eyetracking_parameters
to segmentthe full eye tracking data into epochs.

written by Tara Ghafari 
adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
==============================================
ToDos:
"""
import numpy as np
import pickle
import os.path as op
        
def SequenceEyeData(params, eyeData):
    """
    segments the full eye data into trials. so at each trial timestamp, we grab the data from some pre-onset until
    sometime post-onset to get a window of data that associate with that trial
    :param params: the parameters dictionary. should be the output of initParams
    :param eyeDFs: the data frames with the eye tracker data. should be the ouptut of the asc parser function
    :return: TrialData: a list of dataframes holding the samples for each trial
    :return: TrialInfo: an updated version of the timestamps dataframe; updated to hold the TrialWindowStart and
    TrialWindowEnd columns
    """

    # get the dataframe with the data/samples
    data = eyeData[5]

    # get the trial window start and end for each trial
    trialOnset = np.array(eyeData[1].time[eyeData[1].text == 'Trial Onset'])
    right_cue_onset = np.array(eyeData[1].time[eyeData[1].text == 'Right Cue'])
    left_cue_onset = np.array(eyeData[1].time[eyeData[1].text == 'Left Cue'])
    # Combine the time points into a single variable
    cue_onset = np.concatenate((right_cue_onset, left_cue_onset))
    cue_onset = np.sort(cue_onset)
    right_epoch_start = right_cue_onset + params['CueDur'] * 1000
    left_epoch_start = left_cue_onset + params['CueDur'] * 1000 
    right_epoch_end = right_epoch_start + params['PostCueOffset'] * 1000
    left_epoch_end = left_epoch_start + params['PostCueOffset'] * 1000
    epoch_start = cue_onset + params['CueDur'] * 1000
    epoch_end = epoch_start + params['PostCueOffset'] * 1000

    # initialize outputs
    EpochRight = [None] * len(right_cue_onset)
    EpochLeft = [None] * len(left_cue_onset)
    EpochBoth = [None] * len(trialOnset)
    EpochInfo = dict([])

    
    EpochInfo['right_epoch_start'] = right_epoch_start
    EpochInfo['left_epoch_start'] = left_epoch_start
    EpochInfo['right_epoch_end'] = right_epoch_end
    EpochInfo['left_epoch_end'] = left_epoch_end
    EpochInfo['epoch_start'] = epoch_start
    EpochInfo['epoch_end'] = epoch_end

    print('num right and left epochs = ' + str(len(trialOnset)))

    # loop through epochs
    for epoch in range(0, len(right_epoch_start)):

        # get the indices for the start and end with which to index the data array
        right_stIdx = data.iloc[:, 0] == right_epoch_start[epoch]
        left_endIdx = data.iloc[:, 0] == left_epoch_end[epoch]
        left_stIdx = data.iloc[:, 0] == left_epoch_start[epoch]
        right_endIdx = data.iloc[:, 0] == right_epoch_end[epoch]
        
        if (not np.sum(np.array(right_stIdx)) or not np.sum(np.array(right_endIdx))):
            right_stIdx = data.iloc[:, 0] == right_epoch_start[epoch] -1 
            right_endIdx = data.iloc[:, 0] == right_epoch_end[epoch] - 1
            
        if (not np.sum(np.array(left_stIdx)) or not np.sum(np.array(left_endIdx))):            
            left_stIdx = data.iloc[:, 0] == left_epoch_start[epoch] - 1        
            left_endIdx = data.iloc[:, 0] == left_epoch_end[epoch] - 1
       
        # convert logical indices to numeric indices
        right_stIdx = right_stIdx[right_stIdx].index.values[0]
        right_endIdx = right_endIdx[right_endIdx].index.values[0]
        EpochRight[epoch] = data.iloc[right_stIdx:right_endIdx, :]
        EpochRight[epoch].reset_index(drop=True, inplace=True)
        left_stIdx = left_stIdx[left_stIdx].index.values[0]
        left_endIdx = left_endIdx[left_endIdx].index.values[0]        
        EpochLeft[epoch] = data.iloc[left_stIdx:left_endIdx, :]
        EpochLeft[epoch].reset_index(drop=True, inplace=True)
        
        if (not np.sum(np.array(right_stIdx)) or not np.sum(np.array(right_endIdx)))\
            or (not np.sum(np.array(left_stIdx)) or not np.sum(np.array(left_endIdx))):
            print('The indice of trial start and end could not be found for trial %d,'
                              ' check the raw data' % epoch)
   
    # loop for all the cues (right and left concatenated)  -- needs to be fixed for concat epochs     
    for epoch in range(0, len(trialOnset)):
        # get the indices for the start and end with which to index the data array
        epoch_stIdx = data.iloc[:, 0] == epoch_start[epoch]
        epoch_endIdx = data.iloc[:, 0] == epoch_end[epoch]
        
        if (not np.sum(np.array(epoch_stIdx)) or not np.sum(np.array(epoch_endIdx))):
            epoch_stIdx = data.iloc[:, 0] == epoch_start[epoch] -1 
            epoch_endIdx = data.iloc[:, 0] == epoch_end[epoch] - 1
         
            # convert logical indices to numeric indices
        epoch_stIdx = epoch_stIdx[epoch_stIdx].index.values[0]
        epoch_endIdx = epoch_endIdx[epoch_endIdx].index.values[0]
        EpochBoth[epoch] = data.iloc[epoch_stIdx:epoch_endIdx, :]
        EpochBoth[epoch].reset_index(drop=True, inplace=True)   
            
    return EpochRight, EpochLeft, EpochBoth, EpochInfo

      
platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
deriv_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives')

# load in eyeData and params
for sub_code in range(7,33):
    output_fpath = op.join(deriv_dir, 'target_orientation', 'eyetracking')
    output_dir = op.join(output_fpath,'sub-S' + str(1000+sub_code))
    with open(op.join(output_dir, 'EL_eyeData.json'), 'rb') as f:
        eyeData = pickle.load(f)
        
    with open(op.join(output_dir, 'EL_params.json'), 'rb') as f:
        params = pickle.load(f)
        
    epoch_data = SequenceEyeData(params, eyeData)
    
    # save epoch_data as json file
    with open(op.join(output_dir, 'EL_epochs.json'), 'wb') as f:
        pickle.dump(epoch_data, f)
        
    