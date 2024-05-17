# -*- coding: utf-8 -*-
"""
===============================================
02_Initialise_eyetracking_parameters

this code gets the output of parsing info
from asc function to define screen width,
screen height and camera distance and then
calculates parameters for further analysis.

written by Tara Ghafari 
adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
==============================================
ToDos:
"""

import numpy as np
import os.path as op
import pickle
import AnalysisHelpers

def InitParams(eyeData, participantCode, fs, eye):
    """ defines and returns the parameters that will be used for analysis base on the experiment being analyzed
    :param eyeData: output of 01_parsing_info_from_asc that contains messages from eyelink pc
    :param participantName: name of the participant whose data is being analyzed
    :param fs: sampling rate of the eye tracker
    :param eye: either 'L' or 'R' for left or right eye
    :return: params: a dictionary holding the parameters that will be used for analysis
    """

    # initialize params as an empty dictionary
    params = dict([])
    
    screen_width = round(float(eyeData[1].iloc[2].text[-3:])/10)
    screen_height = round(float(eyeData[1].iloc[2].text[-7:-4])/10)
    view_distance = round(float(eyeData[1].iloc[3].text[-3:])/10)
    screen_resolution = np.array([int(eyeData[1].iloc[1].text[-9:-5]),
                                 int(eyeData[1].iloc[1].text[-4:])])
    
    params['ScreenWidth'] = screen_width
    params['ScreenHeight'] = screen_height
    params['ViewDistance'] = view_distance
    params['ParticipantName'] = participantCode
    params['SamplingFrequency'] = fs
    params['Eye'] = eye
    params['ScreenResolution'] = screen_resolution
    params['cmPerPixel'] = np.array([screen_width, screen_height]) / params['ScreenResolution']
    params['ScreenCenter'] = params['ScreenResolution'] / 2
    params['EventTypes'] = [['Right Cue', 'Left Cue'],
                            ['Stim Onset', 'Stim Offset'],
                            ['Trial Onset', 'Response Onset']]
    params['AcceptedShift'] = 2  # max shift from fixation circle that we accept (2 deg on each side)

    # convert visual angles to pixels
    params['FixationWindow'] = AnalysisHelpers.deg2pix(view_distance, params['AcceptedShift'], params['cmPerPixel'])
    params['DegreesPerPixel'] = params['AcceptedShift'] / params['FixationWindow']

    # define the time before and after the stimulus presentation over which the analysis is performed
    # the post stim duration depends on the duration category (short, medium, long) but we first segment the
    # entire trial from the whole data and then segment the trial itself again into 500ms segments
    params['PreCue'] = 0.25  # in sec, ITI jitters between 400 and 600ms
    params['CueDur'] = 0.6  # in sec, cue = 600ms cue to stim ISI jitters between 500 and 700ms
                               # (we have to remove the 600ms while the cue was displaying, 
                               # better to only include blank screen with attention.)
    params['PostCueOffset'] = 0.5  # in sec, the important duration for microsaccades
    # total number of time points for each trial
    params['TrialTimePts'] = (params['PreCue'] + params['CueDur'] + params['PostCueOffset']) * fs

    # parameters relevant for saccade and microsaccade detection
    params['SaccadeDetection'] = {'threshold': 1,  # upper cutoff for microsaccades (in degrees)
                                  'msOverlap': 2,  # number of overlapping points to count as a binocular saccade
                                  'vfac': 5,  # will be multiplied by E&K criterion to get velocity threshold
                                  'mindur': 5,  # minimum duration of a microsaccade (in indices or samples)
                                  }
    return params

      
platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
deriv_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives')


# load in eyeData
for sub_code in range(10,33):
    output_fpath = op.join(deriv_dir, 'target_orientation', 'eyetracking')
    output_dir = op.join(output_fpath,'sub-S' + str(1000+sub_code))
    with open(op.join(output_dir, 'EL_eyeData.json'), 'rb') as f:
        eyeData = pickle.load(f)
    
    fs = 500  # sampling frequency
    eye = 'B'
    
    params = InitParams(eyeData, 'S'+ str(1000+sub_code), fs, eye)
    
    # save params as json file
    with open(op.join(output_dir, 'EL_params.json'), 'wb') as f:
        pickle.dump(params, f)