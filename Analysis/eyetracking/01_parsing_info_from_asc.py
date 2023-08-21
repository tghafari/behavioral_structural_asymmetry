# -*- coding: utf-8 -*-
"""
===============================================
01_parsing_info_from_asc

this code gets the asc file from the eyetracker 
and detects microsaccades.
edf files (out put of eyelink) should be
converted to .asc using 'visualEDF2ASC' app

written by Tara Ghafari 
adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
==============================================
ToDos:
"""

import pandas as pd
import numpy as np
import os
import os.path as op
import time
import pickle

def ParseEyeLinkAsc(eyetracking_asc_file):
    # dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)
    # -Reads in data files from EyeLink .asc file and produces readable dataframes for further analysis.
    #
    # INPUTS:
    # -elFilename is a string indicating an EyeLink data file 
    #
    # OUTPUTS:
    # -dfRec contains information about recording periods (often trials)
    # -dfMsg contains information about messages (usually sent from stimulus software)
    # -dfFix contains information about fixations
    # -dfSacc contains information about saccades
    # -dfBlink contains information about blinks
    # -dfSamples contains information about individual samples

    # Read in EyeLink file
    t = time.time()
    f = open(eyetracking_asc_file, 'r')
    fileTxt0 = f.read().splitlines(True)  # split into lines
    fileTxt0 = list(filter(None, fileTxt0))  # remove emptys
    fileTxt0 = np.array(fileTxt0)  # concert to np array for simpler indexing
    f.close()
    print('Done! Took %f seconds.' % (time.time() - t))
    
    # Separate lines into samples and messages
    print('Sorting lines...')
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER'] * nLines, dtype='object')
    iStartRec = list([])
    t = time.time()
    for iLine in range(nLines):
        if fileTxt0[iLine] == "**\n" or fileTxt0[iLine] == "\n":
            lineType[iLine] = 'EMPTY'
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = 'COMMENT'
        elif bool(len(fileTxt0[iLine][0])) and fileTxt0[iLine][0].isdigit():
            fileTxt0[iLine] = fileTxt0[iLine].replace('.\t', 'NaN\t')  # fileTxt0[iLine] = fileTxt0[iLine].replace(' . ', ' NaN ')
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if '!MODE RECORD' in fileTxt0[iLine]:  
            iStartRec.append(iLine + 1)    
    
    iStartRec = iStartRec[0]
    print('Done! Took %f seconds.' % (time.time() - t))
    
    # ===== PARSE EYELINK FILE ===== #
    t = time.time()
    # Trials
    print('Parsing recording markers...')
    iNotStart = np.nonzero(lineType != 'START')[0]
    dfRecStart = pd.read_csv(eyetracking_asc_file, skiprows=iNotStart, header=None, delim_whitespace=True, usecols=[1])
    dfRecStart.columns = ['tStart']
    iNotEnd = np.nonzero(lineType != 'END')[0]
    dfRecEnd = pd.read_csv(eyetracking_asc_file, skiprows=iNotEnd, header=None, delim_whitespace=True, usecols=[1, 5, 6])
    dfRecEnd.columns = ['tEnd', 'xRes', 'yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart, dfRecEnd], axis=1)
    nRec = dfRec.shape[0]
    print('%d recording periods found.' % nRec)
    
    # Import Messages
    print('Parsing stimulus messages...')
    t = time.time()
    iMsg = np.nonzero(lineType == 'MSG')[0]
    # set up
    tMsg = []
    txtMsg = []
    t = time.time()
    for i in range(len(iMsg)):
        # separate MSG prefix and timestamp from rest of message
        info = fileTxt0[iMsg[i]].split()
        # extract info
        tMsg.append(int(info[1]))
        txtMsg.append(' '.join(info[2:]))
    # Convert dict to dataframe
    dfMsg = pd.DataFrame({'time': tMsg, 'text': txtMsg})
    print('Done! Took %f seconds.' % (time.time() - t))
    
    # Import Fixations
    print('Parsing fixations...')
    t = time.time()
    iNotEfix = np.nonzero(lineType != 'EFIX')[0]
    dfFix = pd.read_csv(eyetracking_asc_file, skiprows=iNotEfix, header=None, delim_whitespace=True, usecols=range(1, 8))
    dfFix.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg']
    nFix = dfFix.shape[0]
    print('Done! Took %f seconds.' % (time.time() - t))
    
    # Saccades
    print('Parsing saccades...')
    t = time.time()
    iNotEsacc = np.nonzero(lineType != 'ESACC')[0]
    dfSacc = pd.read_csv(eyetracking_asc_file, skiprows=iNotEsacc, header=None, delim_whitespace=True, usecols=range(1, 11))
    dfSacc.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']
    print('Done! Took %f seconds.' % (time.time() - t))
    
    # Blinks
    print('Parsing blinks...')
    iNotEblink = np.nonzero(lineType != 'EBLINK')[0]
    dfBlink = pd.read_csv(eyetracking_asc_file, skiprows=iNotEblink, header=None, delim_whitespace=True, usecols=range(1, 5))
    dfBlink.columns = ['eye', 'tStart', 'tEnd', 'duration']
    print('Done! Took %f seconds.' % (time.time() - t))
 
    # determine sample columns based on eyes recorded in file
    eyesInFile = np.unique(dfFix.eye)
    if eyesInFile.size == 2:
        print('binocular data detected.')
        cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyesInFile[0]
        print('monocular data detected (%c eye).' % eye)
        cols = ['tSample', '%cX' % eye, '%cY' % eye, '%cPupil' % eye]
    # Import samples
    print('Parsing samples...')
    t = time.time()
    iNotSample = np.nonzero(np.logical_or(lineType != 'SAMPLE', np.arange(nLines) < iStartRec))[0]
    dfSamples = pd.read_csv(eyetracking_asc_file, skiprows=iNotSample, header=None, delim_whitespace=True,
                            usecols=range(0, len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyesInFile:
            dfSamples['%cX' % eye] = pd.to_numeric(dfSamples['%cX' % eye], errors='coerce')
            dfSamples['%cY' % eye] = pd.to_numeric(dfSamples['%cY' % eye], errors='coerce')
            dfSamples['%cPupil' % eye] = pd.to_numeric(dfSamples['%cPupil' % eye], errors='coerce')
        else:
            dfSamples['%cX' % eye] = np.nan
            dfSamples['%cY' % eye] = np.nan
            dfSamples['%cPupil' % eye] = np.nan

    print('Done! Took %.1f seconds.' % (time.time() - t))

    # Return new compilation dataframe
    return dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples
       

# Define file names
sub_code = '105'
base_dir = r'Z:\Projects\Subcortical_Structures\SubStr_and_behavioral_bias'
task_dir = r'Data\target_orientation_detection\Results'
sub_dir = op.join('sub-S' + sub_code, r'ses-01\beh')
eyetracking_fpath = op.join(base_dir, task_dir, sub_dir, 'e01S' + sub_code) 
eyetracking_asc_file = eyetracking_fpath + '.asc'

    
eyeData = ParseEyeLinkAsc(eyetracking_asc_file)

# save eyeData as json file
output_fpath = r'Z:\Projects\Subcortical_Structures\SubStr_and_behavioral_bias\Analysis\target_orientation\eyetracking'
output_dir = op.join(output_fpath,'sub-S' + sub_code)
if not op.exists(output_dir):
   os.makedirs(output_dir)
               
with open(op.join(output_dir, 'EL_eyeData.json'), 'wb') as f:
    pickle.dump(eyeData, f)
    
    