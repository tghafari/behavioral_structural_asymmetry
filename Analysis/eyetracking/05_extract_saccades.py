"""
===============================================
05_extract_saccades

this code gets the output of 04_remove_blinks.

written by Tara Ghafari 
Modified by Mohammad Ebrahim Katebi

adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
==============================================
ToDos:
    
"""

import numpy as np
import pandas as pd
import os.path as op
from scipy import stats
import pickle
import os


class Error(Exception):
    pass


class RepresentationError(Error):
    def __init__(self, msg):
        self.message = msg


def ExtractSaccades(gazeData, params, getBinocular=True):
    """
    This is the main function responsible for extracting saccadees and microsaccades. it calls several other helper
    functions that do some heavy lifting.
    :param gazeData: the gaze data segmented per epoch. this should be a a list of nepochs dataframes with each
    dataframe holding 4 columns: LX, LY, RX, RY. should be the output of remove_blinks (EpochDataNoBlinks)
    :param params: the parameters dictionary. should be the output of initParams
    :param getBinocular: a flag to indicate whether or not to get binocular saccades. if true, the function will ignore
    the 'Eye' field in the parameters dict and will get microsaccades for both eyes, then detect binocluar ones based
    on overlap
    :return: SaccadeInfo: a list of dictionaries, one for each trial. A trial's dictionary holds the following fields:
            - 'Velocity': left, right, both
            - 'Microsaccades': left, right, both
            - 'Saccades': left, right, both
            - 'Threshold': in x and y, the treshold used for classifiying saccades
    """

    # get the parameters relevant for saccade detection
    saccParams = params['SaccadeDetection']

    SaccadeInfo = [None] * len(gazeData)

    # loop through the trials' gaze data
    for epoch in range(0, len(gazeData)):
        # initialize the Saccades list
        saccDict = {'Velocity': {'left': None, 'right': None},
                    'Microsaccades': {'left': None, 'right': None, 'both': None},
                    'Saccades': {'left': None, 'right': None, 'both': None},
                    'Threshold': None}

        # get the trials gaze data and convert it to degrees relative to screen center
        epochGaze = np.array(gazeData[epoch])
        # convert zeros to nans
        epochGaze[epochGaze == 0] = np.nan
        # make it relative to screen center
        epochGaze[:, [0, 2]] = epochGaze[:, [0, 2]] - \
            (params['ScreenResolution'][0] / 2)
        # y data is inverted
        epochGaze[:, [1, 3]] = - \
            (epochGaze[:, [1, 3]] - (params['ScreenResolution'][1] / 2))
        epochGaze = epochGaze * params['cmPerPixel'][0]  # convert to cm
        # convert to degrees
        epochGaze = np.degrees(np.arctan(epochGaze / params['ViewDistance']))

        # if we are not getting binocular microsaccades, get the gaze for the specified eye in parameters
        if not getBinocular:
            if params['Eye'] == 'L':
                eyeGazes = [epochGaze[:, [0, 1]]]
                eyes = ['left']
            elif params['Eye'] == 'R':
                eyeGazes = [epochGaze[:, [2, 3]]]
                eyes = ['right']
        else:
            eyeGazes = [epochGaze[:, [0, 1]], epochGaze[:, [2, 3]]]
            eyes = ['left', 'right']

        # get monocular microsaccades
        for eye in range(0, len(eyes)):  # for each eye
            # gaze data for this eye
            eyeGaze = eyeGazes[eye]
            # get velocity
            velocity, speed = GetVelocity(eyeGaze, params['SamplingFrequency'])
            # get the saccades
            trialsacc, radius = ExtractMonocularMS(
                eyeGaze, velocity, params)[0:2]

            # fill out the saccadeinfo list
            saccDict['Threshold'] = radius
            saccDict['Velocity'][eyes[eye]] = velocity
            saccDict['Saccades'][eyes[eye]] = trialsacc

            # get the indices of the microsaccades (saccades less than threshold in amplitude)
            if trialsacc is not None:
                indMS = trialsacc['total_amplitude'] < saccParams['threshold']
                saccDict['Microsaccades'][eyes[eye]
                                          ] = trialsacc.loc[indMS, :].reset_index()

        # get binocular saccades
        saccLeft = saccDict['Saccades']['left']
        saccRight = saccDict['Saccades']['right']
        microLeft = saccDict['Microsaccades']['left']
        microRight = saccDict['Microsaccades']['right']
        if saccLeft is not None and saccRight is not None:
            # for microsaccades only (saccades less than threshold)
            if not microLeft.empty and not microRight.empty:
                ind_both = []
                for k in range(0, microLeft.shape[0]):
                    # get the maximum overlap between this left ms and all right ms
                    max_intersect = 0
                    for j in range(0, microRight.shape[0]):
                        L = len(np.intersect1d(np.arange(microLeft['start'][k], microLeft['end'][k]),
                                               np.arange(microRight['start'][j], microRight['end'][j])))
                        if L > max_intersect:
                            max_intersect = L

                    # check overlap criteria
                    if max_intersect >= saccParams['msOverlap']:
                        ind_both.append(k)

                # add the binocular microsaccades
                saccDict['Microsaccades']['both'] = microLeft.iloc[ind_both, :]

            # for all saccades
            ind_both = []
            for k in range(0, saccLeft.shape[0]):
                # get the maximum overlap between this left saccade and all right saccades
                max_intersect = 0
                for j in range(0, saccRight.shape[0]):
                    L = len(np.intersect1d(np.arange(saccLeft['start'][k], saccLeft['end'][k]),
                                           np.arange(saccRight['start'][j], saccRight['end'][j])))
                    if L > max_intersect:
                        max_intersect = L

                # check overlap criteria
                if max_intersect >= saccParams['msOverlap']:
                    ind_both.append(k)

            # add the binocular saccades
            saccDict['Saccades']['both'] = saccLeft.iloc[ind_both, :]

        SaccadeInfo[epoch] = saccDict

    return SaccadeInfo


def GetVelocity(eyeGaze, fs):
    """

    :param eyeGaze:
    :param fs:
    :return:
    """

    # initialize outputs
    velocity = np.zeros(eyeGaze.shape)
    speed = np.zeros((velocity.shape[0], 1))

    # loop through the data points and calculate a moving average of velocities over 5
    # data samples
    for n in range(2, eyeGaze.shape[0] - 2):
        velocity[n, :] = (eyeGaze[n + 1, :] + eyeGaze[n + 2, :] -
                          eyeGaze[n - 1, :] - eyeGaze[n - 2, :]) * (fs / 6)

    # calculate speed
    speed[:, 0] = np.sqrt(np.power(velocity[:, 0], 2) +
                          np.power(velocity[:, 1], 2))

    return velocity, speed


def ExtractMonocularMS(eyeGaze, velocity, params, refLoc=None, msdx=None, msdy=None):
    """
    This function extracts microsaccades (and generally all saccades) in one eye.
    This is based on Engbert, R., & Mergenthaler, K. (2006) Microsaccades are triggered by low retinal image slip.
    Proceedings of the National Academy of Sciences of the United States of America, 103: 7192-7197.

    :param eyeGaze:
    :param params:
    :param velocity:
    :param msdx:
    :param msdy:
    :param refLoc:
    :return:
    """

    # saccade extraction parameters
    saccParams = params['SaccadeDetection']

    # get the reference location information if indicated
    if refLoc is not None:
        refCoords = params['StimulusCoords'][refLoc]
        deg2pixRef = params['DegreesPerPix']

    # get the velocity thresholds if they are not given as inputs
    if msdx is None or msdy is None:
        # if only one data point exists, replace entire trial with nans
        if sum(~np.isnan(velocity[:, 0])) == 1 or sum(~np.isnan(velocity[:, 1])) == 1:
            velocity[:, 0] = np.nan
            velocity[:, 1] = np.nan
        elif all(velocity[~np.isnan(velocity[:, 0]), 0] == 0) or all(velocity[~np.isnan(velocity[:, 1]), 1] == 0):
            velocity[:, 0] = np.nan
            velocity[:, 1] = np.nan

        MSDX, MSDY, stddev, maddev = GetVelocityThreshold(velocity)
        if msdx is None:
            msdx = MSDX
        if msdy is None:
            msdy = MSDY

    else:
        _, _, stddev, maddev = GetVelocityThreshold(velocity)

    # begin saccade detection
    radiusx = saccParams['vfac'] * msdx
    radiusy = saccParams['vfac'] * msdy
    radius = np.array([radiusx, radiusy])

    # compute test criterion: ellipse equation
    test = np.power((velocity[:, 0] / radiusx), 2) + \
        np.power((velocity[:, 1] / radiusy), 2)
    indx = np.argwhere(test > 1)

    # determine saccades
    N = len(indx)
    if refLoc is None:
        sac = np.zeros((1, 14))
    else:
        sac = np.zeros((1, 15))
    nsac = 0
    dur = 1
    a = 0
    k = 0
    while k < N - 1:
        if indx[k + 1] - indx[k] == 1:
            dur = dur + 1
        else:
            if dur >= saccParams['mindur']:
                nsac = nsac + 1
                b = k
                if nsac == 1:
                    sac[0][0] = indx[a]
                    sac[0][1] = indx[b]
                else:
                    if refLoc is None:
                        sac = np.vstack(
                            (sac, np.array([indx[a][0], indx[b][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
                    else:
                        sac = np.vstack(
                            (sac, np.array([indx[a][0], indx[b][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
            a = k + 1
            dur = 1
        k = k + 1

    # check duration criterion for the last microsaccade
    if dur >= saccParams['mindur']:
        nsac = nsac + 1
        b = k
        if nsac == 1:
            sac[0][0] = indx[a]
            sac[0][1] = indx[b]
        else:
            if refLoc is None:
                sac = np.vstack(
                    (sac, np.array([indx[a][0], indx[b][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
            else:
                sac = np.vstack(
                    (sac, np.array([indx[a][0], indx[b][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))

    # compute peak velocity, horizontal and vertical components, amplitude, and gaze direction
    if nsac > 0:
        for s in range(0, nsac):
            # Onset and offset
            a = int(sac[s][0])
            b = int(sac[s][1])
            idx = range(a, b)

            # peak velocity
            peakvel = max(
                np.sqrt(velocity[idx, 0] ** 2 + velocity[idx, 1] ** 2))
            sac[s][2] = peakvel

            # horz and vert components
            dx = eyeGaze[b, 0] - eyeGaze[a, 0]
            dy = eyeGaze[b, 1] - eyeGaze[a, 1]
            sac[s][3] = dx
            sac[s][4] = dy

            # amplitude (dX,dY)
            minx = min(eyeGaze[idx, 0])
            maxx = max(eyeGaze[idx, 0])
            miny = min(eyeGaze[idx, 1])
            maxy = max(eyeGaze[idx, 1])
            ix1 = np.argmin(eyeGaze[idx, 0])
            ix2 = np.argmax(eyeGaze[idx, 0])
            iy1 = np.argmin(eyeGaze[idx, 1])
            iy2 = np.argmax(eyeGaze[idx, 1])
            dX = np.sign(ix2 - ix1) * (maxx - minx)
            dY = np.sign(iy2 - iy1) * (maxy - miny)
            sac[s][5] = dX
            sac[s][6] = dY

            # total amplitude
            sac[s][7] = np.sqrt(dX ** 2 + dY ** 2)

            # saccade distance to fixation (screen center)

            gazeOnset = eyeGaze[a, :]
            gazeOffset = eyeGaze[b, :]
            deg2pix = params['DegreesPerPixel']

            # distance to center
            distToFixOnset = np.sqrt((gazeOnset[0]) ** 2 +
                                     (gazeOnset[1]) ** 2)
            distToFixOffset = np.sqrt((gazeOffset[0]) ** 2 +
                                      (gazeOffset[1]) ** 2)
            distToFix = (distToFixOffset - distToFixOnset)
            sac[s][8] = distToFix

            # saccade direction
            rad = np.arccos((gazeOffset[0] - gazeOnset[0]) / np.sqrt((gazeOffset[0] - gazeOnset[0]) ** 2 +
                                                                     (gazeOffset[1] - gazeOnset[1]) ** 2))
            angle = np.degrees(rad)
            if (gazeOffset[1] - gazeOnset[1]) >= 0:
                sac[s][9] = angle
            else:
                sac[s][9] = 360 - angle

            # distance to reference
            if refLoc is not None:
                distToRefOnset = np.sqrt((gazeOnset[0] - refCoords[0] * deg2pixRef) ** 2 +
                                         (gazeOnset[1] - refCoords[1] * deg2pixRef) ** 2)
                distToRefOffset = np.sqrt((gazeOffset[0] - refCoords[0] * deg2pixRef) ** 2 +
                                          (gazeOffset[1] - refCoords[1] * deg2pixRef) ** 2)
                distToRef = (distToRefOffset - distToRefOnset)
                sac[s][10] = distToRef

                sac[s][11] = gazeOnset[0]
                sac[s][12] = gazeOnset[1]
                sac[s][13] = gazeOffset[0]
                sac[s][14] = gazeOffset[1]

                # convert to a dataframe
                sacdf = pd.DataFrame(data=sac,
                                     columns=['start', 'end', 'peak_velocity', 'dx', 'dy', 'x_amplitude', 'y_amplitude',
                                              'total_amplitude', 'distance_to_fixation', 'direction',
                                              'distance_to_reference', 'gazeOnset_x', 'gazeOnset_y', 'gazeOffset_x', 'gazeOffset_y'])
            else:
                sac[s][10] = gazeOnset[0]
                sac[s][11] = gazeOnset[1]
                sac[s][12] = gazeOffset[0]
                sac[s][13] = gazeOffset[1]

                sacdf = pd.DataFrame(data=sac,
                                     columns=['start', 'end', 'peak_velocity', 'dx', 'dy', 'x_amplitude', 'y_amplitude',
                                              'total_amplitude', 'distance_to_fixation', 'direction', 'gazeOnset_x', 'gazeOnset_y', 'gazeOffset_x', 'gazeOffset_y'])

        sacdf['start'] = sacdf['start'].apply(int)
        sacdf['end'] = sacdf['end'].apply(int)

    else:
        sacdf = None

    # return all values that were relevant for detection, if a user function doesn't want all values it can just select
    return sacdf, radius, msdx, msdy, stddev, maddev


def GetVelocityThreshold(velocity):
    """

    :param velocity:
    :return: msdx:
    :return msdy:
    :return stddev:
    :return maddev:
    """

    # compute threshold
    msdx = np.sqrt(np.nanmedian(
        np.power(velocity[:, 0], 2)) - np.power(np.nanmedian(velocity[:, 0]), 2))
    msdy = np.sqrt(np.nanmedian(
        np.power(velocity[:, 1], 2)) - np.power(np.nanmedian(velocity[:, 1]), 2))

    if msdx < np.finfo('float').tiny:  # if less than the smallest usable float
        # switch to a mean estimator instead and see
        msdx = np.sqrt(np.nanmean(
            np.power(velocity[:, 0], 2)) - np.power(np.nanmean(velocity[:, 0]), 2))
        # raise an error if still smaller
        if msdx < np.finfo('float').tiny:
            raise RepresentationError('Calculated velocity threshold (msdx) was smaller than the smallest '
                                      'positive representable floating-point number. Did you exclude blinks/'
                                      'missing data before saccade detection?')

    # do the same for the y-component
    if msdy < np.finfo('float').tiny:  # if less than the smallest usable float
        # switch to a mean estimator instead and see
        msdy = np.sqrt(np.nanmean(
            np.power(velocity[:, 1], 2)) - np.power(np.nanmean(velocity[:, 1]), 2))
        # raise an error if still smaller
        if msdy < np.finfo('float').tiny:
            raise RepresentationError('Calculated velocity threshold (msdy) was smaller than the smallest '
                                      'positive representable floating-point number. Did you exclude blinks/'
                                      'missing data before saccade detection?')

    # compute the standard deviation and the median abs deviation for the velocity values in both components
    stddev = np.nanstd(velocity, axis=0, ddof=1)
    maddev = stats.median_abs_deviation(velocity, axis=0, nan_policy='omit')

    return msdx, msdy, stddev, maddev


data_dir = r"../../Results/EyeTracking/Landmark"
os.makedirs(data_dir, exist_ok=True)

for item in os.listdir(data_dir):
    if item.startswith("sub-"):
        sub_dir = os.path.join(data_dir, item)

        for file in os.listdir(sub_dir):
            if file.endswith("_EL_noblinks_All.pkl"):
                print(f"\nProcessing File: {file}")

                with open(op.join(sub_dir, file), 'rb') as f:
                    blink_data = pickle.load(f)

                with open(op.join(sub_dir, f"{file.removesuffix('_EL_noblinks_All.pkl')}_EL_params.pkl"), 'rb') as f:
                    params = pickle.load(f)

                EpochDataNoBlinks = blink_data[0]
                gazeData = [df[['LX', 'LY', 'RX', 'RY']]
                            for df in EpochDataNoBlinks]

                saccadeinfo = ExtractSaccades(
                    gazeData, params, getBinocular=True)

                Output_file_name = f"{file.removesuffix('_EL_noblinks_All.pkl')}_EL_saccadeinfo.pkl"
                Output_file_path = os.path.join(sub_dir, Output_file_name)

                # Save
                with open(Output_file_path, 'wb') as f:
                    pickle.dump(saccadeinfo, f)
