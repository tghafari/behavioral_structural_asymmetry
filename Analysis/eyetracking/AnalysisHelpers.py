# -*- coding: utf-8 -*-
"""
===============================================
AnalysisHelpers

this code contains functions that are 
helpful for further analysis.

written by Tara Ghafari
adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python_new/AnalysisHelpers.py

===============================================
"""

import numpy as np
import math


def deg2pix(viewDistance, degrees, cmPerPixel):
    """
    converts degrees to pixels
    :param viewDistance: viewer distance from the display screen
    :param degrees: degrees visual angle to be converted to no. of pixels
    :param cmPerPixel: the size of one pixel in centimeters
    :return: pixels: the number of pixels corresponding to the degrees visual angle specified
    """

    # get the size of the visual field
    centimeters = math.tan(math.radians(degrees) / 2) * (2 * viewDistance)

    # now convert the centimeters to pixels
    pixels = round(centimeters / cmPerPixel[0])

    return pixels


def CalcFixationDensity(gaze, scale, screenDims):
    """
    This function divides the screen into bins and sums the time during which a gaze was present at each bin
    :param gaze: a tuple where the first element is gazeX and the second is gazeY. gazeX and gazeY are both NxD matrices
                where N is ntrials and D is number of timepoints
    :param scale:
    :param screenDims:
    :return:
    """
    # make sure inputs are arrays
    gazeX = np.array(gaze[0]).flatten()
    gazeY = np.array(gaze[1]).flatten()

    # initialize the fixation density matrix
    fixDensity = np.zeros((int(np.ceil(screenDims[1] / scale)), int(np.ceil(screenDims[0] / scale))))

    # loop through the bins
    L = len(gazeX)
    for i in range(0, fixDensity.shape[1]):
        for j in range(0, fixDensity.shape[0]):
            fixDensity[j, i] = np.sum(((gazeX >= scale * i) & (gazeX <= scale * (i + 1))) &
                                      ((gazeY >= scale * j) & (gazeY <= scale * (j + 1)))) / L

    
    return fixDensity

