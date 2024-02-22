"""
===============================================
00_conver_edf_to_asc

this code gets the edf file (raw output from 
eyelink) and outputs the asc file for further
analyses.
               
written by Tara Ghafari
based on:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/31a08b24b2bc5ebd85334c1af16f8b16cb9e3e4f/Experiment%201/Python/EDFConverter.py
===============================================
"""


import os
import subprocess
import warnings


def convert_file(path, name, exePath):
    """
    This function converts an edf file to an ascii file
    :param path:
    :param name:
    :param exePath:
    :return:
    """
    cmd = exePath + os.sep + "edf2asc.exe "
    ascfile = name[:-3] + "asc"

    # check if an asc file already exists
    if not os.path.isfile(path + os.sep + ascfile):
        subprocess.run([cmd, "-p", path, path + os.sep + name])
    else:
        warnings.warn("An Ascii file for " + name + " already exists!")


def main():
    dataDir = r"C:\Users\abdou\Downloads\Blumenfeld Lab\TWCF\Eye Tracking\Data\Exp 2"
    subnames = ["TA272"]
    

    convert_batch(dataDir, subnames, exeDir)


if __name__ == "__main__":
    main()
    

dataDir = r'Z:\Projects\Subcortical_Structures\SubStr_and_behavioral_bias\Data\target_orientation_detection\Results'
subnames = 
exeDir = r"Z:\Programming\Python\Behavioral Asymmetry\eyetracking"