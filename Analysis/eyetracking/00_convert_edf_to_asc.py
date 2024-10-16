"""
00_convert_edf_to_asc

This code converts EDF files (raw output from EyeLink) to ASC files for further analysis.

Written by Tara Ghafari
Modified by Mohammad Ebrahim Katebi
Based on:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/31a08b24b2bc5ebd85334c1af16f8b16cb9e3e4f/Experiment%201/Python/EDFConverter.py
"""

import os
import subprocess
import warnings


def convert_file(path, name, exe_path):
    """
    Convert an EDF file to an ASCII file.
    
    Args:
        path (str): Path to the directory containing the EDF file.
        name (str): Name of the EDF file.
        exe_path (str): Path to the directory containing edf2asc.exe.
    """
    cmd = os.path.join(exe_path, "edf2asc.exe")
    asc_file = name[:-3] + "asc"
    full_path = os.path.join(path, name)
    output_path = os.path.join(path, asc_file)

    if not os.path.isfile(output_path):
        subprocess.run([cmd, full_path])
    else:
        warnings.warn(f"An ASCII file for {name} already exists!")


def convert_batch(data_dir, exe_dir):
    """
    Process all directories in data_dir starting with "sub-" and convert any EDF files found to ASCII.
    
    Args:
        data_dir (str): Path to the main data directory.
        exe_dir (str): Path to the directory containing edf2asc.exe.
    """
    for item in os.listdir(data_dir):
        if item.startswith("sub-"):
            sub_dir = os.path.join(data_dir, item)
            for session in os.listdir(sub_dir):
                if session.startswith("ses-"):
                    ses_dir = os.path.join(sub_dir, session)
                    beh_dir = os.path.join(ses_dir, "beh")
                    if os.path.isdir(beh_dir):
                        for file in os.listdir(beh_dir):
                            if file.startswith("e01") and file.endswith(".edf"):
                                print(f"\nProcessing File: {file}")
                                convert_file(beh_dir, file, exe_dir)


def main():
    data_dir = r"../../Landmark_Data"
    exe_dir = r"./"

    convert_batch(data_dir, exe_dir)


if __name__ == "__main__":
    main()
