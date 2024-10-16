"""
===============================================
01_parsing_info_from_asc

This code gets the ASC file from the eyetracker
and detects microsaccades.
EDF files (output of EyeLink) should be
converted to .asc using 'visualEDF2ASC' app

Written by Tara Ghafari
Adapted from:
https://github.com/Cogitate-consortium/Eye-Tracking-Code/blob/master/Experiment%201/Python/DataParser.py#L26
===============================================
ToDos:
"""

import os
import os.path as op
import time
import pickle
import numpy as np
import pandas as pd


def parse_eyelink_asc(eyetracking_asc_file):
    """
    Reads in data files from EyeLink .asc file and produces readable dataframes for further analysis.

    Args:
    eyetracking_asc_file (str): Path to the EyeLink data file

    Returns:
    tuple: (dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples)
        dfRec: DataFrame containing information about recording periods (often trials)
        dfMsg: DataFrame containing information about messages (usually sent from stimulus software)
        dfFix: DataFrame containing information about fixations
        dfSacc: DataFrame containing information about saccades
        dfBlink: DataFrame containing information about blinks
        dfSamples: DataFrame containing information about individual samples
    """

    # Read in EyeLink file
    t = time.time()
    with open(eyetracking_asc_file, 'r') as f:
        file_txt = f.read().splitlines(True)
    file_txt = list(filter(None, file_txt))  # remove empty lines
    file_txt = np.array(file_txt)  # convert to np array for simpler indexing
    print(f'Done! Took {time.time() - t:.2f} seconds.')

    # Separate lines into samples and messages
    print('Sorting lines...')
    n_lines = len(file_txt)
    line_type = np.array(['OTHER'] * n_lines, dtype='object')
    i_start_rec = []
    t = time.time()
    for i_line, line in enumerate(file_txt):
        if line in ("**\n", "\n"):
            line_type[i_line] = 'EMPTY'
        elif line.startswith('*') or line.startswith('>>>>>'):
            line_type[i_line] = 'COMMENT'
        elif line[0].isdigit():
            file_txt[i_line] = line.replace('.\t', 'NaN\t')
            line_type[i_line] = 'SAMPLE'
        else:
            line_type[i_line] = line.split()[0]
        if '!MODE RECORD' in line:
            i_start_rec.append(i_line + 1)

    i_start_rec = i_start_rec[0]
    print(f'Done! Took {time.time() - t:.2f} seconds.')

    # Parse EyeLink file
    t = time.time()
    # Trials
    print('Parsing recording markers...')
    i_not_start = np.nonzero(line_type != 'START')[0]
    df_rec_start = pd.read_csv(eyetracking_asc_file, skiprows=i_not_start,
                               header=None, delim_whitespace=True, usecols=[1])
    df_rec_start.columns = ['tStart']
    i_not_end = np.nonzero(line_type != 'END')[0]
    df_rec_end = pd.read_csv(eyetracking_asc_file, skiprows=i_not_end,
                             header=None, delim_whitespace=True, usecols=[1, 5, 6])
    df_rec_end.columns = ['tEnd', 'xRes', 'yRes']
    # combine trial info
    df_rec = pd.concat([df_rec_start, df_rec_end], axis=1)
    n_rec = df_rec.shape[0]
    print(f'{n_rec} recording periods found.')

    # Import Messages
    print('Parsing stimulus messages...')
    t = time.time()
    i_msg = np.nonzero(line_type == 'MSG')[0]
    t_msg = []
    txt_msg = []
    for i in range(len(i_msg)):
        info = file_txt[i_msg[i]].split()
        t_msg.append(int(info[1]))
        txt_msg.append(' '.join(info[2:]))
    df_msg = pd.DataFrame({'time': t_msg, 'text': txt_msg})
    print(f'Done! Took {time.time() - t:.2f} seconds.')

    # Import Fixations
    print('Parsing fixations...')
    t = time.time()
    i_not_efix = np.nonzero(line_type != 'EFIX')[0]
    df_fix = pd.read_csv(eyetracking_asc_file, skiprows=i_not_efix,
                         header=None, delim_whitespace=True, usecols=range(1, 8))
    df_fix.columns = ['eye', 'tStart', 'tEnd',
                      'duration', 'xAvg', 'yAvg', 'pupilAvg']
    print(f'Done! Took {time.time() - t:.2f} seconds.')

    # Saccades
    print('Parsing saccades...')
    t = time.time()
    i_not_esacc = np.nonzero(line_type != 'ESACC')[0]
    df_sacc = pd.read_csv(eyetracking_asc_file, skiprows=i_not_esacc,
                          header=None, delim_whitespace=True, usecols=range(1, 11))
    df_sacc.columns = ['eye', 'tStart', 'tEnd', 'duration',
                       'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']
    print(f'Done! Took {time.time() - t:.2f} seconds.')

    # Blinks
    print('Parsing blinks...')
    i_not_eblink = np.nonzero(line_type != 'EBLINK')[0]
    df_blink = pd.read_csv(eyetracking_asc_file, skiprows=i_not_eblink,
                           header=None, delim_whitespace=True, usecols=range(1, 5))
    df_blink.columns = ['eye', 'tStart', 'tEnd', 'duration']
    print(f'Done! Took {time.time() - t:.2f} seconds.')

    # Determine sample columns based on eyes recorded in file
    eyes_in_file = np.unique(df_fix.eye)
    if eyes_in_file.size == 2:
        print('Binocular data detected.')
        cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyes_in_file[0]
        print(f'Monocular data detected ({eye} eye).')
        cols = ['tSample', f'{eye}X', f'{eye}Y', f'{eye}Pupil']

    # Import samples
    print('Parsing samples...')
    t = time.time()
    i_not_sample = np.nonzero(np.logical_or(
        line_type != 'SAMPLE', np.arange(n_lines) < i_start_rec))[0]
    df_samples = pd.read_csv(eyetracking_asc_file, skiprows=i_not_sample, header=None, delim_whitespace=True,
                             usecols=range(0, len(cols)))
    df_samples.columns = cols

    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyes_in_file:
            for col in [f'{eye}X', f'{eye}Y', f'{eye}Pupil']:
                df_samples[col] = pd.to_numeric(
                    df_samples[col], errors='coerce')
        else:
            for col in [f'{eye}X', f'{eye}Y', f'{eye}Pupil']:
                df_samples[col] = np.nan

    print(f'Done! Took {time.time() - t:.2f} seconds.')

    return df_rec, df_msg, df_fix, df_sacc, df_blink, df_samples


def main():
    data_dir = r"../../Landmark_Data"
    output_folder_path = r"../../Results/EyeTracking"

    os.makedirs(output_folder_path, exist_ok=True)

    for item in os.listdir(data_dir):
        if item.startswith("sub-"):
            sub_dir = os.path.join(data_dir, item)
            for session in os.listdir(sub_dir):
                if session.startswith("ses-"):
                    ses_dir = os.path.join(sub_dir, session)
                    beh_dir = os.path.join(ses_dir, "beh")
                    if os.path.isdir(beh_dir):
                        for file in os.listdir(beh_dir):
                            if file.endswith(".asc"):
                                print(f"\nProcessing File: {file}")
                                eye_data = parse_eyelink_asc(
                                    os.path.join(beh_dir, file))

                                output_dir = op.join(output_folder_path, item)
                                os.makedirs(output_dir, exist_ok=True)

                                output_file_name = f"{file[:-4]}_EL_eyeData.pkl"
                                output_file_path = os.path.join(
                                    output_dir, output_file_name)

                                with open(output_file_path, 'wb') as f:
                                    pickle.dump(eye_data, f)


if __name__ == "__main__":
    main()
