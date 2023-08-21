# -*- coding: utf-8 -*-
"""
===============================================
03. substr volumes to tabe

This code reads the volume of all subcortical
structures from a text file (output of fslstats)
and put them in a table.

written by Tara Ghafari
==============================================
"""

import numpy as np
import os.path as op
import pandas as pd

# Define where to read and write the data
mri_deriv_dir = r'Z:\Projects\Subcortical_Structures\SubStr_and_behavioral_bias\Analysis\MRI_lateralisations'
subStr_segmented_dir = op.join(mri_deriv_dir, 'substr_segmented')
output_dir = op.join(mri_deriv_dir, 'lateralisation_indices')
output_fname = op.join(output_dir, 'all_subs_substr_volumes.csv')

# list of subjects folders
dir_subject_list = [r'20211007#C4DF_nifti_ClemAtkin\20211007#C4DF_nifti.SubVol',
                     r'20211103#C59B_nifti_SiddhantBhutkar\20211103#C59B_nifti.SubVol',
                     r'20221102#C59E_nifti_BethHudson\20221102#C59E_nifti.SubVol',
                     r'20221214#C40F_nifti_VaentinPiscuc\20221214#C40F_nifti.SubVol',
                     r'20230202#C64A_nifti_SrishtiNarang\20230202#C64A_nifti.SubVol',
                     r'20230519#C345_nifti_SumedhaRaj\20230519#C345_nifti.SubVol']

# Specify labels assigned to structures thatwere segmented by FSL
labels = [10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
structures = ['L-Thal', 'L-Caud', 'L-Puta', 'L-Pall', 'BrStem /4th Ventricle',
              'L-Hipp', 'L-Amyg', 'L-Accu', 'R-Thal', 'R-Caud', 'R-Puta',
              'R-Pall', 'R-Hipp', 'R-Amyg', 'R-Accu']

all_subject_substr_volume_table = np.full((6, 15), np.nan)
sub_IDs =[]

# Read good subjects 
for i, subject_dir in enumerate(dir_subject_list):
    substr_dir = op.join(subStr_segmented_dir, subject_dir)
    if op.exists(substr_dir):
        for idx, label in enumerate(labels):
            volume_label = 'volume' + str(label) + '.txt'
            substr_vol_fname = op.join(substr_dir, volume_label)
            if op.exists(substr_vol_fname):
                print(f"reading structure {structures[idx]} in subject # {i}")
                # Read the text file
                with open(substr_vol_fname, "r") as file:
                    line = file.readline()
                substr_volume_array = np.fromstring(line.strip(), sep=' ')[1]     
            else:
                print(f"no volume for substructure {structures[idx]} found for subject # {i}")
                substr_volume_array = np.nan  
            
            # Store the volume of each substr in one columne and data of each subject in one row  
            all_subject_substr_volume_table[i, idx] = substr_volume_array
    else:
        print('no substructures segmented by fsl for subject # ', i)
        all_subject_substr_volume_table[i, :] = np.nan 
    
    sub_IDs.append(i)
    
 
# Create a dataframe for all the data
columns = ['SubID'] + structures
df = pd.DataFrame(np.hstack((np.array(sub_IDs).reshape(-1, 1), all_subject_substr_volume_table)),
                  columns=columns)
df.set_index('SubID', inplace=True)

# Save 
df.to_csv(output_fname)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    