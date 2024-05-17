# -*- coding: utf-8 -*-
"""
===============================================
06_calculate_microsaccade_bias

this code gets saccade information from right
and left attention epochs as input. 
creates a new variable with only direction of 
both-eye microsaccades for each attention condition.
then it averages across the distance to fixation
in right epochs and in left epochs separately.
finally, calculates 
ms_bias_indx = (att_right_avg - att_left_avg) /\
               (att_right_avg + att_left_avg)
               
written by Tara Ghafari
===============================================
"""
import os.path as op
import pandas as pd
import pickle
from statistics import mean    
    
# Loop through the list and extract the 'direction' column
def read_ms_direction(saccadeinfo):
    attention_ms_direction = []
    for saccade_dict in saccadeinfo:
        microsaccade_dict = saccade_dict['Microsaccades']
        if microsaccade_dict['both'] is not None:
            direction = microsaccade_dict['both']['distance_to_fixation']
            attention_ms_direction.extend(direction)
    return attention_ms_direction
    
platform= 'mac'

if platform == 'bluebear':
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define where to read and write the data
deriv_dir = op.join(jenseno_dir,'Projects/subcortical-structures/SubStr-and-behavioral-bias/derivatives')

ms_lateralisation_table =[]
# Load microsaccade info lists
for sub_code in range(7,33):
    output_fpath = op.join(deriv_dir, 'target_orientation', 'eyetracking')
    output_dir = op.join(output_fpath,'sub-S' + str(1000+sub_code))

    with open(op.join(output_dir, 'EL_saccadeinfo_right.json'), 'rb') as f:
        saccadeinfo_right = pickle.load(f)
    with open(op.join(output_dir, 'EL_saccadeinfo_left.json'), 'rb') as f:
        saccadeinfo_left = pickle.load(f)
        
    # pick across direction of microsaccade in attention right and left conditions
    attention_right_ms_direction = read_ms_direction(saccadeinfo_right)
    attention_left_ms_direction = read_ms_direction(saccadeinfo_left)
    
    # average 
    right_att_mean_direction = mean(attention_right_ms_direction)
    left_att_mean_direction = mean(attention_left_ms_direction)
    
    # calculate ms bias index
    microsaccade_lateralisation_idx = (right_att_mean_direction - left_att_mean_direction) /\
                                      (right_att_mean_direction + left_att_mean_direction)  
    ms_lateralisation_table.append(microsaccade_lateralisation_idx)

# first convert table to df then save as .csv
all_subs_fpath = op.join(output_fpath,'all-subs', 'ms_lateralisation_6subs.csv')
ms_lateralisation_df = pd.DataFrame(ms_lateralisation_table)     
ms_lateralisation_df.to_csv(all_subs_fpath,index=True)      
                                      