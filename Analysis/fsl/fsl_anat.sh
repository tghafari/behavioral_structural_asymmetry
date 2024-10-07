#!/bin/bash
#SBATCH --account jenseno-avtemporal-attention
#SBATCH --qos bbdefault
#SBATCH --time 150
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 5 # this will give you circa 40G RAM 
#SBATCH --array=0-3  # Run one task for each subject name


module purge
module load bluebear
module load bear-apps/2022b
module load FSL/6.0.7.9
# module load FSL/6.0.5.1-foss-2021a-fslpython

set -e
source ${FSLDIR}/etc/fslconf/fsl.sh  # set environment variables

# Define the location of the file
export base_dir="/rds/projects/j/jenseno-avtemporal-attention/Projects/subcortical-structures/SubStr-and-behavioral-bias"
output_dir="${base_dir}/derivatives/MRI_lateralisations/substr_segmented"

t1_fnames=('S1045_20240618#C40C_nifti' 'S1044_20240624#C537_nifti' 'S1043_20240729#C388_nifti')
#('S1038_20240507#C5F7_nifti' 'S1041_20240422#C57E_nifti' 'S1039_20240621#C533_nifti'
# 'S1042_20240522#C64D_nifti' 'S1040_20240605#C546_nifti' 'S1041_20240422#C57E_nifti'
# 'S1037_20230525#C4D0_nifti' 'S1036_20240503#C416_nifti' 
# 'S1035_20240411#C453_nifti' 'S1034_20240502#C423_nifti'
# 'S1033_20240503#C389_nifti''S1021_20220923#C47E_nifti' 
# 'S1022_20221102#C5F2_nifti' 'S1023_20240208#C3FA_nifti'
# 'S1024_20230426#C399_nifti' 'S1025_20211029#C3B4_nifti' 'S1026_20240313#C469_nifti'
# 'S1027_20240229#C472_nifti' 'S1028_20221202#C47B_nifti' 'S1029_20240229#C515_nifti' 
#  'S1030_20220308#C3A1_nifti' 'S1031_20240215#C416_nifti' 'S1032_20240229#C472_nifti')

# Get the subject name for each array task
t1_fname="${t1_fnames[$SLURM_ARRAY_TASK_ID]}"  # if wanting to run on one subject, put the name of subject in t1_fname

output_fpath="${output_dir}/${t1_fname:0:5}"
preproc_dir="${base_dir}/T1-scans/${t1_fname}"

mkdir -p "${output_fpath}.anat"
T1W_name="T1_vol_v1_5.nii.gz"
T1W_fpath="${preproc_dir}/${T1W_name}"

# Run fsl anat
fsl_anat -i "$T1W_fpath" -o "$output_fpath" --clobber