#!/bin/bash
#SBATCH --account jenseno-avtemporal-attention
#SBATCH --qos bbdefault
#SBATCH --time 150
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 5 # this will give you circa 40G RAM and will ensure faster conversion to the .sif format

module purge
module load bluebear
module load FSL/6.0.5.1-foss-2021a-fslpython

set -e

# Define the location of the file
export base_dir="/rds/projects/j/jenseno-avtemporal-attention/Projects/subcortical-structures/SubStr-and-behavioral-bias"
output_dir="${base_dir}/results/MRI_lateralisations/substr_segmented"

subject_name="20231110#C5AB_nifti_S1005"
output_fpath="${output_dir}/S1005"
preproc_dir="${base_dir}/T1-scans/${subject_name}"


T1W_name="T1_vol_v1_5.nii.gz"
T1W_fpath="${preproc_dir}/${T1W_name}"

# Run the segmentation function in FSL container
# apptainer exec FSL.sif fsl_anat -i $T1W_fpath -o $output_fpath --clobber

# Run fsl anat in fsl
fsl_anat -i $T1W_fpath -o $output_fpath --clobber

