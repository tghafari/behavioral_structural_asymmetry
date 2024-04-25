#!/bin/bash
#SBATCH --account jenseno-avtemporal-attention
#SBATCH --qos bbdefault
#SBATCH --time 90
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 5 # this will give you circa 40G RAM and will ensure faster conversion to the .sif format

module purge
module load bluebear
module load FSL/6.0.7.9
# module load FSL/6.0.5.1-foss-2021a-fslpython

set -e
source ${FSLDIR}/etc/fslconf/fsl.sh  # set environment variables

# Define the location of the file

export base_dir="/rds/projects/j/jenseno-avtemporal-attention/Projects/subcortical-structures/SubStr-and-behavioral-bias"
mri_deriv_dir="${base_dir}/results/MRI_lateralisations/substr_segmented"
subject_name="S1020"
subject_mri_dir="${mri_deriv_dir}/${subject_name}.anat/first_results"



# Define variables for FSL command
labels=(10 11 12 13 16 17 18 26 49 50 51 52 53 54 58)
structures=("L-Thal" "L-Caud" "L-Puta" "L-Pall" "BrStem /4th Ventricle" \
    "L-Hipp" "L-Amyg" "L-Accu" "R-Thal" "R-Caud" "R-Puta" \
    "R-Pall" "R-Hipp" "R-Amyg" "R-Accu")

# Read segmented MRIs

if [ -d "$subject_mri_dir" ]; then 
   mkdir -p "${mri_deriv_dir}/${subject_name}.SubVol" 
    
   echo "${subject_name}.anat/first_results was found"

   for low in ${labels[@]}; do
    low_minus_point_five=$(echo "$low - 0.5" | bc)
    low_plus_point_five=$(echo "$low + 0.5" | bc)

    VoxVol=$(fslstats "${subject_mri_dir}/T1_first_all_fast_firstseg.nii.gz" -l "$low_minus_point_five" -u "$low_plus_point_five" -V)
    echo "Volumetring: ${subject_name} structure: ${low} = ${VoxVol}"
        
    output_fname="${mri_deriv_dir}/${subject_name}.SubVol/volume${low}.txt"
    echo "$VoxVol" > ${output_fname}  # Save VoxVol values to a text file     
     
   done

   echo "Volumetring ${subjectID} done"
	
else
  echo "no "$subject_mri_dir" found"
fi
