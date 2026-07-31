#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag\ \
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/nnunetv2:1.3"
#SBATCH --qos=high
#SBATCH -o slurm_output/_slurm_output_predict_%j.txt
#SBATCH -e slurm_output/_slurm_error_predict_%j.txt
#SBATCH --exclude=dlc-mewtwo

cd /data/diag/rubenvdw/Pipeline_nnunetv2/Faster_pipeline/Pipeline/

pip3 install numba

# Print the hostname (GPU node) the job is  running on
echo "Running on GPU node: $(hostname)"

python3 -u Pipeline_withoutinternalsaving.py \
    --data_input /data/diag/CARAlab_datasets/PECTUS-obs \
    --data_output /data/diag/rubenvdw/Layered_Plaque_analysis_EuroPCR2026/Pectus_predictions \
    --k 3 \
    --radius 352 \
    --preprocessing  \
    --predict \
    --postprocessing_connected_component_dict /data/diag/rubenvdw/3D_model_OCTsegmentation/Data_info/Pixels_postprocessing_12classes_3D_OCTsegmentation.txt \
    -d Dataset801_LPanalysisEuroPCR \
    -model_results_folder /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_results \
