#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag\ \
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/nnunetv2:1.3"
#SBATCH --qos=high
#SBATCH -o slurm_output/_slurm_output_predict_v2_%j.txt
#SBATCH -e slurm_output/_slurm_error_predict_v2_%j.txt
#SBATCH --exclude=dlc-mewtwo

cd /data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_3DUnet/Pipeline/

pip3 install numba monai

# Print the hostname (GPU node) the job is  running on
echo "Running on GPU node: $(hostname)"

python3 -u Pipeline_withoutinternalsaving_v2.py \
    --data_input /data/diag/rubenvdw/Test/Dicom \
    --data_output /data/diag/rubenvdw/Test/Segmentation \
    --radius 352 \
    --preprocessing  \
    --predict \
    --postprocessing_connected_component_dict /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info/Pixels_postprocessing_Externalvalidation_HP.txt \
    --model_path /data/diag/rubenvdw/Pipeline_3DUnet/Model/Z16_3D_includingpseudo/Weights \
    --model_name 3DUnet_Jasper \
    --overlap_sw 0.93 \
