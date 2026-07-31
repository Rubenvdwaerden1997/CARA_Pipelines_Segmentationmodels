#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag\ \
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH --qos=high
#SBATCH -o slurm_output/_slurm_output_predict_%j.txt
#SBATCH -e slurm_output/_slurm_error_predict_%j.txt
#SBATCH --exclude=dlc-mewtwo

cd /data/diag/rubenvdw/Pipeline_OCTAIDlite/Pipeline/

pip install -U segmentation-models-pytorch numba einops

# Print the hostname (GPU node) the job is  running on
echo "Running on GPU node: $(hostname)"

python3 -u Pipeline_withoutinternalsaving.py \
    --data_input /data/diag/rubenvdw/Test/Dicom \
    --data_output /data/diag/rubenvdw/Test/Segmentation \
    --k 3 \
    --radius 352 \
    --preprocessing  \
    --predict \
    --postprocessing_connected_component_dict /data/diag/rubenvdw/Pipeline_OCTAIDlite/Model_info/Pixels_postprocessing.txt \
    --model_path_student "/data/diag/rubenvdw/Pipeline_OCTAIDlite/Model_info/Dataset601_TS3D3_exp9_Hardloss_manual_pseudo_Softloss_pseudo_artefactexclusionsevereframes_entropy5%/fold_0/Last_model_checkpoint.pth" \
    --model_name_student UNet \
