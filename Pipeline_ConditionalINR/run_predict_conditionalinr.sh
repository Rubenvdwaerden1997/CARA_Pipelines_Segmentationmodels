#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./slurm_output/_slurm_output_predict_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_predict_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-mewtwo,dlc-moltres,dlc-nidoking,dlc-scyther,dlc-lugia,dlc-zapdos

cd /data/diag/rubenvdw/Conditional_INR/Pipeline_ConditionalINR/

pip install -U numba

echo "Running on GPU node: $(hostname)"

python3 -u Pipeline_ConditionalINR.py \
    --data_input  /data/diag/rubenvdw/Test/Dicom \
    --data_output /data/diag/rubenvdw/Test/Segmentation \
    --preprocessing \
    --predict \
    --model_dir  /data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense \
    --checkpoint best \
    --overlap    0.9 \
    --device     cuda \
    --postprocess
