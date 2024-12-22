#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=SOS-GradCAM
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm_gradcam-%A.out


srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main-gradcam.py --model_name=$1 --encoder_name=$2 --epochs=$3
