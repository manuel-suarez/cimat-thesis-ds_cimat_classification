#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=CIMAT-Training
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm_training-%A_%a.out

model_name=$1
encoder_name=$2
epochs=$3
srun /home/${whoami}/tools/anaconda3/envs/py3.9-pt/bin/python main-train.py --epochs=$epochs --encoder_name=$encoder_name
