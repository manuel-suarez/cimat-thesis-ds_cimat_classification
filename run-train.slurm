#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=CIMAT-Training
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-10%1
#SBATCH --output=outputs/slurm_training-%A_%a.out


srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main-train.py --epochs=$1 --encoder_name=base --dataset=17 --trainset=01 --feat_channels=oov
