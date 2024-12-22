#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=CIMAT-TestModels
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm-tests-%A.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python -m unittest -v test_models
