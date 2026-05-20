#!/bin/bash
#SBATCH --job-name=convdirk
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zhaoby9941@outlook.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

/users/zhaoboyan/.conda/envs/meer21cm/bin/python -u \
convolution_dirk.py
