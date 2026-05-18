#!/bin/bash
#SBATCH --job-name=shfreal
#SBATCH --output=log/output_%j.log
#SBATCH --error=log/error_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

/users/zhaoboyan/.conda/envs/meer21cm/bin/python -u \
shuffle_real.py