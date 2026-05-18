#!/bin/bash
#SBATCH --job-name=shfdirk
#SBATCH --output=log/output_%j.log
#SBATCH --error=log/error_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

/users/zhaoboyan/.conda/envs/meer21cm/bin/python -u \
shuffle_dirk.py
