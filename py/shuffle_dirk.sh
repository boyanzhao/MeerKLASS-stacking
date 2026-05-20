#!/bin/bash
#SBATCH --job-name=shfdirk
#SBATCH --output=shuffle_dirk_%j.out
#SBATCH --error=shuffle_dirk_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=869156867@qq.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

/users/zhaoboyan/.conda/envs/meer21cm/bin/python -u \
shuffle_dirk.py