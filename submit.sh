#!/bin/bash
#SBATCH --job-name=goty_predict
#SBATCH --output=slurm_logs/goty_predict_%j.out   
#SBATCH --error=slurm_logs/goty_predict_%j.err    
#SBATCH --partition=gpu                   

source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv

cd ~/Desktop/GOTY
python main.py
