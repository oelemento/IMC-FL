#!/bin/bash
#SBATCH --job-name=utag_b1_S
#SBATCH --output=<PROJECT_ROOT>/utag_b1_S.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/run_utag_b1_S.py
