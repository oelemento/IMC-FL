#!/bin/bash
#SBATCH --job-name=utag_allT
#SBATCH --output=<PROJECT_ROOT>/utag_all_T.log
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/run_utag_all_T.py
