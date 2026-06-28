#!/bin/bash
#SBATCH --job-name=utag_allS
#SBATCH --output=<PROJECT_ROOT>/utag_all_S.log
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/run_utag_all_S.py
