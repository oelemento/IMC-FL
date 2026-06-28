#!/bin/bash
#SBATCH --job-name=utag_v2T
#SBATCH --output=<PROJECT_ROOT>/utag_per_tma_T.log
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/utag_per_tma_T.py
