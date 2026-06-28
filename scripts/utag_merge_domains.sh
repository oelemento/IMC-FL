#!/bin/bash
#SBATCH --job-name=utag_merge
#SBATCH --output=<PROJECT_ROOT>/utag_merge_domains.log
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/utag_merge_domains.py
