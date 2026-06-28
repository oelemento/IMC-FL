#!/bin/bash
#SBATCH --job-name=test_utag
#SBATCH --output=<PROJECT_ROOT>/test_utag.log
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/test_utag_single_roi.py
