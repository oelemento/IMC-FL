#!/bin/bash
#SBATCH --job-name=utag_name
#SBATCH --output=<PROJECT_ROOT>/utag_name_compartments.log
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/utag_name_compartments.py
