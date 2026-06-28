#!/bin/bash
#SBATCH --job-name=plot_FL11
#SBATCH --output=<PROJECT_ROOT>/plot_FL11.log
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/plot_utag_FL11.py
