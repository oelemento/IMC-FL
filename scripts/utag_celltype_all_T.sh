#!/bin/bash
#SBATCH --job-name=utag_ctT
#SBATCH --output=<PROJECT_ROOT>/utag_celltype_all_T.log
#SBATCH --time=04:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python <PROJECT_ROOT>/scripts/utag_celltype_all_T.py
