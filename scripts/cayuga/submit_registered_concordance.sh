#!/bin/bash
#SBATCH --job-name=reg_concord
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/reg_concordance_%j.log

cd <PROJECT_ROOT>
mkdir -p logs output

source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

python scripts/registered_concordance.py \
    --t-panel output/all_TMA_T_global_v8.h5ad \
    --s-panel output/all_TMA_S_global_v8.h5ad \
    --output output/registered_concordance.csv
