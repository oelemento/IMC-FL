#!/bin/bash
#SBATCH --job-name=imc-S-combine
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<PROJECT_ROOT>/logs/S_combine_%j.out
#SBATCH --error=<PROJECT_ROOT>/logs/S_combine_%j.err

CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl

cd "$PROJECT_DIR"

python scripts/combine_results.py \
    --input-dir "$PROJECT_DIR/output/batch_S" \
    --panel S
