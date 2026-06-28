#!/bin/bash
#SBATCH --job-name=imc-retry
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=<PROJECT_ROOT>/logs/retry_%A_%a.out
#SBATCH --error=<PROJECT_ROOT>/logs/retry_%A_%a.err

# Process one failed ROI per task
CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"
DATA_DIR="<DATA_ROOT>/Jan 18 2022_FL_TMA_B1_T"

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl
cd "$PROJECT_DIR"

python scripts/retry_failed.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --data-dir "$DATA_DIR" \
    --output-dir "$PROJECT_DIR/output/batch"
