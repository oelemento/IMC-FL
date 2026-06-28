#!/bin/bash
#SBATCH --job-name=imc-fl-cpu
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-10
#SBATCH --output=<PROJECT_ROOT>/logs/cpu_%A_%a.out
#SBATCH --error=<PROJECT_ROOT>/logs/cpu_%A_%a.err

# Each array task processes 5 ROIs
# Task 0: ROIs 0-4, Task 1: ROIs 5-9, ..., Task 10: ROIs 50-54

CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"
DATA_DIR="<DATA_ROOT>/Jan 18 2022_FL_TMA_B1_T"

BATCH_SIZE=5
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: ROIs ${START}-$((END-1)) ==="
echo "Node: $(hostname)"
echo ""

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl

cd "$PROJECT_DIR"

python scripts/batch_process.py \
    --method hybrid \
    --panel T \
    --data-dir "$DATA_DIR" \
    --output-dir "$PROJECT_DIR/output/batch" \
    --start $START \
    --end $END

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
