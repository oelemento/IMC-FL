#!/bin/bash
#SBATCH --job-name=imc-S-cpu
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=<PROJECT_ROOT>/logs/S_cpu_%A_%a.out
#SBATCH --error=<PROJECT_ROOT>/logs/S_cpu_%A_%a.err

# Each array task processes 5 ROIs
# 47 S-panel ROIs -> 10 tasks (0-9), last task gets remainder

CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"
DATA_DIR="<DATA_ROOT>/May_18_2021_FL_TMA_B1_S"

BATCH_SIZE=5
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))

echo "=== S-panel array task ${SLURM_ARRAY_TASK_ID}: ROIs ${START}-$((END-1)) ==="
echo "Node: $(hostname)"
echo ""

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl

cd "$PROJECT_DIR"

python scripts/batch_process_S.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$PROJECT_DIR/output/batch_S" \
    --start $START \
    --end $END

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
