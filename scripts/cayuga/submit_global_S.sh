#!/bin/bash
#SBATCH --job-name=imc-S-global
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=<PROJECT_ROOT>/logs/S_global_%j.out
#SBATCH --error=<PROJECT_ROOT>/logs/S_global_%j.err

CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"

echo "=== Global S-panel analysis ==="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Mem: $SLURM_MEM_PER_NODE"
echo ""

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl

cd "$PROJECT_DIR"

python scripts/global_analysis.py \
    --panel S \
    --input "$PROJECT_DIR/output/batch_S/TMA_B1_S_combined.h5ad" \
    --output "$PROJECT_DIR/output/batch_S/TMA_B1_S_global.h5ad"

echo ""
echo "=== Done ==="
