#!/bin/bash
# Submit cross-TMA global analysis for T and S panels
# Combines all TMAs (A1, B1, C1, Biomax) per panel into a single unified clustering
# ~2.2M cells for T-panel, ~2.0M cells for S-panel
# Usage: bash scripts/cayuga/submit_cross_tma.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"

echo "=== Submitting cross-TMA global analysis ==="

# T-panel: ~2.2M cells, needs lots of memory
sbatch --parsable \
    --job-name=allTMA-T \
    --partition=scu-cpu \
    --cpus-per-task=16 \
    --mem=128G \
    --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/cross_tma_T_%j.out" \
    --error="$PROJECT_DIR/logs/cross_tma_T_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/cross_tma_global.py --panel T --base-dir $PROJECT_DIR --output $PROJECT_DIR/output/all_TMA_T_global.h5ad"
echo "Submitted T-panel cross-TMA (16 CPUs, 128GB, 48h)"

# S-panel: ~2.0M cells
sbatch --parsable \
    --job-name=allTMA-S \
    --partition=scu-cpu \
    --cpus-per-task=16 \
    --mem=128G \
    --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/cross_tma_S_%j.out" \
    --error="$PROJECT_DIR/logs/cross_tma_S_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/cross_tma_global.py --panel S --base-dir $PROJECT_DIR --output $PROJECT_DIR/output/all_TMA_S_global.h5ad"
echo "Submitted S-panel cross-TMA (16 CPUs, 128GB, 48h)"

echo ""
echo "=== Both jobs submitted ==="
echo "Each: 16 CPUs, 128GB RAM, 48h wall time"
echo "Expected: ~2.2M cells (T), ~2.0M cells (S)"
echo "Monitor: squeue -u <USER>"
