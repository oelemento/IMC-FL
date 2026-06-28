#!/bin/bash
# Cross-TMA T-panel v4: higher-resolution Leiden + fixed annotation
# Uses existing v3 checkpoint (embed already done), runs Leiden at 2.0 and 3.0
# Also re-runs 0.5 and 1.0 with fixed annotation logic (mixed cluster detection)
# Usage: bash scripts/cayuga/submit_cross_tma_t_hires_v4.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"
SCRIPT="scripts/cross_tma_global.py"
ACTIVATE="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR"

echo "=== Cross-TMA T-panel v4 (higher res + annotation fix) ==="

T_CKPT_SRC="$PROJECT_DIR/output/all_TMA_T_ckpt_v3.h5ad"

# Job 1: Leiden 0.5 with fixed annotations (re-run to get corrected cell types)
T_CKPT_1="$PROJECT_DIR/output/all_TMA_T_ckpt_v4.h5ad"
T_OUT_1="$PROJECT_DIR/output/all_TMA_T_global_v4.h5ad"
T_LEI1=$(sbatch --parsable \
    --job-name=T4-lei05 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=02:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T4_leiden05_%j.out" --error="$PROJECT_DIR/logs/xTMA_T4_leiden05_%j.err" \
    --wrap="$ACTIVATE && cp $T_CKPT_SRC $T_CKPT_1 && python $SCRIPT --panel T --step leiden --checkpoint $T_CKPT_1 --resolutions 0.5,1.0 --output $T_OUT_1")
echo "T4 leiden 0.5,1.0: $T_LEI1"

# Job 2: Leiden 2.0,3.0 (higher resolution to split mixed clusters)
T_CKPT_2="$PROJECT_DIR/output/all_TMA_T_ckpt_v4_hires.h5ad"
T_OUT_2="$PROJECT_DIR/output/all_TMA_T_global_v4_hires.h5ad"
T_LEI2=$(sbatch --parsable \
    --job-name=T4-lei30 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=02:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T4_leiden30_%j.out" --error="$PROJECT_DIR/logs/xTMA_T4_leiden30_%j.err" \
    --wrap="$ACTIVATE && cp $T_CKPT_SRC $T_CKPT_2 && python $SCRIPT --panel T --step leiden --checkpoint $T_CKPT_2 --resolutions 2.0,3.0 --output $T_OUT_2")
echo "T4 leiden 2.0,3.0: $T_LEI2"

echo ""
echo "=== 2 jobs submitted (parallel, no dependencies) ==="
echo "Job 1: $T_LEI1 — leiden 0.5,1.0 with fixed annotation (mixed cluster detection)"
echo "Job 2: $T_LEI2 — leiden 2.0,3.0 with fixed annotation"
echo "Both copy v3 checkpoint then run leiden + annotate."
echo "Monitor: squeue -u ole2001"
