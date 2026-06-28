#!/bin/bash
# Cross-TMA S-panel v3: higher-resolution Leiden + T-cell-first annotation
# Uses existing v2 checkpoint (embed already done), runs Leiden at 2.0 and 3.0
# Also re-runs 0.5 and 1.0 with T-cell-first annotation logic
# Usage: bash scripts/cayuga/submit_cross_tma_s_hires_v3.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"
SCRIPT="scripts/cross_tma_global.py"
ACTIVATE="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR"

echo "=== Cross-TMA S-panel v3 (higher res + T-cell-first annotation) ==="

S_CKPT_SRC="$PROJECT_DIR/output/all_TMA_S_ckpt.h5ad"

# Job 1: Leiden 0.5,1.0 with T-cell-first annotation (re-run for corrected cell types)
S_CKPT_1="$PROJECT_DIR/output/all_TMA_S_ckpt_v3.h5ad"
S_OUT_1="$PROJECT_DIR/output/all_TMA_S_global_v3.h5ad"
S_LEI1=$(sbatch --parsable \
    --job-name=S3-lei05 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=02:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_S3_leiden05_%j.out" --error="$PROJECT_DIR/logs/xTMA_S3_leiden05_%j.err" \
    --wrap="$ACTIVATE && cp $S_CKPT_SRC $S_CKPT_1 && python $SCRIPT --panel S --step leiden --checkpoint $S_CKPT_1 --resolutions 0.5,1.0 --output $S_OUT_1")
echo "S3 leiden 0.5,1.0: $S_LEI1"

# Job 2: Leiden 2.0,3.0 (higher resolution to split mixed/LQ clusters)
S_CKPT_2="$PROJECT_DIR/output/all_TMA_S_ckpt_v3_hires.h5ad"
S_OUT_2="$PROJECT_DIR/output/all_TMA_S_global_v3_hires.h5ad"
S_LEI2=$(sbatch --parsable \
    --job-name=S3-lei30 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=02:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_S3_leiden30_%j.out" --error="$PROJECT_DIR/logs/xTMA_S3_leiden30_%j.err" \
    --wrap="$ACTIVATE && cp $S_CKPT_SRC $S_CKPT_2 && python $SCRIPT --panel S --step leiden --checkpoint $S_CKPT_2 --resolutions 2.0,3.0 --output $S_OUT_2")
echo "S3 leiden 2.0,3.0: $S_LEI2"

echo ""
echo "=== 2 jobs submitted (parallel, no dependencies) ==="
echo "Job 1: $S_LEI1 — leiden 0.5,1.0 with T-cell-first annotation"
echo "Job 2: $S_LEI2 — leiden 2.0,3.0 with T-cell-first annotation"
echo "Both copy v2 checkpoint then run leiden + annotate."
echo "Monitor: squeue -u ole2001"
