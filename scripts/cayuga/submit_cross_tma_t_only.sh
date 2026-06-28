#!/bin/bash
# Re-run cross-TMA T-panel only (after B1 raw data fix)
# Step 1: embed (PCA+UMAP) → Step 2a: leiden 0.3,0.5 | Step 2b: leiden 0.8,1.0 (parallel)
# Usage: bash scripts/cayuga/submit_cross_tma_t_only.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"
SCRIPT="scripts/cross_tma_global.py"
ACTIVATE="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR"

echo "=== Cross-TMA T-panel re-run (B1 fix) ==="

T_CKPT="$PROJECT_DIR/output/all_TMA_T_ckpt_v3.h5ad"
T_OUT="$PROJECT_DIR/output/all_TMA_T_global_v3.h5ad"

# Step 1: embed
T_EMBED=$(sbatch --parsable \
    --job-name=T3-embed --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=04:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T3_embed_%j.out" --error="$PROJECT_DIR/logs/xTMA_T3_embed_%j.err" \
    --wrap="$ACTIVATE && python $SCRIPT --panel T --step embed --base-dir $PROJECT_DIR --checkpoint $T_CKPT")
echo "T embed: $T_EMBED (4h)"

# Step 2a: leiden 0.3,0.5 (depends on embed)
T_LEID1=$(sbatch --parsable --dependency=afterok:$T_EMBED \
    --job-name=T3-lei05 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T3_leiden05_%j.out" --error="$PROJECT_DIR/logs/xTMA_T3_leiden05_%j.err" \
    --wrap="$ACTIVATE && python $SCRIPT --panel T --step leiden --checkpoint $T_CKPT --resolutions 0.3,0.5 --output $T_OUT")
echo "T leiden 0.3,0.5: $T_LEID1 (afterok:$T_EMBED)"

# Step 2b: leiden 0.8,1.0 (depends on embed, parallel with 2a)
T_LEID2=$(sbatch --parsable --dependency=afterok:$T_EMBED \
    --job-name=T3-lei10 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T3_leiden10_%j.out" --error="$PROJECT_DIR/logs/xTMA_T3_leiden10_%j.err" \
    --wrap="$ACTIVATE && cp $T_CKPT ${T_CKPT%.h5ad}_hires.h5ad && python $SCRIPT --panel T --step leiden --checkpoint ${T_CKPT%.h5ad}_hires.h5ad --resolutions 0.8,1.0 --output ${T_OUT%.h5ad}_hires.h5ad")
echo "T leiden 0.8,1.0: $T_LEID2 (afterok:$T_EMBED)"

echo ""
echo "=== 3 T-panel jobs submitted ==="
echo "embed($T_EMBED) → leiden05($T_LEID1) + leiden10($T_LEID2)"
echo "Output uses v3 suffix to avoid overwriting previous (broken) results."
echo "Monitor: squeue -u <USER>"
