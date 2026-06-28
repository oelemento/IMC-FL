#!/bin/bash
# Submit cross-TMA global analysis v2: step-based with checkpoints
# Step 1: embed (PCA+UMAP) → Step 2a: leiden 0.3,0.5 | Step 2b: leiden 0.8,1.0 (parallel)
# Each leiden job saves after each resolution and annotates at the end
# Usage: bash scripts/cayuga/submit_cross_tma_v2.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"
SCRIPT="scripts/cross_tma_global.py"
ACTIVATE="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR"

echo "=== Cross-TMA v2: step-based with checkpoints ==="

# --- T-panel ---
echo ""
echo "--- T-panel (~2.2M cells) ---"

T_CKPT="$PROJECT_DIR/output/all_TMA_T_ckpt.h5ad"
T_OUT="$PROJECT_DIR/output/all_TMA_T_global.h5ad"

# Step 1: embed
T_EMBED=$(sbatch --parsable \
    --job-name=T-embed --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=04:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T_embed_%j.out" --error="$PROJECT_DIR/logs/xTMA_T_embed_%j.err" \
    --wrap="$ACTIVATE && python $SCRIPT --panel T --step embed --base-dir $PROJECT_DIR --checkpoint $T_CKPT")
echo "T embed: $T_EMBED (4h)"

# Step 2a: leiden 0.3,0.5 (depends on embed)
T_LEID1=$(sbatch --parsable --dependency=afterok:$T_EMBED \
    --job-name=T-lei05 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T_leiden05_%j.out" --error="$PROJECT_DIR/logs/xTMA_T_leiden05_%j.err" \
    --wrap="$ACTIVATE && python $SCRIPT --panel T --step leiden --checkpoint $T_CKPT --resolutions 0.3,0.5 --output $T_OUT")
echo "T leiden 0.3,0.5: $T_LEID1 (afterok:$T_EMBED)"

# Step 2b: leiden 0.8,1.0 (depends on embed, parallel with 2a)
T_LEID2=$(sbatch --parsable --dependency=afterok:$T_EMBED \
    --job-name=T-lei10 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_T_leiden10_%j.out" --error="$PROJECT_DIR/logs/xTMA_T_leiden10_%j.err" \
    --wrap="$ACTIVATE && cp $T_CKPT ${T_CKPT%.h5ad}_hires.h5ad && python $SCRIPT --panel T --step leiden --checkpoint ${T_CKPT%.h5ad}_hires.h5ad --resolutions 0.8,1.0 --output ${T_OUT%.h5ad}_hires.h5ad")
echo "T leiden 0.8,1.0: $T_LEID2 (afterok:$T_EMBED)"

# --- S-panel ---
echo ""
echo "--- S-panel (~2.0M cells) ---"

S_CKPT="$PROJECT_DIR/output/all_TMA_S_ckpt.h5ad"
S_OUT="$PROJECT_DIR/output/all_TMA_S_global.h5ad"

# Step 1: embed
S_EMBED=$(sbatch --parsable \
    --job-name=S-embed --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=04:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_S_embed_%j.out" --error="$PROJECT_DIR/logs/xTMA_S_embed_%j.err" \
    --wrap="$ACTIVATE && python $SCRIPT --panel S --step embed --base-dir $PROJECT_DIR --checkpoint $S_CKPT")
echo "S embed: $S_EMBED (4h)"

# Step 2a: leiden 0.3,0.5 (depends on embed)
S_LEID1=$(sbatch --parsable --dependency=afterok:$S_EMBED \
    --job-name=S-lei05 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_S_leiden05_%j.out" --error="$PROJECT_DIR/logs/xTMA_S_leiden05_%j.err" \
    --wrap="$ACTIVATE && python $SCRIPT --panel S --step leiden --checkpoint $S_CKPT --resolutions 0.3,0.5 --output $S_OUT")
echo "S leiden 0.3,0.5: $S_LEID1 (afterok:$S_EMBED)"

# Step 2b: leiden 0.8,1.0 (depends on embed, parallel with 2a)
S_LEID2=$(sbatch --parsable --dependency=afterok:$S_EMBED \
    --job-name=S-lei10 --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/xTMA_S_leiden10_%j.out" --error="$PROJECT_DIR/logs/xTMA_S_leiden10_%j.err" \
    --wrap="$ACTIVATE && cp $S_CKPT ${S_CKPT%.h5ad}_hires.h5ad && python $SCRIPT --panel S --step leiden --checkpoint ${S_CKPT%.h5ad}_hires.h5ad --resolutions 0.8,1.0 --output ${S_OUT%.h5ad}_hires.h5ad")
echo "S leiden 0.8,1.0: $S_LEID2 (afterok:$S_EMBED)"

echo ""
echo "=== All 6 jobs submitted ==="
echo "T-panel: embed($T_EMBED) → leiden05($T_LEID1) + leiden10($T_LEID2)"
echo "S-panel: embed($S_EMBED) → leiden05($S_LEID1) + leiden10($S_LEID2)"
echo ""
echo "Checkpoints saved after UMAP and after each Leiden resolution."
echo "leiden 0.3,0.5 jobs produce annotated output you can inspect immediately."
echo "leiden 0.8,1.0 jobs run in parallel on separate checkpoint copies."
echo "Monitor: squeue -u ole2001"
