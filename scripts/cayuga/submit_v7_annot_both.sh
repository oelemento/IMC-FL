#!/bin/bash
#SBATCH --job-name=v7_annot
#SBATCH --output=logs/v7_annot_%j_%a.out
#SBATCH --error=logs/v7_annot_%j_%a.err
#SBATCH --partition=scu-cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-1

source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

cd <PROJECT_ROOT>

PANELS=(T S)
PANEL=${PANELS[$SLURM_ARRAY_TASK_ID]}

if [ "$PANEL" == "T" ]; then
    CKPT=output/all_TMA_T_ckpt_v3_hires.h5ad
    OUT=output/all_TMA_T_global_v7.h5ad
else
    CKPT=output/all_TMA_S_ckpt_v3_hires.h5ad
    OUT=output/all_TMA_S_global_v7.h5ad
fi

echo "Panel: $PANEL, Checkpoint: $CKPT, Output: $OUT"

python scripts/cross_tma_global.py \
    --panel $PANEL \
    --step annotate \
    --checkpoint $CKPT \
    --resolutions 2.0 \
    --output $OUT \
    --base-dir <PROJECT_ROOT>

echo "Done: ${PANEL}-panel v7 annotation (per-cell gating, harmonized thresholds)"
