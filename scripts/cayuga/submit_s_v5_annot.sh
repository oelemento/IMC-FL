#!/bin/bash
#SBATCH --job-name=S5_annot_r2
#SBATCH --output=logs/S5_annot_r2_%j.out
#SBATCH --error=logs/S5_annot_r2_%j.err
#SBATCH --partition=scu-cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

cd <PROJECT_ROOT>

python scripts/cross_tma_global.py \
    --panel S \
    --step annotate \
    --checkpoint output/all_TMA_S_ckpt_v3_hires.h5ad \
    --resolutions 2.0 \
    --output output/all_TMA_S_global_v5.h5ad \
    --base-dir <PROJECT_ROOT>

echo "Done: S-panel v5 annotation (res 2.0)"
