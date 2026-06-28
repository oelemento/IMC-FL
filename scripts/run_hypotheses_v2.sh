#!/bin/bash
#SBATCH --job-name=hyp_v2
#SBATCH --output=<PROJECT_ROOT>/run_hypotheses_v2.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /etc/profile
source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

OUTDIR="<PROJECT_ROOT>/output"

python <PROJECT_ROOT>/scripts/run_hypotheses_v2.py \
    --t-panel "${OUTDIR}/all_TMA_T_global_v8.h5ad" \
    --s-panel "${OUTDIR}/all_TMA_S_global_v8.h5ad" \
    --t-utag  "${OUTDIR}/all_TMA_T_utag_ct_merged.h5ad" \
    --s-utag  "${OUTDIR}/all_TMA_S_utag_ct_merged.h5ad" \
    --output-dir "${OUTDIR}/hypothesis_figs_v2" \
    --cartoon-dir "${OUTDIR}/hypothesis_cartoons"
