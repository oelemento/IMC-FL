#!/bin/bash
# Re-combine all TMAs with raw counts, then re-run global analysis
# Fixes the double-transform bug where combined files had transformed data
# Usage: bash scripts/cayuga/submit_recombine_and_global.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"

echo "=== Step 1: Re-combine with raw counts ==="

# Recombine all 6 non-B1 TMAs (B1-T already has correct raw in combined)
# B1-S also needs recombine
RECOMB=$(sbatch --parsable \
    --job-name=recomb \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=64G \
    --time=04:00:00 \
    --output="$PROJECT_DIR/logs/recombine_%j.out" \
    --error="$PROJECT_DIR/logs/recombine_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/A1_T --panel A1_T && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/A1_S --panel A1_S && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/C1_T --panel C1_T && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/C1_S --panel C1_S && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/Biomax_T --panel Biomax_T && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/Biomax_S --panel Biomax_S && \
python scripts/recombine_raw.py --input-dir $PROJECT_DIR/output/batch_S --panel B1_S && \
echo 'All recombine done'")
echo "Recombine job: $RECOMB"

echo ""
echo "=== Step 2: Re-run per-TMA global analysis (depends on recombine) ==="

# A1 T-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=A1T-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/A1T_global2_%j.out" --error="$PROJECT_DIR/logs/A1T_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel T --input $PROJECT_DIR/output/A1_T/A1_T_raw_combined.h5ad --output $PROJECT_DIR/output/A1_T/TMA_A1_T_global.h5ad"
echo "A1 T-panel global: depends on $RECOMB"

# A1 S-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=A1S-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/A1S_global2_%j.out" --error="$PROJECT_DIR/logs/A1S_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/A1_S/A1_S_raw_combined.h5ad --output $PROJECT_DIR/output/A1_S/TMA_A1_S_global.h5ad"
echo "A1 S-panel global: depends on $RECOMB"

# C1 T-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=C1T-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/C1T_global2_%j.out" --error="$PROJECT_DIR/logs/C1T_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel T --input $PROJECT_DIR/output/C1_T/C1_T_raw_combined.h5ad --output $PROJECT_DIR/output/C1_T/TMA_C1_T_global.h5ad"
echo "C1 T-panel global: depends on $RECOMB"

# C1 S-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=C1S-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/C1S_global2_%j.out" --error="$PROJECT_DIR/logs/C1S_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/C1_S/C1_S_raw_combined.h5ad --output $PROJECT_DIR/output/C1_S/TMA_C1_S_global.h5ad"
echo "C1 S-panel global: depends on $RECOMB"

# Biomax T-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=BmT-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/BmaxT_global2_%j.out" --error="$PROJECT_DIR/logs/BmaxT_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel T --input $PROJECT_DIR/output/Biomax_T/Biomax_T_raw_combined.h5ad --output $PROJECT_DIR/output/Biomax_T/TMA_Biomax_T_global.h5ad"
echo "Biomax T-panel global: depends on $RECOMB"

# Biomax S-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=BmS-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/BmaxS_global2_%j.out" --error="$PROJECT_DIR/logs/BmaxS_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/Biomax_S/Biomax_S_raw_combined.h5ad --output $PROJECT_DIR/output/Biomax_S/TMA_Biomax_S_global.h5ad"
echo "Biomax S-panel global: depends on $RECOMB"

# B1 S-panel
sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=B1S-g2 --partition=scu-cpu --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output="$PROJECT_DIR/logs/B1S_global2_%j.out" --error="$PROJECT_DIR/logs/B1S_global2_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/batch_S/B1_S_raw_combined.h5ad --output $PROJECT_DIR/output/batch_S/TMA_B1_S_global.h5ad"
echo "B1 S-panel global: depends on $RECOMB"

echo ""
echo "=== Step 3: Cross-TMA global analysis (depends on recombine) ==="

# Cross-TMA T-panel (~2.2M cells)
CROSS_T=$(sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=xTMA-T --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/cross_tma_T_%j.out" --error="$PROJECT_DIR/logs/cross_tma_T_%j.err" \
    --wrap="eval \"\\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/cross_tma_global.py --panel T --base-dir $PROJECT_DIR --output $PROJECT_DIR/output/all_TMA_T_global.h5ad")
echo "Cross-TMA T-panel: $CROSS_T (depends on $RECOMB)"

# Cross-TMA S-panel (~2.0M cells)
CROSS_S=$(sbatch --parsable --dependency=afterok:$RECOMB \
    --job-name=xTMA-S --partition=scu-cpu --cpus-per-task=16 --mem=128G --time=2-00:00:00 \
    --output="$PROJECT_DIR/logs/cross_tma_S_%j.out" --error="$PROJECT_DIR/logs/cross_tma_S_%j.err" \
    --wrap="eval \"\\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/cross_tma_global.py --panel S --base-dir $PROJECT_DIR --output $PROJECT_DIR/output/all_TMA_S_global.h5ad")
echo "Cross-TMA S-panel: $CROSS_S (depends on $RECOMB)"

echo ""
echo "=== All jobs submitted ==="
echo "Recombine: $RECOMB (4h, sequential)"
echo "7 per-TMA global analysis jobs depend on recombine completing"
echo "2 cross-TMA global analysis jobs depend on recombine completing"
echo "Note: B1 T-panel already has correct raw data, no per-TMA re-run needed"
echo "Monitor: squeue -u <USER>"
