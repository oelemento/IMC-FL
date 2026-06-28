#!/bin/bash
# Submit global analysis for all TMAs (except B1 which is already done/running)
# Usage: bash scripts/cayuga/submit_global_all.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
CONDA_DIR="<CONDA>"

echo "=== Submitting global analysis jobs ==="

# Common SLURM params
PARTITION="scu-cpu"
CPUS=8
MEM="64G"
TIME="24:00:00"

# --- A1 T-panel ---
sbatch --parsable \
    --job-name=A1T-glob \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --output="$PROJECT_DIR/logs/A1T_global_%j.out" \
    --error="$PROJECT_DIR/logs/A1T_global_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel T --input $PROJECT_DIR/output/A1_T/TMA_B1_A1_T_combined.h5ad --output $PROJECT_DIR/output/A1_T/TMA_A1_T_global.h5ad"
echo "Submitted A1 T-panel global"

# --- A1 S-panel ---
sbatch --parsable \
    --job-name=A1S-glob \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --output="$PROJECT_DIR/logs/A1S_global_%j.out" \
    --error="$PROJECT_DIR/logs/A1S_global_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/A1_S/TMA_B1_A1_S_combined.h5ad --output $PROJECT_DIR/output/A1_S/TMA_A1_S_global.h5ad"
echo "Submitted A1 S-panel global"

# --- C1 T-panel ---
sbatch --parsable \
    --job-name=C1T-glob \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --output="$PROJECT_DIR/logs/C1T_global_%j.out" \
    --error="$PROJECT_DIR/logs/C1T_global_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel T --input $PROJECT_DIR/output/C1_T/TMA_B1_C1_T_combined.h5ad --output $PROJECT_DIR/output/C1_T/TMA_C1_T_global.h5ad"
echo "Submitted C1 T-panel global"

# --- C1 S-panel ---
sbatch --parsable \
    --job-name=C1S-glob \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --output="$PROJECT_DIR/logs/C1S_global_%j.out" \
    --error="$PROJECT_DIR/logs/C1S_global_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/C1_S/TMA_B1_C1_S_combined.h5ad --output $PROJECT_DIR/output/C1_S/TMA_C1_S_global.h5ad"
echo "Submitted C1 S-panel global"

# --- Biomax T-panel ---
sbatch --parsable \
    --job-name=BmxT-glob \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --output="$PROJECT_DIR/logs/BmaxT_global_%j.out" \
    --error="$PROJECT_DIR/logs/BmaxT_global_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel T --input $PROJECT_DIR/output/Biomax_T/TMA_B1_Biomax_T_combined.h5ad --output $PROJECT_DIR/output/Biomax_T/TMA_Biomax_T_global.h5ad"
echo "Submitted Biomax T-panel global"

# --- Biomax S-panel ---
sbatch --parsable \
    --job-name=BmxS-glob \
    --partition=$PARTITION \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --output="$PROJECT_DIR/logs/BmaxS_global_%j.out" \
    --error="$PROJECT_DIR/logs/BmaxS_global_%j.err" \
    --wrap="eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/global_analysis.py --panel S --input $PROJECT_DIR/output/Biomax_S/TMA_B1_Biomax_S_combined.h5ad --output $PROJECT_DIR/output/Biomax_S/TMA_Biomax_S_global.h5ad"
echo "Submitted Biomax S-panel global"

echo ""
echo "=== All 6 global analysis jobs submitted ==="
echo "Each: 8 CPUs, 64GB RAM, 24h wall time"
echo "Monitor: squeue -u ole2001"
