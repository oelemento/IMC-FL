#!/bin/bash
# Submit batch processing for all TMAs (except B1 which is already done)
# Usage: bash scripts/cayuga/submit_all_tmas.sh

set -e
PROJECT_DIR="<PROJECT_ROOT>"
DATA_ROOT="<DATA_ROOT>"

# Create output dirs and log dir
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/output/A1_T" "$PROJECT_DIR/output/A1_S"
mkdir -p "$PROJECT_DIR/output/C1_T" "$PROJECT_DIR/output/C1_S"
mkdir -p "$PROJECT_DIR/output/Biomax_T" "$PROJECT_DIR/output/Biomax_S"

echo "=== Submitting all TMA batch jobs ==="

# --- A1 T-panel (51 ROIs, 11 tasks of 5) ---
A1T=$(sbatch --parsable \
    --job-name=A1-T \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=06:00:00 \
    --array=0-10 \
    --output="$PROJECT_DIR/logs/A1T_%A_%a.out" \
    --error="$PROJECT_DIR/logs/A1T_%A_%a.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/batch_process.py --method hybrid --panel T --data-dir '$DATA_ROOT/March_11_2021_FL_TMA_A1_T' --output-dir '$PROJECT_DIR/output/A1_T' --start \$((SLURM_ARRAY_TASK_ID * 5)) --end \$((SLURM_ARRAY_TASK_ID * 5 + 5))")
echo "A1 T-panel: $A1T (array 0-10)"

# --- A1 S-panel (53 ROIs, 11 tasks of 5) ---
A1S=$(sbatch --parsable \
    --job-name=A1-S \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=06:00:00 \
    --array=0-10 \
    --output="$PROJECT_DIR/logs/A1S_%A_%a.out" \
    --error="$PROJECT_DIR/logs/A1S_%A_%a.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/batch_process_S.py --data-dir '$DATA_ROOT/March_9_2021_FL_TMA_A1_S' --output-dir '$PROJECT_DIR/output/A1_S' --start \$((SLURM_ARRAY_TASK_ID * 5)) --end \$((SLURM_ARRAY_TASK_ID * 5 + 5))")
echo "A1 S-panel: $A1S (array 0-10)"

# --- C1 T-panel (46 ROIs, 10 tasks of 5) ---
C1T=$(sbatch --parsable \
    --job-name=C1-T \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=06:00:00 \
    --array=0-9 \
    --output="$PROJECT_DIR/logs/C1T_%A_%a.out" \
    --error="$PROJECT_DIR/logs/C1T_%A_%a.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/batch_process.py --method hybrid --panel T --data-dir '$DATA_ROOT/Jan_28_2022_FL_TMA_C1_T' --output-dir '$PROJECT_DIR/output/C1_T' --start \$((SLURM_ARRAY_TASK_ID * 5)) --end \$((SLURM_ARRAY_TASK_ID * 5 + 5))")
echo "C1 T-panel: $C1T (array 0-9)"

# --- C1 S-panel (47 ROIs, 10 tasks of 5) ---
C1S=$(sbatch --parsable \
    --job-name=C1-S \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=06:00:00 \
    --array=0-9 \
    --output="$PROJECT_DIR/logs/C1S_%A_%a.out" \
    --error="$PROJECT_DIR/logs/C1S_%A_%a.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/batch_process_S.py --data-dir '$DATA_ROOT/Dec21_2021_FL_TMA_C1_S' --output-dir '$PROJECT_DIR/output/C1_S' --start \$((SLURM_ARRAY_TASK_ID * 5)) --end \$((SLURM_ARRAY_TASK_ID * 5 + 5))")
echo "C1 S-panel: $C1S (array 0-9)"

# --- Biomax T-panel (24 ROIs, 5 tasks of 5) ---
BT=$(sbatch --parsable \
    --job-name=Bmax-T \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=06:00:00 \
    --array=0-4 \
    --output="$PROJECT_DIR/logs/BmaxT_%A_%a.out" \
    --error="$PROJECT_DIR/logs/BmaxT_%A_%a.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/batch_process.py --method hybrid --panel T --data-dir '$DATA_ROOT/Biomax_T' --output-dir '$PROJECT_DIR/output/Biomax_T' --start \$((SLURM_ARRAY_TASK_ID * 5)) --end \$((SLURM_ARRAY_TASK_ID * 5 + 5))")
echo "Biomax T-panel: $BT (array 0-4)"

# --- Biomax S-panel (23 ROIs, 5 tasks of 5) ---
BS=$(sbatch --parsable \
    --job-name=Bmax-S \
    --partition=scu-cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=06:00:00 \
    --array=0-4 \
    --output="$PROJECT_DIR/logs/BmaxS_%A_%a.out" \
    --error="$PROJECT_DIR/logs/BmaxS_%A_%a.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/batch_process_S.py --data-dir '$DATA_ROOT/Biomax_S' --output-dir '$PROJECT_DIR/output/Biomax_S' --start \$((SLURM_ARRAY_TASK_ID * 5)) --end \$((SLURM_ARRAY_TASK_ID * 5 + 5))")
echo "Biomax S-panel: $BS (array 0-4)"

# --- Combine jobs (depend on batch completion) ---
echo ""
echo "=== Submitting combine jobs (afterany dependencies) ==="

sbatch --parsable --dependency=afterany:$A1T \
    --job-name=A1T-comb --partition=scu-cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --output="$PROJECT_DIR/logs/A1T_combine_%j.out" --error="$PROJECT_DIR/logs/A1T_combine_%j.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/combine_results.py --input-dir $PROJECT_DIR/output/A1_T --panel A1_T"
echo "A1 T-panel combine: depends on $A1T"

sbatch --parsable --dependency=afterany:$A1S \
    --job-name=A1S-comb --partition=scu-cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --output="$PROJECT_DIR/logs/A1S_combine_%j.out" --error="$PROJECT_DIR/logs/A1S_combine_%j.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/combine_results.py --input-dir $PROJECT_DIR/output/A1_S --panel A1_S"
echo "A1 S-panel combine: depends on $A1S"

sbatch --parsable --dependency=afterany:$C1T \
    --job-name=C1T-comb --partition=scu-cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --output="$PROJECT_DIR/logs/C1T_combine_%j.out" --error="$PROJECT_DIR/logs/C1T_combine_%j.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/combine_results.py --input-dir $PROJECT_DIR/output/C1_T --panel C1_T"
echo "C1 T-panel combine: depends on $C1T"

sbatch --parsable --dependency=afterany:$C1S \
    --job-name=C1S-comb --partition=scu-cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --output="$PROJECT_DIR/logs/C1S_combine_%j.out" --error="$PROJECT_DIR/logs/C1S_combine_%j.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/combine_results.py --input-dir $PROJECT_DIR/output/C1_S --panel C1_S"
echo "C1 S-panel combine: depends on $C1S"

sbatch --parsable --dependency=afterany:$BT \
    --job-name=BmT-comb --partition=scu-cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --output="$PROJECT_DIR/logs/BmaxT_combine_%j.out" --error="$PROJECT_DIR/logs/BmaxT_combine_%j.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/combine_results.py --input-dir $PROJECT_DIR/output/Biomax_T --panel Biomax_T"
echo "Biomax T-panel combine: depends on $BT"

sbatch --parsable --dependency=afterany:$BS \
    --job-name=BmS-comb --partition=scu-cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --output="$PROJECT_DIR/logs/BmaxS_combine_%j.out" --error="$PROJECT_DIR/logs/BmaxS_combine_%j.err" \
    --wrap="eval \"\$(<CONDA>/bin/conda shell.bash hook)\" && conda activate imc-fl && cd $PROJECT_DIR && python scripts/combine_results.py --input-dir $PROJECT_DIR/output/Biomax_S --panel Biomax_S"
echo "Biomax S-panel combine: depends on $BS"

echo ""
echo "=== All jobs submitted ==="
echo "6 batch arrays (61 total tasks) + 6 combine jobs"
echo "Monitor: squeue -u <USER>"
