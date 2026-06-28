#!/bin/bash
#SBATCH --job-name=imc-fl-batch
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=<PROJECT_ROOT>/logs/batch_%j.out
#SBATCH --error=<PROJECT_ROOT>/logs/batch_%j.err

# ---- Setup ----
CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"
DATA_DIR="<DATA_ROOT>/Jan 18 2022_FL_TMA_B1_T"

# Create log directory
mkdir -p "$PROJECT_DIR/logs"

# Activate conda
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl

# Confirm GPU is visible
echo "=== GPU Info ==="
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo ""

# ---- Run batch processing ----
cd "$PROJECT_DIR"

# Process T-panel with hybrid method (GPU-accelerated Cellpose)
python scripts/batch_process.py \
    --method hybrid \
    --panel T \
    --data-dir "$DATA_DIR" \
    --output-dir "$PROJECT_DIR/output/batch" \
    --gpu

echo ""
echo "=== Job complete ==="
