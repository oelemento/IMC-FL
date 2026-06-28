#!/bin/bash
#SBATCH --job-name=debug-cp
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=<PROJECT_ROOT>/logs/debug_%j.out
#SBATCH --error=<PROJECT_ROOT>/logs/debug_%j.err

CONDA_DIR="<CONDA>"
PROJECT_DIR="<PROJECT_ROOT>"

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate imc-fl
cd "$PROJECT_DIR"

python -c "
import sys, traceback
sys.path.insert(0, '.')
from src.data_loader import load_roi_txt, list_rois
from src.segmentation import segment_roi
from pathlib import Path

DATA_DIR = Path('<DATA_ROOT>/Jan 18 2022_FL_TMA_B1_T')
files = list_rois(DATA_DIR)
f = files[0]
print(f'Loading {f.name}...')
image, markers, metadata = load_roi_txt(f)
print(f'Image: {image.shape}')
try:
    masks, adata = segment_roi(image, markers, 'FL01', method='hybrid', flow_threshold=0.8, min_distance=2, sigma=1.0)
    print(f'Success: {adata.n_obs} cells')
except Exception:
    traceback.print_exc()
"
