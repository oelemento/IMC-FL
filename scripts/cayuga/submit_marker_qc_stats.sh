#!/bin/bash
#SBATCH --job-name=marker_qc_stats
#SBATCH --partition=scu-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=<PROJECT_ROOT>/logs/marker_qc_stats_%j.out
#SBATCH --error=<PROJECT_ROOT>/logs/marker_qc_stats_%j.err

source <CONDA>/etc/profile.d/conda.sh
conda activate imc-fl

cd <PROJECT_ROOT>
python3 scripts/compute_marker_qc_stats.py
