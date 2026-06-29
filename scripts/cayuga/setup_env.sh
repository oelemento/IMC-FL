#!/bin/bash
# Setup conda environment on Cayuga for IMC-FL batch processing.
# Run once from login node: bash setup_env.sh

set -e

# Install miniconda on <SCRATCH> (not home dir, which has limited quota)
CONDA_DIR="<CONDA>"

if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda to $CONDA_DIR..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "Miniconda installed."
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

# Init conda
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# Create environment
ENV_NAME="imc-fl"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, updating..."
    conda activate "$ENV_NAME"
else
    echo "Creating environment $ENV_NAME..."
    conda create -n "$ENV_NAME" python=3.11 -y
    conda activate "$ENV_NAME"
fi

# Install PyTorch with CUDA (for Cellpose GPU)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Cellpose and dependencies
echo "Installing Cellpose and analysis packages..."
pip install cellpose scanpy anndata pandas numpy scipy scikit-image matplotlib tqdm leidenalg

echo ""
echo "=== Setup complete ==="
echo "Activate with:"
echo "  eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\""
echo "  conda activate $ENV_NAME"
