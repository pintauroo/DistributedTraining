#!/bin/bash

# -----------------------------------------------------------------------------
# Script Name: setup_minimal_distributed_training.sh
# Description: Sets up a minimal distributed PyTorch training environment.
#              Assumes CUDA, cuDNN, and PyTorch are already installed.
# -----------------------------------------------------------------------------

# Exit on error
set -e

# Configuration
ENV_NAME="distributed_training"
PYTHON_VERSION=3.10
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_DIR="$HOME/miniconda3"

# Function to install Miniconda if not already installed
install_miniconda() {
    if [ ! -d "${MINICONDA_DIR}" ]; then
        echo "Installing Miniconda..."
        wget -O ~/miniconda.sh ${MINICONDA_URL}
        chmod +x ~/miniconda.sh
        bash ~/miniconda.sh -b -p ${MINICONDA_DIR}
        rm ~/miniconda.sh
        source ${MINICONDA_DIR}/etc/profile.d/conda.sh
    else
        echo "Miniconda is already installed."
        source ${MINICONDA_DIR}/etc/profile.d/conda.sh
    fi
}

# Function to create and activate Conda environment
setup_conda_env() {
    if ! conda info --envs | grep -q "^${ENV_NAME} "; then
        echo "Creating Conda environment '${ENV_NAME}'..."
        conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
    else
        echo "Conda environment '${ENV_NAME}' already exists."
    fi
    echo "Activating Conda environment '${ENV_NAME}'..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
}

# Function to install additional dependencies
install_dependencies() {
    echo "Installing additional Python packages..."
    pip install matplotlib pandas psutil
    echo "Dependencies installed."
}

# Function to verify PyTorch GPU availability
verify_pytorch() {
    echo "Verifying PyTorch installation..."
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
}

# Main Script
install_miniconda
setup_conda_env
install_dependencies
verify_pytorch

echo "Setup complete! To activate the Conda environment, run:"
echo "    conda activate ${ENV_NAME}"
