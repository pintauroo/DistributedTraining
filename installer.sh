#!/bin/bash

# -----------------------------------------------------------------------------
# Script Name: setup_distributed_training.sh
# Description: Automates the setup of a distributed PyTorch training environment
#              with optional NCCL support using Conda on Ubuntu 22.04 with Python 3.10.12.
# Author: OpenAI ChatGPT
# Date: 2024-04-27
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# ----------------------------- Configuration -----------------------------

# Name of the Conda environment
ENV_NAME="distributed_training"

# Python version
PYTHON_VERSION=3.10.12

# CUDA version (Ensure compatibility with your NVIDIA drivers)
CUDA_VERSION=11.8

# cuDNN version (Set to 8.5.0.32 as per available packages)
CUDNN_VERSION=8.8.0.121

# Miniconda installer URL
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Installation directory for Miniconda
MINICONDA_DIR="$HOME/miniconda3"

# Channels
PYTORCH_CHANNEL="pytorch"
NVIDIA_CHANNEL="nvidia"
CONDA_FORGE_CHANNEL="conda-forge"

# ------------------------- Function Definitions -------------------------

# Function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
}

# Function to install Miniconda
install_miniconda() {
    echo "=== Installing Miniconda ==="

    # Download Miniconda installer
    echo "Downloading Miniconda installer..."
    wget -O ~/miniconda.sh ${MINICONDA_URL}

    # Make the installer executable
    chmod +x ~/miniconda.sh

    # Run the installer silently
    echo "Running Miniconda installer..."
    bash ~/miniconda.sh -b -p ${MINICONDA_DIR}

    # Remove the installer
    rm ~/miniconda.sh

    # Initialize Conda
    echo "Initializing Conda..."
    ${MINICONDA_DIR}/bin/conda init bash

    # Source the Conda initialization script to make 'conda' available in this script
    echo "Sourcing Conda initialization script..."
    source ${MINICONDA_DIR}/etc/profile.d/conda.sh

    echo "Miniconda installation completed."
}

# Function to update existing Miniconda installation
update_miniconda() {
    echo "=== Updating Existing Miniconda Installation ==="
    # Source the Conda initialization script
    source ${MINICONDA_DIR}/etc/profile.d/conda.sh

    # Update Conda
    conda update -y conda

    echo "Miniconda update completed."
}

# Function to check if Miniconda is installed
check_miniconda_installed() {
    if [ -d "${MINICONDA_DIR}" ]; then
        echo "Miniconda is already installed at ${MINICONDA_DIR}."
        return 0
    else
        echo "Miniconda is not installed."
        return 1
    fi
}

# Function to check if Conda is available
check_conda_available() {
    if command_exists conda ; then
        echo "Conda is available."
        return 0
    else
        echo "Conda is not available in the current shell."
        return 1
    fi
}

# Function to create a new Conda environment
create_conda_env() {
    echo "=== Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} ==="

    # Check if the environment already exists
    if conda info --envs | grep -q "^${ENV_NAME} "; then
        echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
    else
        conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
        echo "Conda environment '${ENV_NAME}' created."
    fi
}

# Function to activate the Conda environment
activate_conda_env() {
    echo "=== Activating Conda environment '${ENV_NAME}' ==="
    # Initialize Conda in the current shell session
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
}

# Function to install CUDA Toolkit and cuDNN via Conda
install_cuda_cudnn() {
    echo "=== Installing CUDA Toolkit ${CUDA_VERSION} and cuDNN ${CUDNN_VERSION} ==="
    conda install -y -c nvidia -c conda-forge cudatoolkit=${CUDA_VERSION} cudnn=${CUDNN_VERSION}
    echo "CUDA Toolkit and cuDNN installation completed."
}

# Function to install NCCL via Conda
install_nccl() {
    echo "=== Installing NCCL ==="
    conda install -y -c nvidia -c conda-forge nccl
    echo "NCCL installation completed."
}

# Function to install PyTorch with CUDA support
install_pytorch_gpu() {
    echo "=== Installing PyTorch with CUDA ${CUDA_VERSION} support ==="
    conda install -y -c ${PYTORCH_CHANNEL} -c ${NVIDIA_CHANNEL} pytorch torchvision torchaudio cudatoolkit=${CUDA_VERSION}
    echo "PyTorch with CUDA installation completed."
}

# Function to install PyTorch CPU-only
install_pytorch_cpu() {
    echo "=== Installing PyTorch (CPU-only) ==="
    conda install -y -c ${PYTORCH_CHANNEL} pytorch torchvision torchaudio cpuonly
    echo "PyTorch CPU-only installation completed."
}

# Function to install additional dependencies
install_additional_dependencies() {
    echo "=== Installing additional Python packages: matplotlib, pandas, psutil ==="
    conda install -y -c conda-forge matplotlib pandas psutil
    echo "Additional dependencies installation completed."
}

# Function to verify installations
verify_installations() {
    echo "=== Verifying installations ==="

    echo "---- Verifying PyTorch and CUDA ----"
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

    echo "---- Verifying NCCL ----"
    python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"

    echo "---- Verifying other packages ----"
    python -c "import matplotlib; import pandas; import psutil; print('matplotlib version:', matplotlib.__version__); print('pandas version:', pandas.__version__); print('psutil version:', psutil.__version__)"
}

# Function to provide usage instructions
usage_instructions() {
    echo "------------------------------------------------------------"
    echo "Setup Complete!"
    echo "To activate the Conda environment, run:"
    echo "    conda activate ${ENV_NAME}"
    echo ""
    echo "To run your distributed training script, use 'torchrun'."
    echo "Example:"
    echo "    torchrun --nproc_per_node=NUM_GPUS script_name.py NUM_BATCHES BATCH_SIZE [--device cpu|gpu] [--verbose]"
    echo ""
    echo "Replace 'NUM_GPUS', 'script_name.py', 'NUM_BATCHES', and 'BATCH_SIZE' with appropriate values."
    echo "------------------------------------------------------------"
}

# Function to check and install NVIDIA drivers
install_nvidia_drivers() {
    echo "=== Checking NVIDIA Drivers ==="

    if command_exists nvidia-smi ; then
        echo "NVIDIA drivers are already installed."
        return 0
    else
        echo "NVIDIA drivers are not installed. Proceeding with installation."

        # Check if 'ubuntu-drivers' command is available
        if ! command_exists ubuntu-drivers ; then
            echo "'ubuntu-drivers' command not found. Installing 'ubuntu-drivers-common'..."
            sudo apt update
            sudo apt install -y ubuntu-drivers-common
        fi

        # Install recommended NVIDIA drivers
        echo "Installing recommended NVIDIA drivers..."
        sudo ubuntu-drivers autoinstall

        # Inform user to reboot
        echo "NVIDIA drivers installation is complete. A system reboot is required to activate the drivers."
        echo "Please reboot your system by running 'sudo reboot' and then re-run this setup script."
        exit 0
    fi
}

# ------------------------------ Main Script ------------------------------

# Step 1: Check if Miniconda is installed
if check_miniconda_installed ; then
    # Miniconda is installed
    # Step 2: Check if Conda is available
    if ! check_conda_available ; then
        echo "Sourcing Conda initialization script to make 'conda' available..."
        source ${MINICONDA_DIR}/etc/profile.d/conda.sh
    fi
else
    # Miniconda is not installed; proceed to install
    install_miniconda
fi

# Optional: Update Conda to the latest version
# Uncomment the following lines if you want to update Conda every time
# echo "Would you like to update Conda to the latest version? [y/N]"
# read -r response
# if [[ "$response" =~ ^[Yy]$ ]]; then
#     update_miniconda
# else
#     echo "Skipping Conda update."
# fi

# Step 2: Create Conda environment
create_conda_env

# Step 3: Activate Conda environment
activate_conda_env

# Step 4: Check for NVIDIA drivers and decide on GPU or CPU setup
if install_nvidia_drivers ; then
    # NVIDIA drivers are installed; proceed with GPU setup
    echo "Proceeding with GPU-based PyTorch installation."
    install_cuda_cudnn
    install_nccl
    install_pytorch_gpu
else
    # NVIDIA drivers are not installed; proceed with CPU-only setup
    echo "Proceeding with CPU-only PyTorch installation."
    install_pytorch_cpu
fi

# Step 5: Install additional dependencies
install_additional_dependencies

# Step 6: Verify installations
verify_installations

# Step 7: Provide usage instructions
usage_instructions

# Optional: Deactivate the environment after setup
# conda deactivate

# -----------------------------------------------------------------------------
