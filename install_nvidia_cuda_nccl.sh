#!/usr/bin/env bash

set -e  # Exit on any error
set -o pipefail

# Update and upgrade packages
sudo apt-get update
sudo apt-get -y upgrade

# Install necessary dependencies for building kernel modules and HTTPS support
sudo apt-get install -y build-essential dkms apt-transport-https ca-certificates curl gnupg software-properties-common

# -------------------------------------------------------------------------
# Install NVIDIA Driver, CUDA, and NCCL
# -------------------------------------------------------------------------

# Add NVIDIA package repository (for Ubuntu 20.04)
# This fetches the official CUDA repository key and sets up the repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install the recommended NVIDIA driver and CUDA toolkit
# This will install the latest stable CUDA and a compatible driver.
sudo apt-get -y install cuda

# The above 'cuda' package installs a driver and the CUDA toolkit.
# Verify driver after reboot with `nvidia-smi`.

# Install NCCL libraries
sudo apt-get install -y libnccl2 libnccl-dev

# Export paths for CUDA (Add these lines to ~/.bashrc or ~/.profile if you want persistence)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# -------------------------------------------------------------------------
# Install Python and PyTorch with GPU support
# -------------------------------------------------------------------------

# Install Python3, pip, and virtualenv if needed
sudo apt-get install -y python3-pip python3-dev
# Optional: Create a virtual environment (uncomment if you want isolation)
# python3 -m pip install --upgrade pip
# python3 -m pip install --user virtualenv
# python3 -m venv pytorch-env
# source pytorch-env/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support (adjust the CUDA version if needed)
# As of this writing, PyTorch provides wheels for CUDA 11.8 and others.
# Check https://pytorch.org/get-started/locally/ for the latest command.
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: Install other dependencies your script might need
python3 -m pip install psutil

# -------------------------------------------------------------------------
# Post-Installation Checks
# -------------------------------------------------------------------------

# Check that nvidia-smi works (requires a reboot if this is the first time installing drivers)
echo "If 'nvidia-smi' fails, consider rebooting the machine."
nvidia-smi || true

echo "Installation complete."
echo "You may need to reboot to fully load the NVIDIA driver."
