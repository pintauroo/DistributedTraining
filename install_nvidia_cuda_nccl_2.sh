#!/usr/bin/env bash

set -e
set -o pipefail

# Verify that the NVIDIA driver is loaded
nvidia-smi || (echo "NVIDIA driver not loaded. Please reboot or check your driver installation." && exit 1)

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential dkms apt-transport-https ca-certificates curl gnupg software-properties-common

# Add NVIDIA CUDA repository key for Ubuntu 20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install a specific stable CUDA version (e.g. CUDA 11.8)
# You can check https://developer.nvidia.com/cuda-toolkit-archive for available versions
sudo apt-get install -y cuda-11-8

# Install NCCL
sudo apt-get install -y libnccl2 libnccl-dev

# Add CUDA paths to the environment (adjust if needed)
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install Python3, pip, and PyTorch with CUDA support
sudo apt-get install -y python3-pip python3-dev
python3 -m pip install --upgrade pip
# Install PyTorch with CUDA 11.8 support:
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: Install psutil
python3 -m pip install psutil

echo "Installation complete. CUDA, NCCL, and PyTorch (with CUDA support) are now installed."
echo "You can verify with 'nvidia-smi' and a Python session 'import torch; torch.cuda.is_available()'"
