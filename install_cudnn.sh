#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
export DEBIAN_FRONTEND=noninteractive

# Function to display error messages
error_exit() {
    echo "Error: $1"
    exit 1
}

# Verify NVIDIA driver installation
if ! nvidia-smi; then
    error_exit "NVIDIA driver installation failed or GPU not detected."
fi

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin || error_exit "Failed to download CUDA repository pin."
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 || error_exit "Failed to move CUDA repository pin."
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub || error_exit "Failed to fetch CUDA repository public key."
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" || error_exit "Failed to add CUDA repository."

# Update package lists
sudo apt update || error_exit "Failed to update package lists after adding CUDA repository."

# Install CUDA 11.8
sudo apt install -y cuda-11-8 || error_exit "Failed to install CUDA 11.8."

# Set up environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc || error_exit "Failed to source ~/.bashrc."
sudo ldconfig || error_exit "Failed to run ldconfig."

# Install cuDNN 9.5.1
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb || error_exit "Failed to download cuDNN installer."
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb || error_exit "Failed to install cuDNN repository."
sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/ || error_exit "Failed to copy cuDNN keyring."
sudo apt-get update || error_exit "Failed to update package list after adding cuDNN repository."
sudo apt-get -y install cudnn-cuda-11 || error_exit "Failed to install cuDNN for CUDA 11."

# Verify installations
if ! nvidia-smi; then
    error_exit "NVIDIA driver verification failed."
fi

if ! nvcc -V; then
    error_exit "CUDA installation verification failed."
fi

# Install PyTorch 2.0.0
pip install torch==2.0.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 || error_exit "Failed to install PyTorch 2.0.0."

echo "Installation completed successfully."
