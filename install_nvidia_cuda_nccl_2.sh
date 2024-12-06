#!/usr/bin/env bash
set -e
set -o pipefail

############################
# STEP 3: Verify driver and proceed with CUDA installation
############################
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Ensure NVIDIA driver is installed and loaded, then re-run this script."
    exit 1
fi

echo "Verifying NVIDIA driver..."
nvidia-smi

# The above should show the GPU and driver info
# If everything looks good, proceed.

############################
# STEP 4: Install dependencies for CUDA repository
############################
sudo apt-get update
sudo apt-get install -y build-essential dkms apt-transport-https ca-certificates curl gnupg software-properties-common

############################
# STEP 5: Add CUDA repository and install CUDA 12.2
############################
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA 12.2 toolkit
sudo apt-get install -y cuda-12-2

# Install NCCL
sudo apt-get install -y libnccl2 libnccl-dev

############################
# STEP 6: Set environment variables for CUDA
############################
if ! grep -q 'export PATH=/usr/local/cuda-12.2/bin:$PATH' ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
fi
if ! grep -q 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
source ~/.bashrc

############################
# STEP 7: Install Python, pip, and PyTorch with CUDA support
############################
sudo apt-get install -y python3-pip python3-dev
python3 -m pip install --upgrade pip

# As of now, PyTorch official wheels are often provided for CUDA 11.8.
# Install PyTorch for CUDA 11.8:
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install psutil

############################
# STEP 8: Verification
############################
echo "Installation complete."
echo "Please verify your setup by running:"
echo "  nvidia-smi"
echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"

echo "If the above command prints 'True', you are ready to run your training code."
