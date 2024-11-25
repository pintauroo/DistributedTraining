#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

### Step 8: Verify NVIDIA Driver Installation ###
echo "Verifying NVIDIA driver installation..."
if ! command_exists nvidia-smi; then
    echo "nvidia-smi not found. Please ensure the NVIDIA driver is installed correctly."
    exit 1
fi

nvidia-smi

### Step 9: Install CUDA Toolkit 11.8 ###
echo "Installing CUDA Toolkit 11.8..."

# Add CUDA repository GPG key
echo "Adding CUDA repository GPG key..."
wget -O cuda-repo-key.pub https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo apt-key add cuda-repo-key.pub
rm cuda-repo-key.pub

# Add CUDA repository
echo "Adding CUDA repository..."
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

# Install CUDA Toolkit 11.8
echo "Installing CUDA Toolkit 11.8..."
sudo apt-get install -y cuda-toolkit-11-8

echo "CUDA Toolkit 11.8 installation completed."

### Step 10: Set Up Environment Variables ###
echo "Setting up environment variables..."
CUDA_VERSION=11.8
CUDA_PATH=/usr/local/cuda-$CUDA_VERSION

# Backup existing .bashrc
cp ~/.bashrc ~/.bashrc.backup

# Add CUDA to PATH
if ! grep -q "$CUDA_PATH/bin" ~/.bashrc; then
    echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
    echo "Added CUDA bin to PATH."
fi

# Add CUDA libraries to LD_LIBRARY_PATH
if ! grep -q "$CUDA_PATH/lib64" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added CUDA lib64 to LD_LIBRARY_PATH."
fi

# Source the updated .bashrc
source ~/.bashrc

echo "Environment variables set up."

### Step 11: Install cuDNN 8.7 for CUDA 11.8 ###
echo "Installing cuDNN 8.7 for CUDA 11.8..."

# Prompt user to download cuDNN
echo "Please download cuDNN for CUDA 11.8 from NVIDIA's website:"
echo "https://developer.nvidia.com/rdp/cudnn-download"
echo "Ensure you have an NVIDIA Developer account. Place the cuDNN tar file (e.g., cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz) in the current directory."
read -p "Press Enter to continue after placing the cuDNN tar file..."

# Check for cuDNN tar file
CUDNN_TAR_FILE=$(ls cudnn-*-archive.tar.xz 2>/dev/null || true)
if [ -z "$CUDNN_TAR_FILE" ]; then
    echo "cuDNN tar file not found in the current directory. Exiting."
    exit 1
fi

echo "Extracting cuDNN files..."
tar -xf "$CUDNN_TAR_FILE"

# Identify extracted cuDNN folder
CUDNN_FOLDER=$(tar -tf "$CUDNN_TAR_FILE" | head -1 | cut -f1 -d"/")

echo "Copying cuDNN files to CUDA directories..."
sudo cp -P "$CUDNN_FOLDER/include/"* /usr/local/cuda-$CUDA_VERSION/include/
sudo cp -P "$CUDNN_FOLDER/lib/"* /usr/local/cuda-$CUDA_VERSION/lib64/
sudo chmod a+r /usr/local/cuda-$CUDA_VERSION/include/cudnn.h /usr/local/cuda-$CUDA_VERSION/lib64/libcudnn*

# Clean up cuDNN files
rm -rf "$CUDNN_FOLDER"
rm "$CUDNN_TAR_FILE"

echo "cuDNN installation completed."

### Step 12: Verify Installation ###
echo "Verifying CUDA and cuDNN installation..."
nvidia-smi
nvcc -V

echo "CUDA and cuDNN verification completed."

### Step 13: Install PyTorch with CUDA 11.8 Support ###
echo "Installing PyTorch with CUDA 11.8 support..."
# Ensure pip is installed
if ! command_exists pip; then
    echo "pip not found. Installing pip..."
    sudo apt-get install -y python3-pip
fi

# Upgrade pip
pip install --upgrade pip

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "PyTorch installation completed successfully."

echo "All installations completed successfully. Please restart your terminal or source your .bashrc to apply environment changes."
