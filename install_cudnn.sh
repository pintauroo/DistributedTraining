#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Variables
CUDNN_DEB_URL="https://developer.download.nvidia.com/compute/cudnn/8.7.0/local_installers/cudnn-local-repo-ubuntu2204-8.7.0.84_1.0-1_amd64.deb"  # Update this URL to the correct and accessible .deb package
CUDNN_DEB_FILE="cudnn-local-repo-ubuntu2204-8.7.0.84_1.0-1_amd64.deb"  # Update the filename if different
CUDA_VERSION=11.8
CUDA_PATH=/usr/local/cuda-$CUDA_VERSION

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

### Step 1: Verify NVIDIA Driver Installation ###
echo "Verifying NVIDIA driver installation..."
if ! command_exists nvidia-smi; then
    echo "nvidia-smi not found. Please ensure the NVIDIA driver is installed correctly."
    exit 1
fi

echo "NVIDIA driver is installed correctly. Details:"
nvidia-smi

### Step 2: Install CUDA Toolkit 11.8 ###
echo "Installing CUDA Toolkit 11.8..."

# Add CUDA repository GPG key
echo "Adding CUDA repository GPG key..."
wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo apt-key add -

# Add CUDA repository
echo "Adding CUDA repository..."
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'

# Update package lists
echo "Updating package lists..."
sudo apt-get update -y

# Install CUDA Toolkit 11.8
echo "Installing CUDA Toolkit 11.8..."
sudo apt-get install -y cuda-toolkit-11-8

echo "CUDA Toolkit 11.8 installation completed."

### Step 3: Set Up Environment Variables ###
echo "Setting up environment variables for CUDA..."

# Backup existing .bashrc if not already backed up
if [ ! -f ~/.bashrc.backup ]; then
    cp ~/.bashrc ~/.bashrc.backup
    echo "Backup of ~/.bashrc created as ~/.bashrc.backup"
fi

# Add CUDA to PATH if not already present
if ! grep -q "$CUDA_PATH/bin" ~/.bashrc; then
    echo "export PATH=$CUDA_PATH/bin:\$PATH" | sudo tee -a ~/.bashrc
    echo "Added CUDA bin directory to PATH."
fi

# Add CUDA libraries to LD_LIBRARY_PATH if not already present
if ! grep -q "$CUDA_PATH/lib64" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" | sudo tee -a ~/.bashrc
    echo "Added CUDA lib64 directory to LD_LIBRARY_PATH."
fi

# Source the updated .bashrc to apply changes
echo "Applying environment variable changes..."
source ~/.bashrc

echo "Environment variables for CUDA set up successfully."

### Step 4: Install cuDNN 8.7 for CUDA 11.8 via .deb Package ###
echo "Installing cuDNN 8.7 for CUDA 11.8 via .deb package..."

# Download the cuDNN .deb package
echo "Downloading cuDNN .deb package from $CUDNN_DEB_URL..."
wget -O "$CUDNN_DEB_FILE" "$CUDNN_DEB_URL"

# Install the cuDNN local repo package
echo "Installing cuDNN local repo package..."
sudo dpkg -i "$CUDNN_DEB_FILE"

# Copy the GPG keyring
echo "Copying cuDNN GPG keyring..."
sudo cp /var/cudnn-local-repo-ubuntu2204-8.7.0.84/cudnn-*-keyring.gpg /usr/share/keyrings/

# Update package lists
echo "Updating package lists after adding cuDNN repo..."
sudo apt-get update -y

# Install cuDNN for CUDA 11
echo "Installing cuDNN for CUDA 11..."
sudo apt-get install -y libcudnn8 libcudnn8-dev libcudnn8-samples

# Optionally, install the CUDA-specific package if required
# echo "Installing CUDA-specific cuDNN package..."
# sudo apt-get install -y cudnn-cuda-11

# Clean up the .deb package
# echo "Removing the downloaded cuDNN .deb package..."
# rm "$CUDNN_DEB_FILE"

echo "cuDNN installation completed successfully."

### Step 5: Verify CUDA and cuDNN Installation ###
echo "Verifying CUDA and cuDNN installation..."

# Verify NVIDIA driver
echo "NVIDIA Driver Version:"
nvidia-smi

# Verify CUDA compiler
echo "CUDA Compiler Version:"
nvcc -V

# Verify cuDNN
echo "Verifying cuDNN version..."
CUDNN_VERSION=$(cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 | grep #define | awk '{print $3}')
echo "Installed cuDNN Version: $CUDNN_VERSION"

echo "CUDA and cuDNN verification completed successfully."

### Step 6: Install PyTorch with CUDA 11.8 Support ###
echo "Installing PyTorch with CUDA 11.8 support..."

# Ensure pip is installed
if ! command_exists pip3; then
    echo "pip3 not found. Installing pip3..."
    sudo apt-get install -y python3-pip
fi

# Upgrade pip
echo "Upgrading pip..."
sudo pip3 install --upgrade pip

# Install PyTorch, torchvision, torchaudio with CUDA 11.8 support
echo "Installing PyTorch, torchvision, and torchaudio..."
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "PyTorch installation completed successfully."

echo "All installations completed successfully. Please restart your terminal or source your ~/.bashrc to apply environment changes."
