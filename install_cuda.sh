#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

### Step 1: Verify NVIDIA GPU Presence ###
echo "Checking for NVIDIA GPU..."
if ! lspci | grep -i nvidia >/dev/null; then
    echo "No NVIDIA GPU detected. Exiting."
    exit 1
fi
echo "NVIDIA GPU detected."

### Step 2: Clean Up Existing NVIDIA and CUDA Installations ###
echo "Removing existing NVIDIA and CUDA installations..."
sudo apt-get purge -y 'nvidia-*' 'cuda*' 'libnvidia*' 'libcuda*' || true
sudo rm -rf /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove -y
sudo apt-get autoclean -y
sudo rm -rf /usr/local/cuda*
echo "Cleanup completed."

### Step 3: System Update ###
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
echo "System update completed."

### Step 4: Install Required Packages ###
echo "Installing required packages..."
sudo apt-get install -y build-essential dkms freeglut3-dev gcc g++ \
libxi-dev libxmu-dev libglu1-mesa libglu1-mesa-dev wget curl git

### Step 5: Install Kernel Headers ###
echo "Installing kernel headers..."
sudo apt-get install -y linux-headers-$(uname -r)

### Step 6: Add Graphics Drivers PPA ###
echo "Adding graphics drivers PPA..."
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update -y

### Step 7: Identify and Install the Recommended NVIDIA Driver ###
echo "Identifying the recommended NVIDIA driver..."
RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep 'recommended' | awk '{print $3}')

if [ -z "$RECOMMENDED_DRIVER" ]; then
    echo "No recommended NVIDIA driver found. Exiting."
    exit 1
fi

echo "Installing the recommended NVIDIA driver: $RECOMMENDED_DRIVER..."
sudo apt-get install -y $RECOMMENDED_DRIVER

echo "NVIDIA driver installation completed."

### Step 8: Reboot System ###
echo "Rebooting system to load the new NVIDIA driver..."
echo "The system will reboot now. After reboot, please run the second script: install_cuda_cudnn_pytorch.sh"
sudo reboot
