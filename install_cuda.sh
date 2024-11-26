#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# ------------------------------
# Function Definitions
# ------------------------------

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ------------------------------
# Step 1: Verify NVIDIA GPU Presence
# ------------------------------
log "Step 1: Checking for NVIDIA GPU..."

if ! command_exists lspci; then
    log "Error: lspci command not found. Please install pciutils."
    sudo apt-get install -y pciutils
fi

if ! lspci | grep -i nvidia >/dev/null; then
    log "No NVIDIA GPU detected. Exiting."
    exit 1
fi

log "NVIDIA GPU detected."

# ------------------------------
# Step 2: Clean Up Existing NVIDIA and CUDA Installations
# ------------------------------
log "Step 2: Removing existing NVIDIA and CUDA installations..."

sudo apt-get purge -y 'nvidia-*' 'cuda*' 'libnvidia*' 'libcuda*' || true
sudo rm -rf /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove -y
sudo apt-get autoclean -y
sudo rm -rf /usr/local/cuda*
log "Cleanup completed."

# ------------------------------
# Step 3: System Update
# ------------------------------
log "Step 3: Updating system packages..."

sudo apt-get update -y
sudo apt-get upgrade -y
log "System update completed."

# ------------------------------
# Step 4: Install Required Packages
# ------------------------------
log "Step 4: Installing required packages..."

sudo apt-get install -y build-essential dkms freeglut3-dev gcc g++ \
libxi-dev libxmu-dev libglu1-mesa libglu1-mesa-dev wget curl git \
ubuntu-drivers-common software-properties-common ca-certificates gnupg lsb-release

log "Required packages installation completed."

# ------------------------------
# Step 5: Install Kernel Headers
# ------------------------------
log "Step 5: Installing kernel headers..."

sudo apt-get install -y linux-headers-$(uname -r)

log "Kernel headers installation completed."

# ------------------------------
# Step 6: Add Graphics Drivers PPA
# ------------------------------
log "Step 6: Adding graphics drivers PPA..."

sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update -y

log "Graphics drivers PPA added and package lists updated."

# ------------------------------
# Step 7: Identify and Install the Recommended NVIDIA Driver
# ------------------------------
log "Step 7: Identifying the recommended NVIDIA driver..."

if ! command_exists ubuntu-drivers; then
    log "Error: ubuntu-drivers command not found even after installation. Exiting."
    exit 1
fi

RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep 'recommended' | awk '{print $3}')

if [ -z "$RECOMMENDED_DRIVER" ]; then
    log "No recommended NVIDIA driver found. Exiting."
    exit 1
fi

log "Recommended NVIDIA driver identified: $RECOMMENDED_DRIVER"

log "Installing the recommended NVIDIA driver: $RECOMMENDED_DRIVER..."

sudo apt-get install -y "$RECOMMENDED_DRIVER"

log "NVIDIA driver installation completed."

# ------------------------------
# Step 8: Install NVIDIA Modprobe
# ------------------------------
log "Step 8: Installing nvidia-modprobe..."

sudo apt-get install -y nvidia-modprobe

log "nvidia-modprobe installation completed."

# ------------------------------
# Step 9: Blacklist Nouveau Driver (Optional but Recommended)
# ------------------------------
log "Step 9: Blacklisting Nouveau driver..."

sudo bash -c "echo 'blacklist nouveau' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo update-initramfs -u

log "Nouveau driver blacklisted."

# ------------------------------
# Step 10: Reboot System
# ------------------------------
log "Step 10: Rebooting system to load the new NVIDIA driver..."

echo "The system will reboot now. After reboot, please run the second script: install_cuda_cudnn_pytorch.sh"
sudo reboot
