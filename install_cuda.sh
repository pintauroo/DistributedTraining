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
    log "Error: lspci command not found. Installing pciutils..."
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
# Step 6: Install NVIDIA Driver and Components
# ------------------------------
log "Step 6: Installing NVIDIA driver version 535-server and components..."

sudo apt-get install -y nvidia-driver-535-server

log "NVIDIA driver version 535-server installed successfully."

# ------------------------------
# Step 7: Blacklist Nouveau Driver (Optional but Recommended)
# ------------------------------
log "Step 7: Blacklisting Nouveau driver..."

sudo bash -c "echo 'blacklist nouveau' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo update-initramfs -u

log "Nouveau driver blacklisted."

# ------------------------------
# Step 8: Reboot System
# ------------------------------
log "Step 8: Rebooting system to load the new NVIDIA driver..."

echo "The system will reboot now. After reboot, ensure the NVIDIA driver is installed properly by running 'nvidia-smi'."
sudo reboot
