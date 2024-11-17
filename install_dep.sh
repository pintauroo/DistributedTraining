#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display informational messages
echo_info() {
    echo -e "\e[34m[INFO]\e[0m $1"
}

# Function to display warning messages
echo_warning() {
    echo -e "\e[33m[WARNING]\e[0m $1"
}

# Function to display error messages
echo_error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

# ----------------------------
# Step 1: Update Package Lists
# ----------------------------
echo_info "Updating package lists..."
sudo apt-get update -y

# ---------------------------------------
# Step 2: Install Essential Tools
# ---------------------------------------
echo_info "Installing essential tools: net-tools, git, wget, curl..."
sudo apt-get install -y net-tools git wget curl

# --------------------------------------------
# Step 3: Display Current Network Configuration
# --------------------------------------------
echo_info "Displaying current network configuration..."
ifconfig

# -----------------------------------
# Step 4: Install Miniconda
# -----------------------------------
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_SCRIPT="$MINICONDA_DIR/miniconda.sh"

if [ ! -d "$MINICONDA_DIR" ]; then
    echo_info "Creating directory for Miniconda at $MINICONDA_DIR..."
    mkdir -p "$MINICONDA_DIR"
    
    echo_info "Downloading Miniconda installer..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SCRIPT"
    
    echo_info "Installing Miniconda..."
    bash "$MINICONDA_SCRIPT" -b -u -p "$MINICONDA_DIR"
    
    echo_info "Removing Miniconda installer..."
    rm "$MINICONDA_SCRIPT"
else
    echo_warning "Miniconda is already installed at $MINICONDA_DIR. Skipping installation."
fi

# -----------------------------------
# Step 5: Initialize Conda
# -----------------------------------
# Check if Conda is initialized in .bashrc
if ! grep -Fxq "source $MINICONDA_DIR/etc/profile.d/conda.sh" ~/.bashrc; then
    echo_info "Initializing Conda..."
    "$MINICONDA_DIR/bin/conda" init
else
    echo_warning "Conda is already initialized in .bashrc. Skipping initialization."
fi

# Source Conda for the current script
echo_info "Sourcing Conda..."
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# -----------------------------------
# Step 6: Clone the DistributedTraining Repository
# -----------------------------------
REPO_URL="https://github.com/pintauroo/DistributedTraining.git"
REPO_DIR="$HOME/DistributedTraining"

if [ ! -d "$REPO_DIR" ]; then
    echo_info "Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo_warning "Repository '$REPO_DIR' already exists. Skipping clone."
fi

cd "$REPO_DIR"

# -----------------------------------
# Step 7: Create and Activate Conda Environment
# -----------------------------------
ENV_NAME="pytrch"
ENV_FILE="environment.yml"

if conda env list | grep -q "^$ENV_NAME"; then
    echo_warning "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo_info "Creating Conda environment '$ENV_NAME' from $ENV_FILE..."
    conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

echo_info "Activating Conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# -----------------------------------
# Step 8: Set Up NVIDIA CUDA Repository and GPG Key for Ubuntu 22.04
# -----------------------------------
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
CUDA_GPG_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub"
CUDA_LIST_FILE="/etc/apt/sources.list.d/cuda.list"
CUDA_PIN_FILE="/etc/apt/preferences.d/cuda-repository-pin-600"

echo_info "Adding NVIDIA CUDA repository pin..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -O cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin "$CUDA_PIN_FILE"

echo_info "Adding NVIDIA CUDA GPG key..."
sudo mkdir -p /usr/share/keyrings
wget "$CUDA_GPG_KEY_URL" -O cuda-key.pub
gpg --dearmor cuda-key.pub | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg > /dev/null
rm cuda-key.pub

echo_info "Adding NVIDIA CUDA repository to APT sources..."
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $CUDA_REPO_URL /" | sudo tee "$CUDA_LIST_FILE"

# Update package lists after adding CUDA repository
echo_info "Updating package lists after adding CUDA repository..."
sudo apt-get update -y

# -----------------------------------
# Step 9: Install NVIDIA Driver
# -----------------------------------
if ! dpkg -l | grep -q nvidia-driver-535; then
    echo_info "Installing NVIDIA driver version 535..."
    sudo apt-get install -y nvidia-driver-535
else
    echo_warning "NVIDIA driver-535 is already installed. Skipping."
fi

# -----------------------------------
# Step 10: Install CUDA Toolkit 11.8
# -----------------------------------
if ! dpkg -l | grep -q cuda-toolkit-11-8; then
    echo_info "Installing CUDA Toolkit 11.8..."
    sudo apt-get install -y cuda-toolkit-11-8
else
    echo_warning "CUDA Toolkit 11.8 is already installed. Skipping."
fi

# -----------------------------------
# Step 11: Install NCCL
# -----------------------------------
if ! dpkg -l | grep -q libnccl2; then
    echo_info "Installing NCCL libraries..."
    sudo apt-get install -y libnccl2=2.18.3-1+cuda11.8 libnccl-dev=2.18.3-1+cuda11.8
else
    echo_warning "NCCL libraries are already installed. Skipping."
fi

# -----------------------------------
# Step 12: Display Network Configuration After Installations
# -----------------------------------
echo_info "Displaying network configuration after installations..."
ifconfig

# -----------------------------------
# Step 13: Install PyTorch with CUDA Support
# -----------------------------------
if ! python -c "import torch" &> /dev/null; then
    echo_info "Installing PyTorch, torchvision, and torchaudio with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo_warning "PyTorch is already installed. Skipping."
fi

# -----------------------------------
# Step 14: Final Setup and Reboot
# -----------------------------------
echo_info "Setup complete. Rebooting the system to apply NVIDIA driver and CUDA installation..."
sudo reboot
