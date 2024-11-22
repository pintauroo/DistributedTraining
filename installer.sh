#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ----------------------------
# Function Definitions
# ----------------------------

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
# Step 1: Clean Up Existing NVIDIA CUDA Repositories (if any)
# ----------------------------
echo_info "Removing any existing NVIDIA CUDA repository entries to prevent conflicts..."
sudo rm -f /etc/apt/sources.list.d/cuda*.list
sudo rm -f /etc/apt/sources.list.d/nvidia*.list

# ----------------------------
# Step 2: Install Essential Tools (wget and curl)
# ----------------------------
echo_info "Installing essential tools: wget, curl..."
# Update package lists to ensure availability
sudo apt-get update -y

# Install wget and curl if not already installed
sudo apt-get install -y wget curl || {
    echo_error "Failed to install wget and curl. Please check your network connection."
    exit 1
}

# ----------------------------
# Step 3: Install NVIDIA CUDA Keyring and Repository for Ubuntu 22.04
# ----------------------------
echo_info "Installing NVIDIA CUDA GPG keyring and adding repository..."

if ! dpkg -l | grep -q cuda-keyring; then
    # Download the cuda-keyring package
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    
    echo_info "Downloading cuda-keyring package from $CUDA_KEYRING_URL..."
    wget -q "$CUDA_KEYRING_URL" -O cuda-keyring.deb || {
        echo_error "Failed to download cuda-keyring package."
        exit 1
    }
    
    # Install the cuda-keyring package
    echo_info "Installing cuda-keyring package..."
    sudo dpkg -i cuda-keyring.deb || {
        echo_error "Failed to install cuda-keyring package."
        rm -f cuda-keyring.deb
        exit 1
    }
    
    # Remove the downloaded .deb file
    rm -f cuda-keyring.deb
    echo_info "cuda-keyring package installed successfully."
else
    echo_warning "cuda-keyring package is already installed. Skipping."
fi

# ----------------------------
# Step 4: Update Package Lists
# ----------------------------
echo_info "Updating package lists..."
sudo apt-get update -y

# ----------------------------
# Step 5: Install Full List of Essential Tools
# ----------------------------
echo_info "Installing essential tools: net-tools, git, gnupg, lsb-release..."
sudo apt-get install -y net-tools git gnupg lsb-release || {
    echo_error "Failed to install essential tools."
    exit 1
}

# ----------------------------
# Step 6: Display Current Network Configuration
# ----------------------------
echo_info "Displaying current network configuration..."
if command -v ifconfig >/dev/null 2>&1; then
    ifconfig
else
    echo_warning "ifconfig is not installed. Installing net-tools to provide ifconfig..."
    sudo apt-get install -y net-tools
    ifconfig
fi

# ----------------------------
# Step 7: Install Miniconda
# ----------------------------
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_SCRIPT="$HOME/miniconda.sh"

if [ ! -d "$MINICONDA_DIR" ]; then
    echo_info "Downloading Miniconda installer..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SCRIPT" || {
        echo_error "Failed to download Miniconda installer."
        exit 1
    }

    echo_info "Installing Miniconda..."
    bash "$MINICONDA_SCRIPT" -b -u -p "$MINICONDA_DIR" || {
        echo_error "Failed to install Miniconda."
        rm -f "$MINICONDA_SCRIPT"
        exit 1
    }

    echo_info "Removing Miniconda installer..."
    rm -f "$MINICONDA_SCRIPT"
    echo_info "Miniconda installed successfully at $MINICONDA_DIR."
else
    echo_warning "Miniconda is already installed at $MINICONDA_DIR. Skipping installation."
fi

# ----------------------------
# Step 8: Initialize Conda
# ----------------------------
# Check if Conda is initialized in .bashrc
if ! grep -Fxq "source $MINICONDA_DIR/etc/profile.d/conda.sh" ~/.bashrc; then
    echo_info "Initializing Conda..."
    "$MINICONDA_DIR/bin/conda" init || {
        echo_error "Failed to initialize Conda."
        exit 1
    }
else
    echo_warning "Conda is already initialized in .bashrc. Skipping initialization."
fi

# Source Conda for the current script
echo_info "Sourcing Conda..."
source "$MINICONDA_DIR/etc/profile.d/conda.sh" || {
    echo_error "Failed to source Conda."
    exit 1
}

# ----------------------------
# Step 9: Clone the DistributedTraining Repository
# ----------------------------
# REPO_URL="https://github.com/pintauroo/DistributedTraining.git"
REPO_DIR="$HOME/DistributedTraining"

# if [ ! -d "$REPO_DIR" ]; then
#     echo_info "Cloning repository from $REPO_URL..."
#     git clone "$REPO_URL" "$REPO_DIR" || {
#         echo_error "Failed to clone repository."
#         exit 1
#     }
#     echo_info "Repository cloned successfully to $REPO_DIR."
# else
#     echo_warning "Repository '$REPO_DIR' already exists. Skipping clone."
# fi

cd "$REPO_DIR"

# ----------------------------
# Step 10: Create and Activate Conda Environment
# ----------------------------
ENV_NAME="pytrch"
ENV_FILE="environment.yml"

if conda env list | grep -q "^$ENV_NAME"; then
    echo_warning "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    if [ -f "$ENV_FILE" ]; then
        echo_info "Creating Conda environment '$ENV_NAME' from $ENV_FILE..."
        conda env create -f "$ENV_FILE" -n "$ENV_NAME" || {
            echo_error "Failed to create Conda environment '$ENV_NAME'."
            exit 1
        }
        echo_info "Conda environment '$ENV_NAME' created successfully."
    else
        echo_error "Environment file '$ENV_FILE' not found in repository."
        exit 1
    fi
fi

echo_info "Activating Conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME" || {
    echo_error "Failed to activate Conda environment '$ENV_NAME'."
    exit 1
}

# ----------------------------
# Step 11: Install NVIDIA Driver
# ----------------------------
if ! dpkg -l | grep -q nvidia-driver-535; then
    echo_info "Installing NVIDIA driver version 535..."
    sudo apt-get install -y nvidia-driver-535 || {
        echo_error "Failed to install NVIDIA driver-535."
        exit 1
    }
    echo_info "NVIDIA driver-535 installed successfully."
else
    echo_warning "NVIDIA driver-535 is already installed. Skipping."
fi

# ----------------------------
# Step 12: Install CUDA Toolkit 11.8
# ----------------------------
if ! dpkg -l | grep -q cuda-toolkit-11-8; then
    echo_info "Installing CUDA Toolkit 11.8..."
    sudo apt-get install -y cuda-toolkit-11-8 || {
        echo_error "Failed to install CUDA Toolkit 11.8."
        exit 1
    }
    echo_info "CUDA Toolkit 11.8 installed successfully."
else
    echo_warning "CUDA Toolkit 11.8 is already installed. Skipping."
fi

# ----------------------------
# Step 13: Install NCCL Libraries
# ----------------------------
if ! dpkg -l | grep -q libnccl2; then
    echo_info "Installing NCCL libraries..."
    sudo apt-get install -y libnccl2 libnccl-dev || {
        echo_error "Failed to install NCCL libraries."
        exit 1
    }
    echo_info "NCCL libraries installed successfully."
else
    echo_warning "NCCL libraries are already installed. Skipping."
fi

# ----------------------------
# Step 14: Display Network Configuration After Installations
# ----------------------------
echo_info "Displaying network configuration after installations..."
if command -v ifconfig >/dev/null 2>&1; then
    ifconfig
else
    echo_warning "ifconfig is not installed. Installing net-tools to provide ifconfig..."
    sudo apt-get install -y net-tools
    ifconfig
fi

# ----------------------------
# Step 15: Install PyTorch with CUDA Support
# ----------------------------
if ! python -c "import torch" &> /dev/null; then
    echo_info "Installing PyTorch, torchvision, and torchaudio with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
        echo_error "Failed to install PyTorch with CUDA support."
        exit 1
    }
    echo_info "PyTorch with CUDA support installed successfully."
else
    echo_warning "PyTorch is already installed. Skipping."
fi

# ----------------------------
# Step 16: Final Setup and Reboot
# ----------------------------
echo_info "Setup complete. Rebooting the system to apply NVIDIA driver and CUDA installation..."
sudo reboot
