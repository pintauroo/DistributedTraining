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
# Step 2: Install Essential Tools (wget, curl, etc.)
# ----------------------------
echo_info "Installing essential tools: wget, curl, net-tools, git, gnupg, lsb-release..."
sudo apt-get update -y
sudo apt-get install -y wget curl net-tools git gnupg lsb-release || {
    echo_error "Failed to install essential tools. Please check your network connection."
    exit 1
}

# ----------------------------
# Step 3: Install NVIDIA Drivers (if not already installed)
# ----------------------------
echo_info "Checking for NVIDIA driver-535 installation..."
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
# Step 4: Download and Install CUDA Toolkit 11.8 via Runfile Installer
# ----------------------------
CUDA_VERSION="11.8.0"
CUDA_RUNFILE="cuda_${CUDA_VERSION}_520.61.05_linux.run"
CUDA_INSTALL_DIR="/usr/local/cuda-11.8"

# Check if CUDA Toolkit 11.8 is already installed
if [ ! -d "$CUDA_INSTALL_DIR" ]; then
    echo_info "Downloading CUDA Toolkit ${CUDA_VERSION} Runfile Installer..."
    wget -q https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_RUNFILE} -O "/tmp/${CUDA_RUNFILE}" || {
        echo_error "Failed to download CUDA Toolkit ${CUDA_VERSION} Runfile."
        exit 1
    }

    echo_info "Making the CUDA Runfile executable..."
    chmod +x "/tmp/${CUDA_RUNFILE}"

    echo_info "Installing CUDA Toolkit ${CUDA_VERSION} (without driver)..."
    sudo sh "/tmp/${CUDA_RUNFILE}" --silent --toolkit --override || {
        echo_error "Failed to install CUDA Toolkit ${CUDA_VERSION}."
        rm -f "/tmp/${CUDA_RUNFILE}"
        exit 1
    }

    echo_info "CUDA Toolkit ${CUDA_VERSION} installed successfully at ${CUDA_INSTALL_DIR}."

    # Remove the Runfile installer
    rm -f "/tmp/${CUDA_RUNFILE}"
else
    echo_warning "CUDA Toolkit 11.8 is already installed at ${CUDA_INSTALL_DIR}. Skipping installation."
fi

# ----------------------------
# Step 5: Set Environment Variables for CUDA
# ----------------------------
echo_info "Setting up environment variables for CUDA..."
CUDA_PROFILE_LINE='export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

# Add to .bashrc if not already present
if ! grep -Fq "/usr/local/cuda-11.8/bin" ~/.bashrc; then
    echo -e "$CUDA_PROFILE_LINE" >> ~/.bashrc
    echo_info "Environment variables for CUDA added to ~/.bashrc."
else
    echo_warning "Environment variables for CUDA already exist in ~/.bashrc. Skipping."
fi

# Source the updated .bashrc
echo_info "Sourcing ~/.bashrc to apply CUDA environment variables..."
source ~/.bashrc

# Verify CUDA installation
if ! command -v nvcc >/dev/null 2>&1; then
    echo_error "nvcc command not found. CUDA installation might have failed."
    exit 1
else
    echo_info "CUDA installation verified. nvcc version:"
    nvcc --version
fi

# ----------------------------
# Step 6: Install Miniconda (if not already installed)
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
# Step 7: Initialize Conda
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
# Step 8: Clone the DistributedTraining Repository
# ----------------------------
REPO_URL="https://github.com/pintauroo/DistributedTraining.git"
REPO_DIR="$HOME/DistributedTraining"

if [ ! -d "$REPO_DIR" ]; then
    echo_info "Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" "$REPO_DIR" || {
        echo_error "Failed to clone repository."
        exit 1
    }
    echo_info "Repository cloned successfully to $REPO_DIR."
else
    echo_warning "Repository '$REPO_DIR' already exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull || {
        echo_error "Failed to update repository."
        exit 1
    }
fi

cd "$REPO_DIR"

# ----------------------------
# Step 9: Create and Activate Conda Environment
# ----------------------------
ENV_NAME="pytrch"
ENV_FILE="environment.yml"

if conda env list | grep -q "^${ENV_NAME} "; then
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
# Step 10: Install NCCL Libraries
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
# Step 11: Install PyTorch with CUDA Support
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
# Step 12: Final Setup and Reboot
# ----------------------------
echo_info "Setup complete. It is recommended to reboot the system to apply all changes."
read -p "Do you want to reboot now? (y/N): " REBOOT_CONFIRM
if [[ "$REBOOT_CONFIRM" =~ ^[Yy]$ ]]; then
    echo_info "Rebooting the system..."
    sudo reboot
else
    echo_warning "Reboot skipped. Please remember to reboot the system later to apply all changes."
fi
