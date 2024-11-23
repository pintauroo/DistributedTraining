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

# Function to display error messages and exit
echo_error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
    exit 1
}

# ----------------------------
# Step 1: Clean Up Existing NVIDIA CUDA Repositories
# ----------------------------
echo_info "Removing any existing NVIDIA CUDA repository entries to prevent conflicts..."
sudo rm -f /etc/apt/sources.list.d/cuda*.list
sudo rm -f /etc/apt/sources.list.d/nvidia*.list

# ----------------------------
# Step 2: Install Essential Tools
# ----------------------------
echo_info "Installing essential tools: wget, curl, net-tools, git, gnupg, lsb-release..."
sudo apt-get update -y
sudo apt-get install -y wget curl net-tools git gnupg lsb-release || {
    echo_error "Failed to install essential tools. Please check your network connection."
}

# ----------------------------
# Step 3: Install NVIDIA Drivers (if not already installed)
# ----------------------------
DRIVER_VERSION="535"
echo_info "Checking for NVIDIA driver-${DRIVER_VERSION} installation..."
if ! dpkg -l | grep -q "nvidia-driver-${DRIVER_VERSION}"; then
    echo_info "Installing NVIDIA driver version ${DRIVER_VERSION}..."
    sudo apt-get install -y "nvidia-driver-${DRIVER_VERSION}" || {
        echo_error "Failed to install NVIDIA driver-${DRIVER_VERSION}."
    }
    echo_info "NVIDIA driver-${DRIVER_VERSION} installed successfully."
else
    echo_warning "NVIDIA driver-${DRIVER_VERSION} is already installed. Skipping."
fi

# ----------------------------
# Step 4: Add NVIDIA CUDA Repository and GPG Key
# ----------------------------
echo_info "Adding NVIDIA CUDA repository and GPG key..."

# Define CUDA repository details
CUDA_VERSION="11-8"
CUDA_REPO_PKG="cuda-repo-ubuntu2204-${CUDA_VERSION}-local_11.8.0-520.61.05-1_amd64.deb"
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/${CUDA_REPO_PKG}"
CUDA_REPO_PATH="/var/cuda-repo-ubuntu2204-${CUDA_VERSION}-local"

# Download the CUDA repository package if not already downloaded
if [ ! -f "/tmp/${CUDA_REPO_PKG}" ]; then
    echo_info "Downloading CUDA repository package from ${CUDA_REPO_URL}..."
    wget -q "${CUDA_REPO_URL}" -O "/tmp/${CUDA_REPO_PKG}" || {
        echo_error "Failed to download CUDA repository package from ${CUDA_REPO_URL}."
    }
else
    echo_warning "CUDA repository package already downloaded. Skipping download."
fi

# Install the CUDA repository package
echo_info "Installing CUDA repository package..."
sudo dpkg -i "/tmp/${CUDA_REPO_PKG}" || {
    echo_error "Failed to install CUDA repository package."
}

# Check for the presence of the keyring file
KEYRING_FILE=$(ls ${CUDA_REPO_PATH}/cuda-*-keyring.gpg 2>/dev/null || echo "")

if [ -f "${KEYRING_FILE}" ]; then
    echo_info "Adding CUDA GPG key..."
    sudo cp "${KEYRING_FILE}" /usr/share/keyrings/ || {
        echo_error "Failed to copy CUDA GPG key."
    }
else
    echo_warning "CUDA keyring file not found. Attempting to add the key manually..."
    # Manually add the NVIDIA GPG key
    wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg || {
        echo_error "Failed to download and add the NVIDIA GPG key."
    }
fi

# Add the CUDA repository to apt sources
echo_info "Adding CUDA repository to apt sources..."
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] file://${CUDA_REPO_PATH} /" | sudo tee /etc/apt/sources.list.d/cuda.list

# Update the package lists to include the new CUDA repository
echo_info "Updating package lists..."
sudo apt-get update -y

# Clean up the downloaded repository package to save space
echo_info "Removing downloaded CUDA repository package..."
sudo rm -f "/tmp/${CUDA_REPO_PKG}"

# ----------------------------
# Step 5: Install CUDA Toolkit 11.8
# ----------------------------
echo_info "Installing CUDA Toolkit ${CUDA_VERSION}..."
if ! dpkg -l | grep -q "cuda-toolkit-${CUDA_VERSION}"; then
    sudo apt-get install -y "cuda-toolkit-${CUDA_VERSION}" || {
        echo_error "Failed to install CUDA Toolkit ${CUDA_VERSION}."
    }
    echo_info "CUDA Toolkit ${CUDA_VERSION} installed successfully."
else
    echo_warning "CUDA Toolkit ${CUDA_VERSION} is already installed. Skipping."
fi

# ----------------------------
# Step 6: Set Environment Variables for CUDA
# ----------------------------
echo_info "Setting up environment variables for CUDA..."
CUDA_PROFILE_LINES="
# CUDA Toolkit 11.8
export PATH=/usr/local/cuda-11.8/bin\${PATH:+:\${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
"

# Append to ~/.bashrc if not already present
if ! grep -Fq "/usr/local/cuda-11.8/bin" ~/.bashrc; then
    echo -e "${CUDA_PROFILE_LINES}" >> ~/.bashrc
    echo_info "Environment variables for CUDA added to ~/.bashrc."
else
    echo_warning "Environment variables for CUDA already exist in ~/.bashrc. Skipping."
fi

# Source the updated .bashrc to apply changes immediately
echo_info "Sourcing ~/.bashrc to apply CUDA environment variables..."
source ~/.bashrc

# Verify CUDA installation
if command -v nvcc >/dev/null 2>&1; then
    echo_info "CUDA installation verified. nvcc version:"
    nvcc --version
else
    echo_error "nvcc command not found. CUDA installation might have failed."
fi

# ----------------------------
# Step 7: Install Miniconda (if not already installed)
# ----------------------------
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_SCRIPT="/tmp/miniconda.sh"

if [ ! -d "${MINICONDA_DIR}" ]; then
    echo_info "Downloading Miniconda installer..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${MINICONDA_SCRIPT}" || {
        echo_error "Failed to download Miniconda installer."
    }

    echo_info "Installing Miniconda..."
    bash "${MINICONDA_SCRIPT}" -b -u -p "${MINICONDA_DIR}" || {
        echo_error "Failed to install Miniconda."
    }

    echo_info "Removing Miniconda installer..."
    rm -f "${MINICONDA_SCRIPT}"
    echo_info "Miniconda installed successfully at ${MINICONDA_DIR}."
else
    echo_warning "Miniconda is already installed at ${MINICONDA_DIR}. Skipping installation."
fi

# ----------------------------
# Step 8: Initialize Conda
# ----------------------------
echo_info "Initializing Conda..."
# Initialize Conda for bash if not already initialized
if ! grep -Fq "source ${MINICONDA_DIR}/etc/profile.d/conda.sh" ~/.bashrc; then
    "${MINICONDA_DIR}/bin/conda" init bash || {
        echo_error "Failed to initialize Conda."
    }
    echo_info "Conda initialized in ~/.bashrc."
else
    echo_warning "Conda is already initialized in ~/.bashrc. Skipping initialization."
fi

# Source Conda for the current script
echo_info "Sourcing Conda..."
source "${MINICONDA_DIR}/etc/profile.d/conda.sh" || {
    echo_error "Failed to source Conda."
}

# ----------------------------
# Step 9: Clone the DistributedTraining Repository
# ----------------------------
REPO_URL="https://github.com/pintauroo/DistributedTraining.git"
REPO_DIR="$HOME/DistributedTraining"

if [ ! -d "${REPO_DIR}" ]; then
    echo_info "Cloning repository from ${REPO_URL}..."
    git clone "${REPO_URL}" "${REPO_DIR}" || {
        echo_error "Failed to clone repository."
    }
    echo_info "Repository cloned successfully to ${REPO_DIR}."
else
    echo_warning "Repository '${REPO_DIR}' already exists. Pulling latest changes..."
    cd "${REPO_DIR}" || {
        echo_error "Failed to navigate to repository directory."
    }
    git pull || {
        echo_error "Failed to update repository."
    }
fi

# Navigate to the repository directory
cd "${REPO_DIR}" || {
    echo_error "Failed to navigate to repository directory."
}

# ----------------------------
# Step 10: Create and Activate Conda Environment
# ----------------------------
ENV_NAME="pytrch"
ENV_FILE="environment.yml"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo_warning "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    if [ -f "${ENV_FILE}" ]; then
        echo_info "Creating Conda environment '${ENV_NAME}' from ${ENV_FILE}..."
        conda env create -f "${ENV_FILE}" -n "${ENV_NAME}" || {
            echo_error "Failed to create Conda environment '${ENV_NAME}'."
        }
        echo_info "Conda environment '${ENV_NAME}' created successfully."
    else
        echo_error "Environment file '${ENV_FILE}' not found in repository."
    fi
fi

echo_info "Activating Conda environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}" || {
    echo_error "Failed to activate Conda environment '${ENV_NAME}'."
}

# ----------------------------
# Step 11: Install NCCL Libraries
# ----------------------------
echo_info "Installing NCCL libraries..."
if ! dpkg -l | grep -q "libnccl2"; then
    sudo apt-get install -y libnccl2 libnccl-dev || {
        echo_error "Failed to install NCCL libraries."
    }
    echo_info "NCCL libraries installed successfully."
else
    echo_warning "NCCL libraries are already installed. Skipping."
fi

# ----------------------------
# Step 12: Install PyTorch with CUDA Support
# ----------------------------
echo_info "Installing PyTorch, torchvision, and torchaudio with CUDA support..."
if ! python -c "import torch" &> /dev/null; then
    pip install --upgrade pip || {
        echo_error "Failed to upgrade pip."
    }
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
        echo_error "Failed to install PyTorch with CUDA support."
    }
    echo_info "PyTorch with CUDA support installed successfully."
else
    echo_warning "PyTorch is already installed. Skipping."
fi

# ----------------------------
# Step 13: Final Setup and Reboot
# ----------------------------
echo_info "Setup complete. It is recommended to reboot the system to apply all changes."
read -p "Do you want to reboot now? (y/N): " REBOOT_CONFIRM
if [[ "$REBOOT_CONFIRM" =~ ^[Yy]$ ]]; then
    echo_info "Rebooting the system..."
    sudo reboot
else
    echo_warning "Reboot skipped. Please remember to reboot the system later to apply all changes."
fi
