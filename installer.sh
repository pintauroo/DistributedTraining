#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Define Miniconda installation directory
MINICONDA_DIR="$HOME/miniconda3"

# Create Miniconda directory if it doesn't exist
mkdir -p "$MINICONDA_DIR"

# Download the latest Miniconda installer for Linux (64-bit)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_DIR/miniconda.sh"

# Install Miniconda silently (-b), update if already installed (-u), specify installation path (-p)
bash "$MINICONDA_DIR/miniconda.sh" -b -u -p "$MINICONDA_DIR"

# Remove the installer script after installation
rm "$MINICONDA_DIR/miniconda.sh"

# Initialize Conda for the current shell session
source "$MINICONDA_DIR/bin/activate"

# Initialize Conda for all supported shells to ensure it's available in future sessions
conda init --all

# Create a new Conda environment named 'distributed_ps' without prompting (-y)
conda create -n distributed_ps -y

# Activate the newly created environment
conda activate distributed_ps

# Install required packages without prompting (-y)
conda install -y matplotlib pandas psutil

# Optional: Deactivate the environment after setup
# conda deactivate
