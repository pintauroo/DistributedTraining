#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display error messages
error_exit() {
    echo "Error: $1"
    exit 1
}

# Set non-interactive mode to avoid blocking prompts
export DEBIAN_FRONTEND=noninteractive

# Verify if the system has a CUDA-capable GPU
if ! lspci | grep -i nvidia; then
    error_exit "No NVIDIA GPU detected. Ensure your system has a CUDA-capable GPU."
fi

# Remove any existing NVIDIA installations
sudo apt purge -y nvidia* || error_exit "Failed to purge existing NVIDIA packages."
sudo apt autoremove -y && sudo apt autoclean -y || error_exit "Failed to autoremove or autoclean packages."
sudo rm -rf /usr/local/cuda* || error_exit "Failed to remove existing CUDA directories."

# Update the system
sudo apt update && sudo apt upgrade -y || error_exit "System update failed."

# Install essential packages, including ubuntu-drivers-common
sudo apt install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev ubuntu-drivers-common || error_exit "Failed to install essential packages."

# Add the graphics drivers PPA
sudo add-apt-repository -y ppa:graphics-drivers/ppa || error_exit "Failed to add graphics-drivers PPA."
sudo apt update || error_exit "Failed to update package list after adding PPA."

# Install the recommended NVIDIA driver
sudo ubuntu-drivers autoinstall || error_exit "Failed to install NVIDIA driver."

# Remove non-interactive mode to avoid issues for subsequent scripts
unset DEBIAN_FRONTEND

# Reboot the system to apply changes
echo "Rebooting the system to apply NVIDIA driver changes..."
sudo reboot now
