#!/usr/bin/env bash

set -e
set -o pipefail

# Update and upgrade packages
sudo apt-get update
sudo apt-get -y upgrade

# Remove any existing NVIDIA or CUDA packages to avoid conflicts
sudo apt-get remove --purge -y '^nvidia-.*' '^cuda-.*' || true
sudo apt-get autoremove -y
sudo apt-get autoclean

# Update again after purge
sudo apt-get update

# Install required packages for ubuntu-drivers and other tools
sudo apt-get install -y ubuntu-drivers-common

# Autoinstall the recommended NVIDIA driver
sudo ubuntu-drivers autoinstall

echo "NVIDIA driver installed. Please reboot your system now."
echo "After reboot, run the second part of the script to install CUDA, NCCL, and PyTorch."
