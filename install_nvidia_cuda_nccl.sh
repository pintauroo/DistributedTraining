#!/usr/bin/env bash
set -e
set -o pipefail

############################
# STEP 1: Clean up any old NVIDIA/CUDA installs
############################
sudo apt-get update
sudo apt-get -y upgrade

sudo apt-get remove --purge -y '^nvidia-.*' '^cuda-.*' || true
sudo apt-get autoremove -y
sudo apt-get autoclean
sudo apt-get update

############################
# STEP 2: Install NVIDIA driver
############################
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

echo "NVIDIA driver installation complete."
echo "It is strongly recommended to reboot now to ensure the driver is loaded correctly."
read -p "Press Enter to reboot now, or Ctrl+C to cancel." || true
sudo reboot
