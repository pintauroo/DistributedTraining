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
# Step 0: Ensure the Script is Run as Root
# ----------------------------
if [ "$EUID" -ne 0 ]; then
    echo_error "Please run this script with sudo or as root."
fi

# ----------------------------
# Step 1: Define CUDA Version and Runfile
# ----------------------------
# CUDA Version and corresponding Runfile Installer
CUDA_VERSION="12.6.3"
CUDA_RUNFILE="cuda_${CUDA_VERSION}_560.35.05_linux.run"

# Define the download URL
CUDA_DOWNLOAD_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_RUNFILE}"

echo_info "Selected CUDA Version: ${CUDA_VERSION}"
echo_info "CUDA Runfile: ${CUDA_RUNFILE}"
echo_info "Download URL: ${CUDA_DOWNLOAD_URL}"

# ----------------------------
# Step 2: Clean Up Existing CUDA and NVIDIA Driver Installations
# ----------------------------
echo_info "Removing existing CUDA repositories and NVIDIA drivers..."

# Define an array of package patterns to purge
PACKAGES_TO_PURGE=(
    '^nvidia-.*'
    'cuda*'
    'libcudnn*'
    'libnccl*'
)

# Iterate over the package patterns and purge if installed
for pkg_pattern in "${PACKAGES_TO_PURGE[@]}"; do
    echo_info "Checking for installed packages matching pattern: ${pkg_pattern}"
    # List installed packages matching the pattern
    INSTALLED_PACKAGES=$(dpkg -l | grep -E "^ii\s+(${pkg_pattern//\*/.*})" | awk '{print $2}')

    if [ -n "$INSTALLED_PACKAGES" ]; then
        echo_info "Purging packages: $INSTALLED_PACKAGES"
        apt-get purge -y $INSTALLED_PACKAGES
    else
        echo_warning "No installed packages match the pattern: ${pkg_pattern}"
    fi
done

# Autoremove any residual dependencies
echo_info "Autoremoving unused packages..."
apt-get autoremove -y

echo_info "Cleanup of existing CUDA and NVIDIA drivers completed."

# ----------------------------
# Step 3: Install Essential Dependencies
# ----------------------------
echo_info "Installing essential tools: wget, curl, gnupg, build-essential, dkms..."

apt-get update -y
apt-get install -y wget curl gnupg build-essential dkms

# ----------------------------
# Step 4: Download the Correct CUDA Runfile Installer
# ----------------------------
echo_info "Downloading CUDA Runfile Installer (Version: ${CUDA_VERSION}) from ${CUDA_DOWNLOAD_URL}..."

# Download the CUDA Runfile Installer
wget "${CUDA_DOWNLOAD_URL}" -O "/tmp/${CUDA_RUNFILE}" || {
    echo_error "Failed to download CUDA Runfile Installer from ${CUDA_DOWNLOAD_URL}. Please verify the URL and your internet connection."
}

# ----------------------------
# Step 5: Make the Installer Executable
# ----------------------------
echo_info "Making the CUDA Runfile Installer executable..."
chmod +x "/tmp/${CUDA_RUNFILE}"

# ----------------------------
# Step 6: Run the CUDA Runfile Installer
# ----------------------------
echo_info "Running the CUDA Runfile Installer..."

# Install CUDA Toolkit and Samples without installing the driver
sh "/tmp/${CUDA_RUNFILE}" --silent --toolkit --samples --override || {
    echo_error "CUDA Runfile Installer failed. Please check the installer logs for details."
}

# ----------------------------
# Step 7: Set Up Environment Variables
# ----------------------------
echo_info "Setting up environment variables for CUDA..."

# Determine the installed CUDA version
CUDA_PATH=$(readlink -f /usr/local/cuda)
CUDA_VERSION_INSTALLED=$(basename "$CUDA_PATH" | sed 's/cuda-//')

echo_info "Installed CUDA Version: ${CUDA_VERSION_INSTALLED}"
echo_info "CUDA Path: ${CUDA_PATH}"

# Backup existing .bashrc if not already backed up
if [ ! -f ~/.bashrc.cuda_backup ]; then
    cp ~/.bashrc ~/.bashrc.cuda_backup
    echo_info "Backed up your original ~/.bashrc to ~/.bashrc.cuda_backup"
fi

# Add CUDA to PATH if not already present
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
    echo_info "Added CUDA PATH to ~/.bashrc"
else
    echo_warning "CUDA PATH already exists in ~/.bashrc. Skipping."
fi

# Add CUDA to LD_LIBRARY_PATH if not already present
if ! grep -q "/usr/local/cuda/lib64" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo_info "Added CUDA LD_LIBRARY_PATH to ~/.bashrc"
else
    echo_warning "CUDA LD_LIBRARY_PATH already exists in ~/.bashrc. Skipping."
fi

# Apply the changes immediately within the script
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ----------------------------
# Step 8: Create Symbolic Link /usr/local/cuda
# ----------------------------
echo_info "Creating symbolic link /usr/local/cuda pointing to /usr/local/cuda-${CUDA_VERSION_INSTALLED}..."

# Create or update the symbolic link
ln -sfn /usr/local/cuda-${CUDA_VERSION_INSTALLED} /usr/local/cuda

echo_info "Symbolic link /usr/local/cuda created/updated successfully."

# ----------------------------
# Step 9: Verify CUDA Installation
# ----------------------------
echo_info "Verifying CUDA installation..."

if command -v nvcc >/dev/null 2>&1; then
    echo_info "CUDA is installed successfully. Version details:"
    nvcc --version
else
    echo_error "nvcc command not found. CUDA installation might have failed."
fi

# ----------------------------
# Step 10: Install CUDA Samples and Run a Test
# ----------------------------
echo_info "Installing CUDA sample programs..."
cuda-install-samples-${CUDA_VERSION}.sh ~/
echo_info "Compiling and running the deviceQuery sample..."

cd ~/NVIDIA_CUDA-${CUDA_VERSION}_Samples/1_Utilities/deviceQuery || {
    echo_error "Failed to navigate to deviceQuery sample directory."
}

make || {
    echo_error "Failed to compile the deviceQuery sample."
}

./deviceQuery || {
    echo_error "deviceQuery test failed. Please check CUDA installation."
}

echo_info "deviceQuery test passed successfully."

# ----------------------------
# Step 11: Cleanup
# ----------------------------
echo_info "Cleaning up the CUDA Runfile Installer..."
rm -f "/tmp/${CUDA_RUNFILE}"
echo_info "Removed the CUDA Runfile Installer."

# ----------------------------
# Step 12: Final Reboot Prompt
# ----------------------------
read -p "Do you want to reboot the system now to apply all changes? (y/N): " REBOOT_CONFIRM

if [[ "$REBOOT_CONFIRM" =~ ^[Yy]$ ]]; then
    echo_info "Rebooting the system..."
    reboot
else
    echo_warning "Reboot skipped. Please remember to reboot the system later to apply all changes."
fi
