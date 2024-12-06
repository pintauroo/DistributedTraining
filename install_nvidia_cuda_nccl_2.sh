# After reboot, we proceed with CUDA and PyTorch installation
echo "Verifying NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Ensure NVIDIA driver is installed and loaded."
    exit 1
fi
nvidia-smi

############################
# STEP 3: Install dependencies for CUDA repository
############################
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential dkms apt-transport-https ca-certificates curl gnupg software-properties-common

############################
# STEP 4: Add CUDA repository (using CUDA 11.8 for stability)
############################
echo "Adding CUDA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Instead of installing 'cuda-11-8' meta-package (which can bring in unwanted dependencies),
# we install just the toolkit:
############################
# STEP 5: Install CUDA Toolkit 11.8
############################
echo "Installing CUDA Toolkit 11.8..."
sudo apt-get install -y cuda-toolkit-11-8

# Install NCCL
echo "Installing NCCL libraries..."
sudo apt-get install -y libnccl2 libnccl-dev

############################
# STEP 6: Set environment variables for CUDA
############################
echo "Setting environment variables..."
if ! grep -q 'export PATH=/usr/local/cuda-11.8/bin:$PATH' ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
fi
if ! grep -q 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
source ~/.bashrc

############################
# STEP 7: Install Python, pip, and PyTorch with CUDA support
############################
echo "Installing Python, pip, and PyTorch with CUDA 11.8 support..."
sudo apt-get install -y python3-pip python3-dev
python3 -m pip install --upgrade pip
# Install PyTorch with CUDA 11.8 support:
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install psutil

############################
# STEP 8: Verification
############################
echo "Installation complete."
echo "Check with:"
echo "  nvidia-smi"
echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "If it prints 'True', you're good to go!"


