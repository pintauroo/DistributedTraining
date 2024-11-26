wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt install libnccl2=2.15.5-1+cuda11.8 libnccl-dev=2.15.5-1+cuda11.8