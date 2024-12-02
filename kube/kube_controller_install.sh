#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# {
echo "=============================="
echo "Starting Kubernetes Setup"
echo "Date: $(date)"
echo "=============================="

# Step 1: Update package list and install prerequisites
echo "Updating package list and installing prerequisites..."
sudo apt update -y
sudo apt install -y docker.io apt-transport-https curl gnupg lsb-release ca-certificates

# Step 2: Add Kubernetes APT repository GPG key

# curl -sS https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor | sudo tee /usr/share/keyrings/google-cloud-apt.gpg > /dev/null
# echo "deb [signed-by=/usr/share/keyrings/google-cloud-apt.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list   
# sudo apt update -y
# sudo apt install google-cloud-sdk -y

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Step 3: Add Kubernetes APT repository
echo "Adding Kubernetes APT repository..."
sudo rm /etc/apt/sources.list.d/kubernetes.list 2>/dev/null &

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
# curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
sudo chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
# echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo chmod 644 /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update

# Step 4: Update package list and install Kubernetes components
echo "Updating package list..."
sudo apt update -y

echo "Installing kubelet, kubeadm, and kubectl..."
sudo apt-get install -y kubelet kubeadm kubectl

echo "Holding Kubernetes packages at current version..."
sudo apt-mark hold kubelet kubeadm kubectl

# Step 5: Disable swap
echo "Disabling swap..."
sudo swapoff -a
sudo sh -c "echo {                                                  >  /etc/docker/daemon.json"
sudo sh -c 'echo \"exec-opts\": [\"native.cgroupdriver=systemd\"]  >>  /etc/docker/daemon.json'
sudo sh -c "echo }                                                 >>  /etc/docker/daemon.json"


sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker

#     # Step 6: Configure Docker to use systemd as the cgroup driver
#     echo "Configuring Docker to use systemd as the cgroup driver..."
#     sudo mkdir -p /etc/docker
#     sudo tee /etc/docker/daemon.json > /dev/null <<EOF
# {
#   "exec-opts": ["native.cgroupdriver=systemd"],
#   "log-driver": "json-file",
#   "log-opts": {
#     "max-size": "100m"
#   },
#   "storage-driver": "overlay2"
# }
# EOF

#     # Step 7: Enable and restart Docker
#     echo "Enabling and restarting Docker..."
#     sudo systemctl enable docker
#     sudo systemctl daemon-reload
#     sudo systemctl restart docker

#     echo "=============================="
#     echo "Kubernetes Setup Completed Successfully"
#     echo "=============================="

# } 2>&1 | tee -a config_control_plane.log
