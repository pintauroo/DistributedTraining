#!/bin/bash

{

sudo apt update
sudo apt install -y docker.io apt-transport-https curl



curl -sS https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor | sudo tee /usr/share/keyrings/google-cloud-apt.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/google-cloud-apt.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list   
sudo apt update -y
sudo apt install google-cloud-sdk -y

# Step 3: Add Kubernetes APT repository
echo "Adding Kubernetes APT repository..."
sudo rm /etc/apt/sources.list.d/kubernetes.list

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list

#cat /etc/apt/sources.list.d/kubernetes.list

sudo apt update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

sudo swapoff -a

sudo sh -c "echo {                                                  >  /etc/docker/daemon.json"
sudo sh -c 'echo \"exec-opts\": [\"native.cgroupdriver=systemd\"]  >>  /etc/docker/daemon.json'
sudo sh -c "echo }                                                 >>  /etc/docker/daemon.json"


sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker

}  2>&1 | tee -a config_worker_node.log
