#!/bin/bash

subnet=$1
ip=$2

{

yes | sudo kubeadm reset

sudo kubeadm init --pod-network-cidr=${subnet} --apiserver-advertise-address=${ip}
sudo kubeadm init --pod-network-cidr=10.132.3.0/24 --apiserver-advertise-address=10.132.3.2
sudo kubeadm init --pod-network-cidr=192.168.0.0/16

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
kubectl get nodes

}  2>&1 | tee -a start_control_plane.log
