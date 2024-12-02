#!/bin/bash

ip=$1

{

echo ${ip}

yes | sudo kubeadm reset

sudo kubeadm join ${ip}:6443 --token b3sfn1.o6lwqean8csvd6ui --discovery-token-ca-cert-hash   sha256:a473c0a0ee8641d477f324a4bfe98dcba02ae4f36fff449532aa31f2d9a20167

}  2>&1 | tee -a start_worker_node.log
