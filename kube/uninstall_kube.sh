#!/bin/bash

# Function to check for root permissions
check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "This script must be run as root. Use sudo." >&2
        exit 1
    fi
}

# Function to uninstall Kubernetes components
uninstall_kubernetes() {
    echo "Resetting Kubernetes using kubeadm..."
    kubeadm reset -f

    echo "Stopping and disabling kubelet service..."
    systemctl stop kubelet
    systemctl disable kubelet

    echo "Removing Kubernetes packages..."
    apt-get purge -y kubeadm kubelet kubectl kubernetes-cni
    apt-get autoremove -y

    echo "Cleaning up Kubernetes configuration and data..."
    rm -rf /etc/kubernetes/ /var/lib/kubelet/ /var/lib/etcd /root/.kube
    rm -rf /etc/cni /opt/cni/bin

    echo "Flushing iptables rules..."
    iptables -F
    iptables -t nat -F
    iptables -t mangle -F
    iptables -X

    echo "Kubernetes uninstallation complete!"
}

# Function to optionally uninstall Docker
uninstall_docker() {
    read -p "Do you want to uninstall Docker as well? (y/n): " remove_docker
    if [[ "$remove_docker" == "y" ]]; then
        echo "Uninstalling Docker..."
        apt-get purge -y docker.io
        apt-get autoremove -y
        rm -rf /var/lib/docker /etc/docker
        echo "Docker uninstalled successfully!"
    else
        echo "Skipping Docker uninstallation."
    fi
}

# Function to optionally uninstall Helm
uninstall_helm() {
    read -p "Do you want to uninstall Helm as well? (y/n): " remove_helm
    if [[ "$remove_helm" == "y" ]]; then
        echo "Uninstalling Helm..."
        rm -f /usr/local/bin/helm
        rm -rf ~/.helm
        echo "Helm uninstalled successfully!"
    else
        echo "Skipping Helm uninstallation."
    fi
}

# Function to optionally uninstall Minikube and Kind
uninstall_extras() {
    read -p "Do you want to uninstall Minikube? (y/n): " remove_minikube
    if [[ "$remove_minikube" == "y" ]]; then
        echo "Uninstalling Minikube..."
        rm -f /usr/local/bin/minikube
        rm -rf ~/.minikube
        echo "Minikube uninstalled successfully!"
    fi

    read -p "Do you want to uninstall Kind? (y/n): " remove_kind
    if [[ "$remove_kind" == "y" ]]; then
        echo "Uninstalling Kind..."
        rm -f /usr/local/bin/kind
        echo "Kind uninstalled successfully!"
    fi
}

# Main execution flow
check_root
uninstall_kubernetes
uninstall_docker
uninstall_helm
uninstall_extras

echo "All selected components have been uninstalled. System cleanup is complete!"
