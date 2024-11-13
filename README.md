
<!-- SETUP: -->

sudo iptables -F
sudo iptables -X

conda env export > environment.yml
conda env create -f environment.yml
conda activate <environment_name>



<!-- install nccl -->

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install libnccl2 libnccl-dev

verify:

find /usr -name "libnccl-net.so" 2>/dev/null

export:

export LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH


# PS
export TP_SOCKET_IFNAME=enp0s31f6
export GLOO_SOCKET_IFNAME=enp0s31f6
export NCCL_SOCKET_IFNAME=enp0s31f6
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device gpu --verbose


# worker 1
export TP_SOCKET_IFNAME=enxf8e43bb95201
export GLOO_SOCKET_IFNAME=enxf8e43bb95201
export NCCL_SOCKET_IFNAME=enxf8e43bb95201
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device gpu --verbose


# worker 2
export TP_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2
export MASTER_PORT=12345
export MASTER_ADDR=10.172.13.13
export WORLD_SIZE=2
export RANK=0
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device cpu




# 
#  3 rank confgig


# PS
export TP_SOCKET_IFNAME=enp0s31f6
export GLOO_SOCKET_IFNAME=enp0s31f6
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=3
export RANK=0
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device cpu --verbose


# worker 1
export TP_SOCKET_IFNAME=enxf8e43bb95201
export GLOO_SOCKET_IFNAME=enxf8e43bb95201
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=3
export RANK=1
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device cpu --verbose


# worker 2
export TP_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=3
export RANK=2
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device cpu --verbose
