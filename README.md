

sudo iptables -F
sudo iptables -X

conda env export > environment.yml
conda env create -f environment.yml
conda activate <environment_name>

# PS
export TP_SOCKET_IFNAME=enp0s31f6
export GLOO_SOCKET_IFNAME=enp0s31f6
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=0
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device cpu


# worker 1
export TP_SOCKET_IFNAME=enxf8e43bb95201
export GLOO_SOCKET_IFNAME=enxf8e43bb95201
export MASTER_ADDR=10.172.13.13
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=1
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.172.13.13" --master_port=12355 distributed_parameter_server.py 100 32 --device cpu


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
