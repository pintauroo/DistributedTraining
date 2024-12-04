#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --nnodes <number_of_nodes> --node_rank <node_rank> --master_addr <master_address> --epochs <epochs> --batch_size <batch_size>"
    echo "  --nnodes       Number of nodes in the distributed setup (e.g., 2)"
    echo "  --node_rank    Rank of this node (starting from 0)"
    echo "  --master_addr  Master node address (e.g., 192.168.1.2)"
    echo "  --epochs       Number of training epochs (e.g., 1000)"
    echo "  --batch_size   Size of each training batch (e.g., 32)"
    exit 1
}

# Initialize variables
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
EPOCHS=""
BATCH_SIZE=""

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [ -z "$NNODES" ] || [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$EPOCHS" ] || [ -z "$BATCH_SIZE" ]; then
    echo "Error: --nnodes, --node_rank, --master_addr, --epochs, and --batch_size must be provided."
    usage
fi

# Validate that NNODES is a positive integer
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [ "$NNODES" -lt 1 ]; then
    echo "Error: --nnodes must be a positive integer."
    exit 1
fi

# Validate that NODE_RANK is a non-negative integer and less than NNODES
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "Error: --node_rank must be a non-negative integer less than --nnodes."
    exit 1
fi

# Validate that EPOCHS is a positive integer
if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; then
    echo "Error: --epochs must be a positive integer."
    exit 1
fi

# Validate that BATCH_SIZE is a positive integer
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "Error: --batch_size must be a positive integer."
    exit 1
fi

# Function to retrieve the network interface name with IP 192.168.1.x
get_interface() {
    # Use 'ip' to list all IPv4 addresses and filter for 192.168.1.x
    local interface
    interface=$(ip -o -4 addr list | awk '/192\.168\.1\./ {print $2}' | uniq)

    if [ -z "$interface" ]; then
        echo "Error: No network interface found with IP in the range 192.168.1.x"
        exit 1
    fi

    echo "$interface"
}

# Retrieve the interface name
IFNAME=$(get_interface)
echo "Using network interface: $IFNAME"

# Set environment variables
export TP_SOCKET_IFNAME="$IFNAME"
export GLOO_SOCKET_IFNAME="$IFNAME"
export NCCL_SOCKET_IFNAME="$IFNAME"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="12355"
export WORLD_SIZE="$NNODES"
export RANK="$NODE_RANK"
export NCCL_DEBUG="INFO"
export NCCL_IB_DISABLE="1"

# Optional: Print the environment variables for verification
echo "Environment Variables Set:"
echo "TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"

# Execute the torchrun command with the specified epochs and batch size
torchrun \
    --nproc_per_node=1 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    distributed_parameter_server_clean.py "$EPOCHS" "$BATCH_SIZE" \
    --device gpu
    # --verbose
