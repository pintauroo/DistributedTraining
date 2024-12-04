#!/bin/bash

# ==========================
# Distributed Training Script
# ==========================

# Enable debug mode (optional)
# Uncomment the following line to enable shell debug mode
# set -x

# Function to display usage information
usage() {
    echo "Usage: $0 --nnodes <number_of_nodes> --node_rank <node_rank> --master_addr <master_address> [--epochs <epochs>] [--batch_size <batch_size>] [--device <device>]"
    echo ""
    echo "Required Arguments:"
    echo "  --nnodes       Number of nodes in the distributed setup (e.g., 2)"
    echo "  --node_rank    Rank of this node (starting from 0)"
    echo "  --master_addr  Master node address (e.g., 192.168.1.2)"
    echo ""
    echo "Optional Arguments:"
    echo "  --epochs       Number of training epochs (default: 100)"
    echo "  --batch_size   Size of each training batch (default: 32)"
    echo "  --device       Device to use for training (e.g., cpu, gpu; default: gpu)"
    echo ""
    echo "Example:"
    echo "  $0 --nnodes 2 --node_rank 0 --master_addr 192.168.1.1 --epochs 50 --batch_size 64 --device cpu"
    exit 1
}

# Initialize variables with default values
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
EPOCHS=100          # Default epochs
BATCH_SIZE=32       # Default batch size
DEVICE="gpu"        # Default device

echo "===== Script Initialization ====="

# Parse input arguments
echo "Parsing input arguments..."
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nnodes)
            NNODES="$2"
            echo "Parsed --nnodes: $NNODES"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            echo "Parsed --node_rank: $NODE_RANK"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            echo "Parsed --master_addr: $MASTER_ADDR"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            echo "Parsed --epochs: $EPOCHS"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            echo "Parsed --batch_size: $BATCH_SIZE"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            echo "Parsed --device: $DEVICE"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

echo "===== Argument Parsing Completed ====="

# Check if all required arguments are provided
echo "Validating required arguments..."
if [ -z "$NNODES" ] || [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ]; then
    echo "Error: --nnodes, --node_rank, and --master_addr must be provided."
    usage
fi
echo "All required arguments are provided."

# Validate that NNODES is a positive integer
echo "Validating --nnodes..."
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [ "$NNODES" -lt 1 ]; then
    echo "Error: --nnodes must be a positive integer."
    exit 1
fi
echo "--nnodes validation passed."

# Validate that NODE_RANK is a non-negative integer and less than NNODES
echo "Validating --node_rank..."
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "Error: --node_rank must be a non-negative integer less than --nnodes."
    exit 1
fi
echo "--node_rank validation passed."

# Validate that EPOCHS is a positive integer
echo "Validating --epochs..."
if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; then
    echo "Error: --epochs must be a positive integer."
    exit 1
fi
echo "--epochs validation passed."

# Validate that BATCH_SIZE is a positive integer
echo "Validating --batch_size..."
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "Error: --batch_size must be a positive integer."
    exit 1
fi
echo "--batch_size validation passed."

# Validate that DEVICE is either 'cpu' or 'gpu'
echo "Validating --device..."
if ! [[ "$DEVICE" =~ ^(cpu|gpu)$ ]]; then
    echo "Error: --device must be either 'cpu' or 'gpu'."
    exit 1
fi
echo "--device validation passed."

echo "===== All Validations Passed ====="

# Function to retrieve the network interface name with IP 192.168.1.x
get_interface() {
    echo "Retrieving network interface with IP in the range 192.168.1.x..."
    # Use 'ip' to list all IPv4 addresses and filter for 192.168.1.x
    local interface
    interface=$(ip -o -4 addr list | awk '/192\.168\.1\./ {print $2}' | uniq)

    if [ -z "$interface" ]; then
        echo "Error: No network interface found with IP in the range 192.168.1.x"
        exit 1
    fi

    echo "Network interface found: $interface"
    echo "$interface"
}

echo "===== Starting Network Interface Retrieval ====="
# Retrieve the interface name
IFNAME=$(get_interface)
echo "Using network interface: $IFNAME"
echo "===== Network Interface Retrieval Completed ====="

# Set environment variables
echo "Setting environment variables..."
export TP_SOCKET_IFNAME="$IFNAME"
export GLOO_SOCKET_IFNAME="$IFNAME"
export NCCL_SOCKET_IFNAME="$IFNAME"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="12355"
export WORLD_SIZE="$NNODES"
export RANK="$NODE_RANK"
export NCCL_DEBUG="INFO"
export NCCL_IB_DISABLE="1"
echo "Environment variables set successfully."

# Optional: Print the environment variables for verification
echo "===== Environment Variables ====="
echo "TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "=================================="

# Determine the directory where the script resides
echo "Determining script directory..."
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Path to the Python script
PYTHON_SCRIPT="$SCRIPT_DIR/distributed_parameter_server.py"
echo "Python script path: $PYTHON_SCRIPT"

# Check if the Python script exists
echo "Checking if Python script exists..."
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi
echo "Python script found."

# Final summary before execution
echo "===== Execution Summary ====="
echo "Number of Nodes: $NNODES"
echo "Node Rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "============================="

# Execute the torchrun command
echo "Starting torchrun command..."
torchrun \
    --nproc_per_node=1 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PYTHON_SCRIPT" "$EPOCHS" "$BATCH_SIZE" \
    --device "$DEVICE" \
    --verbose

# Capture the exit status of torchrun
EXIT_STATUS=$?
echo "torchrun exited with status: $EXIT_STATUS"

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Error: torchrun command failed."
    exit $EXIT_STATUS
else
    echo "torchrun command completed successfully."
fi

echo "===== Script Execution Completed ====="
