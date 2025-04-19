#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --nnodes <number_of_nodes> --node_rank <node_rank> --master_addr <master_address> --epochs <epochs> --batch_size <batch_size>"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nnodes)      NNODES="$2"; shift 2;;
        --node_rank)   NODE_RANK="$2"; shift 2;;
        --master_addr) MASTER_ADDR="$2"; shift 2;;
        --epochs)      EPOCHS="$2"; shift 2;;
        --batch_size)  BATCH_SIZE="$2"; shift 2;;
        *) echo "Unknown param $1"; usage;;
    esac
done

# Validate
[[ -z "$NNODES" || -z "$NODE_RANK" || -z "$MASTER_ADDR" || -z "$EPOCHS" || -z "$BATCH_SIZE" ]] && usage

# Discover the 192.168.x interface
IFNAME=$(ip -o -4 addr list | awk '/192\.168\./{print $2; exit}')
[ -z "$IFNAME" ] && { echo "No 192.168.* interface"; exit 1; }

echo "Using iface $IFNAME"

# Export env
export TP_SOCKET_IFNAME="$IFNAME"
export GLOO_SOCKET_IFNAME="$IFNAME"
export NCCL_SOCKET_IFNAME="$IFNAME"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="12355"
export WORLD_SIZE="$NNODES"
export RANK="$NODE_RANK"
export NCCL_DEBUG="INFO"
export NCCL_IB_DISABLE="1"

TORCHRUN="/home/ubuntu/.local/bin/torchrun"

COMMON_ARGS=(
  --nproc_per_node=1
  --nnodes="$NNODES"
  --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
)

# Run PS mode
echo "=== Running Parameter‐Server mode ==="
t0=$(date +%s.%N)
$TORCHRUN "${COMMON_ARGS[@]}" distributed_compare.py \
    ps \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"
t1=$(date +%s.%N)
dur_ps=$(echo "$t1 - $t0" | bc)
echo "Parameter‐Server run completed in ${dur_ps} seconds."
echo

# Run Ring mode
echo "=== Running Ring All‑Reduce mode ==="
t2=$(date +%s.%N)
$TORCHRUN "${COMMON_ARGS[@]}" distributed_compare.py \
    ring \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"
t3=$(date +%s.%N)
dur_ring=$(echo "$t3 - $t2" | bc)
echo "Ring All‑Reduce run completed in ${dur_ring} seconds."
