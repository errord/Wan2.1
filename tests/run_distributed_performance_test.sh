#!/bin/bash

# Multi-node distributed performance testing script for Wan2.1
# This script should be run on each node with appropriate environment variables

# Check if required environment variables are set
if [ -z "$RANK" ] || [ -z "$WORLD_SIZE" ] || [ -z "$LOCAL_RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
    echo "Error: Required environment variables not set!"
    echo "Please set the following environment variables:"
    echo "  RANK - rank of this node (0, 1, 2, ...)"
    echo "  WORLD_SIZE - total number of nodes"
    echo "  LOCAL_RANK - local GPU rank on this node (usually 0 for single GPU per node)"
    echo "  MASTER_ADDR - IP address of master node"
    echo "  MASTER_PORT - port for communication"
    echo ""
    echo "Example for master node (rank 0):"
    echo "  export RANK=0"
    echo "  export WORLD_SIZE=2"
    echo "  export LOCAL_RANK=0"
    echo "  export MASTER_ADDR='10.79.79.197'"
    echo "  export MASTER_PORT='7860'"
    echo ""
    echo "Example for worker node (rank 1):"
    echo "  export RANK=1"
    echo "  export WORLD_SIZE=2"
    echo "  export LOCAL_RANK=0"
    echo "  export MASTER_ADDR='10.79.79.197'"
    echo "  export MASTER_PORT='7860'"
    echo ""
    echo "Then run: ./run_distributed_performance_test.sh"
    exit 1
fi

# Print configuration
echo "=== Distributed Performance Test Configuration ==="
echo "Rank: $RANK"
echo "World Size: $WORLD_SIZE"
echo "Local Rank: $LOCAL_RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "=================================================="

# Optional: Set additional NCCL environment variables for better performance
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Optional: Network interface configuration (uncomment and modify as needed)
# export NCCL_SOCKET_IFNAME='enp65s0'  # Adjust to your network interface
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

# Run the distributed performance test
echo "Starting distributed performance test..."
python performance_test.py --test_type distributed --save_results --num_iterations 50

# Check exit status
if [ $? -eq 0 ]; then
    echo "Distributed performance test completed successfully on rank $RANK"
else
    echo "Distributed performance test failed on rank $RANK"
    exit 1
fi