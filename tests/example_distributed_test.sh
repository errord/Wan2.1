#!/bin/bash

# Example scripts for running distributed performance tests on multiple machines
# Based on your start_master.sh and start_rank1.sh configuration

echo "=== Multi-Node Distributed Performance Test Examples ==="
echo ""
echo "These example scripts show how to run distributed performance tests"
echo "on multiple machines using the same configuration as your Wan2.1 setup."
echo ""

echo "1. On Master Node (Rank 0) - Machine with IP 10.79.79.197:"
echo "   Save this as 'run_perf_master.sh':"
echo ""
cat << 'EOF'
#!/bin/bash
# Set environment variables for master node
export PATH=/data/apps/wan-21/wan-env/bin:$PATH
export MASTER_ADDR='10.79.79.197'
export MASTER_PORT='7860'
export RANK='0'
export WORLD_SIZE='2'
export LOCAL_RANK='0'

# NCCL Network configuration
export NCCL_SOCKET_IFNAME='enp65s0'
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Gloo configuration for CPU communication
export GLOO_SOCKET_IFNAME='enp65s0'
export GLOO_TIMEOUT=1800
export GLOO_DEVICE_TRANSPORT=TCP

# Disable problematic features
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Run distributed performance test
cd /data/apps/wan-21/dev/Wan2.1/tests
python performance_test.py --test_type distributed --save_results --num_iterations 50
EOF

echo ""
echo "2. On Worker Node (Rank 1) - Second machine:"
echo "   Save this as 'run_perf_rank1.sh':"
echo ""
cat << 'EOF'
#!/bin/bash
# Set environment variables for worker node
export PATH=/data/apps/wan-21/wan-env/bin:$PATH
export MASTER_ADDR='10.79.79.197'
export MASTER_PORT='7860'
export RANK='1'
export WORLD_SIZE='2'
export LOCAL_RANK='0'

# NCCL Network configuration  
export NCCL_SOCKET_IFNAME='enp33s0'
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Gloo configuration for CPU communication
export GLOO_SOCKET_IFNAME='enp33s0'
export GLOO_TIMEOUT=1800
export GLOO_DEVICE_TRANSPORT=TCP

# Disable problematic features
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Run distributed performance test
cd /data/apps/wan-21/dev/Wan2.1/tests
python performance_test.py --test_type distributed --save_results --num_iterations 50
EOF

echo ""
echo "3. How to run the tests:"
echo ""
echo "   a) Copy the scripts to both machines and make them executable:"
echo "      chmod +x run_perf_master.sh"
echo "      chmod +x run_perf_rank1.sh"
echo ""
echo "   b) Start the worker node first:"
echo "      # On second machine (rank 1):"
echo "      ./run_perf_rank1.sh"
echo ""
echo "   c) Then start the master node:"
echo "      # On master machine (rank 0):"
echo "      ./run_perf_master.sh"
echo ""
echo "   d) Results will be saved on the master node as:"
echo "      'distributed_performance_results_2nodes_TIMESTAMP.json'"
echo ""
echo "4. Tests that will be performed:"
echo "   - AllReduce latency: 1MB to 1GB data sizes"
echo "   - FSDP forward/backward passes"
echo "   - Parameter synchronization frequency impact"
echo "   - Network bandwidth under different loads"
echo ""
echo "5. Key differences from single-machine tests:"
echo "   - Real network latency between machines"
echo "   - Network bandwidth limitations"
echo "   - Cross-machine parameter synchronization overhead"
echo "   - Impact of network interface configuration"
echo ""