# Multi-Node Distributed Performance Testing for Wan2.1

This document explains how to run distributed performance tests across multiple machines for the Wan2.1 model.

## Overview

The updated `performance_test.py` now supports true multi-node distributed testing, similar to your existing `start_master.sh` and `start_rank1.sh` configuration for Wan2.1 inference.

## Key Changes from Original

- **Removed single-machine multi-GPU simulation**: No more `mp.spawn()` which was only for single machine
- **Added environment variable based configuration**: Uses `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`
- **Real multi-node testing**: Tests actual network communication between different machines
- **Enhanced network tests**: Added bandwidth testing under different loads

## Prerequisites

1. **Multiple machines** with CUDA-enabled GPUs
2. **Network connectivity** between machines
3. **Same Wan2.1 environment** installed on all machines
4. **Synchronized clocks** (recommended)

## Configuration

### Environment Variables Required

Each machine needs these environment variables:

```bash
export RANK=<node_rank>           # 0 for master, 1,2,3... for workers
export WORLD_SIZE=<total_nodes>   # Total number of machines
export LOCAL_RANK=0               # Usually 0 for single GPU per machine
export MASTER_ADDR='<master_ip>'  # IP address of master machine
export MASTER_PORT='<port>'       # Communication port
```

### Example Configuration (2 Machines)

Based on your existing setup:

**Master Machine (Rank 0):**
```bash
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0
export MASTER_ADDR='10.79.79.197'
export MASTER_PORT='7860'

# Network configuration
export NCCL_SOCKET_IFNAME='enp65s0'
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

**Worker Machine (Rank 1):**
```bash
export RANK=1
export WORLD_SIZE=2
export LOCAL_RANK=0
export MASTER_ADDR='10.79.79.197'
export MASTER_PORT='7860'

# Network configuration
export NCCL_SOCKET_IFNAME='enp33s0'
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

## Running Distributed Tests

### Method 1: Using the Example Scripts

1. **Generate example scripts:**
   ```bash
   cd tests
   bash example_distributed_test.sh
   ```

2. **Copy and run the generated scripts on each machine**

### Method 2: Manual Execution

1. **On each machine, set environment variables:**
   ```bash
   # Set appropriate RANK, WORLD_SIZE, etc. for each machine
   ```

2. **Start worker nodes first:**
   ```bash
   # On worker machines (rank 1, 2, ...)
   cd /path/to/Wan2.1/tests
   python performance_test.py --test_type distributed --save_results
   ```

3. **Start master node last:**
   ```bash
   # On master machine (rank 0)
   cd /path/to/Wan2.1/tests
   python performance_test.py --test_type distributed --save_results
   ```

## Tests Performed

### 1. AllReduce Latency Testing
- **Purpose**: Measure communication latency between nodes
- **Data sizes**: 1MB to 1GB
- **Metrics**: Latency (ms), Bandwidth (MB/s)

### 2. FSDP Forward/Backward Performance
- **Purpose**: Measure training performance with parameter sharding
- **Batch sizes**: 2, 4, 8, 16
- **Metrics**: Forward time, Backward time, Total time

### 3. Parameter Synchronization Frequency
- **Purpose**: Impact of different sync frequencies on performance
- **Frequencies**: Every 1, 2, 4, 8, 16 iterations
- **Metrics**: Time per iteration

### 4. Network Bandwidth Under Load
- **Purpose**: Network performance with concurrent operations
- **Loads**: 1, 2, 4 concurrent operations
- **Data sizes**: 64MB, 256MB, 1GB
- **Metrics**: Effective bandwidth

## Output and Results

### Console Output
Each machine will show:
- Rank and configuration information
- Test progress and timing results
- Network communication statistics

### Saved Results
Master node (rank 0) saves results to:
```
distributed_performance_results_<world_size>nodes_<timestamp>.json
```

### Key Metrics to Analyze

1. **Network Latency**: Base communication overhead
2. **Bandwidth Utilization**: How well network bandwidth is used
3. **Scaling Efficiency**: Performance vs number of nodes
4. **Synchronization Overhead**: Impact of frequent parameter updates

## Troubleshooting

### Common Issues

1. **Connection Timeout**:
   - Check firewall settings
   - Verify MASTER_ADDR and MASTER_PORT
   - Ensure network connectivity between machines

2. **NCCL Errors**:
   - Verify NCCL_SOCKET_IFNAME matches your network interface
   - Try disabling IB and P2P if not available
   - Check CUDA compatibility between machines

3. **Environment Issues**:
   - Ensure same PyTorch/CUDA versions on all machines
   - Verify PATH and PYTHONPATH are correct
   - Check GPU visibility on each machine

### Debug Commands

```bash
# Check network interface
ip addr show

# Test connectivity
ping <master_ip>
telnet <master_ip> <master_port>

# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Integration with Wan2.1

This distributed testing framework uses the same configuration pattern as your existing Wan2.1 setup:

- **Same environment variables**: RANK, WORLD_SIZE, MASTER_ADDR, etc.
- **Same network configuration**: NCCL settings, interface names
- **Same FSDP patterns**: Compatible with your existing FSDP implementation

The tests will help you understand:
- **Network bottlenecks** in your multi-machine setup
- **Optimal batch sizes** for distributed training
- **Communication overhead** vs computation time
- **Scaling characteristics** of your hardware configuration

## Next Steps

After running these tests, you can:

1. **Compare single-machine vs multi-machine performance**
2. **Optimize network configuration** based on results
3. **Tune batch sizes and synchronization frequencies**
4. **Identify hardware bottlenecks** in your setup
5. **Plan scaling strategies** for larger deployments