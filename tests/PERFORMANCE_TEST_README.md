# Wan2.1 Performance Testing Suite

This directory contains comprehensive performance testing tools for the Wan2.1 model, designed to evaluate three critical aspects:

1. **Memory-GPU Transfer Performance** - Testing data transfer latency between CPU memory and GPU memory
2. **GPU Compute Efficiency Impact** - Measuring how memory transfers affect GPU computation performance  
3. **FSDP Network Performance** - Evaluating multi-node multi-GPU FSDP communication overhead

## Files Overview

- `performance_test.py` - Main comprehensive performance testing script
- `quick_test.py` - Quick verification test for basic functionality
- `analyze_results.py` - Analysis and visualization tool for test results
- `run_performance_test.sh` - Convenient shell script wrapper
- `PERFORMANCE_TEST_README.md` - This documentation

## Quick Start

### 1. Quick Functionality Test

For a fast verification that everything works:

```bash
cd tests/
python quick_test.py
```

This runs a lightweight test in ~10 seconds to verify basic functionality.

### 2. Single GPU Performance Test

To test memory transfer and compute efficiency on a single GPU:

```bash
# Run single GPU tests only
python performance_test.py --test_type single --gpu_id 0 --num_iterations 100 --save_results
```

### 3. Multi-GPU FSDP Test

To test distributed FSDP performance across multiple GPUs:

```bash
# Run distributed tests with 2 GPUs
python performance_test.py --test_type distributed --num_gpus 2 --num_iterations 50 --save_results
```

### 4. Complete Test Suite

To run all tests:

```bash
# Using the shell script (recommended)
./run_performance_test.sh all 2 50

# Or directly with Python
python performance_test.py --test_type all --num_gpus 2 --num_iterations 50 --save_results
```

## Test Details

### Memory Transfer Tests

- **Purpose**: Measure CPU↔GPU transfer latency and bandwidth
- **Data Sizes**: 1MB, 4MB, 16MB, 64MB, 256MB, 1GB
- **Metrics**: 
  - Transfer latency (milliseconds)
  - Transfer bandwidth (GB/s)
  - Statistical variance across multiple runs
- **Scenarios**:
  - Large data, infrequent transfers (simulating model loading)
  - Small data, frequent transfers (simulating dynamic data loading)

### Compute Efficiency Tests

- **Purpose**: Measure GPU compute performance impact from concurrent transfers
- **Test Setup**: Simple neural network (Linear + GELU + Linear + LayerNorm)
- **Batch Sizes**: 8, 16, 32, 64
- **Transfer Sizes**: 1MB, 4MB, 16MB concurrent transfers
- **Metrics**:
  - Baseline compute time without transfers
  - Compute time with concurrent transfers
  - Performance overhead percentage

### FSDP Network Tests

- **Purpose**: Evaluate distributed training communication performance
- **Operations Tested**:
  - AllReduce collective operations
  - Forward/backward pass timing
  - Parameter synchronization frequency impact
- **Message Sizes**: 1MB, 4MB, 16MB, 64MB, 256MB
- **Metrics**:
  - AllReduce latency and bandwidth
  - Forward/backward pass timing
  - Synchronization overhead

## Command Line Options

### performance_test.py

```bash
python performance_test.py [options]

Options:
  --test_type {single,distributed,all}  Type of test to run (default: all)
  --gpu_id GPU_ID                      GPU ID for single GPU tests (default: 0)
  --num_gpus NUM_GPUS                  Number of GPUs for distributed tests (default: 2)
  --num_iterations NUM_ITERATIONS      Number of iterations per test (default: 100)
  --master_addr MASTER_ADDR            Master address for distributed testing (default: localhost)
  --master_port MASTER_PORT            Master port for distributed testing (default: 12355)
  --save_results                       Save results to JSON files
```

### analyze_results.py

```bash
python analyze_results.py [options]

Options:
  --results_dir RESULTS_DIR           Directory containing result JSON files (default: .)
  --output_dir OUTPUT_DIR             Directory to save plots (default: plots)
  --report_file REPORT_FILE           Output file for text report (default: performance_report.txt)
  --no_plots                          Skip generating visualization plots
```

## Result Analysis

After running tests, analyze the results:

```bash
# Generate analysis and visualizations
python analyze_results.py --results_dir results/ --output_dir plots/

# Generate only text report (no plots)
python analyze_results.py --results_dir results/ --no_plots
```

The analysis tool generates:

1. **Text Report** (`performance_report.txt`):
   - Statistical summary of all tests
   - Performance metrics and comparisons
   - Identification of bottlenecks

2. **Visualization Plots** (in `plots/` directory):
   - Memory transfer latency and bandwidth curves
   - Compute efficiency overhead charts
   - FSDP communication performance graphs

## Understanding Results

### Memory Transfer Performance

- **Good Performance**: 
  - CPU→GPU: >10 GB/s for large transfers
  - GPU→CPU: >8 GB/s for large transfers
- **Watch for**: 
  - High latency for small transfers (expected)
  - Unexpectedly low bandwidth for large transfers

### Compute Efficiency Impact

- **Typical Overhead**: 5-15% for concurrent small transfers
- **Concerning**: >25% overhead indicates potential issues
- **Best Case**: <5% overhead with proper non_blocking transfers

### FSDP Performance

- **AllReduce Efficiency**: Should scale with message size
- **Training Overhead**: Forward pass typically faster than backward
- **Sync Frequency**: Less frequent sync reduces communication overhead

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `--num_iterations`
   - Test smaller data sizes first
   - Ensure no other processes are using GPU memory

2. **Distributed Tests Fail**:
   - Check that multiple GPUs are available
   - Verify no other processes are using GPUs
   - Try different `--master_port` if port conflicts occur

3. **Low Performance Results**:
   - Check GPU utilization with `nvidia-smi`
   - Verify PCIe bandwidth and configuration
   - Ensure adequate cooling and power

### Expected Performance Baselines

For reference, typical performance on modern hardware:

| Hardware | CPU→GPU BW | GPU→CPU BW | Compute Overhead |
|----------|------------|------------|------------------|
| RTX 4090 | 20-25 GB/s | 15-20 GB/s | 5-10% |
| RTX 3080 | 15-20 GB/s | 12-15 GB/s | 8-15% |
| V100     | 12-16 GB/s | 10-14 GB/s | 10-20% |

## Integration with Wan2.1

These tests complement the existing Wan2.1 testing infrastructure:

- Use similar argument patterns as `test.sh`
- Compatible with FSDP configurations in `wan/distributed/fsdp.py`
- Results can guide optimization of model offloading strategies

## Advanced Usage

### Custom Test Scenarios

Modify the test parameters in the source code to test specific scenarios:

```python
# In performance_test.py, modify these lists:
transfer_sizes = [1, 4, 16, 64, 256, 1024]  # MB
batch_sizes = [8, 16, 32, 64]
comm_sizes = [1, 4, 16, 64, 256]  # MB
```

### Multi-Node Testing

For true multi-node testing (across different machines):

```bash
# On master node (machine 1):
python performance_test.py --test_type distributed --master_addr 192.168.1.100 --master_port 12355

# On worker nodes (machine 2, 3, etc.):
# Set appropriate RANK and WORLD_SIZE environment variables
```

## Contributing

When adding new tests:

1. Follow the existing class structure (`*Tester` classes)
2. Add timing and memory tracking contexts
3. Store results in the standard format for analysis compatibility
4. Update this README with new test descriptions

## References

- [PyTorch Distributed Training](https://pytorch.org/tutorials/distributed/ddp_tutorial.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)