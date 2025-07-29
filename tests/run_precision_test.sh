#!/bin/bash

# Precision Performance Testing Script for Wan2.1
# Usage: ./run_precision_test.sh [test_type] [num_gpus] [iterations]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

# Default parameters
TEST_TYPE=${1:-"all"}
NUM_GPUS=${2:-2}
NUM_ITERATIONS=${3:-20}

echo "=== Wan2.1 Precision Performance Testing ==="
echo "Test type: $TEST_TYPE"
echo "Number of GPUs: $NUM_GPUS"
echo "Iterations: $NUM_ITERATIONS"
echo "============================================"

# Check if we have enough GPUs
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# Check GPU memory (should be suitable for RTX 5090 32GB)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU Memory: ${GPU_MEMORY}MB"

if [ "$GPU_MEMORY" -lt 20000 ]; then
    echo "Warning: GPU memory is less than 20GB. Tests may fail with OOM."
    echo "Consider reducing model size or batch size."
fi

# Create results directory
mkdir -p results
cd results

# Run the precision test
echo "Starting precision performance tests..."
python ../tests/precision_performance_test.py \
    --test_type "$TEST_TYPE" \
    --num_gpus "$NUM_GPUS" \
    --num_iterations "$NUM_ITERATIONS" \
    --batch_sizes 4 8 \
    --seq_lengths 1024 2048 \
    --save_results

echo ""
echo "=== Test Completed ==="
echo "Results saved in: $(pwd)"
ls -la *.json 2>/dev/null || echo "No result files found"

echo ""
echo "Key findings:"
echo "1. Check precision_results_gpu_*.json for single GPU fp32/fp16/bf16 comparison"
echo "2. Check distributed_precision_results_*.json for FSDP communication impact"
echo "3. Look for 'Precision Benefit Analysis' in logs to see if network negates precision gains"

echo ""
echo "To analyze results:"
echo "python ../tests/analyze_precision_results.py"