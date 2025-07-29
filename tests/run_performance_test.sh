#!/bin/bash

# Performance testing script for Wan2.1
# Usage: ./run_performance_test.sh [test_type] [num_gpus] [iterations]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

# Default parameters
TEST_TYPE=${1:-"all"}
NUM_GPUS=${2:-2}
NUM_ITERATIONS=${3:-50}

echo "=== Wan2.1 Performance Testing ==="
echo "Test type: $TEST_TYPE"
echo "Number of GPUs: $NUM_GPUS"
echo "Iterations: $NUM_ITERATIONS"
echo "================================"

# Check if we have enough GPUs
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# Create results directory
mkdir -p results
cd results

# Run the performance test
python ../tests/performance_test.py \
    --test_type "$TEST_TYPE" \
    --num_gpus "$NUM_GPUS" \
    --num_iterations "$NUM_ITERATIONS" \
    --save_results

echo ""
echo "=== Test Completed ==="
echo "Results saved in: $(pwd)"
ls -la *.json 2>/dev/null || echo "No result files found"

echo ""
echo "To analyze results, run:"
echo "python ../tests/analyze_results.py"