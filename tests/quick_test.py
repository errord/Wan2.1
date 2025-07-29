#!/usr/bin/env python3
"""
Quick performance test for basic functionality verification
Runs a smaller subset of tests for faster results
"""

import time
import statistics
import torch
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def timer(description: str):
    """Simple timer context manager"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {(end - start) * 1000:.2f} ms")


def quick_memory_transfer_test():
    """Quick test of memory transfer performance"""
    print("=== Quick Memory Transfer Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    sizes_mb = [1, 16, 64]  # Small subset for quick test
    
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4
        
        # Test CPU to GPU
        times = []
        for _ in range(10):  # Fewer iterations for speed
            tensor_cpu = torch.randn(num_elements, dtype=torch.float32)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            tensor_gpu = tensor_cpu.to(device)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
            del tensor_gpu
        
        avg_time = statistics.mean(times)
        bandwidth = (size_mb / 1024) / (avg_time / 1000)  # GB/s
        print(f"CPU->GPU {size_mb}MB: {avg_time:.2f} ms, {bandwidth:.2f} GB/s")


def quick_compute_efficiency_test():
    """Quick test of compute efficiency impact"""
    print("\n=== Quick Compute Efficiency Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    
    # Simple test model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.GELU(),
        nn.Linear(2048, 1024)
    ).to(device)
    
    batch_size = 32
    input_tensor = torch.randn(batch_size, 1024, device=device)
    
    # Baseline performance
    baseline_times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        baseline_times.append((end - start) * 1000)
    
    baseline_avg = statistics.mean(baseline_times)
    
    # Performance with concurrent transfer
    transfer_times = []
    transfer_size = 16 * 1024 * 1024 // 4  # 16MB
    
    for _ in range(20):
        transfer_tensor = torch.randn(transfer_size)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            output = model(input_tensor)
            # Concurrent transfer
            gpu_transfer = transfer_tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        transfer_times.append((end - start) * 1000)
        del gpu_transfer
    
    transfer_avg = statistics.mean(transfer_times)
    overhead = ((transfer_avg - baseline_avg) / baseline_avg) * 100
    
    print(f"Baseline compute: {baseline_avg:.2f} ms")
    print(f"With 16MB transfer: {transfer_avg:.2f} ms")
    print(f"Overhead: {overhead:.1f}%")


def quick_distributed_test():
    """Quick test of distributed operations (single process)"""
    print("\n=== Quick Distributed Communication Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for distributed test, skipping")
        return
    
    # Simulate AllReduce operation timing (without actual distributed setup)
    device = torch.device('cuda:0')
    sizes_mb = [1, 16, 64]
    
    print("Simulating AllReduce operations...")
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4
        
        times = []
        for _ in range(10):
            tensor = torch.randn(num_elements, device=device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Simulate collective communication by doing some operations
            # In real scenario, this would be dist.all_reduce()
            tensor_copy = tensor.clone()
            result = tensor + tensor_copy  # Simulate reduction operation
            torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = statistics.mean(times)
        bandwidth = (size_mb / 1024) / (avg_time / 1000)  # GB/s (simulated)
        print(f"Simulated AllReduce {size_mb}MB: {avg_time:.2f} ms, ~{bandwidth:.2f} GB/s")


def main():
    print("Wan2.1 Quick Performance Test")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print()
    
    with timer("Total test time"):
        quick_memory_transfer_test()
        quick_compute_efficiency_test()
        quick_distributed_test()
    
    print("\nQuick test completed!")
    print("For comprehensive testing, run: python tests/performance_test.py")


if __name__ == '__main__':
    main()