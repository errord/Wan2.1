#!/usr/bin/env python3
"""
Precise bandwidth testing for CPU-GPU memory transfers
Tests various optimization techniques to find peak bandwidth
"""

import time
import statistics
import torch
import numpy as np
from contextlib import contextmanager


@contextmanager
def timer(description: str):
    """Context manager for timing operations"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {(end - start) * 1000:.2f} ms")


def test_basic_transfer(size_mb: int, device: torch.device, num_iterations: int = 20):
    """Basic transfer test (current implementation)"""
    print(f"\n--- Basic Transfer Test ({size_mb}MB) ---")
    
    num_elements = (size_mb * 1024 * 1024) // 4
    
    # CPU to GPU
    cpu_to_gpu_times = []
    for _ in range(num_iterations):
        tensor_cpu = torch.randn(num_elements, dtype=torch.float32)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        tensor_gpu = tensor_cpu.to(device)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        cpu_to_gpu_times.append((end - start) * 1000)
        del tensor_gpu
        torch.cuda.empty_cache()
    
    # GPU to CPU
    gpu_to_cpu_times = []
    tensor_gpu = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tensor_cpu = tensor_gpu.cpu()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        gpu_to_cpu_times.append((end - start) * 1000)
        del tensor_cpu
    
    del tensor_gpu
    torch.cuda.empty_cache()
    
    cpu_avg = statistics.mean(cpu_to_gpu_times)
    gpu_avg = statistics.mean(gpu_to_cpu_times)
    cpu_bw = (size_mb / 1024) / (cpu_avg / 1000)
    gpu_bw = (size_mb / 1024) / (gpu_avg / 1000)
    
    print(f"CPU→GPU: {cpu_avg:.2f} ms, {cpu_bw:.2f} GB/s")
    print(f"GPU→CPU: {gpu_avg:.2f} ms, {gpu_bw:.2f} GB/s")


def test_pinned_memory_transfer(size_mb: int, device: torch.device, num_iterations: int = 20):
    """Test with pinned memory for better performance"""
    print(f"\n--- Pinned Memory Transfer Test ({size_mb}MB) ---")
    
    num_elements = (size_mb * 1024 * 1024) // 4
    
    # CPU to GPU with pinned memory
    cpu_to_gpu_times = []
    for _ in range(num_iterations):
        tensor_cpu = torch.randn(num_elements, dtype=torch.float32).pin_memory()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        tensor_gpu = tensor_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        cpu_to_gpu_times.append((end - start) * 1000)
        del tensor_gpu
        torch.cuda.empty_cache()
    
    # GPU to CPU with pinned memory
    gpu_to_cpu_times = []
    tensor_gpu = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tensor_cpu = tensor_gpu.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        gpu_to_cpu_times.append((end - start) * 1000)
        del tensor_cpu
    
    del tensor_gpu
    torch.cuda.empty_cache()
    
    cpu_avg = statistics.mean(cpu_to_gpu_times)
    gpu_avg = statistics.mean(gpu_to_cpu_times)
    cpu_bw = (size_mb / 1024) / (cpu_avg / 1000)
    gpu_bw = (size_mb / 1024) / (gpu_avg / 1000)
    
    print(f"CPU→GPU (pinned): {cpu_avg:.2f} ms, {cpu_bw:.2f} GB/s")
    print(f"GPU→CPU (pinned): {gpu_avg:.2f} ms, {gpu_bw:.2f} GB/s")


def test_contiguous_memory_transfer(size_mb: int, device: torch.device, num_iterations: int = 20):
    """Test with pre-allocated contiguous memory"""
    print(f"\n--- Contiguous Memory Transfer Test ({size_mb}MB) ---")
    
    num_elements = (size_mb * 1024 * 1024) // 4
    
    # Pre-allocate pinned memory
    pinned_tensor = torch.empty(num_elements, dtype=torch.float32).pin_memory()
    gpu_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
    
    # CPU to GPU with pre-allocated memory
    cpu_to_gpu_times = []
    for _ in range(num_iterations):
        # Fill with random data
        pinned_tensor.copy_(torch.randn(num_elements, dtype=torch.float32))
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        gpu_tensor.copy_(pinned_tensor, non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        cpu_to_gpu_times.append((end - start) * 1000)
    
    # GPU to CPU with pre-allocated memory
    gpu_to_cpu_times = []
    gpu_tensor.copy_(torch.randn(num_elements, dtype=torch.float32, device=device))
    
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        pinned_tensor.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        gpu_to_cpu_times.append((end - start) * 1000)
    
    del pinned_tensor, gpu_tensor
    torch.cuda.empty_cache()
    
    cpu_avg = statistics.mean(cpu_to_gpu_times)
    gpu_avg = statistics.mean(gpu_to_cpu_times)
    cpu_bw = (size_mb / 1024) / (cpu_avg / 1000)
    gpu_bw = (size_mb / 1024) / (gpu_avg / 1000)
    
    print(f"CPU→GPU (contiguous): {cpu_avg:.2f} ms, {cpu_bw:.2f} GB/s")
    print(f"GPU→CPU (contiguous): {gpu_avg:.2f} ms, {gpu_bw:.2f} GB/s")


def test_cudamemcpy_equivalent(size_mb: int, device: torch.device, num_iterations: int = 20):
    """Test using CUDA memory copy operations directly"""
    print(f"\n--- CUDA Memcpy Equivalent Test ({size_mb}MB) ---")
    
    num_elements = (size_mb * 1024 * 1024) // 4
    
    try:
        # Use CuPy for more direct CUDA operations if available
        import cupy as cp
        
        # CPU to GPU
        cpu_to_gpu_times = []
        cpu_array = np.random.randn(num_elements).astype(np.float32)
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            gpu_array = cp.asarray(cpu_array)
            cp.cuda.Device().synchronize()
            end = time.perf_counter()
            
            cpu_to_gpu_times.append((end - start) * 1000)
            del gpu_array
        
        # GPU to CPU
        gpu_to_cpu_times = []
        gpu_array = cp.random.randn(num_elements, dtype=cp.float32)
        
        for _ in range(num_iterations):
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            cpu_result = cp.asnumpy(gpu_array)
            cp.cuda.Device().synchronize()
            end = time.perf_counter()
            
            gpu_to_cpu_times.append((end - start) * 1000)
            del cpu_result
        
        del gpu_array
        
        cpu_avg = statistics.mean(cpu_to_gpu_times)
        gpu_avg = statistics.mean(gpu_to_cpu_times)
        cpu_bw = (size_mb / 1024) / (cpu_avg / 1000)
        gpu_bw = (size_mb / 1024) / (gpu_avg / 1000)
        
        print(f"CPU→GPU (CuPy): {cpu_avg:.2f} ms, {cpu_bw:.2f} GB/s")
        print(f"GPU→CPU (CuPy): {gpu_avg:.2f} ms, {gpu_bw:.2f} GB/s")
        
    except ImportError:
        print("CuPy not available, skipping direct CUDA test")


def test_multiple_streams(size_mb: int, device: torch.device, num_iterations: int = 20):
    """Test with multiple CUDA streams"""
    print(f"\n--- Multiple Streams Test ({size_mb}MB) ---")
    
    num_elements = (size_mb * 1024 * 1024) // 4
    num_streams = 4
    chunk_size = num_elements // num_streams
    
    # Create streams
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    # CPU to GPU with multiple streams
    cpu_to_gpu_times = []
    for _ in range(num_iterations):
        tensors_cpu = [torch.randn(chunk_size, dtype=torch.float32).pin_memory() 
                      for _ in range(num_streams)]
        tensors_gpu = []
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                tensor_gpu = tensors_cpu[i].to(device, non_blocking=True)
                tensors_gpu.append(tensor_gpu)
        
        # Wait for all streams
        for stream in streams:
            stream.synchronize()
        
        end = time.perf_counter()
        cpu_to_gpu_times.append((end - start) * 1000)
        
        # Cleanup
        for tensor in tensors_gpu:
            del tensor
        torch.cuda.empty_cache()
    
    cpu_avg = statistics.mean(cpu_to_gpu_times)
    cpu_bw = (size_mb / 1024) / (cpu_avg / 1000)
    
    print(f"CPU→GPU (multi-stream): {cpu_avg:.2f} ms, {cpu_bw:.2f} GB/s")


def analyze_system_info():
    """Analyze system configuration that might affect bandwidth"""
    print("\n=== System Information ===")
    
    # GPU information
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {props.total_memory / (1024**3):.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Multi-processors: {props.multi_processor_count}")
        
        # RTX 5090 specifications (known values)
        if "RTX 5090" in torch.cuda.get_device_name(device):
            print(f"Memory Type: GDDR6X (estimated)")
            print(f"Memory Clock Rate: ~21000 MHz (estimated)")
            print(f"Memory Bus Width: 512 bits (estimated)")
            print(f"Theoretical Memory BW: ~1344 GB/s (estimated)")
        else:
            print(f"Memory specifications: Unknown for this GPU")
        
        # PCIe information (general)
        print(f"PCIe: Likely PCIe 4.0 x16 (~32 GB/s peak)")
        
        # Current GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    # Check if pinned memory is supported
    try:
        test_tensor = torch.randn(1000).pin_memory()
        print("Pinned memory: Supported")
        del test_tensor
    except Exception as e:
        print(f"Pinned memory: Not supported ({e})")
    
    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.total / (1024**3):.1f} GB")
        print(f"Available RAM: {mem.available / (1024**3):.1f} GB")
    except ImportError:
        print("psutil not available for system memory info")
    
    # PyTorch and CUDA versions
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Check CUDA device count and current device
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")


def main():
    print("=== Precise GPU Memory Bandwidth Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device('cuda:0')
    analyze_system_info()
    
    # Test different sizes with different methods
    test_sizes = [256, 1024, 4096]  # MB
    
    for size_mb in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing {size_mb}MB transfers")
        print('='*60)
        
        try:
            test_basic_transfer(size_mb, device)
            test_pinned_memory_transfer(size_mb, device)
            test_contiguous_memory_transfer(size_mb, device)
            test_cudamemcpy_equivalent(size_mb, device)
            test_multiple_streams(size_mb, device)
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {size_mb}MB, skipping larger tests")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"Error testing {size_mb}MB: {e}")
            torch.cuda.empty_cache()
            continue
    
    print("\n=== Summary ===")
    print("Compare the different methods to identify:")
    print("1. Peak achievable bandwidth")
    print("2. Most efficient transfer method")
    print("3. Potential system bottlenecks")


if __name__ == '__main__':
    main()