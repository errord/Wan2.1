#!/usr/bin/env python3
"""
Quick performance test for basic functionality verification
Runs a larger scale test for better performance analysis
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


class LargeTestModel(nn.Module):
    """Large test model for 20GB parameter size"""
    
    def __init__(self, hidden_size: int = 8192, num_layers: int = 12, vocab_size: int = 50000):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer (large vocab for bigger model)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Multiple large transformer-like layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),  # 4x expansion
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Additional large layers to reach ~20GB
        self.extra_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(8)
        ])
        
        # Large output layer
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        print(f"Model parameter count: {self.get_param_count():,}")
        print(f"Estimated model size: {self.get_model_size_gb():.2f} GB")
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_gb(self):
        param_count = self.get_param_count()
        # 4 bytes per parameter (fp32)
        return (param_count * 4) / (1024 ** 3)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        # Apply extra layers
        for layer in self.extra_layers:
            x = x + layer(x)
        
        x = self.ln_f(x)
        return self.lm_head(x)


def quick_memory_transfer_test():
    """Quick test of memory transfer performance up to 5GB"""
    print("=== Quick Memory Transfer Test (up to 5GB) ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    
    # Check available GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    # Test sizes up to 5GB, but respect available memory
    max_test_size = min(5120, int(gpu_memory_gb * 0.8 * 1024))  # 80% of available memory
    sizes_mb = [64, 256, 1024, 2048, max_test_size]  # Up to 5GB or available memory
    
    print(f"Testing sizes: {sizes_mb} MB")
    
    for size_mb in sizes_mb:
        print(f"\nTesting {size_mb}MB transfer...")
        try:
            num_elements = (size_mb * 1024 * 1024) // 4
            
            # Test CPU to GPU
            cpu_to_gpu_times = []
            for i in range(5):  # Fewer iterations for large sizes
                print(f"  CPU->GPU iteration {i+1}/5", end="\r")
                tensor_cpu = torch.randn(num_elements, dtype=torch.float32)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                tensor_gpu = tensor_cpu.to(device)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                cpu_to_gpu_times.append((end - start) * 1000)
                del tensor_gpu
                torch.cuda.empty_cache()
            
            # Test GPU to CPU
            gpu_to_cpu_times = []
            tensor_gpu = torch.randn(num_elements, dtype=torch.float32, device=device)
            
            for i in range(5):
                print(f"  GPU->CPU iteration {i+1}/5", end="\r")
                torch.cuda.synchronize()
                start = time.perf_counter()
                tensor_cpu = tensor_gpu.cpu()
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                gpu_to_cpu_times.append((end - start) * 1000)
                del tensor_cpu
            
            del tensor_gpu
            torch.cuda.empty_cache()
            
            # Calculate statistics
            cpu_avg = statistics.mean(cpu_to_gpu_times)
            gpu_avg = statistics.mean(gpu_to_cpu_times)
            cpu_bandwidth = (size_mb / 1024) / (cpu_avg / 1000)  # GB/s
            gpu_bandwidth = (size_mb / 1024) / (gpu_avg / 1000)  # GB/s
            
            print(f"  CPU->GPU {size_mb}MB: {cpu_avg:.2f} ms, {cpu_bandwidth:.2f} GB/s")
            print(f"  GPU->CPU {size_mb}MB: {gpu_avg:.2f} ms, {gpu_bandwidth:.2f} GB/s")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM for {size_mb}MB, skipping larger sizes")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  Error testing {size_mb}MB: {e}")
            torch.cuda.empty_cache()
            break


def quick_compute_efficiency_test():
    """Quick test of compute efficiency impact with 20GB model and 5GB transfer"""
    print("\n=== Quick Compute Efficiency Test (20GB model, 5GB transfer) ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    
    # Check available GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb < 25:  # Need at least 25GB for 20GB model + overhead
        print("Warning: GPU memory < 25GB, reducing model size to fit")
        # Scale down model size based on available memory
        scale_factor = min(1.0, (gpu_memory_gb - 5) / 20)  # Reserve 5GB for overhead
        hidden_size = int(8192 * scale_factor)
        num_layers = max(1, int(12 * scale_factor))
        vocab_size = max(10000, int(50000 * scale_factor))
    else:
        hidden_size = 8192
        num_layers = 12
        vocab_size = 50000
    
    try:
        # Create large test model (targeting ~20GB)
        print("Creating large test model...")
        model = LargeTestModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size
        ).to(device)
        
        batch_size = 4  # Smaller batch for large model
        seq_len = 512
        input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        print("\nTesting baseline performance...")
        # Baseline performance
        baseline_times = []
        for i in range(10):  # Fewer iterations for large model
            print(f"  Baseline iteration {i+1}/10", end="\r")
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)
            
            del output
            torch.cuda.empty_cache()
        
        baseline_avg = statistics.mean(baseline_times)
        print(f"\nBaseline compute: {baseline_avg:.2f} ms")
        
        # Performance with concurrent large transfer (5GB)
        print("\nTesting with concurrent 5GB transfer...")
        max_transfer_gb = min(5.0, gpu_memory_gb * 0.2)  # 20% of GPU memory or 5GB
        transfer_size = int((max_transfer_gb * 1024 * 1024 * 1024) // 4)  # Convert to float32 elements
        print(f"Transfer size: {max_transfer_gb:.1f} GB")
        
        transfer_times = []
        for i in range(10):
            print(f"  Transfer iteration {i+1}/10", end="\r")
            try:
                # Create large transfer tensor on CPU
                transfer_tensor = torch.randn(transfer_size, dtype=torch.float32)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    # Start model computation
                    output = model(input_tensor)
                    
                    # Concurrent large transfer
                    gpu_transfer = transfer_tensor.to(device, non_blocking=True)
                    
                    # Ensure all operations complete
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                transfer_times.append((end - start) * 1000)
                
                del output, gpu_transfer, transfer_tensor
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n  OOM during transfer test, reducing transfer size")
                transfer_size = transfer_size // 2
                max_transfer_gb = max_transfer_gb / 2
                if max_transfer_gb < 0.5:  # Less than 500MB
                    print("  Transfer size too small, skipping transfer test")
                    break
                continue
        
        if transfer_times:
            transfer_avg = statistics.mean(transfer_times)
            overhead = ((transfer_avg - baseline_avg) / baseline_avg) * 100
            
            print(f"\nWith {max_transfer_gb:.1f}GB transfer: {transfer_avg:.2f} ms")
            print(f"Overhead: {overhead:.1f}%")
            
            # Calculate effective bandwidth during compute
            if transfer_avg > baseline_avg:
                transfer_time = transfer_avg - baseline_avg
                effective_bandwidth = max_transfer_gb / (transfer_time / 1000)
                print(f"Effective transfer bandwidth during compute: {effective_bandwidth:.2f} GB/s")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except torch.cuda.OutOfMemoryError:
        print("OOM creating large model. GPU memory insufficient for 20GB model test.")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error in compute efficiency test: {e}")
        torch.cuda.empty_cache()


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
    sizes_mb = [64, 256, 1024, 2048]  # Larger sizes for more realistic testing
    
    print("Simulating AllReduce operations...")
    for size_mb in sizes_mb:
        try:
            num_elements = (size_mb * 1024 * 1024) // 4
            
            times = []
            for _ in range(5):  # Fewer iterations for large sizes
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
                
                del tensor, tensor_copy, result
                torch.cuda.empty_cache()
            
            avg_time = statistics.mean(times)
            bandwidth = (size_mb / 1024) / (avg_time / 1000)  # GB/s (simulated)
            print(f"Simulated AllReduce {size_mb}MB: {avg_time:.2f} ms, ~{bandwidth:.2f} GB/s")
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {size_mb}MB AllReduce simulation")
            torch.cuda.empty_cache()
            break


def main():
    print("Wan2.1 Quick Performance Test (Large Scale)")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print()
    
    with timer("Total test time"):
        quick_memory_transfer_test()
        quick_compute_efficiency_test()
        quick_distributed_test()
    
    print("\nLarge scale quick test completed!")
    print("For comprehensive testing, run: python tests/performance_test.py")
    print("For precision testing, run: python tests/precision_performance_test.py")


if __name__ == '__main__':
    main()