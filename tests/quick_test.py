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
    """Large test model for ~20GB parameter size"""
    
    def __init__(self, hidden_size: int = 6144, num_layers: int = 8, vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer-like layers with controlled size
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),  # 4x expansion
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Additional layers to reach target size more precisely
        # Calculate remaining parameters needed for ~20GB
        current_params = self._estimate_current_params()
        target_params = int((20 * 1024**3) / 4)  # 20GB in float32 parameters
        
        if current_params < target_params:
            remaining_params = target_params - current_params
            # Add one large linear layer to reach target
            extra_dim = int((remaining_params / hidden_size) ** 0.5)
            extra_dim = min(extra_dim, 16384)  # Cap at reasonable size
            if extra_dim > hidden_size:
                self.extra_layer = nn.Linear(hidden_size, extra_dim)
            else:
                self.extra_layer = None
        else:
            self.extra_layer = None
        
        # Output layer
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Report actual model statistics
        actual_params = self.get_param_count()
        actual_size_gb = self.get_model_size_gb()
        print(f"Model parameter count: {actual_params:,}")
        print(f"Actual model size: {actual_size_gb:.2f} GB")
        
        # Validate size is reasonable
        if actual_size_gb > 22:
            print(f"Warning: Model size {actual_size_gb:.2f}GB may be too large!")
    
    def _estimate_current_params(self):
        """Estimate parameters without the extra layer"""
        # Embedding
        embed_params = self.embedding.weight.numel()
        
        # Transformer layers
        layer_params = 0
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    layer_params += module.weight.numel()
                    if module.bias is not None:
                        layer_params += module.bias.numel()
                elif isinstance(module, nn.LayerNorm):
                    layer_params += module.weight.numel() + module.bias.numel()
        
        # Output layers (estimated)
        ln_params = self.hidden_size * 2  # LayerNorm
        lm_head_params = self.hidden_size * 32000  # vocab_size
        
        return embed_params + layer_params + ln_params + lm_head_params
    
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
        
        # Apply extra layer if exists
        if self.extra_layer is not None:
            x_extra = self.extra_layer(x)
            # Use only a subset to maintain original dimensions
            x = x + x_extra[:, :, :self.hidden_size]
        
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
    """Quick test of compute efficiency impact with ~20GB model and large transfer"""
    print("\n=== Quick Compute Efficiency Test (~20GB model, large transfer) ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    
    # Check available GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    # Calculate safe model and transfer sizes
    # Reserve memory: model + activations + transfer + overhead
    available_memory = gpu_memory_gb * 0.9  # Use 90% of available memory
    
    # Target: ~18GB model + ~3GB activations + ~5GB transfer = ~26GB total
    if available_memory < 26:
        print(f"Warning: Only {available_memory:.1f}GB available, adjusting sizes...")
        model_target_gb = max(8, available_memory * 0.6)  # 60% for model
        transfer_target_gb = max(1, available_memory * 0.2)  # 20% for transfer
    else:
        model_target_gb = 18  # Target 18GB model (safer than 20GB)
        transfer_target_gb = 5   # Target 5GB transfer
    
    # Scale model parameters based on available memory
    if gpu_memory_gb < 30:
        print("Scaling model size for available memory...")
        scale_factor = min(1.0, model_target_gb / 18)
        hidden_size = max(1024, int(6144 * scale_factor))
        num_layers = max(2, int(8 * scale_factor))
        vocab_size = max(5000, int(32000 * scale_factor))
    else:
        hidden_size = 6144
        num_layers = 8
        vocab_size = 32000
    
    print(f"Target model size: {model_target_gb:.1f}GB, Transfer size: {transfer_target_gb:.1f}GB")
    
    try:
        # Create model with target size
        print("Creating test model...")
        model = LargeTestModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size
        ).to(device)
        
        # Check actual model size and adjust if needed
        actual_model_size = model.get_model_size_gb()
        if actual_model_size > model_target_gb * 1.2:  # 20% tolerance
            print(f"Model too large ({actual_model_size:.1f}GB), recreating with smaller parameters...")
            del model
            torch.cuda.empty_cache()
            
            # Reduce model size more aggressively
            hidden_size = max(1024, int(hidden_size * 0.7))
            num_layers = max(2, int(num_layers * 0.7))
            vocab_size = max(5000, int(vocab_size * 0.8))
            
            model = LargeTestModel(
                hidden_size=hidden_size,
                num_layers=num_layers,
                vocab_size=vocab_size
            ).to(device)
        
        # Use smaller batch and sequence length for large model
        batch_size = 2
        seq_len = 256
        input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        print("\nTesting baseline performance...")
        # Baseline performance with memory cleanup
        baseline_times = []
        for i in range(8):  # Fewer iterations
            print(f"  Baseline iteration {i+1}/8", end="\r")
            torch.cuda.empty_cache()  # Clean before each iteration
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
        
        # Calculate safe transfer size
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Free memory in GB
        safe_transfer_gb = min(transfer_target_gb, free_memory * 0.8)  # Use 80% of free memory
        transfer_size = int((safe_transfer_gb * 1024 * 1024 * 1024) // 4)  # Convert to float32 elements
        
        print(f"\nTesting with concurrent {safe_transfer_gb:.1f}GB transfer...")
        print(f"Free GPU memory: {free_memory:.1f}GB")
        
        if safe_transfer_gb < 0.5:
            print("Insufficient memory for transfer test, skipping...")
            return
        
        transfer_times = []
        successful_iterations = 0
        
        for i in range(8):
            print(f"  Transfer iteration {i+1}/8", end="\r")
            try:
                torch.cuda.empty_cache()  # Clean before each iteration
                
                # Create transfer tensor on CPU
                transfer_tensor = torch.randn(transfer_size, dtype=torch.float32)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    # Start model computation
                    output = model(input_tensor)
                    
                    # Concurrent transfer
                    gpu_transfer = transfer_tensor.to(device, non_blocking=True)
                    
                    # Ensure all operations complete
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                transfer_times.append((end - start) * 1000)
                successful_iterations += 1
                
                del output, gpu_transfer, transfer_tensor
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n  OOM during iteration {i+1}, reducing transfer size")
                transfer_size = int(transfer_size * 0.7)  # Reduce by 30%
                safe_transfer_gb = safe_transfer_gb * 0.7
                if safe_transfer_gb < 0.3:  # Less than 300MB
                    print("  Transfer size too small, stopping transfer test")
                    break
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"\n  Error in iteration {i+1}: {e}")
                torch.cuda.empty_cache()
                continue
        
        if transfer_times and successful_iterations >= 3:
            transfer_avg = statistics.mean(transfer_times)
            overhead = ((transfer_avg - baseline_avg) / baseline_avg) * 100
            
            print(f"\nWith {safe_transfer_gb:.1f}GB transfer: {transfer_avg:.2f} ms")
            print(f"Overhead: {overhead:.1f}%")
            print(f"Successful iterations: {successful_iterations}/8")
            
            # Calculate effective bandwidth during compute
            if transfer_avg > baseline_avg:
                transfer_time = transfer_avg - baseline_avg
                effective_bandwidth = safe_transfer_gb / (transfer_time / 1000)
                print(f"Effective transfer bandwidth during compute: {effective_bandwidth:.2f} GB/s")
        else:
            print(f"\nInsufficient successful iterations ({successful_iterations}) for reliable results")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except torch.cuda.OutOfMemoryError:
        print("OOM creating model. GPU memory insufficient for large model test.")
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