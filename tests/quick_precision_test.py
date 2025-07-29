#!/usr/bin/env python3
"""
Quick precision test for basic verification
Tests fp32, fp16, bf16 performance with smaller models
"""

import time
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


@contextmanager
def timer(description: str):
    """Simple timer context manager"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {(end - start) * 1000:.2f} ms")


class QuickTestModel(nn.Module):
    """Smaller model for quick testing"""
    
    def __init__(self, hidden_size: int = 2048, num_layers: int = 6, vocab_size: int = 10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        x = self.ln_f(x)
        return self.lm_head(x)


def test_precision_performance():
    """Test performance across fp32, fp16, bf16"""
    print("=== Quick Precision Performance Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    batch_size = 8
    seq_len = 512
    num_iterations = 10
    
    precisions = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    
    results = {}
    
    for precision_name, dtype in precisions.items():
        print(f"\nTesting {precision_name.upper()}...")
        
        try:
            # Create model
            model = QuickTestModel().to(device=device, dtype=dtype)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Test data
            input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
            targets = torch.randint(0, 10000, (batch_size, seq_len), device=device)
            
            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(dtype != torch.float32)):
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            memory_usage = []
            
            for _ in range(num_iterations):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
                
                optimizer.zero_grad()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.cuda.amp.autocast(enabled=(dtype != torch.float32)):
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append((end - start) * 1000)
                peak_mem = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_mem / 1024**2)  # MB
            
            # Results
            avg_time = statistics.mean(times)
            avg_memory = statistics.mean(memory_usage)
            throughput = (batch_size * seq_len) / (avg_time / 1000)  # tokens/s
            
            results[precision_name] = {
                'time': avg_time,
                'memory': avg_memory,
                'throughput': throughput
            }
            
            print(f"  Time: {avg_time:.2f} ms")
            print(f"  Memory: {avg_memory:.1f} MB")
            print(f"  Throughput: {throughput:.0f} tokens/s")
            
            # Model size estimation
            param_count = sum(p.numel() for p in model.parameters())
            bytes_per_param = 4 if dtype == torch.float32 else 2
            model_size_mb = (param_count * bytes_per_param) / (1024**2)
            print(f"  Model size: {model_size_mb:.1f} MB ({param_count:,} parameters)")
            
            del model, optimizer
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM error for {precision_name}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  Error for {precision_name}: {e}")
            continue
    
    # Compare results
    if len(results) > 1:
        print("\n=== Performance Comparison ===")
        if 'fp32' in results:
            baseline = results['fp32']
            print(f"{'Precision':<8} {'Speedup':<8} {'Memory Ratio':<12} {'Throughput Gain':<15}")
            print("-" * 50)
            
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in results:
                    r = results[precision]
                    speedup = baseline['time'] / r['time']
                    memory_ratio = r['memory'] / baseline['memory']
                    throughput_gain = r['throughput'] / baseline['throughput']
                    
                    print(f"{precision.upper():<8} {speedup:.2f}x    {memory_ratio:.2f}x        {throughput_gain:.2f}x")


def test_mixed_precision_impact():
    """Test impact of mixed precision training"""
    print("\n=== Mixed Precision Impact Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    batch_size = 16
    seq_len = 1024
    
    # Test with and without autocast
    model = QuickTestModel(hidden_size=1024, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    
    scenarios = [
        ("FP32", False),
        ("Mixed Precision", True)
    ]
    
    for scenario_name, use_autocast in scenarios:
        times = []
        memory_usage = []
        
        for _ in range(5):  # Fewer iterations for speed
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.cuda.amp.autocast(enabled=use_autocast):
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
        
        avg_time = statistics.mean(times)
        avg_memory = statistics.mean(memory_usage)
        
        print(f"{scenario_name}: {avg_time:.2f} ms, {avg_memory:.1f} MB")


def main():
    print("Wan2.1 Quick Precision Test")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print()
    
    with timer("Total test time"):
        test_precision_performance()
        test_mixed_precision_impact()
    
    print("\nQuick precision test completed!")
    print("For comprehensive testing, run: python tests/precision_performance_test.py")
    print("Or use the shell script: ./tests/run_precision_test.sh")


if __name__ == '__main__':
    main()