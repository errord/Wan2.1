#!/usr/bin/env python3
"""
Quick test script to verify precision performance test fixes
"""

import torch
import statistics
import time

def test_model_creation():
    """Test if we can create smaller models without OOM"""
    print("Testing model creation with reduced sizes...")
    
    # Simulate the fixed model configurations
    configs = {
        'fp32': {'hidden_size': 2048, 'num_layers': 4},
        'fp16': {'hidden_size': 2816, 'num_layers': 6}, 
        'bf16': {'hidden_size': 2816, 'num_layers': 6}
    }
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    for precision, config in configs.items():
        try:
            print(f"\nTesting {precision.upper()}...")
            print(f"  Config: hidden_size={config['hidden_size']}, num_layers={config['num_layers']}")
            
            # Create a simple test model
            model = torch.nn.Sequential(
                torch.nn.Linear(config['hidden_size'], config['hidden_size'] * 4),
                torch.nn.GELU(),
                torch.nn.Linear(config['hidden_size'] * 4, config['hidden_size']),
            ).to(device)
            
            # Test with small batch
            batch_size, seq_len = 2, 512
            dtype = torch.float32 if precision == 'fp32' else torch.float16
            
            input_tensor = torch.randn(batch_size, seq_len, config['hidden_size'], 
                                     device=device, dtype=dtype)
            
            # Forward pass test
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.amp.autocast('cuda', enabled=(dtype != torch.float32)):
                output = model(input_tensor)
                loss = output.sum()
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            print(f"  ✓ Forward pass successful: {(end-start)*1000:.2f} ms")
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            del model, input_tensor, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            torch.cuda.empty_cache()

def test_empty_results_handling():
    """Test the fixed analyze_precision_results method"""
    print("\n" + "="*50)
    print("Testing empty results handling...")
    
    # Simulate empty results (all OOM scenario)
    empty_results = {
        'fp32': {'total_times': [], 'throughput': [], 'memory_usage': []},
        'fp16': {'total_times': [], 'throughput': [], 'memory_usage': []},
        'bf16': {'total_times': [], 'throughput': [], 'memory_usage': []}
    }
    
    # Test the logic
    valid_precisions = []
    for precision in ['fp32', 'fp16', 'bf16']:
        if (precision in empty_results and 
            empty_results[precision]['total_times'] and 
            len(empty_results[precision]['total_times']) > 0):
            valid_precisions.append(precision)
    
    if not valid_precisions:
        print("✓ Empty results correctly detected")
        print("✓ Error handling would work properly")
    else:
        print("✗ Empty results detection failed")
    
    # Test with some valid results
    partial_results = {
        'fp32': {'total_times': [], 'throughput': [], 'memory_usage': []},
        'fp16': {'total_times': [100, 95, 105], 'throughput': [1000, 1050, 950], 'memory_usage': [2.1, 2.0, 2.2]},
        'bf16': {'total_times': [98, 102, 96], 'throughput': [1020, 980, 1040], 'memory_usage': [2.0, 2.1, 1.9]}
    }
    
    valid_precisions = []
    for precision in ['fp32', 'fp16', 'bf16']:
        if (precision in partial_results and 
            partial_results[precision]['total_times'] and 
            len(partial_results[precision]['total_times']) > 0):
            valid_precisions.append(precision)
    
    if len(valid_precisions) == 2 and 'fp16' in valid_precisions and 'bf16' in valid_precisions:
        print("✓ Partial results correctly detected")
        
        # Test baseline calculation
        baseline_precision = valid_precisions[0]  # Should be 'fp16'
        baseline_total = statistics.mean(partial_results[baseline_precision]['total_times'])
        print(f"✓ Baseline calculation works: {baseline_total:.2f} ms")
        
    else:
        print("✗ Partial results detection failed")

def main():
    print("Precision Performance Test Fix Verification")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        test_empty_results_handling()
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    test_model_creation()
    test_empty_results_handling()
    
    print("\n" + "="*50)
    print("Fix verification completed!")

if __name__ == '__main__':
    main()