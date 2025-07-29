#!/usr/bin/env python3
"""
Precision Performance Testing for Wan2.1
Tests fp32, fp16, bf16 computation performance and distributed precision impact
"""

import os
import sys
import time
import argparse
import logging
import statistics
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
import torch.nn as nn
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@contextmanager
def timer(description: str):
    """Context manager for timing operations"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{description}: {(end - start) * 1000:.2f} ms")


@contextmanager
def memory_tracker(description: str):
    """Context manager for tracking CUDA memory usage"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    start = time.perf_counter()
    
    yield
    
    end = time.perf_counter()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    
    logger.info(f"{description}:")
    logger.info(f"  Time: {(end - start) * 1000:.2f} ms")
    logger.info(f"  Memory - Peak: {peak_mem/1024/1024:.2f} MB, Delta: {(peak_mem - start_mem)/1024/1024:.2f} MB")


class LargeTestModel(nn.Module):
    """Large test model designed to use significant GPU memory for RTX 5090 32GB"""
    
    def __init__(self, hidden_size: int = 8192, num_layers: int = 24, vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)  # [batch, seq_len, hidden_size]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_model_size_gb(self, dtype=torch.float32):
        """Calculate model size in GB for given dtype"""
        param_size = sum(p.numel() for p in self.parameters())
        bytes_per_param = 4 if dtype == torch.float32 else 2  # fp32=4bytes, fp16/bf16=2bytes
        return (param_size * bytes_per_param) / (1024 ** 3)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    
    def __init__(self, hidden_size: int, num_heads: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads or hidden_size // 128  # Default head dimension = 128
        self.head_dim = hidden_size // self.num_heads
        
        # Multi-head attention
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, self.num_heads)
        
        # MLP
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)
    
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        
        return output


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(self, hidden_size: int, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class PrecisionPerformanceTester:
    """Test performance across different precisions (fp32, fp16, bf16)"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results: Dict[str, Dict[str, List[float]]] = {}
        
        # Calculate optimal model size for RTX 5090 32GB
        self.target_memory_gb = 28  # Leave 4GB for activations and overhead
        self.model_configs = self._calculate_model_configs()
    
    def _calculate_model_configs(self) -> Dict[str, Dict]:
        """Calculate model configurations for different precisions targeting ~28GB"""
        configs = {}
        
        # Start with much smaller base config to avoid OOM
        base_hidden = 4096  # Reduced from 8192
        base_layers = 8     # Reduced from 24
        
        # Test model to get parameter count
        test_model = LargeTestModel(hidden_size=base_hidden, num_layers=base_layers)
        
        # Target much smaller memory usage (leave more room for activations)
        target_memory_gb = 20  # Reduced from 28
        
        for precision in ['fp32', 'fp16', 'bf16']:
            dtype = torch.float32 if precision == 'fp32' else torch.float16
            model_size_gb = test_model.get_model_size_gb(dtype)
            
            # Scale model size to fit target memory
            scale_factor = (target_memory_gb / model_size_gb) ** 0.5
            
            # Adjust hidden size and layers with more conservative scaling
            scaled_hidden = int(base_hidden * scale_factor * 0.8)  # Extra conservative factor
            scaled_layers = max(1, int(base_layers * scale_factor * 0.8))
            
            # Ensure hidden size is divisible by attention heads
            scaled_hidden = (scaled_hidden // 128) * 128
            
            # For fp16/bf16, we can use slightly larger models
            if precision in ['fp16', 'bf16']:
                scaled_hidden = int(scaled_hidden * 1.4)
                scaled_layers = int(scaled_layers * 1.2)
                scaled_hidden = (scaled_hidden // 128) * 128
            
            configs[precision] = {
                'hidden_size': scaled_hidden,
                'num_layers': scaled_layers,
                'estimated_size_gb': model_size_gb * (scale_factor ** 2)
            }
            
            logger.info(f"{precision}: hidden={scaled_hidden}, layers={scaled_layers}, "
                       f"estimated_size={configs[precision]['estimated_size_gb']:.1f}GB")
        
        del test_model
        torch.cuda.empty_cache()
        return configs
    
    def test_precision_performance(self, batch_sizes: List[int] = [2, 4], # Reduced batch sizes
                                 seq_lengths: List[int] = [512, 1024],   # Reduced seq lengths 
                                 num_iterations: int = 10):              # Reduced iterations
        """Test forward and backward pass performance across precisions"""
        logger.info("=== Testing Precision Performance ===")
        
        precisions = {
            'fp32': torch.float32,
            'fp16': torch.float16, 
            'bf16': torch.bfloat16
        }
        
        for precision_name, dtype in precisions.items():
            if precision_name not in self.model_configs:
                continue
                
            logger.info(f"\nTesting {precision_name.upper()}...")
            
            # Initialize results structure
            self.results[precision_name] = {
                'forward_times': [],
                'backward_times': [],
                'total_times': [],
                'throughput': [],
                'memory_usage': []
            }
            
            config = self.model_configs[precision_name]
            
            try:
                # Create model with calculated configuration
                model = LargeTestModel(
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers']
                ).to(self.device, dtype=dtype)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                successful_runs = 0
                
                for batch_size in batch_sizes:
                    for seq_len in seq_lengths:
                        if successful_runs >= 3:  # Limit successful runs to avoid OOM accumulation
                            break
                            
                        try:
                            # Generate test data
                            input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
                            target_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
                            
                            # Warmup (reduced)
                            for _ in range(2):
                                optimizer.zero_grad()
                                with torch.amp.autocast('cuda', enabled=(dtype != torch.float32)):
                                    logits = model(input_ids)
                                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                                loss.backward()
                                optimizer.step()
                                torch.cuda.empty_cache()
                            
                            # Benchmark
                            forward_times = []
                            backward_times = []
                            total_times = []
                            
                            for _ in range(num_iterations):
                                optimizer.zero_grad()
                                torch.cuda.empty_cache()  # Clear cache before each iteration
                                
                                # Measure memory before
                                torch.cuda.synchronize()
                                start_mem = torch.cuda.memory_allocated()
                                
                                # Forward pass
                                start_total = time.perf_counter()
                                torch.cuda.synchronize()
                                start_forward = time.perf_counter()
                                
                                with torch.amp.autocast('cuda', enabled=(dtype != torch.float32)):
                                    logits = model(input_ids)
                                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                                
                                torch.cuda.synchronize()
                                end_forward = time.perf_counter()
                                
                                # Backward pass
                                start_backward = time.perf_counter()
                                loss.backward()
                                torch.cuda.synchronize()
                                end_backward = time.perf_counter()
                                
                                optimizer.step()
                                torch.cuda.synchronize()
                                end_total = time.perf_counter()
                                
                                # Record timings
                                forward_time = (end_forward - start_forward) * 1000
                                backward_time = (end_backward - start_backward) * 1000
                                total_time = (end_total - start_total) * 1000
                                
                                forward_times.append(forward_time)
                                backward_times.append(backward_time)
                                total_times.append(total_time)
                                
                                # Calculate throughput
                                tokens_processed = batch_size * seq_len
                                throughput = tokens_processed / (total_time / 1000)  # tokens/sec
                                
                                # Memory usage
                                peak_mem = torch.cuda.max_memory_allocated()
                                memory_usage = (peak_mem - start_mem) / (1024**3)  # GB
                                
                                self.results[precision_name]['forward_times'].append(forward_time)
                                self.results[precision_name]['backward_times'].append(backward_time)
                                self.results[precision_name]['total_times'].append(total_time)
                                self.results[precision_name]['throughput'].append(throughput)
                                self.results[precision_name]['memory_usage'].append(memory_usage)
                            
                            successful_runs += 1
                            logger.info(f"  Batch {batch_size}, Seq {seq_len}: "
                                      f"Forward: {statistics.mean(forward_times):.2f}ms, "
                                      f"Backward: {statistics.mean(backward_times):.2f}ms, "
                                      f"Total: {statistics.mean(total_times):.2f}ms")
                            
                        except torch.cuda.OutOfMemoryError:
                            logger.warning(f"OOM for {precision_name} batch_size={batch_size}, seq_len={seq_len}")
                            torch.cuda.empty_cache()
                            continue
                    
                    if successful_runs >= 3:
                        break
                
                del model, optimizer
                torch.cuda.empty_cache()
                
                if successful_runs == 0:
                    logger.warning(f"No successful runs for {precision_name}")
                else:
                    logger.info(f"Completed {successful_runs} successful configurations for {precision_name}")
                    
            except Exception as e:
                logger.error(f"Error testing {precision_name}: {e}")
                torch.cuda.empty_cache()
    
    def analyze_precision_results(self):
        """Analyze and display precision performance results"""
        logger.info("\n=== Precision Performance Analysis ===")
        
        # Check if we have any valid results
        valid_precisions = []
        for precision in ['fp32', 'fp16', 'bf16']:
            if (precision in self.results and 
                self.results[precision]['total_times'] and 
                len(self.results[precision]['total_times']) > 0):
                valid_precisions.append(precision)
        
        if not valid_precisions:
            logger.error("No valid test results found! All tests failed due to OOM.")
            logger.error("Model configurations may be too large for available GPU memory.")
            logger.error("Consider:")
            logger.error("1. Reducing hidden_size and num_layers in model configs")
            logger.error("2. Using smaller batch_sizes and seq_lengths")
            logger.error("3. Ensuring sufficient GPU memory is available")
            return
        
        logger.info(f"Valid results for precisions: {valid_precisions}")
        
        # Use the first valid precision as baseline
        baseline_precision = valid_precisions[0]
        baseline_total = statistics.mean(self.results[baseline_precision]['total_times'])
        baseline_throughput = statistics.mean(self.results[baseline_precision]['throughput'])
        baseline_memory = statistics.mean(self.results[baseline_precision]['memory_usage'])
        
        logger.info(f"Using {baseline_precision.upper()} as baseline")
        logger.info(f"{'Precision':<10} {'Speedup':<10} {'Throughput':<15} {'Memory':<15}")
        logger.info("-" * 60)
        
        for precision in valid_precisions:
            if self.results[precision]['total_times']:
                avg_total = statistics.mean(self.results[precision]['total_times'])
                avg_throughput = statistics.mean(self.results[precision]['throughput'])
                avg_memory = statistics.mean(self.results[precision]['memory_usage'])
                
                speedup = baseline_total / avg_total
                throughput_ratio = avg_throughput / baseline_throughput
                memory_ratio = avg_memory / baseline_memory
                
                logger.info(f"{precision.upper():<10} {speedup:.2f}x     "
                          f"{throughput_ratio:.2f}x         {memory_ratio:.2f}x")
        
        # Additional detailed statistics
        logger.info("\n=== Detailed Statistics ===")
        for precision in valid_precisions:
            results = self.results[precision]
            if results['total_times']:
                logger.info(f"\n{precision.upper()}:")
                logger.info(f"  Forward:    {statistics.mean(results['forward_times']):.2f} ± "
                          f"{statistics.stdev(results['forward_times']) if len(results['forward_times']) > 1 else 0:.2f} ms")
                logger.info(f"  Backward:   {statistics.mean(results['backward_times']):.2f} ± "
                          f"{statistics.stdev(results['backward_times']) if len(results['backward_times']) > 1 else 0:.2f} ms")
                logger.info(f"  Total:      {statistics.mean(results['total_times']):.2f} ± "
                          f"{statistics.stdev(results['total_times']) if len(results['total_times']) > 1 else 0:.2f} ms")
                logger.info(f"  Throughput: {statistics.mean(results['throughput']):.1f} ± "
                          f"{statistics.stdev(results['throughput']) if len(results['throughput']) > 1 else 0:.1f} tokens/s")
                logger.info(f"  Memory:     {statistics.mean(results['memory_usage']):.2f} ± "
                          f"{statistics.stdev(results['memory_usage']) if len(results['memory_usage']) > 1 else 0:.2f} GB")


class DistributedPrecisionTester:
    """Test how network communication affects precision performance gains"""
    
    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.results: Dict[str, Dict[str, List[float]]] = {}
    
    def setup_distributed(self, master_addr: str = "localhost", master_port: str = "12355"):
        """Setup distributed environment"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.device)
    
    def test_distributed_precision_performance(self, 
                                             batch_size: int = 8, 
                                             seq_len: int = 1024,
                                             num_iterations: int = 20):
        """Test FSDP performance across different precisions"""
        logger.info(f"=== Distributed Precision Testing (Rank {self.rank}) ===")
        
        precisions = {
            'fp32': (torch.float32, torch.float32),
            'fp16': (torch.float16, torch.float32),  # param_dtype, reduce_dtype
            'bf16': (torch.bfloat16, torch.float32)
        }
        
        for precision_name, (param_dtype, reduce_dtype) in precisions.items():
            if self.rank == 0:
                logger.info(f"\nTesting distributed {precision_name.upper()}...")
            
            try:
                # Create smaller model for distributed testing
                model = LargeTestModel(hidden_size=4096, num_layers=12)
                
                # Wrap with FSDP using specific precision
                mixed_precision = MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    buffer_dtype=torch.float32
                )
                
                model = FSDP(
                    model,
                    device_id=self.device,
                    mixed_precision=mixed_precision,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                )
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                # Generate test data
                input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
                target_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
                
                # Warmup
                for _ in range(3):
                    optimizer.zero_grad()
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    loss.backward()
                    optimizer.step()
                
                dist.barrier()
                
                # Benchmark
                compute_times = []
                communication_times = []
                total_times = []
                
                for i in range(num_iterations):
                    optimizer.zero_grad()
                    
                    torch.cuda.synchronize()
                    start_total = time.perf_counter()
                    
                    # Forward pass (includes parameter all-gather)
                    start_compute = time.perf_counter()
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    torch.cuda.synchronize()
                    end_compute = time.perf_counter()
                    
                    # Backward pass (includes gradient reduce-scatter)
                    start_backward = time.perf_counter()
                    loss.backward()
                    torch.cuda.synchronize()
                    end_backward = time.perf_counter()
                    
                    optimizer.step()
                    torch.cuda.synchronize()
                    end_total = time.perf_counter()
                    
                    # Estimate communication time as total - pure compute
                    compute_time = (end_compute - start_compute) + (end_backward - start_backward)
                    total_time = end_total - start_total
                    communication_time = total_time - compute_time
                    
                    compute_times.append(compute_time * 1000)
                    communication_times.append(max(0, communication_time * 1000))  # Avoid negative
                    total_times.append(total_time * 1000)
                
                # Store results
                self.results[precision_name] = {
                    'compute_times': compute_times,
                    'communication_times': communication_times,
                    'total_times': total_times
                }
                
                if self.rank == 0:
                    avg_compute = statistics.mean(compute_times)
                    avg_comm = statistics.mean(communication_times)
                    avg_total = statistics.mean(total_times)
                    comm_overhead = (avg_comm / avg_total) * 100
                    
                    logger.info(f"{precision_name.upper()} Distributed Results:")
                    logger.info(f"  Compute: {avg_compute:.2f} ms")
                    logger.info(f"  Communication: {avg_comm:.2f} ms") 
                    logger.info(f"  Total: {avg_total:.2f} ms")
                    logger.info(f"  Communication overhead: {comm_overhead:.1f}%")
                
                # Cleanup
                del model, optimizer
                torch.cuda.empty_cache()
                dist.barrier()
                
            except Exception as e:
                if self.rank == 0:
                    logger.error(f"Error testing {precision_name}: {e}")
                dist.barrier()
                continue
    
    def analyze_distributed_precision_impact(self):
        """Analyze how communication affects precision performance gains"""
        if self.rank != 0:
            return
            
        logger.info("\n=== Distributed Precision Impact Analysis ===")
        
        if not self.results:
            logger.warning("No distributed results to analyze")
            return
        
        # Compare precision efficiency in distributed setting
        if 'fp32' in self.results:
            fp32_total = statistics.mean(self.results['fp32']['total_times'])
            fp32_compute = statistics.mean(self.results['fp32']['compute_times'])
            fp32_comm = statistics.mean(self.results['fp32']['communication_times'])
            
            logger.info("Distributed Precision Comparison:")
            logger.info(f"{'Precision':<10} {'Total Time':<12} {'Compute Speedup':<16} {'Comm Overhead':<15} {'Net Benefit':<12}")
            logger.info("-" * 75)
            
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in self.results:
                    total_time = statistics.mean(self.results[precision]['total_times'])
                    compute_time = statistics.mean(self.results[precision]['compute_times'])
                    comm_time = statistics.mean(self.results[precision]['communication_times'])
                    
                    compute_speedup = fp32_compute / compute_time
                    total_speedup = fp32_total / total_time
                    comm_overhead = (comm_time / total_time) * 100
                    
                    logger.info(f"{precision.upper():<10} {total_time:.2f}ms     "
                              f"{compute_speedup:.2f}x           "
                              f"{comm_overhead:.1f}%           "
                              f"{total_speedup:.2f}x")
            
            # Analysis of whether communication negates precision benefits
            if 'fp16' in self.results and 'bf16' in self.results:
                logger.info("\nPrecision Benefit Analysis:")
                
                for precision in ['fp16', 'bf16']:
                    compute_benefit = fp32_compute / statistics.mean(self.results[precision]['compute_times'])
                    total_benefit = fp32_total / statistics.mean(self.results[precision]['total_times'])
                    benefit_retention = (total_benefit / compute_benefit) * 100
                    
                    logger.info(f"{precision.upper()}: Compute benefit {compute_benefit:.2f}x, "
                              f"Total benefit {total_benefit:.2f}x, "
                              f"Retention {benefit_retention:.1f}%")
    
    def cleanup(self):
        """Cleanup distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()


def run_precision_tests(args):
    """Run precision performance tests"""
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)
    
    logger.info(f"Running precision tests on {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Test single GPU precision performance
    tester = PrecisionPerformanceTester(device)
    tester.test_precision_performance(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_iterations=args.num_iterations
    )
    tester.analyze_precision_results()
    
    # Save results
    if args.save_results:
        import json
        with open(f'precision_results_gpu_{args.gpu_id}.json', 'w') as f:
            json.dump(tester.results, f, indent=2)
        logger.info(f"Results saved to precision_results_gpu_{args.gpu_id}.json")


def run_distributed_precision_worker(rank: int, world_size: int, args):
    """Worker function for distributed precision testing"""
    device = torch.device(f'cuda:{rank}')
    
    tester = DistributedPrecisionTester(rank, world_size, device)
    
    try:
        tester.setup_distributed(args.master_addr, args.master_port)
        
        if rank == 0:
            logger.info(f"Running distributed precision tests with {world_size} GPUs")
        
        tester.test_distributed_precision_performance(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_iterations=args.num_iterations
        )
        tester.analyze_distributed_precision_impact()
        
        # Save results from rank 0
        if rank == 0 and args.save_results:
            import json
            with open(f'distributed_precision_results_{world_size}gpus.json', 'w') as f:
                json.dump(tester.results, f, indent=2)
            logger.info(f"Distributed results saved to distributed_precision_results_{world_size}gpus.json")
    
    finally:
        tester.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Precision Performance Testing for Wan2.1')
    parser.add_argument('--test_type', choices=['single', 'distributed', 'all'], 
                       default='all', help='Type of test to run')
    parser.add_argument('--gpu_id', type=int, default=0, 
                       help='GPU ID for single GPU tests')
    parser.add_argument('--num_gpus', type=int, default=2,
                       help='Number of GPUs for distributed tests')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[1024, 2048],
                       help='Sequence lengths to test')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for distributed tests')
    parser.add_argument('--seq_len', type=int, default=1024,
                       help='Sequence length for distributed tests')
    parser.add_argument('--num_iterations', type=int, default=20,
                       help='Number of iterations for each test')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='Master address for distributed testing')
    parser.add_argument('--master_port', type=str, default='12356',
                       help='Master port for distributed testing')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON files')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)
    
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    if args.test_type in ['single', 'all']:
        logger.info("Starting single GPU precision tests...")
        run_precision_tests(args)
    
    if args.test_type in ['distributed', 'all']:
        if args.num_gpus > torch.cuda.device_count():
            logger.error(f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available")
            sys.exit(1)
        
        logger.info("Starting distributed precision tests...")
        mp.spawn(
            run_distributed_precision_worker,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    
    logger.info("All precision tests completed!")


if __name__ == '__main__':
    main()