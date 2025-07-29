#!/usr/bin/env python3
"""
Performance testing script for Wan2.1 model
Tests:
1. Memory-GPU transfer latency for different data sizes
2. GPU compute efficiency impact from memory transfers
3. Multi-node multi-GPU FSDP network performance simulation
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
from torch.distributed.fsdp import ShardingStrategy
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
def cuda_memory_tracker(description: str):
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
    logger.info(f"  Memory - Start: {start_mem/1024/1024:.2f} MB, End: {end_mem/1024/1024:.2f} MB")
    logger.info(f"  Memory - Peak: {peak_mem/1024/1024:.2f} MB, Delta: {(peak_mem - start_mem)/1024/1024:.2f} MB")


class MemoryTransferTester:
    """Test memory to GPU and GPU to memory transfer performance"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results: Dict[str, List[float]] = {}
    
    def test_transfer_latency(self, sizes: List[int], num_iterations: int = 100):
        """Test transfer latency for different data sizes"""
        logger.info("=== Testing Memory-GPU Transfer Latency ===")
        
        for size_mb in sizes:
            # Calculate tensor size (assuming float32)
            num_elements = (size_mb * 1024 * 1024) // 4
            tensor_shape = (num_elements,)
            
            # Test CPU to GPU transfer
            cpu_to_gpu_times = []
            for _ in range(num_iterations):
                # Create tensor on CPU
                tensor_cpu = torch.randn(tensor_shape, dtype=torch.float32)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                tensor_gpu = tensor_cpu.to(self.device, non_blocking=False)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                cpu_to_gpu_times.append((end - start) * 1000)  # Convert to ms
                del tensor_gpu
                torch.cuda.empty_cache()
            
            # Test GPU to CPU transfer
            gpu_to_cpu_times = []
            tensor_gpu = torch.randn(tensor_shape, dtype=torch.float32, device=self.device)
            
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                tensor_cpu = tensor_gpu.cpu()
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                gpu_to_cpu_times.append((end - start) * 1000)  # Convert to ms
                del tensor_cpu
            
            del tensor_gpu
            torch.cuda.empty_cache()
            
            # Store results
            self.results[f"cpu_to_gpu_{size_mb}MB"] = cpu_to_gpu_times
            self.results[f"gpu_to_cpu_{size_mb}MB"] = gpu_to_cpu_times
            
            # Log statistics
            cpu_avg = statistics.mean(cpu_to_gpu_times)
            cpu_std = statistics.stdev(cpu_to_gpu_times) if len(cpu_to_gpu_times) > 1 else 0
            gpu_avg = statistics.mean(gpu_to_cpu_times)
            gpu_std = statistics.stdev(gpu_to_cpu_times) if len(gpu_to_cpu_times) > 1 else 0
            
            bandwidth_cpu_to_gpu = size_mb / (cpu_avg / 1000)  # MB/s
            bandwidth_gpu_to_cpu = size_mb / (gpu_avg / 1000)  # MB/s
            
            logger.info(f"Size: {size_mb}MB")
            logger.info(f"  CPU->GPU: {cpu_avg:.2f}±{cpu_std:.2f} ms, Bandwidth: {bandwidth_cpu_to_gpu:.2f} MB/s")
            logger.info(f"  GPU->CPU: {gpu_avg:.2f}±{gpu_std:.2f} ms, Bandwidth: {bandwidth_gpu_to_cpu:.2f} MB/s")


class ComputeEfficiencyTester:
    """Test how memory transfers impact GPU compute efficiency"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results: Dict[str, List[float]] = {}
    
    def create_test_model(self, hidden_size: int = 1024) -> nn.Module:
        """Create a simple test model for compute benchmarking"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(self.device)
    
    def test_compute_without_transfer(self, batch_sizes: List[int], num_iterations: int = 50):
        """Baseline compute performance without memory transfers"""
        logger.info("=== Testing Baseline Compute Performance ===")
        
        model = self.create_test_model()
        
        for batch_size in batch_sizes:
            compute_times = []
            
            # Pre-allocate input tensor on GPU
            input_tensor = torch.randn(batch_size, 1024, device=self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_tensor)
            torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                compute_times.append((end - start) * 1000)
            
            self.results[f"baseline_compute_batch_{batch_size}"] = compute_times
            avg_time = statistics.mean(compute_times)
            std_time = statistics.stdev(compute_times) if len(compute_times) > 1 else 0
            
            logger.info(f"Batch size {batch_size}: {avg_time:.2f}±{std_time:.2f} ms")
    
    def test_compute_with_frequent_transfers(self, batch_sizes: List[int], 
                                           transfer_sizes: List[int], num_iterations: int = 50):
        """Test compute performance with frequent small memory transfers"""
        logger.info("=== Testing Compute Performance with Frequent Transfers ===")
        
        model = self.create_test_model()
        
        for batch_size in batch_sizes:
            for transfer_size_mb in transfer_sizes:
                compute_times = []
                transfer_elements = (transfer_size_mb * 1024 * 1024) // 4
                
                # Pre-allocate main input tensor on GPU
                input_tensor = torch.randn(batch_size, 1024, device=self.device)
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(input_tensor)
                    # Small transfer during warmup
                    temp_tensor = torch.randn(transfer_elements)
                    _ = temp_tensor.to(self.device)
                torch.cuda.synchronize()
                
                # Benchmark with transfers
                for _ in range(num_iterations):
                    # Create transfer tensor on CPU
                    transfer_tensor = torch.randn(transfer_elements)
                    
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    
                    # Interleave compute and transfer
                    with torch.no_grad():
                        # Start compute
                        output = model(input_tensor)
                        
                        # Concurrent transfer (simulating data loading)
                        gpu_transfer = transfer_tensor.to(self.device, non_blocking=True)
                        
                        # Force synchronization to measure total time including transfer
                        torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    compute_times.append((end - start) * 1000)
                    
                    del gpu_transfer
                
                key = f"compute_with_transfer_batch_{batch_size}_transfer_{transfer_size_mb}MB"
                self.results[key] = compute_times
                
                avg_time = statistics.mean(compute_times)
                std_time = statistics.stdev(compute_times) if len(compute_times) > 1 else 0
                
                baseline_key = f"baseline_compute_batch_{batch_size}"
                if baseline_key in self.results:
                    baseline_avg = statistics.mean(self.results[baseline_key])
                    overhead = ((avg_time - baseline_avg) / baseline_avg) * 100
                    logger.info(f"Batch {batch_size}, Transfer {transfer_size_mb}MB: "
                              f"{avg_time:.2f}±{std_time:.2f} ms (overhead: {overhead:.1f}%)")
                else:
                    logger.info(f"Batch {batch_size}, Transfer {transfer_size_mb}MB: "
                              f"{avg_time:.2f}±{std_time:.2f} ms")


class MockModel(nn.Module):
    """Mock model for FSDP testing"""
    
    def __init__(self, hidden_size: int = 2048, num_layers: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, 1000)
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return self.output_layer(x)


class FSDPNetworkTester:
    """Test FSDP network performance in multi-node setup"""
    
    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.results: Dict[str, List[float]] = {}
    
    def setup_distributed(self, master_addr: str = "localhost", master_port: str = "12355"):
        """Setup distributed environment"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.device)
    
    def test_allreduce_latency(self, sizes: List[int], num_iterations: int = 50):
        """Test allreduce communication latency for different tensor sizes"""
        logger.info(f"=== Testing AllReduce Latency (Rank {self.rank}) ===")
        
        for size_mb in sizes:
            latencies = []
            num_elements = (size_mb * 1024 * 1024) // 4  # float32
            
            for _ in range(num_iterations):
                tensor = torch.randn(num_elements, device=self.device)
                
                dist.barrier()
                start = time.perf_counter()
                
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            if self.rank == 0:
                avg_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                bandwidth = (size_mb * self.world_size) / (avg_latency / 1000)  # MB/s
                
                logger.info(f"AllReduce {size_mb}MB: {avg_latency:.2f}±{std_latency:.2f} ms, "
                          f"Bandwidth: {bandwidth:.2f} MB/s")
                
                self.results[f"allreduce_{size_mb}MB"] = latencies
    
    def test_fsdp_forward_backward(self, batch_sizes: List[int], num_iterations: int = 20):
        """Test FSDP forward/backward pass with network communication"""
        logger.info(f"=== Testing FSDP Forward/Backward (Rank {self.rank}) ===")
        
        # Create and wrap model with FSDP
        model = MockModel()
        model = FSDP(
            model,
            device_id=self.device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for batch_size in batch_sizes:
            forward_times = []
            backward_times = []
            total_times = []
            
            # Warmup
            for _ in range(5):
                inputs = torch.randn(batch_size, 2048, device=self.device)
                targets = torch.randint(0, 1000, (batch_size,), device=self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            dist.barrier()
            
            # Benchmark
            for _ in range(num_iterations):
                inputs = torch.randn(batch_size, 2048, device=self.device)
                targets = torch.randint(0, 1000, (batch_size,), device=self.device)
                
                optimizer.zero_grad()
                
                # Forward pass timing
                torch.cuda.synchronize()
                start_total = time.perf_counter()
                start_forward = time.perf_counter()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                torch.cuda.synchronize()
                end_forward = time.perf_counter()
                
                # Backward pass timing
                start_backward = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                end_backward = time.perf_counter()
                
                optimizer.step()
                torch.cuda.synchronize()
                end_total = time.perf_counter()
                
                forward_times.append((end_forward - start_forward) * 1000)
                backward_times.append((end_backward - start_backward) * 1000)
                total_times.append((end_total - start_total) * 1000)
            
            if self.rank == 0:
                fwd_avg = statistics.mean(forward_times)
                bwd_avg = statistics.mean(backward_times)
                total_avg = statistics.mean(total_times)
                
                logger.info(f"FSDP Batch {batch_size}:")
                logger.info(f"  Forward: {fwd_avg:.2f} ms")
                logger.info(f"  Backward: {bwd_avg:.2f} ms") 
                logger.info(f"  Total: {total_avg:.2f} ms")
                
                self.results[f"fsdp_forward_batch_{batch_size}"] = forward_times
                self.results[f"fsdp_backward_batch_{batch_size}"] = backward_times
                self.results[f"fsdp_total_batch_{batch_size}"] = total_times
    
    def test_parameter_sync_frequency(self, sync_frequencies: List[int], num_iterations: int = 30):
        """Test impact of different parameter synchronization frequencies"""
        logger.info(f"=== Testing Parameter Sync Frequency (Rank {self.rank}) ===")
        
        model = MockModel(hidden_size=1024, num_layers=4)
        model = FSDP(model, device_id=self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        batch_size = 16
        
        for sync_freq in sync_frequencies:
            iteration_times = []
            
            # Warmup
            for _ in range(5):
                inputs = torch.randn(batch_size, 1024, device=self.device)
                targets = torch.randint(0, 1000, (batch_size,), device=self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            dist.barrier()
            
            # Benchmark with different sync frequencies
            for iteration in range(num_iterations):
                inputs = torch.randn(batch_size, 1024, device=self.device)
                targets = torch.randint(0, 1000, (batch_size,), device=self.device)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                optimizer.zero_grad()
                
                # Control synchronization based on frequency
                if iteration % sync_freq == 0:
                    # Full synchronization
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    # Local computation without sync (simulated)
                    with model.no_sync():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                    # Accumulate gradients without optimizer step
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                iteration_times.append((end - start) * 1000)
            
            if self.rank == 0:
                avg_time = statistics.mean(iteration_times)
                logger.info(f"Sync frequency every {sync_freq} iterations: {avg_time:.2f} ms/iter")
                self.results[f"sync_freq_{sync_freq}"] = iteration_times
    
    def cleanup(self):
        """Cleanup distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()


def run_single_gpu_tests(args):
    """Run single GPU tests"""
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)
    
    logger.info(f"Running single GPU tests on {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Test 1: Memory transfer latency
    memory_tester = MemoryTransferTester(device)
    transfer_sizes = [1, 4, 16, 64, 256, 1024]  # MB
    memory_tester.test_transfer_latency(transfer_sizes, num_iterations=args.num_iterations)
    
    # Test 2: Compute efficiency impact
    compute_tester = ComputeEfficiencyTester(device)
    batch_sizes = [8, 16, 32, 64]
    compute_tester.test_compute_without_transfer(batch_sizes, num_iterations=args.num_iterations)
    
    # Test with frequent small transfers
    small_transfer_sizes = [1, 4, 16]  # MB
    compute_tester.test_compute_with_frequent_transfers(
        batch_sizes, small_transfer_sizes, num_iterations=args.num_iterations
    )
    
    # Save results
    if args.save_results:
        import json
        all_results = {**memory_tester.results, **compute_tester.results}
        with open(f'performance_results_gpu_{args.gpu_id}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to performance_results_gpu_{args.gpu_id}.json")


def run_distributed_worker(rank: int, world_size: int, args):
    """Worker function for distributed testing"""
    device = torch.device(f'cuda:{rank}')
    
    fsdp_tester = FSDPNetworkTester(rank, world_size, device)
    
    try:
        fsdp_tester.setup_distributed(args.master_addr, args.master_port)
        
        if rank == 0:
            logger.info(f"Running distributed tests with {world_size} GPUs")
        
        # Test network communication
        comm_sizes = [1, 4, 16, 64, 256]  # MB
        fsdp_tester.test_allreduce_latency(comm_sizes, num_iterations=args.num_iterations)
        
        # Test FSDP training performance
        batch_sizes = [4, 8, 16]
        fsdp_tester.test_fsdp_forward_backward(batch_sizes, num_iterations=args.num_iterations // 2)
        
        # Test synchronization frequency impact
        sync_frequencies = [1, 2, 4, 8]
        fsdp_tester.test_parameter_sync_frequency(sync_frequencies, num_iterations=args.num_iterations)
        
        # Save results from rank 0
        if rank == 0 and args.save_results:
            import json
            with open(f'fsdp_performance_results_{world_size}gpus.json', 'w') as f:
                json.dump(fsdp_tester.results, f, indent=2)
            logger.info(f"FSDP results saved to fsdp_performance_results_{world_size}gpus.json")
    
    finally:
        fsdp_tester.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Wan2.1 Performance Testing')
    parser.add_argument('--test_type', choices=['single', 'distributed', 'all'], 
                       default='all', help='Type of test to run')
    parser.add_argument('--gpu_id', type=int, default=0, 
                       help='GPU ID for single GPU tests')
    parser.add_argument('--num_gpus', type=int, default=2,
                       help='Number of GPUs for distributed tests')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of iterations for each test')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='Master address for distributed testing')
    parser.add_argument('--master_port', type=str, default='12355',
                       help='Master port for distributed testing')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON files')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)
    
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    if args.test_type in ['single', 'all']:
        logger.info("Starting single GPU tests...")
        run_single_gpu_tests(args)
    
    if args.test_type in ['distributed', 'all']:
        if args.num_gpus > torch.cuda.device_count():
            logger.error(f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available")
            sys.exit(1)
        
        logger.info("Starting distributed tests...")
        mp.spawn(
            run_distributed_worker,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    
    logger.info("All tests completed!")


if __name__ == '__main__':
    main()