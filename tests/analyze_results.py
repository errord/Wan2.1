#!/usr/bin/env python3
"""
Analysis tool for performance test results
Generates reports and visualizations from JSON result files
"""

import json
import os
import sys
import statistics
import argparse
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class PerformanceAnalyzer:
    """Analyze and visualize performance test results"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.load_results()
    
    def load_results(self):
        """Load all JSON result files"""
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON result files found in {self.results_dir}")
            return
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.results[json_file.stem] = data
        
        print(f"Loaded {len(self.results)} result files")
    
    def analyze_memory_transfer(self):
        """Analyze memory transfer performance"""
        print("\n=== Memory Transfer Analysis ===")
        
        for file_name, data in self.results.items():
            if 'performance_results_gpu' in file_name:
                print(f"\nFile: {file_name}")
                
                # Analyze CPU to GPU transfers
                cpu_to_gpu_results = {}
                gpu_to_cpu_results = {}
                
                for key, values in data.items():
                    if 'cpu_to_gpu' in key:
                        size = key.split('_')[3]  # Extract size like "1MB"
                        cpu_to_gpu_results[size] = {
                            'mean': statistics.mean(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0,
                            'min': min(values),
                            'max': max(values)
                        }
                    elif 'gpu_to_cpu' in key:
                        size = key.split('_')[3]  # Extract size like "1MB"
                        gpu_to_cpu_results[size] = {
                            'mean': statistics.mean(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0,
                            'min': min(values),
                            'max': max(values)
                        }
                
                # Print summary table
                print("\nTransfer Performance Summary:")
                print(f"{'Size':<8} {'CPU->GPU (ms)':<15} {'GPU->CPU (ms)':<15} {'CPU->GPU BW (GB/s)':<18} {'GPU->CPU BW (GB/s)':<18}")
                print("-" * 80)
                
                for size in sorted(cpu_to_gpu_results.keys(), key=lambda x: int(x[:-2])):
                    cpu_stats = cpu_to_gpu_results[size]
                    gpu_stats = gpu_to_cpu_results[size]
                    
                    size_mb = int(size[:-2])
                    cpu_bandwidth = (size_mb / 1024) / (cpu_stats['mean'] / 1000)  # GB/s
                    gpu_bandwidth = (size_mb / 1024) / (gpu_stats['mean'] / 1000)  # GB/s
                    
                    print(f"{size:<8} {cpu_stats['mean']:.2f}±{cpu_stats['std']:.2f}     "
                          f"{gpu_stats['mean']:.2f}±{gpu_stats['std']:.2f}     "
                          f"{cpu_bandwidth:.2f}               "
                          f"{gpu_bandwidth:.2f}")
    
    def analyze_compute_efficiency(self):
        """Analyze compute efficiency impact"""
        print("\n=== Compute Efficiency Analysis ===")
        
        for file_name, data in self.results.items():
            if 'performance_results_gpu' in file_name:
                print(f"\nFile: {file_name}")
                
                # Extract baseline performance
                baseline_results = {}
                transfer_results = {}
                
                for key, values in data.items():
                    if 'baseline_compute_batch' in key:
                        batch_size = int(key.split('_')[3])
                        baseline_results[batch_size] = statistics.mean(values)
                    elif 'compute_with_transfer_batch' in key:
                        parts = key.split('_')
                        batch_size = int(parts[4])
                        transfer_size = parts[6]  # e.g., "1MB"
                        
                        if batch_size not in transfer_results:
                            transfer_results[batch_size] = {}
                        transfer_results[batch_size][transfer_size] = statistics.mean(values)
                
                # Calculate overhead
                print("\nCompute Efficiency Impact:")
                print(f"{'Batch Size':<12} {'Baseline (ms)':<15} {'Transfer Size':<15} {'With Transfer (ms)':<20} {'Overhead (%)':<12}")
                print("-" * 85)
                
                for batch_size in sorted(baseline_results.keys()):
                    baseline = baseline_results[batch_size]
                    print(f"{batch_size:<12} {baseline:.2f}           {'N/A':<15} {'N/A':<20} {'0.0':<12}")
                    
                    if batch_size in transfer_results:
                        for transfer_size in sorted(transfer_results[batch_size].keys(), key=lambda x: int(x[:-2])):
                            with_transfer = transfer_results[batch_size][transfer_size]
                            overhead = ((with_transfer - baseline) / baseline) * 100
                            print(f"{'':12} {'':15} {transfer_size:<15} {with_transfer:.2f}              {overhead:.1f}")
    
    def analyze_fsdp_performance(self):
        """Analyze FSDP network performance"""
        print("\n=== FSDP Network Performance Analysis ===")
        
        for file_name, data in self.results.items():
            if 'fsdp_performance_results' in file_name:
                print(f"\nFile: {file_name}")
                
                # Analyze AllReduce performance
                allreduce_results = {}
                for key, values in data.items():
                    if 'allreduce' in key:
                        size = key.split('_')[1]  # Extract size like "1MB"
                        allreduce_results[size] = {
                            'mean': statistics.mean(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0
                        }
                
                if allreduce_results:
                    print("\nAllReduce Performance:")
                    print(f"{'Size':<8} {'Latency (ms)':<15} {'Bandwidth (GB/s)':<18}")
                    print("-" * 45)
                    
                    for size in sorted(allreduce_results.keys(), key=lambda x: int(x[:-2])):
                        stats = allreduce_results[size]
                        size_mb = int(size[:-2])
                        bandwidth = (size_mb / 1024) / (stats['mean'] / 1000)  # GB/s
                        print(f"{size:<8} {stats['mean']:.2f}±{stats['std']:.2f}     {bandwidth:.2f}")
                
                # Analyze FSDP training performance
                fsdp_results = {}
                for key, values in data.items():
                    if 'fsdp_forward_batch' in key or 'fsdp_backward_batch' in key or 'fsdp_total_batch' in key:
                        batch_size = int(key.split('_')[3])
                        phase = key.split('_')[1]  # forward, backward, or total
                        
                        if batch_size not in fsdp_results:
                            fsdp_results[batch_size] = {}
                        fsdp_results[batch_size][phase] = statistics.mean(values)
                
                if fsdp_results:
                    print("\nFSDP Training Performance:")
                    print(f"{'Batch Size':<12} {'Forward (ms)':<15} {'Backward (ms)':<16} {'Total (ms)':<12}")
                    print("-" * 60)
                    
                    for batch_size in sorted(fsdp_results.keys()):
                        results = fsdp_results[batch_size]
                        forward = results.get('forward', 0)
                        backward = results.get('backward', 0)
                        total = results.get('total', 0)
                        print(f"{batch_size:<12} {forward:.2f}           {backward:.2f}            {total:.2f}")
                
                # Analyze sync frequency impact
                sync_results = {}
                for key, values in data.items():
                    if 'sync_freq' in key:
                        freq = int(key.split('_')[2])
                        sync_results[freq] = statistics.mean(values)
                
                if sync_results:
                    print("\nSynchronization Frequency Impact:")
                    print(f"{'Sync Frequency':<15} {'Time per Iteration (ms)':<25}")
                    print("-" * 45)
                    
                    for freq in sorted(sync_results.keys()):
                        time_per_iter = sync_results[freq]
                        print(f"Every {freq} iters    {time_per_iter:.2f}")
    
    def create_visualizations(self, output_dir: str = "plots"):
        """Create performance visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n=== Creating Visualizations in {output_path} ===")
        
        # Plot memory transfer performance
        self._plot_memory_transfer(output_path)
        
        # Plot compute efficiency
        self._plot_compute_efficiency(output_path)
        
        # Plot FSDP performance
        self._plot_fsdp_performance(output_path)
        
        print("Visualizations completed!")
    
    def _plot_memory_transfer(self, output_path: Path):
        """Plot memory transfer performance"""
        for file_name, data in self.results.items():
            if 'performance_results_gpu' not in file_name:
                continue
            
            sizes = []
            cpu_to_gpu_times = []
            gpu_to_cpu_times = []
            cpu_to_gpu_bandwidths = []
            gpu_to_cpu_bandwidths = []
            
            for key, values in data.items():
                if 'cpu_to_gpu' in key:
                    size_str = key.split('_')[3]  # e.g., "1MB"
                    size_mb = int(size_str[:-2])
                    sizes.append(size_mb)
                    
                    mean_time = statistics.mean(values)
                    cpu_to_gpu_times.append(mean_time)
                    
                    bandwidth = (size_mb / 1024) / (mean_time / 1000)  # GB/s
                    cpu_to_gpu_bandwidths.append(bandwidth)
                elif 'gpu_to_cpu' in key:
                    mean_time = statistics.mean(values)
                    gpu_to_cpu_times.append(mean_time)
                    
                    size_str = key.split('_')[3]
                    size_mb = int(size_str[:-2])
                    bandwidth = (size_mb / 1024) / (mean_time / 1000)  # GB/s
                    gpu_to_cpu_bandwidths.append(bandwidth)
            
            if sizes:
                # Sort by size
                sorted_data = sorted(zip(sizes, cpu_to_gpu_times, gpu_to_cpu_times, 
                                       cpu_to_gpu_bandwidths, gpu_to_cpu_bandwidths))
                sizes, cpu_to_gpu_times, gpu_to_cpu_times, cpu_to_gpu_bandwidths, gpu_to_cpu_bandwidths = zip(*sorted_data)
                
                # Plot latency
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.loglog(sizes, cpu_to_gpu_times, 'bo-', label='CPU -> GPU')
                ax1.loglog(sizes, gpu_to_cpu_times, 'ro-', label='GPU -> CPU')
                ax1.set_xlabel('Transfer Size (MB)')
                ax1.set_ylabel('Latency (ms)')
                ax1.set_title('Memory Transfer Latency')
                ax1.legend()
                ax1.grid(True)
                
                # Plot bandwidth
                ax2.semilogx(sizes, cpu_to_gpu_bandwidths, 'bo-', label='CPU -> GPU')
                ax2.semilogx(sizes, gpu_to_cpu_bandwidths, 'ro-', label='GPU -> CPU')
                ax2.set_xlabel('Transfer Size (MB)')
                ax2.set_ylabel('Bandwidth (GB/s)')
                ax2.set_title('Memory Transfer Bandwidth')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(output_path / f'memory_transfer_{file_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_compute_efficiency(self, output_path: Path):
        """Plot compute efficiency impact"""
        for file_name, data in self.results.items():
            if 'performance_results_gpu' not in file_name:
                continue
            
            batch_sizes = []
            overheads = {1: [], 4: [], 16: []}  # MB transfer sizes
            
            baseline_results = {}
            for key, values in data.items():
                if 'baseline_compute_batch' in key:
                    batch_size = int(key.split('_')[3])
                    baseline_results[batch_size] = statistics.mean(values)
                    if batch_size not in batch_sizes:
                        batch_sizes.append(batch_size)
            
            for key, values in data.items():
                if 'compute_with_transfer_batch' in key:
                    parts = key.split('_')
                    batch_size = int(parts[4])
                    transfer_size_str = parts[6]  # e.g., "1MB"
                    transfer_size = int(transfer_size_str[:-2])
                    
                    if transfer_size in overheads and batch_size in baseline_results:
                        baseline = baseline_results[batch_size]
                        with_transfer = statistics.mean(values)
                        overhead = ((with_transfer - baseline) / baseline) * 100
                        
                        idx = batch_sizes.index(batch_size) if batch_size in batch_sizes else -1
                        if idx >= 0:
                            while len(overheads[transfer_size]) <= idx:
                                overheads[transfer_size].append(0)
                            overheads[transfer_size][idx] = overhead
            
            if batch_sizes:
                batch_sizes.sort()
                
                plt.figure(figsize=(10, 6))
                
                x = np.arange(len(batch_sizes))
                width = 0.25
                
                for i, (transfer_size, overhead_list) in enumerate(overheads.items()):
                    if overhead_list:
                        plt.bar(x + i * width, overhead_list[:len(batch_sizes)], 
                               width, label=f'{transfer_size}MB Transfer')
                
                plt.xlabel('Batch Size')
                plt.ylabel('Compute Overhead (%)')
                plt.title('Impact of Memory Transfers on Compute Performance')
                plt.xticks(x + width, batch_sizes)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / f'compute_efficiency_{file_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_fsdp_performance(self, output_path: Path):
        """Plot FSDP performance"""
        for file_name, data in self.results.items():
            if 'fsdp_performance_results' not in file_name:
                continue
            
            # Plot AllReduce performance
            sizes = []
            latencies = []
            bandwidths = []
            
            for key, values in data.items():
                if 'allreduce' in key:
                    size_str = key.split('_')[1]  # e.g., "1MB"
                    size_mb = int(size_str[:-2])
                    sizes.append(size_mb)
                    
                    mean_latency = statistics.mean(values)
                    latencies.append(mean_latency)
                    
                    bandwidth = (size_mb / 1024) / (mean_latency / 1000)  # GB/s
                    bandwidths.append(bandwidth)
            
            if sizes:
                sorted_data = sorted(zip(sizes, latencies, bandwidths))
                sizes, latencies, bandwidths = zip(*sorted_data)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.loglog(sizes, latencies, 'go-')
                ax1.set_xlabel('Message Size (MB)')
                ax1.set_ylabel('AllReduce Latency (ms)')
                ax1.set_title('FSDP AllReduce Latency')
                ax1.grid(True)
                
                ax2.semilogx(sizes, bandwidths, 'go-')
                ax2.set_xlabel('Message Size (MB)')
                ax2.set_ylabel('Bandwidth (GB/s)')
                ax2.set_title('FSDP AllReduce Bandwidth')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(output_path / f'fsdp_allreduce_{file_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, output_file: str = "performance_report.txt"):
        """Generate a comprehensive text report"""
        with open(output_file, 'w') as f:
            f.write("Wan2.1 Performance Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Redirect stdout to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            try:
                self.analyze_memory_transfer()
                self.analyze_compute_efficiency()
                self.analyze_fsdp_performance()
            finally:
                sys.stdout = original_stdout
        
        print(f"Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze performance test results')
    parser.add_argument('--results_dir', type=str, default='.',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save visualization plots')
    parser.add_argument('--report_file', type=str, default='performance_report.txt',
                       help='Output file for text report')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.results_dir)
    
    if not analyzer.results:
        print("No result files found. Please run performance tests first.")
        sys.exit(1)
    
    # Generate analysis
    analyzer.analyze_memory_transfer()
    analyzer.analyze_compute_efficiency() 
    analyzer.analyze_fsdp_performance()
    
    # Generate report
    analyzer.generate_report(args.report_file)
    
    # Create visualizations
    if not args.no_plots:
        try:
            analyzer.create_visualizations(args.output_dir)
        except ImportError:
            print("matplotlib not available, skipping visualizations")
        except Exception as e:
            print(f"Error creating visualizations: {e}")


if __name__ == '__main__':
    main()