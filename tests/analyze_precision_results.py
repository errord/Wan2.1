#!/usr/bin/env python3
"""
Analysis tool for precision performance test results
Generates comprehensive reports and visualizations for fp32/fp16/bf16 performance
"""

import json
import os
import sys
import statistics
import argparse
from typing import Dict, List, Any
from pathlib import Path


class PrecisionAnalyzer:
    """Analyze and visualize precision test results"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.single_gpu_results = {}
        self.distributed_results = {}
        self.load_results()
    
    def load_results(self):
        """Load precision test result files"""
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON result files found in {self.results_dir}")
            return
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                if 'precision_results_gpu' in json_file.name:
                    self.single_gpu_results[json_file.stem] = data
                elif 'distributed_precision_results' in json_file.name:
                    self.distributed_results[json_file.stem] = data
        
        print(f"Loaded {len(self.single_gpu_results)} single GPU and {len(self.distributed_results)} distributed result files")
    
    def analyze_single_gpu_precision(self):
        """Analyze single GPU precision performance"""
        print("\n" + "=" * 60)
        print("SINGLE GPU PRECISION PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        for file_name, data in self.single_gpu_results.items():
            print(f"\nResults from: {file_name}")
            print("-" * 40)
            
            # Collect statistics for each precision
            precision_stats = {}
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in data and data[precision].get('total_times'):
                    stats = {
                        'forward_time': statistics.mean(data[precision]['forward_times']),
                        'backward_time': statistics.mean(data[precision]['backward_times']),
                        'total_time': statistics.mean(data[precision]['total_times']),
                        'memory_usage': statistics.mean(data[precision]['memory_usage']),
                        'throughput': statistics.mean(data[precision]['throughput'])
                    }
                    precision_stats[precision] = stats
            
            if not precision_stats:
                print("No valid precision data found")
                continue
            
            # Performance summary table
            print(f"\n{'Precision':<10} {'Forward':<10} {'Backward':<10} {'Total':<10} {'Memory':<10} {'Throughput':<12}")
            print(f"{'Type':<10} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} {'(GB)':<10} {'(tokens/s)':<12}")
            print("-" * 72)
            
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in precision_stats:
                    stats = precision_stats[precision]
                    print(f"{precision.upper():<10} {stats['forward_time']:<10.2f} "
                          f"{stats['backward_time']:<10.2f} {stats['total_time']:<10.2f} "
                          f"{stats['memory_usage']:<10.2f} {stats['throughput']:<12.0f}")
            
            # Relative performance analysis
            if 'fp32' in precision_stats:
                print(f"\nPerformance relative to FP32:")
                print(f"{'Precision':<10} {'Speed':<10} {'Memory':<10} {'Throughput':<12} {'Efficiency':<10}")
                print(f"{'Type':<10} {'Gain':<10} {'Reduction':<10} {'Gain':<12} {'Score':<10}")
                print("-" * 62)
                
                fp32_stats = precision_stats['fp32']
                
                for precision in ['fp32', 'fp16', 'bf16']:
                    if precision in precision_stats:
                        stats = precision_stats[precision]
                        
                        speed_gain = fp32_stats['total_time'] / stats['total_time']
                        memory_reduction = fp32_stats['memory_usage'] / stats['memory_usage']
                        throughput_gain = stats['throughput'] / fp32_stats['throughput']
                        
                        # Efficiency score: balance of speed and memory
                        efficiency_score = (speed_gain * memory_reduction) ** 0.5
                        
                        print(f"{precision.upper():<10} {speed_gain:<10.2f} "
                              f"{memory_reduction:<10.2f} {throughput_gain:<12.2f} "
                              f"{efficiency_score:<10.2f}")
                
                # Memory bandwidth analysis
                print(f"\nMemory Bandwidth Analysis:")
                print(f"{'Precision':<10} {'Model Size':<12} {'Bandwidth':<12} {'Utilization':<12}")
                print(f"{'Type':<10} {'(GB)':<12} {'(GB/s)':<12} {'(%)':<12}")
                print("-" * 48)
                
                for precision in ['fp32', 'fp16', 'bf16']:
                    if precision in precision_stats:
                        stats = precision_stats[precision]
                        
                        # Estimate model size (this is approximate)
                        if precision == 'fp32':
                            model_size_gb = stats['memory_usage'] * 0.6  # Rough estimate
                        else:
                            model_size_gb = stats['memory_usage'] * 0.7  # fp16/bf16 models are smaller
                        
                        # Estimate memory bandwidth
                        bandwidth = (model_size_gb * 2) / (stats['total_time'] / 1000)  # Read + Write
                        
                        # Theoretical peak bandwidth (example for modern GPUs)
                        peak_bandwidth = 800  # GB/s for high-end GPUs
                        utilization = (bandwidth / peak_bandwidth) * 100
                        
                        print(f"{precision.upper():<10} {model_size_gb:<12.1f} "
                              f"{bandwidth:<12.0f} {utilization:<12.1f}")
    
    def analyze_distributed_precision(self):
        """Analyze distributed precision performance"""
        print("\n" + "=" * 60)
        print("DISTRIBUTED PRECISION PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        for file_name, data in self.distributed_results.items():
            print(f"\nResults from: {file_name}")
            print("-" * 40)
            
            # Collect statistics for each precision
            distributed_stats = {}
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in data and data[precision].get('total_times'):
                    stats = {
                        'compute_time': statistics.mean(data[precision]['compute_times']),
                        'communication_time': statistics.mean(data[precision]['communication_times']),
                        'total_time': statistics.mean(data[precision]['total_times'])
                    }
                    stats['communication_ratio'] = (stats['communication_time'] / stats['total_time']) * 100
                    distributed_stats[precision] = stats
            
            if not distributed_stats:
                print("No valid distributed precision data found")
                continue
            
            # Distributed performance table
            print(f"\n{'Precision':<10} {'Compute':<10} {'Comm':<10} {'Total':<10} {'Comm %':<10}")
            print(f"{'Type':<10} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} {'Overhead':<10}")
            print("-" * 52)
            
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in distributed_stats:
                    stats = distributed_stats[precision]
                    print(f"{precision.upper():<10} {stats['compute_time']:<10.2f} "
                          f"{stats['communication_time']:<10.2f} {stats['total_time']:<10.2f} "
                          f"{stats['communication_ratio']:<10.1f}")
            
            # Communication impact analysis
            if 'fp32' in distributed_stats:
                print(f"\nCommunication Impact on Precision Benefits:")
                print(f"{'Precision':<10} {'Compute':<12} {'Total':<12} {'Benefit':<12} {'Network':<12}")
                print(f"{'Type':<10} {'Speedup':<12} {'Speedup':<12} {'Retention':<12} {'Impact':<12}")
                print("-" * 62)
                
                fp32_stats = distributed_stats['fp32']
                
                for precision in ['fp32', 'fp16', 'bf16']:
                    if precision in distributed_stats:
                        stats = distributed_stats[precision]
                        
                        compute_speedup = fp32_stats['compute_time'] / stats['compute_time']
                        total_speedup = fp32_stats['total_time'] / stats['total_time']
                        
                        if compute_speedup > 1.0:
                            benefit_retention = (total_speedup / compute_speedup) * 100
                        else:
                            benefit_retention = 100.0
                        
                        network_impact = 100 - benefit_retention
                        
                        print(f"{precision.upper():<10} {compute_speedup:<12.2f} "
                              f"{total_speedup:<12.2f} {benefit_retention:<12.1f} "
                              f"{network_impact:<12.1f}")
                
                # Network efficiency analysis
                print(f"\nNetwork Efficiency Analysis:")
                
                for precision in ['fp16', 'bf16']:
                    if precision in distributed_stats and 'fp32' in distributed_stats:
                        fp32_compute = fp32_stats['compute_time']
                        precision_compute = distributed_stats[precision]['compute_time']
                        precision_total = distributed_stats[precision]['total_time']
                        
                        # Calculate theoretical speedup if no communication overhead
                        theoretical_speedup = fp32_stats['total_time'] / precision_compute
                        actual_speedup = fp32_stats['total_time'] / precision_total
                        efficiency = (actual_speedup / theoretical_speedup) * 100
                        
                        print(f"\n{precision.upper()} Network Efficiency:")
                        print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
                        print(f"  Actual speedup: {actual_speedup:.2f}x") 
                        print(f"  Network efficiency: {efficiency:.1f}%")
                        
                        if efficiency < 80:
                            print(f"  ⚠️  Network communication significantly impacts {precision.upper()} benefits")
                        elif efficiency < 90:
                            print(f"  ⚡ Moderate network impact on {precision.upper()} benefits")
                        else:
                            print(f"  ✅ Good network efficiency for {precision.upper()}")
    
    def generate_recommendations(self):
        """Generate optimization recommendations based on results"""
        print("\n" + "=" * 60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Analyze single GPU results for recommendations
        for file_name, data in self.single_gpu_results.items():
            precision_stats = {}
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in data and data[precision].get('total_times'):
                    precision_stats[precision] = {
                        'total_time': statistics.mean(data[precision]['total_times']),
                        'memory_usage': statistics.mean(data[precision]['memory_usage']),
                        'throughput': statistics.mean(data[precision]['throughput'])
                    }
            
            if 'fp32' in precision_stats:
                fp32_time = precision_stats['fp32']['total_time']
                
                # Check FP16 performance
                if 'fp16' in precision_stats:
                    fp16_speedup = fp32_time / precision_stats['fp16']['total_time']
                    if fp16_speedup > 1.5:
                        recommendations.append("🚀 FP16 shows excellent speedup (>1.5x). Strongly recommended for production.")
                    elif fp16_speedup > 1.2:
                        recommendations.append("⚡ FP16 shows good speedup (>1.2x). Recommended with accuracy validation.")
                    else:
                        recommendations.append("⚠️  FP16 shows limited speedup. Consider mixed precision instead.")
                
                # Check BF16 performance
                if 'bf16' in precision_stats:
                    bf16_speedup = fp32_time / precision_stats['bf16']['total_time']
                    if bf16_speedup > 1.3:
                        recommendations.append("🎯 BF16 shows strong performance with better numerical stability than FP16.")
                    elif bf16_speedup > 1.1:
                        recommendations.append("✅ BF16 provides moderate speedup with good stability. Good for training.")
                
                # Memory recommendations
                fp32_memory = precision_stats['fp32']['memory_usage']
                if 'fp16' in precision_stats:
                    memory_savings = (fp32_memory - precision_stats['fp16']['memory_usage']) / fp32_memory * 100
                    if memory_savings > 30:
                        recommendations.append(f"💾 FP16 reduces memory usage by {memory_savings:.1f}%. Enables larger models/batches.")
        
        # Analyze distributed results for network recommendations
        for file_name, data in self.distributed_results.items():
            distributed_stats = {}
            for precision in ['fp32', 'fp16', 'bf16']:
                if precision in data and data[precision].get('total_times'):
                    distributed_stats[precision] = {
                        'compute_time': statistics.mean(data[precision]['compute_times']),
                        'communication_time': statistics.mean(data[precision]['communication_times']),
                        'total_time': statistics.mean(data[precision]['total_times'])
                    }
            
            if 'fp32' in distributed_stats:
                fp32_comm_ratio = (distributed_stats['fp32']['communication_time'] / 
                                 distributed_stats['fp32']['total_time']) * 100
                
                if fp32_comm_ratio > 50:
                    recommendations.append("🌐 High communication overhead (>50%). Consider gradient accumulation or larger local batch sizes.")
                
                # Check if precision benefits are preserved in distributed setting
                for precision in ['fp16', 'bf16']:
                    if precision in distributed_stats:
                        compute_speedup = (distributed_stats['fp32']['compute_time'] / 
                                         distributed_stats[precision]['compute_time'])
                        total_speedup = (distributed_stats['fp32']['total_time'] / 
                                       distributed_stats[precision]['total_time'])
                        
                        if total_speedup / compute_speedup < 0.7:
                            recommendations.append(f"⚠️  Network communication reduces {precision.upper()} benefits significantly. "
                                                 "Consider optimizing network or increasing computation/communication ratio.")
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No specific recommendations generated. Check if test data is available.")
        
        # General best practices
        print(f"\nGeneral Best Practices:")
        print("• Use FP16/BF16 for inference when accuracy is acceptable")
        print("• BF16 is preferred for training due to better numerical stability")
        print("• Mixed precision training can provide benefits of both FP32 and FP16")
        print("• In distributed settings, optimize the computation/communication ratio")
        print("• Profile your specific workload as results vary by model architecture")
    
    def create_summary_report(self, output_file: str = "precision_performance_report.txt"):
        """Generate a comprehensive text report"""
        with open(output_file, 'w') as f:
            # Redirect stdout to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            try:
                print("PRECISION PERFORMANCE ANALYSIS REPORT")
                print("=" * 50)
                print(f"Generated from results in: {self.results_dir}")
                print()
                
                self.analyze_single_gpu_precision()
                self.analyze_distributed_precision()
                self.generate_recommendations()
                
            finally:
                sys.stdout = original_stdout
        
        print(f"Comprehensive report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze precision performance test results')
    parser.add_argument('--results_dir', type=str, default='.',
                       help='Directory containing result JSON files')
    parser.add_argument('--report_file', type=str, default='precision_performance_report.txt',
                       help='Output file for comprehensive report')
    
    args = parser.parse_args()
    
    analyzer = PrecisionAnalyzer(args.results_dir)
    
    if not analyzer.single_gpu_results and not analyzer.distributed_results:
        print("No precision test result files found. Please run precision tests first:")
        print("  python tests/precision_performance_test.py --save_results")
        print("  or")
        print("  ./tests/run_precision_test.sh")
        sys.exit(1)
    
    # Generate analysis
    analyzer.analyze_single_gpu_precision()
    analyzer.analyze_distributed_precision()
    analyzer.generate_recommendations()
    
    # Generate comprehensive report
    analyzer.create_summary_report(args.report_file)


if __name__ == '__main__':
    main()