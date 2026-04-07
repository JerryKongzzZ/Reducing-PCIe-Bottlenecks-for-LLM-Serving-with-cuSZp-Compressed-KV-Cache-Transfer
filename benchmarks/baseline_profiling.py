"""
Baseline performance profiling script

This script is used to measure the baseline performance of vLLM before integrating cuSZp,
specifically the CPU-GPU data transfer time.
"""

import torch
import time
import argparse
import logging
from typing import List, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineProfiler:
    """Baseline performance profiler"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
    def profile_h2d_transfer(
        self,
        tensor_sizes: List[int],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile Host-to-Device transfer performance
        
        Args:
            tensor_sizes: List of tensor sizes (number of elements)
            num_iterations: Number of iterations
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Profiling H2D transfers...")
        
        results = {
            "sizes": [],
            "transfer_times": [],
            "bandwidths": []
        }
        
        for size in tensor_sizes:
            # Create CPU tensor
            cpu_tensor = torch.randn(size, dtype=torch.float32)
            
            # Warm up
            for _ in range(10):
                _ = cpu_tensor.to(self.device, non_blocking=False)
            torch.cuda.synchronize()
            
            # Measure transfer time
            transfer_times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
                torch.cuda.synchronize()
                end = time.perf_counter()
                transfer_times.append(end - start)
            
            avg_time = np.mean(transfer_times)
            std_time = np.std(transfer_times)
            data_size_bytes = size * 4  # float32 = 4 bytes
            bandwidth = data_size_bytes / avg_time / 1e9  # GB/s
            
            results["sizes"].append(size)
            results["transfer_times"].append(avg_time)
            results["bandwidths"].append(bandwidth)
            
            logger.info(
                f"Size: {size}, Time: {avg_time*1000:.3f}ms ± {std_time*1000:.3f}ms, "
                f"Bandwidth: {bandwidth:.2f} GB/s"
            )
        
        return results
    
    def profile_d2h_transfer(
        self,
        tensor_sizes: List[int],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile Device-to-Host transfer performance
        
        Args:
            tensor_sizes: List of tensor sizes (number of elements)
            num_iterations: Number of iterations
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Profiling D2H transfers...")
        
        results = {
            "sizes": [],
            "transfer_times": [],
            "bandwidths": []
        }
        
        for size in tensor_sizes:
            # Create GPU tensor
            gpu_tensor = torch.randn(size, dtype=torch.float32, device=self.device)
            
            # Warm up
            for _ in range(10):
                _ = gpu_tensor.cpu()
            torch.cuda.synchronize()
            
            # Measure transfer time
            transfer_times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                cpu_tensor = gpu_tensor.cpu()
                torch.cuda.synchronize()
                end = time.perf_counter()
                transfer_times.append(end - start)
            
            avg_time = np.mean(transfer_times)
            std_time = np.std(transfer_times)
            data_size_bytes = size * 4  # float32 = 4 bytes
            bandwidth = data_size_bytes / avg_time / 1e9  # GB/s
            
            results["sizes"].append(size)
            results["transfer_times"].append(avg_time)
            results["bandwidths"].append(bandwidth)
            
            logger.info(
                f"Size: {size}, Time: {avg_time*1000:.3f}ms ± {std_time*1000:.3f}ms, "
                f"Bandwidth: {bandwidth:.2f} GB/s"
            )
        
        return results
    
    def profile_async_overlap(
        self,
        tensor_size: int,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile the effect of overlapping asynchronous transfer and computation
        
        Args:
            tensor_size: Tensor size
            num_iterations: Number of iterations
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Profiling async overlap...")
        
        # Create streams
        transfer_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        
        cpu_tensor = torch.randn(tensor_size, dtype=torch.float32)
        gpu_tensor = torch.randn(tensor_size, dtype=torch.float32, device=self.device)
        
        # Warm up
        for _ in range(10):
            with torch.cuda.stream(transfer_stream):
                _ = cpu_tensor.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        
        # Measure synchronous transfer time
        sync_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = cpu_tensor.to(self.device, non_blocking=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            sync_times.append(end - start)
        
        # Measure asynchronous transfer time (overlapped with computation)
        async_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(transfer_stream):
                _ = cpu_tensor.to(self.device, non_blocking=True)
            with torch.cuda.stream(compute_stream):
                # Simulate some computation
                _ = gpu_tensor * 2.0
            torch.cuda.synchronize()
            end = time.perf_counter()
            async_times.append(end - start)
        
        avg_sync = np.mean(sync_times)
        avg_async = np.mean(async_times)
        overlap_efficiency = (avg_sync - avg_async) / avg_sync * 100
        
        logger.info(
            f"Sync time: {avg_sync*1000:.3f}ms, "
            f"Async time: {avg_async*1000:.3f}ms, "
            f"Overlap efficiency: {overlap_efficiency:.2f}%"
        )
        
        return {
            "sync_time": avg_sync,
            "async_time": avg_async,
            "overlap_efficiency": overlap_efficiency
        }


def main():
    parser = argparse.ArgumentParser(description="Baseline performance profiling")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--tensor-sizes", type=int, nargs="+", 
                       default=[1024, 4096, 16384, 65536, 262144, 1048576],
                       help="Tensor sizes to profile")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations per measurement")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    profiler = BaselineProfiler(device_id=args.device_id)
    
    # Print GPU info
    logger.info(f"GPU: {torch.cuda.get_device_name(args.device_id)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Profile H2D transfer
    h2d_results = profiler.profile_h2d_transfer(
        args.tensor_sizes, args.iterations
    )
    
    # Profile D2H transfer
    d2h_results = profiler.profile_d2h_transfer(
        args.tensor_sizes, args.iterations
    )
    
    # Profile async overlap
    overlap_results = profiler.profile_async_overlap(
        args.tensor_sizes[-1], args.iterations
    )
    
    # Save results
    import json
    results = {
        "h2d": h2d_results,
        "d2h": d2h_results,
        "overlap": overlap_results,
        "device": torch.cuda.get_device_name(args.device_id),
        "cuda_version": torch.version.cuda
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
