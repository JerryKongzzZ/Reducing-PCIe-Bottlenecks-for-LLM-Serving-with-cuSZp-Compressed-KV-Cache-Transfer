"""
Compression performance benchmark script

This script is used to test the performance of cuSZp compression, including:
- Compression/Decompression speed
- Compression ratio
- Error analysis
"""

import torch
import time
import argparse
import logging
import numpy as np
import json
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the compiled extension directory to the system path
# Assuming the script is in benchmarks/ and the extension is in integration/compression_pipeline/
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.abspath(os.path.join(current_dir, '..', 'integration', 'compression_pipeline'))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

# Try to import cuSZp wrapper
try:
    import cuszp_wrapper_cpp
except ImportError as e:
    logger.error(f"Failed to import cuszp_wrapper_cpp: {e}")
    logger.error(f"Make sure the C++ extension is compiled and available in {pipeline_dir}")
    sys.exit(1)


class CompressionBenchmark:
    """Compression performance benchmark"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
    def benchmark(
        self,
        tensor_size: int,
        error_bound: float,
        encoding_mode: str = "plain",
        num_iterations: int = 50,
        real_kv_path: str = None
    ) -> dict:
        """
        Benchmark compression performance
        
        Args:
            tensor_size: Tensor size (number of elements)
            error_bound: Error bound
            encoding_mode: Encoding mode
            num_iterations: Number of iterations
            real_kv_path: Path to real KV cache tensor
            
        Returns:
            Dictionary of performance metrics
        """
        # Create test data
        if real_kv_path and os.path.exists(real_kv_path):
            logger.info(f"Loading real KV Cache from {real_kv_path}")
            real_tensor = torch.load(real_kv_path, weights_only=True).to(self.device).to(torch.float32)
            # Repeat or slice to match tensor_size
            if real_tensor.numel() >= tensor_size:
                test_tensor = real_tensor[:tensor_size].contiguous()
            else:
                repeats = (tensor_size // real_tensor.numel()) + 1
                test_tensor = real_tensor.repeat(repeats)[:tensor_size].contiguous()
        else:
            logger.info("Using random normally distributed tensor")
            test_tensor = torch.randn(tensor_size, dtype=torch.float32, device=self.device)
        
        # Create compressor
        mode = self._parse_encoding_mode(encoding_mode)
        config = cuszp_wrapper_cpp.CompressionConfig(
            error_bound=error_bound,
            use_relative_error=True,
            processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
            encoding_mode=mode,
            data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT
        )
        compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, self.device_id)
        
        # Estimate compressed buffer size
        estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(
            test_tensor.numel() * test_tensor.element_size()
        )
        
        # Allocate compressed buffer
        compressed_buffer = torch.empty(
            estimated_size, 
            dtype=torch.uint8, 
            device=self.device
        )
        
        # Warm up
        for _ in range(5):
            success, compressed_buffer, actual_size, actual_eb = compressor.compress(test_tensor, compressed_buffer)
            
            if success:
                decompressed_tensor = torch.empty_like(test_tensor)
                compressor.decompress(compressed_buffer, actual_size, decompressed_tensor, actual_eb)
        torch.cuda.synchronize()
        
        # Measure compression time
        compression_times = []
        compressed_sizes = []
        actual_eb_last = 0.0
        
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            success, compressed_buffer, actual_size, actual_eb = compressor.compress(test_tensor, compressed_buffer)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            if success:
                compression_times.append(end - start)
                compressed_sizes.append(actual_size)
                actual_eb_last = actual_eb
        
        # Measure decompression time
        decompression_times = []
        
        if len(compressed_sizes) > 0:
            actual_size = compressed_sizes[0]
            
            for _ in range(num_iterations):
                decompressed_tensor = torch.empty_like(test_tensor)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                compressor.decompress(compressed_buffer, actual_size, decompressed_tensor, actual_eb_last)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                decompression_times.append(end - start)
        
        # Calculate performance metrics
        original_size_bytes = tensor_size * 4  # float32 = 4 bytes
        avg_comp_time = np.mean(compression_times) if compression_times else 0
        avg_decomp_time = np.mean(decompression_times) if decompression_times else 0
        
        comp_bandwidth = (original_size_bytes / avg_comp_time / 1e9) if avg_comp_time > 0 else 0
        decomp_bandwidth = (original_size_bytes / avg_decomp_time / 1e9) if avg_decomp_time > 0 else 0
        
        avg_compressed_size = np.mean(compressed_sizes) if compressed_sizes else original_size_bytes
        compression_ratio = original_size_bytes / avg_compressed_size if avg_compressed_size > 0 else 1.0
        
        # Calculate error
        decompressed_tensor = torch.empty_like(test_tensor)
        compressor.decompress(compressed_buffer, int(avg_compressed_size), decompressed_tensor, actual_eb_last)
        
        abs_errors = torch.abs(test_tensor - decompressed_tensor)
        max_error = torch.max(abs_errors).item()
        mean_error = torch.mean(abs_errors).item()
        
        logger.info(
            f"Size: {tensor_size}, "
            f"Compression: {comp_bandwidth:.2f} GB/s, "
            f"Decompression: {decomp_bandwidth:.2f} GB/s, "
            f"Ratio: {compression_ratio:.2f}x, "
            f"Max Error: {max_error:.2e}"
        )
        
        return {
            "tensor_size": tensor_size,
            "original_size_bytes": original_size_bytes,
            "compressed_size_bytes": float(avg_compressed_size),
            "compression_ratio": float(compression_ratio),
            "compression_time_ms": float(avg_comp_time * 1000),
            "decompression_time_ms": float(avg_decomp_time * 1000),
            "compression_bandwidth_GB_s": float(comp_bandwidth),
            "decompression_bandwidth_GB_s": float(decomp_bandwidth),
            "max_absolute_error": float(max_error),
            "mean_absolute_error": float(mean_error),
            "error_bound_setting": error_bound
        }

    def _parse_encoding_mode(self, mode_str: str):
        """Parse encoding mode"""
        mode_str = mode_str.lower()
        if mode_str == "fixed":
            return cuszp_wrapper_cpp.CuszpMode.MODE_FIXED
        elif mode_str == "plain":
            return cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN
        elif mode_str == "outlier":
            return cuszp_wrapper_cpp.CuszpMode.MODE_OUTLIER
        else:
            logger.warning(f"Unknown mode {mode_str}, using PLAIN")
            return cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN
            
    def scan_error_bounds(
        self,
        tensor_size: int,
        error_bounds: list,
        encoding_mode: str = "plain",
        real_kv_path: str = None
    ):
        """Scan performance with different error bounds"""
        results = []
        logger.info(f"Scanning error bounds for tensor size {tensor_size}...")
        
        for eb in error_bounds:
            logger.info(f"Testing error bound: {eb}")
            res = self.benchmark(
                tensor_size=tensor_size,
                error_bound=eb,
                encoding_mode=encoding_mode,
                num_iterations=20,  # Reduce iterations for scanning
                real_kv_path=real_kv_path
            )
            results.append(res)
            
        return results


def main():
    parser = argparse.ArgumentParser(description="cuSZp compression benchmarking")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--tensor-size", type=int, default=1048576, help="Tensor size to profile")
    parser.add_argument("--error-bound", type=float, default=1e-4, help="Relative error bound")
    parser.add_argument("--encoding-mode", type=str, default="plain", 
                       choices=["fixed", "plain", "outlier"], help="Encoding mode")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--scan-eb", action="store_true", help="Scan different error bounds")
    parser.add_argument("--use-real-kv", type=str, default=None, help="Path to real KV cache .pt file. e.g. data/real_kv_cache.pt")
    parser.add_argument("--output", type=str, default="compression_results.json", help="Output file")
    
    args = parser.parse_args()
    
    benchmark = CompressionBenchmark(device_id=args.device_id)
    
    # Print GPU info
    logger.info(f"GPU: {torch.cuda.get_device_name(args.device_id)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    
    if args.scan_eb:
        # Scan error bounds
        ebs = [1e-2, 1e-3, 1e-4, 1e-5]
        results = benchmark.scan_error_bounds(
            args.tensor_size, ebs, args.encoding_mode, args.use_real_kv
        )
    else:
        # Single test
        res = benchmark.benchmark(
            args.tensor_size, 
            args.error_bound, 
            args.encoding_mode, 
            args.iterations,
            args.use_real_kv
        )
        results = [res]
        
    # Save results
    output_data = {
        "results": results,
        "device": torch.cuda.get_device_name(args.device_id),
        "cuda_version": torch.version.cuda
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
