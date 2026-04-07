"""
压缩性能基准测试脚本

这个脚本用于测试cuSZp压缩的性能，包括：
- 压缩/解压缩速度
- 压缩比
- 错误分析
"""

import torch
import time
import argparse
import logging
import numpy as np
from typing import Dict, List
import json
import ctypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
# 将编译好的扩展目录加入到系统路径中
# 假设脚本在 benchmarks/ 目录下，扩展在 integration/compression_pipeline/ 目录下
current_dir = os.path.dirname(os.path.abspath(__file__))
extension_dir = os.path.join(current_dir, '../integration/compression_pipeline')
sys.path.append(extension_dir)

# 尝试导入cuSZp包装器
try:
    import cuszp_wrapper_cpp
    CUSZP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"cuSZp wrapper not available. Error: {e}")
    CUSZP_AVAILABLE = False

class CompressionBenchmark:
    """压缩性能基准测试"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
    
    def benchmark_compression(
        self,
        tensor_size: int,
        error_bound: float,
        encoding_mode: str = "plain",
        num_iterations: int = 100
    ) -> Dict:
        """
        基准测试压缩性能
        
        Args:
            tensor_size: 张量大小（元素数量）
            error_bound: 错误边界
            encoding_mode: 编码模式
            num_iterations: 迭代次数
            
        Returns:
            性能指标字典
        """
        if not CUSZP_AVAILABLE:
            logger.error("cuSZp wrapper not available")
            return {}
        
        # 创建测试数据
        test_tensor = torch.randn(tensor_size, dtype=torch.float32, device=self.device)
        
        # 创建压缩器
        config = cuszp_wrapper_cpp.CompressionConfig()
        config.error_bound = error_bound
        config.use_relative_error = True
        config.encoding_mode = self._parse_mode(encoding_mode)
        config.processing_dim = cuszp_wrapper_cpp.CuszpDim.DIM_1D
        config.data_type = cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT
        
        compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, self.device_id)
        
        # 估算压缩缓冲区大小
        original_size_bytes = tensor_size * 4
        estimated_buffer_size = compressor.estimate_compressed_buffer_size(original_size_bytes)
        
        # 分配压缩缓冲区
        compressed_buffer = torch.empty(
            (estimated_buffer_size,),
            dtype=torch.uint8,
            device=self.device
        )
        
        # 预热
        for _ in range(10):
            # 💡 直接接收返回值，不需要 ctypes
            _, compressed_buffer, _ = compressor.compress(
                test_tensor,
                compressed_buffer
            )
        torch.cuda.synchronize()
        
        # 测量压缩时间
        compression_times = []
        compressed_sizes = []
        
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # 💡 使用新 API
            success, compressed_buffer, actual_size = compressor.compress(
                test_tensor,
                compressed_buffer
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            if success:
                compression_times.append(end - start)
                compressed_sizes.append(actual_size) # 💡 直接用 actual_size
        
        # 测量解压缩时间
        decompression_times = []
        
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            decompressed_tensor = torch.empty_like(test_tensor)
            success = compressor.decompress(
                compressed_buffer,
                compressed_sizes[0],
                decompressed_tensor
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            if success:
                decompression_times.append(end - start)
        
        # 计算性能指标
        avg_compression_time = np.mean(compression_times)
        avg_decompression_time = np.mean(decompression_times)
        avg_compressed_size = np.mean(compressed_sizes)
        
        compression_speed = original_size_bytes / avg_compression_time / 1e9  # GB/s
        decompression_speed = original_size_bytes / avg_decompression_time / 1e9  # GB/s
        compression_ratio = original_size_bytes / avg_compressed_size
        
        # 计算错误
        decompressed_tensor = torch.empty_like(test_tensor)
        compressor.decompress(
            compressed_buffer,
            compressed_sizes[0],
            decompressed_tensor
        )
        
        abs_errors = torch.abs(test_tensor - decompressed_tensor)
        max_error = torch.max(abs_errors).item()
        mean_error = torch.mean(abs_errors).item()
        
        results = {
            "tensor_size": tensor_size,
            "original_size_bytes": original_size_bytes,
            "compressed_size_bytes": avg_compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time_ms": avg_compression_time * 1000,
            "decompression_time_ms": avg_decompression_time * 1000,
            "compression_speed_gbps": compression_speed,
            "decompression_speed_gbps": decompression_speed,
            "max_error": max_error,
            "mean_error": mean_error,
            "error_bound": error_bound,
            "encoding_mode": encoding_mode
        }
        
        logger.info(
            f"Size: {tensor_size}, "
            f"Compression: {compression_speed:.2f} GB/s, "
            f"Decompression: {decompression_speed:.2f} GB/s, "
            f"Ratio: {compression_ratio:.2f}x, "
            f"Max Error: {max_error:.2e}"
        )
        
        return results
    
    def _parse_mode(self, mode: str):
        """解析编码模式"""
        mode_map = {
            "fixed": cuszp_wrapper_cpp.CuszpMode.MODE_FIXED,
            "plain": cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
            "outlier": cuszp_wrapper_cpp.CuszpMode.MODE_OUTLIER
        }
        return mode_map.get(mode.lower(), cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN)
    
    def benchmark_error_bound_sweep(
        self,
        tensor_size: int,
        error_bounds: List[float],
        encoding_mode: str = "plain"
    ) -> List[Dict]:
        """扫描不同错误边界的性能"""
        results = []
        
        for error_bound in error_bounds:
            result = self.benchmark_compression(
                tensor_size, error_bound, encoding_mode
            )
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Compression performance benchmark")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--tensor-size", type=int, default=1048576,
                       help="Tensor size (number of elements)")
    parser.add_argument("--error-bound", type=float, default=1e-4,
                       help="Error bound")
    parser.add_argument("--encoding-mode", type=str, default="plain",
                       choices=["fixed", "plain", "outlier"],
                       help="Encoding mode")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations")
    parser.add_argument("--error-bound-sweep", action="store_true",
                       help="Sweep error bounds")
    parser.add_argument("--error-bounds", type=float, nargs="+",
                       default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                       help="Error bounds for sweep")
    parser.add_argument("--output", type=str, default="compression_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    if not CUSZP_AVAILABLE:
        logger.error("cuSZp wrapper not available. Please compile it first.")
        return
    
    benchmark = CompressionBenchmark(device_id=args.device_id)
    
    # 打印GPU信息
    logger.info(f"GPU: {torch.cuda.get_device_name(args.device_id)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    
    if args.error_bound_sweep:
        # 扫描错误边界
        results = benchmark.benchmark_error_bound_sweep(
            args.tensor_size,
            args.error_bounds,
            args.encoding_mode
        )
    else:
        # 单次测试
        result = benchmark.benchmark_compression(
            args.tensor_size,
            args.error_bound,
            args.encoding_mode,
            args.iterations
        )
        results = [result]
    
    # 保存结果
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

