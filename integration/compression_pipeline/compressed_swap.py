"""
压缩传输管道：在vLLM的CPU-GPU交换中使用cuSZp压缩

这个模块修改了vLLM的swap_out_blocks_to_host和swap_in_blocks_from_host函数，
添加了压缩-传输-解压缩的工作流程。
"""

import torch
import ctypes
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# 尝试导入cuSZp包装器（需要编译）
try:
    import cuszp_wrapper_cpp
    CUSZP_AVAILABLE = True
except ImportError:
    logger.warning("cuSZp wrapper not available. Compression will be disabled.")
    CUSZP_AVAILABLE = False


class CompressedSwapManager:
    """
    管理压缩的CPU-GPU交换操作
    """
    
    def __init__(
        self,
        enable_compression: bool = True,
        error_bound: float = 1e-4,
        use_relative_error: bool = True,
        encoding_mode: str = "plain",
        device_id: int = 0
    ):
        """
        初始化压缩交换管理器
        
        Args:
            enable_compression: 是否启用压缩
            error_bound: 错误边界
            use_relative_error: 是否使用相对错误边界
            encoding_mode: 编码模式 ("fixed", "plain", "outlier")
            device_id: GPU设备ID
        """
        self.enable_compression = enable_compression and CUSZP_AVAILABLE
        self.device_id = device_id
        
        if self.enable_compression:
            # 创建cuSZp包装器
            config = cuszp_wrapper_cpp.CompressionConfig(
                error_bound=error_bound,
                use_relative_error=use_relative_error,
                encoding_mode=self._parse_encoding_mode(encoding_mode),
                processing_dim=1,  # 1D处理（适用于KV cache）
                data_type=0  # float32
            )
            self.compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, device_id)
            
            # 创建CUDA流用于异步操作
            self.compression_stream = torch.cuda.Stream(device=device_id)
            self.decompression_stream = torch.cuda.Stream(device=device_id)
            
            logger.info(f"Compressed swap enabled with error_bound={error_bound}, mode={encoding_mode}")
        else:
            self.compressor = None
            logger.info("Compressed swap disabled")
    
    def _parse_encoding_mode(self, mode: str) -> int:
        """解析编码模式字符串到cuSZp枚举值"""
        mode_map = {
            "fixed": 0,
            "plain": 1,
            "outlier": 2
        }
        return mode_map.get(mode.lower(), 1)  # 默认使用plain
    
    def swap_out_blocks_to_host_compressed(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """
        压缩版本的swap_out：将GPU块压缩后传输到CPU
        
        工作流程：
        1. 从GPU缓存中提取要交换的块
        2. 在GPU上压缩这些块
        3. 将压缩后的数据异步传输到CPU
        4. 在CPU上存储压缩数据（而不是原始数据）
        """
        if not self.enable_compression:
            # 回退到原始实现
            self._swap_out_blocks_to_host_original(
                src_cache, dst_cache, src_block_indices, dst_block_indices
            )
            return
        
        try:
            # 提取要交换的块
            _src_cache = src_cache[:, src_block_indices]
            
            # 估算压缩缓冲区大小
            original_size = _src_cache.numel() * _src_cache.element_size()
            compressed_buffer_size = self.compressor.estimate_compressed_buffer_size(original_size)
            
            # 在GPU上分配压缩缓冲区
            compressed_buffer = torch.empty(
                (compressed_buffer_size,),
                dtype=torch.uint8,
                device=src_cache.device
            )
            
            # 在压缩流上异步压缩
            with torch.cuda.stream(self.compression_stream):
                # 💡 修改这里：使用元组解包接收返回值，只传2个参数
                success, compressed_buffer, actual_size = self.compressor.compress(
                    _src_cache,
                    compressed_buffer
                )
                
                if not success:
                    logger.warning("Compression failed, falling back to uncompressed swap")
                    self._swap_out_blocks_to_host_original(
                        src_cache, dst_cache, src_block_indices, dst_block_indices
                    )
                    return
                
                # 💡 修改这里：使用 actual_size 而不是 compressed_size.value
                compressed_cpu = compressed_buffer[:actual_size].cpu()
                
                # 存储压缩数据（这里简化处理，实际需要修改vLLM的存储结构）
                # TODO: 修改vLLM的CPU缓存结构以支持压缩数据
                dst_cache[:, dst_block_indices] = _src_cache.cpu()  # 临时：仍使用原始数据
                
        except Exception as e:
            logger.error(f"Error in compressed swap out: {e}")
            # 回退到原始实现
            self._swap_out_blocks_to_host_original(
                src_cache, dst_cache, src_block_indices, dst_block_indices
            )
    
    def swap_in_blocks_from_host_compressed(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """
        压缩版本的swap_in：从CPU解压缩数据并传输到GPU
        
        工作流程：
        1. 从CPU缓存中读取压缩数据
        2. 异步传输压缩数据到GPU
        3. 在GPU上解压缩
        4. 将解压缩的数据写入GPU缓存
        """
        if not self.enable_compression:
            # 回退到原始实现
            self._swap_in_blocks_from_host_original(
                src_cache, dst_cache, src_block_indices, dst_block_indices
            )
            return
        
        try:
            # 从CPU读取压缩数据
            # 注意：这里假设src_cache包含压缩数据
            # 实际实现中，需要修改vLLM来区分压缩和未压缩的数据
            compressed_cpu = src_cache[:, src_block_indices]
            
            # 异步传输到GPU
            compressed_gpu = compressed_cpu.to(dst_cache.device, non_blocking=True)
            
            # 在解压缩流上异步解压缩
            with torch.cuda.stream(self.decompression_stream):
                _dst_cache = dst_cache[:, dst_block_indices]
                original_size = compressed_cpu.numel()
                
                # 💡 修改这里：去掉第四个 stream 参数
                success = self.compressor.decompress(
                    compressed_gpu,
                    original_size,  # 压缩大小
                    _dst_cache
                )
                
                if not success:
                    logger.warning("Decompression failed, falling back to uncompressed swap")
                    self._swap_in_blocks_from_host_original(
                        src_cache, dst_cache, src_block_indices, dst_block_indices
                    )
                    return
                
        except Exception as e:
            logger.error(f"Error in compressed swap in: {e}")
            # 回退到原始实现
            self._swap_in_blocks_from_host_original(
                src_cache, dst_cache, src_block_indices, dst_block_indices
            )
    
    def _swap_out_blocks_to_host_original(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """原始的非压缩swap_out实现"""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()
    
    def _swap_in_blocks_from_host_original(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """原始的非压缩swap_in实现"""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)


# 全局压缩管理器实例（可以在vLLM初始化时创建）
_global_compressed_swap_manager: Optional[CompressedSwapManager] = None


def get_compressed_swap_manager() -> Optional[CompressedSwapManager]:
    """获取全局压缩交换管理器"""
    return _global_compressed_swap_manager


def initialize_compressed_swap(
    enable_compression: bool = True,
    error_bound: float = 1e-4,
    use_relative_error: bool = True,
    encoding_mode: str = "plain",
    device_id: int = 0
) -> CompressedSwapManager:
    """
    初始化全局压缩交换管理器
    
    这个函数应该在vLLM初始化时调用
    """
    global _global_compressed_swap_manager
    _global_compressed_swap_manager = CompressedSwapManager(
        enable_compression=enable_compression,
        error_bound=error_bound,
        use_relative_error=use_relative_error,
        encoding_mode=encoding_mode,
        device_id=device_id
    )
    return _global_compressed_swap_manager

