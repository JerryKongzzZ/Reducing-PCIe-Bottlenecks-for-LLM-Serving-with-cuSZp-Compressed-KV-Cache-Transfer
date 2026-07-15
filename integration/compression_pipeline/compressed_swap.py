"""
Integration module for vLLM to support compressed KV cache swapping.
This module provides a wrapper and monkey-patching utilities to replace
vLLM's original BlockSpaceManager or CacheEngine swapping methods.
"""

import torch
import ctypes
import logging
from typing import Dict, List, Tuple, Optional
import sys
import os
import time
import zlib
import numpy as np

logger = logging.getLogger(__name__)

# Try to import cuSZp wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import cuszp_wrapper_cpp
except ImportError as e:
    logger.error(f"Failed to import cuszp_wrapper_cpp: {e}")
    logger.error("cuSZp compression will not be available.")
    cuszp_wrapper_cpp = None

# Optional adaptive scheduler (local import when present)
try:
    from adaptive_scheduler import PCIEAdaptiveScheduler, Block as SchedulerBlock
except Exception:
    PCIEAdaptiveScheduler = None
    SchedulerBlock = None


class CuSZpCompressor:
    """Python wrapper for cuSZp C++ extension"""
    
    def __init__(
        self, 
        error_bound: float = 1e-4,
        encoding_mode: str = "plain",
        device_id: int = 0
    ):
        self.error_bound = error_bound
        self.device_id = device_id

        # Parse encoding mode
        self.encoding_mode = encoding_mode.lower() if encoding_mode is not None else "plain"

        # Attempt to initialize cuSZp if available
        if cuszp_wrapper_cpp is not None:
            mode = cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN
            if self.encoding_mode == "fixed":
                mode = cuszp_wrapper_cpp.CuszpMode.MODE_FIXED
            elif self.encoding_mode == "outlier":
                mode = cuszp_wrapper_cpp.CuszpMode.MODE_OUTLIER

            # Initialize configuration
            self.config = cuszp_wrapper_cpp.CompressionConfig(
                error_bound=error_bound,
                use_relative_error=True,
                processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
                encoding_mode=mode,
                data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT
            )

            # Initialize compressor
            self.compressor = cuszp_wrapper_cpp.CuSZpWrapper(self.config, device_id)
        else:
            self.config = None
            self.compressor = None

        # Performance counters
        self.stats = {
            "bytes_original": 0,
            "bytes_compressed": 0,
            "swap_out_count": 0,
            "swap_in_count": 0
        }

    def compress_with_eps(self, input_tensor, compressed_buffer, eps_rel: float):
        """Update runtime error bound and compress using the underlying C++ wrapper.

        Returns the same tuple as the pybind `compress` wrapper:
        (success: bool, compressed_buffer: Tensor, compressed_size: int, actual_error_bound: float)
        """
        # If cuSZp binding is available, use it
        if self.compressor is not None:
            return self.compressor.compress(input_tensor, compressed_buffer, float(eps_rel))

        # Fallback: zlib-based compression on CPU
        try:
            # Move to CPU float32 contiguous
            cpu = input_tensor.detach().cpu().contiguous().view(-1)
            orig_bytes = cpu.numpy().tobytes()
            comp_bytes = zlib.compress(orig_bytes)
            comp_size = len(comp_bytes)

            # Create CPU uint8 tensor containing compressed bytes
            buf = torch.frombuffer(comp_bytes, dtype=torch.uint8).to(torch.device("cpu"))
            actual_eb = float(eps_rel)

            # Return (success, buffer, size, actual_eb)
            return True, buf, comp_size, actual_eb
        except Exception:
            logger.exception("Fallback compression failed")
            return False, compressed_buffer, 0, float(eps_rel)

    def decompress(self, compressed_buffer, compressed_size, output_tensor, actual_error_bound):
        """Decompress using cuSZp if available, otherwise fallback to zlib.

        Returns True/False.
        """
        if self.compressor is not None:
            return self.compressor.decompress(compressed_buffer, compressed_size, output_tensor, actual_error_bound)

        try:
            # compressed_buffer is a CPU uint8 tensor
            data = bytes(compressed_buffer[:int(compressed_size)].cpu().numpy().tobytes())
            decomp = zlib.decompress(data)
            arr = np.frombuffer(decomp, dtype=np.float32)
            # reshape to output_tensor numel
            out = torch.from_numpy(arr).view_as(output_tensor).to(output_tensor.device)
            output_tensor.copy_(out)
            return True
        except Exception:
            logger.exception("Fallback decompression failed")
            return False
        
    def get_compression_ratio(self) -> float:
        """Calculate average compression ratio"""
        if self.stats["bytes_compressed"] == 0:
            return 1.0
        return self.stats["bytes_original"] / self.stats["bytes_compressed"]


class CompressedCacheEngineMonkeyPatch:
    """
    Monkey patching utility for vLLM's CacheEngine.
    This replaces the swap_in and swap_out methods to inject compression.
    """
    
    def __init__(
        self,
        cache_engine,
        compressor: CuSZpCompressor,
        scheduler: Optional[PCIEAdaptiveScheduler] = None,
        enable_auto_layer_infer: bool = False,
        low_threshold_bytes: int = 32 * 1024 * 1024,
        high_threshold_bytes: int = 128 * 1024 * 1024,
    ):
        self.engine = cache_engine
        self.compressor = compressor

        # Optional adaptive scheduler (can be supplied externally or auto-inferred)
        self.scheduler = scheduler
        self.enable_auto_layer_infer = enable_auto_layer_infer
        self.low_threshold_bytes = low_threshold_bytes
        self.high_threshold_bytes = high_threshold_bytes

        # Save original methods
        self._swap_out_blocks_to_host_original = self.engine._swap_out_blocks_to_host
        self._swap_in_blocks_from_host_original = self.engine._swap_in_blocks_from_host

        # Host memory buffer for compressed data (can't be stored in regular cache)
        # In a real implementation, we would need a dedicated compressed block manager
        self.compressed_blocks_store = {}
        # Decompression streams per sensitivity class (lazy-created)
        self._decompression_streams: Dict[str, torch.cuda.Stream] = {}
        
    def patch(self):
        """Apply monkey patch"""
        logger.info("Patching vLLM CacheEngine with cuSZp compression...")
        self.engine._swap_out_blocks_to_host = self._swap_out_blocks_to_host_patched
        self.engine._swap_in_blocks_from_host = self._swap_in_blocks_from_host_patched
        
    def unpatch(self):
        """Restore original methods"""
        logger.info("Restoring original vLLM CacheEngine methods...")
        self.engine._swap_out_blocks_to_host = self._swap_out_blocks_to_host_original
        self.engine._swap_in_blocks_from_host = self._swap_in_blocks_from_host_original
        
    def _swap_out_blocks_to_host_patched(self, src_cache, dst_cache, src_block_indices, dst_block_indices):
        """Patched swap_out_blocks_to_host method with congestion-aware eps allocation."""
        # Note: This is a simplified prototype implementation
        # In production vLLM, cache operations are highly optimized with custom CUDA kernels

        num_blocks = len(src_block_indices)
        if num_blocks == 0:
            return

        # Prepare pending block list for scheduler
        pending_blocks = []
        for i, (src_idx, dst_idx) in enumerate(zip(src_block_indices, dst_block_indices)):
            for layer_idx in range(len(src_cache)):
                _src_cache = src_cache[layer_idx][src_idx]
                size_bytes = _src_cache.numel() * _src_cache.element_size()
                key = (layer_idx, int(dst_idx))
                pending_blocks.append(SchedulerBlock(key=key, layer_idx=layer_idx, block_idx=int(dst_idx), size_bytes=size_bytes, tensor=_src_cache))

        # Auto-infer a default layer sensitivity map on first use if requested
        if self.scheduler is None and self.enable_auto_layer_infer:
            try:
                num_layers = len(src_cache)
                layer_map = PCIEAdaptiveScheduler.default_layer_sensitivity(num_layers)
                self.scheduler = PCIEAdaptiveScheduler(layer_map, low_threshold_bytes=self.low_threshold_bytes, high_threshold_bytes=self.high_threshold_bytes)
                logger.info("Adaptive scheduler auto-initialized with %d layers", num_layers)
            except Exception:
                self.scheduler = None

        # Compute eps assignment map
        eps_map = {}
        if self.scheduler is not None:
            try:
                eps_map = self.scheduler.compute_eps_map(pending_blocks)
            except Exception:
                logger.exception("Scheduler failed; falling back to static error bound")
                eps_map = {}

        # Process and compress block-by-block using eps_map (or default)
        for b in pending_blocks:
            layer_idx = b.layer_idx
            dst_idx = b.block_idx
            _src_cache = b.tensor

            estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(
                _src_cache.numel() * _src_cache.element_size()
            )

            compressed_buffer = torch.empty(
                estimated_size,
                dtype=torch.uint8,
                device=_src_cache.device,
            )

            eps_rel = eps_map.get(b.key, self.compressor.error_bound)

            # Use the new compress_with_eps wrapper which updates runtime config
            success, compressed_buffer, actual_size, actual_eb = self.compressor.compress_with_eps(
                _src_cache, compressed_buffer, float(eps_rel)
            )

            if not success:
                logger.warning("Compression failed, falling back to uncompressed swap")
                self._swap_out_blocks_to_host_original(
                    src_cache, dst_cache, src_block_indices, dst_block_indices
                )
                return

            cpu_compressed = compressed_buffer[:actual_size].cpu()

            # Move compressed blob to pinned host memory for async H2D transfer
            try:
                cpu_compressed = compressed_buffer[:actual_size].cpu().pin_memory()
            except Exception:
                # pin_memory may fail on non-CPU or if not supported; keep regular CPU tensor
                cpu_compressed = compressed_buffer[:actual_size].cpu()

            # Determine sensitivity category for this layer
            if self.scheduler is not None:
                sensitivity = self.scheduler.layer_sensitivity.get(layer_idx, "deep")
            else:
                sensitivity = "deep"

            store_key = (layer_idx, int(dst_idx))
            self.compressed_blocks_store[store_key] = {
                'data': cpu_compressed,
                'original_size': _src_cache.numel(),
                'shape': _src_cache.shape,
                'actual_eb': actual_eb,
                'sensitivity': sensitivity,
                'arrival_ts': time.time(),
            }

            # Update statistics
            self.compressor.stats["bytes_original"] += _src_cache.numel() * _src_cache.element_size()
            self.compressor.stats["bytes_compressed"] += actual_size
            self.compressor.stats["swap_out_count"] += 1
                
    def _swap_in_blocks_from_host_patched(self, src_cache, dst_cache, src_block_indices, dst_block_indices):
        """Patched swap_in_blocks_from_host method with prioritized async decompression.

        Strategy:
        - Collect requested blocks and sort by sensitivity (shallow first).
        - For `shallow` blocks: perform H2D + decompress on a dedicated stream and
          synchronize before returning (ensures low TTFT for critical layers).
        - For `mid`/`deep` blocks: launch H2D + decompress asynchronously and
          record a CUDA event; these complete in background.
        """
        num_blocks = len(src_block_indices)
        if num_blocks == 0:
            return

        # Build task list for all requested blocks
        tasks = []  # each task: dict with keys below
        for i, (src_idx, dst_idx) in enumerate(zip(src_block_indices, dst_block_indices)):
            for layer_idx in range(len(dst_cache)):
                store_key = (layer_idx, int(src_idx))
                if store_key not in self.compressed_blocks_store:
                    # If any requested block missing, fallback to original behavior for this pair
                    self._swap_in_blocks_from_host_original(src_cache, dst_cache, [src_idx], [dst_idx])
                    return

                block_info = self.compressed_blocks_store[store_key]
                compressed_cpu = block_info['data']
                compressed_size = int(compressed_cpu.numel()) if hasattr(compressed_cpu, 'numel') else int(len(compressed_cpu))
                sensitivity = block_info.get('sensitivity', 'deep')

                tasks.append({
                    'store_key': store_key,
                    'layer_idx': layer_idx,
                    'dst_idx': dst_idx,
                    'compressed_cpu': compressed_cpu,
                    'compressed_size': compressed_size,
                    'actual_eb': block_info['actual_eb'],
                    'sensitivity': sensitivity,
                    'dst_tensor': dst_cache[layer_idx][dst_idx],
                    'meta': block_info,
                })

        # Sort by sensitivity: shallow -> mid -> deep
        priority = {'shallow': 0, 'mid': 1, 'deep': 2}
        tasks.sort(key=lambda t: priority.get(t['sensitivity'], 2))

        # Launch decompression tasks. Shallow blocks are synchronized before return.
        for task in tasks:
            sens = task['sensitivity']
            # Create/get stream for this sensitivity class
            stream = self._decompression_streams.get(sens)
            if stream is None:
                stream = torch.cuda.Stream()
                self._decompression_streams[sens] = stream

            dst_tensor = task['dst_tensor']
            dev = dst_tensor.device

            with torch.cuda.stream(stream):
                # Asynchronous H2D from pinned host memory
                try:
                    compressed_gpu = task['compressed_cpu'].to(device=dev, non_blocking=True)
                except Exception:
                    # Fallback: cuda() call
                    compressed_gpu = task['compressed_cpu'].cuda(non_blocking=True)

                # Decompress on the selected stream (pybind uses current CUDA stream)
                success = self.compressor.compressor.decompress(
                    compressed_gpu,
                    task['compressed_size'],
                    dst_tensor,
                    task['actual_eb']
                )

                if not success:
                    logger.warning("Decompression failed for %s; falling back to original swap_in", task['store_key'])
                    self._swap_in_blocks_from_host_original(src_cache, dst_cache, [task['store_key'][1]], [task['dst_idx']])
                    return

                # Record completion event
                ev = torch.cuda.Event()
                ev.record(stream)
                # Save event handle for possible later synchronization
                self.compressed_blocks_store[task['store_key']]['decompress_event'] = ev

            # If this is a shallow (critical) block, wait for completion before returning
            if sens == 'shallow':
                ev.synchronize()
                # Optionally free host-side compressed buffer after completion
                try:
                    del self.compressed_blocks_store[task['store_key']]['data']
                except Exception:
                    pass
                self.compressor.stats['swap_in_count'] += 1
            else:
                # For mid/deep: launched in background; update counter and continue
                self.compressor.stats['swap_in_count'] += 1

# Example usage function
def setup_vllm_compression(engine, error_bound=1e-4, enable_adaptive: bool = False, low_threshold_bytes: int = 32 * 1024 * 1024, high_threshold_bytes: int = 128 * 1024 * 1024):
    """
    Helper function to set up compression in an existing vLLM engine
    """
    try:
        compressor = CuSZpCompressor(error_bound=error_bound)
        # Optionally enable adaptive scheduler. If enabled but no explicit scheduler
        # is provided, the patcher will auto-infer a per-layer sensitivity map
        # on first use based on the observed number of layers.
        scheduler = None
        # Apply the original CacheEngine monkey patch for backwards compatibility
        patcher = CompressedCacheEngineMonkeyPatch(
            engine.cache_engine,
            compressor,
            scheduler=scheduler,
            enable_auto_layer_infer=enable_adaptive,
            low_threshold_bytes=low_threshold_bytes,
            high_threshold_bytes=high_threshold_bytes,
        )
        patcher.patch()

        # Also patch vLLM's OffloadingWorker.register_handler to install
        # a proxy handler that can later implement compression-aware
        # transfers at the worker level.
        try:
            from integration.compression_pipeline.offloading_wrapper import patch_offloading_worker_register

            patch_offloading_worker_register(compressor=compressor, scheduler=scheduler)
        except Exception:
            logger.exception("Failed to apply OffloadingWorker register patch")

        return patcher
    except Exception as e:
        logger.error(f"Failed to setup vLLM compression: {e}")
        return None
