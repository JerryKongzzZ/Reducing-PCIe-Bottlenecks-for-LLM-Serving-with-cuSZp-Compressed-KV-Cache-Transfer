"""Wrapper to patch vLLM's OffloadingWorker.register_handler

This installs a proxy handler that can later be extended to perform
compression-aware transfers. For now it preserves original behavior
but exposes hooks and stores references to `cuszp_wrapper_cpp` and
the `PCIEAdaptiveScheduler` when available.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple
import threading
import numpy as np
import torch

try:
    import cuszp_wrapper_cpp
except Exception:
    cuszp_wrapper_cpp = None

try:
    from vllm.v1.kv_offload.worker.worker import OffloadingHandler, OffloadingWorker
except Exception:
    OffloadingHandler = None  # type: ignore
    OffloadingWorker = None  # type: ignore

logger = logging.getLogger(__name__)


class CompressedProxyHandler:
    """Proxy that wraps an existing OffloadingHandler.

    Methods delegate to the wrapped handler. This proxy records submitted
    transfer specs and asynchronously compresses GPU->CPU completed
    transfers using the `cuszp_wrapper_cpp` binding when available.
    """

    def __init__(
        self,
        inner: Any,
        compressor: Any = None,
        scheduler: Any = None,
        src_medium: str | None = None,
        dst_medium: str | None = None,
    ):
        self._inner = inner
        self.compressor = compressor
        self.scheduler = scheduler
        self.src_medium = src_medium
        self.dst_medium = dst_medium

        # Track submitted specs: job_id -> spec
        self._submitted_specs: dict[int, Any] = {}
        # Store compressed blobs: (tensor_idx, block_idx) -> dict
        self._compressed_store: dict[tuple[int, int], dict] = {}
        # Background lock
        self._bg_lock = threading.Lock()

    def transfer_async(self, job_id: int, spec: Any) -> bool:
        # Record the spec for possible post-copy compression
        try:
            self._submitted_specs[job_id] = spec
        except Exception:
            logger.exception("Failed to record submitted spec")

        try:
            return self._inner.transfer_async(job_id, spec)
        except Exception:
            logger.exception("Inner handler transfer_async failed")
            # Cleanup recorded spec on failure
            try:
                del self._submitted_specs[job_id]
            except Exception:
                pass
            return False

    def get_finished(self) -> list:
        finished = self._inner.get_finished()

        # Post-process finished GPU->CPU transfers: compress CPU-side rows
        processed = []
        for tr in finished:
            try:
                if (
                    tr.transfer_type is not None
                    and tr.transfer_type[0] == "GPU"
                    and tr.transfer_type[1] == "CPU"
                ):
                    spec = self._submitted_specs.pop(tr.job_id, None)
                    if spec is not None and cuszp_wrapper_cpp is not None:
                        # Schedule compression in background to avoid blocking
                        t = threading.Thread(target=self._compress_transfer, args=(spec,))
                        t.daemon = True
                        t.start()
            except Exception:
                logger.exception("Error in post-processing finished transfer %s", tr.job_id)
            processed.append(tr)

        return processed

    def wait(self, job_ids: set[int]) -> None:
        return self._inner.wait(job_ids)

    def shutdown(self) -> None:
        try:
            return self._inner.shutdown()
        except Exception:
            logger.exception("Inner handler shutdown failed")

    def _compress_transfer(self, spec: Any) -> None:
        """Compress CPU-side destination blocks for a finished GPU->CPU transfer.

        This reads the CPU int8-backed rows from the inner handler and
        reinterprets the bytes as float32 for cuSZp compression. Compressed
        blobs are stored in `self._compressed_store`.
        """
        try:
            # spec is a tuple (src_spec, dst_spec)
            _, dst_spec = spec
            # dst_spec likely has attribute `block_ids` (numpy array)
            block_ids = getattr(dst_spec, "block_ids", None)
            if block_ids is None:
                return

            # iterate through handler's CPU tensors (destination)
            dst_tensors = getattr(self._inner, "dst_tensors", None)
            if dst_tensors is None:
                return

            # For each tensor in dst_tensors, compress requested rows
            for t_idx, cpu_tensor in enumerate(dst_tensors):
                # cpu_tensor: int8 CPU tensor with shape (num_blocks, page_size_bytes)
                page_size_bytes = cpu_tensor.shape[1]
                for blk in block_ids.tolist():
                    try:
                        row = cpu_tensor[int(blk)]
                        # Convert to numpy bytes and reinterpret as float32
                        arr = row.cpu().numpy()
                        if arr.size % 4 != 0:
                            # unexpected; skip
                            continue
                        floats = arr.view(np.float32)

                        # Move to GPU for compressor
                        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                        tensor_f = torch.from_numpy(floats).to(dev)

                        # Prepare compressor and compressed buffer
                        cfg = cuszp_wrapper_cpp.CompressionConfig(
                            error_bound=1e-4,
                            use_relative_error=True,
                            processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
                            encoding_mode=cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
                            data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT,
                        )
                        comp = cuszp_wrapper_cpp.CuSZpWrapper(cfg, 0)
                        est = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(
                            tensor_f.numel() * tensor_f.element_size()
                        )
                        compressed_buffer = torch.empty(est, dtype=torch.uint8, device=tensor_f.device)
                        success, comp_buf, comp_size, actual_eb = comp.compress(tensor_f, compressed_buffer, float(1e-4))
                        if not success:
                            continue

                        # Move compressed bytes to pinned CPU memory for storage
                        cpu_comp = comp_buf[:int(comp_size)].cpu()
                        try:
                            cpu_comp = cpu_comp.pin_memory()
                        except Exception:
                            pass

                        meta = {
                            "data": cpu_comp,
                            "size": int(comp_size),
                            "eb": float(actual_eb),
                            "page_bytes": int(page_size_bytes),
                        }
                        with self._bg_lock:
                            self._compressed_store[(t_idx, int(blk))] = meta
                    except Exception:
                        logger.exception("Failed compressing block %s in tensor %d", blk, t_idx)
        except Exception:
            logger.exception("Error in _compress_transfer")


def patch_offloading_worker_register(compressor: Any = None, scheduler: Any = None):
    """Monkey-patch OffloadingWorker.register_handler to wrap handlers.

    This is intentionally conservative: it wraps the handler objects
    with `CompressedProxyHandler` so we can later implement compression
    behavior without touching vLLM internals further.
    """
    if OffloadingWorker is None:
        logger.warning("vLLM OffloadingWorker not importable; skipping patch")
        return

    orig_register = OffloadingWorker.register_handler

    def _register(self, src_cls, dst_cls, handler):
        logger.info("Registering offloading handler for %s -> %s", src_cls, dst_cls)
        try:
            proxy = CompressedProxyHandler(handler, compressor=compressor, scheduler=scheduler)
            return orig_register(self, src_cls, dst_cls, proxy)
        except Exception:
            logger.exception("Failed to wrap handler; registering original")
            return orig_register(self, src_cls, dst_cls, handler)

    OffloadingWorker.register_handler = _register
    logger.info("Patched OffloadingWorker.register_handler with compression proxy")
