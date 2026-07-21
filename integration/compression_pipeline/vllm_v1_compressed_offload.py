"""Compressed CPU offloading connector for vLLM v1 (tested with vLLM 0.23).

The stock CPU connector copies full KV pages between GPU and host. This
connector compresses each real GPU KV page before D2H and stores the variable-
length payload in pinned CPU memory. On load it transfers only that payload,
decompresses on the GPU, and casts back to the cache's original dtype.

The first implementation intentionally supports ``block_size_factor == 1``.
Rejecting unsupported layouts is safer than silently copying the wrong bytes.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterable
import zlib

import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

module_dir = str(Path(__file__).resolve().parent)
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

import cuszp_wrapper_cpp
from integration.compression_pipeline import native_lossless


MIN_CUSZP_ELEMENTS = 36864
logger = logging.getLogger(__name__)
CUSZP_MODES = {
    "plain": cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
    "fixed": cuszp_wrapper_cpp.CuszpMode.MODE_FIXED,
    "outlier": cuszp_wrapper_cpp.CuszpMode.MODE_OUTLIER,
}


@dataclass
class BundleComponent:
    tensor_idx: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    numel: int
    layer_axis: int | None = None
    layer_indices: tuple[int, ...] = ()
    destination_view_shape: tuple[int, ...] = ()


@dataclass
class EncodedSegment:
    payload: torch.Tensor
    compressed_size: int
    actual_error_bound: float
    encoding: str
    components: tuple[BundleComponent, ...]
    unpadded_numel: int
    compressed_numel: int
    original_bytes: int
    requested_error_bound: float
    quant_scale: float | None = None
    cuszp_mode: str | None = None


@dataclass
class CompressedBundle:
    segments: tuple[EncodedSegment, ...]
    compressed_size: int
    original_bytes: int


@dataclass
class RestoreStageTimings:
    """Profiled CPU-to-GPU restore stages for one encoded segment."""

    cpu_decode_seconds: float = 0.0
    h2d_seconds: float = 0.0
    gpu_decode_seconds: float = 0.0
    scatter_seconds: float = 0.0

    def add(self, other: "RestoreStageTimings") -> None:
        self.cpu_decode_seconds += other.cpu_decode_seconds
        self.h2d_seconds += other.h2d_seconds
        self.gpu_decode_seconds += other.gpu_decode_seconds
        self.scatter_seconds += other.scatter_seconds

    def as_dict(self) -> dict[str, float]:
        return {
            "cpu_decode_seconds": self.cpu_decode_seconds,
            "h2d_seconds": self.h2d_seconds,
            "gpu_decode_seconds": self.gpu_decode_seconds,
            "scatter_seconds": self.scatter_seconds,
        }


@dataclass(frozen=True)
class AdaptiveDecision:
    state: str
    pressure: float
    backlog_bytes: float
    layer_error_bounds: dict[int, float]
    state_changed: bool
    cuszp_mode: str


@dataclass(frozen=True)
class RestoreCostEstimate:
    original_bytes: int
    compressed_bytes: float
    raw_seconds: float
    compressed_seconds: float
    worthwhile: bool


class RestoreCostModel:
    """Predict whether compressed CPU-to-GPU restore beats raw H2D."""

    def __init__(
        self,
        h2d_bytes_per_second: float,
        compression_ratios: dict[float, float],
        decompression_bytes_per_second: dict[float, float],
        fixed_overhead_seconds: float = 0.0,
        min_savings_fraction: float = 0.05,
    ):
        if h2d_bytes_per_second <= 0:
            raise ValueError("H2D bandwidth must be positive")
        if fixed_overhead_seconds < 0:
            raise ValueError("restore fixed overhead cannot be negative")
        if not 0.0 <= min_savings_fraction < 1.0:
            raise ValueError("minimum savings fraction must be in [0, 1)")
        self.h2d_rate = float(h2d_bytes_per_second)
        self.ratios = {
            float(bound): float(ratio)
            for bound, ratio in compression_ratios.items()
            if float(bound) > 0 and float(ratio) > 1.0
        }
        self.decompression_rates = {
            float(bound): float(rate)
            for bound, rate in decompression_bytes_per_second.items()
            if float(bound) > 0 and float(rate) > 0
        }
        self.fixed_overhead = float(fixed_overhead_seconds)
        self.min_savings_fraction = float(min_savings_fraction)

    def estimate(self, original_bytes: int, error_bound: float) -> RestoreCostEstimate:
        original_bytes = int(original_bytes)
        bound = float(error_bound)
        if original_bytes < 0:
            raise ValueError("restore size cannot be negative")
        raw_seconds = original_bytes / self.h2d_rate
        ratio = self.ratios.get(bound)
        decompression_rate = self.decompression_rates.get(bound)
        if bound <= 0 or ratio is None or decompression_rate is None:
            return RestoreCostEstimate(
                original_bytes=original_bytes,
                compressed_bytes=float(original_bytes),
                raw_seconds=raw_seconds,
                compressed_seconds=raw_seconds,
                worthwhile=False,
            )
        compressed_bytes = original_bytes / ratio
        compressed_seconds = (
            compressed_bytes / self.h2d_rate
            + original_bytes / decompression_rate
            + self.fixed_overhead
        )
        worthwhile = compressed_seconds <= raw_seconds * (
            1.0 - self.min_savings_fraction
        )
        return RestoreCostEstimate(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            raw_seconds=raw_seconds,
            compressed_seconds=compressed_seconds,
            worthwhile=worthwhile,
        )


class AdaptiveErrorBoundController:
    """Deadline-based pressure controller constrained by layer safety limits.

    The backlog is a small fluid model: bytes are added when an offload job
    arrives and drained at the *measured* uncompressed PCIe service rate.  A
    pressure of 1.0 therefore means one transfer deadline worth of queued
    work.  Hysteresis avoids switching bounds around the deadline boundary.
    """

    def __init__(
        self,
        layer_safe_bounds: dict[int, float],
        service_rate_bytes_per_second: float,
        transfer_deadline_seconds: float,
        candidate_bounds: tuple[float, ...] = (1e-5, 1e-4, 1e-3),
        restore_cost_model: RestoreCostModel | None = None,
        mode_cost_models: dict[str, RestoreCostModel] | None = None,
        candidate_modes: tuple[str, ...] = ("plain",),
    ):
        if service_rate_bytes_per_second <= 0:
            raise ValueError("PCIe service rate must be positive")
        if transfer_deadline_seconds <= 0:
            raise ValueError("transfer deadline must be positive")
        candidates = tuple(sorted({float(x) for x in candidate_bounds if x > 0}))
        if not candidates:
            raise ValueError("adaptive controller needs positive candidate bounds")
        self.layer_safe_bounds = {
            int(layer): max(0.0, float(bound))
            for layer, bound in layer_safe_bounds.items()
        }
        self.service_rate = float(service_rate_bytes_per_second)
        self.deadline = float(transfer_deadline_seconds)
        self.candidate_bounds = candidates
        self.restore_cost_model = restore_cost_model
        modes = tuple(dict.fromkeys(str(mode).lower() for mode in candidate_modes))
        if not modes:
            raise ValueError("adaptive controller needs a cuSZp candidate mode")
        invalid_modes = tuple(mode for mode in modes if mode not in CUSZP_MODES)
        if invalid_modes:
            raise ValueError(f"unsupported adaptive cuSZp modes: {invalid_modes}")
        self.candidate_modes = modes
        self.mode_cost_models = dict(mode_cost_models or {})
        if restore_cost_model is not None:
            self.mode_cost_models.setdefault(modes[0], restore_cost_model)
        unknown_profiles = tuple(
            mode for mode in self.mode_cost_models if mode not in modes
        )
        if unknown_profiles:
            raise ValueError(
                f"cost profiles have no matching candidate mode: {unknown_profiles}"
            )
        self.backlog_bytes = 0.0
        self.last_arrival: float | None = None
        self.state = "green"

    def _bounds_and_mode_for_state(
        self, state: str, arriving_bytes: int
    ) -> tuple[dict[int, float], str]:
        target_bounds = {}
        for layer, safe_bound in self.layer_safe_bounds.items():
            eligible = [x for x in self.candidate_bounds if x <= safe_bound]
            if state == "green" or not eligible:
                bound = 0.0
            elif state == "yellow":
                bound = eligible[(len(eligible) - 1) // 2]
            else:
                bound = eligible[-1]
            target_bounds[layer] = bound

        default_mode = self.candidate_modes[0]
        if not self.mode_cost_models or not any(target_bounds.values()):
            return target_bounds, default_mode

        per_layer_bytes = max(
            1, int(arriving_bytes) // max(len(self.layer_safe_bounds), 1)
        )
        best_mode = default_mode
        best_bounds: dict[int, float] | None = None
        best_seconds = float("inf")
        for mode in self.candidate_modes:
            model = self.mode_cost_models.get(mode)
            if model is None:
                continue
            selected = {}
            total_seconds = 0.0
            for layer, bound in target_bounds.items():
                estimate = model.estimate(per_layer_bytes, bound)
                if bound > 0 and estimate.worthwhile:
                    selected[layer] = bound
                    total_seconds += estimate.compressed_seconds
                else:
                    selected[layer] = 0.0
                    total_seconds += estimate.raw_seconds
            if total_seconds < best_seconds:
                best_mode = mode
                best_bounds = selected
                best_seconds = total_seconds

        if best_bounds is None:
            return {layer: 0.0 for layer in target_bounds}, default_mode
        return best_bounds, best_mode

    def decide(self, arriving_bytes: int, now: float | None = None) -> AdaptiveDecision:
        if arriving_bytes < 0:
            raise ValueError("arriving bytes cannot be negative")
        now = time.perf_counter() if now is None else float(now)
        if self.last_arrival is not None:
            elapsed = max(0.0, now - self.last_arrival)
            self.backlog_bytes = max(
                0.0, self.backlog_bytes - elapsed * self.service_rate
            )
        self.last_arrival = now
        self.backlog_bytes += float(arriving_bytes)
        pressure = self.backlog_bytes / (self.service_rate * self.deadline)

        previous = self.state
        if self.state == "green":
            if pressure >= 1.0:
                self.state = "red"
            elif pressure >= 0.5:
                self.state = "yellow"
        elif self.state == "yellow":
            if pressure >= 1.0:
                self.state = "red"
            elif pressure < 0.4:
                self.state = "green"
        elif self.state == "red":
            if pressure < 0.8:
                self.state = "yellow" if pressure >= 0.4 else "green"

        selected_bounds, selected_mode = self._bounds_and_mode_for_state(
            self.state, arriving_bytes
        )
        return AdaptiveDecision(
            state=self.state,
            pressure=pressure,
            backlog_bytes=self.backlog_bytes,
            layer_error_bounds=selected_bounds,
            state_changed=self.state != previous,
            cuszp_mode=selected_mode,
        )


class MetricsRecorder:
    def __init__(self, path: str | None = None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, event: dict) -> None:
        if not self.path:
            return
        payload = dict(event)
        payload["timestamp_ns"] = time.time_ns()
        payload["pid"] = os.getpid()
        line = json.dumps(payload, sort_keys=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class CompressedPageStore:
    """Shared variable-length host storage for the two transfer directions."""

    def __init__(self):
        self._pages: dict[int, CompressedBundle] = {}
        self._lock = threading.Lock()
        self._gpu_pack_slab: torch.Tensor | None = None

    def put(self, cpu_block_id: int, bundle: CompressedBundle) -> None:
        with self._lock:
            self._pages[cpu_block_id] = bundle
    def put_many_packed(
        self,
        entries: list[tuple[int, CompressedBundle]],
    ) -> None:
        """Store one transfer job in a contiguous pinned host slab."""
        segments = [
            segment
            for _cpu_block_id, bundle in entries
            for segment in bundle.segments
        ]
        payloads = [
            segment.payload.view(torch.uint8).view(-1)
            for segment in segments
        ]
        alignment = 8
        layout: list[tuple[EncodedSegment, torch.Tensor, int, int]] = []
        total_bytes = 0
        for segment, payload in zip(segments, payloads):
            start = (total_bytes + alignment - 1) // alignment * alignment
            end = start + int(payload.numel())
            layout.append((segment, payload, start, end))
            total_bytes = end
        try:
            slab = torch.empty(
                total_bytes, dtype=torch.uint8, pin_memory=True
            )
        except RuntimeError:
            slab = torch.empty(total_bytes, dtype=torch.uint8)
        all_gpu = bool(payloads) and all(
            payload.is_cuda and payload.device == payloads[0].device
            for payload in payloads
        )
        if all_gpu:
            device = payloads[0].device
            if (
                self._gpu_pack_slab is None
                or self._gpu_pack_slab.device != device
                or self._gpu_pack_slab.numel() < total_bytes
            ):
                self._gpu_pack_slab = torch.empty(
                    total_bytes, dtype=torch.uint8, device=device
                )
            packed_gpu = self._gpu_pack_slab[:total_bytes]
            if len(layout) == 1 and layout[0][2] == 0:
                packed_gpu.copy_(layout[0][1])
            else:
                padding = torch.empty(
                    alignment, dtype=torch.uint8, device=device
                )
                parts = []
                cursor = 0
                for _segment, payload, start, end in layout:
                    if start > cursor:
                        parts.append(padding[: start - cursor])
                    parts.append(payload)
                    cursor = end
                torch.cat(parts, out=packed_gpu)
            slab.copy_(packed_gpu, non_blocking=True)
            torch.cuda.current_stream(device).synchronize()
            for segment, _payload, start, end in layout:
                segment.payload = slab[start:end]
            with self._lock:
                for cpu_block_id, bundle in entries:
                    self._pages[cpu_block_id] = bundle
            return

        cuda_devices = set()
        for segment, payload, start, end in layout:
            slab[start:end].copy_(payload, non_blocking=payload.is_cuda)
            if payload.is_cuda:
                cuda_devices.add(payload.device)
            segment.payload = slab[start:end]
        for device in cuda_devices:
            torch.cuda.current_stream(device).synchronize()
        with self._lock:
            for cpu_block_id, bundle in entries:
                self._pages[cpu_block_id] = bundle


    def get(self, cpu_block_id: int) -> CompressedBundle:
        with self._lock:
            bundle = self._pages.get(cpu_block_id)
        if bundle is None:
            raise KeyError(f"compressed bundle not found for CPU block {cpu_block_id}")
        return bundle


def _iter_page_operations(
    gpu_spec: GPULoadStoreSpec,
    cpu_spec: CPULoadStoreSpec,
    group_data_refs,
) -> Iterable[tuple[tuple[int, ...], int, int]]:
    """Yield ``(tensor_indices, gpu_block_id, cpu_block_id)`` bundles."""
    gpu_offset = 0
    cpu_offset = 0
    for group_size, refs in zip(gpu_spec.group_sizes, group_data_refs):
        group_size = int(group_size)
        gpu_ids = gpu_spec.block_ids[gpu_offset : gpu_offset + group_size]
        cpu_ids = cpu_spec.block_ids[cpu_offset : cpu_offset + group_size]
        if len(gpu_ids) != group_size or len(cpu_ids) != group_size:
            raise ValueError("GPU/CPU block specification length mismatch")
        tensor_indices = tuple(int(ref.tensor_idx) for ref in refs)
        for gpu_id, cpu_id in zip(gpu_ids, cpu_ids):
            yield tensor_indices, int(gpu_id), int(cpu_id)
        gpu_offset += group_size
        cpu_offset += group_size
    if gpu_offset != len(gpu_spec.block_ids) or cpu_offset != len(cpu_spec.block_ids):
        raise ValueError("unconsumed block IDs in offload transfer specification")


class CompressedOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        store: CompressedPageStore,
        gpu_to_cpu: bool,
        error_bound: float,
        device_id: int,
        metrics: MetricsRecorder,
        layer_error_bounds: dict[int, float] | None = None,
        layer_axis: int = 1,
        flat_layer_prefix: int = 2,
        codec: str = "cuszp",
        adaptive_controller: AdaptiveErrorBoundController | None = None,
        async_store: bool = False,
        profile_restore_stages: bool = False,
        batch_restore_transfers: bool = False,
        cuszp_mode: str = "plain",
    ):
        self.kv_caches = kv_caches
        self.store = store
        self.gpu_to_cpu = gpu_to_cpu
        self.error_bound = float(error_bound)
        self.device_id = int(device_id)
        self.metrics = metrics
        self.layer_error_bounds = layer_error_bounds
        self.layer_axis = int(layer_axis)
        self.flat_layer_prefix = int(flat_layer_prefix)
        if codec not in ("raw", "cuszp", "int8", "zlib", "zstd", "lz4"):
            raise ValueError(f"unsupported offload codec: {codec}")
        self.codec = codec
        if cuszp_mode not in CUSZP_MODES:
            raise ValueError(
                f"unsupported cuSZp encoding mode: {cuszp_mode}; "
                f"choose one of {tuple(CUSZP_MODES)}"
            )
        self.cuszp_mode = cuszp_mode
        self.adaptive_controller = adaptive_controller
        self.async_store = bool(async_store and gpu_to_cpu)
        self.profile_restore_stages = bool(
            profile_restore_stages and not gpu_to_cpu
        )
        self.batch_restore_transfers = bool(
            batch_restore_transfers and not gpu_to_cpu
        )
        self.transfer_type = ("GPU", "CPU") if gpu_to_cpu else ("CPU", "GPU")
        self._finished: deque[TransferResult] = deque()
        self._lock = threading.Lock()
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuszp-offload")
            if self.async_store
            else None
        )
        self._futures: dict[int, Future] = {}
        self._layer_index_cache: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}
        self._compression_output_slab: torch.Tensor | None = None

        self.compressors = {}
        for mode, encoding_mode in CUSZP_MODES.items():
            config = cuszp_wrapper_cpp.CompressionConfig(
                error_bound=self.error_bound,
                use_relative_error=True,
                processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
                encoding_mode=encoding_mode,
                data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT,
            )
            self.compressors[mode] = cuszp_wrapper_cpp.CuSZpWrapper(
                config, self.device_id
            )
        # Preserve the original attribute for downstream experiments and tests.
        self.compressor = self.compressors[self.cuszp_mode]

    def _layer_index(
        self, device: torch.device, layer_indices: tuple[int, ...]
    ) -> torch.Tensor:
        """Reuse immutable GPU indices across blocks and transfer jobs."""
        key = (str(device), layer_indices)
        index = self._layer_index_cache.get(key)
        if index is None:
            index = torch.tensor(layer_indices, dtype=torch.long, device=device)
            self._layer_index_cache[key] = index
        return index

    def _encode_segment(
        self,
        tensor_indices: tuple[int, ...],
        gpu_block_id: int,
        layer_indices: tuple[int, ...] | None,
        requested_error_bound: float,
        cuszp_mode: str | None = None,
        defer_host_copy: bool = False,
    ) -> EncodedSegment:
        selected_cuszp_mode = cuszp_mode or self.cuszp_mode
        if selected_cuszp_mode not in self.compressors:
            raise ValueError(
                f"cuSZp compressor is unavailable for mode {selected_cuszp_mode}"
            )
        components = []
        pieces = []
        raw_pieces = []
        original_bytes = 0
        needs_numeric_encoding = (
            requested_error_bound > 0 and self.codec in ("cuszp", "int8")
        )
        for tensor_idx in tensor_indices:
            source_storage = self.kv_caches.tensors[tensor_idx].tensor[gpu_block_id]
            source = (
                source_storage.view(torch.uint8).view(torch.bfloat16)
                if source_storage.dtype in (torch.uint8, torch.int8)
                else source_storage
            )
            if layer_indices is None:
                selected = source.contiguous()
                component_axis = None
                component_indices = ()
                destination_view_shape = tuple(source.shape)
            else:
                source_view = source
                component_axis = self.layer_axis
                if source.ndim == 1:
                    num_layers = len(self.layer_error_bounds or {})
                    divisor = self.flat_layer_prefix * num_layers
                    if num_layers <= 0 or source.numel() % divisor:
                        raise ValueError(
                            "cannot recover layer view from flattened KV page "
                            f"with numel={source.numel()}, prefix={self.flat_layer_prefix}, "
                            f"layers={num_layers}"
                        )
                    source_view = source.view(
                        self.flat_layer_prefix, num_layers, -1
                    )
                    component_axis = 1
                index = self._layer_index(source.device, layer_indices)
                selected = source_view.index_select(
                    component_axis, index
                ).contiguous()
                component_indices = layer_indices
                destination_view_shape = tuple(source_view.shape)
            if needs_numeric_encoding:
                pieces.append(selected.to(torch.float32).contiguous().view(-1))
            raw_pieces.append(selected.view(torch.uint8).view(-1))
            components.append(
                BundleComponent(
                    tensor_idx=tensor_idx,
                    original_shape=tuple(selected.shape),
                    original_dtype=source.dtype,
                    numel=selected.numel(),
                    layer_axis=component_axis,
                    layer_indices=component_indices,
                    destination_view_shape=destination_view_shape,
                )
            )
            original_bytes += selected.numel() * selected.element_size()

        unpadded_numel = sum(component.numel for component in components)
        payload_gpu = raw_pieces[0] if len(raw_pieces) == 1 else torch.cat(raw_pieces)
        if requested_error_bound <= 0 or self.codec == "raw":
            payload = payload_gpu if defer_host_copy else payload_gpu.to("cpu")
            if not defer_host_copy:
                try:
                    payload = payload.pin_memory()
                except RuntimeError:
                    pass
            return EncodedSegment(
                payload=payload,
                compressed_size=original_bytes,
                actual_error_bound=0.0,
                encoding="raw",
                components=tuple(components),
                unpadded_numel=unpadded_numel,
                compressed_numel=unpadded_numel,
                original_bytes=int(original_bytes),
                requested_error_bound=float(requested_error_bound),
            )

        if self.codec in ("zlib", "zstd", "lz4"):
            raw_cpu = payload_gpu.to("cpu").contiguous()
            raw_bytes = raw_cpu.numpy().tobytes()
            encoded_bytes = (
                zlib.compress(raw_bytes)
                if self.codec == "zlib"
                else native_lossless.compress(self.codec, raw_bytes)
            )
            if len(encoded_bytes) < original_bytes:
                payload = torch.frombuffer(
                    bytearray(encoded_bytes), dtype=torch.uint8
                )
                encoding = self.codec
                payload_size = len(encoded_bytes)
            else:
                payload = raw_cpu
                encoding = "raw"
                payload_size = original_bytes
            try:
                payload = payload.pin_memory()
            except RuntimeError:
                pass
            return EncodedSegment(
                payload=payload,
                compressed_size=payload_size,
                actual_error_bound=0.0,
                encoding=encoding,
                components=tuple(components),
                unpadded_numel=unpadded_numel,
                compressed_numel=unpadded_numel,
                original_bytes=int(original_bytes),
                requested_error_bound=float(requested_error_bound),
            )

        source_unpadded = (
            pieces[0] if len(pieces) == 1 else torch.cat(pieces).contiguous()
        )
        source_fp32 = source_unpadded
        if self.codec == "cuszp" and unpadded_numel < MIN_CUSZP_ELEMENTS:
            source_fp32 = torch.nn.functional.pad(
                source_fp32, (0, MIN_CUSZP_ELEMENTS - unpadded_numel)
            ).contiguous()
        capacity = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(
            source_fp32.numel() * source_fp32.element_size()
        )
        compressed = torch.empty(
            capacity, dtype=torch.uint8, device=source_fp32.device
        )
        encoding = "raw"
        payload_size = original_bytes
        actual_eb = 0.0
        quant_scale = None
        if requested_error_bound > 0:
            if self.codec == "int8":
                scale_tensor = source_unpadded.abs().max() / 127.0
                if float(scale_tensor.item()) == 0.0:
                    scale_tensor = torch.ones_like(scale_tensor)
                quantized = torch.clamp(
                    torch.round(source_unpadded / scale_tensor), -127, 127
                ).to(torch.int8)
                candidate_size = quantized.numel() + 4
                if candidate_size < original_bytes:
                    encoding = "int8"
                    payload_gpu = quantized.view(torch.uint8)
                    payload_size = candidate_size
                    quant_scale = float(scale_tensor.item())
                    actual_eb = quant_scale / 2.0
            elif self.codec == "cuszp":
                success, compressed, size, actual_eb = self.compressors[selected_cuszp_mode].compress(
                    source_fp32, compressed, requested_error_bound
                )
                if not success:
                    raise RuntimeError("cuSZp failed to compress a KV page")
                # Decide on the GPU and transfer raw KV bytes if cuSZp would
                # expand relative to the native vLLM dtype.
                if int(size) < original_bytes:
                    encoding = "cuszp"
                    payload_gpu = compressed[: int(size)]
                    payload_size = int(size)
                else:
                    actual_eb = 0.0

        payload = payload_gpu if defer_host_copy else payload_gpu.to("cpu")
        if not defer_host_copy:
            try:
                payload = payload.pin_memory()
            except RuntimeError:
                pass
        return EncodedSegment(
            payload=payload,
            compressed_size=payload_size,
            actual_error_bound=float(actual_eb),
            encoding=encoding,
            components=tuple(components),
            unpadded_numel=unpadded_numel,
            compressed_numel=source_fp32.numel(),
            original_bytes=int(original_bytes),
            requested_error_bound=float(requested_error_bound),
            quant_scale=quant_scale,
            cuszp_mode=selected_cuszp_mode if encoding == "cuszp" else None,
        )

    def _compress_bundle(
        self,
        tensor_indices: tuple[int, ...],
        gpu_block_id: int,
        layer_error_bounds: dict[int, float] | None = None,
        cuszp_mode: str | None = None,
        defer_host_copy: bool = False,
    ) -> CompressedBundle:
        active_bounds = (
            self.layer_error_bounds
            if layer_error_bounds is None
            else layer_error_bounds
        )
        if not active_bounds:
            segments = (
                self._encode_segment(
                    tensor_indices, gpu_block_id, None, self.error_bound,
                    cuszp_mode, defer_host_copy
                ),
            )
        else:
            first = self.kv_caches.tensors[tensor_indices[0]].tensor[gpu_block_id]
            if first.ndim == 1:
                num_layers = len(active_bounds)
            else:
                if self.layer_axis < 0 or self.layer_axis >= first.ndim:
                    raise ValueError(
                        f"layer_axis={self.layer_axis} is invalid for KV page {tuple(first.shape)}"
                    )
                num_layers = int(first.shape[self.layer_axis])
            grouped: dict[float, list[int]] = {}
            for layer_idx in range(num_layers):
                bound = float(
                    active_bounds.get(layer_idx, self.error_bound)
                )
                grouped.setdefault(bound, []).append(layer_idx)
            if len(grouped) == 1:
                # GREEN adaptive decisions (and any uniform layer profile) cover
                # the complete page. Avoid an identity index_select here and the
                # matching index_copy_ during restore.
                uniform_bound = next(iter(grouped))
                segments = (
                    self._encode_segment(
                        tensor_indices, gpu_block_id, None, uniform_bound,
                        cuszp_mode, defer_host_copy
                    ),
                )
            else:
                segments = tuple(
                    self._encode_segment(
                        tensor_indices,
                        gpu_block_id,
                        tuple(indices),
                        bound,
                        cuszp_mode,
                        defer_host_copy,
                    )
                    for bound, indices in sorted(grouped.items())
                )
        return CompressedBundle(
            segments=segments,
            compressed_size=sum(segment.compressed_size for segment in segments),
            original_bytes=sum(segment.original_bytes for segment in segments),
        )

    def _compress_fixed_bundles_batched(
        self,
        operations: list[tuple[tuple[int, ...], int, int]],
        active_bounds: dict[int, float] | None,
        active_mode: str,
    ) -> list[tuple[int, CompressedBundle]] | None:
        """Compress uniform full KV pages with one fixed-mode native call."""
        effective_bounds = (
            self.layer_error_bounds
            if active_bounds is None
            else active_bounds
        )
        batch_error_bound = self.error_bound
        if effective_bounds:
            layer_ids = sorted(effective_bounds)
            if layer_ids != list(range(len(layer_ids))):
                return None
            uniform_bounds = {
                float(effective_bounds[layer_idx])
                for layer_idx in layer_ids
            }
            if len(uniform_bounds) != 1:
                return None
            batch_error_bound = next(iter(uniform_bounds))
        if (
            self.codec != "cuszp"
            or active_mode != "fixed"
            or batch_error_bound <= 0
            or len(operations) <= 1
        ):
            return None
        compressor = self.compressors["fixed"]
        batch_compress = getattr(
            compressor, "compress_batch_fixed_bf16", None
        )
        if batch_compress is None:
            return None

        prepared = []
        common_numel = None
        common_device = None
        for page_index, (
            tensor_indices, gpu_block_id, cpu_block_id
        ) in enumerate(operations):
            components = []
            pieces = []
            raw_pieces = []
            original_bytes = 0
            for tensor_idx in tensor_indices:
                source_storage = self.kv_caches.tensors[
                    tensor_idx
                ].tensor[gpu_block_id]
                source = (
                    source_storage.view(torch.uint8).view(torch.bfloat16)
                    if source_storage.dtype in (torch.uint8, torch.int8)
                    else source_storage
                )
                selected = source.contiguous()
                if selected.dtype != torch.bfloat16:
                    return None
                pieces.append(selected.view(-1))
                raw_pieces.append(selected.view(torch.uint8).view(-1))
                components.append(
                    BundleComponent(
                        tensor_idx=tensor_idx,
                        original_shape=tuple(selected.shape),
                        original_dtype=source.dtype,
                        numel=selected.numel(),
                        layer_axis=None,
                        layer_indices=(),
                        destination_view_shape=tuple(source.shape),
                    )
                )
                original_bytes += selected.numel() * selected.element_size()

            unpadded_numel = sum(
                component.numel for component in components
            )
            source_unpadded = (
                pieces[0]
                if len(pieces) == 1
                else torch.cat(pieces).contiguous()
            )
            source_numeric = source_unpadded
            if unpadded_numel < MIN_CUSZP_ELEMENTS:
                source_numeric = torch.nn.functional.pad(
                    source_numeric,
                    (0, MIN_CUSZP_ELEMENTS - unpadded_numel),
                ).contiguous()
            if common_numel is None:
                common_numel = source_numeric.numel()
                common_device = source_numeric.device
            elif (
                source_numeric.numel() != common_numel
                or source_numeric.device != common_device
            ):
                return None
            capacity = (
                cuszp_wrapper_cpp.CuSZpWrapper.
                estimate_compressed_buffer_size(
                    source_numeric.numel() * 4
                )
            )
            required_capacity = len(operations) * capacity
            if (
                self._compression_output_slab is None
                or self._compression_output_slab.device != source_numeric.device
                or self._compression_output_slab.numel() < required_capacity
            ):
                self._compression_output_slab = torch.empty(
                    required_capacity,
                    dtype=torch.uint8,
                    device=source_numeric.device,
                )
            compressed = self._compression_output_slab[
                page_index * capacity:(page_index + 1) * capacity
            ]
            raw_payload = (
                raw_pieces[0]
                if len(raw_pieces) == 1
                else torch.cat(raw_pieces).contiguous()
            )
            prepared.append(
                (
                    cpu_block_id,
                    tuple(components),
                    source_numeric,
                    compressed,
                    raw_payload,
                    unpadded_numel,
                    int(original_bytes),
                )
            )

        success, sizes, actual_bounds = batch_compress(
            [item[2] for item in prepared],
            [item[3] for item in prepared],
            [batch_error_bound] * len(prepared),
        )
        if (
            not success
            or len(sizes) != len(prepared)
            or len(actual_bounds) != len(prepared)
        ):
            logger.warning(
                "fixed cuSZp batch compression failed; using per-page path"
            )
            return None

        entries = []
        for item, size, actual_bound in zip(
            prepared, sizes, actual_bounds
        ):
            (
                cpu_block_id,
                components,
                source_numeric,
                compressed,
                raw_payload,
                unpadded_numel,
                original_bytes,
            ) = item
            size = int(size)
            if size <= 0 or size > compressed.numel():
                logger.warning(
                    "fixed cuSZp batch returned invalid size; "
                    "using per-page path"
                )
                return None
            if size < original_bytes:
                payload = compressed[:size]
                encoding = "cuszp"
                payload_size = size
                stored_bound = float(actual_bound)
                stored_mode = "fixed"
            else:
                payload = raw_payload
                encoding = "raw"
                payload_size = original_bytes
                stored_bound = 0.0
                stored_mode = None
            segment = EncodedSegment(
                payload=payload,
                compressed_size=payload_size,
                actual_error_bound=stored_bound,
                encoding=encoding,
                components=components,
                unpadded_numel=unpadded_numel,
                compressed_numel=source_numeric.numel(),
                original_bytes=original_bytes,
                requested_error_bound=batch_error_bound,
                cuszp_mode=stored_mode,
            )
            entries.append(
                (
                    cpu_block_id,
                    CompressedBundle(
                        segments=(segment,),
                        compressed_size=payload_size,
                        original_bytes=original_bytes,
                    ),
                )
            )
        return entries

    def _compress_adaptive_fixed_bundles_batched(
        self,
        operations: list[tuple[tuple[int, ...], int, int]],
        active_bounds: dict[int, float] | None,
        active_mode: str,
    ) -> list[tuple[int, CompressedBundle]] | None:
        """Batch equal-shaped adaptive layer segments across KV pages."""
        if (
            self.codec != "cuszp"
            or active_mode != "fixed"
            or not active_bounds
            or len(operations) <= 1
        ):
            return None
        compressor = self.compressors["fixed"]
        batch_compress = getattr(
            compressor, "compress_batch_fixed_bf16", None
        )
        if batch_compress is None:
            return None

        first_indices, first_gpu_id, _ = operations[0]
        first_storage = self.kv_caches.tensors[
            first_indices[0]
        ].tensor[first_gpu_id]
        first = (
            first_storage.view(torch.uint8).view(torch.bfloat16)
            if first_storage.dtype in (torch.uint8, torch.int8)
            else first_storage
        )
        if first.dtype != torch.bfloat16:
            return None
        if first.ndim == 1:
            num_layers = len(active_bounds)
        else:
            if self.layer_axis < 0 or self.layer_axis >= first.ndim:
                return None
            num_layers = int(first.shape[self.layer_axis])
        grouped: dict[float, list[int]] = {}
        for layer_idx in range(num_layers):
            bound = float(active_bounds.get(layer_idx, self.error_bound))
            grouped.setdefault(bound, []).append(layer_idx)
        if len(grouped) <= 1:
            return None

        page_segments: list[list[EncodedSegment | None]] = [
            [None] * len(grouped) for _ in operations
        ]
        compressed_requests: list[dict] = []
        cpu_ids = []
        for page_index, (
            tensor_indices, gpu_block_id, cpu_block_id
        ) in enumerate(operations):
            cpu_ids.append(cpu_block_id)
            for segment_index, (bound, layer_indices_list) in enumerate(
                sorted(grouped.items())
            ):
                layer_indices = tuple(layer_indices_list)
                components = []
                pieces = []
                raw_pieces = []
                original_bytes = 0
                for tensor_idx in tensor_indices:
                    source_storage = self.kv_caches.tensors[
                        tensor_idx
                    ].tensor[gpu_block_id]
                    source = (
                        source_storage.view(torch.uint8).view(torch.bfloat16)
                        if source_storage.dtype in (torch.uint8, torch.int8)
                        else source_storage
                    )
                    if source.dtype != torch.bfloat16:
                        return None
                    source_view = source
                    component_axis = self.layer_axis
                    if source.ndim == 1:
                        divisor = self.flat_layer_prefix * num_layers
                        if num_layers <= 0 or source.numel() % divisor:
                            return None
                        source_view = source.view(
                            self.flat_layer_prefix, num_layers, -1
                        )
                        component_axis = 1
                    index = self._layer_index(
                        source.device, layer_indices
                    )
                    selected = source_view.index_select(
                        component_axis, index
                    ).contiguous()
                    pieces.append(selected.view(-1))
                    raw_pieces.append(selected.view(torch.uint8).view(-1))
                    components.append(
                        BundleComponent(
                            tensor_idx=tensor_idx,
                            original_shape=tuple(selected.shape),
                            original_dtype=source.dtype,
                            numel=selected.numel(),
                            layer_axis=component_axis,
                            layer_indices=layer_indices,
                            destination_view_shape=tuple(source_view.shape),
                        )
                    )
                    original_bytes += (
                        selected.numel() * selected.element_size()
                    )

                source_unpadded = (
                    pieces[0]
                    if len(pieces) == 1
                    else torch.cat(pieces).contiguous()
                )
                raw_payload = (
                    raw_pieces[0]
                    if len(raw_pieces) == 1
                    else torch.cat(raw_pieces).contiguous()
                )
                unpadded_numel = source_unpadded.numel()
                source_numeric = source_unpadded
                if unpadded_numel < MIN_CUSZP_ELEMENTS:
                    source_numeric = torch.nn.functional.pad(
                        source_numeric,
                        (0, MIN_CUSZP_ELEMENTS - unpadded_numel),
                    ).contiguous()
                metadata = {
                    "page_index": page_index,
                    "segment_index": segment_index,
                    "bound": bound,
                    "components": tuple(components),
                    "source": source_numeric,
                    "raw_payload": raw_payload,
                    "unpadded_numel": unpadded_numel,
                    "original_bytes": int(original_bytes),
                }
                if bound <= 0:
                    page_segments[page_index][segment_index] = EncodedSegment(
                        payload=raw_payload,
                        compressed_size=int(original_bytes),
                        actual_error_bound=0.0,
                        encoding="raw",
                        components=tuple(components),
                        unpadded_numel=unpadded_numel,
                        compressed_numel=unpadded_numel,
                        original_bytes=int(original_bytes),
                        requested_error_bound=bound,
                    )
                else:
                    capacity = (
                        cuszp_wrapper_cpp.CuSZpWrapper.
                        estimate_compressed_buffer_size(
                            source_numeric.numel() * 4
                        )
                    )
                    metadata["capacity"] = capacity
                    compressed_requests.append(metadata)

        if not compressed_requests:
            return None
        total_capacity = sum(
            int(request["capacity"])
            for request in compressed_requests
        )
        first_device = compressed_requests[0]["source"].device
        if (
            self._compression_output_slab is None
            or self._compression_output_slab.device != first_device
            or self._compression_output_slab.numel() < total_capacity
        ):
            self._compression_output_slab = torch.empty(
                total_capacity, dtype=torch.uint8, device=first_device
            )
        offset = 0
        request_groups: dict[tuple[str, int], list[dict]] = {}
        for request in compressed_requests:
            source = request["source"]
            if source.device != first_device:
                return None
            capacity = int(request["capacity"])
            request["compressed"] = self._compression_output_slab[
                offset:offset + capacity
            ]
            offset += capacity
            key = (str(source.device), source.numel())
            request_groups.setdefault(key, []).append(request)

        for requests in request_groups.values():
            success, sizes, actual_bounds = batch_compress(
                [request["source"] for request in requests],
                [request["compressed"] for request in requests],
                [request["bound"] for request in requests],
            )
            if (
                not success
                or len(sizes) != len(requests)
                or len(actual_bounds) != len(requests)
            ):
                logger.warning(
                    "adaptive fixed batch compression failed; "
                    "using per-segment path"
                )
                return None
            for request, size, actual_bound in zip(
                requests, sizes, actual_bounds
            ):
                size = int(size)
                compressed = request["compressed"]
                original_bytes = int(request["original_bytes"])
                if size <= 0 or size > compressed.numel():
                    return None
                if size < original_bytes:
                    payload = compressed[:size]
                    encoding = "cuszp"
                    payload_size = size
                    stored_bound = float(actual_bound)
                    stored_mode = "fixed"
                else:
                    payload = request["raw_payload"]
                    encoding = "raw"
                    payload_size = original_bytes
                    stored_bound = 0.0
                    stored_mode = None
                page_segments[request["page_index"]][
                    request["segment_index"]
                ] = EncodedSegment(
                    payload=payload,
                    compressed_size=payload_size,
                    actual_error_bound=stored_bound,
                    encoding=encoding,
                    components=request["components"],
                    unpadded_numel=request["unpadded_numel"],
                    compressed_numel=request["source"].numel(),
                    original_bytes=original_bytes,
                    requested_error_bound=request["bound"],
                    cuszp_mode=stored_mode,
                )

        entries = []
        for cpu_block_id, segments_with_none in zip(cpu_ids, page_segments):
            if any(segment is None for segment in segments_with_none):
                return None
            segments = tuple(segments_with_none)
            entries.append(
                (
                    cpu_block_id,
                    CompressedBundle(
                        segments=segments,
                        compressed_size=sum(
                            segment.compressed_size for segment in segments
                        ),
                        original_bytes=sum(
                            segment.original_bytes for segment in segments
                        ),
                    ),
                )
            )
        return entries

    def _compress_adaptive_fixed_bundles_indexed(
        self,
        operations: list[tuple[tuple[int, ...], int, int]],
        active_bounds: dict[int, float] | None,
        active_mode: str,
    ) -> list[tuple[int, CompressedBundle]] | None:
        """Compress adaptive layer groups without materializing GPU gathers."""
        if (
            self.codec != "cuszp"
            or active_mode != "fixed"
            or not active_bounds
            or len(operations) <= 1
            or any(len(tensor_indices) != 1 for tensor_indices, _, _ in operations)
        ):
            return None
        compressor = self.compressors["fixed"]
        batch_compress = getattr(
            compressor, "compress_batch_fixed_bf16_indexed", None
        )
        if batch_compress is None:
            return None

        num_layers = len(active_bounds)
        prefix_count = self.flat_layer_prefix
        if num_layers <= 0 or prefix_count <= 0:
            return None
        grouped: dict[float, list[int]] = {}
        for layer_idx in range(num_layers):
            bound = float(active_bounds.get(layer_idx, self.error_bound))
            grouped.setdefault(bound, []).append(layer_idx)
        if len(grouped) <= 1:
            return None

        page_segments: list[list[EncodedSegment | None]] = [
            [None] * len(grouped) for _ in operations
        ]
        compressed_requests = []
        cpu_ids = []
        common_elements_per_layer = None
        common_device = None
        for page_index, (
            tensor_indices, gpu_block_id, cpu_block_id
        ) in enumerate(operations):
            cpu_ids.append(cpu_block_id)
            tensor_idx = tensor_indices[0]
            source_storage = self.kv_caches.tensors[
                tensor_idx
            ].tensor[gpu_block_id]
            source = (
                source_storage.view(torch.uint8).view(torch.bfloat16)
                if source_storage.dtype in (torch.uint8, torch.int8)
                else source_storage
            )
            divisor = prefix_count * num_layers
            if (
                source.dtype != torch.bfloat16
                or source.ndim != 1
                or source.numel() % divisor
            ):
                return None
            elements_per_layer = source.numel() // divisor
            if common_elements_per_layer is None:
                common_elements_per_layer = elements_per_layer
                common_device = source.device
            elif (
                elements_per_layer != common_elements_per_layer
                or source.device != common_device
            ):
                return None
            source_view = source.view(
                prefix_count, num_layers, elements_per_layer
            )
            for segment_index, (bound, selected_list) in enumerate(
                sorted(grouped.items())
            ):
                selected_layers = tuple(selected_list)
                selected_numel = (
                    prefix_count * len(selected_layers) * elements_per_layer
                )
                component = BundleComponent(
                    tensor_idx=tensor_idx,
                    original_shape=(
                        prefix_count,
                        len(selected_layers),
                        elements_per_layer,
                    ),
                    original_dtype=source.dtype,
                    numel=selected_numel,
                    layer_axis=1,
                    layer_indices=selected_layers,
                    destination_view_shape=tuple(source_view.shape),
                )
                if bound <= 0:
                    index = self._layer_index(
                        source.device, selected_layers
                    )
                    raw_selected = source_view.index_select(
                        1, index
                    ).contiguous()
                    raw_payload = raw_selected.view(
                        torch.uint8
                    ).view(-1)
                    original_bytes = (
                        selected_numel * source.element_size()
                    )
                    page_segments[page_index][segment_index] = EncodedSegment(
                        payload=raw_payload,
                        compressed_size=original_bytes,
                        actual_error_bound=0.0,
                        encoding="raw",
                        components=(component,),
                        unpadded_numel=selected_numel,
                        compressed_numel=selected_numel,
                        original_bytes=original_bytes,
                        requested_error_bound=bound,
                    )
                    continue
                if selected_numel < MIN_CUSZP_ELEMENTS:
                    return None
                capacity = (
                    cuszp_wrapper_cpp.CuSZpWrapper.
                    estimate_compressed_buffer_size(selected_numel * 4)
                )
                compressed_requests.append(
                    {
                        "page_index": page_index,
                        "segment_index": segment_index,
                        "bound": bound,
                        "source": source.contiguous(),
                        "selected_layers": selected_layers,
                        "selected_index": self._layer_index(
                            source.device, selected_layers
                        ),
                        "component": component,
                        "selected_numel": selected_numel,
                        "original_bytes": (
                            selected_numel * source.element_size()
                        ),
                        "capacity": capacity,
                    }
                )

        if not compressed_requests:
            return None
        total_capacity = sum(
            request["capacity"] for request in compressed_requests
        )
        if (
            self._compression_output_slab is None
            or self._compression_output_slab.device != common_device
            or self._compression_output_slab.numel() < total_capacity
        ):
            self._compression_output_slab = torch.empty(
                total_capacity, dtype=torch.uint8, device=common_device
            )
        offset = 0
        request_groups: dict[tuple[float, tuple[int, ...]], list[dict]] = {}
        for request in compressed_requests:
            capacity = request["capacity"]
            request["compressed"] = self._compression_output_slab[
                offset:offset + capacity
            ]
            offset += capacity
            key = (request["bound"], request["selected_layers"])
            request_groups.setdefault(key, []).append(request)

        grouped_requests = list(request_groups.values())
        batch_results = []
        grouped_compress = getattr(
            compressor,
            "compress_batch_fixed_bf16_indexed_groups",
            None,
        )
        if grouped_compress is not None and len(grouped_requests) > 1:
            flattened_requests = [
                request
                for requests in grouped_requests
                for request in requests
            ]
            success, sizes, actual_bounds = grouped_compress(
                [request["source"] for request in flattened_requests],
                [
                    request["compressed"]
                    for request in flattened_requests
                ],
                [
                    requests[0]["selected_index"]
                    for requests in grouped_requests
                ],
                [len(requests) for requests in grouped_requests],
                prefix_count,
                num_layers,
                common_elements_per_layer,
                [
                    request["bound"]
                    for request in flattened_requests
                ],
            )
            if (
                success
                and len(sizes) == len(flattened_requests)
                and len(actual_bounds) == len(flattened_requests)
            ):
                batch_results.append(
                    (flattened_requests, sizes, actual_bounds)
                )
            else:
                logger.warning(
                    "grouped indexed compression failed; retrying each "
                    "adaptive layer group"
                )

        if not batch_results:
            for requests in grouped_requests:
                first_request = requests[0]
                success, sizes, actual_bounds = batch_compress(
                    [request["source"] for request in requests],
                    [request["compressed"] for request in requests],
                    first_request["selected_index"],
                    prefix_count,
                    num_layers,
                    common_elements_per_layer,
                    [request["bound"] for request in requests],
                )
                if (
                    not success
                    or len(sizes) != len(requests)
                    or len(actual_bounds) != len(requests)
                ):
                    logger.warning(
                        "indexed adaptive fixed compression failed; "
                        "using gathered batch path"
                    )
                    return None
                batch_results.append((requests, sizes, actual_bounds))

        for requests, sizes, actual_bounds in batch_results:
            for request, size, actual_bound in zip(
                requests, sizes, actual_bounds
            ):
                size = int(size)
                original_bytes = request["original_bytes"]
                compressed = request["compressed"]
                if (
                    size <= 0
                    or size > compressed.numel()
                    or size >= original_bytes
                ):
                    return None
                page_segments[request["page_index"]][
                    request["segment_index"]
                ] = EncodedSegment(
                    payload=compressed[:size],
                    compressed_size=size,
                    actual_error_bound=float(actual_bound),
                    encoding="cuszp",
                    components=(request["component"],),
                    unpadded_numel=request["selected_numel"],
                    compressed_numel=request["selected_numel"],
                    original_bytes=original_bytes,
                    requested_error_bound=request["bound"],
                    cuszp_mode="fixed",
                )

        entries = []
        for cpu_block_id, segments_with_none in zip(cpu_ids, page_segments):
            if any(segment is None for segment in segments_with_none):
                return None
            segments = tuple(segments_with_none)
            entries.append(
                (
                    cpu_block_id,
                    CompressedBundle(
                        segments=segments,
                        compressed_size=sum(
                            segment.compressed_size for segment in segments
                        ),
                        original_bytes=sum(
                            segment.original_bytes for segment in segments
                        ),
                    ),
                )
            )
        return entries

    def _decompress_bundle(
        self, gpu_block_id: int, bundle: CompressedBundle
    ) -> RestoreStageTimings:
        timings = RestoreStageTimings()
        for segment in bundle.segments:
            timings.add(self._decode_segment(gpu_block_id, segment))
        return timings
    def _restore_raw_direct_batched(
        self,
        items: list[tuple[int, EncodedSegment]],
    ) -> RestoreStageTimings:
        """Copy full raw components from pinned CPU memory into final KV pages."""
        timings = RestoreStageTimings()
        h2d_started = self._stage_start()
        for gpu_block_id, segment in items:
            byte_offset = 0
            for component in segment.components:
                destination_page = self.kv_caches.tensors[
                    component.tensor_idx
                ].tensor[gpu_block_id]
                destination = destination_page.view(
                    component.original_dtype
                ).view(component.destination_view_shape)
                if (
                    component.layer_axis is not None
                    or tuple(destination.shape) != component.original_shape
                ):
                    raise ValueError(
                        "raw direct destination is not a complete KV component"
                    )
                component_bytes = (
                    component.numel * destination.element_size()
                )
                end = byte_offset + component_bytes
                source = segment.payload[byte_offset:end].view(
                    component.original_dtype
                ).view(component.original_shape)
                destination.copy_(source, non_blocking=True)
                byte_offset = end
            if byte_offset != segment.compressed_size:
                raise RuntimeError(
                    "raw direct metadata does not cover the payload"
                )
        timings.h2d_seconds = self._stage_elapsed(h2d_started)
        return timings


    def _decompress_bundles_batched(
        self, bundles: list[tuple[int, CompressedBundle]]
    ) -> RestoreStageTimings:
        """Queue pinned payload copies together before decode and scatter."""
        timings = RestoreStageTimings()
        direct: list[tuple[int, EncodedSegment]] = []
        raw_direct: list[tuple[int, EncodedSegment]] = []
        lossless: list[tuple[int, EncodedSegment]] = []
        for gpu_block_id, bundle in bundles:
            for segment in bundle.segments:
                if segment.encoding == "raw" and all(
                    component.layer_axis is None
                    for component in segment.components
                ):
                    raw_direct.append((gpu_block_id, segment))
                    continue
                target = (
                    lossless
                    if segment.encoding in ("zlib", "zstd", "lz4")
                    else direct
                )
                target.append((gpu_block_id, segment))

        if raw_direct:
            timings.add(self._restore_raw_direct_batched(raw_direct))
        prefetched: list[tuple[int, EncodedSegment, torch.Tensor]] = []
        h2d_started = self._stage_start()
        if direct:
            first_tensor = self.kv_caches.tensors[
                direct[0][1].components[0].tensor_idx
            ].tensor
            total_payload_bytes = sum(
                int(segment.payload.numel()) for _, segment in direct
            )
            packed_payloads = torch.empty(
                total_payload_bytes,
                dtype=torch.uint8,
                device=first_tensor.device,
            )
            host_payloads = [segment.payload for _, segment in direct]
            first_payload = host_payloads[0]
            storage_pointer = first_payload.untyped_storage().data_ptr()
            expected_offset = first_payload.storage_offset()
            host_contiguous = True
            for payload in host_payloads:
                if (
                    payload.untyped_storage().data_ptr() != storage_pointer
                    or payload.storage_offset() != expected_offset
                ):
                    host_contiguous = False
                    break
                expected_offset += payload.numel()
            if host_contiguous:
                packed_host = first_payload.as_strided(
                    (total_payload_bytes,),
                    (1,),
                    first_payload.storage_offset(),
                )
                packed_payloads.copy_(packed_host, non_blocking=True)

            payload_offset = 0
            for (gpu_block_id, segment), payload in zip(
                direct, host_payloads
            ):
                end = payload_offset + payload.numel()
                payload_gpu = packed_payloads[payload_offset:end]
                if not host_contiguous:
                    payload_gpu.copy_(payload, non_blocking=True)
                prefetched.append(
                    (
                        gpu_block_id,
                        segment,
                        payload_gpu,
                    )
                )
                payload_offset = end
        timings.h2d_seconds += self._stage_elapsed(h2d_started)

        cuszp_prefetched = [
            item for item in prefetched if item[1].encoding == "cuszp"
        ]
        if cuszp_prefetched:
            timings.add(self._decode_cuszp_prefetched_batched(cuszp_prefetched))
        for gpu_block_id, segment, payload_gpu in prefetched:
            if segment.encoding == "cuszp":
                continue
            timings.add(
                self._decode_segment(
                    gpu_block_id, segment, prefetched_payload_gpu=payload_gpu
                )
            )
        for gpu_block_id, segment in lossless:
            timings.add(self._decode_segment(gpu_block_id, segment))
        return timings

    def _fixed_bf16_indexed_destinations(
        self,
        items: list[tuple[int, EncodedSegment, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], torch.Tensor, int, int, int] | None:
        """Return full page bases and one shared partial-layer index."""
        first_segment = items[0][1]
        if len(first_segment.components) != 1:
            return None
        first_component = first_segment.components[0]
        selected_layers = first_component.layer_indices
        if (
            first_component.layer_axis != 1
            or first_component.original_dtype != torch.bfloat16
            or not selected_layers
            or len(first_component.destination_view_shape) != 3
        ):
            return None
        prefix_count, source_layers, elements_per_layer = (
            first_component.destination_view_shape
        )
        expected_shape = (
            prefix_count, len(selected_layers), elements_per_layer
        )
        outputs = []
        for gpu_block_id, segment, _payload in items:
            if (
                len(segment.components) != 1
                or segment.unpadded_numel != segment.compressed_numel
            ):
                return None
            component = segment.components[0]
            if (
                component.layer_axis != 1
                or component.original_dtype != torch.bfloat16
                or component.layer_indices != selected_layers
                or component.destination_view_shape
                != first_component.destination_view_shape
                or component.original_shape != expected_shape
            ):
                return None
            destination_page = self.kv_caches.tensors[
                component.tensor_idx
            ].tensor[gpu_block_id]
            destination = destination_page.view(
                torch.uint8
            ).view(torch.bfloat16).view(-1)
            if (
                not destination.is_contiguous()
                or destination.numel()
                < prefix_count * source_layers * elements_per_layer
            ):
                return None
            outputs.append(destination)
        index = self._layer_index(outputs[0].device, selected_layers)
        return (
            outputs,
            index,
            prefix_count,
            source_layers,
            elements_per_layer,
        )

    def _fixed_bf16_direct_destinations(
        self,
        items: list[tuple[int, EncodedSegment, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], int, int] | None:
        """Return uniform complete-component destinations for fused decode."""
        component_count = len(items[0][1].components)
        if component_count == 0:
            return None
        elements_per_destination = items[0][1].components[0].numel
        destinations = []
        for gpu_block_id, segment, _payload in items:
            if (
                len(segment.components) != component_count
                or segment.unpadded_numel != segment.compressed_numel
                or segment.compressed_numel
                != component_count * elements_per_destination
            ):
                return None
            for component in segment.components:
                if (
                    component.layer_axis is not None
                    or component.original_dtype != torch.bfloat16
                    or component.numel != elements_per_destination
                ):
                    return None
                destination_page = self.kv_caches.tensors[
                    component.tensor_idx
                ].tensor[gpu_block_id]
                destination = destination_page.view(
                    component.original_dtype
                ).view(component.destination_view_shape)
                if (
                    tuple(destination.shape) != component.original_shape
                    or not destination.is_contiguous()
                ):
                    return None
                destinations.append(destination.view(-1))
        return destinations, component_count, elements_per_destination


    def _decode_cuszp_prefetched_batched(
        self,
        prefetched: list[tuple[int, EncodedSegment, torch.Tensor]],
    ) -> RestoreStageTimings:
        timings = RestoreStageTimings()
        grouped: dict[
            tuple[object, ...],
            list[tuple[int, EncodedSegment, torch.Tensor]],
        ] = {}
        for item in prefetched:
            segment = item[1]
            mode = segment.cuszp_mode or self.cuszp_mode
            if mode == "fixed":
                component_layout = tuple(
                    (
                        component.tensor_idx,
                        component.original_shape,
                        component.original_dtype,
                        component.numel,
                        component.layer_axis,
                        component.layer_indices,
                        component.destination_view_shape,
                    )
                    for component in segment.components
                )
                batch_key = (
                    mode,
                    segment.compressed_numel,
                    segment.unpadded_numel,
                    component_layout,
                )
            else:
                batch_key = (mode,)
            grouped.setdefault(batch_key, []).append(item)

        for batch_key, items in grouped.items():
            mode = str(batch_key[0])
            if mode not in self.compressors:
                raise ValueError(f"unknown cuSZp mode in segment: {mode}")
            compressor = self.compressors[mode]
            if len(items) == 1 or not hasattr(compressor, "decompress_batch"):
                for gpu_block_id, segment, payload_gpu in items:
                    timings.add(
                        self._decode_segment(
                            gpu_block_id,
                            segment,
                            prefetched_payload_gpu=payload_gpu,
                        )
                    )
                continue

            component_dtypes = {
                component.original_dtype
                for _, segment, _ in items
                for component in segment.components
            }
            page_numels = {
                segment.compressed_numel for _, segment, _ in items
            }
            use_fixed_bf16 = (
                mode == "fixed"
                and component_dtypes == {torch.bfloat16}
                and len(page_numels) == 1
                and hasattr(compressor, "decompress_batch_fixed_bf16")
            )
            indexed_destinations = None
            if use_fixed_bf16 and hasattr(
                compressor,
                "decompress_batch_fixed_bf16_indexed_scatter",
            ):
                indexed_destinations = (
                    self._fixed_bf16_indexed_destinations(items)
                )
            if indexed_destinations is not None:
                (
                    outputs,
                    layer_index,
                    prefix_count,
                    source_layers,
                    elements_per_layer,
                ) = indexed_destinations
                decode_started = self._stage_start()
                success = (
                    compressor.
                    decompress_batch_fixed_bf16_indexed_scatter(
                        [item[2] for item in items],
                        [item[1].compressed_size for item in items],
                        outputs,
                        layer_index,
                        prefix_count,
                        source_layers,
                        elements_per_layer,
                        [item[1].actual_error_bound for item in items],
                    )
                )
                timings.gpu_decode_seconds += self._stage_elapsed(
                    decode_started
                )
                if not success:
                    raise RuntimeError(
                        "cuSZp failed to indexed-scatter a KV batch"
                    )
                continue
            direct_destinations = None
            if use_fixed_bf16 and hasattr(
                compressor, "decompress_batch_fixed_bf16_scatter"
            ):
                direct_destinations = (
                    self._fixed_bf16_direct_destinations(items)
                )
            if direct_destinations is not None:
                outputs, destinations_per_page, elements_per_output = (
                    direct_destinations
                )
                decode_started = self._stage_start()
                success = compressor.decompress_batch_fixed_bf16_scatter(
                    [item[2] for item in items],
                    [item[1].compressed_size for item in items],
                    outputs,
                    next(iter(page_numels)),
                    destinations_per_page,
                    elements_per_output,
                    [item[1].actual_error_bound for item in items],
                )
                timings.gpu_decode_seconds += self._stage_elapsed(
                    decode_started
                )
                if not success:
                    raise RuntimeError(
                        "cuSZp failed to direct-scatter a KV batch"
                    )
                continue
            total_numel = sum(item[1].compressed_numel for item in items)
            storage = torch.empty(
                total_numel,
                dtype=torch.bfloat16 if use_fixed_bf16 else torch.float32,
                device=items[0][2].device,
            )
            restored_views = []
            offset = 0
            for _, segment, _ in items:
                end = offset + segment.compressed_numel
                restored_views.append(storage[offset:end])
                offset = end

            decode_started = self._stage_start()
            if use_fixed_bf16:
                success = compressor.decompress_batch_fixed_bf16(
                    [item[2] for item in items],
                    [item[1].compressed_size for item in items],
                    storage,
                    next(iter(page_numels)),
                    [item[1].actual_error_bound for item in items],
                )
            else:
                success = compressor.decompress_batch(
                    [item[2] for item in items],
                    [item[1].compressed_size for item in items],
                    restored_views,
                    [item[1].actual_error_bound for item in items],
                )
            timings.gpu_decode_seconds += self._stage_elapsed(decode_started)
            if not success:
                raise RuntimeError("cuSZp failed to decompress a KV batch")

            scatter_views = restored_views
            if not use_fixed_bf16 and len(component_dtypes) == 1:
                destination_dtype = next(iter(component_dtypes))
                convert_started = self._stage_start()
                scatter_storage = storage.to(destination_dtype)
                timings.scatter_seconds += self._stage_elapsed(convert_started)
                scatter_views = []
                offset = 0
                for _, segment, _ in items:
                    end = offset + segment.compressed_numel
                    scatter_views.append(scatter_storage[offset:end])
                    offset = end



            for (gpu_block_id, segment, payload_gpu), restored in zip(
                items, scatter_views
            ):
                timings.add(
                    self._decode_segment(
                        gpu_block_id,
                        segment,
                        prefetched_payload_gpu=payload_gpu,
                        prefetched_restored_fp32=restored,
                    )
                )
        return timings


    def _stage_start(self) -> float | None:
        if not self.profile_restore_stages:
            return None
        torch.cuda.synchronize(self.device_id)
        return time.perf_counter()

    def _stage_elapsed(self, started: float | None) -> float:
        if started is None:
            return 0.0
        torch.cuda.synchronize(self.device_id)
        return time.perf_counter() - started

    def _decode_segment(
        self,
        gpu_block_id: int,
        segment: EncodedSegment,
        prefetched_payload_gpu: torch.Tensor | None = None,
        prefetched_restored_fp32: torch.Tensor | None = None,
    ) -> RestoreStageTimings:
        timings = RestoreStageTimings()
        first_tensor = self.kv_caches.tensors[segment.components[0].tensor_idx].tensor
        payload_gpu = prefetched_payload_gpu
        if payload_gpu is None and segment.encoding in ("zlib", "zstd", "lz4"):
            cpu_started = time.perf_counter()
            encoded_bytes = segment.payload.numpy().tobytes()
            decoded_bytes = (
                zlib.decompress(encoded_bytes)
                if segment.encoding == "zlib"
                else native_lossless.decompress(
                    segment.encoding, encoded_bytes, segment.original_bytes
                )
            )
            if len(decoded_bytes) != segment.original_bytes:
                raise RuntimeError(
                    f"{segment.encoding} payload size does not match metadata"
                )
            payload_cpu = torch.frombuffer(
                bytearray(decoded_bytes), dtype=torch.uint8
            )
            try:
                payload_cpu = payload_cpu.pin_memory()
            except RuntimeError:
                pass
            timings.cpu_decode_seconds = time.perf_counter() - cpu_started
        elif payload_gpu is None:
            payload_cpu = segment.payload
        if payload_gpu is None:
            h2d_started = self._stage_start()
            payload_gpu = payload_cpu.to(first_tensor.device, non_blocking=False)
            timings.h2d_seconds = self._stage_elapsed(h2d_started)
        if segment.encoding in ("raw", "zlib", "zstd", "lz4"):
            scatter_started = self._stage_start()
            byte_offset = 0
            for component in segment.components:
                destination_page = self.kv_caches.tensors[component.tensor_idx].tensor[
                    gpu_block_id
                ]
                destination = destination_page.view(component.original_dtype).view(
                    component.destination_view_shape
                )
                component_bytes = component.numel * destination.element_size()
                end = byte_offset + component_bytes
                restored = payload_gpu[byte_offset:end].view(component.original_dtype)
                restored = restored.view(component.original_shape)
                if component.layer_axis is None:
                    destination.copy_(restored)
                else:
                    index = self._layer_index(
                        destination.device, component.layer_indices
                    )
                    destination.index_copy_(component.layer_axis, index, restored)
                byte_offset = end
            timings.scatter_seconds = self._stage_elapsed(scatter_started)
            expected_bytes = (
                segment.original_bytes
                if segment.encoding in ("zlib", "zstd", "lz4")
                else segment.compressed_size
            )
            if byte_offset != expected_bytes:
                raise RuntimeError("raw bundle metadata does not cover the payload")
            return timings
        if segment.encoding == "int8":
            if segment.quant_scale is None:
                raise RuntimeError("INT8 segment is missing its quantization scale")
            decode_started = self._stage_start()
            restored_fp32 = (
                payload_gpu.view(torch.int8).to(torch.float32)
                * segment.quant_scale
            )
            timings.gpu_decode_seconds = self._stage_elapsed(decode_started)
            scatter_started = self._stage_start()
            offset = 0
            for component in segment.components:
                destination_page = self.kv_caches.tensors[
                    component.tensor_idx
                ].tensor[gpu_block_id]
                destination = destination_page.view(component.original_dtype).view(
                    component.destination_view_shape
                )
                end = offset + component.numel
                restored = restored_fp32[offset:end].view(
                    component.original_shape
                ).to(component.original_dtype)
                if component.layer_axis is None:
                    destination.copy_(restored)
                else:
                    index = self._layer_index(
                        destination.device, component.layer_indices
                    )
                    destination.index_copy_(component.layer_axis, index, restored)
                offset = end
            timings.scatter_seconds = self._stage_elapsed(scatter_started)
            if offset != segment.unpadded_numel:
                raise RuntimeError("INT8 metadata does not cover the payload")
            return timings
        if segment.encoding != "cuszp":
            raise ValueError(f"unknown KV bundle encoding: {segment.encoding}")

        restored_fp32 = prefetched_restored_fp32
        if restored_fp32 is None:
            restored_fp32 = torch.empty(
                segment.compressed_numel,
                dtype=torch.float32,
                device=first_tensor.device,
            )
            decode_started = self._stage_start()
            decode_mode = segment.cuszp_mode or self.cuszp_mode
            if decode_mode not in self.compressors:
                raise ValueError(f"unknown cuSZp mode in segment: {decode_mode}")
            compressor = self.compressors[decode_mode]
            success = compressor.decompress(
                payload_gpu,
                segment.compressed_size,
                restored_fp32,
                segment.actual_error_bound,
            )
            timings.gpu_decode_seconds = self._stage_elapsed(decode_started)
            if not success:
                raise RuntimeError("cuSZp failed to decompress a KV bundle")
        elif restored_fp32.numel() != segment.compressed_numel:
            raise ValueError("batched cuSZp output size does not match metadata")

        scatter_started = self._stage_start()
        offset = 0
        for component in segment.components:
            destination_page = self.kv_caches.tensors[component.tensor_idx].tensor[
                gpu_block_id
            ]
            destination = destination_page.view(component.original_dtype).view(
                component.destination_view_shape
            )
            if component.layer_axis is None:
                shape_matches = tuple(destination.shape) == component.original_shape
            else:
                expected = list(destination.shape)
                expected[component.layer_axis] = len(component.layer_indices)
                shape_matches = tuple(expected) == component.original_shape
            if not shape_matches:
                raise ValueError("destination KV page shape changed since offload")
            end = offset + component.numel
            restored = restored_fp32[offset:end].view(component.original_shape)
            restored = restored.to(component.original_dtype)
            if component.layer_axis is None:
                destination.copy_(restored)
            else:
                index = self._layer_index(
                    destination.device, component.layer_indices
                )
                destination.index_copy_(component.layer_axis, index, restored)
            offset = end
        timings.scatter_seconds = self._stage_elapsed(scatter_started)
        if offset != segment.unpadded_numel:
            raise RuntimeError("compressed bundle metadata does not cover the payload")
        return timings

    def _transfer_sync(self, job_id: int, spec: TransferSpec) -> bool:
        started = time.perf_counter()
        original_bytes = 0
        transferred_bytes = 0
        encoding_counts = {
            "cuszp": 0,
            "int8": 0,
            "zlib": 0,
            "zstd": 0,
            "lz4": 0,
            "raw": 0,
        }
        bound_counts: dict[str, int] = {}
        mode_counts: dict[str, int] = {}
        adaptive_event = None
        restore_timings = RestoreStageTimings()
        try:
            src_spec, dst_spec = spec
            if self.gpu_to_cpu:
                if not isinstance(src_spec, GPULoadStoreSpec) or not isinstance(
                    dst_spec, CPULoadStoreSpec
                ):
                    raise TypeError("expected GPU-to-CPU transfer specs")
                operations = list(_iter_page_operations(
                    src_spec, dst_spec, self.kv_caches.group_data_refs
                ))
                active_bounds = None
                active_mode = self.cuszp_mode
                if self.adaptive_controller is not None:
                    arriving_bytes = 0
                    for tensor_indices, gpu_id, _cpu_id in operations:
                        for tensor_idx in tensor_indices:
                            storage = self.kv_caches.tensors[tensor_idx].tensor[gpu_id]
                            arriving_bytes += storage.numel() * storage.element_size()
                    decision = self.adaptive_controller.decide(arriving_bytes)
                    active_bounds = decision.layer_error_bounds
                    active_mode = decision.cuszp_mode
                    adaptive_event = {
                        "state": decision.state,
                        "pressure": decision.pressure,
                        "backlog_bytes": decision.backlog_bytes,
                        "arriving_bytes": arriving_bytes,
                        "service_rate_bytes_per_second": (
                            getattr(self.adaptive_controller, "service_rate", None)
                        ),
                        "transfer_deadline_seconds": (
                            getattr(self.adaptive_controller, "deadline", None)
                        ),
                        "state_changed": decision.state_changed,
                        "cuszp_mode": decision.cuszp_mode,
                    }
                store_entries = self._compress_fixed_bundles_batched(
                    operations, active_bounds, active_mode
                )
                if store_entries is None:
                    store_entries = self._compress_adaptive_fixed_bundles_indexed(
                        operations, active_bounds, active_mode
                    )
                if store_entries is None:
                    store_entries = self._compress_adaptive_fixed_bundles_batched(
                        operations, active_bounds, active_mode
                    )
                if store_entries is None:
                    store_entries = []
                    for tensor_indices, gpu_id, cpu_id in operations:
                        bundle = self._compress_bundle(
                            tensor_indices,
                            gpu_id,
                            active_bounds,
                            active_mode,
                            defer_host_copy=True,
                        )
                        store_entries.append((cpu_id, bundle))
                for _cpu_id, bundle in store_entries:
                    original_bytes += bundle.original_bytes
                    transferred_bytes += bundle.compressed_size
                    for segment in bundle.segments:
                        encoding_counts[segment.encoding] += 1
                        key = f"{segment.requested_error_bound:.8g}"
                        bound_counts[key] = bound_counts.get(key, 0) + 1
                        if segment.cuszp_mode:
                            mode_counts[segment.cuszp_mode] = mode_counts.get(segment.cuszp_mode, 0) + 1
                self.store.put_many_packed(store_entries)
            else:
                if not isinstance(src_spec, CPULoadStoreSpec) or not isinstance(
                    dst_spec, GPULoadStoreSpec
                ):
                    raise TypeError("expected CPU-to-GPU transfer specs")
                operations = list(_iter_page_operations(
                    dst_spec, src_spec, self.kv_caches.group_data_refs
                ))
                restore_bundles: list[tuple[int, CompressedBundle]] = []
                for _tensor_indices, gpu_id, cpu_id in operations:
                    bundle = self.store.get(cpu_id)
                    restore_bundles.append((gpu_id, bundle))
                    original_bytes += bundle.original_bytes
                    transferred_bytes += bundle.compressed_size
                    for segment in bundle.segments:
                        encoding_counts[segment.encoding] += 1
                        key = f"{segment.requested_error_bound:.8g}"
                        bound_counts[key] = bound_counts.get(key, 0) + 1
                        if segment.cuszp_mode:
                            mode_counts[segment.cuszp_mode] = mode_counts.get(segment.cuszp_mode, 0) + 1
                if self.batch_restore_transfers:
                    restore_timings.add(
                        self._decompress_bundles_batched(restore_bundles)
                    )
                else:
                    for gpu_id, bundle in restore_bundles:
                        restore_timings.add(
                            self._decompress_bundle(gpu_id, bundle)
                        )

            elapsed = time.perf_counter() - started
            result = TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=transferred_bytes,
                transfer_time=elapsed,
                transfer_type=self.transfer_type,
            )
            self.metrics.record(
                {
                    "job_id": job_id,
                    "direction": "gpu_to_cpu" if self.gpu_to_cpu else "cpu_to_gpu",
                    "original_bytes": original_bytes,
                    "transferred_bytes": transferred_bytes,
                    "compression_ratio": (
                        original_bytes / transferred_bytes if transferred_bytes else 1.0
                    ),
                    "elapsed_seconds": elapsed,
                    "error_bound": self.error_bound,
                    "codec": self.codec,
                    "cuszp_mode": self.cuszp_mode if self.codec == "cuszp" else None,
                    "encoding_counts": encoding_counts,
                    "cuszp_mode_counts": mode_counts,
                    "bound_counts": bound_counts,
                    "adaptive": adaptive_event,
                    "restore_stages": (
                        restore_timings.as_dict()
                        if not self.gpu_to_cpu and self.profile_restore_stages
                        else None
                    ),
                    "effective_h2d_gbps": (
                        transferred_bytes * 8.0 / restore_timings.h2d_seconds / 1e9
                        if restore_timings.h2d_seconds > 0
                        else None
                    ),
                    "success": True,
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.exception("compressed offload job %s failed", job_id)
            result = TransferResult(
                job_id=job_id,
                success=False,
                transfer_size=transferred_bytes,
                transfer_time=elapsed,
                transfer_type=self.transfer_type,
            )
            self.metrics.record(
                {
                    "job_id": job_id,
                    "direction": "gpu_to_cpu" if self.gpu_to_cpu else "cpu_to_gpu",
                    "original_bytes": original_bytes,
                    "transferred_bytes": transferred_bytes,
                    "elapsed_seconds": elapsed,
                    "error_bound": self.error_bound,
                    "error": repr(exc),
                    "success": False,
                }
            )
            with self._lock:
                self._finished.append(result)
            return False

        with self._lock:
            self._finished.append(result)
        return True

    def _run_async_store(
        self, job_id: int, spec: TransferSpec, ready_event: torch.cuda.Event
    ) -> bool:
        torch.cuda.set_device(self.device_id)
        # cuSZp currently corrupts results on a non-default stream despite its
        # stream parameter. Wait for model writes on the host, then let the
        # wrapper use this worker thread's default CUDA stream.
        ready_event.synchronize()
        return self._transfer_sync(job_id, spec)

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        if self._executor is None:
            return self._transfer_sync(job_id, spec)
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(self.device_id))
        future = self._executor.submit(
            self._run_async_store, job_id, spec, ready_event
        )
        with self._lock:
            self._futures[job_id] = future
        return True

    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            completed = [
                job_id for job_id, future in self._futures.items()
                if future.done()
            ]
            for job_id in completed:
                self._futures.pop(job_id).result()
            results = list(self._finished)
            self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> None:
        with self._lock:
            futures = [self._futures.get(job_id) for job_id in job_ids]
        for future in futures:
            if future is not None:
                future.result()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class CompressedCpuGpuOffloadingHandlers:
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        error_bound: float,
        device_id: int,
        metrics_path: str | None,
        layer_error_bounds: dict[int, float] | None = None,
        layer_axis: int = 1,
        flat_layer_prefix: int = 2,
        codec: str = "cuszp",
        adaptive_controller: AdaptiveErrorBoundController | None = None,
        async_store: bool = False,
        profile_restore_stages: bool = False,
        batch_restore_transfers: bool = False,
        cuszp_mode: str = "plain",
    ):
        if block_size_factor != 1:
            raise NotImplementedError(
                "compressed offload currently requires block_size_factor == 1"
            )
        store = CompressedPageStore()
        metrics = MetricsRecorder(metrics_path)
        self.gpu_to_cpu_handler = CompressedOffloadingHandler(
            kv_caches,
            store,
            True,
            error_bound,
            device_id,
            metrics,
            layer_error_bounds,
            layer_axis,
            flat_layer_prefix,
            codec,
            adaptive_controller,
            async_store,
            False,
            False,
            cuszp_mode,
        )
        self.cpu_to_gpu_handler = CompressedOffloadingHandler(
            kv_caches,
            store,
            False,
            error_bound,
            device_id,
            metrics,
            layer_error_bounds,
            layer_axis,
            flat_layer_prefix,
            codec,
            None,
            False,
            profile_restore_stages,
            batch_restore_transfers,
            cuszp_mode,
        )


class CompressedCPUOffloadingSpec(CPUOffloadingSpec):
    """vLLM-loadable spec selected with ``spec_module_path``."""

    def create_handlers(self, kv_caches: CanonicalKVCaches):
        error_bound = float(self.extra_config.get("error_bound", 1e-4))
        device_id = int(self.extra_config.get("device_id", 0))
        metrics_path = self.extra_config.get("metrics_path")
        layer_axis = int(self.extra_config.get("layer_axis", 1))
        flat_layer_prefix = int(self.extra_config.get("flat_layer_prefix", 2))
        codec = self.extra_config.get("codec", "cuszp")
        layer_error_bounds = None
        cuszp_mode = str(self.extra_config.get("cuszp_mode", "plain")).lower()
        sensitivity_path = self.extra_config.get("sensitivity_profile")
        if sensitivity_path:
            with open(sensitivity_path, "r", encoding="utf-8") as fh:
                profile = json.load(fh)
            layers = profile.get("layers", profile)
            policy = self.extra_config.get(
                "sensitivity_policy", "tolerant_only"
            )
            layer_error_bounds = {}
            for layer_idx, entry in layers.items():
                safe_bound = float(entry.get("max_safe_eps", 0.0))
                if policy == "tolerant_only":
                    safe_bound = (
                        safe_bound if entry.get("category") == "deep" else 0.0
                    )
                elif policy != "all_safe":
                    raise ValueError(
                        f"unknown sensitivity_policy: {policy}"
                    )
                layer_error_bounds[int(layer_idx)] = safe_bound
        adaptive_controller = None
        async_store = str(self.extra_config.get("async_store", "false")).lower() in (
            "1", "true", "yes"
        )
        profile_restore_stages = str(
            self.extra_config.get("profile_restore_stages", "false")
        ).lower() in ("1", "true", "yes")
        batch_restore_transfers = str(
            self.extra_config.get("batch_restore_transfers", "false")
        ).lower() in ("1", "true", "yes")
        adaptive_enabled = str(
            self.extra_config.get("adaptive_error_bound", "false")
        ).lower() in ("1", "true", "yes")
        cost_aware_enabled = str(
            self.extra_config.get("cost_aware_restore", "false")
        ).lower() in ("1", "true", "yes")
        if cost_aware_enabled and not adaptive_enabled:
            raise ValueError("cost-aware restore requires adaptive_error_bound")
        if adaptive_enabled:
            if codec != "cuszp":
                raise ValueError("adaptive error bounds currently require codec=cuszp")
            if not layer_error_bounds:
                raise ValueError(
                    "adaptive error bounds require a sensitivity_profile"
                )
            rate_gbps = float(self.extra_config.get("pcie_service_rate_gbps", 0.0))
            deadline_ms = float(self.extra_config.get("transfer_deadline_ms", 0.0))
            candidates = tuple(
                float(x)
                for x in self.extra_config.get(
                    "adaptive_candidates", [1e-5, 1e-4, 1e-3]
                )
            )
            candidate_modes = tuple(
                str(mode).lower()
                for mode in self.extra_config.get(
                    "adaptive_cuszp_modes", [cuszp_mode]
                )
            )
            restore_cost_model = None
            mode_cost_models = {}
            if cost_aware_enabled:
                h2d_gbps = float(
                    self.extra_config.get("restore_h2d_bandwidth_gbps", 0.0)
                )
                min_savings = float(
                    self.extra_config.get("restore_min_savings_fraction", 0.05)
                )
                default_fixed_ms = float(
                    self.extra_config.get("restore_fixed_overhead_ms", 0.0)
                )
                mode_profiles = self.extra_config.get("restore_mode_profiles", {})
                if mode_profiles:
                    for mode, profile in mode_profiles.items():
                        ratios = {
                            float(bound): float(value)
                            for bound, value in profile.get(
                                "compression_ratios", {}
                            ).items()
                        }
                        decompression_gbps = {
                            float(bound): float(value)
                            for bound, value in profile.get(
                                "decompression_gbps", {}
                            ).items()
                        }
                        mode_cost_models[str(mode).lower()] = RestoreCostModel(
                            h2d_bytes_per_second=h2d_gbps * 1e9 / 8.0,
                            compression_ratios=ratios,
                            decompression_bytes_per_second={
                                bound: rate * 1e9 / 8.0
                                for bound, rate in decompression_gbps.items()
                            },
                            fixed_overhead_seconds=float(
                                profile.get("fixed_overhead_ms", default_fixed_ms)
                            ) / 1000.0,
                            min_savings_fraction=min_savings,
                        )
                else:
                    ratios = {
                        float(bound): float(value)
                        for bound, value in self.extra_config.get(
                            "restore_compression_ratios", {}
                        ).items()
                    }
                    decompression_gbps = {
                        float(bound): float(value)
                        for bound, value in self.extra_config.get(
                            "restore_decompression_gbps", {}
                        ).items()
                    }
                    restore_cost_model = RestoreCostModel(
                        h2d_bytes_per_second=h2d_gbps * 1e9 / 8.0,
                        compression_ratios=ratios,
                        decompression_bytes_per_second={
                            bound: rate * 1e9 / 8.0
                            for bound, rate in decompression_gbps.items()
                        },
                        fixed_overhead_seconds=default_fixed_ms / 1000.0,
                        min_savings_fraction=min_savings,
                    )
            adaptive_controller = AdaptiveErrorBoundController(
                layer_safe_bounds=layer_error_bounds,
                service_rate_bytes_per_second=rate_gbps * 1e9 / 8.0,
                transfer_deadline_seconds=deadline_ms / 1000.0,
                candidate_bounds=candidates,
                restore_cost_model=restore_cost_model,
                mode_cost_models=mode_cost_models,
                candidate_modes=candidate_modes,
            )
        return CompressedCpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            error_bound=error_bound,
            device_id=device_id,
            metrics_path=metrics_path,
            layer_error_bounds=layer_error_bounds,
            layer_axis=layer_axis,
            flat_layer_prefix=flat_layer_prefix,
            codec=codec,
            adaptive_controller=adaptive_controller,
            async_store=async_store,
            profile_restore_stages=profile_restore_stages,
            batch_restore_transfers=batch_restore_transfers,
            cuszp_mode=cuszp_mode,
        )
