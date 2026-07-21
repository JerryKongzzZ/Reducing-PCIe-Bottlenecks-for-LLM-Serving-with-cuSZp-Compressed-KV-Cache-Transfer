import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_native_fixed_batch_compression_round_trip():
    from integration.compression_pipeline import vllm_v1_compressed_offload as offload

    cpp = offload.cuszp_wrapper_cpp
    config = cpp.CompressionConfig(
        error_bound=1e-5,
        use_relative_error=True,
        processing_dim=cpp.CuszpDim.DIM_1D,
        encoding_mode=cpp.CuszpMode.MODE_FIXED,
        data_type=cpp.CuszpType.TYPE_FLOAT,
    )
    wrapper = cpp.CuSZpWrapper(config, 0)
    inputs = [
        torch.randn(65536, dtype=torch.float32, device="cuda"),
        torch.randn(65536, dtype=torch.float32, device="cuda") * 3.0,
    ]
    capacity = cpp.CuSZpWrapper.estimate_compressed_buffer_size(
        inputs[0].numel() * inputs[0].element_size()
    )
    outputs = [
        torch.empty(capacity, dtype=torch.uint8, device="cuda")
        for _ in inputs
    ]

    success, sizes, bounds = wrapper.compress_batch_fixed(
        inputs, outputs, [1e-5, 1e-5]
    )
    assert success
    assert len(sizes) == len(inputs)
    assert len(bounds) == len(inputs)

    for source, compressed, size, bound in zip(
        inputs, outputs, sizes, bounds
    ):
        restored = torch.empty_like(source)
        assert wrapper.decompress(compressed, size, restored, bound)
        max_error = float((restored - source).abs().max().item())
        assert max_error <= float(bound) * 1.01 + 1e-7

def test_native_fixed_bf16_indexed_batch_round_trip():
    from integration.compression_pipeline import vllm_v1_compressed_offload as offload

    cpp = offload.cuszp_wrapper_cpp
    config = cpp.CompressionConfig(
        error_bound=1e-5,
        use_relative_error=True,
        processing_dim=cpp.CuszpDim.DIM_1D,
        encoding_mode=cpp.CuszpMode.MODE_FIXED,
        data_type=cpp.CuszpType.TYPE_FLOAT,
    )
    wrapper = cpp.CuSZpWrapper(config, 0)
    prefix_count = 2
    source_layers = 8
    elements_per_layer = 4096
    selected_layers = torch.tensor(
        [1, 3, 6], dtype=torch.long, device="cuda"
    )
    inputs = [
        torch.randn(
            prefix_count * source_layers * elements_per_layer,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(2)
    ]
    logical_numel = (
        prefix_count * selected_layers.numel() * elements_per_layer
    )
    capacity = cpp.CuSZpWrapper.estimate_compressed_buffer_size(
        logical_numel * 4
    )
    outputs = [
        torch.empty(capacity, dtype=torch.uint8, device="cuda")
        for _ in inputs
    ]

    success, sizes, bounds = wrapper.compress_batch_fixed_bf16_indexed(
        inputs,
        outputs,
        selected_layers,
        prefix_count,
        source_layers,
        elements_per_layer,
        [1e-5] * len(inputs),
    )
    assert success

    for source, compressed, size, bound in zip(
        inputs, outputs, sizes, bounds
    ):
        expected = source.view(
            prefix_count, source_layers, elements_per_layer
        ).index_select(1, selected_layers).contiguous().view(-1)
        restored = torch.empty(
            logical_numel, dtype=torch.float32, device="cuda"
        )
        assert wrapper.decompress(
            compressed, size, restored, bound
        )
        max_error = float(
            (restored - expected.to(torch.float32)).abs().max().item()
        )
        assert max_error <= float(bound) * 1.01 + 1e-7




def test_adaptive_controller_uses_deadline_pressure_and_layer_caps():
    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        AdaptiveErrorBoundController,
    )

    controller = AdaptiveErrorBoundController(
        layer_safe_bounds={0: 0.0, 1: 1e-5, 2: 1e-4, 3: 1e-3},
        service_rate_bytes_per_second=1000.0,
        transfer_deadline_seconds=1.0,
        candidate_bounds=(1e-5, 1e-4, 1e-3),
    )

    green = controller.decide(100, now=0.0)
    assert green.state == "green"
    assert set(green.layer_error_bounds.values()) == {0.0}

    yellow = controller.decide(500, now=0.0)
    assert yellow.state == "yellow"
    assert yellow.layer_error_bounds == {
        0: 0.0,
        1: 1e-5,
        2: 1e-5,
        3: 1e-4,
    }

    red = controller.decide(500, now=0.0)
    assert red.state == "red"
    assert red.layer_error_bounds == {
        0: 0.0,
        1: 1e-5,
        2: 1e-4,
        3: 1e-3,
    }

    drained = controller.decide(0, now=1.0)
    assert drained.state == "green"
    assert drained.state_changed


def test_restore_cost_model_finds_bandwidth_break_even():
    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        RestoreCostModel,
    )

    payload_bytes = 10_000_000
    constrained = RestoreCostModel(
        h2d_bytes_per_second=1_000_000_000,
        compression_ratios={1e-4: 2.0},
        decompression_bytes_per_second={1e-4: 10_000_000_000},
    )
    fast_link = RestoreCostModel(
        h2d_bytes_per_second=10_000_000_000,
        compression_ratios={1e-4: 2.0},
        decompression_bytes_per_second={1e-4: 10_000_000_000},
    )

    assert constrained.estimate(payload_bytes, 1e-4).worthwhile
    assert not fast_link.estimate(payload_bytes, 1e-4).worthwhile


def test_adaptive_cost_gate_falls_back_to_raw_when_restore_cannot_win():
    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        AdaptiveErrorBoundController,
        RestoreCostModel,
    )

    cost_model = RestoreCostModel(
        h2d_bytes_per_second=10_000_000_000,
        compression_ratios={1e-4: 2.0},
        decompression_bytes_per_second={1e-4: 1_000_000_000},
    )
    controller = AdaptiveErrorBoundController(
        layer_safe_bounds={0: 1e-4, 1: 1e-4},
        service_rate_bytes_per_second=1000.0,
        transfer_deadline_seconds=1.0,
        candidate_bounds=(1e-4,),
        restore_cost_model=cost_model,
    )

    decision = controller.decide(2000, now=0.0)
    assert decision.state == "red"
    assert set(decision.layer_error_bounds.values()) == {0.0}


def test_joint_gate_selects_fastest_calibrated_cuszp_mode():
    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        AdaptiveErrorBoundController,
        RestoreCostModel,
    )

    models = {
        "plain": RestoreCostModel(
            h2d_bytes_per_second=1_000_000_000,
            compression_ratios={1e-4: 2.0},
            decompression_bytes_per_second={1e-4: 2_000_000_000},
        ),
        "fixed": RestoreCostModel(
            h2d_bytes_per_second=1_000_000_000,
            compression_ratios={1e-4: 2.2},
            decompression_bytes_per_second={1e-4: 10_000_000_000},
        ),
    }
    controller = AdaptiveErrorBoundController(
        layer_safe_bounds={0: 1e-4, 1: 1e-4},
        service_rate_bytes_per_second=1_000_000.0,
        transfer_deadline_seconds=1.0,
        candidate_bounds=(1e-4,),
        mode_cost_models=models,
        candidate_modes=("plain", "fixed"),
    )

    decision = controller.decide(2_000_000, now=0.0)
    assert decision.state == "red"
    assert decision.cuszp_mode == "fixed"
    assert set(decision.layer_error_bounds.values()) == {1e-4}


@pytest.mark.parametrize("cuszp_mode", ["plain", "fixed", "outlier"])
def test_real_cuszp_page_round_trip_through_vllm_specs(cuszp_mode):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    tensor = torch.randn(2, 36864, dtype=torch.float32, device="cuda")
    expected = tensor[0].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-5,
        device_id=0,
        metrics_path=None,
        cuszp_mode=cuszp_mode,
    )

    gpu_source = GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0])
    cpu_block = CPULoadStoreSpec([0])
    assert handlers.gpu_to_cpu_handler.transfer_async(
        1, (gpu_source, cpu_block)
    )
    store_result = handlers.gpu_to_cpu_handler.get_finished()
    assert len(store_result) == 1 and store_result[0].success
    assert store_result[0].transfer_size < page_bytes
    stored_segment = handlers.gpu_to_cpu_handler.store.get(0).segments[0]
    assert stored_segment.cuszp_mode == cuszp_mode
    assert stored_segment.unpadded_numel == 36864
    assert stored_segment.compressed_numel == 36864

    tensor[1].zero_()
    gpu_destination = GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    assert handlers.cpu_to_gpu_handler.transfer_async(
        2, (cpu_block, gpu_destination)
    )
    load_result = handlers.cpu_to_gpu_handler.get_finished()
    assert len(load_result) == 1 and load_result[0].success
    assert torch.allclose(
        tensor[1],
        expected,
        atol=stored_segment.actual_error_bound * 1.01 + 1e-7,
    )


def test_batched_restore_round_trip_for_multiple_pinned_pages():
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    tensor = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    expected = tensor[:2].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=0.0,
        device_id=0,
        metrics_path=None,
        codec="raw",
        profile_restore_stages=True,
        batch_restore_transfers=True,
    )

    gpu_source = GPULoadStoreSpec([0, 1], group_sizes=[2], block_indices=[0])
    cpu_blocks = CPULoadStoreSpec([0, 1])
    assert handlers.gpu_to_cpu_handler.transfer_async(
        1, (gpu_source, cpu_blocks)
    )
    assert handlers.gpu_to_cpu_handler.get_finished()[0].success

    tensor[2:].zero_()
    gpu_destination = GPULoadStoreSpec(
        [2, 3], group_sizes=[2], block_indices=[0]
    )
    assert handlers.cpu_to_gpu_handler.transfer_async(
        2, (cpu_blocks, gpu_destination)
    )
    assert handlers.cpu_to_gpu_handler.get_finished()[0].success
    assert torch.equal(tensor[2:], expected)


@pytest.mark.parametrize("adaptive_uniform", [False, True])
def test_fixed_bf16_batched_fast_path_round_trip(adaptive_uniform):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        AdaptiveDecision,
        CompressedCpuGpuOffloadingHandlers,
    )

    class UniformAdaptiveController:
        def decide(self, _arriving_bytes):
            return AdaptiveDecision(
                state="red",
                pressure=2.0,
                backlog_bytes=1.0,
                layer_error_bounds={idx: 1e-4 for idx in range(28)},
                state_changed=False,
                cuszp_mode="fixed",
            )

    num_pages = 8
    page_numel = 229376
    tensor = torch.randn(
        2 * num_pages,
        page_numel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = tensor[:num_pages].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)
        ],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-5,
        device_id=0,
        metrics_path=None,
        cuszp_mode="fixed",
        batch_restore_transfers=True,
        adaptive_controller=(
            UniformAdaptiveController() if adaptive_uniform else None
        ),
    )

    native_compressor = handlers.gpu_to_cpu_handler.compressors["fixed"]

    class CountingCompressor:
        def __init__(self):
            self.batch_calls = 0
            self.batch_success = False

        def compress_batch_fixed_bf16(self, *args):
            self.batch_calls += 1
            result = native_compressor.compress_batch_fixed_bf16(*args)
            self.batch_success = bool(result[0])
            return result

        def __getattr__(self, name):
            return getattr(native_compressor, name)

    counting_compressor = CountingCompressor()
    handlers.gpu_to_cpu_handler.compressors[
        "fixed"
    ] = counting_compressor

    source_ids = list(range(num_pages))
    cpu_ids = list(range(num_pages))
    destination_ids = list(range(num_pages, 2 * num_pages))
    gpu_source = GPULoadStoreSpec(
        source_ids, group_sizes=[num_pages], block_indices=[0]
    )
    cpu_blocks = CPULoadStoreSpec(cpu_ids)
    assert handlers.gpu_to_cpu_handler.transfer_async(
        1, (gpu_source, cpu_blocks)
    )
    assert handlers.gpu_to_cpu_handler.get_finished()[0].success
    assert counting_compressor.batch_calls == 1
    assert counting_compressor.batch_success

    stored_segments = [
        handlers.gpu_to_cpu_handler.store.get(cpu_id).segments[0]
        for cpu_id in cpu_ids
    ]
    assert all(segment.encoding == "cuszp" for segment in stored_segments)
    assert all(
        segment.requested_error_bound == (1e-4 if adaptive_uniform else 1e-5)
        for segment in stored_segments
    )
    tensor[num_pages:].zero_()
    gpu_destination = GPULoadStoreSpec(
        destination_ids, group_sizes=[num_pages], block_indices=[0]
    )
    assert handlers.cpu_to_gpu_handler.transfer_async(
        2, (cpu_blocks, gpu_destination)
    )
    assert handlers.cpu_to_gpu_handler.get_finished()[0].success

    restored = tensor[num_pages:].to(torch.float32)
    reference = expected.to(torch.float32)
    max_error = float((restored - reference).abs().max().item())
    max_bound = max(segment.actual_error_bound for segment in stored_segments)
    assert torch.isfinite(restored).all()
    assert max_error <= max_bound * 2.0 + 1e-7



@pytest.mark.parametrize("lower_bound", [0.0, 1e-4])
def test_adaptive_fixed_segments_use_cross_page_bf16_batch_path(
    lower_bound,
):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        AdaptiveDecision,
        CompressedCpuGpuOffloadingHandlers,
    )

    num_pages = 4
    num_layers = 24
    page_numel = 2 * num_layers * 2 * 16 * 64
    backing = torch.randn(
        2 * num_pages,
        page_numel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    tensor = backing.view(torch.uint8).view(torch.int8).reshape(
        2 * num_pages, -1
    )
    expected = backing[:num_pages].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)
        ],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    selected_bounds = {
        layer: (lower_bound if layer < num_layers // 2 else 1e-5)
        for layer in range(num_layers)
    }

    class FixedAdaptiveController:
        def decide(self, _arriving_bytes):
            return AdaptiveDecision(
                state="red",
                pressure=2.0,
                backlog_bytes=1.0,
                layer_error_bounds=selected_bounds,
                state_changed=False,
                cuszp_mode="fixed",
            )

    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-5,
        device_id=0,
        metrics_path=None,
        adaptive_controller=FixedAdaptiveController(),
        cuszp_mode="fixed",
        batch_restore_transfers=True,
    )
    native_compressor = handlers.gpu_to_cpu_handler.compressors["fixed"]

    class CountingCompressor:
        def __init__(self):
            self.batch_calls = 0
            self.indexed_calls = 0
            self.indexed_group_calls = 0

        def compress_batch_fixed_bf16(self, *args):
            self.batch_calls += 1
            return native_compressor.compress_batch_fixed_bf16(*args)

        def compress_batch_fixed_bf16_indexed(self, *args):
            self.indexed_calls += 1
            return native_compressor.compress_batch_fixed_bf16_indexed(*args)

        def compress_batch_fixed_bf16_indexed_groups(self, *args):
            self.indexed_group_calls += 1
            return (
                native_compressor.
                compress_batch_fixed_bf16_indexed_groups(*args)
            )

        def __getattr__(self, name):
            return getattr(native_compressor, name)

    counting_compressor = CountingCompressor()
    handlers.gpu_to_cpu_handler.compressors["fixed"] = counting_compressor

    native_decoder = handlers.cpu_to_gpu_handler.compressors["fixed"]

    class CountingDecoder:
        def __init__(self):
            self.indexed_scatter_calls = 0

        def decompress_batch_fixed_bf16_indexed_scatter(self, *args):
            self.indexed_scatter_calls += 1
            return (
                native_decoder.
                decompress_batch_fixed_bf16_indexed_scatter(*args)
            )

        def __getattr__(self, name):
            return getattr(native_decoder, name)

    counting_decoder = CountingDecoder()
    handlers.cpu_to_gpu_handler.compressors["fixed"] = counting_decoder


    source_ids = list(range(num_pages))
    cpu_ids = list(range(num_pages))
    destination_ids = list(range(num_pages, 2 * num_pages))
    gpu_source = GPULoadStoreSpec(
        source_ids, group_sizes=[num_pages], block_indices=[0]
    )
    cpu_blocks = CPULoadStoreSpec(cpu_ids)
    assert handlers.gpu_to_cpu_handler.transfer_async(
        1, (gpu_source, cpu_blocks)
    )
    result = handlers.gpu_to_cpu_handler.get_finished()[0]
    assert result.success
    assert result.transfer_size < num_pages * page_bytes
    assert counting_compressor.batch_calls == 0
    expected_indexed_calls = 1 if lower_bound == 0.0 else 0
    expected_group_calls = 0 if lower_bound == 0.0 else 1
    assert counting_compressor.indexed_calls == expected_indexed_calls
    assert counting_compressor.indexed_group_calls == expected_group_calls

    stored_bundles = [
        handlers.gpu_to_cpu_handler.store.get(cpu_id) for cpu_id in cpu_ids
    ]
    assert all(len(bundle.segments) == 2 for bundle in stored_bundles)
    expected_encodings = (
        {"raw", "cuszp"} if lower_bound == 0.0 else {"cuszp"}
    )
    assert all(
        {segment.encoding for segment in bundle.segments}
        == expected_encodings
        for bundle in stored_bundles
    )

    tensor[num_pages:].zero_()
    gpu_destination = GPULoadStoreSpec(
        destination_ids, group_sizes=[num_pages], block_indices=[0]
    )
    assert handlers.cpu_to_gpu_handler.transfer_async(
        2, (cpu_blocks, gpu_destination)
    )
    assert handlers.cpu_to_gpu_handler.get_finished()[0].success
    assert (
        counting_decoder.indexed_scatter_calls
        == (1 if lower_bound == 0.0 else 2)
    )

    restored = tensor[num_pages:].view(torch.bfloat16).view(
        num_pages, 2, num_layers, 2, 16, 64
    )
    reference = expected.view(num_pages, 2, num_layers, 2, 16, 64)
    if lower_bound == 0.0:
        assert torch.equal(
            restored[:, :, :num_layers // 2],
            reference[:, :, :num_layers // 2],
        )
    compressed_error = (
        restored[:, :, num_layers // 2:].to(torch.float32)
        - reference[:, :, num_layers // 2:].to(torch.float32)
    ).abs().max().item()
    max_bound = max(
        segment.actual_error_bound
        for bundle in stored_bundles
        for segment in bundle.segments
    )
    assert torch.isfinite(restored).all()
    assert compressed_error <= max_bound * 2.0 + 1e-7
    if lower_bound > 0.0:
        lower_error = (
            restored[:, :, :num_layers // 2].to(torch.float32)
            - reference[:, :, :num_layers // 2].to(torch.float32)
        ).abs().max().item()
        lower_actual_bound = max(
            segment.actual_error_bound
            for bundle in stored_bundles
            for segment in bundle.segments
            if segment.requested_error_bound == lower_bound
        )
        assert lower_error <= lower_actual_bound * 2.0 + 1e-7


def test_mixed_layer_policy_preserves_sensitive_layers_and_saves_bytes():
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    # vLLM logs the allocation as [block, K/V, layer, head, token, head_dim]
    # but exposes each canonical page as a flattened vector to the connector.
    page_numel = 2 * 24 * 2 * 16 * 64
    backing = torch.randn(2, page_numel, dtype=torch.bfloat16, device="cuda")
    tensor = backing.view(torch.uint8).view(torch.int8).reshape(2, -1)
    expected = backing[0].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    layer_bounds = {idx: (0.0 if idx < 12 else 1e-2) for idx in range(24)}
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-4,
        device_id=0,
        metrics_path=None,
        layer_error_bounds=layer_bounds,
        layer_axis=1,
    )

    green_bundle = handlers.gpu_to_cpu_handler._compress_bundle(
        (0,), 0, {idx: 0.0 for idx in range(24)}
    )
    assert len(green_bundle.segments) == 1
    assert green_bundle.segments[0].encoding == "raw"
    assert green_bundle.segments[0].components[0].layer_axis is None

    gpu_source = GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0])
    cpu_block = CPULoadStoreSpec([0])
    assert handlers.gpu_to_cpu_handler.transfer_async(1, (gpu_source, cpu_block))
    stored = handlers.gpu_to_cpu_handler.get_finished()[0]
    assert stored.transfer_size < page_bytes

    tensor[1].zero_()
    gpu_destination = GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    assert handlers.cpu_to_gpu_handler.transfer_async(
        2, (cpu_block, gpu_destination)
    )
    handlers.cpu_to_gpu_handler.get_finished()
    restored_view = tensor[1].view(torch.bfloat16).view(2, 24, 2, 16, 64)
    expected_view = expected.view(2, 24, 2, 16, 64)
    assert torch.equal(restored_view[:, :12], expected_view[:, :12])
    assert torch.isfinite(restored_view[:, 12:]).all()
    assert torch.allclose(
        restored_view[:, 12:], expected_view[:, 12:], atol=0.12, rtol=0.12
    )


def test_int8_codec_reinterprets_uint8_backing_and_halves_bytes():
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    backing = torch.randn(2, 65536, dtype=torch.bfloat16, device="cuda")
    tensor = backing.view(torch.uint8).view(torch.int8).reshape(2, -1)
    page_bytes = tensor[0].numel()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-4,
        device_id=0,
        metrics_path=None,
        codec="int8",
    )
    bundle = handlers.gpu_to_cpu_handler._compress_bundle((0,), 0)
    assert bundle.segments[0].encoding == "int8"
    assert bundle.compressed_size == 65536 + 4
    assert bundle.original_bytes == 2 * 65536


def test_zero_bound_segment_skips_numeric_compressor_path():
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    tensor = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-4,
        device_id=0,
        metrics_path=None,
    )

    class CompressorMustNotRun:
        def compress(self, *_args, **_kwargs):
            raise AssertionError("zero-bound raw segment reached cuSZp")

    handler = handlers.gpu_to_cpu_handler
    handler.compressor = CompressorMustNotRun()
    segment = handler._encode_segment((0,), 0, None, 0.0)
    assert segment.encoding == "raw"
    assert segment.compressed_size == page_bytes


@pytest.mark.parametrize("codec", ["zlib", "zstd", "lz4"])
def test_cpu_lossless_codec_restores_page_and_counts_payload(codec):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    tensor = torch.zeros(2, 65536, dtype=torch.bfloat16, device="cuda")
    tensor[0, ::257] = 1.0
    expected = tensor[0].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-4,
        device_id=0,
        metrics_path=None,
        codec=codec,
    )
    bundle = handlers.gpu_to_cpu_handler._compress_bundle((0,), 0)
    assert bundle.segments[0].encoding == codec
    assert bundle.compressed_size < bundle.original_bytes

    tensor[1].fill_(2.0)
    handlers.cpu_to_gpu_handler._decompress_bundle(1, bundle)
    assert torch.equal(tensor[1], expected)


def test_async_store_waits_for_background_compression_and_restores_page():
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        CompressedCpuGpuOffloadingHandlers,
    )

    tensor = torch.randn(2, 65536, dtype=torch.bfloat16, device="cuda")
    expected = tensor[0].clone()
    page_bytes = tensor[0].numel() * tensor.element_size()
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_bytes)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=1e-5,
        device_id=0,
        metrics_path=None,
        async_store=True,
    )
    gpu_source = GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0])
    cpu_block = CPULoadStoreSpec([0])
    assert handlers.gpu_to_cpu_handler.transfer_async(1, (gpu_source, cpu_block))
    handlers.gpu_to_cpu_handler.wait({1})
    result = handlers.gpu_to_cpu_handler.get_finished()
    assert len(result) == 1 and result[0].success

    tensor[1].zero_()
    gpu_destination = GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    assert handlers.cpu_to_gpu_handler.transfer_async(
        2, (cpu_block, gpu_destination)
    )
    handlers.cpu_to_gpu_handler.get_finished()
    restored = tensor[1].to(torch.float32)
    reference = expected.to(torch.float32)
    assert torch.isfinite(restored).all()
    bundle = handlers.gpu_to_cpu_handler.store.get(0)
    assert all(segment.payload.device.type == "cpu" for segment in bundle.segments)
    assert all(segment.payload.is_pinned() for segment in bundle.segments)
    absolute_bound = max(
        segment.actual_error_bound for segment in bundle.segments
    )
    # cuSZp bounds its FP32 reconstruction; storing that reconstruction back
    # into the native BF16 KV page adds at most one BF16 relative rounding unit.
    assert torch.allclose(
        restored,
        reference,
        atol=absolute_bound * 1.01 + 1e-7,
        rtol=torch.finfo(torch.bfloat16).eps,
    )
    handlers.gpu_to_cpu_handler.shutdown()

def test_packed_store_aligns_raw_bf16_after_odd_compressed_payload():
    from integration.compression_pipeline.vllm_v1_compressed_offload import (
        BundleComponent,
        CompressedBundle,
        CompressedPageStore,
        EncodedSegment,
    )

    component = BundleComponent(
        tensor_idx=0,
        original_shape=(2,),
        original_dtype=torch.bfloat16,
        numel=2,
        destination_view_shape=(2,),
    )
    odd_segment = EncodedSegment(
        payload=torch.tensor([1, 2, 3], dtype=torch.uint8, device="cuda"),
        compressed_size=3,
        actual_error_bound=0.0,
        encoding="lz4",
        components=(component,),
        unpadded_numel=2,
        compressed_numel=2,
        original_bytes=4,
        requested_error_bound=0.0,
    )
    raw_values = torch.tensor(
        [1.0, 2.0], dtype=torch.bfloat16, device="cuda"
    )
    raw_segment = EncodedSegment(
        payload=raw_values.view(torch.uint8),
        compressed_size=4,
        actual_error_bound=0.0,
        encoding="raw",
        components=(component,),
        unpadded_numel=2,
        compressed_numel=2,
        original_bytes=4,
        requested_error_bound=0.0,
    )
    first = CompressedBundle(
        segments=(odd_segment,), compressed_size=3, original_bytes=4
    )
    second = CompressedBundle(
        segments=(raw_segment,), compressed_size=4, original_bytes=4
    )

    store = CompressedPageStore()
    store.put_many_packed([(0, first), (1, second)])

    stored_raw = store.get(1).segments[0].payload
    assert stored_raw.storage_offset() % 8 == 0
    assert stored_raw.numel() == raw_segment.compressed_size
    assert torch.equal(stored_raw.view(torch.bfloat16), raw_values.cpu())
