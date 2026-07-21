#include "cuszp_wrapper.h"
#include "cuszp_fastpath.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <tuple>

namespace {

bool check_cuda_error(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << "Error: " << operation << ": "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

}  // namespace

CuSZpWrapper::CuSZpWrapper(const CompressionConfig& config, int device_id)
    : config_(config), device_id_(device_id),
      temp_compression_buffer_(nullptr), temp_buffer_size_(0),
      decompression_cmp_offsets_(nullptr),
      decompression_local_offsets_(nullptr), decompression_flags_(nullptr),
      decompression_workspace_entries_(0),
      decompression_input_ready_(nullptr),
      batch_compressed_ptrs_(nullptr), batch_error_bounds_(nullptr),
      batch_metadata_capacity_(0), batch_output_ptrs_(nullptr),
      batch_output_capacity_(0), batch_compressed_sizes_host_(nullptr),
      batch_error_bounds_host_(nullptr),
      batch_compressed_sizes_capacity_(0),
      batch_compressed_output_ptrs_(nullptr),
      batch_compression_metadata_capacity_(0) {
    cudaSetDevice(device_id);
    temp_buffer_size_ = 0;
}

CuSZpWrapper::~CuSZpWrapper() {
    if (temp_compression_buffer_) {
        cudaFree(temp_compression_buffer_);
        temp_compression_buffer_ = nullptr;
    }
    if (decompression_cmp_offsets_) cudaFree(decompression_cmp_offsets_);
    if (decompression_local_offsets_) cudaFree(decompression_local_offsets_);
    if (decompression_flags_) cudaFree(decompression_flags_);
    for (cudaEvent_t event : decompression_events_) cudaEventDestroy(event);
    for (cudaStream_t stream : decompression_streams_) cudaStreamDestroy(stream);
    if (decompression_input_ready_) {
        cudaEventDestroy(decompression_input_ready_);
    }
    if (batch_compressed_ptrs_) cudaFree(batch_compressed_ptrs_);
    if (batch_error_bounds_) cudaFree(batch_error_bounds_);
    if (batch_output_ptrs_) cudaFree(batch_output_ptrs_);
    if (batch_compressed_sizes_host_) {
        cudaFreeHost(batch_compressed_sizes_host_);
    }
    if (batch_error_bounds_host_) {
        cudaFreeHost(batch_error_bounds_host_);
    }
    if (batch_compressed_output_ptrs_) {
        cudaFree(batch_compressed_output_ptrs_);
    }
}

bool CuSZpWrapper::ensure_decompression_workspace(size_t entries) {
    if (entries <= decompression_workspace_entries_) return true;

    if (decompression_cmp_offsets_) cudaFree(decompression_cmp_offsets_);
    if (decompression_local_offsets_) cudaFree(decompression_local_offsets_);
    if (decompression_flags_) cudaFree(decompression_flags_);
    decompression_cmp_offsets_ = nullptr;
    decompression_local_offsets_ = nullptr;
    decompression_flags_ = nullptr;
    decompression_workspace_entries_ = 0;

    const cudaError_t cmp_status = cudaMalloc(
        reinterpret_cast<void**>(&decompression_cmp_offsets_),
        entries * sizeof(unsigned int));
    const cudaError_t local_status = cudaMalloc(
        reinterpret_cast<void**>(&decompression_local_offsets_),
        entries * sizeof(unsigned int));
    const cudaError_t flag_status = cudaMalloc(
        reinterpret_cast<void**>(&decompression_flags_), entries * sizeof(int));
    if (cmp_status != cudaSuccess || local_status != cudaSuccess ||
        flag_status != cudaSuccess) {
        if (decompression_cmp_offsets_) cudaFree(decompression_cmp_offsets_);
        if (decompression_local_offsets_) cudaFree(decompression_local_offsets_);
        if (decompression_flags_) cudaFree(decompression_flags_);
        decompression_cmp_offsets_ = nullptr;
        decompression_local_offsets_ = nullptr;
        decompression_flags_ = nullptr;
        return false;
    }
    decompression_workspace_entries_ = entries;
    return true;
}
bool CuSZpWrapper::ensure_decompression_streams(size_t count) {
    if (!decompression_input_ready_ &&
        cudaEventCreateWithFlags(
            &decompression_input_ready_, cudaEventDisableTiming) != cudaSuccess) {
        return false;
    }
    while (decompression_streams_.size() < count) {
        cudaStream_t stream = nullptr;
        cudaEvent_t event = nullptr;
        if (cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking) != cudaSuccess) {
            return false;
        }
        if (cudaEventCreateWithFlags(
                &event, cudaEventDisableTiming) != cudaSuccess) {
            cudaStreamDestroy(stream);
            return false;
        }
        decompression_streams_.push_back(stream);
        decompression_events_.push_back(event);
    }
    return true;
}
bool CuSZpWrapper::ensure_batch_metadata(size_t count) {
    if (count <= batch_metadata_capacity_) return true;
    if (batch_compressed_ptrs_) cudaFree(batch_compressed_ptrs_);
    if (batch_error_bounds_) cudaFree(batch_error_bounds_);
    batch_compressed_ptrs_ = nullptr;
    batch_error_bounds_ = nullptr;
    batch_metadata_capacity_ = 0;

    const cudaError_t pointer_status = cudaMalloc(
        reinterpret_cast<void**>(&batch_compressed_ptrs_),
        count * sizeof(unsigned char*));
    const cudaError_t bound_status = cudaMalloc(
        reinterpret_cast<void**>(&batch_error_bounds_),
        count * sizeof(float));
    if (pointer_status != cudaSuccess || bound_status != cudaSuccess) {
        if (batch_compressed_ptrs_) cudaFree(batch_compressed_ptrs_);
        if (batch_error_bounds_) cudaFree(batch_error_bounds_);
        batch_compressed_ptrs_ = nullptr;
        batch_error_bounds_ = nullptr;
        return false;
    }
    batch_metadata_capacity_ = count;
    return true;
}
bool CuSZpWrapper::ensure_batch_compressed_sizes_host(size_t count) {
    if (count <= batch_compressed_sizes_capacity_) return true;
    if (batch_compressed_sizes_host_) {
        cudaFreeHost(batch_compressed_sizes_host_);
    }
    if (batch_error_bounds_host_) {
        cudaFreeHost(batch_error_bounds_host_);
    }
    batch_compressed_sizes_host_ = nullptr;
    batch_error_bounds_host_ = nullptr;
    batch_compressed_sizes_capacity_ = 0;
    const cudaError_t size_status = cudaHostAlloc(
        reinterpret_cast<void**>(&batch_compressed_sizes_host_),
        count * sizeof(unsigned int), cudaHostAllocDefault);
    const cudaError_t bound_status = cudaHostAlloc(
        reinterpret_cast<void**>(&batch_error_bounds_host_),
        count * sizeof(float), cudaHostAllocDefault);
    if (size_status != cudaSuccess || bound_status != cudaSuccess) {
        if (batch_compressed_sizes_host_) {
            cudaFreeHost(batch_compressed_sizes_host_);
        }
        if (batch_error_bounds_host_) {
            cudaFreeHost(batch_error_bounds_host_);
        }
        batch_compressed_sizes_host_ = nullptr;
        batch_error_bounds_host_ = nullptr;
        return false;
    }
    batch_compressed_sizes_capacity_ = count;
    return true;
}
bool CuSZpWrapper::ensure_batch_compression_metadata(size_t count) {
    if (count <= batch_compression_metadata_capacity_) return true;
    if (batch_compressed_output_ptrs_) {
        cudaFree(batch_compressed_output_ptrs_);
    }
    batch_compressed_output_ptrs_ = nullptr;
    batch_compression_metadata_capacity_ = 0;
    const cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&batch_compressed_output_ptrs_),
        count * sizeof(unsigned char*));
    if (status != cudaSuccess) {
        batch_compressed_output_ptrs_ = nullptr;
        return false;
    }
    batch_compression_metadata_capacity_ = count;
    return true;
}
bool CuSZpWrapper::ensure_batch_output_metadata(size_t count) {
    if (count <= batch_output_capacity_) return true;
    if (batch_output_ptrs_) cudaFree(batch_output_ptrs_);
    batch_output_ptrs_ = nullptr;
    batch_output_capacity_ = 0;

    const cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&batch_output_ptrs_),
        count * sizeof(void*));
    if (status != cudaSuccess) {
        batch_output_ptrs_ = nullptr;
        return false;
    }
    batch_output_capacity_ = count;
    return true;
}





bool CuSZpWrapper::compress(
    torch::Tensor input_tensor,
    torch::Tensor& compressed_buffer,
    size_t& compressed_size,
    float& actual_error_bound,
    float eps_override,
    cudaStream_t stream) {
    
    if (!input_tensor.is_cuda()) {
        std::cerr << "Error: Input tensor must be on GPU" << std::endl;
        return false;
    }
    
    size_t nb_elements = input_tensor.numel();
    void* d_input_data = input_tensor.data_ptr();
    
    size_t estimated_size = estimate_compressed_buffer_size(input_tensor.nbytes());
    const size_t compressed_capacity = static_cast<size_t>(
        compressed_buffer.numel() * compressed_buffer.element_size());
    if (compressed_capacity < estimated_size) {
        compressed_buffer = torch::empty(
            {static_cast<long>(estimated_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input_tensor.device())
        );
    }
    
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    
    // Determine per-call eps to use (eps_override > 0 means override)
    float eps_to_use = (eps_override > 0.0f) ? eps_override : config_.error_bound;

    // Dynamically calculate the error bound from one fused min/max reduction.
    // Calling min() and max() separately launched two reductions and caused two
    // host-visible scalar reads. aminmax computes both extrema together; the
    // first item() waits for that one reduction and the second is already ready.
    actual_error_bound = eps_to_use;
    if (config_.use_relative_error) {
        auto minmax = at::aminmax(input_tensor);
        float min_val = std::get<0>(minmax).item<float>();
        float max_val = std::get<1>(minmax).item<float>();
        float range_estimate = max_val - min_val;
        if (range_estimate < 1e-6f) range_estimate = 1e-6f; // Prevent near-zero range
        actual_error_bound = eps_to_use * range_estimate;
    }
    
    uint3 dims = get_tensor_dims(input_tensor);
    
    try {
        size_t compressed_size_temp = 0;
        if (config_.processing_dim == CUSZP_DIM_1D &&
            config_.data_type == CUSZP_TYPE_FLOAT &&
            config_.encoding_mode == CUSZP_MODE_FIXED) {
            constexpr size_t kElementsPerBlock = 32 * 1024;
            const size_t grid_size =
                (nb_elements + kElementsPerBlock - 1) / kElementsPerBlock;
            const size_t entries = grid_size + 1;
            if (!ensure_decompression_workspace(entries)) {
                return false;
            }
            cudaMemsetAsync(
                decompression_cmp_offsets_, 0,
                entries * sizeof(unsigned int), stream);
            cudaMemsetAsync(
                decompression_local_offsets_, 0,
                entries * sizeof(unsigned int), stream);
            cudaMemsetAsync(
                decompression_flags_, 0, entries * sizeof(int), stream);
            if (!launch_cuszp_compress_1d_fixed_f32(
                    static_cast<const float*>(d_input_data), d_compressed,
                    decompression_cmp_offsets_, decompression_local_offsets_,
                    decompression_flags_, nb_elements, actual_error_bound,
                    stream)) {
                return false;
            }
            unsigned int final_offset = 0;
            cudaMemcpyAsync(
                &final_offset, decompression_cmp_offsets_ + grid_size,
                sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                return false;
            }
            const size_t rate_bytes =
                grid_size * kElementsPerBlock / 32;
            compressed_size_temp =
                static_cast<size_t>(final_offset) + rate_bytes;
        } else {
            cuSZp_compress(
                d_input_data,
                d_compressed,
                nb_elements,
                &compressed_size_temp,
                actual_error_bound,
                config_.processing_dim,
                dims,
                config_.data_type,
                config_.encoding_mode,
                stream
            );
        }
        compressed_size = compressed_size_temp;
        
        // Both the fixed fast path and the upstream fallback have already made
        // the compressed size host-visible. Avoid a second device-wide sync;
        // retain explicit launch/runtime error reporting instead.
        return check_cuda_error("cuSZp compression failed");
    } catch (...) {
        std::cerr << "Error: cuSZp compression failed" << std::endl;
        return false;
    }
}

bool CuSZpWrapper::compress_batch_fixed(
    const std::vector<torch::Tensor>& input_tensors,
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<float>& eps_overrides,
    std::vector<size_t>& compressed_sizes,
    std::vector<float>& actual_error_bounds,
    cudaStream_t stream) {
    const size_t batch_size = input_tensors.size();
    if (batch_size == 0 ||
        compressed_buffers.size() != batch_size ||
        (!eps_overrides.empty() && eps_overrides.size() != batch_size) ||
        config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        return false;
    }

    const size_t num_elements = input_tensors[0].numel();
    const size_t required_capacity =
        estimate_compressed_buffer_size(num_elements * sizeof(float));
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> minmax_values;
    if (config_.use_relative_error) {
        minmax_values.reserve(batch_size);
    }
    for (size_t i = 0; i < batch_size; ++i) {
        const torch::Tensor& input = input_tensors[i];
        const torch::Tensor& output = compressed_buffers[i];
        if (!input.is_cuda() || !output.is_cuda() ||
            input.scalar_type() != torch::kFloat32 ||
            output.scalar_type() != torch::kUInt8 ||
            !input.is_contiguous() || !output.is_contiguous() ||
            static_cast<size_t>(input.numel()) != num_elements ||
            static_cast<size_t>(output.numel()) < required_capacity) {
            return false;
        }
        if (config_.use_relative_error) {
            minmax_values.push_back(at::aminmax(input));
        }
    }

    actual_error_bounds.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const float eps = (
            !eps_overrides.empty() && eps_overrides[i] > 0.0f)
                ? eps_overrides[i]
                : config_.error_bound;
        float actual = eps;
        if (config_.use_relative_error) {
            const float min_value =
                std::get<0>(minmax_values[i]).item<float>();
            const float max_value =
                std::get<1>(minmax_values[i]).item<float>();
            actual = eps * std::max(max_value - min_value, 1e-6f);
        }
        actual_error_bounds[i] = actual;
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    const size_t grid_size =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
    const size_t entries_per_page = grid_size + 1;
    const size_t total_entries = entries_per_page * batch_size;
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_batch_compressed_sizes_host(batch_size)) {
        return false;
    }
    cudaMemsetAsync(
        decompression_cmp_offsets_, 0,
        total_entries * sizeof(unsigned int), stream);
    cudaMemsetAsync(
        decompression_local_offsets_, 0,
        total_entries * sizeof(unsigned int), stream);
    cudaMemsetAsync(
        decompression_flags_, 0, total_entries * sizeof(int), stream);

    for (size_t i = 0; i < batch_size; ++i) {
        const size_t workspace_offset = i * entries_per_page;
        if (!launch_cuszp_compress_1d_fixed_f32(
                input_tensors[i].data_ptr<float>(),
                compressed_buffers[i].data_ptr<unsigned char>(),
                decompression_cmp_offsets_ + workspace_offset,
                decompression_local_offsets_ + workspace_offset,
                decompression_flags_ + workspace_offset,
                num_elements, actual_error_bounds[i], stream)) {
            return false;
        }
        cudaMemcpyAsync(
            batch_compressed_sizes_host_ + i,
            decompression_cmp_offsets_ + workspace_offset + grid_size,
            sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }

    const size_t rate_bytes = grid_size * kElementsPerBlock / 32;
    compressed_sizes.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        compressed_sizes[i] =
            static_cast<size_t>(batch_compressed_sizes_host_[i]) + rate_bytes;
    }
    return check_cuda_error("cuSZp fixed batch compression failed");
}

bool CuSZpWrapper::compress_batch_fixed_bf16(
    const std::vector<torch::Tensor>& input_tensors,
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<float>& eps_overrides,
    std::vector<size_t>& compressed_sizes,
    std::vector<float>& actual_error_bounds,
    cudaStream_t stream) {
    const size_t batch_size = input_tensors.size();
    if (batch_size == 0 ||
        compressed_buffers.size() != batch_size ||
        (!eps_overrides.empty() && eps_overrides.size() != batch_size) ||
        config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        return false;
    }

    const size_t num_elements = input_tensors[0].numel();
    const size_t required_capacity =
        estimate_compressed_buffer_size(num_elements * sizeof(float));
    std::vector<void*> host_inputs;
    std::vector<unsigned char*> host_outputs;
    host_inputs.reserve(batch_size);
    host_outputs.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const torch::Tensor& input = input_tensors[i];
        const torch::Tensor& output = compressed_buffers[i];
        if (!input.is_cuda() || !output.is_cuda() ||
            input.scalar_type() != torch::kBFloat16 ||
            output.scalar_type() != torch::kUInt8 ||
            !input.is_contiguous() || !output.is_contiguous() ||
            static_cast<size_t>(input.numel()) != num_elements ||
            static_cast<size_t>(output.numel()) < required_capacity ||
            input.get_device() != device_id_ ||
            output.get_device() != device_id_) {
            return false;
        }
        host_inputs.push_back(input.data_ptr());
        host_outputs.push_back(output.data_ptr<unsigned char>());
    }

    actual_error_bounds.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const float eps = (
            !eps_overrides.empty() && eps_overrides[i] > 0.0f)
                ? eps_overrides[i]
                : config_.error_bound;
        actual_error_bounds[i] = eps;
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    const size_t blocks_per_page =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
    const size_t entries_per_page = blocks_per_page + 1;
    const size_t total_entries = entries_per_page * batch_size;
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_batch_metadata(batch_size) ||
        !ensure_batch_output_metadata(batch_size) ||
        !ensure_batch_compression_metadata(batch_size) ||
        !ensure_batch_compressed_sizes_host(batch_size)) {
        return false;
    }

    cudaMemcpyAsync(
        batch_output_ptrs_, host_inputs.data(),
        batch_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        batch_compressed_output_ptrs_, host_outputs.data(),
        batch_size * sizeof(unsigned char*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        batch_error_bounds_, actual_error_bounds.data(),
        batch_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(
        decompression_cmp_offsets_, 0,
        total_entries * sizeof(unsigned int), stream);
    cudaMemsetAsync(
        decompression_local_offsets_, 0,
        total_entries * sizeof(unsigned int), stream);
    cudaMemsetAsync(
        decompression_flags_, 0, total_entries * sizeof(int), stream);

    if (!launch_cuszp_compress_batch_fixed_bf16(
            reinterpret_cast<const void* const*>(batch_output_ptrs_),
            batch_compressed_output_ptrs_, decompression_cmp_offsets_,
            decompression_local_offsets_, decompression_flags_,
            batch_error_bounds_, num_elements, batch_size,
            config_.use_relative_error, stream)) {
        return false;
    }
    cudaMemcpyAsync(
        batch_error_bounds_host_, batch_error_bounds_,
        batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMemcpyAsync(
            batch_compressed_sizes_host_ + i,
            decompression_cmp_offsets_ +
                i * entries_per_page + blocks_per_page,
            sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }

    const size_t rate_bytes =
        blocks_per_page * kElementsPerBlock / 32;
    compressed_sizes.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        actual_error_bounds[i] = batch_error_bounds_host_[i];
        compressed_sizes[i] =
            static_cast<size_t>(batch_compressed_sizes_host_[i]) +
            rate_bytes;
    }
    return check_cuda_error("cuSZp fixed BF16 batch compression failed");
}



bool CuSZpWrapper::compress_batch_fixed_bf16_indexed(
    const std::vector<torch::Tensor>& input_tensors,
    const std::vector<torch::Tensor>& compressed_buffers,
    torch::Tensor layer_indices,
    size_t prefix_count,
    size_t source_layers,
    size_t elements_per_layer,
    const std::vector<float>& eps_overrides,
    std::vector<size_t>& compressed_sizes,
    std::vector<float>& actual_error_bounds,
    cudaStream_t stream) {
    const size_t batch_size = input_tensors.size();
    const size_t selected_layers =
        static_cast<size_t>(layer_indices.numel());
    if (batch_size == 0 ||
        compressed_buffers.size() != batch_size ||
        (!eps_overrides.empty() && eps_overrides.size() != batch_size) ||
        prefix_count == 0 || source_layers == 0 ||
        elements_per_layer == 0 || selected_layers == 0 ||
        selected_layers > source_layers ||
        !layer_indices.is_cuda() ||
        layer_indices.scalar_type() != torch::kInt64 ||
        !layer_indices.is_contiguous() ||
        layer_indices.get_device() != device_id_ ||
        config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        return false;
    }

    const size_t source_elements =
        prefix_count * source_layers * elements_per_layer;
    const size_t num_elements =
        prefix_count * selected_layers * elements_per_layer;
    const size_t required_capacity =
        estimate_compressed_buffer_size(num_elements * sizeof(float));
    std::vector<void*> host_inputs;
    std::vector<unsigned char*> host_outputs;
    host_inputs.reserve(batch_size);
    host_outputs.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const torch::Tensor& input = input_tensors[i];
        const torch::Tensor& output = compressed_buffers[i];
        if (!input.is_cuda() || !output.is_cuda() ||
            input.scalar_type() != torch::kBFloat16 ||
            output.scalar_type() != torch::kUInt8 ||
            !input.is_contiguous() || !output.is_contiguous() ||
            static_cast<size_t>(input.numel()) < source_elements ||
            static_cast<size_t>(output.numel()) < required_capacity ||
            input.get_device() != device_id_ ||
            output.get_device() != device_id_) {
            return false;
        }
        host_inputs.push_back(input.data_ptr());
        host_outputs.push_back(output.data_ptr<unsigned char>());
    }

    actual_error_bounds.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const float eps = (
            !eps_overrides.empty() && eps_overrides[i] > 0.0f)
                ? eps_overrides[i]
                : config_.error_bound;
        actual_error_bounds[i] = eps;
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    const size_t blocks_per_page =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
    const size_t entries_per_page = blocks_per_page + 1;
    const size_t total_entries = entries_per_page * batch_size;
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_batch_metadata(batch_size) ||
        !ensure_batch_output_metadata(batch_size) ||
        !ensure_batch_compression_metadata(batch_size) ||
        !ensure_batch_compressed_sizes_host(batch_size)) {
        return false;
    }

    cudaMemcpyAsync(
        batch_output_ptrs_, host_inputs.data(),
        batch_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        batch_compressed_output_ptrs_, host_outputs.data(),
        batch_size * sizeof(unsigned char*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        batch_error_bounds_, actual_error_bounds.data(),
        batch_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(
        decompression_cmp_offsets_, 0,
        total_entries * sizeof(unsigned int), stream);
    cudaMemsetAsync(
        decompression_local_offsets_, 0,
        total_entries * sizeof(unsigned int), stream);
    cudaMemsetAsync(
        decompression_flags_, 0, total_entries * sizeof(int), stream);

    if (!launch_cuszp_compress_batch_fixed_bf16_indexed(
            reinterpret_cast<const void* const*>(batch_output_ptrs_),
            batch_compressed_output_ptrs_, decompression_cmp_offsets_,
            decompression_local_offsets_, decompression_flags_,
            batch_error_bounds_,
            layer_indices.data_ptr<int64_t>(),
            prefix_count, source_layers, elements_per_layer,
            selected_layers, batch_size,
            config_.use_relative_error, stream)) {
        return false;
    }
    cudaMemcpyAsync(
        batch_error_bounds_host_, batch_error_bounds_,
        batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMemcpyAsync(
            batch_compressed_sizes_host_ + i,
            decompression_cmp_offsets_ +
                i * entries_per_page + blocks_per_page,
            sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }

    const size_t rate_bytes =
        blocks_per_page * kElementsPerBlock / 32;
    compressed_sizes.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        actual_error_bounds[i] = batch_error_bounds_host_[i];
        compressed_sizes[i] =
            static_cast<size_t>(batch_compressed_sizes_host_[i]) +
            rate_bytes;
    }
    return check_cuda_error(
        "cuSZp indexed fixed BF16 batch compression failed");
}

bool CuSZpWrapper::compress_batch_fixed_bf16_indexed_groups(
    const std::vector<torch::Tensor>& input_tensors,
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<torch::Tensor>& layer_indices,
    const std::vector<size_t>& group_sizes,
    size_t prefix_count,
    size_t source_layers,
    size_t elements_per_layer,
    const std::vector<float>& eps_overrides,
    std::vector<size_t>& compressed_sizes,
    std::vector<float>& actual_error_bounds,
    cudaStream_t stream) {
    const size_t request_count = input_tensors.size();
    if (request_count == 0 || compressed_buffers.size() != request_count ||
        (!eps_overrides.empty() && eps_overrides.size() != request_count) ||
        layer_indices.empty() || layer_indices.size() != group_sizes.size() ||
        prefix_count == 0 || source_layers == 0 ||
        elements_per_layer == 0 ||
        config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        return false;
    }

    size_t grouped_request_count = 0;
    size_t max_group_size = 0;
    size_t max_workspace_entries = 0;
    std::vector<size_t> blocks_per_page_by_group;
    std::vector<size_t> entries_per_page_by_group;
    std::vector<size_t> rate_bytes_by_group;
    blocks_per_page_by_group.reserve(group_sizes.size());
    entries_per_page_by_group.reserve(group_sizes.size());
    rate_bytes_by_group.reserve(group_sizes.size());
    constexpr size_t kElementsPerBlock = 32 * 1024;
    for (size_t group = 0; group < group_sizes.size(); ++group) {
        const torch::Tensor& indices = layer_indices[group];
        const size_t selected_layers =
            static_cast<size_t>(indices.numel());
        if (group_sizes[group] == 0 || selected_layers == 0 ||
            selected_layers > source_layers || !indices.is_cuda() ||
            indices.scalar_type() != torch::kInt64 ||
            !indices.is_contiguous() || indices.get_device() != device_id_) {
            return false;
        }
        grouped_request_count += group_sizes[group];
        max_group_size = std::max(max_group_size, group_sizes[group]);
        const size_t num_elements =
            prefix_count * selected_layers * elements_per_layer;
        const size_t blocks_per_page =
            (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
        const size_t entries_per_page = blocks_per_page + 1;
        max_workspace_entries = std::max(
            max_workspace_entries, entries_per_page * group_sizes[group]);
        blocks_per_page_by_group.push_back(blocks_per_page);
        entries_per_page_by_group.push_back(entries_per_page);
        rate_bytes_by_group.push_back(
            blocks_per_page * kElementsPerBlock / 32);
    }
    if (grouped_request_count != request_count) {
        return false;
    }

    const size_t source_elements =
        prefix_count * source_layers * elements_per_layer;
    std::vector<void*> host_inputs;
    std::vector<unsigned char*> host_outputs;
    host_inputs.reserve(request_count);
    host_outputs.reserve(request_count);
    size_t request_offset = 0;
    for (size_t group = 0; group < group_sizes.size(); ++group) {
        const size_t selected_layers =
            static_cast<size_t>(layer_indices[group].numel());
        const size_t selected_elements =
            prefix_count * selected_layers * elements_per_layer;
        const size_t required_capacity =
            estimate_compressed_buffer_size(
                selected_elements * sizeof(float));
        for (size_t i = 0; i < group_sizes[group]; ++i) {
            const torch::Tensor& input = input_tensors[request_offset + i];
            const torch::Tensor& output =
                compressed_buffers[request_offset + i];
            if (!input.is_cuda() || !output.is_cuda() ||
                input.scalar_type() != torch::kBFloat16 ||
                output.scalar_type() != torch::kUInt8 ||
                !input.is_contiguous() || !output.is_contiguous() ||
                static_cast<size_t>(input.numel()) < source_elements ||
                static_cast<size_t>(output.numel()) < required_capacity ||
                input.get_device() != device_id_ ||
                output.get_device() != device_id_) {
                return false;
            }
            host_inputs.push_back(input.data_ptr());
            host_outputs.push_back(output.data_ptr<unsigned char>());
        }
        request_offset += group_sizes[group];
    }

    actual_error_bounds.resize(request_count);
    for (size_t i = 0; i < request_count; ++i) {
        actual_error_bounds[i] =
            (!eps_overrides.empty() && eps_overrides[i] > 0.0f)
                ? eps_overrides[i]
                : config_.error_bound;
    }
    if (!ensure_decompression_workspace(max_workspace_entries) ||
        !ensure_batch_metadata(max_group_size) ||
        !ensure_batch_output_metadata(max_group_size) ||
        !ensure_batch_compression_metadata(max_group_size) ||
        !ensure_batch_compressed_sizes_host(request_count)) {
        return false;
    }

    request_offset = 0;
    for (size_t group = 0; group < group_sizes.size(); ++group) {
        const size_t group_size = group_sizes[group];
        const size_t selected_layers =
            static_cast<size_t>(layer_indices[group].numel());
        const size_t entries_per_page = entries_per_page_by_group[group];
        const size_t total_entries = entries_per_page * group_size;
        cudaMemcpyAsync(
            batch_output_ptrs_, host_inputs.data() + request_offset,
            group_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(
            batch_compressed_output_ptrs_,
            host_outputs.data() + request_offset,
            group_size * sizeof(unsigned char*),
            cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(
            batch_error_bounds_,
            actual_error_bounds.data() + request_offset,
            group_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(
            decompression_cmp_offsets_, 0,
            total_entries * sizeof(unsigned int), stream);
        cudaMemsetAsync(
            decompression_local_offsets_, 0,
            total_entries * sizeof(unsigned int), stream);
        cudaMemsetAsync(
            decompression_flags_, 0, total_entries * sizeof(int), stream);

        if (!launch_cuszp_compress_batch_fixed_bf16_indexed(
                reinterpret_cast<const void* const*>(batch_output_ptrs_),
                batch_compressed_output_ptrs_, decompression_cmp_offsets_,
                decompression_local_offsets_, decompression_flags_,
                batch_error_bounds_,
                layer_indices[group].data_ptr<int64_t>(),
                prefix_count, source_layers, elements_per_layer,
                selected_layers, group_size,
                config_.use_relative_error, stream)) {
            return false;
        }
        cudaMemcpyAsync(
            batch_error_bounds_host_ + request_offset,
            batch_error_bounds_, group_size * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        const size_t blocks_per_page =
            blocks_per_page_by_group[group];
        for (size_t i = 0; i < group_size; ++i) {
            cudaMemcpyAsync(
                batch_compressed_sizes_host_ + request_offset + i,
                decompression_cmp_offsets_ +
                    i * entries_per_page + blocks_per_page,
                sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        }
        request_offset += group_size;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }

    compressed_sizes.resize(request_count);
    request_offset = 0;
    for (size_t group = 0; group < group_sizes.size(); ++group) {
        for (size_t i = 0; i < group_sizes[group]; ++i) {
            const size_t request = request_offset + i;
            actual_error_bounds[request] =
                batch_error_bounds_host_[request];
            compressed_sizes[request] =
                static_cast<size_t>(batch_compressed_sizes_host_[request]) +
                rate_bytes_by_group[group];
        }
        request_offset += group_sizes[group];
    }
    return check_cuda_error(
        "cuSZp grouped indexed fixed BF16 batch compression failed");
}

bool CuSZpWrapper::decompress(
    torch::Tensor compressed_buffer,
    size_t compressed_size,
    torch::Tensor output_tensor,
    float actual_error_bound,
    cudaStream_t stream) {
    
    if (!compressed_buffer.is_cuda() || !output_tensor.is_cuda()) {
        std::cerr << "Error: Both compressed buffer and output tensor must be on GPU" << std::endl;
        return false;
    }
    
    size_t nb_elements = output_tensor.numel();
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    void* d_output_data = output_tensor.data_ptr();
    
    uint3 dims = get_tensor_dims(output_tensor);
    
    try {
        // Upstream cuSZp allocates and frees three metadata arrays for every
        // page. vLLM restores hundreds of equally shaped pages per request, so
        // allocator barriers dominate the decoder. Reuse one grow-only
        // workspace and launch the original cuSZp kernel on PyTorch's stream.
        if (config_.processing_dim == CUSZP_DIM_1D &&
            config_.data_type == CUSZP_TYPE_FLOAT) {
            constexpr size_t kElementsPerBlock = 32 * 1024;
            const size_t grid_size =
                (nb_elements + kElementsPerBlock - 1) / kElementsPerBlock;
            const size_t entries = grid_size + 1;
            if (!ensure_decompression_workspace(entries)) {
                std::cerr
                    << "Error: unable to allocate cuSZp decompression workspace"
                    << std::endl;
                return false;
            }
            cudaMemsetAsync(decompression_cmp_offsets_, 0,
                            entries * sizeof(unsigned int), stream);
            cudaMemsetAsync(decompression_local_offsets_, 0,
                            entries * sizeof(unsigned int), stream);
            cudaMemsetAsync(decompression_flags_, 0,
                            entries * sizeof(int), stream);
            if (!launch_cuszp_decompress_1d_f32(
                    static_cast<float*>(d_output_data), d_compressed,
                    decompression_cmp_offsets_, decompression_local_offsets_,
                    decompression_flags_, nb_elements, actual_error_bound,
                    config_.encoding_mode, stream)) {
                return false;
            }
            return check_cuda_error("cuSZp fast decompression failed");
        }


        cuSZp_decompress(
            d_output_data,
            d_compressed,
            nb_elements,
            compressed_size,
            actual_error_bound,
            config_.processing_dim,
            dims,
            config_.data_type,
            config_.encoding_mode,
            stream
        );
        
        // cuSZp frees the temporary metadata used by the decompression kernel
        // before returning, so the output is complete at this point. Avoid a
        // redundant whole-stream/device barrier while still surfacing errors.
        return check_cuda_error("cuSZp decompression failed");
    } catch (...) {
        std::cerr << "Error: cuSZp decompression failed" << std::endl;
        return false;
    }
}
bool CuSZpWrapper::decompress_batch(
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<size_t>& compressed_sizes,
    const std::vector<torch::Tensor>& output_tensors,
    const std::vector<float>& actual_error_bounds,
    cudaStream_t parent_stream) {
    const size_t batch_size = compressed_buffers.size();
    if (batch_size == 0) return true;
    if (compressed_sizes.size() != batch_size ||
        output_tensors.size() != batch_size ||
        actual_error_bounds.size() != batch_size) {
        std::cerr << "Error: cuSZp batch metadata lengths differ" << std::endl;
        return false;
    }
    if (config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT) {
        std::cerr << "Error: cuSZp batch fast path requires 1D f32" << std::endl;
        return false;
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    constexpr size_t kMaxStreams = 8;
    std::vector<size_t> workspace_entries(batch_size);
    size_t total_entries = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        if (!compressed_buffers[i].is_cuda() || !output_tensors[i].is_cuda() ||
            output_tensors[i].scalar_type() != torch::kFloat32 ||
            compressed_sizes[i] == 0) {
            std::cerr << "Error: invalid tensor in cuSZp batch" << std::endl;
            return false;
        }
        const size_t num_elements = output_tensors[i].numel();
        const size_t grid_size =
            (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
        workspace_entries[i] = grid_size + 1;
        total_entries += workspace_entries[i];
    }

    const size_t stream_count = std::min(kMaxStreams, batch_size);
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_decompression_streams(stream_count)) {
        std::cerr << "Error: unable to prepare cuSZp batch resources"
                  << std::endl;
        return false;
    }
    if (cudaEventRecord(decompression_input_ready_, parent_stream) != cudaSuccess) {
        return false;
    }
    for (size_t i = 0; i < stream_count; ++i) {
        if (cudaStreamWaitEvent(
                decompression_streams_[i], decompression_input_ready_, 0) !=
            cudaSuccess) {
            return false;
        }
    }

    size_t workspace_offset = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        cudaStream_t stream = decompression_streams_[i % stream_count];
        const size_t entries = workspace_entries[i];
        unsigned int* cmp_offsets =
            decompression_cmp_offsets_ + workspace_offset;
        unsigned int* local_offsets =
            decompression_local_offsets_ + workspace_offset;
        int* flags = decompression_flags_ + workspace_offset;
        cudaMemsetAsync(cmp_offsets, 0, entries * sizeof(unsigned int), stream);
        cudaMemsetAsync(
            local_offsets, 0, entries * sizeof(unsigned int), stream);
        cudaMemsetAsync(flags, 0, entries * sizeof(int), stream);
        if (!launch_cuszp_decompress_1d_f32(
                output_tensors[i].data_ptr<float>(),
                compressed_buffers[i].data_ptr<unsigned char>(),
                cmp_offsets, local_offsets, flags, output_tensors[i].numel(),
                actual_error_bounds[i], config_.encoding_mode, stream)) {
            return false;
        }
        workspace_offset += entries;
    }

    for (size_t i = 0; i < stream_count; ++i) {
        if (cudaEventRecord(
                decompression_events_[i], decompression_streams_[i]) !=
                cudaSuccess ||
            cudaStreamWaitEvent(
                parent_stream, decompression_events_[i], 0) != cudaSuccess) {
            return false;
        }
    }
    return check_cuda_error("cuSZp batched decompression failed");
}
bool CuSZpWrapper::decompress_batch_fixed_bf16(
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<size_t>& compressed_sizes,
    torch::Tensor output_tensor,
    size_t num_elements_per_page,
    const std::vector<float>& actual_error_bounds,
    cudaStream_t parent_stream) {
    const size_t batch_size = compressed_buffers.size();
    if (batch_size == 0) return true;
    if (compressed_sizes.size() != batch_size ||
        actual_error_bounds.size() != batch_size ||
        num_elements_per_page == 0 ||
        !output_tensor.is_cuda() ||
        output_tensor.scalar_type() != torch::kBFloat16 ||
        static_cast<size_t>(output_tensor.numel()) <
            batch_size * num_elements_per_page) {
        std::cerr << "Error: invalid cuSZp fixed BF16 batch metadata"
                  << std::endl;
        return false;
    }
    if (config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        std::cerr << "Error: BF16 batch kernel requires fixed 1D f32 cuSZp"
                  << std::endl;
        return false;
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    const size_t blocks_per_page =
        (num_elements_per_page + kElementsPerBlock - 1) / kElementsPerBlock;
    const size_t entries_per_page = blocks_per_page + 1;
    const size_t total_entries = entries_per_page * batch_size;
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_batch_metadata(batch_size)) {
        std::cerr << "Error: unable to prepare cuSZp BF16 batch resources"
                  << std::endl;
        return false;
    }

    std::vector<const unsigned char*> host_pointers;
    host_pointers.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (!compressed_buffers[i].is_cuda() || compressed_sizes[i] == 0) {
            std::cerr << "Error: invalid compressed page in BF16 batch"
                      << std::endl;
            return false;
        }
        host_pointers.push_back(
            compressed_buffers[i].data_ptr<unsigned char>());
    }
    cudaMemcpyAsync(
        batch_compressed_ptrs_, host_pointers.data(),
        batch_size * sizeof(unsigned char*), cudaMemcpyHostToDevice,
        parent_stream);
    cudaMemcpyAsync(
        batch_error_bounds_, actual_error_bounds.data(),
        batch_size * sizeof(float), cudaMemcpyHostToDevice, parent_stream);
    cudaMemsetAsync(
        decompression_cmp_offsets_, 0,
        total_entries * sizeof(unsigned int), parent_stream);
    cudaMemsetAsync(
        decompression_local_offsets_, 0,
        total_entries * sizeof(unsigned int), parent_stream);
    cudaMemsetAsync(
        decompression_flags_, 0,
        total_entries * sizeof(int), parent_stream);

    if (!launch_cuszp_decompress_batch_fixed_bf16(
            output_tensor.data_ptr(), batch_compressed_ptrs_,
            decompression_cmp_offsets_, decompression_local_offsets_,
            decompression_flags_, batch_error_bounds_,
            num_elements_per_page, batch_size, parent_stream)) {
        return false;
    }
    return check_cuda_error("cuSZp fixed BF16 batch decompression failed");
}
bool CuSZpWrapper::decompress_batch_fixed_bf16_scatter(
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<size_t>& compressed_sizes,
    const std::vector<torch::Tensor>& output_tensors,
    size_t num_elements_per_page,
    size_t destinations_per_page,
    size_t elements_per_destination,
    const std::vector<float>& actual_error_bounds,
    cudaStream_t parent_stream) {
    const size_t batch_size = compressed_buffers.size();
    if (batch_size == 0) return true;
    const size_t output_count = batch_size * destinations_per_page;
    if (compressed_sizes.size() != batch_size ||
        actual_error_bounds.size() != batch_size ||
        destinations_per_page == 0 ||
        elements_per_destination == 0 ||
        num_elements_per_page !=
            destinations_per_page * elements_per_destination ||
        output_tensors.size() != output_count) {
        std::cerr << "Error: invalid cuSZp direct-scatter metadata"
                  << std::endl;
        return false;
    }
    if (config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        std::cerr << "Error: direct scatter requires fixed 1D f32 cuSZp"
                  << std::endl;
        return false;
    }

    std::vector<const unsigned char*> host_compressed;
    host_compressed.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (!compressed_buffers[i].is_cuda() || compressed_sizes[i] == 0) {
            std::cerr << "Error: invalid compressed direct-scatter page"
                      << std::endl;
            return false;
        }
        host_compressed.push_back(
            compressed_buffers[i].data_ptr<unsigned char>());
    }

    std::vector<void*> host_outputs;
    host_outputs.reserve(output_count);
    for (const torch::Tensor& output : output_tensors) {
        if (!output.is_cuda() ||
            output.scalar_type() != torch::kBFloat16 ||
            !output.is_contiguous() ||
            static_cast<size_t>(output.numel()) <
                elements_per_destination) {
            std::cerr << "Error: invalid BF16 direct-scatter destination"
                      << std::endl;
            return false;
        }
        host_outputs.push_back(output.data_ptr());
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    const size_t blocks_per_page =
        (num_elements_per_page + kElementsPerBlock - 1) / kElementsPerBlock;
    const size_t entries_per_page = blocks_per_page + 1;
    const size_t total_entries = entries_per_page * batch_size;
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_batch_metadata(batch_size) ||
        !ensure_batch_output_metadata(output_count)) {
        std::cerr << "Error: unable to prepare direct-scatter resources"
                  << std::endl;
        return false;
    }

    cudaMemcpyAsync(
        batch_compressed_ptrs_, host_compressed.data(),
        batch_size * sizeof(unsigned char*), cudaMemcpyHostToDevice,
        parent_stream);
    cudaMemcpyAsync(
        batch_output_ptrs_, host_outputs.data(),
        output_count * sizeof(void*), cudaMemcpyHostToDevice, parent_stream);
    cudaMemcpyAsync(
        batch_error_bounds_, actual_error_bounds.data(),
        batch_size * sizeof(float), cudaMemcpyHostToDevice, parent_stream);
    cudaMemsetAsync(
        decompression_cmp_offsets_, 0,
        total_entries * sizeof(unsigned int), parent_stream);
    cudaMemsetAsync(
        decompression_local_offsets_, 0,
        total_entries * sizeof(unsigned int), parent_stream);
    cudaMemsetAsync(
        decompression_flags_, 0,
        total_entries * sizeof(int), parent_stream);

    if (!launch_cuszp_decompress_batch_fixed_bf16_scatter(
            batch_output_ptrs_, destinations_per_page,
            elements_per_destination, batch_compressed_ptrs_,
            decompression_cmp_offsets_, decompression_local_offsets_,
            decompression_flags_, batch_error_bounds_,
            num_elements_per_page, batch_size, parent_stream)) {
        return false;
    }
    return check_cuda_error(
        "cuSZp fixed BF16 direct-scatter decompression failed");
}

bool CuSZpWrapper::decompress_batch_fixed_bf16_indexed_scatter(
    const std::vector<torch::Tensor>& compressed_buffers,
    const std::vector<size_t>& compressed_sizes,
    const std::vector<torch::Tensor>& output_tensors,
    torch::Tensor layer_indices,
    size_t prefix_count,
    size_t source_layers,
    size_t elements_per_layer,
    const std::vector<float>& actual_error_bounds,
    cudaStream_t parent_stream) {
    const size_t batch_size = compressed_buffers.size();
    const size_t selected_layers =
        static_cast<size_t>(layer_indices.numel());
    if (batch_size == 0) return true;
    const size_t num_elements =
        prefix_count * selected_layers * elements_per_layer;
    const size_t source_elements =
        prefix_count * source_layers * elements_per_layer;
    if (compressed_sizes.size() != batch_size ||
        actual_error_bounds.size() != batch_size ||
        output_tensors.size() != batch_size ||
        prefix_count == 0 || source_layers == 0 ||
        elements_per_layer == 0 || selected_layers == 0 ||
        selected_layers > source_layers ||
        !layer_indices.is_cuda() ||
        layer_indices.scalar_type() != torch::kInt64 ||
        !layer_indices.is_contiguous() ||
        layer_indices.get_device() != device_id_ ||
        config_.processing_dim != CUSZP_DIM_1D ||
        config_.data_type != CUSZP_TYPE_FLOAT ||
        config_.encoding_mode != CUSZP_MODE_FIXED) {
        return false;
    }

    std::vector<const unsigned char*> host_compressed;
    std::vector<void*> host_outputs;
    host_compressed.reserve(batch_size);
    host_outputs.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const torch::Tensor& compressed = compressed_buffers[i];
        const torch::Tensor& output = output_tensors[i];
        if (!compressed.is_cuda() || compressed_sizes[i] == 0 ||
            !output.is_cuda() ||
            output.scalar_type() != torch::kBFloat16 ||
            !output.is_contiguous() ||
            static_cast<size_t>(output.numel()) < source_elements ||
            compressed.get_device() != device_id_ ||
            output.get_device() != device_id_) {
            return false;
        }
        host_compressed.push_back(
            compressed.data_ptr<unsigned char>());
        host_outputs.push_back(output.data_ptr());
    }

    constexpr size_t kElementsPerBlock = 32 * 1024;
    const size_t blocks_per_page =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
    const size_t entries_per_page = blocks_per_page + 1;
    const size_t total_entries = entries_per_page * batch_size;
    if (!ensure_decompression_workspace(total_entries) ||
        !ensure_batch_metadata(batch_size) ||
        !ensure_batch_output_metadata(batch_size)) {
        return false;
    }

    cudaMemcpyAsync(
        batch_compressed_ptrs_, host_compressed.data(),
        batch_size * sizeof(unsigned char*), cudaMemcpyHostToDevice,
        parent_stream);
    cudaMemcpyAsync(
        batch_output_ptrs_, host_outputs.data(),
        batch_size * sizeof(void*), cudaMemcpyHostToDevice,
        parent_stream);
    cudaMemcpyAsync(
        batch_error_bounds_, actual_error_bounds.data(),
        batch_size * sizeof(float), cudaMemcpyHostToDevice,
        parent_stream);
    cudaMemsetAsync(
        decompression_cmp_offsets_, 0,
        total_entries * sizeof(unsigned int), parent_stream);
    cudaMemsetAsync(
        decompression_local_offsets_, 0,
        total_entries * sizeof(unsigned int), parent_stream);
    cudaMemsetAsync(
        decompression_flags_, 0,
        total_entries * sizeof(int), parent_stream);

    if (!launch_cuszp_decompress_batch_fixed_bf16_indexed_scatter(
            batch_output_ptrs_, batch_compressed_ptrs_,
            decompression_cmp_offsets_, decompression_local_offsets_,
            decompression_flags_, batch_error_bounds_,
            layer_indices.data_ptr<int64_t>(),
            prefix_count, source_layers, elements_per_layer,
            selected_layers, batch_size, parent_stream)) {
        return false;
    }
    return check_cuda_error(
        "cuSZp indexed fixed BF16 scatter decompression failed");
}





size_t CuSZpWrapper::estimate_compressed_buffer_size(size_t original_size) {
    return original_size * 2;
}

uint3 CuSZpWrapper::get_tensor_dims(torch::Tensor tensor) {
    uint3 dims = {0, 0, 0};
    auto shape = tensor.sizes();
    
    if (config_.processing_dim == CUSZP_DIM_1D) {
        dims.x = dims.y = dims.z = 0;
    } else if (config_.processing_dim == CUSZP_DIM_2D) {
        if (shape.size() >= 2) {
            dims.y = static_cast<unsigned int>(shape[shape.size() - 2]);
            dims.x = static_cast<unsigned int>(shape[shape.size() - 1]);
            dims.z = 1;
        }
    } else if (config_.processing_dim == CUSZP_DIM_3D) {
        if (shape.size() >= 3) {
            dims.z = static_cast<unsigned int>(shape[shape.size() - 3]);
            dims.y = static_cast<unsigned int>(shape[shape.size() - 2]);
            dims.x = static_cast<unsigned int>(shape[shape.size() - 1]);
        }
    }
    
    return dims;
}
