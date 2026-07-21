#ifndef CUSZP_WRAPPER_H
#define CUSZP_WRAPPER_H

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuSZp.h>
#include <vector>

class CuSZpWrapper {
public:
    struct CompressionConfig {
        float error_bound = 1e-4f;
        bool use_relative_error = true;
        cuszp_dim_t processing_dim = CUSZP_DIM_1D;
        cuszp_mode_t encoding_mode = CUSZP_MODE_PLAIN;
        cuszp_type_t data_type = CUSZP_TYPE_FLOAT;
    };

    CuSZpWrapper(const CompressionConfig& config, int device_id = 0);
    ~CuSZpWrapper();

    CuSZpWrapper(const CuSZpWrapper&) = delete;
    CuSZpWrapper& operator=(const CuSZpWrapper&) = delete;

    bool compress(
        torch::Tensor input_tensor,
        torch::Tensor& compressed_buffer,
        size_t& compressed_size,
        float& actual_error_bound,
        float eps_override = -1.0f,
        cudaStream_t stream = nullptr
    );
    bool compress_batch_fixed(
        const std::vector<torch::Tensor>& input_tensors,
        const std::vector<torch::Tensor>& compressed_buffers,
        const std::vector<float>& eps_overrides,
        std::vector<size_t>& compressed_sizes,
        std::vector<float>& actual_error_bounds,
        cudaStream_t stream = nullptr
    );
    bool compress_batch_fixed_bf16(
        const std::vector<torch::Tensor>& input_tensors,
        const std::vector<torch::Tensor>& compressed_buffers,
        const std::vector<float>& eps_overrides,
        std::vector<size_t>& compressed_sizes,
        std::vector<float>& actual_error_bounds,
        cudaStream_t stream = nullptr
    );


    bool compress_batch_fixed_bf16_indexed(
        const std::vector<torch::Tensor>& input_tensors,
        const std::vector<torch::Tensor>& compressed_buffers,
        torch::Tensor layer_indices,
        size_t prefix_count,
        size_t source_layers,
        size_t elements_per_layer,
        const std::vector<float>& eps_overrides,
        std::vector<size_t>& compressed_sizes,
        std::vector<float>& actual_error_bounds,
        cudaStream_t stream = nullptr
    );
    bool compress_batch_fixed_bf16_indexed_groups(
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
        cudaStream_t stream = nullptr
    );
    bool decompress(
        torch::Tensor compressed_buffer,
        size_t compressed_size,
        torch::Tensor output_tensor,
        float actual_error_bound,
        cudaStream_t stream = nullptr
    );
    bool decompress_batch(
        const std::vector<torch::Tensor>& compressed_buffers,
        const std::vector<size_t>& compressed_sizes,
        const std::vector<torch::Tensor>& output_tensors,
        const std::vector<float>& actual_error_bounds,
        cudaStream_t parent_stream = nullptr
    );
    bool decompress_batch_fixed_bf16(
        const std::vector<torch::Tensor>& compressed_buffers,
        const std::vector<size_t>& compressed_sizes,
        torch::Tensor output_tensor,
        size_t num_elements_per_page,
        const std::vector<float>& actual_error_bounds,
        cudaStream_t parent_stream = nullptr
    );
    bool decompress_batch_fixed_bf16_scatter(
        const std::vector<torch::Tensor>& compressed_buffers,
        const std::vector<size_t>& compressed_sizes,
        const std::vector<torch::Tensor>& output_tensors,
        size_t num_elements_per_page,
        size_t destinations_per_page,
        size_t elements_per_destination,
        const std::vector<float>& actual_error_bounds,
        cudaStream_t parent_stream = nullptr
    );

    bool decompress_batch_fixed_bf16_indexed_scatter(
        const std::vector<torch::Tensor>& compressed_buffers,
        const std::vector<size_t>& compressed_sizes,
        const std::vector<torch::Tensor>& output_tensors,
        torch::Tensor layer_indices,
        size_t prefix_count,
        size_t source_layers,
        size_t elements_per_layer,
        const std::vector<float>& actual_error_bounds,
        cudaStream_t parent_stream = nullptr
    );




    static size_t estimate_compressed_buffer_size(size_t original_size);

    CompressionConfig get_config() const { return config_; }
    void update_config(const CompressionConfig& config) { config_ = config; }

private:
    CompressionConfig config_;
    int device_id_;
    void* temp_compression_buffer_;
    size_t temp_buffer_size_;
    uint3 get_tensor_dims(torch::Tensor tensor);
    unsigned int* decompression_cmp_offsets_;
    unsigned int* decompression_local_offsets_;
    int* decompression_flags_;
    size_t decompression_workspace_entries_;
    std::vector<cudaStream_t> decompression_streams_;
    std::vector<cudaEvent_t> decompression_events_;
    cudaEvent_t decompression_input_ready_;
    bool ensure_decompression_streams(size_t count);
    const unsigned char** batch_compressed_ptrs_;
    float* batch_error_bounds_;
    size_t batch_metadata_capacity_;
    bool ensure_batch_metadata(size_t count);
    void** batch_output_ptrs_;
    size_t batch_output_capacity_;
    unsigned int* batch_compressed_sizes_host_;
    float* batch_error_bounds_host_;
    size_t batch_compressed_sizes_capacity_;
    unsigned char** batch_compressed_output_ptrs_;
    size_t batch_compression_metadata_capacity_;
    bool ensure_batch_compression_metadata(size_t count);
    bool ensure_batch_compressed_sizes_host(size_t count);
    bool ensure_batch_output_metadata(size_t count);
    bool ensure_decompression_workspace(size_t entries);
};

#endif // CUSZP_WRAPPER_H
