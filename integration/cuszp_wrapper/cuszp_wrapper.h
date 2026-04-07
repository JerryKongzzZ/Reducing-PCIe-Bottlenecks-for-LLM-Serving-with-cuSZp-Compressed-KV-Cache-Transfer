#ifndef CUSZP_WRAPPER_H
#define CUSZP_WRAPPER_H

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuSZp.h>

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

    bool compress(
        torch::Tensor input_tensor,
        torch::Tensor& compressed_buffer,
        size_t& compressed_size,
        float& actual_error_bound,
        cudaStream_t stream = nullptr
    );

    bool decompress(
        torch::Tensor compressed_buffer,
        size_t compressed_size,
        torch::Tensor output_tensor,
        float actual_error_bound,
        cudaStream_t stream = nullptr
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
};

#endif // CUSZP_WRAPPER_H
