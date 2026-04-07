#ifndef CUSZP_WRAPPER_H
#define CUSZP_WRAPPER_H

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuSZp.h>

/**
 * cuSZp wrapper class for integrating cuSZp compression functionality in vLLM
 * 
 * This class provides interfaces for compressing and decompressing KV cache pages, supporting:
 * - Asynchronous compression/decompression (using CUDA streams)
 * - Memory pool management
 * - Error bound configuration
 */
class CuSZpWrapper {
public:
    /**
     * Compression configuration structure
     */
    struct CompressionConfig {
        float error_bound = 1e-4f;           // Error bound (relative or absolute)
        bool use_relative_error = true;     // Whether to use relative error bound
        cuszp_dim_t processing_dim = CUSZP_DIM_1D;  // Processing dimension (1D/2D/3D)
        cuszp_mode_t encoding_mode = CUSZP_MODE_PLAIN;  // Encoding mode
        cuszp_type_t data_type = CUSZP_TYPE_FLOAT;  // Data type (float/double)
    };

    /**
     * Constructor
     * @param config Compression configuration
     * @param device_id GPU device ID
     */
    CuSZpWrapper(const CompressionConfig& config, int device_id = 0);
    
    /**
     * Destructor
     */
    ~CuSZpWrapper();

    /**
     * Compress GPU tensor (D2H scenario)
     * @param input_tensor Input tensor (on GPU)
     * @param compressed_buffer Compressed buffer (on GPU, used for transfer)
     * @param compressed_size Compressed size (output, reference)
     * @param stream CUDA stream (optional, for asynchronous execution)
     * @return Success status
     */
    bool compress(
        torch::Tensor input_tensor,
        torch::Tensor& compressed_buffer,
        size_t& compressed_size,
        cudaStream_t stream = nullptr
    );

    /**
     * Decompress to GPU tensor (H2D scenario)
     * @param compressed_buffer Compressed buffer (on GPU)
     * @param compressed_size Compressed size
     * @param output_tensor Output tensor (on GPU)
     * @param stream CUDA stream (optional, for asynchronous execution)
     * @return Success status
     */
    bool decompress(
        torch::Tensor compressed_buffer,
        size_t compressed_size,
        torch::Tensor output_tensor,
        cudaStream_t stream = nullptr
    );

    /**
     * Estimate the maximum required size for the compressed buffer
     * @param original_size Original data size (bytes)
     * @return Estimated maximum compressed buffer size
     */
    static size_t estimate_compressed_buffer_size(size_t original_size);

    /**
     * Get compression configuration
     */
    CompressionConfig get_config() const { return config_; }

    /**
     * Update compression configuration
     */
    void update_config(const CompressionConfig& config) { config_ = config; }

private:
    CompressionConfig config_;
    int device_id_;
    
    // Memory pool: used to store temporary buffers for compression/decompression
    void* temp_compression_buffer_;
    size_t temp_buffer_size_;
    
    // Calculate the actual value of relative error bound
    float compute_actual_error_bound(torch::Tensor tensor);
    
    // Get dimension information of the tensor
    uint3 get_tensor_dims(torch::Tensor tensor);
};

#endif // CUSZP_WRAPPER_H
