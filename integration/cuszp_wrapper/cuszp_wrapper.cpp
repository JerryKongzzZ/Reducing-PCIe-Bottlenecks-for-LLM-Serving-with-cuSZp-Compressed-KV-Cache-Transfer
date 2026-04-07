#include "cuszp_wrapper.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <algorithm>

CuSZpWrapper::CuSZpWrapper(const CompressionConfig& config, int device_id)
    : config_(config), device_id_(device_id),
      temp_compression_buffer_(nullptr), temp_buffer_size_(0) {
    // Set current CUDA device
    cudaSetDevice(device_id);
    
    // Initialize temporary buffer (lazy allocation)
    temp_buffer_size_ = 0;
}

CuSZpWrapper::~CuSZpWrapper() {
    if (temp_compression_buffer_) {
        cudaFree(temp_compression_buffer_);
        temp_compression_buffer_ = nullptr;
    }
}

bool CuSZpWrapper::compress(
    torch::Tensor input_tensor,
    torch::Tensor& compressed_buffer,
    size_t& compressed_size,
    cudaStream_t stream) {
    
    // Check input tensor
    if (!input_tensor.is_cuda()) {
        std::cerr << "Error: Input tensor must be on GPU" << std::endl;
        return false;
    }
    
    // Get tensor information
    size_t nb_elements = input_tensor.numel();
    void* d_input_data = input_tensor.data_ptr();
    
    // Ensure compressed buffer is large enough
    size_t estimated_size = estimate_compressed_buffer_size(input_tensor.nbytes());
    if (compressed_buffer.numel() * compressed_buffer.element_size() < estimated_size) {
        // Need to reallocate buffer
        compressed_buffer = torch::empty(
            {static_cast<long>(estimated_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input_tensor.device())
        );
    }
    
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    
    // Calculate actual error bound
    float actual_error_bound = config_.error_bound;
    if (config_.use_relative_error) {
        actual_error_bound = compute_actual_error_bound(input_tensor);
    }
    
    // Get dimension information
    uint3 dims = get_tensor_dims(input_tensor);
    
    // Call cuSZp compression
    try {
        size_t compressed_size_temp = 0;
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
        compressed_size = compressed_size_temp;
        
        // Synchronize stream (if provided)
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
        
        return true;
    } catch (...) {
        std::cerr << "Error: cuSZp compression failed" << std::endl;
        return false;
    }
}

bool CuSZpWrapper::decompress(
    torch::Tensor compressed_buffer,
    size_t compressed_size,
    torch::Tensor output_tensor,
    cudaStream_t stream) {
    
    // Check input
    if (!compressed_buffer.is_cuda() || !output_tensor.is_cuda()) {
        std::cerr << "Error: Both compressed buffer and output tensor must be on GPU" << std::endl;
        return false;
    }
    
    // Get tensor information
    size_t nb_elements = output_tensor.numel();
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    void* d_output_data = output_tensor.data_ptr();
    
    // Calculate actual error bound (must be exactly consistent with compression!)
    float actual_error_bound = config_.error_bound;
    if (config_.use_relative_error) {
        // The logic here must be exactly the same as during compression
        // Ideally, this value should be saved with compressed data or passed as metadata
        actual_error_bound = config_.error_bound * 2.0f; // Consistent with hardcoded 2.0f in compute_actual_error_bound
    }
    
    // Get dimension information
    uint3 dims = get_tensor_dims(output_tensor);
    
    // Call cuSZp decompression
    try {
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
        
        // Synchronize stream (if provided)
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
        
        return true;
    } catch (...) {
        std::cerr << "Error: cuSZp decompression failed" << std::endl;
        return false;
    }
}

size_t CuSZpWrapper::estimate_compressed_buffer_size(size_t original_size) {
    // Conservative estimate: compression ratio is usually 2-10x, we use 2x as a safe boundary
    // Actual compression ratio depends on data characteristics and error bound
    return original_size * 2;
}

float CuSZpWrapper::compute_actual_error_bound(torch::Tensor tensor) {
    // Calculating relative error bound requires knowing the data range
    // Simplified handling here: assuming data range is between [-1, 1]
    // More precise calculations might be needed in actual applications
    
    // Note: this calculation should be performed on GPU, simplified here
    // In actual implementation, CUDA kernel can be used to compute min/max
    float range_estimate = 2.0f;  // Simplified assumption
    
    return config_.error_bound * range_estimate;
}

uint3 CuSZpWrapper::get_tensor_dims(torch::Tensor tensor) {
    uint3 dims = {0, 0, 0};
    auto shape = tensor.sizes();
    
    if (config_.processing_dim == CUSZP_DIM_1D) {
        // 1D processing: all dimension information set to 0 (ignored by cuSZp)
        dims.x = dims.y = dims.z = 0;
    } else if (config_.processing_dim == CUSZP_DIM_2D) {
        // 2D processing: use the last two dimensions
        if (shape.size() >= 2) {
            dims.y = static_cast<unsigned int>(shape[shape.size() - 2]);
            dims.x = static_cast<unsigned int>(shape[shape.size() - 1]);
            dims.z = 1;
        }
    } else if (config_.processing_dim == CUSZP_DIM_3D) {
        // 3D processing: use the last three dimensions
        if (shape.size() >= 3) {
            dims.z = static_cast<unsigned int>(shape[shape.size() - 3]);
            dims.y = static_cast<unsigned int>(shape[shape.size() - 2]);
            dims.x = static_cast<unsigned int>(shape[shape.size() - 1]);
        }
    }
    
    return dims;
}
