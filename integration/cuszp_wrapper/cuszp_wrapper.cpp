#include "cuszp_wrapper.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <algorithm>

CuSZpWrapper::CuSZpWrapper(const CompressionConfig& config, int device_id)
    : config_(config), device_id_(device_id),
      temp_compression_buffer_(nullptr), temp_buffer_size_(0) {
    cudaSetDevice(device_id);
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
    if (compressed_buffer.numel() * compressed_buffer.element_size() < estimated_size) {
        compressed_buffer = torch::empty(
            {static_cast<long>(estimated_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input_tensor.device())
        );
    }
    
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    
    // Determine per-call eps to use (eps_override > 0 means override)
    float eps_to_use = (eps_override > 0.0f) ? eps_override : config_.error_bound;

    // Dynamically calculate error bound based on true min/max of tensor
    actual_error_bound = eps_to_use;
    if (config_.use_relative_error) {
        float min_val = input_tensor.min().item<float>();
        float max_val = input_tensor.max().item<float>();
        float range_estimate = max_val - min_val;
        if (range_estimate < 1e-6f) range_estimate = 1e-6f; // Prevent near-zero range
        actual_error_bound = eps_to_use * range_estimate;
    }
    
    uint3 dims = get_tensor_dims(input_tensor);
    
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
