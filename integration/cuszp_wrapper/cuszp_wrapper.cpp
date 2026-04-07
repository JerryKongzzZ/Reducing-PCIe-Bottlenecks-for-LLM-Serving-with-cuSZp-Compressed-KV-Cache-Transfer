#include "cuszp_wrapper.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <algorithm>

CuSZpWrapper::CuSZpWrapper(const CompressionConfig& config, int device_id)
    : config_(config), device_id_(device_id),
      temp_compression_buffer_(nullptr), temp_buffer_size_(0) {
    // 设置当前CUDA设备
    cudaSetDevice(device_id);
    
    // 初始化临时缓冲区（延迟分配）
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
    
    // 检查输入张量
    if (!input_tensor.is_cuda()) {
        std::cerr << "Error: Input tensor must be on GPU" << std::endl;
        return false;
    }
    
    // 获取张量信息
    size_t nb_elements = input_tensor.numel();
    void* d_input_data = input_tensor.data_ptr();
    
    // 确保压缩缓冲区足够大
    size_t estimated_size = estimate_compressed_buffer_size(input_tensor.nbytes());
    if (compressed_buffer.numel() * compressed_buffer.element_size() < estimated_size) {
        // 需要重新分配缓冲区
        compressed_buffer = torch::empty(
            {static_cast<long>(estimated_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input_tensor.device())
        );
    }
    
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    
    // 计算实际错误边界
    float actual_error_bound = config_.error_bound;
    if (config_.use_relative_error) {
        actual_error_bound = compute_actual_error_bound(input_tensor);
    }
    
    // 获取维度信息
    uint3 dims = get_tensor_dims(input_tensor);
    
            // 调用cuSZp压缩
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
        
        // 同步流（如果提供了）
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
    
    // 检查输入
    if (!compressed_buffer.is_cuda() || !output_tensor.is_cuda()) {
        std::cerr << "Error: Both compressed buffer and output tensor must be on GPU" << std::endl;
        return false;
    }
    
    // 获取张量信息
    size_t nb_elements = output_tensor.numel();
    unsigned char* d_compressed = compressed_buffer.data_ptr<unsigned char>();
    void* d_output_data = output_tensor.data_ptr();
    
    // 计算实际错误边界（需要从原始数据计算，这里使用配置值）
    float actual_error_bound = config_.error_bound;
    
    // 获取维度信息
    uint3 dims = get_tensor_dims(output_tensor);
    
    // 调用cuSZp解压缩
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
        
        // 同步流（如果提供了）
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
    // 保守估计：压缩比通常为2-10倍，我们使用2倍作为安全边界
    // 实际压缩比取决于数据特征和错误边界
    return original_size * 2;
}

float CuSZpWrapper::compute_actual_error_bound(torch::Tensor tensor) {
    // 计算相对错误边界需要知道数据的范围
    // 这里简化处理：假设数据范围在[-1, 1]之间
    // 实际应用中可能需要更精确的计算
    
    // 注意：这个计算应该在GPU上进行，这里简化处理
    // 实际实现中，可以使用CUDA kernel来计算min/max
    float range_estimate = 2.0f;  // 简化假设
    
    return config_.error_bound * range_estimate;
}

uint3 CuSZpWrapper::get_tensor_dims(torch::Tensor tensor) {
    uint3 dims = {0, 0, 0};
    auto shape = tensor.sizes();
    
    if (config_.processing_dim == CUSZP_DIM_1D) {
        // 1D处理：所有维度信息设为0（cuSZp会忽略）
        dims.x = dims.y = dims.z = 0;
    } else if (config_.processing_dim == CUSZP_DIM_2D) {
        // 2D处理：使用最后两个维度
        if (shape.size() >= 2) {
            dims.y = static_cast<unsigned int>(shape[shape.size() - 2]);
            dims.x = static_cast<unsigned int>(shape[shape.size() - 1]);
            dims.z = 1;
        }
    } else if (config_.processing_dim == CUSZP_DIM_3D) {
        // 3D处理：使用最后三个维度
        if (shape.size() >= 3) {
            dims.z = static_cast<unsigned int>(shape[shape.size() - 3]);
            dims.y = static_cast<unsigned int>(shape[shape.size() - 2]);
            dims.x = static_cast<unsigned int>(shape[shape.size() - 1]);
        }
    }
    
    return dims;
}

