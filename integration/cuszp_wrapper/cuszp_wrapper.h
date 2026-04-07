#ifndef CUSZP_WRAPPER_H
#define CUSZP_WRAPPER_H

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuSZp.h>

/**
 * cuSZp包装器类，用于在vLLM中集成cuSZp压缩功能
 * 
 * 这个类提供了压缩和解压缩KV cache页面的接口，支持：
 * - 异步压缩/解压缩（使用CUDA流）
 * - 内存池管理
 * - 错误边界配置
 */
class CuSZpWrapper {
public:
    /**
     * 压缩配置结构
     */
    struct CompressionConfig {
        float error_bound = 1e-4f;           // 错误边界（相对或绝对）
        bool use_relative_error = true;     // 是否使用相对错误边界
        cuszp_dim_t processing_dim = CUSZP_DIM_1D;  // 处理维度（1D/2D/3D）
        cuszp_mode_t encoding_mode = CUSZP_MODE_PLAIN;  // 编码模式
        cuszp_type_t data_type = CUSZP_TYPE_FLOAT;  // 数据类型（float/double）
    };

    /**
     * 构造函数
     * @param config 压缩配置
     * @param device_id GPU设备ID
     */
    CuSZpWrapper(const CompressionConfig& config, int device_id = 0);
    
    /**
     * 析构函数
     */
    ~CuSZpWrapper();

    /**
     * 压缩GPU张量（D2H场景）
     * @param input_tensor 输入张量（GPU上）
     * @param compressed_buffer 压缩后的缓冲区（GPU上，用于传输）
     * @param compressed_size 压缩后的大小（输出，引用）
     * @param stream CUDA流（可选，用于异步执行）
     * @return 是否成功
     */
    bool compress(
        torch::Tensor input_tensor,
        torch::Tensor& compressed_buffer,
        size_t& compressed_size,
        cudaStream_t stream = nullptr
    );

    /**
     * 解压缩到GPU张量（H2D场景）
     * @param compressed_buffer 压缩后的缓冲区（GPU上）
     * @param compressed_size 压缩后的大小
     * @param output_tensor 输出张量（GPU上）
     * @param stream CUDA流（可选，用于异步执行）
     * @return 是否成功
     */
    bool decompress(
        torch::Tensor compressed_buffer,
        size_t compressed_size,
        torch::Tensor output_tensor,
        cudaStream_t stream = nullptr
    );

    /**
     * 估算压缩缓冲区所需的最大大小
     * @param original_size 原始数据大小（字节）
     * @return 估算的最大压缩缓冲区大小
     */
    static size_t estimate_compressed_buffer_size(size_t original_size);

    /**
     * 获取压缩配置
     */
    CompressionConfig get_config() const { return config_; }

    /**
     * 更新压缩配置
     */
    void update_config(const CompressionConfig& config) { config_ = config; }

private:
    CompressionConfig config_;
    int device_id_;
    
    // 内存池：用于存储压缩/解压缩的临时缓冲区
    void* temp_compression_buffer_;
    size_t temp_buffer_size_;
    
    // 计算相对错误边界的实际值
    float compute_actual_error_bound(torch::Tensor tensor);
    
    // 获取张量的维度信息
    uint3 get_tensor_dims(torch::Tensor tensor);
};

#endif // CUSZP_WRAPPER_H

