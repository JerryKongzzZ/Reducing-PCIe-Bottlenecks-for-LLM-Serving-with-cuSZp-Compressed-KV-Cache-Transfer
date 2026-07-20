#ifndef CUSZP_FASTPATH_H
#define CUSZP_FASTPATH_H

#include <cuda_runtime.h>
#include <cuSZp.h>

bool launch_cuszp_compress_1d_fixed_f32(
    const float* input,
    unsigned char* compressed,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    size_t num_elements,
    float error_bound,
    cudaStream_t stream);

bool launch_cuszp_compress_batch_fixed_bf16(
    const void* const* input_pages,
    unsigned char* const* compressed_pages,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    float* error_bounds,
    size_t num_elements,
    size_t batch_size,
    bool use_relative_error,
    cudaStream_t stream);

bool launch_cuszp_compress_batch_fixed_bf16_indexed(
    const void* const* input_pages,
    unsigned char* const* compressed_pages,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    float* error_bounds,
    const int64_t* layer_indices,
    size_t prefix_count,
    size_t source_layers,
    size_t elements_per_layer,
    size_t selected_layers,
    size_t batch_size,
    bool use_relative_error,
    cudaStream_t stream);

bool launch_cuszp_decompress_1d_f32(
    float* output,
    const unsigned char* compressed,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    size_t num_elements,
    float error_bound,
    cuszp_mode_t mode,
    cudaStream_t stream);
bool launch_cuszp_decompress_batch_fixed_bf16(
    void* output,
    const unsigned char* const* compressed_pages,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    const float* error_bounds,
    size_t num_elements,
    size_t batch_size,
    cudaStream_t stream);
bool launch_cuszp_decompress_batch_fixed_bf16_scatter(
    void* const* destination_pages,
    size_t destinations_per_page,
    size_t elements_per_destination,
    const unsigned char* const* compressed_pages,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    const float* error_bounds,
    size_t num_elements,
    size_t batch_size,
    cudaStream_t stream);

bool launch_cuszp_decompress_batch_fixed_bf16_indexed_scatter(
    void* const* destination_pages,
    const unsigned char* const* compressed_pages,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    const float* error_bounds,
    const int64_t* layer_indices,
    size_t prefix_count,
    size_t source_layers,
    size_t elements_per_layer,
    size_t selected_layers,
    size_t batch_size,
    cudaStream_t stream);


#endif  // CUSZP_FASTPATH_H
