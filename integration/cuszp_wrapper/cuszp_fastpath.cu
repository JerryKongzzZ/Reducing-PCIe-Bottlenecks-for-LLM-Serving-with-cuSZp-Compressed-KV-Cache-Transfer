#include "cuszp_fastpath.h"

#include <cuSZp/cuSZp_kernels_1D_f32.h>

bool launch_cuszp_compress_1d_fixed_f32(
    const float* input,
    unsigned char* compressed,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    size_t num_elements,
    float error_bound,
    cudaStream_t stream) {
    const int grid_size = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(grid_size);
    const size_t shared_bytes = sizeof(unsigned int) * 2;
    cuSZp_compress_kernel_1D_fixed_f32<<<
        grid, block, shared_bytes, stream>>>(
            input, compressed, compressed_offsets, local_offsets, flags,
            error_bound, num_elements);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_cuszp_decompress_1d_f32(
    float* output,
    const unsigned char* compressed,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    size_t num_elements,
    float error_bound,
    cuszp_mode_t mode,
    cudaStream_t stream) {
    const int grid_size = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(grid_size);
    const size_t shared_bytes = sizeof(unsigned int) * 2;

    switch (mode) {
        case CUSZP_MODE_FIXED:
            cuSZp_decompress_kernel_1D_fixed_f32<<<grid, block, shared_bytes, stream>>>(
                output, compressed, compressed_offsets, local_offsets, flags,
                error_bound, num_elements);
            break;
        case CUSZP_MODE_PLAIN:
            cuSZp_decompress_kernel_1D_plain_f32<<<grid, block, shared_bytes, stream>>>(
                output, compressed, compressed_offsets, local_offsets, flags,
                error_bound, num_elements);
            break;
        case CUSZP_MODE_OUTLIER:
            cuSZp_decompress_kernel_1D_outlier_f32<<<grid, block, shared_bytes, stream>>>(
                output, compressed, compressed_offsets, local_offsets, flags,
                error_bound, num_elements);
            break;
        default:
            return false;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}
