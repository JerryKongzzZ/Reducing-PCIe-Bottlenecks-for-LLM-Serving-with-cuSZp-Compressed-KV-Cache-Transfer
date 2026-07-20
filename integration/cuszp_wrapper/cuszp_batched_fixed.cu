#include "cuszp_fastpath.h"

#include <cuda_bf16.h>
#include <cuSZp/cuSZp_kernels_1D_f32.h>
#include <cfloat>
#include <cstdio>

namespace {

__device__ inline int batch_quantization(float data, float reciprocal_precision) {
    int result;
    asm("{\n\t"
        ".reg .f32 dataRecip;\n\t"
        ".reg .f32 temp1;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dataRecip, %1, %2;\n\t"
        "setp.ge.f32 p, dataRecip, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 temp1, dataRecip, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, temp1;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}" : "=r"(result) : "f"(data), "f"(reciprocal_precision));
    return result;
}

__device__ inline int batch_bit_count(unsigned int value) {
    int leading_zeros;
    asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(value));
    return 32 - leading_zeros;
}

__device__ inline size_t selected_bf16_source_index(
    const size_t logical_index,
    const int64_t* const layer_indices,
    const size_t selected_layers,
    const size_t source_layers,
    const size_t elements_per_layer) {
    if (layer_indices == nullptr) return logical_index;
    const size_t elements_per_prefix =
        selected_layers * elements_per_layer;
    const size_t prefix_index = logical_index / elements_per_prefix;
    const size_t within_prefix = logical_index % elements_per_prefix;
    const size_t selected_layer = within_prefix / elements_per_layer;
    const size_t inner_index = within_prefix % elements_per_layer;
    return (
        prefix_index * source_layers +
        static_cast<size_t>(layer_indices[selected_layer])
    ) * elements_per_layer + inner_index;
}

// This is the cuSZp fixed 1D packing procedure with two scheduling changes:
// pages share one grid, and native BF16 values are converted in registers.
// The byte layout remains compatible with the upstream fixed-mode decoder.
__global__ void compute_relative_bounds_bf16_kernel(
    const __nv_bfloat16* const* const input_pages,
    float* const error_bounds,
    const size_t num_elements,
    const int64_t* const layer_indices,
    const size_t selected_layers,
    const size_t source_layers,
    const size_t elements_per_layer) {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    __shared__ float warp_mins[kWarps];
    __shared__ float warp_maxs[kWarps];

    const int page = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const __nv_bfloat16* const input = input_pages[page];
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    for (size_t index = threadIdx.x; index < num_elements;
         index += blockDim.x) {
        const size_t source_index = selected_bf16_source_index(
            index, layer_indices, selected_layers, source_layers,
            elements_per_layer);
        const float value = __bfloat162float(input[source_index]);
        local_min = fminf(local_min, value);
        local_max = fmaxf(local_max, value);
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_min = fminf(
            local_min,
            __shfl_down_sync(0xffffffff, local_min, delta));
        local_max = fmaxf(
            local_max,
            __shfl_down_sync(0xffffffff, local_max, delta));
    }
    if (lane == 0) {
        warp_mins[warp] = local_min;
        warp_maxs[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        local_min = lane < kWarps ? warp_mins[lane] : FLT_MAX;
        local_max = lane < kWarps ? warp_maxs[lane] : -FLT_MAX;
        for (int delta = 16; delta > 0; delta >>= 1) {
            local_min = fminf(
                local_min,
                __shfl_down_sync(0xffffffff, local_min, delta));
            local_max = fmaxf(
                local_max,
                __shfl_down_sync(0xffffffff, local_max, delta));
        }
        if (lane == 0) {
            const float value_range = fmaxf(local_max - local_min, 1e-6f);
            error_bounds[page] *= value_range;
        }
    }
}

__global__ void compress_batch_fixed_bf16_kernel(
    const __nv_bfloat16* const* const input_pages,
    unsigned char* const* const compressed_pages,
    volatile unsigned int* const compressed_offsets,
    volatile unsigned int* const local_offsets,
    volatile int* const flags,
    const float* const error_bounds,
    const size_t num_elements,
    const int blocks_per_page,
    const int64_t* const layer_indices,
    const size_t selected_layers,
    const size_t source_layers,
    const size_t elements_per_layer) {
    const int page = blockIdx.x / blocks_per_page;
    const int local_block = blockIdx.x - page * blocks_per_page;
    const int entries_per_page = blocks_per_page + 1;
    const __nv_bfloat16* const input = input_pages[page];
    unsigned char* const compressed = compressed_pages[page];
    volatile unsigned int* const page_compressed_offsets =
        compressed_offsets + page * entries_per_page;
    volatile unsigned int* const page_local_offsets =
        local_offsets + page * entries_per_page;
    volatile int* const page_flags = flags + page * entries_per_page;

    __shared__ unsigned int exclusive_sum;
    __shared__ unsigned int base_index;

    const int lane = threadIdx.x;
    const int warp = local_block;
    const int chunks_per_block = thread_chunk >> 5;
    const int rate_offset = blocks_per_page * (tblock_size * thread_chunk) / 32;
    const float reciprocal_precision = 0.5f / error_bounds[page];

    if (lane == 0) {
        exclusive_sum = 0;
        base_index = 0;
    }
    __syncthreads();

    int absolute_quantized[thread_chunk];
    unsigned int sign_flags[chunks_per_block];
    int fixed_rate[chunks_per_block];
    unsigned int thread_offset = 0;
    const int base_start_index = warp * thread_chunk * 32;

    for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
        const int block_start =
            base_start_index + chunk * 1024 + lane * 32;
        sign_flags[chunk] = 0;
        int max_quantized = 0;
        const int quantized_start = chunk * 32;

        #pragma unroll 32
        for (int value = 0; value < 32; ++value) {
            const int input_index = block_start + value;
            int quantized = 0;
            if (input_index < num_elements) {
                const size_t source_index = selected_bf16_source_index(
                    input_index, layer_indices, selected_layers,
                    source_layers, elements_per_layer);
                quantized = batch_quantization(
                    __bfloat162float(input[source_index]),
                    reciprocal_precision);
            }
            if (quantized < 0) {
                sign_flags[chunk] |= 1u << (31 - value);
                quantized = -quantized;
            }
            absolute_quantized[quantized_start + value] = quantized;
            max_quantized =
                max_quantized > quantized ? max_quantized : quantized;
        }

        fixed_rate[chunk] =
            batch_bit_count(static_cast<unsigned int>(max_quantized));
        thread_offset += fixed_rate[chunk]
            ? 4 + static_cast<unsigned int>(fixed_rate[chunk]) * 4
            : 0;
        if (block_start < num_elements) {
            compressed[block_start / 32] =
                static_cast<unsigned char>(fixed_rate[chunk]);
        }
        __syncthreads();
    }

    #pragma unroll 5
    for (int delta = 1; delta < 32; delta <<= 1) {
        const unsigned int prior =
            __shfl_up_sync(0xffffffff, thread_offset, delta);
        if (lane >= delta) thread_offset += prior;
    }
    __syncthreads();

    if (lane == 31) {
        page_local_offsets[warp + 1] = thread_offset;
        __threadfence();
        if (warp == 0) {
            page_flags[0] = 2;
            __threadfence();
            page_flags[1] = 1;
            __threadfence();
        } else {
            page_flags[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (warp > 0 && lane == 0) {
        int lookback = warp;
        unsigned int local_exclusive_sum = 0;
        while (lookback > 0) {
            int status;
            do {
                status = page_flags[lookback];
                __threadfence();
            } while (status == 0);
            if (status == 2) {
                local_exclusive_sum += page_compressed_offsets[lookback];
                __threadfence();
                break;
            }
            local_exclusive_sum += page_local_offsets[lookback];
            --lookback;
            __threadfence();
        }
        exclusive_sum = local_exclusive_sum;
    }
    __syncthreads();

    if (lane == 0) {
        if (warp == 0) {
            page_compressed_offsets[0] = 0;
            if (blocks_per_page == 1) {
                page_compressed_offsets[1] = page_local_offsets[1];
            }
        } else {
            page_compressed_offsets[warp] = exclusive_sum;
            __threadfence();
            if (warp == blocks_per_page - 1) {
                page_compressed_offsets[warp + 1] =
                    page_compressed_offsets[warp] +
                    page_local_offsets[warp + 1];
            }
            __threadfence();
            page_flags[warp] = 2;
            __threadfence();
        }
    }
    __syncthreads();

    if (lane == 0) base_index = exclusive_sum + rate_offset;
    __syncthreads();

    const unsigned int base_compressed_byte_offset = base_index;
    unsigned int current_byte_offset = 0;
    for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
        unsigned int chunk_bytes = fixed_rate[chunk]
            ? 4 + static_cast<unsigned int>(fixed_rate[chunk]) * 4
            : 0;
        #pragma unroll 5
        for (int delta = 1; delta < 32; delta <<= 1) {
            const unsigned int prior =
                __shfl_up_sync(0xffffffff, chunk_bytes, delta);
            if (lane >= delta) chunk_bytes += prior;
        }
        const unsigned int previous_thread =
            __shfl_up_sync(0xffffffff, chunk_bytes, 1);
        unsigned int compressed_byte_offset =
            base_compressed_byte_offset + current_byte_offset;
        if (lane != 0) compressed_byte_offset += previous_thread;

        if (fixed_rate[chunk]) {
            uchar4 packed;
            packed.x = 0xff & (sign_flags[chunk] >> 24);
            packed.y = 0xff & (sign_flags[chunk] >> 16);
            packed.z = 0xff & (sign_flags[chunk] >> 8);
            packed.w = 0xff & sign_flags[chunk];
            reinterpret_cast<uchar4*>(compressed)[
                compressed_byte_offset / 4] = packed;
            compressed_byte_offset += 4;

            int mask = 1;
            const int quantized_start = chunk * 32;
            for (int bit = 0; bit < fixed_rate[chunk]; ++bit) {
                packed.x = packed.y = packed.z = packed.w = 0;
                #pragma unroll 32
                for (int value = 0; value < 32; ++value) {
                    const unsigned char encoded =
                        static_cast<unsigned char>(
                            (absolute_quantized[
                                quantized_start + value] & mask) >> bit);
                    if (value < 8) {
                        packed.x |= encoded << (7 - value);
                    } else if (value < 16) {
                        packed.y |= encoded << (15 - value);
                    } else if (value < 24) {
                        packed.z |= encoded << (23 - value);
                    } else {
                        packed.w |= encoded << (31 - value);
                    }
                }
                reinterpret_cast<uchar4*>(compressed)[
                    compressed_byte_offset / 4] = packed;
                compressed_byte_offset += 4;
                mask <<= 1;
            }
        }
        current_byte_offset +=
            __shfl_sync(0xffffffff, chunk_bytes, 31);
    }
}

__global__ void decompress_batch_fixed_bf16_kernel(
    __nv_bfloat16* const output,
    __nv_bfloat16* const* const destination_pages,
    const int destinations_per_page,
    const size_t elements_per_destination,
    const unsigned char* const* const compressed_pages,
    volatile unsigned int* const compressed_offsets,
    volatile unsigned int* const local_offsets,
    volatile int* const flags,
    const float* const error_bounds,
    const size_t num_elements,
    const int blocks_per_page,
    const int64_t* const layer_indices,
    const size_t selected_layers,
    const size_t source_layers,
    const size_t elements_per_layer) {
    const int page = blockIdx.x / blocks_per_page;
    const int local_block = blockIdx.x - page * blocks_per_page;
    const int entries_per_page = blocks_per_page + 1;
    const unsigned char* const compressed = compressed_pages[page];
    __nv_bfloat16* const decoded =
        output == nullptr ? nullptr : output + page * num_elements;
    __nv_bfloat16* const direct_single_destination =
        destination_pages != nullptr && destinations_per_page == 1
            ? destination_pages[page]
            : nullptr;
    volatile unsigned int* const page_compressed_offsets =
        compressed_offsets + page * entries_per_page;
    volatile unsigned int* const page_local_offsets =
        local_offsets + page * entries_per_page;
    volatile int* const page_flags = flags + page * entries_per_page;
    const float error_bound = error_bounds[page];

    __shared__ unsigned int exclusive_sum;
    __shared__ unsigned int base_index;

    const int lane = threadIdx.x;
    const int warp = local_block;
    const int chunks_per_block = thread_chunk >> 5;
    const int rate_offset =
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk) * (tblock_size * thread_chunk) / 32;

    if (lane == 0) {
        exclusive_sum = 0;
        base_index = 0;
    }
    __syncthreads();

    int fixed_rate[chunks_per_block];
    unsigned int thread_offset = 0;
    for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
        const int block_index =
            warp * thread_chunk + chunk * 32 + lane;
        fixed_rate[chunk] = static_cast<int>(compressed[block_index]);
        thread_offset += fixed_rate[chunk] ? 4 + fixed_rate[chunk] * 4 : 0;
        __syncthreads();
    }

    #pragma unroll 5
    for (int delta = 1; delta < 32; delta <<= 1) {
        const int prior = __shfl_up_sync(0xffffffff, thread_offset, delta);
        if (lane >= delta) thread_offset += prior;
    }
    __syncthreads();

    if (lane == 31) {
        page_local_offsets[warp + 1] = thread_offset;
        __threadfence();
        if (warp == 0) {
            page_flags[0] = 2;
            __threadfence();
            page_flags[1] = 1;
            __threadfence();
        } else {
            page_flags[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (warp > 0 && lane == 0) {
        int lookback = warp;
        int local_exclusive_sum = 0;
        while (lookback > 0) {
            int status;
            do {
                status = page_flags[lookback];
                __threadfence();
            } while (status == 0);
            if (status == 2) {
                local_exclusive_sum += page_compressed_offsets[lookback];
                __threadfence();
                break;
            }
            local_exclusive_sum += page_local_offsets[lookback];
            --lookback;
            __threadfence();
        }
        exclusive_sum = local_exclusive_sum;
    }
    __syncthreads();

    if (warp > 0) {
        if (lane == 0) page_compressed_offsets[warp] = exclusive_sum;
        __threadfence();
        if (lane == 0) page_flags[warp] = 2;
        __threadfence();
    }
    __syncthreads();

    if (lane == 0) base_index = exclusive_sum + rate_offset;
    __syncthreads();

    const unsigned int base_compressed_byte_offset = base_index;
    unsigned int current_byte_offset = 0;
    const int base_start_index = warp * thread_chunk * 32;
    for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
        const int block_start =
            base_start_index + chunk * 1024 + lane * 32;
        unsigned int sign_flags = 0;

        unsigned int chunk_bytes =
            fixed_rate[chunk] ? 4 + fixed_rate[chunk] * 4 : 0;
        #pragma unroll 5
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int prior = __shfl_up_sync(0xffffffff, chunk_bytes, delta);
            if (lane >= delta) chunk_bytes += prior;
        }
        const unsigned int previous_thread =
            __shfl_up_sync(0xffffffff, chunk_bytes, 1);
        unsigned int compressed_byte_offset =
            base_compressed_byte_offset + current_byte_offset;
        if (lane != 0) compressed_byte_offset += previous_thread;

        if (fixed_rate[chunk]) {
            uchar4 packed = reinterpret_cast<const uchar4*>(compressed)[
                compressed_byte_offset / 4];
            sign_flags =
                (static_cast<unsigned int>(packed.x) << 24) |
                (static_cast<unsigned int>(packed.y) << 16) |
                (static_cast<unsigned int>(packed.z) << 8) |
                static_cast<unsigned int>(packed.w);
            compressed_byte_offset += 4;

            int absolute_quantized[32];
            #pragma unroll 32
            for (int value = 0; value < 32; ++value) {
                absolute_quantized[value] = 0;
            }
            for (int bit = 0; bit < fixed_rate[chunk]; ++bit) {
                packed = reinterpret_cast<const uchar4*>(compressed)[
                    compressed_byte_offset / 4];
                compressed_byte_offset += 4;
                const unsigned char bytes[4] = {
                    packed.x, packed.y, packed.z, packed.w};
                #pragma unroll 32
                for (int value = 0; value < 32; ++value) {
                    absolute_quantized[value] |=
                        ((bytes[value >> 3] >> (7 - (value & 7))) & 1) << bit;
                }
            }

            #pragma unroll 16
            for (int value = 0; value < 32; value += 2) {
                int first_quantized = absolute_quantized[value];
                int second_quantized = absolute_quantized[value + 1];
                if (sign_flags & (1u << (31 - value))) {
                    first_quantized = -first_quantized;
                }
                if (sign_flags & (1u << (30 - value))) {
                    second_quantized = -second_quantized;
                }
                const int output_index = block_start + value;
                const float first_value =
                    first_quantized * error_bound * 2.0f;
                const float second_value =
                    second_quantized * error_bound * 2.0f;
                if (layer_indices != nullptr) {
                    __nv_bfloat16* const destination =
                        destination_pages[page];
                    if (output_index < num_elements) {
                        const size_t first_destination =
                            selected_bf16_source_index(
                                output_index, layer_indices,
                                selected_layers, source_layers,
                                elements_per_layer);
                        destination[first_destination] =
                            __float2bfloat16_rn(first_value);
                    }
                    if (output_index + 1 < num_elements) {
                        const size_t second_destination =
                            selected_bf16_source_index(
                                output_index + 1, layer_indices,
                                selected_layers, source_layers,
                                elements_per_layer);
                        destination[second_destination] =
                            __float2bfloat16_rn(second_value);
                    }
                } else if (output_index + 1 < num_elements) {
                    if (destination_pages == nullptr ||
                        direct_single_destination != nullptr) {
                        __nv_bfloat16* const contiguous_destination =
                            destination_pages == nullptr
                                ? decoded
                                : direct_single_destination;
                        reinterpret_cast<__nv_bfloat162*>(
                            contiguous_destination)[output_index / 2] =
                                __floats2bfloat162_rn(
                                    first_value, second_value);
                    } else {
                        const size_t destination_index =
                            output_index / elements_per_destination;
                        const size_t destination_offset =
                            output_index % elements_per_destination;
                        __nv_bfloat16* const destination =
                            destination_pages[
                                page * destinations_per_page +
                                destination_index];
                        if ((destination_offset & 1) == 0 &&
                            destination_offset + 1 <
                                elements_per_destination) {
                            reinterpret_cast<__nv_bfloat162*>(destination)[
                                destination_offset / 2] =
                                    __floats2bfloat162_rn(
                                        first_value, second_value);
                        } else {
                            destination[destination_offset] =
                                __float2bfloat16_rn(first_value);
                            const size_t next_index = output_index + 1;
                            const size_t next_destination_index =
                                next_index / elements_per_destination;
                            const size_t next_destination_offset =
                                next_index % elements_per_destination;
                            destination_pages[
                                page * destinations_per_page +
                                next_destination_index][
                                    next_destination_offset] =
                                        __float2bfloat16_rn(second_value);
                        }
                    }
                } else if (output_index < num_elements) {
                    __nv_bfloat16* const destination =
                        destination_pages == nullptr
                            ? decoded + output_index
                            : direct_single_destination != nullptr
                                ? direct_single_destination + output_index
                                : destination_pages[
                                      page * destinations_per_page +
                                      output_index /
                                          elements_per_destination] +
                                      output_index %
                                          elements_per_destination;
                    *destination = __float2bfloat16_rn(first_value);
                }
            }
        }

        current_byte_offset += __shfl_sync(0xffffffff, chunk_bytes, 31);
    }
}

}  // namespace

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
    cudaStream_t stream) {
    const int blocks_per_page = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(blocks_per_page * batch_size);
    const auto typed_inputs =
        reinterpret_cast<const __nv_bfloat16* const*>(input_pages);
    if (use_relative_error) {
        compute_relative_bounds_bf16_kernel<<<
            static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
                typed_inputs, error_bounds, num_elements,
                nullptr, 0, 0, 0);
        const cudaError_t reduction_status = cudaPeekAtLastError();
        if (reduction_status != cudaSuccess) {
            std::fprintf(stderr, "cuSZp BF16 bound reduction failed: %s\n",
                         cudaGetErrorString(reduction_status));
            return false;
        }
    }
    compress_batch_fixed_bf16_kernel<<<grid, block, 0, stream>>>(
        typed_inputs,
        compressed_pages, compressed_offsets, local_offsets, flags,
        error_bounds, num_elements, blocks_per_page,
        nullptr, 0, 0, 0);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cuSZp BF16 compression batch launch failed: %s\n",
                     cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}

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
    cudaStream_t stream) {
    const size_t num_elements =
        prefix_count * selected_layers * elements_per_layer;
    if (num_elements == 0 || source_layers == 0 || selected_layers == 0) {
        return false;
    }
    const int blocks_per_page = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(blocks_per_page * batch_size);
    const auto typed_inputs =
        reinterpret_cast<const __nv_bfloat16* const*>(input_pages);
    if (use_relative_error) {
        compute_relative_bounds_bf16_kernel<<<
            static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
                typed_inputs, error_bounds, num_elements,
                layer_indices, selected_layers, source_layers,
                elements_per_layer);
        const cudaError_t reduction_status = cudaPeekAtLastError();
        if (reduction_status != cudaSuccess) {
            std::fprintf(
                stderr, "cuSZp indexed BF16 bound reduction failed: %s\n",
                cudaGetErrorString(reduction_status));
            return false;
        }
    }
    compress_batch_fixed_bf16_kernel<<<grid, block, 0, stream>>>(
        typed_inputs, compressed_pages, compressed_offsets, local_offsets,
        flags, error_bounds, num_elements, blocks_per_page,
        layer_indices, selected_layers, source_layers, elements_per_layer);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "cuSZp indexed BF16 compression launch failed: %s\n",
            cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}

bool launch_cuszp_decompress_batch_fixed_bf16(
    void* output,
    const unsigned char* const* compressed_pages,
    volatile unsigned int* compressed_offsets,
    volatile unsigned int* local_offsets,
    volatile int* flags,
    const float* error_bounds,
    size_t num_elements,
    size_t batch_size,
    cudaStream_t stream) {
    const int blocks_per_page = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(blocks_per_page * batch_size);
    decompress_batch_fixed_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(output), nullptr, 0, 0,
        compressed_pages,
        compressed_offsets, local_offsets, flags, error_bounds,
        num_elements, blocks_per_page,
        nullptr, 0, 0, 0);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cuSZp BF16 batch launch failed: %s\n",
                     cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}

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
    cudaStream_t stream) {
    const size_t num_elements =
        prefix_count * selected_layers * elements_per_layer;
    const int blocks_per_page = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(blocks_per_page * batch_size);
    decompress_batch_fixed_bf16_kernel<<<grid, block, 0, stream>>>(
        nullptr,
        reinterpret_cast<__nv_bfloat16* const*>(destination_pages),
        1, 0, compressed_pages, compressed_offsets, local_offsets,
        flags, error_bounds, num_elements, blocks_per_page,
        layer_indices, selected_layers, source_layers,
        elements_per_layer);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "cuSZp indexed BF16 scatter launch failed: %s\n",
            cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}
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
    cudaStream_t stream) {
    const int blocks_per_page = static_cast<int>(
        (num_elements + tblock_size * thread_chunk - 1) /
        (tblock_size * thread_chunk));
    const dim3 block(tblock_size);
    const dim3 grid(blocks_per_page * batch_size);
    decompress_batch_fixed_bf16_kernel<<<grid, block, 0, stream>>>(
        nullptr,
        reinterpret_cast<__nv_bfloat16* const*>(destination_pages),
        static_cast<int>(destinations_per_page),
        elements_per_destination, compressed_pages, compressed_offsets,
        local_offsets, flags, error_bounds, num_elements, blocks_per_page,
        nullptr, 0, 0, 0);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cuSZp BF16 direct-scatter launch failed: %s\n",
                     cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}
