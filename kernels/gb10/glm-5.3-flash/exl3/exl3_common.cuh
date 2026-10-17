// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
constexpr float H128_NORM = 0.08838834613561630249f;

__device__ __forceinline__ float hadamard128_scaled(float value, float* scratch, float scale) {
    const unsigned int lane = threadIdx.x;
    #pragma unroll
    for (unsigned int distance = 1; distance < 128; distance <<= 1) {
        scratch[lane] = value;
        __syncthreads();
        const float other = scratch[lane ^ distance];
        value = (lane & distance) ? __fsub_rn(other, value) : __fadd_rn(value, other);
        // A second barrier prevents the next stage overwriting a partner's
        // value while threads from another warp are still reading it.
        __syncthreads();
    }
    return __fmul_rn(value, scale);
}

__device__ __forceinline__ float hadamard128(float value, float* scratch) {
    return hadamard128_scaled(value, scratch, H128_NORM);
}

__device__ __forceinline__ float mul1(unsigned int state) {
    const unsigned int product = state * 0x83dcd12du;
    const unsigned int sum = (product & 255u) + ((product >> 8) & 255u)
        + ((product >> 16) & 255u) + (product >> 24);
    // These FP16 bit patterns are format constants. A single half FMA
    // rounds once, exactly as the CPU codebook oracle requires.
    const __half value = __ushort_as_half(static_cast<unsigned short>(0x6400u + sum));
    return __half2float(__hfma(value, __ushort_as_half(0x1eeeu), __ushort_as_half(0xc931u)));
}

__device__ __forceinline__ unsigned int tile_state(
    const unsigned int* words, unsigned int row, unsigned int column, unsigned int bits
) {
    // Inverse of the packed stream -> 16x16 matrix permutation.
    const unsigned int stream = (column & 7u) * 32u + ((row >> 1) & 3u) * 8u
        + ((column >> 3) & 1u) * 4u + ((row >> 3) & 1u) * 2u + (row & 1u);
    const unsigned int end = (stream + 257u) * bits;
    const unsigned int first = (end - 16u) / 32u;
    const unsigned int last = (end - 1u) / 32u;
    const unsigned int shift = (last + 1u) * 32u - end;
    const unsigned int count = 8u * bits;
    const unsigned long long pair = (static_cast<unsigned long long>(words[first % count]) << 32)
        | words[last % count];
    return static_cast<unsigned int>(pair >> shift) & 65535u;
}
} // namespace
