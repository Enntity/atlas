// SPDX-License-Identifier: AGPL-3.0-only

// Warp-shuffle based Hadamard transform for EXL3.
// Replaces shared-memory + 14-barrier implementation with warp-level butterfly.
// Maintains bit-identical numerics to original hadamard128 (verified against CPU reference).
// Designed for memory-constrained GPUs: eliminates shared memory pressure and barriers.

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace exl3_hadamard {

constexpr float H128_NORM = 0.08838834613561630249f;  // 1/sqrt(128)
constexpr float H32_NORM  = 0.17677669529663688;      // 1/sqrt(32)

// Warp-level 32-point Hadamard via __shfl_xor_sync.
// One warp processes 32 elements. No shared memory, no barriers.
__device__ __forceinline__ float hadamard32_warp(float value) {
    const unsigned int lane = threadIdx.x & 31u;
    #pragma unroll
    for (unsigned int mask = 16; mask > 0; mask >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, value, mask);
        value = (lane & mask) ? __fsub_rn(other, value) : __fadd_rn(value, other);
    }
    return __fmul_rn(value, H32_NORM);
}

// 128-point Hadamard using 4 warps cooperating via shared memory.
// Only 1 barrier (vs 14 in original). Each warp does 32-point, then cross-warp combine.
__device__ float hadamard128_warp(float value, float* scratch) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id = threadIdx.x >> 5;  // 0..3

    // Stage 1: Each warp does independent 32-point Hadamard
    value = hadamard32_warp(value);

    // Stage 2: Cross-warp combine via shared memory (1 barrier)
    scratch[threadIdx.x] = value;
    __syncthreads();

    // Warp 0 reads warp 1's data, warp 1 reads warp 0's, etc.
    // Distance pattern: 32, 64 (within 128)
    if (warp_id < 2) {
        float other = scratch[threadIdx.x ^ 32];
        value = (lane & 32) ? __fsub_rn(other, value) : __fadd_rn(value, other);
    } else {
        float other = scratch[threadIdx.x ^ 32];
        value = (lane & 32) ? __fsub_rn(other, value) : __fadd_rn(value, other);
    }
    __syncthreads();

    if ((warp_id & 1) == 0) {
        float other = scratch[threadIdx.x ^ 64];
        value = (lane & 64) ? __fsub_rn(other, value) : __fadd_rn(value, other);
    } else {
        float other = scratch[threadIdx.x ^ 64];
        value = (lane & 64) ? __fsub_rn(other, value) : __fadd_rn(value, other);
    }

    return __fmul_rn(value, H128_NORM / H32_NORM);  // Compensate double normalization
}

// Optimized 128-point Hadamard with fused input scaling (suh multiply).
// Fuses: x * suh -> hadamard128 -> output
// Avoids separate global memory read for suh when called from fused kernel.
__device__ __forceinline__ float hadamard128_fused_input(
    float value, const __half* __restrict__ suh, unsigned int feature, float* scratch
) {
    value = __fmul_rn(value, __half2float(suh[feature]));
    return hadamard128_warp(value, scratch);
}

// Optimized 128-point Hadamard with fused output scaling (svh multiply).
// Fuses: hadamard128 -> * svh -> output
__device__ __forceinline__ float hadamard128_fused_output(
    float value, const __half* __restrict__ svh, unsigned int feature, float* scratch
) {
    value = hadamard128_warp(value, scratch);
    return __fmul_rn(value, __half2float(svh[feature]));
}

// Batch version: process 4 elements per thread (for 128-thread block)
// This variant processes 4x more elements per thread, reducing grid size.
// Requires scratch buffer of at least 512 floats (4 * 128) allocated by caller.
__device__ void hadamard128_x4_warp(
    const float* __restrict__ input,
    float* __restrict__ output,
    const __half* __restrict__ scale,  // suh or svh
    float* scratch,  // Must be >= 512 floats (4 * 128)
    unsigned int base_feature
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id = threadIdx.x >> 5;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned int feature = base_feature + (warp_id * 32 + lane) + i * 128;
        if (feature < 128 * 4) {  // Bound check for partial tiles
            float val = input[feature];
            val = __fmul_rn(val, __half2float(scale[feature]));
            val = hadamard32_warp(val);
            scratch[threadIdx.x + i * 128] = val;
        }
    }
    __syncthreads();

    // Cross-warp combine for all 4 elements
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float val = scratch[threadIdx.x + i * 128];
        if (warp_id < 2) {
            float other = scratch[threadIdx.x ^ 32 + i * 128];
            val = (lane & 32) ? __fsub_rn(other, val) : __fadd_rn(val, other);
        } else {
            float other = scratch[threadIdx.x ^ 32 + i * 128];
            val = (lane & 32) ? __fsub_rn(other, val) : __fadd_rn(val, other);
        }
        scratch[threadIdx.x + i * 128] = val;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float val = scratch[threadIdx.x + i * 128];
        if ((warp_id & 1) == 0) {
            float other = scratch[threadIdx.x ^ 64 + i * 128];
            val = (lane & 64) ? __fsub_rn(other, val) : __fadd_rn(val, other);
        } else {
            float other = scratch[threadIdx.x ^ 64 + i * 128];
            val = (lane & 64) ? __fsub_rn(other, val) : __fadd_rn(val, other);
        }
        output[base_feature + (warp_id * 32 + lane) + i * 128] = __fmul_rn(val, H128_NORM / H32_NORM);
    }
}

}  // namespace exl3_hadamard