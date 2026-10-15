// SPDX-License-Identifier: AGPL-3.0-only

// Optimized EXL3 persistent kernels module.
// Module name: exl3_persistent_opt
// Entry points:
//   exl3_persistent_mul1_matmul_mma
//   exl3_persistent_mul1_matmul_scalar
//   exl3_persistent_output_hadamard
//   exl3_persistent_fused_linear

#include "exl3_persistent.cuh"

// Persistent MMA kernel for decode (M=1)
extern "C" __global__ void exl3_persistent_mul1_matmul_mma(
    const float* __restrict__ xh,
    const unsigned int* __restrict__ trellis,
    float* __restrict__ yh,
    unsigned int rows,
    unsigned int n,
    unsigned int k,
    unsigned int bits,
    unsigned int tile_k,
    unsigned int tile_n,
    unsigned int slices,
    unsigned int half_partial
) {
    exl3_persistent::exl3_persistent_mul1_matmul_mma(
        xh, trellis, yh, rows, n, k, bits, tile_k, tile_n, slices, half_partial
    );
}

// Persistent scalar matmul (fallback)
extern "C" __global__ void exl3_persistent_mul1_matmul_scalar(
    const float* __restrict__ xh,
    const unsigned int* __restrict__ trellis,
    float* __restrict__ yh,
    unsigned int rows,
    unsigned int n,
    unsigned int k,
    unsigned int bits
) {
    exl3_persistent::exl3_persistent_mul1_matmul_scalar(xh, trellis, yh, rows, n, k, bits);
}

// Persistent output hadamard
extern "C" __global__ void exl3_persistent_output_hadamard(
    const float* __restrict__ yh,
    const __half* __restrict__ svh,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int n
) {
    exl3_persistent::exl3_persistent_output_hadamard(yh, svh, output, rows, n);
}

// Fully fused persistent linear kernel
extern "C" __global__ void exl3_persistent_fused_linear(
    const __nv_bfloat16* __restrict__ x,
    const __half* __restrict__ suh,
    const unsigned int* __restrict__ trellis,
    const __half* __restrict__ svh,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int n,
    unsigned int k,
    unsigned int bits
) {
    exl3_persistent::exl3_persistent_fused_linear(x, suh, trellis, svh, output, rows, n, k, bits);
}