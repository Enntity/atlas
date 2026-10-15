// SPDX-License-Identifier: AGPL-3.0-only

// Optimized EXL3 linear MMA module: fused kernels, persistent launch, cooperative MMA.
// Module name: exl3_linear_mma_opt
// Entry points:
//   exl3_fused_linear              - Fused input hadamard + matmul + output hadamard
//   exl3_persistent_mul1_matmul_mma - Persistent kernel for decode (M=1)
//   exl3_persistent_fused_linear   - Fully fused persistent kernel

#include "exl3_fused_linear.cuh"
#include "exl3_persistent.cuh"

// Re-export original kernels with _opt suffix for comparison
// The original implementations are in exl3_linear_mma.cu

// Fused linear kernel entry point
// Grid: (ceil(N/128), rows, 1), Block: (128, 1, 1)
// Note: This is a placeholder - the real fused kernel needs more work for correctness
extern "C" __global__ void exl3_fused_linear(
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
    // Delegate to the implementation in exl3_fused_linear.cuh
    exl3_fused::exl3_fused_linear(x, suh, trellis, svh, output, rows, n, k, bits);
}

// Persistent MMA kernel for decode
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
    exl3_persistent::exl3_persistent_fused_linear(
        x, suh, trellis, svh, output, rows, n, k, bits
    );
}