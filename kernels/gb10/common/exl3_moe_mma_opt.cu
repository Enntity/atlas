// SPDX-License-Identifier: AGPL-3.0-only

// Optimized EXL3 MoE MMA module: cooperative weight caching
// Module name: exl3_moe_mma_opt
// Entry points:
//   exl3_moe_mul1_matmul_coop_mma    - Cooperative MMA with shared weight cache

#include "exl3_moe_fused.cu"

// Cooperative MMA for MoE (thin experts)
extern "C" __global__ void exl3_moe_mul1_matmul_coop_mma(
    const float* __restrict__ xh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    float* __restrict__ yh,
    const unsigned int* __restrict__ counts,
    unsigned int rows,
    unsigned int n,
    unsigned int k,
    unsigned int topk,
    unsigned int experts,
    unsigned int thin_cap,
    unsigned int tile_k,
    unsigned int tile_n,
    unsigned int slices,
    unsigned int half_partial,
    unsigned int fat_tile_k,
    unsigned int fat_tile_n,
    unsigned int fat_slices,
    unsigned int fat_half_partial
) {
    exl3_moe_fused::exl3_moe_mul1_matmul_coop_mma(
        xh, ids, table, yh, counts,
        rows, n, k, topk, experts,
        thin_cap, tile_k, tile_n, slices, half_partial,
        fat_tile_k, fat_tile_n, fat_slices, fat_half_partial
    );
}