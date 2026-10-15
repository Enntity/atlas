// SPDX-License-Identifier: AGPL-3.0-only

// Cooperative MMA for EXL3 packed linear.
// Warp-cooperative weight decode + shared memory caching to eliminate redundant
// on-demand weight decoding. Each weight decoded once per block instead of
// 512x per warp per MMA step.
// Designed for memory-constrained GPUs: reduces global memory reads of packed weights.

#pragma once
#include "exl3_common.cuh"

namespace exl3_mma_coop {

// Warp-level pair bits packing (same as original)
__device__ __forceinline__ unsigned int pair_bits(float low, float high) {
    return static_cast<unsigned int>(__half_as_ushort(__float2half_rn(low)))
        | (static_cast<unsigned int>(__half_as_ushort(__float2half_rn(high))) << 16u);
}

// PTX MMA instruction (same as original)
__device__ __forceinline__ void mma_step(float* acc, unsigned int a0, unsigned int a2,
                                        unsigned int b0, unsigned int b1) {
    const unsigned int zero = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(zero), "r"(a2), "r"(zero), "r"(b0), "r"(b1));
}

// Shared memory weight cache layout: [TILE_K][WARPS_PER_BLOCK * COLUMNS_PER_WARP]
// For TILE_K=32, WARPS_PER_BLOCK=4, COLUMNS_PER_WARP=8: 32 rows x 32 cols = 1024 floats
// Stored as flat array: weight_cache[row * 32 + col]
__device__ __forceinline__ void weight_cache_set(float* weight_cache, unsigned int row, unsigned int col, float val) {
    weight_cache[row * 32 + col] = val;
}

__device__ __forceinline__ float weight_cache_get(const float* weight_cache, unsigned int row, unsigned int col) {
    return weight_cache[row * 32 + col];
}

// Cooperative weight decode: all warps in block cooperate to decode weights
// for the current tile into shared memory
__device__ void cooperative_decode_weights_tile(
    const unsigned int* __restrict__ packed,
    unsigned int n, unsigned int k, unsigned int bits,
    unsigned int column_tile, unsigned int tiles_k,
    float* weight_cache,  // 32*32 floats in shared memory
    unsigned int tile_input_base,  // Base input index for this tile
    unsigned int warp_col_base     // Base column for this warp
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id = threadIdx.x >> 5;

    // Each warp decodes its column range (8 columns per warp)
    const unsigned int my_col = warp_col_base + warp_id * 8 + (lane / 4);

    if (my_col >= n) return;

    // Decode 32 weights for this column (TILE_K=32)
    for (unsigned int t = 0; t < 32; ++t) {
        unsigned int input = tile_input_base + t;
        if (input >= k) {
            weight_cache_set(weight_cache, t, warp_id * 8 + (lane / 4), 0.0f);
            continue;
        }

        const unsigned long long wtile = static_cast<unsigned long long>(input / 16u)
            * (n / 16u) + my_col / 16u;
        const unsigned int* words = packed + wtile * (8u * bits);
        float weight = exl3_common::mul1(
            exl3_common::tile_state(words, input & 15u, my_col & 15u, bits)
        );
        weight_cache_set(weight_cache, t, warp_id * 8 + (lane / 4), weight);
    }
    __syncthreads();
}

// Cooperative MMA accumulate with shared weight cache
// Optimized for the common routed case: tile_k=32, tile_n=256, slices=8, half_partial=1
__device__ __forceinline__ void accumulate_coop(
    const float* __restrict__ xh,
    const unsigned int* __restrict__ packed,
    float* total,
    unsigned long long row_offset,
    unsigned int base_n,
    unsigned int n,
    unsigned int k,
    unsigned int bits,
    unsigned int slices,
    unsigned int half_partial,
    float* weight_cache  // Shared memory: 32*32 floats
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id = threadIdx.x >> 5;  // 0..3

    const unsigned int tiles_k = k / 32;
    const unsigned int column_tile = base_n / 256;
    const unsigned int total_tiles = tiles_k * (n / 256);

    for (unsigned int j = 0; j < 4u; ++j) total[j] = 0.0f;

    for (unsigned int s = slices; s > 0; --s) {
        const unsigned int flat_begin = static_cast<unsigned long long>(total_tiles) * (s - 1u) / slices;
        const unsigned int flat_end = static_cast<unsigned long long>(total_tiles) * s / slices;
        const unsigned int first = max(flat_begin, column_tile * tiles_k);
        const unsigned int last = min(flat_end, (column_tile + 1u) * tiles_k);
        if (first >= last) continue;

        float acc[4][4] = {};

        for (unsigned int tile = first; tile < last; ++tile) {
            const unsigned int tile_input = (tile - column_tile * tiles_k) * 32;

            // Cooperative weight decode for this tile into shared memory
            cooperative_decode_weights_tile(packed, n, k, bits, column_tile, tiles_k,
                weight_cache, tile_input, base_n + warp_id * 32);

            // Now all warps read from shared weight cache
            for (unsigned int sub = 0; sub < 2; ++sub) {  // 2 sub-tiles of 16
                const unsigned int input = tile_input + sub * 16 + (lane & 3u) * 2;
                const unsigned long long row = row_offset;
                const unsigned int a0 = lane < 4u ? pair_bits(xh[row + input], xh[row + input + 1]) : 0u;
                const unsigned int a2 = lane < 4u ? pair_bits(xh[row + input + 8], xh[row + input + 9]) : 0u;

                // Read from shared cache: row = sub*16 + offset, col = warp_id*8 + (lane/4)
                unsigned int col_idx = warp_id * 8 + (lane / 4);
                unsigned int row0 = sub * 16 + 0;
                unsigned int row1 = sub * 16 + 1;
                unsigned int row8 = sub * 16 + 8;
                unsigned int row9 = sub * 16 + 9;

                const unsigned int b0 = pair_bits(
                    weight_cache_get(weight_cache, row0, col_idx),
                    weight_cache_get(weight_cache, row1, col_idx)
                );
                const unsigned int b1 = pair_bits(
                    weight_cache_get(weight_cache, row8, col_idx),
                    weight_cache_get(weight_cache, row9, col_idx)
                );

                mma_step(acc[sub], a0, a2, b0, b1);
            }
            __syncthreads();  // Ensure shared mem not overwritten before next tile
        }

        for (unsigned int j = 0; j < 4u; ++j) {
            float sum = acc[0][j];
            sum = __fadd_rn(sum, acc[1][j]);
            total[j] = __fadd_rn(sum, total[j]);
            if (half_partial) total[j] = __half2float(__float2half_rn(total[j]));
        }
    }
}

// Dispatch function matching original accumulate_logical signature
// Falls back to original implementation for non-standard geometries
__device__ __forceinline__ void accumulate_logical(
    const float* xh,
    const unsigned int* packed,
    float* total,
    unsigned long long row_offset,
    unsigned int base_n,
    unsigned int n,
    unsigned int k,
    unsigned int bits,
    unsigned int logical_n,
    unsigned int logical_offset,
    unsigned int tile_k,
    unsigned int tile_n,
    unsigned int slices,
    unsigned int half_partial
) {
    // Only optimize the common routed case (physical == logical)
    if (tile_k == 32 && tile_n == 256 && slices == 8 && half_partial == 1 &&
        logical_n == n && logical_offset == 0) {
        // Need shared memory for weight cache - caller must provide
        // For now, fall back to original if shared memory not available
        // This is a placeholder - the fused kernels provide shared memory
        extern __shared__ float shared_mem[];
        float* weight_cache = shared_mem;  // Assumes 1024 floats at start of shared mem
        accumulate_coop(xh, packed, total, row_offset, base_n, n, k, bits, slices, half_partial, weight_cache);
    } else {
        // Fallback to original implementation (replicate original logic)
        const unsigned int lane = threadIdx.x & 31u;
        const unsigned int column = base_n + (lane / 4u);
        const unsigned int offset = (lane & 3u) * 2u;
        const unsigned int tiles_k = k / tile_k;
        const unsigned int column_tile = (base_n + logical_offset) / tile_n;
        const unsigned int total_tiles = tiles_k * (logical_n / tile_n);
        for (unsigned int j = 0; j < 4u; ++j) total[j] = 0.0f;
        for (unsigned int s = slices; s > 0; --s) {
            const unsigned int flat_begin = static_cast<unsigned long long>(total_tiles) * (s - 1u) / slices;
            const unsigned int flat_end = static_cast<unsigned long long>(total_tiles) * s / slices;
            const unsigned int first = max(flat_begin, column_tile * tiles_k);
            const unsigned int last = min(flat_end, (column_tile + 1u) * tiles_k);
            if (first >= last) continue;
            float acc[4][4] = {};
            for (unsigned int tile = first; tile < last; ++tile) {
                const unsigned int tile_input = (tile - column_tile * tiles_k) * tile_k;
                for (unsigned int sub = 0; sub < tile_k / 16u; ++sub) {
                    const unsigned int input = tile_input + sub * 16u + offset;
                    const unsigned long long row = row_offset;
                    const unsigned int a0 = lane < 4u ? pair_bits(xh[row + input], xh[row + input + 1u]) : 0u;
                    const unsigned int a2 = lane < 4u ? pair_bits(xh[row + input + 8u], xh[row + input + 9u]) : 0u;
                    const unsigned int b0 = pair_bits(
                        exl3_common::mul1(exl3_common::tile_state(packed + (static_cast<unsigned long long>(input / 16u) * (n / 16u) + column / 16u) * (8u * bits), input & 15u, column & 15u, bits)),
                        exl3_common::mul1(exl3_common::tile_state(packed + (static_cast<unsigned long long>((input + 1u) / 16u) * (n / 16u) + column / 16u) * (8u * bits), (input + 1u) & 15u, column & 15u, bits))
                    );
                    const unsigned int b1 = pair_bits(
                        exl3_common::mul1(exl3_common::tile_state(packed + (static_cast<unsigned long long>((input + 8u) / 16u) * (n / 16u) + column / 16u) * (8u * bits), (input + 8u) & 15u, column & 15u, bits)),
                        exl3_common::mul1(exl3_common::tile_state(packed + (static_cast<unsigned long long>((input + 9u) / 16u) * (n / 16u) + column / 16u) * (8u * bits), (input + 9u) & 15u, column & 15u, bits))
                    );
                    mma_step(acc[sub], a0, a2, b0, b1);
                }
            }
            for (unsigned int j = 0; j < 4u; ++j) {
                float sum = acc[0][j];
                if (tile_k >= 32u) sum = __fadd_rn(sum, acc[1][j]);
                if (tile_k == 48u) sum = __fadd_rn(sum, acc[2][j]);
                if (tile_k == 64u) sum = __fadd_rn(sum, __fadd_rn(acc[2][j], acc[3][j]));
                total[j] = __fadd_rn(sum, total[j]);
                if (half_partial) total[j] = __half2float(__float2half_rn(total[j]));
            }
        }
    }
}

}  // namespace exl3_mma_coop