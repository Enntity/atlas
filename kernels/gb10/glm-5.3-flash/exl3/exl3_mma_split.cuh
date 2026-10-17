// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Slice-parallel variant of exl3_mma::accumulate_logical.
//
// The serial kernel walks every split-K slice of a fragment inside one warp.
// Here each (fragment, slice) pair is computed by its own warp with exactly the
// same per-slice MMA chain, the per-slice sums land in shared memory, and one
// thread per output column folds them in the original order (s = slices..1,
// empty slices skipped, __fadd_rn, optional FP16 rounding after every add).
// Inputs, MMA order, per-slice partial sums and the fold are identical, so the
// output is bit-identical to accumulate_logical; only the parallel schedule
// changes. TK is the tile_k (16 or 32), a template so acc[] stays in registers.
#include "exl3_mma.cuh"

namespace exl3_split {
constexpr unsigned int FRAGMENTS = 2;   // N8 fragments per block (16 columns)
constexpr unsigned int WARPS = 4;       // block = 128 threads
constexpr unsigned int MAX_SLICES = 48;

struct Range {
    unsigned int first;
    unsigned int last;
};

// Identical arithmetic to accumulate_logical's slice bounds.
__device__ __forceinline__ Range slice_range(unsigned int s, unsigned int total_tiles,
                                             unsigned int slices, unsigned int column_tile,
                                             unsigned int tiles_k) {
    const unsigned int flat_begin = static_cast<unsigned long long>(total_tiles) * (s - 1u) / slices;
    const unsigned int flat_end = static_cast<unsigned long long>(total_tiles) * s / slices;
    return {max(flat_begin, column_tile * tiles_k), min(flat_end, (column_tile + 1u) * tiles_k)};
}

// One slice's `sum` for row 0 of the M16 tile (lanes < 4 own it): same loop
// body as accumulate_logical, accumulators held in registers.
template <unsigned int TK>
__device__ __forceinline__ void slice_sum(const float* xh, const unsigned int* packed, float* out,
                                          unsigned long long row_offset, unsigned int base_n,
                                          unsigned int n, unsigned int bits, unsigned int first,
                                          unsigned int last, unsigned int column_tile,
                                          unsigned int tiles_k) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int column = base_n + (lane / 4u);
    const unsigned int offset = (lane & 3u) * 2u;
    float acc[TK / 16u][4];
#pragma unroll
    for (unsigned int sub = 0; sub < TK / 16u; ++sub)
#pragma unroll
        for (unsigned int j = 0; j < 4u; ++j) acc[sub][j] = 0.0f;
    for (unsigned int tile = first; tile < last; ++tile) {
        const unsigned int tile_input = (tile - column_tile * tiles_k) * TK;
#pragma unroll
        for (unsigned int sub = 0; sub < TK / 16u; ++sub) {
            const unsigned int input = tile_input + sub * 16u + offset;
            const unsigned long long row = row_offset;
            const unsigned int a0 = lane < 4u ? exl3_mma::pair_bits(xh[row + input], xh[row + input + 1u]) : 0u;
            const unsigned int a2 = lane < 4u ? exl3_mma::pair_bits(xh[row + input + 8u], xh[row + input + 9u]) : 0u;
            const unsigned int b0 = exl3_mma::pair_bits(exl3_mma::decoded(packed, input, column, n, bits),
                                                        exl3_mma::decoded(packed, input + 1u, column, n, bits));
            const unsigned int b1 = exl3_mma::pair_bits(exl3_mma::decoded(packed, input + 8u, column, n, bits),
                                                        exl3_mma::decoded(packed, input + 9u, column, n, bits));
            exl3_mma::mma_step(acc[sub], a0, a2, b0, b1);
        }
    }
#pragma unroll
    for (unsigned int j = 0; j < 2u; ++j) {
        float sum = acc[0][j];
        if (TK >= 32u) sum = __fadd_rn(sum, acc[TK >= 32u ? 1u : 0u][j]);
        out[j] = sum;
    }
}

// Computes 16 output columns [block_n, block_n+16) of one row into yh_row.
// Caller has validated geometry (tile_k in {16,32}, slices <= MAX_SLICES).
template <unsigned int TK>
__device__ __forceinline__ void block_columns(const float* xh, const unsigned int* packed,
                                              float* yh_row, unsigned long long row_offset,
                                              unsigned int block_n, unsigned int n, unsigned int k,
                                              unsigned int bits, unsigned int logical_n,
                                              unsigned int logical_offset, unsigned int tile_n,
                                              unsigned int slices, unsigned int half_partial) {
    __shared__ float partial[FRAGMENTS][MAX_SLICES][8];
    const unsigned int warp = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int tiles_k = k / TK;
    const unsigned int total_tiles = tiles_k * (logical_n / tile_n);
    // Pair p -> (fragment p % 2, slice slices - p / 2): a slice's two fragments
    // are adjacent, so the contiguous non-empty slice range spreads over warps.
    for (unsigned int p = warp; p < FRAGMENTS * slices; p += WARPS) {
        const unsigned int f = p % FRAGMENTS;
        const unsigned int s = slices - p / FRAGMENTS;
        const unsigned int base_n = block_n + f * 8u;
        if (base_n >= n) continue;
        const unsigned int column_tile = (base_n + logical_offset) / tile_n;
        const Range r = slice_range(s, total_tiles, slices, column_tile, tiles_k);
        if (r.first >= r.last) continue;
        float out[2];
        slice_sum<TK>(xh, packed, out, row_offset, base_n, n, bits, r.first, r.last, column_tile, tiles_k);
        if (lane < 4u) {
            partial[f][s - 1u][lane * 2u] = out[0];
            partial[f][s - 1u][lane * 2u + 1u] = out[1];
        }
    }
    __syncthreads();
    if (threadIdx.x >= FRAGMENTS * 8u) return;
    const unsigned int f = threadIdx.x / 8u;
    const unsigned int c = threadIdx.x & 7u;
    const unsigned int base_n = block_n + f * 8u;
    if (base_n >= n) return;
    const unsigned int column_tile = (base_n + logical_offset) / tile_n;
    float total = 0.0f;
    for (unsigned int s = slices; s > 0; --s) {
        const Range r = slice_range(s, total_tiles, slices, column_tile, tiles_k);
        if (r.first >= r.last) continue;
        total = __fadd_rn(partial[f][s - 1u][c], total);
        if (half_partial) total = __half2float(__float2half_rn(total));
    }
    yh_row[base_n + c] = total;
}
}  // namespace exl3_split
