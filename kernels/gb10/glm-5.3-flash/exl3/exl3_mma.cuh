// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Native on-demand packed MMA. NVIDIA PTX m16n8k16 F16 fragment mapping:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
#include "exl3_common.cuh"
namespace exl3_mma {
__device__ unsigned int pair_bits(float low, float high) {
    return static_cast<unsigned int>(__half_as_ushort(__float2half_rn(low)))
        | (static_cast<unsigned int>(__half_as_ushort(__float2half_rn(high))) << 16u);
}

__device__ float decoded(const unsigned int* packed, unsigned int input,
                         unsigned int output, unsigned int n, unsigned int bits) {
    const unsigned long long tile = static_cast<unsigned long long>(input / 16u)
        * (n / 16u) + output / 16u;
    return mul1(tile_state(packed + tile * (8u * bits), input & 15u, output & 15u, bits));
}

__device__ void mma_step(float* acc, unsigned int a0, unsigned int a2,
                        unsigned int b0, unsigned int b1) {
    const unsigned int zero = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(zero), "r"(a2), "r"(zero), "r"(b0), "r"(b1));
}

// All warp lanes participate. One actual row is padded to M16; only lane<4
// owns that row's C0/C1 outputs. Geometry and rounding policy are explicit.
__device__ __forceinline__ void accumulate_logical(
    const float* xh, const unsigned int* packed, float* total,
    unsigned long long row_offset, unsigned int base_n,
    unsigned int n, unsigned int k, unsigned int bits,
    unsigned int logical_n, unsigned int logical_offset, unsigned int tile_k, unsigned int tile_n, unsigned int slices,
    unsigned int half_partial) {
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
                const unsigned int b0 = pair_bits(decoded(packed,input,column,n,bits),
                                                  decoded(packed,input+1u,column,n,bits));
                const unsigned int b1 = pair_bits(decoded(packed,input+8u,column,n,bits),
                                                  decoded(packed,input+9u,column,n,bits));
                mma_step(acc[sub],a0,a2,b0,b1);
            }
        }
        for (unsigned int j = 0; j < 4u; ++j) {
            float sum = acc[0][j];
            if (tile_k >= 32u) sum = __fadd_rn(sum,acc[1][j]);
            if (tile_k == 48u) sum = __fadd_rn(sum,acc[2][j]);
            if (tile_k == 64u) sum = __fadd_rn(sum,__fadd_rn(acc[2][j],acc[3][j]));
            total[j] = __fadd_rn(sum,total[j]);
            if (half_partial) total[j] = __half2float(__float2half_rn(total[j]));
        }
    }
}
// Routed matrices have identical physical and logical geometry. Preserve this
// entry point and its explicit rounding policy for the existing expert kernels.
__device__ __forceinline__ void accumulate(
    const float* xh, const unsigned int* packed, float* total,
    unsigned long long row_offset, unsigned int base_n,
    unsigned int n, unsigned int k, unsigned int bits,
    unsigned int tile_k, unsigned int tile_n, unsigned int slices,
    unsigned int half_partial) {
    accumulate_logical(xh, packed, total, row_offset, base_n, n, k, bits,
        n, 0u, tile_k, tile_n, slices, half_partial);
}

}
