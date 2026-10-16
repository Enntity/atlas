// SPDX-License-Identifier: AGPL-3.0-only

// Explicit generic packed-linear F32 MMA partials. Decoding uses the physical
// packed matrix width; split ranges use the full fused reference geometry.
// Grid(ceil(physical_n/128),rows,1), block(128,1,1).
#include "exl3_mma.cuh"

extern "C" __global__ void exl3_mul1_matmul_mma(
    const float* __restrict__ xh, const unsigned int* __restrict__ trellis,
    float* __restrict__ yh, unsigned int rows, unsigned int physical_n,
    unsigned int k, unsigned int bits, unsigned int logical_n,
    unsigned int logical_offset, unsigned int tile_k, unsigned int tile_n,
    unsigned int slices, unsigned int half_partial) {
    const unsigned int row = blockIdx.y;
    if (row >= rows) return;
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long output_row = static_cast<unsigned long long>(row) * physical_n;
    // The host validates an explicit named profile before allocating scratch.
    // These guards prevent unsupported geometry from returning plausible data.
    if (physical_n == 0u || physical_n % 128u || k == 0u || k % 128u
        || bits < 1u || bits > 8u || (tile_k != 16u && tile_k != 32u)
        || (tile_n != 128u && tile_n != 256u && tile_n != 512u)
        || (tile_k == 32u && tile_n == 512u) || (tile_k == 16u && tile_n == 256u)
        || logical_n == 0u || logical_n % tile_n || logical_offset % 128u
        || logical_offset > logical_n || physical_n > logical_n - logical_offset
        || slices == 0u || slices > 48u || half_partial != 0u) {
        if (output < physical_n) yh[output_row + output] = __int_as_float(0x7fc00000);
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    for (unsigned int fragment = 0; fragment < 4u; ++fragment) {
        const unsigned int base_n = blockIdx.x * 128u + (threadIdx.x / 32u) * 8u + fragment * 32u;
        if (base_n >= physical_n) continue;
        float total[4];
        exl3_mma::accumulate_logical(xh, trellis, total,
            static_cast<unsigned long long>(row) * k, base_n,
            physical_n, k, bits, logical_n, logical_offset,
            tile_k, tile_n, slices, half_partial);
        if (lane < 4u) {
            const unsigned int column = base_n + (lane & 3u) * 2u;
            yh[output_row + column] = total[0];
            yh[output_row + column + 1u] = total[1];
        }
    }
}

// Slice-parallel variant (exl3_mma_split.cuh): same signature and guards,
// bit-identical output. Grid(physical_n/16,rows,1), block(128,1,1).
#include "exl3_mma_split.cuh"
extern "C" __global__ void exl3_mul1_matmul_mma_split(
    const float* __restrict__ xh, const unsigned int* __restrict__ trellis,
    float* __restrict__ yh, unsigned int rows, unsigned int physical_n,
    unsigned int k, unsigned int bits, unsigned int logical_n,
    unsigned int logical_offset, unsigned int tile_k, unsigned int tile_n,
    unsigned int slices, unsigned int half_partial) {
    const unsigned int row = blockIdx.y;
    if (row >= rows) return;
    const unsigned int block_n = blockIdx.x * 16u;
    const unsigned long long output_row = static_cast<unsigned long long>(row) * physical_n;
    if (physical_n == 0u || physical_n % 128u || k == 0u || k % 128u
        || bits < 1u || bits > 8u || (tile_k != 16u && tile_k != 32u)
        || (tile_n != 128u && tile_n != 256u && tile_n != 512u)
        || (tile_k == 32u && tile_n == 512u) || (tile_k == 16u && tile_n == 256u)
        || logical_n == 0u || logical_n % tile_n || logical_offset % 128u
        || logical_offset > logical_n || physical_n > logical_n - logical_offset
        || slices == 0u || slices > exl3_split::MAX_SLICES || half_partial != 0u) {
        if (threadIdx.x < 16u && block_n + threadIdx.x < physical_n)
            yh[output_row + block_n + threadIdx.x] = __int_as_float(0x7fc00000);
        return;
    }
    if (tile_k == 16u)
        exl3_split::block_columns<16u>(xh, trellis, yh + output_row, static_cast<unsigned long long>(row) * k,
            block_n, physical_n, k, bits, logical_n, logical_offset, tile_n, slices, half_partial);
    else
        exl3_split::block_columns<32u>(xh, trellis, yh + output_row, static_cast<unsigned long long>(row) * k,
            block_n, physical_n, k, bits, logical_n, logical_offset, tile_n, slices, half_partial);
}
