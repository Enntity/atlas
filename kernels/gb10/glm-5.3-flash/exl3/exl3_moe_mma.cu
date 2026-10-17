// SPDX-License-Identifier: AGPL-3.0-only

// Scoped routed arithmetic. Thin experts use the explicit tensor-core and
// ordered FP16 split-K contract. Fat experts use sequential K16 MMA with
// F32 accumulation and no partial FP16 rounding.
#include "exl3_mma.cuh"
#include "exl3_expert.cuh"

// Grid(ceil(n/128),rows*topk,1), block(128,1,1). Host validates the named
// thin profile32/256/8/1 and fat profile16/128/1/0 before allocation.
// Scratch is F32, with thin outputs explicitly FP16-representable.
extern "C" __global__ void exl3_moe_mul1_matmul_precision_mma(
    const float* __restrict__ xh, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, float* __restrict__ yh,
    const unsigned int* __restrict__ counts,
    unsigned int rows, unsigned int n, unsigned int k, unsigned int topk,
    unsigned int experts, unsigned int thin_cap,
    unsigned int tile_k, unsigned int tile_n, unsigned int slices,
    unsigned int half_partial, unsigned int fat_tile_k, unsigned int fat_tile_n,
    unsigned int fat_slices, unsigned int fat_half_partial) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long output_row = static_cast<unsigned long long>(slot) * n;
    const int expert = ids[slot];
    if (expert < 0 || static_cast<unsigned int>(expert) >= experts
        || tile_k != 32u || tile_n != 256u || slices != 8u || half_partial != 1u
        || fat_tile_k != 16u || fat_tile_n != 128u || fat_slices != 1u
        || fat_half_partial != 0u || k % 128u || n % 256u || thin_cap == 0u) {
        if (output < n) yh[output_row + output] = __int_as_float(0x7fc00000);
        return;
    }
    const Exl3Expert descriptor = table[expert];
    const unsigned int* packed = reinterpret_cast<const unsigned int*>(descriptor.trellis);
    const unsigned long long input_row = static_cast<unsigned long long>(slot) * k;
    const bool fat = counts[expert] > thin_cap;
    const unsigned int selected_k = fat ? fat_tile_k : tile_k;
    const unsigned int selected_n = fat ? fat_tile_n : tile_n;
    const unsigned int selected_slices = fat ? fat_slices : slices;
    const unsigned int selected_half = fat ? fat_half_partial : half_partial;
    const unsigned int lane = threadIdx.x & 31u;
    // Four N8 fragments per warp cover the same N128 block as the scalar ABI.
    // Every lane enters each MMA; only lanes owning the padded actual row store.
    for (unsigned int fragment = 0; fragment < 4u; ++fragment) {
        const unsigned int base_n = blockIdx.x * 128u + (threadIdx.x / 32u) * 8u + fragment * 32u;
        if (base_n >= n) continue;
        float total[4];
        exl3_mma::accumulate(xh, packed, total, input_row, base_n,
            n, k, descriptor.bits, selected_k, selected_n, selected_slices, selected_half);
        if (lane < 4u) {
            const unsigned int column = base_n + (lane & 3u) * 2u;
            yh[output_row + column] = total[0];
            yh[output_row + column + 1u] = total[1];
        }
    }
}

// Slice-parallel variant (exl3_mma_split.cuh): same signature and guards,
// bit-identical output. Grid(n/16,rows*topk,1), block(128,1,1).
#include "exl3_mma_split.cuh"
extern "C" __global__ void exl3_moe_mul1_matmul_precision_mma_split(
    const float* __restrict__ xh, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, float* __restrict__ yh,
    const unsigned int* __restrict__ counts,
    unsigned int rows, unsigned int n, unsigned int k, unsigned int topk,
    unsigned int experts, unsigned int thin_cap,
    unsigned int tile_k, unsigned int tile_n, unsigned int slices,
    unsigned int half_partial, unsigned int fat_tile_k, unsigned int fat_tile_n,
    unsigned int fat_slices, unsigned int fat_half_partial) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;
    const unsigned int block_n = blockIdx.x * 16u;
    const unsigned long long output_row = static_cast<unsigned long long>(slot) * n;
    const int expert = ids[slot];
    if (expert < 0 || static_cast<unsigned int>(expert) >= experts
        || tile_k != 32u || tile_n != 256u || slices != 8u || half_partial != 1u
        || fat_tile_k != 16u || fat_tile_n != 128u || fat_slices != 1u
        || fat_half_partial != 0u || k % 128u || n % 256u || thin_cap == 0u) {
        if (threadIdx.x < 16u && block_n + threadIdx.x < n)
            yh[output_row + block_n + threadIdx.x] = __int_as_float(0x7fc00000);
        return;
    }
    const Exl3Expert descriptor = table[expert];
    const unsigned int* packed = reinterpret_cast<const unsigned int*>(descriptor.trellis);
    const unsigned long long input_row = static_cast<unsigned long long>(slot) * k;
    const bool fat = counts[expert] > thin_cap;
    if (fat)
        exl3_split::block_columns<16u>(xh, packed, yh + output_row, input_row, block_n, n, k,
            descriptor.bits, n, 0u, fat_tile_n, fat_slices, fat_half_partial);
    else
        exl3_split::block_columns<32u>(xh, packed, yh + output_row, input_row, block_n, n, k,
            descriptor.bits, n, 0u, tile_n, slices, half_partial);
}
