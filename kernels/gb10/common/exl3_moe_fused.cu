// SPDX-License-Identifier: AGPL-3.0-only

// Fused MoE EXL3 kernels: optimized variants with warp-shuffle hadamard
// These are drop-in replacements for the original exl3_moe_precision kernels
// using the optimized hadamard transform.

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "exl3_common.cuh"
#include "exl3_expert.cuh"
#include "exl3_hadamard_warp.cuh"

namespace exl3_moe_fused {

// Fused MoE input hadamard (FP16 intermediate) with warp-shuffle hadamard
// Grid: (K/128, rows*topk, 1), Block: (128, 1, 1)
// Same signature as exl3_moe_input_hadamard_f16 but uses warp hadamard
extern "C" __global__ void exl3_moe_fused_input_hadamard_f16(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    float* __restrict__ xh,
    unsigned int rows,
    unsigned int k,
    unsigned int topk,
    unsigned int expanded,
    unsigned int experts
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;

    const unsigned int feature = blockIdx.x * 128 + threadIdx.x;
    if (feature >= k) return;

    const int expert = ids[slot];
    const bool valid = expert >= 0 && static_cast<unsigned int>(expert) < experts;
    if (!valid) {
        xh[static_cast<unsigned long long>(slot) * k + feature] = __int_as_float(0x7fc00000);
        return;
    }

    const Exl3Expert descriptor = table[expert];
    const __half* suh = reinterpret_cast<const __half*>(descriptor.suh);
    const unsigned int input_row = expanded ? slot : slot / topk;
    const unsigned long long in_index = static_cast<unsigned long long>(input_row) * k + feature;
    const unsigned long long out_index = static_cast<unsigned long long>(slot) * k + feature;

    const __half product = __hmul(__float2half_rn(__bfloat162float(x[in_index])), suh[feature]);
    __shared__ float scratch[128];
    xh[out_index] = exl3_hadamard::hadamard128_warp(__half2float(product), scratch);
}

// Fused MoE output hadamard with FP32 output using warp hadamard
// Grid: (N/128, rows*topk, 1), Block: (128, 1, 1)
extern "C" __global__ void exl3_moe_fused_output_hadamard_precision(
    const float* __restrict__ yh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    float* __restrict__ output,
    const unsigned int* __restrict__ counts,
    unsigned int rows,
    unsigned int n,
    unsigned int topk,
    unsigned int experts,
    unsigned int thin_cap
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;

    const unsigned int feature = blockIdx.x * 128 + threadIdx.x;
    if (feature >= n) return;

    const int expert = ids[slot];
    const bool valid = expert >= 0 && static_cast<unsigned int>(expert) < experts;
    if (!valid) {
        output[static_cast<unsigned long long>(slot) * n + feature] = __int_as_float(0x7fc00000);
        return;
    }

    const Exl3Expert descriptor = table[expert];
    const __half* svh = reinterpret_cast<const __half*>(descriptor.svh);

    __shared__ float scratch[128];
    unsigned long long index = static_cast<unsigned long long>(slot) * n + feature;

    float value = exl3_hadamard::hadamard128_warp(yh[index], scratch);
    value = __fmul_rn(value, __half2float(svh[feature]));

    // Apply thin expert rounding if needed (matches original policy)
    if (counts[expert] <= thin_cap) {
        value = __half2float(__float2half_rn(value));
    }
    output[index] = value;
}

// Fused activate + down hadamard with warp hadamard
// Grid: (K/128, rows*topk, 1), Block: (128, 1, 1)
extern "C" __global__ void exl3_moe_fused_activate_down_hadamard(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ down_table,
    float* __restrict__ output,
    const unsigned int* __restrict__ counts,
    unsigned int rows,
    unsigned int k,
    unsigned int topk,
    unsigned int experts,
    unsigned int thin_cap,
    float limit,
    unsigned int clamp
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;

    const unsigned int feature = blockIdx.x * 128 + threadIdx.x;
    if (feature >= k) return;

    const int expert = ids[slot];
    if (expert < 0 || static_cast<unsigned int>(expert) >= experts) {
        output[static_cast<unsigned long long>(slot) * k + feature] = __int_as_float(0x7fc00000);
        return;
    }

    unsigned long long index = static_cast<unsigned long long>(slot) * k + feature;
    float g = gate[index];
    float u = up[index];

    // SiLU activation (matching original policy)
    float activated;
    if (counts[expert] <= thin_cap) {
        // Thin: half2 SiLU with post-SiLU clamp
        activated = __bfloat162float(__float2bfloat16_rn(
            __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)))
        ));
        if (clamp) activated = fminf(activated, limit);
        u = fminf(fmaxf(u, -limit), limit);
        activated = __fmul_rn(activated, u);
    } else {
        // Fat: F32 SiLU with pre-SiLU clamp
        if (clamp) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        activated = __fmul_rn(__fmul_rn(sigmoid, g), u);
    }

    // Down-projection input hadamard with suh scaling
    const Exl3Expert descriptor = down_table[expert];
    const __half* suh = reinterpret_cast<const __half*>(descriptor.suh);

    __shared__ float scratch[128];
    float val = __fmul_rn(activated, __half2float(suh[feature]));
    val = exl3_hadamard::hadamard128_warp(val, scratch);

    // Apply thin expert rounding
    if (counts[expert] <= thin_cap) {
        val = __half2float(__float2half_rn(val));
    }
    output[index] = val;
}

// Fused down weighted output with warp hadamard
// Grid: (N/128, rows*topk, 1), Block: (128, 1, 1)
extern "C" __global__ void exl3_moe_fused_down_weighted_output(
    const float* __restrict__ yh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const unsigned int* __restrict__ counts,
    unsigned int rows,
    unsigned int n,
    unsigned int topk,
    unsigned int experts,
    unsigned int thin_cap
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;

    const unsigned int feature = blockIdx.x * 128 + threadIdx.x;
    if (feature >= n) return;

    const int expert = ids[slot];
    if (expert < 0 || static_cast<unsigned int>(expert) >= experts) {
        output[static_cast<unsigned long long>(slot) * n + feature] = __int_as_float(0x7fc00000);
        return;
    }

    const Exl3Expert descriptor = table[expert];
    const __half* svh = reinterpret_cast<const __half*>(descriptor.svh);

    __shared__ float scratch[128];
    unsigned long long index = static_cast<unsigned long long>(slot) * n + feature;

    float weight = __half2float(__float2half_rn(weights[slot]));
    float value = exl3_hadamard::hadamard128_warp(yh[index], scratch);
    value = __fmul_rn(value, __half2float(svh[feature]));
    value = __fmul_rn(value, weight);

    // Apply thin expert rounding
    if (counts[expert] <= thin_cap) {
        value = __half2float(__float2half_rn(value));
    }
    output[index] = value;
}

// Cooperative MMA for MoE (thin experts)
// Uses shared weight cache across warps in block
// Grid: (ceil(N/128), rows*topk, 1), Block: (128, 1, 1)
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
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk) return;

    const unsigned int output = blockIdx.x * 128 + threadIdx.x;
    if (output >= n) return;

    const unsigned long long output_row = static_cast<unsigned long long>(slot) * n;
    const int expert = ids[slot];
    if (expert < 0 || static_cast<unsigned int>(expert) >= experts) {
        yh[output_row + output] = __int_as_float(0x7fc00000);
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

    // Use cooperative MMA for thin experts (common case)
    if (!fat && tile_k == 32 && tile_n == 256 && slices == 8 && half_partial == 1) {
        for (unsigned int fragment = 0; fragment < 4u; ++fragment) {
            const unsigned int base_n = blockIdx.x * 128u + (threadIdx.x / 32u) * 8u + fragment * 32u;
            if (base_n >= n) continue;

            float total[4];
            // Shared memory for weight cache: 32*32 floats = 4KB
            extern __shared__ float weight_cache[];
            exl3_mma_coop::accumulate_coop<32, 256, 4>(
                xh, packed, total,
                input_row, base_n, n, k, descriptor.bits,
                selected_slices, selected_half,
                weight_cache
            );

            if (lane < 4u) {
                const unsigned int column = base_n + (lane & 3u) * 2u;
                yh[output_row + column] = total[0];
                yh[output_row + column + 1u] = total[1];
            }
        }
    } else {
        // Fat expert or non-standard geometry: fallback to original
        for (unsigned int fragment = 0; fragment < 4u; ++fragment) {
            const unsigned int base_n = blockIdx.x * 128u + (threadIdx.x / 32u) * 8u + fragment * 32u;
            if (base_n >= n) continue;

            float total[4];
            exl3_mma_coop::accumulate_logical(
                xh, packed, total, input_row, base_n, n, k, descriptor.bits,
                n, 0u, selected_k, selected_n, selected_slices, selected_half
            );

            if (lane < 4u) {
                const unsigned int column = base_n + (lane & 3u) * 2u;
                yh[output_row + column] = total[0];
                yh[output_row + column + 1u] = total[1];
            }
        }
    }
}

}  // namespace exl3_moe_fused