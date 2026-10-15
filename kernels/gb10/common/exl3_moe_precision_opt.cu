// SPDX-License-Identifier: AGPL-3.0-only

// Optimized EXL3 MoE precision module: fused kernels with warp hadamard.
// Module name: exl3_moe_precision_opt
// Entry points (fused variants of exl3_moe_precision kernels):
//   exl3_moe_fused_input_hadamard_f16
//   exl3_moe_fused_output_hadamard_precision
//   exl3_moe_fused_activate_down_hadamard
//   exl3_moe_fused_down_weighted_output

#include "exl3_moe_fused.cu"

// Fused MoE input hadamard (FP16 intermediate)
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
    exl3_moe_fused::exl3_moe_fused_input_hadamard_f16(
        x, ids, table, xh, rows, k, topk, expanded, experts
    );
}

// Fused MoE output hadamard with FP32 output
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
    exl3_moe_fused::exl3_moe_fused_output_hadamard_precision(
        yh, ids, table, output, counts, rows, n, topk, experts, thin_cap
    );
}

// Fused activate + down hadamard
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
    exl3_moe_fused::exl3_moe_fused_activate_down_hadamard(
        gate, up, ids, down_table, output, counts,
        rows, k, topk, experts, thin_cap, limit, clamp
    );
}

// Fused down weighted output
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
    exl3_moe_fused::exl3_moe_fused_down_weighted_output(
        yh, ids, table, weights, output, counts,
        rows, n, topk, experts, thin_cap
    );
}