// SPDX-License-Identifier: AGPL-3.0-only

// EXL3 reference-policy KDA decode recurrence: a byte-exact port of the pinned
// FLA `fused_recurrent_gated_delta_rule_fwd_kernel` variant used for decode
// (IS_KDA, USE_QK_L2NORM_IN_KERNEL, SIGMOID_BETA, COMPUTE_GATE, SAFE_GATE,
// LOWER_BOUND = -5, BK = 128, BV = 8, num_warps = 1).
//
// Reference layout per warp: lane L owns k = 4L..4L+3 for 8 consecutive value
// rows; every reduction over k is an in-thread chain (product of element 1,
// then fma of elements 0, 2, 3) followed by a butterfly over lanes 16..1.
// The operation sequence below mirrors the reference PTX instruction for
// instruction (see docs/design/exl3-kda-reference-core.md §3.4, §3.6a).
//
// State is Atlas layout [heads, K, V] f32 (V contiguous); the reference's
// [heads, V, K] differs only in indexing. Updated in place.
#pragma once
#include "kda_fla_ops.cuh"

namespace kda_fla_decode {

using namespace kda_fla;

constexpr int D = 128;
constexpr int ROWS_PER_WARP = 8;  // reference BV
constexpr int WARPS = 4;          // value rows per block = 32

// sum over the warp's 128 k values of a[i] * b[i], reference order.
__device__ __forceinline__ float dot4_reduce(const float (&a)[4], const float (&b)[4]) {
    float acc = fmul(a[1], b[1]);
    acc = ffma(a[0], b[0], acc);
    acc = ffma(a[2], b[2], acc);
    acc = ffma(a[3], b[3], acc);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc = fadd(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    return acc;
}

}  // namespace kda_fla_decode

// One decode row of one sequence.
// Grid (D / 32, heads), block 128 (4 warps x 8 value rows).
// qkv: [3 * heads * D] bf16 conv output (q | k | v planes); raw_gate: [heads * D]
// bf16; raw_beta: [heads] bf16 logits; a_log: [heads] f32; dt_bias: [heads * D]
// f32; state: [heads, D, D] f32 (k-major); output: [heads, D] bf16.
extern "C" __global__ void __launch_bounds__(128) kda_fla_recurrent_decode(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int heads
) {
    using namespace kda_fla;
    using namespace kda_fla_decode;
    const unsigned int head = blockIdx.y;
    if (head >= heads || blockDim.x != 32 * WARPS) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int v0 = (blockIdx.x * WARPS + warp) * ROWS_PER_WARP;
    const int k0 = lane * 4;
    // f32(128 ** -0.5), the reference's `scale` argument.
    const float scale = __int_as_float(0x3DB504F3);

    float* H = state + (size_t)head * D * D;
    // b_h = load(h0) + 0 (the reference adds its zero-initialised tile).
    // The warp's 8 value rows are contiguous and 32-byte aligned per k row, so
    // the state moves as float4 pairs (2.5x faster than scalar; same values).
    float h[ROWS_PER_WARP][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r4 = 0; r4 < ROWS_PER_WARP / 4; ++r4) {
            const float4 t = *reinterpret_cast<const float4*>(H + (size_t)(k0 + j) * D + v0 + 4 * r4);
            h[4 * r4 + 0][j] = fadd(t.x, 0.0f);
            h[4 * r4 + 1][j] = fadd(t.y, 0.0f);
            h[4 * r4 + 2][j] = fadd(t.z, 0.0f);
            h[4 * r4 + 3][j] = fadd(t.w, 0.0f);
        }

    const __nv_bfloat16* qp = qkv + (size_t)head * D;
    const __nv_bfloat16* kp = qkv + (size_t)(heads + head) * D;
    const __nv_bfloat16* vp = qkv + (size_t)(2 * heads + head) * D;

    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = __bfloat162float(qp[k0 + j]);
        k[j] = __bfloat162float(kp[k0 + j]);
    }
    // L2 norms: x / sqrt.approx.ftz(sum(x * x) + 1e-6).
    const float eps = __int_as_float(0x358637BD);
    const float qs = sqrt_approx_ftz(fadd(dot4_reduce(q, q), eps));
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = div_full(q[j], qs);
    const float ks = sqrt_approx_ftz(fadd(dot4_reduce(k, k), eps));
#pragma unroll
    for (int j = 0; j < 4; ++j) k[j] = div_full(k[j], ks);
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = fmul(scale, q[j]);

    // Safe gate: -5 / (1 + exp(-(exp(A_log) * (g + bias)))), then exp(gate).
    const float neg_a = -tl_exp(a_log[head]);
    float decay[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gk = fadd(__bfloat162float(raw_gate[(size_t)head * D + k0 + j]),
                              dt_bias[(size_t)head * D + k0 + j]);
        const float gate = div_full(-5.0f, fadd(tl_exp(ffma(neg_a, gk, 0.0f)), 1.0f));
        decay[j] = tl_exp(gate);
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) h[r][j] = fmul(h[r][j], decay[j]);

    // Beta: in-kernel tl.sigmoid on the f32-widened logit.
    const float beta = inductor_sigmoid(__bfloat162float(raw_beta[head]));

    float o[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float hk = dot4_reduce(k, h[r]);
        const float v = fmul(fsub(__bfloat162float(vp[v0 + r]), hk), beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) h[r][j] = ffma(k[j], v, h[r][j]);
        o[r] = dot4_reduce(q, h[r]);
    }

    if (lane < ROWS_PER_WARP) {
        float mine = o[0];
#pragma unroll
        for (int r = 1; r < ROWS_PER_WARP; ++r)
            if (lane == r) mine = o[r];
        output[(size_t)head * D + v0 + lane] = to_bf16(mine);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r4 = 0; r4 < ROWS_PER_WARP / 4; ++r4)
            *reinterpret_cast<float4*>(H + (size_t)(k0 + j) * D + v0 + 4 * r4) =
                make_float4(h[4 * r4][j], h[4 * r4 + 1][j], h[4 * r4 + 2][j], h[4 * r4 + 3][j]);
}
