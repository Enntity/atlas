// SPDX-License-Identifier: AGPL-3.0-only

// EXL3 reference-policy (FLA) KDA preparation stages: causal conv + SiLU,
// q/k L2 norm, beta sigmoid and the chunk-local gate cumsum.
//
// Each kernel reproduces the arithmetic of the pinned reference Triton kernel
// (PTX in arithmetic-audit/kda-reference-ptx-spec-20261010) operation for
// operation, so its output bytes equal the reference stage oracle:
//   conv       _causal_conv1d_fwd_kernel / _causal_conv1d_update_kernel
//   l2norm     l2norm_fwd_kernel2 (MBLOCK 32, BD 128, 4 warps)
//   beta       Inductor triton_poi_fused__to_copy_sigmoid_0
//   gate       kda_gate_cumsum_fwd_kernel (SAFE_GATE, HAS_BIAS, BT 64)
// Layouts are Atlas's: rows of [q | k | v] planes, heads contiguous per plane.
#pragma once
#include "kda_fla_ops.cuh"

namespace kda_fla_prep {

using namespace kda_fla;

// Reference SiLU: acc / (1 + exp(0 - acc)) with tl.exp and div.full.
__device__ __forceinline__ float silu(float acc) {
    return div_full(acc, fadd(tl_exp(fsub(0.0f, acc)), 1.0f));
}

// One channel of the width-4 depthwise conv over `seq_len` rows.
// state[0..3] (Atlas f32 [dim, 4]) holds the last four raw inputs, newest at
// state[3]; the reference keeps the last three ([dim, 3] = state[1..3]).
// Products of BF16 inputs and BF16-valued f32 weights are exact, and the
// reference sums them oldest tap first starting from 0.
__device__ __forceinline__ void conv_channel(float* __restrict__ state, const __nv_bfloat16* __restrict__ input,
                                             const __nv_bfloat16* __restrict__ weight,
                                             __nv_bfloat16* __restrict__ output, unsigned int ch,
                                             unsigned int seq_len, unsigned long long stride,
                                             bool has_initial_state) {
    float* st = state + (unsigned long long)ch * 4;
    float w[4], s[4];
    for (int j = 0; j < 4; ++j) {
        w[j] = __bfloat162float(weight[(unsigned long long)ch * 4 + j]);
        s[j] = has_initial_state ? st[j] : 0.0f;
    }
    for (unsigned int t = 0; t < seq_len; ++t) {
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = __bfloat162float(input[(unsigned long long)t * stride + ch]);
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) acc = fadd(acc, fmul(s[j], w[j]));
        output[(unsigned long long)t * stride + ch] = to_bf16(silu(acc));
    }
    for (int j = 0; j < 4; ++j) st[j] = s[j];
}

}  // namespace kda_fla_prep

// Prefill conv: x/out [T, dim] BF16 (dim = 3*HL*D), weight BF16 [dim, 4],
// conv_state f32 [dim, 4] (read only if has_initial_state, always written).
// Grid (ceil(dim/256)), block 256.
extern "C" __global__ void kda_fla_conv_prefill(float* __restrict__ conv_state,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ weight,
                                                __nv_bfloat16* __restrict__ out, unsigned int dim,
                                                unsigned int tokens, unsigned int has_initial_state) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    kda_fla_prep::conv_channel(conv_state, x, weight, out, ch, tokens, dim, has_initial_state != 0);
}

// Decode conv (one row, existing history). Same arithmetic as the reference
// _causal_conv1d_update_kernel. Grid (ceil(dim/256)), block 256.
extern "C" __global__ void kda_fla_conv_decode(float* __restrict__ conv_state, const __nv_bfloat16* __restrict__ x,
                                               const __nv_bfloat16* __restrict__ weight,
                                               __nv_bfloat16* __restrict__ out, unsigned int dim) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    kda_fla_prep::conv_channel(conv_state, x, weight, out, ch, 1, dim, true);
}

// L2 norm of the q and k planes of conv_out [T, 3*heads*128] into
// q_l2 / k_l2 [T, heads, 128] BF16. One 128-vector per 16 lanes: each lane
// owns 8 contiguous elements; per-lane sum starts e1*e1, then fma e0, e2..e7;
// lane butterfly xor 8, 4, 2, 1; rsqrt.approx.ftz(eps + sum); y = x * r.
// Grid (ceil(2*T*heads / 8)), block 128.
extern "C" __global__ void kda_fla_l2norm(const __nv_bfloat16* __restrict__ conv_out,
                                          __nv_bfloat16* __restrict__ q_l2, __nv_bfloat16* __restrict__ k_l2,
                                          unsigned int tokens, unsigned int heads, float eps) {
    using namespace kda_fla;
    const unsigned int vec = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const unsigned int lane16 = threadIdx.x % 16;
    const unsigned int count = 2 * tokens * heads;
    const bool live = vec < count;
    const unsigned int v = live ? vec : 0;
    const unsigned int plane = v / (tokens * heads);  // 0 = q, 1 = k
    const unsigned int row = (v % (tokens * heads)) / heads;
    const unsigned int head = v % heads;
    const unsigned long long src =
        (unsigned long long)row * 3 * heads * 128 + (unsigned long long)plane * heads * 128 + head * 128 + lane16 * 8;
    float e[8];
    for (int i = 0; i < 8; ++i) e[i] = __bfloat162float(conv_out[src + i]);
    float acc = fmul(e[1], e[1]);
    acc = ffma(e[0], e[0], acc);
    for (int i = 2; i < 8; ++i) acc = ffma(e[i], e[i], acc);
    for (int off = 8; off > 0; off >>= 1) acc = fadd(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    const float r = rsqrt_approx_ftz(fadd(eps, acc));
    if (!live) return;
    __nv_bfloat16* dst = (plane == 0 ? q_l2 : k_l2) + ((unsigned long long)row * heads + head) * 128 + lane16 * 8;
    for (int i = 0; i < 8; ++i) dst[i] = to_bf16(fmul(e[i], r));
}

// beta = sigmoid(float(raw)) as Inductor lowers it. raw BF16 [n], beta f32 [n].
// Grid (ceil(n/256)), block 256.
extern "C" __global__ void kda_fla_beta(const __nv_bfloat16* __restrict__ raw, float* __restrict__ beta,
                                        unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    beta[i] = kda_fla::inductor_sigmoid(__bfloat162float(raw[i]));
}

// Chunk-local gate cumsum: g1 BF16 [T, heads, 128], dt_bias f32 [heads*128],
// a_log f32 [heads] -> g f32 [T, heads, 128].
// gate = lower_bound / (1 + exp(-(exp(A_log) * (g1 + bias)))) and
// y = cumsum_scale * (sequential f32 sum over the row's 64-row chunk).
// Grid (ceil(heads*128 / 128)), block 128; one thread per (head, column).
extern "C" __global__ void kda_fla_gate_cumsum(const __nv_bfloat16* __restrict__ g1, const float* __restrict__ dt_bias,
                                               const float* __restrict__ a_log, float* __restrict__ g,
                                               unsigned int tokens, unsigned int heads, float lower_bound,
                                               float cumsum_scale) {
    using namespace kda_fla;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= heads * 128) return;
    const unsigned int head = col / 128;
    const float a = tl_exp(a_log[head]);
    const float bias = dt_bias[col];
    float acc = 0.0f;
    for (unsigned int t = 0; t < tokens; ++t) {
        if (t % 64 == 0) acc = 0.0f;
        const unsigned long long idx = (unsigned long long)t * heads * 128 + col;
        const float bg = fadd(__bfloat162float(g1[idx]), bias);
        const float inner = ffma(-a, bg, 0.0f);
        const float gate = div_full(lower_bound, fadd(tl_exp(inner), 1.0f));
        // The reference's fma.rn(1.0, gate, acc) is an exactly rounded add.
        // Spell it add.rn: nvcc folds fmaf(1, x, y) into a contractible
        // add.f32 that ptxas then fuses with div.full's multiply.
        acc = fadd(acc, gate);
        g[idx] = fmul(cumsum_scale, acc);
    }
}
