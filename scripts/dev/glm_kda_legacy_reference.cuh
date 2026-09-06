// SPDX-License-Identifier: AGPL-3.0-only

// TEST-ONLY frozen Atlas scalar reference, snapshotted 2026-09-06 BEFORE
// shared arithmetic extraction. Never regenerate from refactored production.
// The two original function definitions are byte-identical except exported names.
#pragma once
#include <cuda_bf16.h>

// Source: kernels/gb10/common/causal_conv1d.cu
extern "C" __global__ void frozen_causal_conv1d_update_prefill(
    float* __restrict__ conv_state,             // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ input,    // N tokens, input[t * input_stride + ch]
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // N tokens, output[t * output_stride + ch]
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,          // number of tokens
    unsigned int input_stride,     // BF16 elements between consecutive tokens in input
    unsigned int output_stride     // BF16 elements between consecutive tokens in output
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;

    float* state = conv_state + ch * d_conv;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    // Load weights into registers (d_conv = 4 for Qwen3-Next)
    float w_reg[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
    }

    // Load current sliding window state into registers
    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        s[k] = state[k];
    }

    // Process all tokens sequentially — state stays in registers
    for (unsigned int t = 0; t < seq_len; t++) {
        float new_val = (float)input[(unsigned long long)t * input_stride + ch];

        // Shift state left by 1, insert new value
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;

        // Depthwise convolution
        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1] + s[2]*w_reg[2] + s[3]*w_reg[3];

        // SiLU activation
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sigmoid_acc);
    }

    // Write final state back to global memory
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        state[k] = s[k];
    }
}

// Source: kernels/gb10/common/kda.cu
extern "C" __global__ void frozen_kda_recurrent_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int vrow = threadIdx.x;
    if (head >= heads || dim > 128 || blockDim.x < dim) return;

    __shared__ float qv[128];
    __shared__ float kv[128];
    __shared__ float gate_exp[128];
    __shared__ float red_q[128];
    __shared__ float red_k[128];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float beta;

    // H is [key_dim, value_dim], with value_dim contiguous. Each thread owns
    // one value column, so every warp reads and writes adjacent FP32 elements.
    // The original [value_dim, key_dim] traversal issued one 32-byte memory
    // transaction per lane for every state access on decode.
    float* H = state + (unsigned long long)head * dim * dim;
    const float a = expf(a_log[head]);
    const float scale = rsqrtf((float)dim);

    for (unsigned int t = 0; t < tokens; ++t) {
        const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
        if (vrow < dim) {
            float q = (float)qkv[qbase + (unsigned long long)head * dim + vrow];
            float k = (float)qkv[qbase + (unsigned long long)heads * dim
                              + (unsigned long long)head * dim + vrow];
            qv[vrow] = q;
            kv[vrow] = k;
            red_q[vrow] = q * q;
            red_k[vrow] = k * k;
            float g = (float)raw_gate[((unsigned long long)t * heads + head) * dim + vrow];
            float log_decay = lower_bound /
                (1.0f + expf(-a * (g + dt_bias[(unsigned long long)head * dim + vrow])));
            gate_exp[vrow] = expf(log_decay);
        }
        __syncthreads();

        for (unsigned int stride = 64; stride > 0; stride >>= 1) {
            if (vrow < stride) {
                red_q[vrow] += red_q[vrow + stride];
                red_k[vrow] += red_k[vrow + stride];
            }
            __syncthreads();
        }
        if (vrow == 0) {
            inv_q = rsqrtf(red_q[0] + 1.0e-6f) * scale;
            inv_k = rsqrtf(red_k[0] + 1.0e-6f);
            float b = (float)raw_beta[(unsigned long long)t * heads + head];
            beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();

        if (vrow < dim) {
            float dot_k = 0.0f;
            #pragma unroll 4
            for (unsigned int k = 0; k < dim; k += 4) {
                float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * gate_exp[k + 0];
                float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * gate_exp[k + 1];
                float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * gate_exp[k + 2];
                float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * gate_exp[k + 3];
                dot_k += h0 * (kv[k + 0] * inv_k) + h1 * (kv[k + 1] * inv_k)
                       + h2 * (kv[k + 2] * inv_k) + h3 * (kv[k + 3] * inv_k);
            }
            const unsigned long long vbase = qbase + (unsigned long long)2 * heads * dim;
            float delta = ((float)qkv[vbase + (unsigned long long)head * dim + vrow]
                           - dot_k) * beta;
            float out = 0.0f;
            #pragma unroll 4
            for (unsigned int k = 0; k < dim; k += 4) {
                float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * gate_exp[k + 0]
                         + delta * (kv[k + 0] * inv_k);
                float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * gate_exp[k + 1]
                         + delta * (kv[k + 1] * inv_k);
                float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * gate_exp[k + 2]
                         + delta * (kv[k + 2] * inv_k);
                float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * gate_exp[k + 3]
                         + delta * (kv[k + 3] * inv_k);
                H[(unsigned long long)(k + 0) * dim + vrow] = h0;
                H[(unsigned long long)(k + 1) * dim + vrow] = h1;
                H[(unsigned long long)(k + 2) * dim + vrow] = h2;
                H[(unsigned long long)(k + 3) * dim + vrow] = h3;
                out += h0 * (qv[k + 0] * inv_q) + h1 * (qv[k + 1] * inv_q)
                     + h2 * (qv[k + 2] * inv_q) + h3 * (qv[k + 3] * inv_q);
            }
            output[((unsigned long long)t * heads + head) * dim + vrow] =
                __float2bfloat16(out);
        }
        __syncthreads();
    }
}


