// SPDX-License-Identifier: AGPL-3.0-only

// Test-only value32 decomposition of Atlas's FP32 recurrent arithmetic.
// No production include/registration. See glm_kda_value_tile_plan.md.
// Host requires H32/D128, N1..4, unique live slots and validated pool spans.
// Grid(heads,rows,4), block(32,1,1); state remains [head,key,value].
#pragma once
#include <cuda_bf16.h>

extern "C" __global__ void glm_kda_recurrent_value32(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state_pool,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int heads,
    unsigned int dim,
    float lower_bound,
    const int* __restrict__ state_slots,
    unsigned int slot_count,
    unsigned long long state_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int vrow = blockIdx.z * 32 + lane;
    // Every rejection is CTA-uniform and precedes all barriers/state accesses.
    // Do not add dim==128: preserve scalar runtime rsqrtf(dim) lowering.
    if (row >= rows || head >= heads || dim > 128 || blockIdx.z >= 4
        || blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1) return;
    const int slot = state_slots[row];
    if (slot < 0 || (unsigned int)slot >= slot_count
        || state_stride < (unsigned long long)heads * dim * dim) return;
    float* H = state_pool + (unsigned long long)slot * state_stride
        + (unsigned long long)head * dim * dim;
    qkv += (unsigned long long)row * 3 * heads * dim;
    raw_gate += (unsigned long long)row * heads * dim;
    raw_beta += (unsigned long long)row * heads;
    output += (unsigned long long)row * heads * dim;

    __shared__ float value32_qv[128];
    __shared__ float value32_kv[128];
    __shared__ float value32_gate_exp[128];
    __shared__ float value32_red_q[128];
    __shared__ float value32_red_k[128];
    __shared__ float value32_inv_q;
    __shared__ float value32_inv_k;
    __shared__ float value32_beta;

    const float a = expf(a_log[head]);
    const float scale = rsqrtf((float)dim);
    // Four coalesced lane passes populate exactly the same norm/gate arrays.
    for (unsigned int key = lane; key < dim; key += 32) {
        float q = (float)qkv[(unsigned long long)head * dim + key];
        float k = (float)qkv[(unsigned long long)heads * dim
                          + (unsigned long long)head * dim + key];
        value32_qv[key] = q;
        value32_kv[key] = k;
        value32_red_q[key] = q * q;
        value32_red_k[key] = k * k;
        float g = (float)raw_gate[(unsigned long long)head * dim + key];
        float log_decay = lower_bound /
            (1.0f + expf(-a * (g + dt_bias[(unsigned long long)head * dim + key])));
        value32_gate_exp[key] = expf(log_decay);
    }
    __syncthreads();

    // Same tree/addition grouping as scalar. At stride64 each lane performs
    // two independent additions; no result is consumed until the barrier.
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        for (unsigned int key = lane; key < stride; key += 32) {
            value32_red_q[key] += value32_red_q[key + stride];
            value32_red_k[key] += value32_red_k[key + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        value32_inv_q = rsqrtf(value32_red_q[0] + 1.0e-6f) * scale;
        value32_inv_k = rsqrtf(value32_red_k[0] + 1.0e-6f);
        float b = (float)raw_beta[head];
        value32_beta = 1.0f / (1.0f + expf(-b));
    }
    __syncthreads();

    if (vrow < dim) {
        float dot_k = 0.0f;
        #pragma unroll 4
        for (unsigned int k = 0; k < dim; k += 4) {
            float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * value32_gate_exp[k + 0];
            float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * value32_gate_exp[k + 1];
            float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * value32_gate_exp[k + 2];
            float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * value32_gate_exp[k + 3];
            dot_k += h0 * (value32_kv[k + 0] * value32_inv_k) + h1 * (value32_kv[k + 1] * value32_inv_k)
                   + h2 * (value32_kv[k + 2] * value32_inv_k) + h3 * (value32_kv[k + 3] * value32_inv_k);
        }
        const unsigned long long vbase = (unsigned long long)2 * heads * dim;
        float delta = ((float)qkv[vbase + (unsigned long long)head * dim + vrow]
                       - dot_k) * value32_beta;
        float out = 0.0f;
        #pragma unroll 4
        for (unsigned int k = 0; k < dim; k += 4) {
            float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * value32_gate_exp[k + 0]
                     + delta * (value32_kv[k + 0] * value32_inv_k);
            float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * value32_gate_exp[k + 1]
                     + delta * (value32_kv[k + 1] * value32_inv_k);
            float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * value32_gate_exp[k + 2]
                     + delta * (value32_kv[k + 2] * value32_inv_k);
            float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * value32_gate_exp[k + 3]
                     + delta * (value32_kv[k + 3] * value32_inv_k);
            H[(unsigned long long)(k + 0) * dim + vrow] = h0;
            H[(unsigned long long)(k + 1) * dim + vrow] = h1;
            H[(unsigned long long)(k + 2) * dim + vrow] = h2;
            H[(unsigned long long)(k + 3) * dim + vrow] = h3;
            out += h0 * (value32_qv[k + 0] * value32_inv_q) + h1 * (value32_qv[k + 1] * value32_inv_q)
                 + h2 * (value32_qv[k + 2] * value32_inv_q) + h3 * (value32_qv[k + 3] * value32_inv_q);
        }
        output[(unsigned long long)head * dim + vrow] = __float2bfloat16(out);
    }
}
