// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5 KDA primitives. The recurrent kernel intentionally follows the
// reference token order; it is the conservative correctness path used by the
// first GB10 port and can later be replaced by a chunked implementation.

#include <cuda_bf16.h>

extern "C" __global__ void kda_pack_qkv(
    const __nv_bfloat16* __restrict__ planes,
    __nv_bfloat16* __restrict__ packed,
    unsigned int tokens,
    unsigned int dim
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)tokens * 3 * dim;
    if (i >= count) return;
    unsigned int c = i % (3 * dim);
    unsigned int t = i / (3 * dim);
    unsigned int plane = c / dim;
    unsigned int d = c % dim;
    packed[i] = planes[((unsigned long long)plane * tokens + t) * dim + d];
}

extern "C" __global__ void kda_recurrent_bf16(
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

// Precompute normalized Q/K and row decay once per token/head. The recurrence
// fans each token out across 32 column groups, so doing the reductions and
// transcendental operations there would repeat the expensive work 32 times.
// The caller supplies FP32 planes borrowed from the otherwise-idle MoE expert
// buffers; no persistent allocation is added.
extern "C" __global__ void kda_preprocess_regresident(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ q_norm,
    float* __restrict__ k_norm,
    float* __restrict__ decay,
    float* __restrict__ beta,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int row = threadIdx.x;
    if (head >= heads || t >= tokens || row >= dim || dim != 128) return;

    __shared__ float red_q[128];
    __shared__ float red_k[128];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float a;
    const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
    const unsigned long long off = ((unsigned long long)t * heads + head) * dim + row;
    const float q = (float)qkv[qbase + (unsigned long long)head * dim + row];
    const float k = (float)qkv[qbase + (unsigned long long)heads * dim
                               + (unsigned long long)head * dim + row];
    red_q[row] = q * q;
    red_k[row] = k * k;
    __syncthreads();

    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (row < stride) {
            red_q[row] += red_q[row + stride];
            red_k[row] += red_k[row + stride];
        }
        __syncthreads();
    }
    if (row == 0) {
        inv_q = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)dim);
        inv_k = rsqrtf(red_k[0] + 1.0e-6f);
        a = expf(a_log[head]);
        const float b = (float)raw_beta[(unsigned long long)t * heads + head];
        beta[(unsigned long long)t * heads + head] = 1.0f / (1.0f + expf(-b));
    }
    __syncthreads();

    q_norm[off] = q * inv_q;
    k_norm[off] = k * inv_k;
    const float g = (float)raw_gate[off];
    const float bias = dt_bias[(unsigned long long)head * dim + row];
    decay[off] = expf(lower_bound / (1.0f + expf(-a * (g + bias))));
}

// Register-resident prefill recurrence for GLM-5 KDA.
//
// The reference kernel assigns one thread to each state column, leaving the
// 128x128 FP32 state in global memory throughout the token loop. That streams
// every state matrix twice per prompt token. Here one warp owns one column:
// its lanes retain four state rows each in registers for the whole prompt.
// State columns are independent under KDA's row-wise decay, so 32 CTAs per
// head expose enough parallelism while loading and storing H exactly once.
//
// Grid:  (heads, dim / 4, 1)
// Block: (128, 1, 1), four warps and therefore four columns per CTA.
extern "C" __global__ void __launch_bounds__(128, 4)
kda_recurrent_bf16_regresident(
    const __nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int col = blockIdx.y * 4 + warp;
    if (head >= heads || col >= dim || dim != 128) return;

    const unsigned int r0 = lane;
    const unsigned int r1 = lane + 32;
    const unsigned int r2 = lane + 64;
    const unsigned int r3 = lane + 96;
    float* H = state + (unsigned long long)head * dim * dim;

    float s0 = H[(unsigned long long)r0 * dim + col];
    float s1 = H[(unsigned long long)r1 * dim + col];
    float s2 = H[(unsigned long long)r2 * dim + col];
    float s3 = H[(unsigned long long)r3 * dim + col];

    for (unsigned int t = 0; t < tokens; ++t) {
        const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
        const unsigned long long off = ((unsigned long long)t * heads + head) * dim;
        const float q0 = q_norm[off + r0];
        const float q1 = q_norm[off + r1];
        const float q2 = q_norm[off + r2];
        const float q3 = q_norm[off + r3];
        const float k0 = k_norm[off + r0];
        const float k1 = k_norm[off + r1];
        const float k2 = k_norm[off + r2];
        const float k3 = k_norm[off + r3];
        const float d0 = decay[off + r0];
        const float d1 = decay[off + r1];
        const float d2 = decay[off + r2];
        const float d3 = decay[off + r3];

        float dot_k = s0 * d0 * k0 + s1 * d1 * k1
                    + s2 * d2 * k2 + s3 * d3 * k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot_k += __shfl_xor_sync(0xFFFFFFFFu, dot_k, offset);

        const unsigned long long voff = qbase + (unsigned long long)2 * heads * dim
                                         + (unsigned long long)head * dim + col;
        const float delta = ((float)qkv[voff] - dot_k)
                          * beta[(unsigned long long)t * heads + head];

        s0 = s0 * d0 + delta * k0;
        s1 = s1 * d1 + delta * k1;
        s2 = s2 * d2 + delta * k2;
        s3 = s3 * d3 + delta * k3;

        float dot_q = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot_q += __shfl_xor_sync(0xFFFFFFFFu, dot_q, offset);
        if (lane == 0) {
            output[((unsigned long long)t * heads + head) * dim + col] =
                __float2bfloat16(dot_q);
        }
    }

    H[(unsigned long long)r0 * dim + col] = s0;
    H[(unsigned long long)r1 * dim + col] = s1;
    H[(unsigned long long)r2 * dim + col] = s2;
    H[(unsigned long long)r3 * dim + col] = s3;
}

extern "C" __global__ void kda_sigmoid_gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int heads,
    unsigned int dim,
    float eps
) {
    const unsigned int row_id = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (dim > 128 || d >= dim) return;
    __shared__ float red[128];
    __shared__ float inv;
    const unsigned long long off = (unsigned long long)row_id * dim + d;
    float x = (float)input[off];
    red[d] = x * x;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) red[d] += red[d + stride];
        __syncthreads();
    }
    if (d == 0) inv = rsqrtf(red[0] / (float)dim + eps);
    __syncthreads();
    float g = (float)gate[off];
    float y = x * inv * (float)weight[d] / (1.0f + expf(-g));
    output[off] = __float2bfloat16(y);
}
