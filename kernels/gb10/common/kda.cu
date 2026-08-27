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

    float* row = state + ((unsigned long long)head * dim + vrow) * dim;
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
            for (unsigned int k = 0; k < dim; ++k) {
                float s = row[k] * gate_exp[k];
                row[k] = s;
                dot_k += s * (kv[k] * inv_k);
            }
            const unsigned long long vbase = qbase + (unsigned long long)2 * heads * dim;
            float delta = ((float)qkv[vbase + (unsigned long long)head * dim + vrow]
                           - dot_k) * beta;
            float out = 0.0f;
            for (unsigned int k = 0; k < dim; ++k) {
                float s = row[k] + delta * (kv[k] * inv_k);
                row[k] = s;
                out += s * (qv[k] * inv_q);
            }
            output[((unsigned long long)t * heads + head) * dim + vrow] =
                __float2bfloat16(out);
        }
        __syncthreads();
    }
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
