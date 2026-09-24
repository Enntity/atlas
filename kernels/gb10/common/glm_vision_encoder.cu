// SPDX-License-Identifier: AGPL-3.0-only

// Native GLM-5.3 vision primitives.  These deliberately live in a separate
// module from the Qwen vision kernels: GLM uses RMSNorm, clamped SwiGLU,
// partial 2-D RoPE, and a convolutional merger.

#include <cuda_bf16.h>
#include <math.h>

__device__ inline float glm_bf16(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline __nv_bfloat16 glm_bf16_from_f32(float x) {
    return __float2bfloat16(x);
}

extern "C" __global__ void glm_vision_f32_to_bf16(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = glm_bf16_from_f32(src[i]);
}

extern "C" __global__ void glm_vision_gemm_bias(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c,
    unsigned int m, unsigned int n, unsigned int k) {
    unsigned int row = blockIdx.y * 16 + threadIdx.y;
    unsigned int col = blockIdx.x * 16 + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i)
        acc += glm_bf16(a[row * k + i]) * glm_bf16(b[col * k + i]);
    c[row * n + col] = glm_bf16_from_f32(acc + glm_bf16(bias[col]));
}

extern "C" __global__ void glm_vision_rms_norm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    unsigned int rows, unsigned int hidden, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const __nv_bfloat16* xr = x + row * hidden;
    __nv_bfloat16* yr = y + row * hidden;
    float sum = 0.0f;
    for (unsigned int d = 0; d < hidden; ++d) {
        float v = glm_bf16(xr[d]);
        sum += v * v;
    }
    float inv = rsqrtf(sum / hidden + eps);
    for (unsigned int d = 0; d < hidden; ++d)
        yr[d] = glm_bf16_from_f32(glm_bf16(xr[d]) * inv * glm_bf16(w[d]));
}

extern "C" __global__ void glm_vision_add(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = glm_bf16_from_f32(glm_bf16(dst[i]) + glm_bf16(src[i]));
}

extern "C" __global__ void glm_vision_layer_norm(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
    unsigned int rows, unsigned int hidden, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    __nv_bfloat16* xr = x + row * hidden;
    float mean = 0.0f;
    for (unsigned int d = 0; d < hidden; ++d) mean += glm_bf16(xr[d]);
    mean /= hidden;
    float var = 0.0f;
    for (unsigned int d = 0; d < hidden; ++d) {
        float delta = glm_bf16(xr[d]) - mean;
        var += delta * delta;
    }
    float inv = rsqrtf(var / hidden + eps);
    for (unsigned int d = 0; d < hidden; ++d) {
        float v = (glm_bf16(xr[d]) - mean) * inv;
        xr[d] = glm_bf16_from_f32(v * glm_bf16(w[d]) + glm_bf16(b[d]));
    }
}

extern "C" __global__ void glm_vision_gelu(
    __nv_bfloat16* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = glm_bf16(x[i]);
    x[i] = glm_bf16_from_f32(0.5f * v * (1.0f + erff(v * 0.7071067811865475f)));
}

extern "C" __global__ void glm_vision_swiglu_clamp(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    unsigned int rows, unsigned int hidden, float limit) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    unsigned int n = rows * hidden;
    if (i >= n) return;
    unsigned int row = i / hidden;
    unsigned int col = i % hidden;
    // vLLM's SiluAndMulWithClamp clamps the gate only on the upper side;
    // the up branch is clamped symmetrically below.
    float gate = fminf(glm_bf16(x[row * (2 * hidden) + col]), limit);
    float up = fminf(fmaxf(glm_bf16(x[row * (2 * hidden) + hidden + col]), -limit), limit);
    y[i] = glm_bf16_from_f32((gate / (1.0f + expf(-gate))) * up);
}

__device__ inline float glm_rotated(
    const __nv_bfloat16* qkv, unsigned int base, unsigned int dim,
    const __nv_bfloat16* cos_row, const __nv_bfloat16* sin_row,
    const __nv_bfloat16* norm, float inv, unsigned int rotary_dim) {
    // ApplyRotaryEmb is Neox-style over the full head.  GLM's partial rotary
    // factor makes the cache row `head_dim / 2` wide; its first half carries
    // H frequencies and its second half W frequencies, while the two halves
    // of the head are paired by the rotary transform.
    unsigned int partner = dim < rotary_dim ? dim + rotary_dim : dim - rotary_dim;
    // fused_q_kv_rmsnorm returns BF16.  Preserve that write boundary before
    // loading the values for the subsequent BF16 rotary operation.
    float x = glm_bf16(glm_bf16_from_f32(
        glm_bf16(qkv[base + dim]) * inv * glm_bf16(norm[dim])));
    float p = glm_bf16(glm_bf16_from_f32(
        glm_bf16(qkv[base + partner]) * inv * glm_bf16(norm[partner])));
    float c = glm_bf16(cos_row[dim % rotary_dim]);
    float s = glm_bf16(sin_row[dim % rotary_dim]);
    float rotated = dim < rotary_dim ? x * c - p * s : x * c + p * s;
    // The CUDA rotary op writes back to the BF16 q/k tensor before attention.
    return glm_bf16(glm_bf16_from_f32(rotated));
}

// One thread computes one (query, head) row.  It is intentionally a scalar
// correctness kernel while the numerical reference is established; it keeps
// q/k RMSNorm, partial 2-D RoPE, and variable-sequence softmax in one place.
extern "C" __global__ void glm_vision_attention(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ out,
    unsigned int seq, unsigned int heads, unsigned int head_dim) {
    unsigned int query = blockIdx.x;
    unsigned int head = blockIdx.y;
    if (query >= seq || head >= heads || threadIdx.x != 0) return;
    unsigned int hidden = heads * head_dim;
    unsigned int row_stride = 3 * hidden;
    unsigned int q_base = query * row_stride + head * head_dim;
    unsigned int rotary_dim = head_dim / 2;
    float q[128];
    float q_sq = 0.0f;
    for (unsigned int d = 0; d < head_dim; ++d) {
        float raw = glm_bf16(qkv[q_base + d]);
        q[d] = raw;
        q_sq += raw * raw;
    }
    float q_inv = rsqrtf(q_sq / head_dim + 1.0e-5f);
    const __nv_bfloat16* cos_q = cos_table + query * rotary_dim;
    const __nv_bfloat16* sin_q = sin_table + query * rotary_dim;
    for (unsigned int d = 0; d < head_dim; ++d)
        q[d] = glm_rotated(qkv, q_base, d, cos_q, sin_q, q_norm, q_inv, rotary_dim);

    float max_score = -3.402823466e+38f;
    for (unsigned int key = 0; key < seq; ++key) {
        unsigned int k_base = key * row_stride + hidden + head * head_dim;
        float k_sq = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d) {
            float raw = glm_bf16(qkv[k_base + d]);
            k_sq += raw * raw;
        }
        float k_inv = rsqrtf(k_sq / head_dim + 1.0e-5f);
        const __nv_bfloat16* cos_k = cos_table + key * rotary_dim;
        const __nv_bfloat16* sin_k = sin_table + key * rotary_dim;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d)
            score += q[d] * glm_rotated(qkv, k_base, d, cos_k, sin_k, k_norm, k_inv, rotary_dim);
        score *= rsqrtf((float)head_dim);
        max_score = fmaxf(max_score, score);
    }
    float denom = 0.0f;
    for (unsigned int key = 0; key < seq; ++key) {
        unsigned int k_base = key * row_stride + hidden + head * head_dim;
        float k_sq = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d) {
            float raw = glm_bf16(qkv[k_base + d]);
            k_sq += raw * raw;
        }
        float k_inv = rsqrtf(k_sq / head_dim + 1.0e-5f);
        const __nv_bfloat16* cos_k = cos_table + key * rotary_dim;
        const __nv_bfloat16* sin_k = sin_table + key * rotary_dim;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d)
            score += q[d] * glm_rotated(qkv, k_base, d, cos_k, sin_k, k_norm, k_inv, rotary_dim);
        denom += expf(score * rsqrtf((float)head_dim) - max_score);
    }
    for (unsigned int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (unsigned int key = 0; key < seq; ++key) {
            unsigned int k_base = key * row_stride + hidden + head * head_dim;
            float k_sq = 0.0f;
            for (unsigned int j = 0; j < head_dim; ++j) {
                float raw = glm_bf16(qkv[k_base + j]);
                k_sq += raw * raw;
            }
            float k_inv = rsqrtf(k_sq / head_dim + 1.0e-5f);
            const __nv_bfloat16* cos_k = cos_table + key * rotary_dim;
            const __nv_bfloat16* sin_k = sin_table + key * rotary_dim;
            float score = 0.0f;
            for (unsigned int j = 0; j < head_dim; ++j)
                score += q[j] * glm_rotated(qkv, k_base, j, cos_k, sin_k, k_norm, k_inv, rotary_dim);
            float p = expf(score * rsqrtf((float)head_dim) - max_score) / denom;
            acc += p * glm_bf16(qkv[key * row_stride + 2 * hidden + head * head_dim + d]);
        }
        out[query * hidden + head * head_dim + d] = glm_bf16_from_f32(acc);
    }
}

extern "C" __global__ void glm_vision_conv2d(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int grid_h, unsigned int grid_w, unsigned int hidden, unsigned int out_hidden) {
    unsigned int out_col = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int out_h = grid_h / 2;
    unsigned int out_w = grid_w / 2;
    if (out_col >= out_hidden || token >= out_h * out_w || threadIdx.x != 0) return;
    unsigned int bh = token / out_w;
    unsigned int bw = token % out_w;
    float acc = glm_bf16(bias[out_col]);
    for (unsigned int ih = 0; ih < 2; ++ih)
        for (unsigned int iw = 0; iw < 2; ++iw)
            for (unsigned int c = 0; c < hidden; ++c) {
                // The processor emits each 2x2 merge block contiguously
                // (`bh`, `bw`, `ih`, `iw`). The post-layernorm tensor is
                // therefore already laid out as Conv2d's [N,C,H,W] view;
                // indexing it as a row-major patch plane would mix the four
                // members of neighbouring merge blocks.
                unsigned int row = (bh * out_w + bw) * 4 + ih * 2 + iw;
                unsigned int woff = ((out_col * hidden + c) * 2 + ih) * 2 + iw;
                acc += glm_bf16(input[row * hidden + c]) * glm_bf16(weight[woff]);
            }
    output[token * out_hidden + out_col] = glm_bf16_from_f32(acc);
}
