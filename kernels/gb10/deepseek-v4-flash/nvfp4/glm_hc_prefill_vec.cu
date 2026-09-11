// SPDX-License-Identifier: AGPL-3.0-only
// Opt-in GLM HC4/4096 prefill finalizer with warp Sinkhorn and vector collapse.
// Same RMS reduction, raw mix and arithmetic as hc_pre_from_raw_mix, with
// only the independent 4x4 Sinkhorn cells distributed across 16 lanes.
#include <cuda_bf16.h>

#ifndef HC_BLOCK
#define HC_BLOCK 256
#define HC_MAX_MULT 4
#define HC_MAX_MIX 24
#endif
__device__ __forceinline__ float glm_hc_vec_block_reduce(float* red, unsigned tid) {
    for (unsigned s = HC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

extern "C" __global__ void glm_hc_pre_from_raw_mix_vec(
    const float* __restrict__ streams,
    const float* __restrict__ raw_mix,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    if (hc_mult != 4 || hidden_size != 4096 || blockDim.x != 256) return;
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int H = 4096;
    constexpr unsigned int hc = 4;
    const unsigned int hc_dim = hc * H;
    const unsigned int mix_hc = (2 + hc) * hc;
    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];

    float ss = 0.f;
    #pragma unroll 1
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        const float v = x[k];
        ss += v * v;
    }

    red[tid] = ss;
    __syncthreads();
    float ssum = glm_hc_vec_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    if (tid < mix_hc) s_mix[tid] = raw_mix[(size_t)t * mix_hc + tid];
    __syncthreads();

    // Independent gates retain the baseline expression and operation order.
    if (tid < 4) {
        const unsigned int i = tid;
        float pr = s_mix[i] * s_rsqrt * hc_scale[0] + hc_base[i];
        s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
        float po = s_mix[4 + i] * s_rsqrt * hc_scale[1] + hc_base[4 + i];
        post_out[(size_t)t * 4 + i] = 2.f * (1.f / (1.f + expf(-po)));
    }
    // One half-warp owns the 4x4 matrix. Broadcasts enumerate each row/column
    // in the baseline's 0,1,2,3 order; only independent cells run in parallel.
    if (tid < 16) {
        constexpr unsigned mask = 0xffffu;
        const unsigned int row = tid / 4, col = tid % 4;
        float value = s_mix[8 + tid] * s_rsqrt * hc_scale[2] + hc_base[8 + tid];
        float mx = -1e30f;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            mx = fmaxf(mx, __shfl_sync(mask, value, row * 4 + j));
        value = expf(value - mx);
        float sum = 0.f;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            sum += __shfl_sync(mask, value, row * 4 + j);
        value = value / sum + hc_eps;
        float c = hc_eps;
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i)
            c += __shfl_sync(mask, value, i * 4 + col);
        value /= c;
        for (unsigned it = 0; it + 1 < sinkhorn_iters; ++it) {
            float r = hc_eps;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                r += __shfl_sync(mask, value, row * 4 + j);
            value /= r;
            c = hc_eps;
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                c += __shfl_sync(mask, value, i * 4 + col);
            value /= c;
        }
        c = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i)
            c += __shfl_sync(mask, value, i * 4 + col);
        const float inv = (c > 0.f) ? (1.f / c) : 0.f;
        value *= inv;
        comb_out[(size_t)t * 16 + tid] = value;
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int chunk = 0; chunk < 4; ++chunk) {
        const unsigned int d = tid * 4 + chunk * HC_BLOCK * 4;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            const float4 v = *(const float4*)&x[i * H + d];
            acc.x += s_pre[i] * v.x; acc.y += s_pre[i] * v.y;
            acc.z += s_pre[i] * v.z; acc.w += s_pre[i] * v.w;
        }
        const unsigned lo = (unsigned)__bfloat16_as_ushort(__float2bfloat16(acc.x))
            | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(acc.y)) << 16);
        const unsigned hi = (unsigned)__bfloat16_as_ushort(__float2bfloat16(acc.z))
            | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(acc.w)) << 16);
        *(uint2*)&y_out[(size_t)t * H + d] = make_uint2(lo, hi);
    }

}

