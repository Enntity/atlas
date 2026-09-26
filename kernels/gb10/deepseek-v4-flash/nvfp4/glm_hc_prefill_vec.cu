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
// Four consecutive highway values as FP32 (FP32 or BF16 highway storage).
__device__ __forceinline__ float4 hc_ld4(const float* p) { return *(const float4*)p; }
__device__ __forceinline__ float4 hc_ld4(const __nv_bfloat16* p) {
    const uint2 u = *(const uint2*)p;
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                       __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}

__device__ __forceinline__ float glm_hc_vec_block_reduce(float* red, unsigned tid) {
    for (unsigned s = HC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// Everything after the RMS scale: split, Sinkhorn and the vector collapse.
// `s_rsqrt` and `s_mix` must be populated and visible (after a barrier).
template <typename HT>
__device__ __forceinline__ void glm_hc_vec_finalize(
    const HT* __restrict__ x,
    const float s_rsqrt,
    const float* __restrict__ s_mix,
    float* __restrict__ s_pre,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int t,
    const unsigned int tid,
    const unsigned int sinkhorn_iters,
    const float hc_eps
) {
    constexpr unsigned int H = 4096;

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
            const float4 v = hc_ld4(&x[i * H + d]);
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

template <typename HT>
__device__ __forceinline__ void glm_hc_pre_from_raw_mix_vec_t(
    const HT* __restrict__ streams,
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
    constexpr unsigned int hc_dim = 4 * 4096;
    constexpr unsigned int mix_hc = 24;
    const HT* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];

    float ss = 0.f;
    #pragma unroll 1
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        const float v = (float)x[k];
        ss += v * v;
    }

    red[tid] = ss;
    __syncthreads();
    float ssum = glm_hc_vec_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    if (tid < mix_hc) s_mix[tid] = raw_mix[(size_t)t * mix_hc + tid];
    __syncthreads();
    glm_hc_vec_finalize(x, s_rsqrt, s_mix, s_pre, hc_scale, hc_base, y_out, post_out,
                        comb_out, t, tid, sinkhorn_iters, hc_eps);
}

// FP32 prefill pre-mix: one pass over the highway yields both the 24 raw mix
// dot products and the RMS sum of squares, replacing the TF32 GEMM and the
// finalizer's separate RMS read. Each block stages a 24x256 slice of hc_fn in
// shared memory and reuses it for GLM_HC_MIX_TOKENS tokens (TPW per warp).
// Grid: (ceil(T / GLM_HC_MIX_TOKENS), 1, 1)  Block: (256, 1, 1).
#define GLM_HC_MIX_TPW 4
#define GLM_HC_MIX_TOKENS (8 * GLM_HC_MIX_TPW)
#define GLM_HC_MIX_KC 256
template <typename HT>
__device__ __forceinline__ void glm_hc_mix_ss_t(
    const HT* __restrict__ streams, // [T, 4, 4096]
    const float* __restrict__ hc_fn,   // [24, 16384]
    float* __restrict__ raw_mix,       // [T, 24]
    float* __restrict__ ss_out,        // [T]
    const unsigned int tokens
) {
    constexpr unsigned int K = 4 * 4096;
    constexpr unsigned int M = 24;
    __shared__ float4 s_fn[M * GLM_HC_MIX_KC / 4];
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned int t0 = blockIdx.x * GLM_HC_MIX_TOKENS + warp * GLM_HC_MIX_TPW;
    float acc[GLM_HC_MIX_TPW][M];
    float ss[GLM_HC_MIX_TPW];
    #pragma unroll
    for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) {
        ss[j] = 0.f;
        #pragma unroll
        for (unsigned int m = 0; m < M; ++m) acc[j][m] = 0.f;
    }
    const HT* xs[GLM_HC_MIX_TPW];
    #pragma unroll
    for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) {
        const unsigned int t = min(t0 + j, tokens - 1);
        xs[j] = streams + (size_t)t * K;
    }
    #pragma unroll 1
    for (unsigned int k0 = 0; k0 < K; k0 += GLM_HC_MIX_KC) {
        __syncthreads();
        #pragma unroll
        for (unsigned int i = tid; i < M * GLM_HC_MIX_KC / 4; i += 256) {
            const unsigned int m = i / (GLM_HC_MIX_KC / 4), c = i % (GLM_HC_MIX_KC / 4);
            s_fn[i] = __ldg((const float4*)(hc_fn + (size_t)m * K + k0) + c);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int half = 0; half < GLM_HC_MIX_KC / 128; ++half) {
            const unsigned int c = half * 32 + lane;
            float4 v[GLM_HC_MIX_TPW];
            #pragma unroll
            for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) v[j] = hc_ld4(xs[j] + k0 + 4 * c);
            #pragma unroll
            for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j)
                ss[j] += v[j].x * v[j].x + v[j].y * v[j].y + v[j].z * v[j].z + v[j].w * v[j].w;
            #pragma unroll
            for (unsigned int m = 0; m < M; ++m) {
                const float4 f = s_fn[m * (GLM_HC_MIX_KC / 4) + c];
                #pragma unroll
                for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j)
                    acc[j][m] += f.x * v[j].x + f.y * v[j].y + f.z * v[j].z + f.w * v[j].w;
            }
        }
    }
    #pragma unroll
    for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) {
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) {
            ss[j] += __shfl_xor_sync(0xffffffffu, ss[j], o);
            #pragma unroll
            for (unsigned int m = 0; m < M; ++m)
                acc[j][m] += __shfl_xor_sync(0xffffffffu, acc[j][m], o);
        }
        const unsigned int t = t0 + j;
        if (t < tokens) {
            #pragma unroll
            for (unsigned int m = 0; m < M; ++m)
                if (lane == m) raw_mix[(size_t)t * M + m] = acc[j][m];
            if (lane == 0) ss_out[t] = ss[j];
        }
    }
}

// Finalizer for glm_hc_mix_ss: identical split/Sinkhorn/collapse, RMS scale
// taken from the fused pass instead of a second highway read.
template <typename HT>
__device__ __forceinline__ void glm_hc_pre_finalize_ss_vec_t(
    const HT* __restrict__ streams,
    const float* __restrict__ raw_mix,
    const float* __restrict__ ss_in,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int hc_dim = 4 * 4096;
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];
    if (tid == 0) s_rsqrt = rsqrtf(ss_in[t] / (float)hc_dim + norm_eps);
    if (tid < 24) s_mix[tid] = raw_mix[(size_t)t * 24 + tid];
    __syncthreads();
    glm_hc_vec_finalize(streams + (size_t)t * hc_dim, s_rsqrt, s_mix, s_pre, hc_scale, hc_base,
                        y_out, post_out, comb_out, t, tid, sinkhorn_iters, hc_eps);
}

// ── FP32 (existing) and BF16-highway entry points ──

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
    glm_hc_pre_from_raw_mix_vec_t<float>(streams, raw_mix, hc_scale, hc_base, y_out, post_out, comb_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps, hc_eps);
}

extern "C" __global__ void glm_hc_pre_from_raw_mix_vec_bf16(
    const __nv_bfloat16* __restrict__ streams,
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
    glm_hc_pre_from_raw_mix_vec_t<__nv_bfloat16>(streams, raw_mix, hc_scale, hc_base, y_out, post_out, comb_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps, hc_eps);
}

extern "C" __global__ void __launch_bounds__(256) glm_hc_mix_ss(
    const float* __restrict__ streams, // [T, 4, 4096]
    const float* __restrict__ hc_fn,   // [24, 16384]
    float* __restrict__ raw_mix,       // [T, 24]
    float* __restrict__ ss_out,        // [T]
    const unsigned int tokens
) {
    glm_hc_mix_ss_t<float>(streams, hc_fn, raw_mix, ss_out, tokens);
}

extern "C" __global__ void __launch_bounds__(256) glm_hc_mix_ss_bf16(
    const __nv_bfloat16* __restrict__ streams, // [T, 4, 4096]
    const float* __restrict__ hc_fn,   // [24, 16384]
    float* __restrict__ raw_mix,       // [T, 24]
    float* __restrict__ ss_out,        // [T]
    const unsigned int tokens
) {
    glm_hc_mix_ss_t<__nv_bfloat16>(streams, hc_fn, raw_mix, ss_out, tokens);
}

extern "C" __global__ void glm_hc_pre_finalize_ss_vec(
    const float* __restrict__ streams,
    const float* __restrict__ raw_mix,
    const float* __restrict__ ss_in,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    glm_hc_pre_finalize_ss_vec_t<float>(streams, raw_mix, ss_in, hc_scale, hc_base, y_out, post_out, comb_out, sinkhorn_iters, norm_eps, hc_eps);
}

extern "C" __global__ void glm_hc_pre_finalize_ss_vec_bf16(
    const __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ raw_mix,
    const float* __restrict__ ss_in,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    glm_hc_pre_finalize_ss_vec_t<__nv_bfloat16>(streams, raw_mix, ss_in, hc_scale, hc_base, y_out, post_out, comb_out, sinkhorn_iters, norm_eps, hc_eps);
}
