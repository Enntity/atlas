// SPDX-License-Identifier: AGPL-3.0-only
// Opt-in GLM HC4/4096 prefill finalizer with warp Sinkhorn and vector collapse.
// Same RMS reduction, raw mix and arithmetic as hc_pre_from_raw_mix, with
// only the independent 4x4 Sinkhorn cells distributed across 16 lanes.
#include "../../common/atlas_pdl.cuh"
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

// hc_post of one site fused with glm_hc_mix_ss of the next: one highway pass
// computes out[t,j,d] = post[t,j]*block_out[t,d] + sum_i comb[t,i,j]*x[t,i,d]
// (hc_post_t's exact expression and order), stores it in place, and feeds the
// stored (rounded) values into the next site's 24 mix dots and sum of squares.
// Only the FP32 order of those dot sums differs from glm_hc_mix_ss.
// Grid: (ceil(T / GLM_HC_MIX_TOKENS), 1, 1)  Block: (256, 1, 1).
#define GLM_HC_PM_DC 64
__device__ __forceinline__ float2 hc_ld2(const float* p) { return *(const float2*)p; }
__device__ __forceinline__ float2 hc_ld2(const __nv_bfloat16* p) {
    const unsigned u = *(const unsigned*)p;
    return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ float hc_round(float v, const float*) { return v; }
__device__ __forceinline__ float hc_round(float v, const __nv_bfloat16*) {
    return __bfloat162float(__float2bfloat16(v));
}
__device__ __forceinline__ void hc_st2(float* p, float a, float b) { *(float2*)p = make_float2(a, b); }
__device__ __forceinline__ void hc_st2(__nv_bfloat16* p, float a, float b) {
    *(unsigned*)p = (unsigned)__bfloat16_as_ushort(__float2bfloat16(a))
        | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}
template <typename HT>
__device__ __forceinline__ void glm_hc_post_mix_ss_t(
    const __nv_bfloat16* __restrict__ block_out, // [T, 4096]
    HT* __restrict__ streams,                    // [T, 4, 4096], updated in place
    const float* __restrict__ post,              // [T, 4]
    const float* __restrict__ comb,              // [T, 4, 4]
    const float* __restrict__ hc_fn,             // [24, 16384] of the next site
    float* __restrict__ raw_mix,                 // [T, 24]
    float* __restrict__ ss_out,                  // [T]
    const unsigned int tokens
) {
    constexpr unsigned int H = 4096, K = 4 * H, M = 24, PC = 20;
    __shared__ float2 s_fn[M * 4 * (GLM_HC_PM_DC / 2)]; // [m][stream][d/2]
    __shared__ float s_pc[GLM_HC_MIX_TOKENS * PC];      // post[4], comb[16]
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned int b0 = blockIdx.x * GLM_HC_MIX_TOKENS;
    for (unsigned int i = tid; i < GLM_HC_MIX_TOKENS * PC; i += 256) {
        const unsigned int t = min(b0 + i / PC, tokens - 1), e = i % PC;
        s_pc[i] = e < 4 ? post[(size_t)t * 4 + e] : comb[(size_t)t * 16 + e - 4];
    }
    float acc[GLM_HC_MIX_TPW][M];
    float ss[GLM_HC_MIX_TPW];
    #pragma unroll
    for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) {
        ss[j] = 0.f;
        #pragma unroll
        for (unsigned int m = 0; m < M; ++m) acc[j][m] = 0.f;
    }
    const unsigned int t0 = b0 + warp * GLM_HC_MIX_TPW;
    #pragma unroll 1
    for (unsigned int d0 = 0; d0 < H; d0 += GLM_HC_PM_DC) {
        __syncthreads();
        for (unsigned int i = tid; i < M * 4 * (GLM_HC_PM_DC / 2); i += 256) {
            const unsigned int row = i / (GLM_HC_PM_DC / 2), c = i % (GLM_HC_PM_DC / 2);
            const unsigned int m = row / 4, st = row % 4;
            s_fn[i] = __ldg((const float2*)(hc_fn + (size_t)m * K + st * H + d0) + c);
        }
        __syncthreads();
        const unsigned int d = d0 + 2 * lane;
        // Old highway and block output of this warp's tokens at columns d, d+1.
        float2 rv[GLM_HC_MIX_TPW][4], o[GLM_HC_MIX_TPW];
        #pragma unroll
        for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) {
            const unsigned int t = min(t0 + j, tokens - 1);
            o[j] = hc_ld2(block_out + (size_t)t * H + d);
            #pragma unroll
            for (unsigned int i = 0; i < 4; ++i) rv[j][i] = hc_ld2(streams + (size_t)t * K + i * H + d);
        }
        #pragma unroll
        for (unsigned int st = 0; st < 4; ++st) {
            float2 nv[GLM_HC_MIX_TPW];
            #pragma unroll
            for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j) {
                const float* pc = s_pc + (warp * GLM_HC_MIX_TPW + j) * PC;
                float a = pc[st] * o[j].x, b = pc[st] * o[j].y;
                #pragma unroll
                for (unsigned int i = 0; i < 4; ++i) {
                    a += pc[4 + i * 4 + st] * rv[j][i].x;
                    b += pc[4 + i * 4 + st] * rv[j][i].y;
                }
                // Clamped duplicate rows (t >= tokens) never store.
                if (t0 + j < tokens) hc_st2(streams + (size_t)(t0 + j) * K + st * H + d, a, b);
                nv[j] = make_float2(hc_round(a, streams), hc_round(b, streams));
                ss[j] += nv[j].x * nv[j].x + nv[j].y * nv[j].y;
            }
            #pragma unroll
            for (unsigned int m = 0; m < M; ++m) {
                const float2 f = s_fn[(m * 4 + st) * (GLM_HC_PM_DC / 2) + lane];
                #pragma unroll
                for (unsigned int j = 0; j < GLM_HC_MIX_TPW; ++j)
                    acc[j][m] += f.x * nv[j].x + f.y * nv[j].y;
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

// ── Decode/verify mHC seam (T <= 32 rows) ────────────────────────────────
// The per-token kernels leave most SMs idle at verify widths (one CTA per
// row, three launches per site). Here the 4x4096 highway is split across
// GLM_HCD_SPLIT CTAs of 4 warps (warp = stream, 2 columns per lane): each
// optionally applies the finishing site's hc_post (hc_post_t's expression and
// order, so the stored highway is bitwise identical), then accumulates its
// slice of the next site's 24 mix dots and sum of squares from an hc_fn slice
// staged once. glm_hc_decode_finalize sums the partials and runs the shared
// vector finalizer. Mix/RMS sums differ from glm_hc_mix_ss only in FP32 order.
// partial: [GLM_HCD_SPLIT][T][25]. Grid (GLM_HCD_SPLIT, ceil(T / GLM_HCD_TG)),
// block 128.
#define GLM_HCD_SPLIT 64
#define GLM_HCD_TG 4
#define GLM_HCD_COLS (4096 / GLM_HCD_SPLIT)   // 64 columns per stream per CTA
template <typename HT, bool POST>
__device__ __forceinline__ void glm_hc_decode_partial_t(
    const __nv_bfloat16* __restrict__ block_out, // [T, 4096] (POST only)
    HT* __restrict__ streams,                    // [T, 4, 4096]
    const float* __restrict__ post,              // [T, 4]   (POST only)
    const float* __restrict__ comb,              // [T, 4, 4] (POST only)
    const float* __restrict__ hc_fn,             // [24, 16384] of the next site
    float* __restrict__ partial,
    const unsigned int tokens
) {
    constexpr unsigned int H = 4096, K = 4 * H, M = 24;
    __shared__ float s_fn[M][4][GLM_HCD_COLS];
    __shared__ float s_x[4][GLM_HCD_COLS];
    __shared__ float s_red[4][M + 1];
    const unsigned int tid = threadIdx.x, st = tid >> 5, lane = tid & 31;
    const unsigned int d0 = blockIdx.x * GLM_HCD_COLS;
    for (unsigned int i = tid; i < M * 4 * GLM_HCD_COLS; i += 128) {
        const unsigned int m = i / (4 * GLM_HCD_COLS), r = i % (4 * GLM_HCD_COLS);
        const unsigned int s4 = r / GLM_HCD_COLS, c = r % GLM_HCD_COLS;
        s_fn[m][s4][c] = hc_fn[(size_t)m * K + s4 * H + d0 + c];
    }
    __syncthreads();
    const unsigned int t_end = min(tokens, (blockIdx.y + 1) * GLM_HCD_TG);
    for (unsigned int t = blockIdx.y * GLM_HCD_TG; t < t_end; ++t) {
        HT* x = streams + (size_t)t * K;
        float v[2];
        #pragma unroll
        for (unsigned int e = 0; e < 2; ++e) {
            const unsigned int c = lane + e * 32, d = d0 + c;
            if constexpr (POST) {
                s_x[st][c] = (float)x[st * H + d];
                __syncthreads();
                float acc = post[t * 4 + st] * __bfloat162float(block_out[(size_t)t * H + d]);
                #pragma unroll
                for (unsigned int i = 0; i < 4; ++i) acc += comb[t * 16 + i * 4 + st] * s_x[i][c];
                x[st * H + d] = (HT)acc;
                v[e] = (float)(HT)acc;
                __syncthreads();
            } else {
                v[e] = (float)x[st * H + d];
            }
        }
        float ss = v[0] * v[0] + v[1] * v[1];
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) s_red[st][M] = ss;
        #pragma unroll
        for (unsigned int m = 0; m < M; ++m) {
            float a = s_fn[m][st][lane] * v[0] + s_fn[m][st][lane + 32] * v[1];
            #pragma unroll
            for (unsigned int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) s_red[st][m] = a;
        }
        __syncthreads();
        if (tid <= M)
            partial[((size_t)blockIdx.x * tokens + t) * (M + 1) + tid] =
                s_red[0][tid] + s_red[1][tid] + s_red[2][tid] + s_red[3][tid];
        __syncthreads();
    }
}

// Sum the GLM_HCD_SPLIT partials of row blockIdx.x, then the shared finalizer
// (split, Sinkhorn, collapse). Grid (T), block 256.
template <typename HT>
__device__ __forceinline__ void glm_hc_decode_finalize_t(
    const HT* __restrict__ streams,
    const float* __restrict__ partial,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int tokens,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];
    if (tid <= 24) {
        float acc = 0.f;
        for (unsigned int c = 0; c < GLM_HCD_SPLIT; ++c)
            acc += partial[((size_t)c * tokens + t) * 25 + tid];
        if (tid < 24) s_mix[tid] = acc;
        else s_rsqrt = rsqrtf(acc / (float)(4 * 4096) + norm_eps);
    }
    __syncthreads();
    glm_hc_vec_finalize(streams + (size_t)t * 4 * 4096, s_rsqrt, s_mix, s_pre, hc_scale, hc_base,
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

extern "C" __global__ void __launch_bounds__(256) glm_hc_post_mix_ss(
    const __nv_bfloat16* __restrict__ block_out,
    float* __restrict__ streams,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    const float* __restrict__ hc_fn,
    float* __restrict__ raw_mix,
    float* __restrict__ ss_out,
    const unsigned int tokens
) {
    glm_hc_post_mix_ss_t<float>(block_out, streams, post, comb, hc_fn, raw_mix, ss_out, tokens);
}

extern "C" __global__ void __launch_bounds__(256) glm_hc_post_mix_ss_bf16(
    const __nv_bfloat16* __restrict__ block_out,
    __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    const float* __restrict__ hc_fn,
    float* __restrict__ raw_mix,
    float* __restrict__ ss_out,
    const unsigned int tokens
) {
    glm_hc_post_mix_ss_t<__nv_bfloat16>(block_out, streams, post, comb, hc_fn, raw_mix, ss_out, tokens);
}

extern "C" __global__ void __launch_bounds__(128) glm_hc_decode_post_partial(
    const __nv_bfloat16* __restrict__ block_out, float* __restrict__ streams,
    const float* __restrict__ post, const float* __restrict__ comb,
    const float* __restrict__ hc_fn, float* __restrict__ partial, const unsigned int tokens
) {
    glm_hc_decode_partial_t<float, true>(block_out, streams, post, comb, hc_fn, partial, tokens);
}

extern "C" __global__ void __launch_bounds__(128) glm_hc_decode_partial(
    const __nv_bfloat16* __restrict__ block_out, float* __restrict__ streams,
    const float* __restrict__ post, const float* __restrict__ comb,
    const float* __restrict__ hc_fn, float* __restrict__ partial, const unsigned int tokens
) {
    glm_hc_decode_partial_t<float, false>(block_out, streams, post, comb, hc_fn, partial, tokens);
}

extern "C" __global__ void __launch_bounds__(256) glm_hc_decode_finalize(
    const float* __restrict__ streams, const float* __restrict__ partial,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    const unsigned int tokens, const unsigned int sinkhorn_iters, const float norm_eps, const float hc_eps
) {
    glm_hc_decode_finalize_t<float>(streams, partial, hc_scale, hc_base, y_out, post_out, comb_out,
                                  tokens, sinkhorn_iters, norm_eps, hc_eps);
}

extern "C" __global__ void __launch_bounds__(128) glm_hc_decode_post_partial_bf16(
    const __nv_bfloat16* __restrict__ block_out, __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ post, const float* __restrict__ comb,
    const float* __restrict__ hc_fn, float* __restrict__ partial, const unsigned int tokens
) {
    atlas_pdl_enter();
    glm_hc_decode_partial_t<__nv_bfloat16, true>(block_out, streams, post, comb, hc_fn, partial, tokens);
}

extern "C" __global__ void __launch_bounds__(128) glm_hc_decode_partial_bf16(
    const __nv_bfloat16* __restrict__ block_out, __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ post, const float* __restrict__ comb,
    const float* __restrict__ hc_fn, float* __restrict__ partial, const unsigned int tokens
) {
    atlas_pdl_enter();
    glm_hc_decode_partial_t<__nv_bfloat16, false>(block_out, streams, post, comb, hc_fn, partial, tokens);
}

extern "C" __global__ void __launch_bounds__(256) glm_hc_decode_finalize_bf16(
    const __nv_bfloat16* __restrict__ streams, const float* __restrict__ partial,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    const unsigned int tokens, const unsigned int sinkhorn_iters, const float norm_eps, const float hc_eps
) {
    atlas_pdl_enter();
    glm_hc_decode_finalize_t<__nv_bfloat16>(streams, partial, hc_scale, hc_base, y_out, post_out, comb_out,
                                  tokens, sinkhorn_iters, norm_eps, hc_eps);
}
