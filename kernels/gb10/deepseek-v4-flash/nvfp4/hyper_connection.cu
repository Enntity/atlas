// SPDX-License-Identifier: AGPL-3.0-only

// Manifold-Constrained Hyper-Connections (mHC) kernels for DeepSeek-V4.
//
// Every transformer block keeps `hc_mult` parallel residual streams. The
// stream state is stored BF16 as [T, hc_mult, H] (stream-major per token).
// Per attention/FFN site:
//   hc_pre : collapse hc streams -> 1 (RMS-rescaled mix-logits -> sigmoid
//            `pre` weights), and emit `post`/`comb` (Sinkhorn) for hc_post.
//   hc_post: expand the sublayer output back into hc streams, mixing the
//            saved residual streams through the doubly-stochastic `comb`.
// Final collapse before the LM head:
//   hc_head: a single learned weighted sum over the hc streams.
//
// Reference: deepseek-ai/DeepSeek-V4-Pro inference/model.py (hc_split_sinkhorn,
// Block.hc_pre/hc_post, ParallelHead.hc_head). All HC params are float32.
//
// These kernels support hc_mult <= 4 (DeepSeek-V4 uses 4); mix_hc = (2+hc)*hc.

#include <cuda_bf16.h>

#define HC_BLOCK 256
#define HC_MAX_MULT 4
#define HC_MAX_MIX 24 // (2 + HC_MAX_MULT) * HC_MAX_MULT

// Block-wide sum reduction over red[0..HC_BLOCK).
__device__ __forceinline__ float hc_block_reduce(float* red, unsigned int tid) {
    for (unsigned int s = HC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// ── hc_expand ──
// Broadcast a single hidden state into `hc_mult` identical streams:
// streams[t, i, d] = hidden[t, d].  Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void hc_expand(
    const __nv_bfloat16* __restrict__ hidden, // [T, H]
    float* __restrict__ streams,              // [T, hc, H] FP32 highway (mHC)
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)t * H;
    float* s = streams + (size_t)t * hc_mult * H;
    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}

// GLM-5 has no learned HC head: contract the final residual highway by mean.
extern "C" __global__ void hc_contract(
    const float* __restrict__ streams,
    __nv_bfloat16* __restrict__ hidden,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    for (unsigned int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float sum = 0.0f;
        const float* x = streams + (size_t)t * hc_mult * hidden_size + d;
        for (unsigned int i = 0; i < hc_mult; ++i) sum += x[(size_t)i * hidden_size];
        hidden[(size_t)t * hidden_size + d] = __float2bfloat16(sum / (float)hc_mult);
    }
}

// ── hc_pre ──
// streams [T, hc, H] -> y_out [T, H] (collapsed), post_out [T, hc],
// comb_out [T, hc, hc].  Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void hc_pre(
    const float* __restrict__ streams,  // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ hc_fn,    // [mix_hc, hc*H]
    const float* __restrict__ hc_scale, // [3]
    const float* __restrict__ hc_base,  // [mix_hc]
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;
    const unsigned int mix_hc = (2 + hc) * hc;

    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];

    // Pass 1: RMS over the flattened hc*H vector.
    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    __syncthreads();
    const float rsqrt = s_rsqrt;

    // Pass 2: mixes[m] = (sum_k fn[m,k] * x[k]) * rsqrt
    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* fn_row = hc_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
            acc += fn_row[k] * (float)x[k];
        }
        red[tid] = acc;
        __syncthreads();
        float r = hc_block_reduce(red, tid);
        if (tid == 0) s_mix[m] = r * rsqrt;
        __syncthreads();
    }

    // Thread 0: split + Sinkhorn (tiny hc x hc problem).
    if (tid == 0) {
        float comb[HC_MAX_MULT * HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) {
            float pr = s_mix[i] * hc_scale[0] + hc_base[i];
            s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
            float po = s_mix[hc + i] * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] =
                    s_mix[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j];
        // softmax over j (dim=-1) + eps
        for (unsigned int i = 0; i < hc; ++i) {
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }
        // col-norm first (dim=-2, over i)
        for (unsigned int j = 0; j < hc; ++j) {
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }
        // Sinkhorn: (iters - 1) alternating row/col passes
        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            for (unsigned int i = 0; i < hc; ++i) {
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            for (unsigned int j = 0; j < hc; ++j) {
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
        }
        // Final EXACT column projection onto the doubly-stochastic manifold.
        // hc_post mixes streams as out[j] = sum_i comb[i][j] * res[i], so the
        // residual-mixing operator is column-indexed: its spectral radius equals
        // max_j (sum_i comb[i][j]). The Sinkhorn passes above divide by
        // (sum + hc_eps), which leaves each column summing to sum/(sum+eps) — a
        // value that is < 1 but whose denominator carries the eps of EVERY prior
        // pass, so the realized column sums drift off exactly 1 in fp32. Pin the
        // columns to sum exactly to 1 here (no eps), guaranteeing the mixing map
        // is non-expansive (eigenvalue == 1) regardless of logit magnitude — the
        // manifold constraint the kernel's name promises. Matches the reference,
        // which likewise ends its Sinkhorn on a column normalization.
        // NOTE (2026-07-05): dropping this to match the reference's eps-ending
        // Sinkhorn was A/B-tested (portv4b11) and REGRESSED coherence onset
        // (~150→~90 tok) — the extra projection compensates for another mHC
        // deviation, so it stays. eps-Sinkhorn is NOT the ~150 base-degrade lever.
        for (unsigned int j = 0; j < hc; ++j) {
            float c = 0.f;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            float inv = (c > 0.f) ? (1.f / c) : 0.f;
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] *= inv;
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb_out[(size_t)t * hc * hc + i * hc + j] = comb[i * hc + j];
    }
    __syncthreads();

    // Pass 3: collapse y[d] = sum_i pre[i] * x[i, d]
    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}

// Finalize an mHC pre block after a batched TF32 GEMM has produced
// raw_mix[t,m] = dot(streams[t], hc_fn[m]).  Keeping the large MxNxK product
// separate lets cuBLASLt reuse both operands across tokens/mix rows instead of
// hc_pre rereading the same 16K-float highway 24 times per token.
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void hc_pre_from_raw_mix(
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
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;
    const unsigned int mix_hc = (2 + hc) * hc;
    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];

    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        float v = x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    if (tid < mix_hc) s_mix[tid] = raw_mix[(size_t)t * mix_hc + tid];
    __syncthreads();

    if (tid == 0) {
        float comb[HC_MAX_MULT * HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) {
            float pr = s_mix[i] * s_rsqrt * hc_scale[0] + hc_base[i];
            s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
            float po = s_mix[hc + i] * s_rsqrt * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] = s_mix[2 * hc + i * hc + j] * s_rsqrt
                    * hc_scale[2] + hc_base[2 * hc + i * hc + j];
        for (unsigned int i = 0; i < hc; ++i) {
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }
        for (unsigned int j = 0; j < hc; ++j) {
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }
        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            for (unsigned int i = 0; i < hc; ++i) {
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            for (unsigned int j = 0; j < hc; ++j) {
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
        }
        // Atlas's FP32 highway relies on an exact final column projection.
        for (unsigned int j = 0; j < hc; ++j) {
            float c = 0.f;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            float inv = (c > 0.f) ? (1.f / c) : 0.f;
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] *= inv;
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb_out[(size_t)t * hc * hc + i * hc + j] = comb[i * hc + j];
    }
    __syncthreads();

    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}

// ── hc_post ──
// out[t,j,d] = post[t,j]*block_out[t,d] + sum_i comb[t,i,j]*residual[t,i,d].
// `out` may alias `residual` (all hc residual values are read before write).
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void hc_post(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ residual,          // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ post,              // [T, hc]
    const float* __restrict__ comb,              // [T, hc, hc]
    float* __restrict__ out,                     // [T, hc, H] FP32 highway (mHC)
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float xd = (float)x[d];
        float rv[HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
        for (unsigned int j = 0; j < hc; ++j) {
            float acc = p[j] * xd;
            for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
            o[j * H + d] = acc;
        }
    }
}

// Two-rank KDA K=5 fast seam. This intentionally evaluates the exact same
// BF16 `__hadd(local, peer)` that the existing send/recv all-reduce path stores
// before `hc_post` reads it. The reduced BF16 value is only consumed here, so
// fusing removes the transient store/load without changing its arithmetic.
extern "C" __global__ void hc_post_bf16_add(
    const __nv_bfloat16* __restrict__ local_block_out, // [T, H]
    const __nv_bfloat16* __restrict__ peer_block_out,  // [T, H]
    const float* __restrict__ residual,                // [T, hc, H]
    const float* __restrict__ post,                    // [T, hc]
    const float* __restrict__ comb,                    // [T, hc, hc]
    float* __restrict__ out,                           // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* local = local_block_out + (size_t)t * H;
    const __nv_bfloat16* peer = peer_block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        const float xd = (float)__hadd(local[d], peer[d]);
        float rv[HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
        for (unsigned int j = 0; j < hc; ++j) {
            float acc = p[j] * xd;
            for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
            o[j * H + d] = acc;
        }
    }
}

// ── hc_head ──
// Final collapse: streams [T, hc, H] -> y_out [T, H] via a single learned
// sigmoid-weighted sum.  Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void hc_head(
    const float* __restrict__ streams,    // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ head_fn,    // [hc, hc*H]
    const float* __restrict__ head_scale, // [1]
    const float* __restrict__ head_base,  // [hc]
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;

    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_pre[HC_MAX_MULT];

    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    __syncthreads();
    const float rsqrt = s_rsqrt;
    const float scale = head_scale[0];

    for (unsigned int m = 0; m < hc; ++m) {
        const float* fn_row = head_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
            acc += fn_row[k] * (float)x[k];
        }
        red[tid] = acc;
        __syncthreads();
        float r = hc_block_reduce(red, tid);
        if (tid == 0) {
            float v = r * rsqrt * scale + head_base[m];
            s_pre[m] = 1.f / (1.f + expf(-v)) + hc_eps;
        }
        __syncthreads();
    }

    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}
