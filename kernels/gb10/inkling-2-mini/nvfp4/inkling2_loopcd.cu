// SPDX-License-Identifier: AGPL-3.0-only

// LoopCD-Logits for Inkling 2 (arXiv 2610.02185), as oMLX serves it
// (jundot/omlx#5640, Apache-2.0: omlx/patches/inkling2/loopcd.py `guide`).
// Both rows arrive softcapped (the head's logit_softcap_bf16). Per row:
//   p = softmax(final[: scored])          (the checkpoint's real vocabulary)
//   w = strength * (1 - (p1 - p2))        (top-two probability margin)
//   guided = final + w * (final - reference)   for ids < ordinary
// thinking[row] != 0 leaves the row as it is. keep_control rescales the guided
// ids so they keep the final row's total probability: the ids at or past
// `ordinary` (control tokens) keep theirs. Ids past `ordinary` are never
// written. fp32 arithmetic, BF16 out.
//
// Grid: (rows, 1, 1)  Block: (LOOPCD_THREADS, 1, 1)

#include <cuda_bf16.h>
#include <math_constants.h>

#define LOOPCD_THREADS 1024

struct Top2 {
    float a, b;  // a >= b; a tie leaves b == a, a zero margin as in the reference
};

__device__ __forceinline__ Top2 top2_add(Top2 t, float x) {
    if (x > t.a) {
        t.b = t.a;
        t.a = x;
    } else if (x > t.b) {
        t.b = x;
    }
    return t;
}

__device__ __forceinline__ Top2 top2_merge(Top2 t, Top2 u) { return top2_add(top2_add(t, u.a), u.b); }

__device__ Top2 block_top2(Top2 t, Top2* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        Top2 u{__shfl_xor_sync(0xffffffff, t.a, off), __shfl_xor_sync(0xffffffff, t.b, off)};
        t = top2_merge(t, u);
    }
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = t;
    __syncthreads();
    if (threadIdx.x == 0) {
        Top2 s = red[0];
        for (unsigned int i = 1; i < blockDim.x / 32; i++) s = top2_merge(s, red[i]);
        red[0] = s;
    }
    __syncthreads();
    Top2 out = red[0];
    __syncthreads();
    return out;
}

// sum (or max, `max_op`) over the CTA, in a fixed order
__device__ float block_reduce(float v, float* red, bool max_op) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float u = __shfl_xor_sync(0xffffffff, v, off);
        v = max_op ? fmaxf(v, u) : v + u;
    }
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = red[0];
        for (unsigned int i = 1; i < blockDim.x / 32; i++) s = max_op ? fmaxf(s, red[i]) : s + red[i];
        red[0] = s;
    }
    __syncthreads();
    const float out = red[0];
    __syncthreads();
    return out;
}

extern "C" __global__ void inkling2_loopcd_guide(__nv_bfloat16* __restrict__ logits,
                                                 const __nv_bfloat16* __restrict__ reference,
                                                 const unsigned char* __restrict__ thinking, unsigned int vocab,
                                                 unsigned int ordinary, unsigned int scored, float strength,
                                                 unsigned int keep_control) {
    const unsigned long long row = blockIdx.x;
    if (thinking != nullptr && thinking[row] != 0) return;
    __nv_bfloat16* __restrict__ f = logits + row * vocab;
    const __nv_bfloat16* __restrict__ r = reference + row * vocab;
    __shared__ Top2 red2[LOOPCD_THREADS / 32];
    __shared__ float red[LOOPCD_THREADS / 32];

    Top2 t{-CUDART_INF_F, -CUDART_INF_F};
    for (unsigned int j = threadIdx.x; j < scored; j += blockDim.x) t = top2_add(t, __bfloat162float(f[j]));
    t = block_top2(t, red2);
    float z = 0.0f;
    for (unsigned int j = threadIdx.x; j < scored; j += blockDim.x) z += expf(__bfloat162float(f[j]) - t.a);
    z = block_reduce(z, red, false);
    const float margin = (1.0f - expf(t.b - t.a)) / z;
    const float w = strength * (1.0f - margin);

    float shift = 0.0f;
    if (keep_control) {
        // log-sum-exp of the final and the guided ids below `ordinary`
        float gmax = -CUDART_INF_F;
        for (unsigned int j = threadIdx.x; j < ordinary; j += blockDim.x) {
            const float x = __bfloat162float(f[j]);
            gmax = fmaxf(gmax, x + w * (x - __bfloat162float(r[j])));
        }
        gmax = block_reduce(gmax, red, true);
        float zf = 0.0f, zg = 0.0f;
        for (unsigned int j = threadIdx.x; j < ordinary; j += blockDim.x) {
            const float x = __bfloat162float(f[j]);
            zf += expf(x - t.a);
            zg += expf(x + w * (x - __bfloat162float(r[j])) - gmax);
        }
        zf = block_reduce(zf, red, false);
        zg = block_reduce(zg, red, false);
        shift = (t.a + logf(zf)) - (gmax + logf(zg));
    }
    for (unsigned int j = threadIdx.x; j < ordinary; j += blockDim.x) {
        const float x = __bfloat162float(f[j]);
        f[j] = __float2bfloat16(x + w * (x - __bfloat162float(r[j])) + shift);
    }
}
