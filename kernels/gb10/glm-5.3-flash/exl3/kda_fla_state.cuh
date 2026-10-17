// SPDX-License-Identifier: AGPL-3.0-only

// EXL3 reference-policy KDA: chunked state recurrence (delta h) and output.
//
// Byte-exact ports of the pinned FLA kernels chunk_gated_delta_rule_fwd_kernel_h_blockdim64
// (USE_GK, use_exp2) and chunk_gla_fwd_kernel_o as lowered by Triton 3.7.1 for
// sm_121a (docs/design/exl3-kda-reference-core.md §3.6a). Arithmetic contract
// read from the reference TTGIR/PTX:
//
// delta h, per 64-row chunk (state f32, carried across chunks):
//   h_bf16 = cvt.rn(h)                                  (stored, consumers' view)
//   acc    = mma_bf16(w[t, k], h_bf16[k, v]), k = 0..127 in ascending k16 steps,
//            one accumulator starting from +0
//   v_new  = sub.f32(u, acc), stored as cvt.rn bf16
//   d      = mma_bf16(kg[t, k]^T, v_new_bf16[t, v]), t = 0..63 ascending, from +0
//   h      = fma.rn(h, ex2.approx.ftz(g[last, k]), d)   (fma.rn.f32x2 in the reference)
// output, per 64-row chunk:
//   qg  = cvt.rn(mul(cvt.rn(mul(q, scale)), ex2.approx.ftz(g)))
//   o   = mma_bf16(qg[t, k], h_bf16[k, v]) over k ascending, then in place
//         += mma_bf16(cvt.rn(t >= s ? Aqk[t, s] : 0), v_new[s, v]) over s ascending
//   o stored as cvt.rn bf16.
// Rows past the sequence end load as zero, as in the reference's boundary checks.
//
// Layouts (single sequence of T rows, HL heads, K = V = 128, chunk 64):
//   kg, w, u, q, v_new, o : bf16 [T, HL, 128]      g : f32 [T, HL, 128]
//   aqk : f32 [T, HL, 64]                          h : bf16 [NT, HL, V, K] (scratch)
//   state : f32 Atlas layout [HL, K, V] (V contiguous), read then overwritten in place.
#pragma once
#include "kda_fla_ops.cuh"

namespace kda_fla_state {

constexpr int D = 128;
constexpr int BT = 64;
constexpr int DELTA_BV = 32;
constexpr int OUT_BV = 64;

__device__ __forceinline__ uint32_t ld_bf16x2(const __nv_bfloat16* p) {
    return kda_fla::pack_bf16x2(p[0], p[1]);
}

}  // namespace kda_fla_state

// Grid (V / 32, HL), block 128 (4 warps). Dynamic shared memory: none.
extern "C" __global__ void __launch_bounds__(128) kda_fla_delta_h(
    const __nv_bfloat16* __restrict__ kg,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ u,
    const float* __restrict__ g,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ v_new,
    unsigned int tokens,
    unsigned int heads
) {
    using namespace kda_fla;
    using namespace kda_fla_state;
    constexpr int BV = DELTA_BV;
    __shared__ float hs[D][BV];                    // f32 state slice, [k][v]
    __shared__ __nv_bfloat16 hb[D][BV + 8];        // bf16(h), [k][v]
    __shared__ __nv_bfloat16 vn[BT][BV + 8];       // bf16(v_new), [t][v]
    __shared__ float e_last[D];

    const unsigned int v0 = blockIdx.x * BV;
    const unsigned int head = blockIdx.y;
    if (head >= heads) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tq = lane & 3;
    const size_t row_stride = (size_t)heads * D;
    float* st = state + (size_t)head * D * D;

    for (int i = tid; i < D * BV; i += blockDim.x) {
        int k = i / BV, v = i % BV;
        hs[k][v] = fadd(0.0f, st[(size_t)k * D + v0 + v]);  // zeros + load(h0)
    }
    __syncthreads();

    const unsigned int nt = (tokens + BT - 1) / BT;
    for (unsigned int it = 0; it < nt; ++it) {
        const unsigned int t0 = it * BT;
        // Store bf16(h) for consumers and stage it as the mma B operand.
        for (int i = tid; i < D * BV; i += blockDim.x) {
            int k = i / BV, v = i % BV;
            __nv_bfloat16 b = to_bf16(hs[k][v]);
            hb[k][v] = b;
            h_out[(((size_t)it * heads + head) * D + v0 + v) * D + k] = b;
        }
        __syncthreads();

        // v_new = u - w . bf16(h); warp owns rows t0 + 16*warp .. +15.
        {
            float acc[BV / 8][4];
            for (int n = 0; n < BV / 8; ++n)
                for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
            const unsigned int ra = t0 + warp * 16 + gq, rb = ra + 8;
            for (int ks = 0; ks < D / 16; ++ks) {
                const int kc = ks * 16 + 2 * tq;
                uint32_t a[4];
                const __nv_bfloat16 z = __float2bfloat16_rn(0.0f);
                const __nv_bfloat16* pa = w + (size_t)ra * row_stride + head * D;
                const __nv_bfloat16* pb = w + (size_t)rb * row_stride + head * D;
                a[0] = ra < tokens ? ld_bf16x2(pa + kc) : pack_bf16x2(z, z);
                a[1] = rb < tokens ? ld_bf16x2(pb + kc) : pack_bf16x2(z, z);
                a[2] = ra < tokens ? ld_bf16x2(pa + kc + 8) : pack_bf16x2(z, z);
                a[3] = rb < tokens ? ld_bf16x2(pb + kc + 8) : pack_bf16x2(z, z);
                for (int n = 0; n < BV / 8; ++n) {
                    const int col = n * 8 + gq;
                    uint32_t b[2] = {pack_bf16x2(hb[kc][col], hb[kc + 1][col]),
                                     pack_bf16x2(hb[kc + 8][col], hb[kc + 9][col])};
                    mma_bf16_16816(acc[n], a, b, acc[n]);
                }
            }
            for (int n = 0; n < BV / 8; ++n) {
                for (int r = 0; r < 4; ++r) {
                    const unsigned int row = (r < 2 ? ra : rb);
                    const int col = n * 8 + 2 * tq + (r & 1);
                    float uu = row < tokens
                        ? __bfloat162float(u[(size_t)row * row_stride + head * D + v0 + col]) : 0.0f;
                    __nv_bfloat16 b = to_bf16(fsub(uu, acc[n][r]));
                    vn[row - t0][col] = b;
                    if (row < tokens) v_new[(size_t)row * row_stride + head * D + v0 + col] = b;
                }
            }
        }
        const unsigned int last = min(t0 + BT, tokens) - 1;
        for (int k = tid; k < D; k += blockDim.x)
            e_last[k] = ex2_approx_ftz(g[(size_t)last * row_stride + head * D + k]);
        __syncthreads();

        // d[k, v] = sum_t kg[t, k] * v_new[t, v]; warp owns k rows 32*warp .. +31.
        for (int m = 0; m < 2; ++m) {
            const int ka = warp * 32 + m * 16 + gq, kb = ka + 8;
            float acc[BV / 8][4];
            for (int n = 0; n < BV / 8; ++n)
                for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
            for (int ts = 0; ts < BT / 16; ++ts) {
                const int tc = ts * 16 + 2 * tq;
                auto kgv = [&](int t, int k) {
                    unsigned int row = t0 + t;
                    return row < tokens ? kg[(size_t)row * row_stride + head * D + k] : __float2bfloat16_rn(0.0f);
                };
                uint32_t a[4] = {pack_bf16x2(kgv(tc, ka), kgv(tc + 1, ka)),
                                 pack_bf16x2(kgv(tc, kb), kgv(tc + 1, kb)),
                                 pack_bf16x2(kgv(tc + 8, ka), kgv(tc + 9, ka)),
                                 pack_bf16x2(kgv(tc + 8, kb), kgv(tc + 9, kb))};
                for (int n = 0; n < BV / 8; ++n) {
                    const int col = n * 8 + gq;
                    uint32_t b[2] = {pack_bf16x2(vn[tc][col], vn[tc + 1][col]),
                                     pack_bf16x2(vn[tc + 8][col], vn[tc + 9][col])};
                    mma_bf16_16816(acc[n], a, b, acc[n]);
                }
            }
            for (int n = 0; n < BV / 8; ++n) {
                for (int r = 0; r < 4; ++r) {
                    const int k = (r < 2 ? ka : kb);
                    const int col = n * 8 + 2 * tq + (r & 1);
                    hs[k][col] = ffma(hs[k][col], e_last[k], acc[n][r]);
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < D * BV; i += blockDim.x) {
        int k = i / BV, v = i % BV;
        st[(size_t)k * D + v0 + v] = hs[k][v];
    }
}

// Grid (V / 64, NT, HL), block 128 (4 warps; warp w owns rows 16w..16w+15).
extern "C" __global__ void __launch_bounds__(128) kda_fla_output(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ v_new,
    const float* __restrict__ g,
    const float* __restrict__ aqk,
    const __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ o,
    float scale,
    unsigned int tokens,
    unsigned int heads
) {
    using namespace kda_fla;
    using namespace kda_fla_state;
    constexpr int BV = OUT_BV;
    const unsigned int v0 = blockIdx.x * BV;
    const unsigned int it = blockIdx.y;
    const unsigned int head = blockIdx.z;
    if (head >= heads) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gq = lane >> 2, tq = lane & 3;
    const size_t row_stride = (size_t)heads * D;
    const unsigned int t0 = it * BT;
    const unsigned int ra = t0 + warp * 16 + gq, rb = ra + 8;
    const __nv_bfloat16 z = __float2bfloat16_rn(0.0f);
    const __nv_bfloat16* hp = h + ((size_t)it * heads + head) * D * D;

    auto qg = [&](unsigned int row, int k) -> __nv_bfloat16 {
        if (row >= tokens) return z;
        size_t idx = (size_t)row * row_stride + head * D + k;
        float qs = bf16r(fmul(__bfloat162float(q[idx]), scale));
        return to_bf16(fmul(qs, ex2_approx_ftz(g[idx])));
    };

    float acc[BV / 8][4];
    for (int n = 0; n < BV / 8; ++n)
        for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
    for (int ks = 0; ks < D / 16; ++ks) {
        const int kc = ks * 16 + 2 * tq;
        uint32_t a[4] = {pack_bf16x2(qg(ra, kc), qg(ra, kc + 1)), pack_bf16x2(qg(rb, kc), qg(rb, kc + 1)),
                         pack_bf16x2(qg(ra, kc + 8), qg(ra, kc + 9)), pack_bf16x2(qg(rb, kc + 8), qg(rb, kc + 9))};
        for (int n = 0; n < BV / 8; ++n) {
            const __nv_bfloat16* hv = hp + (size_t)(v0 + n * 8 + gq) * D;  // h[v][k]
            uint32_t b[2] = {ld_bf16x2(hv + kc), ld_bf16x2(hv + kc + 8)};
            mma_bf16_16816(acc[n], a, b, acc[n]);
        }
    }
    auto am = [&](unsigned int row, int s) -> __nv_bfloat16 {
        if (row >= tokens || (int)(row - t0) < s) return z;
        return to_bf16(aqk[((size_t)row * heads + head) * BT + s]);
    };
    auto vv = [&](int s, int col) -> __nv_bfloat16 {
        unsigned int row = t0 + s;
        return row < tokens ? v_new[(size_t)row * row_stride + head * D + v0 + col] : z;
    };
    for (int ss = 0; ss < BT / 16; ++ss) {
        const int sc = ss * 16 + 2 * tq;
        uint32_t a[4] = {pack_bf16x2(am(ra, sc), am(ra, sc + 1)), pack_bf16x2(am(rb, sc), am(rb, sc + 1)),
                         pack_bf16x2(am(ra, sc + 8), am(ra, sc + 9)), pack_bf16x2(am(rb, sc + 8), am(rb, sc + 9))};
        for (int n = 0; n < BV / 8; ++n) {
            const int col = n * 8 + gq;
            uint32_t b[2] = {pack_bf16x2(vv(sc, col), vv(sc + 1, col)), pack_bf16x2(vv(sc + 8, col), vv(sc + 9, col))};
            mma_bf16_16816(acc[n], a, b, acc[n]);
        }
    }
    for (int n = 0; n < BV / 8; ++n) {
        for (int r = 0; r < 4; ++r) {
            const unsigned int row = (r < 2 ? ra : rb);
            const int col = n * 8 + 2 * tq + (r & 1);
            if (row < tokens) o[(size_t)row * row_stride + head * D + v0 + col] = to_bf16(acc[n][r]);
        }
    }
}
