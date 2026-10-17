// SPDX-License-Identifier: AGPL-3.0-only

// EXL3 reference-policy KDA: (I + A)^-1 per 64-row chunk and the WY
// recomputation (w, u, kg). Byte-exact ports of the pinned FLA Triton kernels
// `merge_16x16_to_64x64_inverse_kernel` (num_warps 2) and
// `recompute_w_u_fwd_kernel` (num_warps 4, BK = BV = 64) as lowered for GB10
// (docs/design/exl3-kda-reference-core.md §3.6a). Operation order follows the
// reference PTX, including LLVM's contraction of the first butterfly level of
// the forward-substitution reduction into an fma.
//
// Layouts (single sequence of T rows, HL heads, K = V = 128, chunk BT = 64):
//   A   f32  [T, HL, 64]   strictly lower within each chunk (from the KKT kernels)
//   Ai  bf16 [T, HL, 64]   (I + A)^-1 per chunk, upper blocks zero
//   k, kg, w bf16 [T, HL, 128]; g f32 [T, HL, 128] (chunk-local log2 cumsum)
//   v   bf16 rows with caller stride (the v plane of the packed conv output)
//   u   bf16 [T, HL, 128]; beta f32 [T, HL]
#pragma once
#include "kda_fla_ops.cuh"

namespace kda_fla_wy {

using namespace kda_fla;

constexpr int BT = 64;
constexpr int KD = 128;

// 16x16x16 f32 FMA dot (Triton ieee dot lowering): acc chain over k ascending,
// starting from `init` (0 or a previous dot's result). 64 threads, 4 outputs each.
__device__ __forceinline__ void dot16(float (*out)[16], const float (*x)[16], const float (*y)[16],
                                      const float (*init)[16], bool negate) {
    const int t = threadIdx.x;
    for (int e = 0; e < 4; ++e) {
        const int idx = t * 4 + e;
        const int m = idx >> 4, n = idx & 15;
        float acc = init ? init[m][n] : 0.0f;
        for (int kk = 0; kk < 16; ++kk) acc = ffma(x[m][kk], y[kk][n], acc);
        out[m][n] = negate ? fsub(0.0f, acc) : acc;
    }
}

}  // namespace kda_fla_wy

// grid (ceil(T/64), heads), block 64.
extern "C" __global__ void __launch_bounds__(64) kda_fla_solve_tril(const float* __restrict__ A,
                                                                   __nv_bfloat16* __restrict__ Ai,
                                                                   unsigned int T, unsigned int heads) {
    using namespace kda_fla;
    using namespace kda_fla_wy;
    // Blocks: d[0..3] diagonal (inverted in place), o[0..5] = A21 A31 A32 A41 A42 A43,
    // r[0..5] = results Ai21 Ai31 Ai32 Ai41 Ai42 Ai43, s[0..1] scratch.
    __shared__ float d[4][16][16];
    __shared__ float o[6][16][16];
    __shared__ float r[6][16][16];
    __shared__ float s[2][16][16];

    const int tid = threadIdx.x;
    const unsigned int c0 = blockIdx.x * BT;
    const unsigned int h = blockIdx.y;
    const int nvalid = (int)min((unsigned)BT, T - c0);
    const size_t row_stride = (size_t)heads * BT;
    const float* Ab = A + (size_t)c0 * row_stride + (size_t)h * BT;

    // Load with boundary zero fill (rows >= T read as 0).
    static const int obr[6] = {1, 2, 2, 3, 3, 3};
    static const int obc[6] = {0, 0, 1, 0, 1, 2};
    for (int e = tid; e < 16 * 16; e += 64) {
        const int rr = e >> 4, cc = e & 15;
        for (int b = 0; b < 4; ++b) {
            const int row = b * 16 + rr;
            d[b][rr][cc] = row < nvalid ? Ab[(size_t)row * row_stride + b * 16 + cc] : 0.0f;
        }
        for (int b = 0; b < 6; ++b) {
            const int row = obr[b] * 16 + rr;
            o[b][rr][cc] = row < nvalid ? Ab[(size_t)row * row_stride + obc[b] * 16 + cc] : 0.0f;
        }
    }
    __syncthreads();

    // Forward substitution on each diagonal block: one thread per (block, column).
    {
        const int b = tid >> 4, c = tid & 15;
        float col[16];
        for (int rr = 0; rr < 16; ++rr) col[rr] = rr > c ? fsub(0.0f, d[b][rr][c]) : 0.0f;
        const int lim = min(16, nvalid - b * 16);
        for (int i = 2; i < lim; ++i) {
            const float* arow = Ab + (size_t)(b * 16 + i) * row_stride + b * 16;
            float neg[16], p[16];
            for (int j = 0; j < 16; ++j) {
                neg[j] = fsub(0.0f, arow[j]);
                p[j] = fmul(col[j], neg[j]);
            }
            float w[2];
            for (int wp = 0; wp < 2; ++wp) {
                float v[4];
                for (int q = 0; q < 4; ++q) {
                    const int j = wp * 8 + q;
                    v[q] = ffma(col[j], neg[j], p[j + 4]);
                }
                w[wp] = fadd(fadd(v[0], v[2]), fadd(v[1], v[3]));
            }
            const float sum = fadd(w[0], w[1]);
            col[i] = fadd(fsub(0.0f, arow[c]), sum);
        }
        __syncthreads();
        for (int rr = 0; rr < 16; ++rr) d[b][rr][c] = fadd(col[rr], rr == c ? 1.0f : 0.0f);
    }
    __syncthreads();

    // Off-diagonal blocks (all dots are ieee f32 FMA chains).
    dot16(s[0], d[1], o[0], nullptr, false);  // Ai22 @ A21
    dot16(s[1], d[2], o[2], nullptr, false);  // Ai33 @ A32
    __syncthreads();
    dot16(r[0], s[0], d[0], nullptr, true);  // Ai21 = -(.) @ Ai11
    dot16(r[2], s[1], d[1], nullptr, true);  // Ai32 = -(.) @ Ai22
    __syncthreads();
    dot16(s[0], d[3], o[5], nullptr, false);  // Ai44 @ A43
    __syncthreads();
    dot16(r[5], s[0], d[2], nullptr, true);  // Ai43
    // Ai31 = -(Ai33 @ (A31 @ Ai11 + A32 @ Ai21)), chained accumulator.
    dot16(s[1], o[1], d[0], nullptr, false);
    __syncthreads();
    dot16(s[1], o[2], r[0], s[1], false);
    __syncthreads();
    dot16(r[1], d[2], s[1], nullptr, true);
    // Ai42 = -(Ai44 @ (A42 @ Ai22 + A43 @ Ai32)).
    dot16(s[0], o[4], d[1], nullptr, false);
    __syncthreads();
    dot16(s[0], o[5], r[2], s[0], false);
    __syncthreads();
    dot16(r[4], d[3], s[0], nullptr, true);
    __syncthreads();
    // Ai41 = -(Ai44 @ (A41 @ Ai11 + A42 @ Ai21 + A43 @ Ai31)).
    dot16(s[1], o[3], d[0], nullptr, false);
    __syncthreads();
    dot16(s[1], o[4], r[0], s[1], false);
    __syncthreads();
    dot16(s[1], o[5], r[1], s[1], false);
    __syncthreads();
    dot16(r[3], d[3], s[1], nullptr, true);
    __syncthreads();

    // Store every column of every valid row (upper blocks are zero).
    __nv_bfloat16* out = Ai + (size_t)c0 * row_stride + (size_t)h * BT;
    for (int e = tid; e < nvalid * BT; e += 64) {
        const int row = e >> 6, col = e & 63;
        const int rb = row >> 4, cb = col >> 4, rr = row & 15, cc = col & 15;
        float val = 0.0f;
        if (rb == cb) {
            val = d[rb][rr][cc];
        } else if (cb < rb) {
            // r index: (1,0)->0 (2,0)->1 (2,1)->2 (3,0)->3 (3,1)->4 (3,2)->5
            const int ri = rb == 1 ? 0 : rb == 2 ? 1 + cb : 3 + cb;
            val = r[ri][rr][cc];
        }
        out[(size_t)row * row_stride + col] = to_bf16(val);
    }
}

// grid (ceil(T/64), heads), block 128 (4 warps, 16 rows each).
// v points at the v plane of row 0 (head 0); element (t, h, d) is
// v[t * v_row_stride + h * 128 + d].
extern "C" __global__ void __launch_bounds__(128) kda_fla_recompute_w_u(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, unsigned int v_row_stride,
    const float* __restrict__ beta, const __nv_bfloat16* __restrict__ Ai, const float* __restrict__ g,
    __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ kg,
    unsigned int T, unsigned int heads) {
    using namespace kda_fla;
    using namespace kda_fla_wy;
    __shared__ __nv_bfloat16 sa[BT][BT + 8];
    __shared__ __nv_bfloat16 sx[BT][KD + 8];

    const int tid = threadIdx.x;
    const unsigned int c0 = blockIdx.x * BT;
    const unsigned int h = blockIdx.y;
    const int nvalid = (int)min((unsigned)BT, T - c0);
    const size_t hk = (size_t)heads * KD;
    const size_t ha = (size_t)heads * BT;

    for (int e = tid; e < BT * BT; e += 128) {
        const int rr = e >> 6, cc = e & 63;
        sa[rr][cc] = rr < nvalid ? Ai[(size_t)(c0 + rr) * ha + (size_t)h * BT + cc] : __float2bfloat16_rn(0.0f);
    }

    const int warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int m0 = warp * 16;

    // pass 0: u = Ai @ bf16(v * beta); pass 1: w = Ai @ bf16(k * beta * exp2(g)).
    for (int pass = 0; pass < 2; ++pass) {
        __syncthreads();
        for (int e = tid; e < BT * KD; e += 128) {
            const int rr = e >> 7, cc = e & 127;
            float val = 0.0f;
            if (rr < nvalid) {
                const unsigned int t = c0 + rr;
                const float b = beta[(size_t)t * heads + h];
                if (pass == 0) {
                    const float x = __bfloat162float(v[(size_t)t * v_row_stride + (size_t)h * KD + cc]);
                    val = fmul(x, b);
                } else {
                    const size_t idx = (size_t)t * hk + (size_t)h * KD + cc;
                    const float x = __bfloat162float(k[idx]);
                    val = fmul(fmul(x, b), ex2_approx_ftz(g[idx]));
                }
            }
            sx[rr][cc] = to_bf16(val);
        }
        __syncthreads();
        __nv_bfloat16* dst = pass == 0 ? u : w;
        for (int nt = 0; nt < KD / 8; ++nt) {
            const int n0 = nt * 8;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int ks = 0; ks < BT / 16; ++ks) {
                const int k0 = ks * 16;
                uint32_t a[4], bb[2];
                a[0] = pack_bf16x2(sa[m0 + gid][k0 + 2 * tig], sa[m0 + gid][k0 + 2 * tig + 1]);
                a[1] = pack_bf16x2(sa[m0 + gid + 8][k0 + 2 * tig], sa[m0 + gid + 8][k0 + 2 * tig + 1]);
                a[2] = pack_bf16x2(sa[m0 + gid][k0 + 2 * tig + 8], sa[m0 + gid][k0 + 2 * tig + 9]);
                a[3] = pack_bf16x2(sa[m0 + gid + 8][k0 + 2 * tig + 8], sa[m0 + gid + 8][k0 + 2 * tig + 9]);
                bb[0] = pack_bf16x2(sx[k0 + 2 * tig][n0 + gid], sx[k0 + 2 * tig + 1][n0 + gid]);
                bb[1] = pack_bf16x2(sx[k0 + 2 * tig + 8][n0 + gid], sx[k0 + 2 * tig + 9][n0 + gid]);
                float c[4] = {acc[0], acc[1], acc[2], acc[3]};
                mma_bf16_16816(acc, a, bb, c);
            }
            const int r0 = m0 + gid, r1 = m0 + gid + 8, cc = n0 + 2 * tig;
            if (r0 < nvalid) {
                const size_t base = (size_t)(c0 + r0) * hk + (size_t)h * KD + cc;
                dst[base] = to_bf16(acc[0]);
                dst[base + 1] = to_bf16(acc[1]);
            }
            if (r1 < nvalid) {
                const size_t base = (size_t)(c0 + r1) * hk + (size_t)h * KD + cc;
                dst[base] = to_bf16(acc[2]);
                dst[base + 1] = to_bf16(acc[3]);
            }
        }
    }

    // kg = k * exp2(g_last - g), g_last = the chunk's last valid row.
    const unsigned int last = c0 + nvalid - 1;
    for (int e = tid; e < nvalid * KD; e += 128) {
        const int rr = e >> 7, cc = e & 127;
        const size_t idx = (size_t)(c0 + rr) * hk + (size_t)h * KD + cc;
        const float gn = g[(size_t)last * hk + (size_t)h * KD + cc];
        kg[idx] = to_bf16(fmul(__bfloat162float(k[idx]), ex2_approx_ftz(fsub(gn, g[idx]))));
    }
}
