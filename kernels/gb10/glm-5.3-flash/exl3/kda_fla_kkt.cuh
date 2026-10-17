// SPDX-License-Identifier: AGPL-3.0-only

// EXL3 reference-policy KDA: KKT stage (A = beta * K K^T decay, Aqk = scaled
// Q K^T decay) for one sequence of T rows, mirroring the pinned FLA kernels
// chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter (off-diagonal 16x16
// blocks, TF32 mma) and ..._intra_sub_intra (diagonal blocks, f32 reductions).
//
// Layouts: q, k bf16 [T, H, 128]; g f32 [T, H, 128] (chunk-local cumsum,
// log2 space); beta f32 [T, H]; A, Aqk f32 [T, H, 64]. The caller zeroes A and
// Aqk (the reference allocates them with torch.zeros): the upper blocks are
// never written. Launch inter then intra on the same stream.
#pragma once
#include "kda_fla_ops.cuh"

namespace kda_fla_kkt {

constexpr int K = 128;
constexpr int BT = 64;
constexpr int BC = 16;
constexpr int NC = BT / BC;

__device__ __forceinline__ float ld_bf16(const __nv_bfloat16* p) { return __bfloat162float(*p); }

}  // namespace kda_fla_kkt

// Grid (ceil(T/64), NC*NC, H), block 32 (one warp per (chunk, i, j, head)).
// Matches the reference: A/Aqk accumulate in the mma accumulator across all
// K/8 tf32 k-steps in ascending order; A is scaled by beta per row afterwards.
extern "C" __global__ void __launch_bounds__(32) kda_fla_kkt_inter(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ A,
    float* __restrict__ Aqk,
    float scale,
    unsigned int T,
    unsigned int H
) {
    using namespace kda_fla;
    using namespace kda_fla_kkt;
    const int i_t = blockIdx.x, i_c = blockIdx.y, i_h = blockIdx.z;
    const int i_i = i_c / NC, i_j = i_c % NC;
    const int row0 = i_t * BT + i_i * BC;
    const int col0 = i_t * BT + i_j * BC;
    if (row0 >= (int)T || i_i <= i_j) return;

    const int lane = threadIdx.x;
    const int gq = lane >> 2, tq = lane & 3;
    const size_t rs = (size_t)H * K;  // row stride of q/k/g
    const size_t hoff = (size_t)i_h * K;
    const float* gn = g + (size_t)row0 * rs + hoff;  // first row of block i

    // Fragment rows (A operand) and columns (B operand) owned by this lane.
    const int ra = row0 + gq, rb = row0 + gq + 8;
    const bool va = ra < (int)T, vb = rb < (int)T;
    float acc_a[2][4], acc_q[2][4];
    for (int n = 0; n < 2; ++n)
        for (int e = 0; e < 4; ++e) acc_a[n][e] = acc_q[n][e] = 0.0f;

    // Triton's tf32 dot operands use kWidth=2: within each 16-wide k block
    // the first mma takes the even k (2t, 8+2t) and the second the odd ones.
    for (int step = 0; step < K / 8; ++step) {
        const int k16 = (step >> 1) * 16, par = step & 1;
        const int ka = k16 + 2 * tq + par, kb = k16 + 8 + 2 * tq + par;
        const float gna = gn[ka], gnb = gn[kb];
        // A-operand values at (row, k); out-of-range rows load zeros.
        float kx[4], qx[4];
        const int rows[4] = {ra, rb, ra, rb};
        const bool vr[4] = {va, vb, va, vb};
        const int ks[4] = {ka, ka, kb, kb};
        for (int e = 0; e < 4; ++e) {
            const size_t off = (size_t)rows[e] * rs + hoff + ks[e];
            const float gv = vr[e] ? g[off] : 0.0f;
            const float kv = vr[e] ? ld_bf16(k + off) : 0.0f;
            const float qv = vr[e] ? ld_bf16(q + off) : 0.0f;
            const float ex = ex2_approx_ftz(fsub(gv, ks[e] == ka ? gna : gnb));
            kx[e] = fmul(kv, ex);
            qx[e] = fmul(fmul(qv, ex), scale);
        }
        const uint32_t ak[4] = {__float_as_uint(kx[0]), __float_as_uint(kx[1]), __float_as_uint(kx[2]),
                                __float_as_uint(kx[3])};
        const uint32_t aq[4] = {__float_as_uint(qx[0]), __float_as_uint(qx[1]), __float_as_uint(qx[2]),
                                __float_as_uint(qx[3])};
        for (int n = 0; n < 2; ++n) {
            // B-operand values at (k, col): ktg = kt * exp2(gn - gk).
            const int col = col0 + n * 8 + gq;
            const bool vc = col < (int)T;
            uint32_t bk[2];
            const int kcol[2] = {ka, kb};
            for (int e = 0; e < 2; ++e) {
                const size_t off = (size_t)col * rs + hoff + kcol[e];
                const float gk = vc ? g[off] : 0.0f;
                const float kt = vc ? ld_bf16(k + off) : 0.0f;
                const float ex = ex2_approx_ftz(fsub(e == 0 ? gna : gnb, gk));
                bk[e] = __float_as_uint(fmul(kt, ex));
            }
            mma_tf32_1688(acc_a[n], ak, bk, acc_a[n]);
            mma_tf32_1688(acc_q[n], aq, bk, acc_q[n]);
        }
    }

    const float ba = va ? beta[(size_t)ra * H + i_h] : 0.0f;
    const float bb = vb ? beta[(size_t)rb * H + i_h] : 0.0f;
    for (int n = 0; n < 2; ++n) {
        const int c = i_j * BC + n * 8 + 2 * tq;  // column within the 64-wide chunk row
        if (va) {
            float* pa = A + ((size_t)ra * H + i_h) * BT + c;
            float* pq = Aqk + ((size_t)ra * H + i_h) * BT + c;
            pa[0] = fmul(acc_a[n][0], ba);
            pa[1] = fmul(acc_a[n][1], ba);
            pq[0] = acc_q[n][0];
            pq[1] = acc_q[n][1];
        }
        if (vb) {
            float* pa = A + ((size_t)rb * H + i_h) * BT + c;
            float* pq = Aqk + ((size_t)rb * H + i_h) * BT + c;
            pa[0] = fmul(acc_a[n][2], bb);
            pa[1] = fmul(acc_a[n][3], bb);
            pq[0] = acc_q[n][2];
            pq[1] = acc_q[n][3];
        }
    }
}

// Grid (ceil(T/64), NC, H), block 256 (8 warps x 2 rows; 16 lanes x 8 k per row).
// Per-thread reduction mirrors the reference PTX: acc = k1*x1, then fma over
// elements 0, 2..7, then butterfly adds over lane offsets 8, 4, 2, 1.
extern "C" __global__ void __launch_bounds__(256) kda_fla_kkt_intra(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ A,
    float* __restrict__ Aqk,
    float scale,
    unsigned int T,
    unsigned int H
) {
    using namespace kda_fla;
    using namespace kda_fla_kkt;
    const int i_t = blockIdx.x, i_i = blockIdx.y, i_h = blockIdx.z;
    const int row0 = i_t * BT + i_i * BC;
    if (row0 >= (int)T) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o_i = warp * 2 + (lane >> 4);
    const int k0 = (lane & 15) * 8;
    const int row = row0 + o_i;
    const bool vr = row < (int)T;
    const size_t rs = (size_t)H * K;
    const size_t hoff = (size_t)i_h * K;

    float kb[8], qv[8], gv[8];
    const float b = vr ? beta[(size_t)row * H + i_h] : 0.0f;
    for (int e = 0; e < 8; ++e) {
        const size_t off = (size_t)row * rs + hoff + k0 + e;
        kb[e] = fmul(vr ? ld_bf16(k + off) : 0.0f, b);
        qv[e] = vr ? ld_bf16(q + off) : 0.0f;
        gv[e] = vr ? g[off] : 0.0f;
    }
    const int jn = min(BC, (int)T - row0);
    for (int j = 0; j < jn; ++j) {
        const size_t offj = (size_t)(row0 + j) * rs + hoff + k0;
        float x[8];
        for (int e = 0; e < 8; ++e)
            x[e] = fmul(ex2_approx_ftz(fsub(gv[e], g[offj + e])), ld_bf16(k + offj + e));
        float sa = fmul(kb[1], x[1]);
        float sq = fmul(x[1], qv[1]);
        sa = ffma(kb[0], x[0], sa);
        sq = ffma(x[0], qv[0], sq);
        for (int e = 2; e < 8; ++e) {
            sa = ffma(kb[e], x[e], sa);
            sq = ffma(x[e], qv[e], sq);
        }
        for (int off = 8; off > 0; off >>= 1) {
            sa = fadd(sa, __shfl_xor_sync(0xffffffffu, sa, off));
            sq = fadd(sq, __shfl_xor_sync(0xffffffffu, sq, off));
        }
        if (vr && (lane & 15) == 0) {
            const size_t o = ((size_t)row * H + i_h) * BT + i_i * BC + j;
            A[o] = o_i > j ? sa : 0.0f;
            Aqk[o] = o_i >= j ? fmul(sq, scale) : 0.0f;
        }
    }
}
