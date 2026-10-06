// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) exact GEMVs over 9..32 rows in ONE weight
// pass (ATLAS_QWEN4EXP_BATCH_FAST=1, `layers/ops/qwen4exp_wide_rows.rs`):
// the 16- and 32-row tiers above the 8-row `dense_gemv_bf16_batchm` and the
// 4-row `w4a16_gemv_qg_batch4`, for a batched speculative verify of C4..C8
// sequences x K=4 rows.
//
//   qwen4exp_bf16_rows16/32  row t byte-identical to `dense_gemv_bf16` on row t
//   qwen4exp_qg_rows16/32    row t byte-identical to `w4a16_gemv_qg` on row t
//
// THE INVARIANT. Every (row t, output n) is one FP32 accumulator walked
// exactly as the single-row kernel walks it, and nothing else touches it:
// "virtual lane" l in 0..63 of the output takes the 8-element vectors
// kv = l, l + 64, l + 128, ... < K/8 in order, each vector's elements in the
// order lo, hi of words 0..3, `acc += a * w` as a separate multiply and add
// (this directory builds with --fmad=false), then lanes 0..31 and 32..63
// each reduce by the shfl_down 16/8/4/2/1 tree and the output is
// `lane0 + lane32` rounded once to BF16. The rows of a launch only add
// independent accumulators, so a row's bits depend neither on M nor on which
// rows share the launch (tested at every M = 1..32,
// scripts/dev/qwen4exp_wide_rows_bench.cu).
//
// WHAT CHANGES vs the narrow tiers is only where a row's activation comes
// from: a K-step (64 vectors, 1 KiB a row) of every row is staged in shared
// memory by cp.async, so the 64 x NPB threads of a CTA read the activation
// rows once per CTA from L2 instead of once per output from L1 (at 32 rows
// the activation block is 160 KiB, past L1). The weight vector of the next
// step is loaded into registers before the barrier, so a CTA keeps a weight
// load in flight while it computes.
//
//   tier     CTA          stages  regs  shared   bound at its full width
//   rows16   4 outputs    2       56    32 KiB   DRAM (16 BF16 rows cost 8)
//   rows32   8 outputs    1       64    32 KiB   issue (2 CTA/SM; 64 KiB of
//                                                 double buffer would be 1)
//
// GB10 (ennspark03), us per call, weights streamed from DRAM, against the
// chunks the lane ran before (scripts/dev/qwen4exp_wide_rows_bench.cu):
//
//   rows                          8      16      24      32
//   GDN qkvz TP2   8-row chunks  182     398     575     781
//                  rows16/32     173     178     258     310
//   LM head half   8-row chunks  2.80ms  5.55ms  8.38ms  11.09ms
//                  rows16/32     2.63ms  2.55ms  3.86ms  4.49ms
//   Q+gate TP2     4-row chunks   83     161     238     324
//                  rows16/32       -     130     202     242
//
// At <= 8 rows the narrow kernels stay (they are as fast there); 32 rows are
// issue-bound on the per-row BF16 unpack + separate multiply and add, which
// --fmad=false and the bit-identity forbid fusing.
//
// Layouts: A [M, K] BF16 rows K apart; C row t at C + t * out_stride.
//   BF16  B [N, K] BF16.
//   QG    B [N, K/2] packed E2M1, S [N, K/16] E4M3 block scales, s2 FP32;
//         n indexes the interleaved [Q_h(hd) | G_h(hd)] rows and lands
//         deinterleaved [Q | G], as `w4a16_gemv_qg` writes it.
// K must be a multiple of 8 (BF16) / 16 (QG): no scalar tail (the host
// refuses other K). Grid: (ceil(N / NPB), 1, 1), block (64 * NPB, 1, 1).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QW_TPO 64   // threads per output: the reference's two warps
#define QW_WARP 32

__device__ __constant__ float QW_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

struct QwArgs {
    const __nv_bfloat16* A;
    const unsigned char* B;   // BF16 or packed E2M1 weight rows
    const unsigned char* S;   // E4M3 scales (QG)
    float s2;                 // second-level scale (QG)
    __nv_bfloat16* C;
    unsigned M, N, K, out_stride, heads, head_dim;
};

// BF16 weights: one 16-byte vector a step, as `dense_gemv_bf16` loads it.
struct QwBf16 {
    typedef uint4 Raw;
    static __device__ __forceinline__ Raw load(const QwArgs& p, unsigned n, unsigned kv) {
        return ((const uint4*)(p.B + (unsigned long long)n * p.K * 2))[kv];
    }
    static __device__ __forceinline__ void decode(const QwArgs&, const Raw& r, const float*,
                                                  float w[8]) {
        const unsigned br[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            w[2 * i] = __uint_as_float(br[i] << 16);              // exact BF16 -> FP32
            w[2 * i + 1] = __uint_as_float(br[i] & 0xFFFF0000u);
        }
    }
    static __device__ __forceinline__ unsigned out_col(const QwArgs&, unsigned n) { return n; }
};

// NVFP4 Q+gate: 4 packed bytes and their group scale a step, each weight
// `lut[nibble] * (e4m3(scale) * s2)`, as `w4a16_gemv_qg` decodes it.
struct QwQg {
    struct Raw { unsigned packed; unsigned char scale; };
    static __device__ __forceinline__ Raw load(const QwArgs& p, unsigned n, unsigned kv) {
        Raw r;
        r.packed = *(const unsigned*)(p.B + (unsigned long long)n * (p.K / 2) + kv * 4);
        r.scale = p.S[(unsigned long long)n * (p.K / 16) + kv * 8 / 16];
        return r;
    }
    static __device__ __forceinline__ void decode(const QwArgs& p, const Raw& r, const float* lut,
                                                  float w[8]) {
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = r.scale;
        const float scale = (float)fp8 * p.s2;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const unsigned byte_val = (r.packed >> (b * 8)) & 0xFFu;
            w[2 * b] = lut[byte_val & 0xFu] * scale;
            w[2 * b + 1] = lut[byte_val >> 4] * scale;
        }
    }
    // Interleaved [Q_h | G_h] row n -> deinterleaved [Q | G] column.
    static __device__ __forceinline__ unsigned out_col(const QwArgs& p, unsigned n) {
        const unsigned group = 2 * p.head_dim, h = n / group, idx = n % group;
        return idx < p.head_dim ? h * p.head_dim + idx
                                : p.heads * p.head_dim + h * p.head_dim + (idx - p.head_dim);
    }
};

__device__ __forceinline__ void qw_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}

template <int MAX_M, int NPB, int STAGES, typename W>
__device__ __forceinline__ void qw_rows_impl(const QwArgs p) {
    constexpr unsigned NT = NPB * QW_TPO;
    const unsigned local_out = threadIdx.x / QW_TPO;
    const unsigned lane = threadIdx.x % QW_TPO;
    const unsigned n = blockIdx.x * NPB + local_out;
    const bool live = n < p.N;
    const unsigned M = p.M < MAX_M ? p.M : MAX_M;
    const unsigned K_VEC = p.K / 8;
    const unsigned steps = (K_VEC + QW_TPO - 1) / QW_TPO;

    __shared__ __align__(16) uint4 s_a[STAGES][MAX_M][QW_TPO];
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QW_E2M1_LUT[threadIdx.x];

    // Stage step `step`'s activation vectors of every row into buffer `buf`.
    auto stage = [&](unsigned step, unsigned buf) {
        for (unsigned i = threadIdx.x; i < M * QW_TPO; i += NT) {
            const unsigned t = i / QW_TPO, v = i % QW_TPO, kv = step * QW_TPO + v;
            if (kv < K_VEC)
                qw_cp16(&s_a[buf][t][v], (const uint4*)(p.A + (unsigned long long)t * p.K) + kv);
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc[MAX_M];
#pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    typename W::Raw next{};
    if (live && lane < K_VEC) next = W::load(p, n, lane);
    stage(0, 0);
    for (unsigned s = 0; s < steps; s++) {
        const typename W::Raw cur = next;
        const unsigned kv = s * QW_TPO + lane;
        if (live && kv + QW_TPO < K_VEC) next = W::load(p, n, kv + QW_TPO);
        if (STAGES == 2 && s + 1 < steps) {
            stage(s + 1, (s + 1) % STAGES);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        if (live && kv < K_VEC) {
            float w[8];
            W::decode(p, cur, s_lut, w);
            const uint4(*buf)[QW_TPO] = s_a[s % STAGES];
#pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned)t >= M) break;
                const uint4 a = buf[t][lane];
                const unsigned ar[4] = {a.x, a.y, a.z, a.w};
                float x = acc[t];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    x += __uint_as_float(ar[i] << 16) * w[2 * i];
                    x += __uint_as_float(ar[i] & 0xFFFF0000u) * w[2 * i + 1];
                }
                acc[t] = x;
            }
        }
        __syncthreads();
        if (STAGES == 1 && s + 1 < steps) stage(s + 1, 0);
    }

#pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        float a = acc[t];
#pragma unroll
        for (int off = QW_WARP / 2; off > 0; off >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, off);
        acc[t] = a;
    }
    // The warp partials of every (row, output), in the activation buffer
    // (free since the loop's last barrier).
    static_assert(sizeof(float) * MAX_M * NPB * 2 <= sizeof(s_a), "reduction fits s_a");
    float* red = (float*)s_a;
    if (threadIdx.x % QW_WARP == 0) {
#pragma unroll
        for (int t = 0; t < MAX_M; t++)
            if ((unsigned)t < M) red[(t * NPB + local_out) * 2 + lane / QW_WARP] = acc[t];
    }
    __syncthreads();
    // Thread `lane` of the output writes row `lane`: lane0 + lane32, as the
    // single-row kernel's `smem[w0] + smem[w1]`.
    if (live && lane < M) {
        const float r = red[(lane * NPB + local_out) * 2] + red[(lane * NPB + local_out) * 2 + 1];
        p.C[(unsigned long long)lane * p.out_stride + W::out_col(p, n)] = __float2bfloat16(r);
    }
}

// Production entry points, spelled out so the kernel-name check
// (scripts/dev/check_qwen4exp_kernel_names.py) sees them.
#define QW_BF16_PARAMS                                                                 \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B,          \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K, unsigned out_stride
#define QW_BF16_ARGS QwArgs{A, (const unsigned char*)B, nullptr, 0.0f, C, M, N, K, out_stride, 0, 0}
#define QW_QG_PARAMS                                                                   \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B_packed,   \
        const unsigned char *__restrict__ B_scale, const float scale2,                 \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K,             \
        unsigned out_stride, unsigned num_heads, unsigned head_dim
#define QW_QG_ARGS QwArgs{A, B_packed, B_scale, scale2, C, M, N, K, out_stride, num_heads, head_dim}

extern "C" __global__ void __launch_bounds__(4 * QW_TPO, 1) qwen4exp_bf16_rows16(QW_BF16_PARAMS) {
    qw_rows_impl<16, 4, 2, QwBf16>(QW_BF16_ARGS);
}
extern "C" __global__ void __launch_bounds__(8 * QW_TPO, 2) qwen4exp_bf16_rows32(QW_BF16_PARAMS) {
    qw_rows_impl<32, 8, 1, QwBf16>(QW_BF16_ARGS);
}
extern "C" __global__ void __launch_bounds__(4 * QW_TPO, 1) qwen4exp_qg_rows16(QW_QG_PARAMS) {
    qw_rows_impl<16, 4, 2, QwQg>(QW_QG_ARGS);
}
extern "C" __global__ void __launch_bounds__(8 * QW_TPO, 2) qwen4exp_qg_rows32(QW_QG_PARAMS) {
    qw_rows_impl<32, 8, 1, QwQg>(QW_QG_ARGS);
}

#ifdef QW_SWEEP
// Bench-only shapes (scripts/dev/qwen4exp_wide_rows_bench.cu sweep):
// rows, outputs per CTA, stages, min CTAs per SM.
#define QW_SWEEP_BF16(name, MAX_M, NPB, STAGES, MINB)                                   \
    extern "C" __global__ void __launch_bounds__(NPB * QW_TPO, MINB) name(QW_BF16_PARAMS) { \
        qw_rows_impl<MAX_M, NPB, STAGES, QwBf16>(QW_BF16_ARGS);                         \
    }
#define QW_SWEEP_QG(name, MAX_M, NPB, STAGES, MINB)                                     \
    extern "C" __global__ void __launch_bounds__(NPB * QW_TPO, MINB) name(QW_QG_PARAMS) { \
        qw_rows_impl<MAX_M, NPB, STAGES, QwQg>(QW_QG_ARGS);                             \
    }
QW_SWEEP_BF16(qw_sweep_bf16_m16_n8_s2, 16, 8, 2, 1)
QW_SWEEP_BF16(qw_sweep_bf16_m32_n16_s2, 32, 16, 2, 1)
QW_SWEEP_BF16(qw_sweep_bf16_m32_n4_s1, 32, 4, 1, 3)
QW_SWEEP_QG(qw_sweep_qg_m16_n8_s2, 16, 8, 2, 1)
QW_SWEEP_QG(qw_sweep_qg_m32_n16_s2, 32, 16, 2, 1)
QW_SWEEP_QG(qw_sweep_qg_m8_n4_s2, 8, 4, 2, 1)
#endif
