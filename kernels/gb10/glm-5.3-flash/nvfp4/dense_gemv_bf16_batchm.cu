// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense BF16 batched GEMV (M rows) for SM121 (GB10).
//
// The M-row generalisation of dense_gemv_bf16_batch2: computes M output rows
// from ONE pass over the BF16 weight matrix, so weight bandwidth is paid once
// instead of M times. Bit-identical to running dense_gemv_bf16 M times — each
// row's accumulator follows the exact same K-iteration order and reduction
// tree; the extra rows only add independent accumulators over the same loop.
// (KERNEL.toml builds this dir with --fmad=false, which is what makes that
// identity hold rather than merely "close".)
//
//   C[t, n] = dot(A[t, :], B[n, :])   for t in [0, M)
//
//   A: [M, K] BF16 (activation rows, contiguous — this is exactly the layout
//      the multi-seq decode path already has: `normed.offset(i * h * bf16)`)
//   B: [N, K] BF16 (weights, row-major)
//   C: M rows at C + t * out_stride (BF16 elements)
//
// `out_stride` decouples the output row stride from N so callers can write
// straight into per-token strided layouts (e.g. the multi-seq qkv buffer,
// whose rows are `per_seq_qkv` apart, not `N` apart).
//
// WHY THIS EXISTS: at decode, Laguna's q/k/v/o and shared-expert projections
// are BF16 (the checkpoint ships them unquantized and they stay that way), and
// the BF16 path had no batched tier — only the quantized paths did
// (w4a16_gemv_batch2/3/4, w8a16_gemv_batch2/4). So every sequence in a decode
// batch re-read the whole weight matrix, making 54% of the decode step scale
// linearly with concurrency.
//
// A tile-based GEMM is the wrong tool here: at M<=4 an M64-tile GEMM is ~94%
// padding, and was measured 3.6x SLOWER than the batched GEMV on this exact
// workload (see the note in multi_seq/qkv.rs::wide_verify_gemm).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8   // BF16 values per vectorized load (uint4 = 16 bytes)
#define MAX_M 8      // compile-time cap on the generic entry point

template <int ROWS>
__device__ __forceinline__ void dense_gemv_bf16_batchm_impl(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // rows at C + t*out_stride
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,               // BF16 elements between output rows
    float* __restrict__ smem
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int m = (M > ROWS) ? ROWS : M;

    float acc[ROWS];
    #pragma unroll
    for (int t = 0; t < ROWS; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

#if defined(__HIP_PLATFORM_AMD__)
    // v_dot2_f32_bf16 path: both operands are already packed bf16x2 in the
    // loaded words — no unpacking at all. Each fdot2 does the two products
    // and the accumulate in one instruction; at M>=4 the scalar unpack+FMA
    // chain was issue-bound, not bandwidth-bound. Products are bit-exact
    // bf16*bf16->f32 (identical to the float path's); only the intra-8
    // summation grouping differs (2-term dots vs serial adds).
    // Inline asm, NOT __builtin_amdgcn_fdot2: the builtin resolves its bf16x2
    // operands to llvm.amdgcn.fdot2(<2 x half>) — v_dot2_f32_F16 — silently
    // miscompiling bf16 pairs as fp16.
    // NOTE: an earlier variant software-pipelined the A-row loads into
    // a_cur/a_next[MAX_M] register arrays — the static MAX_M=8 arrays spilled
    // to local memory and REGRESSED qkvz 1.24ms -> 3.07ms. Do not prefetch
    // into per-row arrays here; the loads stay per-iteration.
    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        for (unsigned int t = 0; t < m; t++) {
            const uint4* At_vec = (const uint4*)(A + (unsigned long long)t * K);
            uint4 a_data = At_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            float a = acc[t];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0"
                             : "+v"(a) : "v"(a_raw[i]), "v"(b_raw[i]));
            }
            acc[t] = a;
        }
    }
#else
    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        // ONE weight load feeds every row — this is the whole point.
        uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        float bf[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            bf[2 * i] = __bfloat162float(b_lo);
            bf[2 * i + 1] = __bfloat162float(b_hi);
        }

        for (unsigned int t = 0; t < m; t++) {
            const uint4* At_vec = (const uint4*)(A + (unsigned long long)t * K);
            uint4 a_data = At_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            float a = acc[t];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                // Same add order as dense_gemv_bf16: lo then hi, per vector slot.
                a += __bfloat162float(a_lo) * bf[2 * i];
                a += __bfloat162float(a_hi) * bf[2 * i + 1];
            }
            acc[t] = a;
        }
    }
#endif

    // Scalar tail for K not divisible by VEC_SIZE (never hits for model dims).
    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    // 2 warps per output: cross-warp reduce via shared memory, per row.
    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) {
            smem[t * (N_PER_BLOCK * 2) + smem_idx] = acc[t];
        }
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const unsigned int base = t * (N_PER_BLOCK * 2) + local_out * 2;
            const float r = smem[base] + smem[base + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}

extern "C" __global__ void dense_gemv_bf16_batchm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    atlas_pdl_enter();
    __shared__ float smem[MAX_M * N_PER_BLOCK * 2];
    dense_gemv_bf16_batchm_impl<MAX_M>(A, B, C, M, N, K, out_stride, smem);
}

// Exact five-row verifier tier. Keeping ROWS compile-time constant avoids the
// generic M<=8 kernel's three unused accumulator lanes and lets ptxas keep the
// hot accumulator array register-resident for GLM's K=5 verification batch.
extern "C" __global__ void dense_gemv_bf16_batch5(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    __shared__ float smem[5 * N_PER_BLOCK * 2];
    dense_gemv_bf16_batchm_impl<5>(A, B, C, 5, N, K, out_stride, smem);
}

// Two independent exact-five-row BF16 projections with the same [N,K]
// shape. Grid Z selects the input, weight, and output plane; the underlying
// dot-product body and reduction order are unchanged.
extern "C" __global__ void dense_gemv_bf16_batch5_dual(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ B0,
    const __nv_bfloat16* __restrict__ B1,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N,
    unsigned int K
) {
    __shared__ float smem[5 * N_PER_BLOCK * 2];
    const bool second = blockIdx.z != 0u;
    const __nv_bfloat16* A = second ? A1 : A0;
    const __nv_bfloat16* B = second ? B1 : B0;
    __nv_bfloat16* C = second ? C1 : C0;
    dense_gemv_bf16_batchm_impl<5>(A, B, C, 5, N, K, N, smem);
}

// Three exact-five-row BF16 projections sharing one input and K dimension.
// Plane zero may use a smaller output width (GLM beta); planes one and two
// cover the equal-width f_a/g_a pair. CTAs beyond a plane's N return inside
// the unchanged projection body.
extern "C" __global__ void dense_gemv_bf16_batch5_triple_n(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B0,
    const __nv_bfloat16* __restrict__ B1,
    const __nv_bfloat16* __restrict__ B2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N0,
    unsigned int N12,
    unsigned int K
) {
    __shared__ float smem[5 * N_PER_BLOCK * 2];
    const unsigned int plane = blockIdx.z;
    const __nv_bfloat16* B = plane == 0u ? B0 : (plane == 1u ? B1 : B2);
    __nv_bfloat16* C = plane == 0u ? C0 : (plane == 1u ? C1 : C2);
    const unsigned int N = plane == 0u ? N0 : N12;
    dense_gemv_bf16_batchm_impl<5>(A, B, C, 5, N, K, N, smem);
}

// Generic-M (<= MAX_M) twins of the two fused five-row tiers above: each
// plane runs the unchanged dense_gemv_bf16_batchm body, so it is bit-identical
// to its own batchm launch; fusing only puts the planes in one grid.
extern "C" __global__ void dense_gemv_bf16_batchm_dual(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ B0,
    const __nv_bfloat16* __restrict__ B1,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    atlas_pdl_enter();
    __shared__ float smem[MAX_M * N_PER_BLOCK * 2];
    const bool second = blockIdx.z != 0u;
    dense_gemv_bf16_batchm_impl<MAX_M>(second ? A1 : A0, second ? B1 : B0, second ? C1 : C0,
        M, N, K, N, smem);
}

extern "C" __global__ void dense_gemv_bf16_batchm_triple_n(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B0,
    const __nv_bfloat16* __restrict__ B1,
    const __nv_bfloat16* __restrict__ B2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    __nv_bfloat16* __restrict__ C2,
    unsigned int M,
    unsigned int N0,
    unsigned int N12,
    unsigned int K
) {
    atlas_pdl_enter();
    __shared__ float smem[MAX_M * N_PER_BLOCK * 2];
    const unsigned int plane = blockIdx.z;
    const __nv_bfloat16* B = plane == 0u ? B0 : (plane == 1u ? B1 : B2);
    __nv_bfloat16* C = plane == 0u ? C0 : (plane == 1u ? C1 : C2);
    dense_gemv_bf16_batchm_impl<MAX_M>(A, B, C, M, plane == 0u ? N0 : N12, K,
        plane == 0u ? N0 : N12, smem);
}

// K = 128 tier of dense_gemv_bf16_batchm_dual (GLM KDA f_b/g_b). At K = 128
// the generic dual feeds 48 of an output's 64 lanes exact zeros through the
// shuffle tree, a barrier and the cross-warp add: 12-26 us at N = 4096,
// M = 2..8 on GB10, vs 9.5-10.8 us here and an 8.5 us weight-read floor
// (scripts/dev/dense_gemv_dual_k128_bench.cu). A 16-lane group owns one
// output: each lane runs the generic single-slot product chain unchanged, the
// group reduces with the generic tree minus its offset-16 level (which only
// added +0.0), and "+ 0.0f" stands in for the add of the empty second warp.
// +0.0 only turns -0.0 into +0.0, so the outputs are bit-identical.
// Grid: (ceil(N / 16), 1, 2)   Block: (256, 1, 1)   K must be 128.
#define DUAL_K128_LANES 16

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) dense_gemv_bf16_batchm_dual_k128(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ B0,
    const __nv_bfloat16* __restrict__ B1,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    unsigned int M,
    unsigned int N,
    unsigned int   // K == DUAL_K128_LANES * VEC_SIZE
) {
    atlas_pdl_enter();
    const unsigned int K = DUAL_K128_LANES * VEC_SIZE;
    const bool second = blockIdx.z != 0u;
    const __nv_bfloat16* A = second ? A1 : A0;
    __nv_bfloat16* C = second ? C1 : C0;
    const unsigned int lane = threadIdx.x % DUAL_K128_LANES;
    const unsigned int n = blockIdx.x * (BLOCK_SIZE / DUAL_K128_LANES) + threadIdx.x / DUAL_K128_LANES;
    const bool live = n < N;  // no early return: the whole warp shuffles
    const unsigned int m = (M > MAX_M) ? MAX_M : M;

    uint4 b_data = make_uint4(0u, 0u, 0u, 0u);
    if (live) b_data = ((const uint4*)((second ? B1 : B0) + (unsigned long long)n * K))[lane];
    const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
    float bf[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_bfloat16 b_lo, b_hi;
        *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
        *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
        bf[2 * i] = __bfloat162float(b_lo);
        bf[2 * i + 1] = __bfloat162float(b_hi);
    }

    #pragma unroll
    for (unsigned int t = 0; t < MAX_M; t++) {
        if (t >= m) break;
        const uint4 a_data = ((const uint4*)(A + (unsigned long long)t * K))[lane];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            a += __bfloat162float(a_lo) * bf[2 * i];
            a += __bfloat162float(a_hi) * bf[2 * i + 1];
        }
        #pragma unroll
        for (int offset = DUAL_K128_LANES / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset, DUAL_K128_LANES);
        }
        if (live && lane == 0) C[(unsigned long long)t * N + n] = __float2bfloat16(a + 0.0f);
    }
}

// ============================================================
// BF16 GEMV on tensor cores for 9..32 rows (batched verify / drafter blocks).
// ============================================================
// dense_gemv_bf16_batchm caps at 8 rows; wider row counts used to loop it
// (re-reading the weight per 8 rows) or fall to a 128-row tiled GEMM that is
// mostly padding. Here `mma.m16n8k16` takes 16 weight rows as A and each
// 8-row slice of the activations as one B tile, so a CTA reads its weight
// slice ONCE for up to NT*8 rows. FP32 accumulation; not bit-identical to the
// scalar batch-M kernel.
//
// Mapping: CTA = 16 output rows, DG_TC_WARPS warps split K in 32-wide chunks.
// Lane (g = lane/4, c = lane%4) loads 8 consecutive K values (16 bytes) of
// weight rows g, g+8 and of activation row 8t+g for each tile t, prefetching
// its next chunk. MMA step j (0..1) feeds k-slots {2c,2c+1} <- values
// 4j+{0,1} and {2c+8,2c+9} <- 4j+{2,3} of the lane's run in both A and B.
//
// A:[M,K] BF16 (row stride K), B:[N,K] BF16, C rows at C + m*out_stride.
// K % 8 == 0. Grid: (ceil(N/16),1,1) Block: (DG_TC_WARPS*32,1,1).
#define DG_TC_WARPS 8

template <int NT>
struct DgTcChunk {
    uint4 w0, w1;
    uint4 x[NT];
};

template <int NT>
__device__ __forceinline__ void dg_tc_load(
    DgTcChunk<NT>& t, const __nv_bfloat16* w0p, const __nv_bfloat16* w1p,
    const __nv_bfloat16* A, unsigned int g, unsigned int M, unsigned int K,
    unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z;
    #pragma unroll
    for (int i = 0; i < NT; i++) t.x[i] = z;
    if (kb >= K) return;
    if (v0) t.w0 = *(const uint4*)(w0p + kb);
    if (v1) t.w1 = *(const uint4*)(w1p + kb);
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) t.x[i] = *(const uint4*)(A + (unsigned long long)row * K + kb);
    }
}

template <int NT>
__device__ __forceinline__ void dense_gemv_bf16_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    __shared__ float s_red[DG_TC_WARPS][WARP_SIZE][NT * 4];
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = blockIdx.x * 16u + g, r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N;
    const __nv_bfloat16* w0p = B + (unsigned long long)(v0 ? r0 : 0u) * K;
    const __nv_bfloat16* w1p = B + (unsigned long long)(v1 ? r1 : 0u) * K;

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    const unsigned int chunks = (K + 31u) / 32u;
    DgTcChunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks) dg_tc_load<NT>(cur, w0p, w1p, A, g, M, K, ch * 32u + 8u * c, v0, v1);
    for (; ch < chunks; ch += DG_TC_WARPS) {
        const unsigned int next = ch + DG_TC_WARPS;
        if (next < chunks) dg_tc_load<NT>(nxt, w0p, w1p, A, g, M, K, next * 32u + 8u * c, v0, v1);
        const unsigned int w0w[4] = {cur.w0.x, cur.w0.y, cur.w0.z, cur.w0.w};
        const unsigned int w1w[4] = {cur.w1.x, cur.w1.y, cur.w1.z, cur.w1.w};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            // Values 4j..4j+3 of each row's 8-value run = words 2j, 2j+1.
            const unsigned int a0 = w0w[2 * j], a1 = w1w[2 * j];
            const unsigned int a2 = w0w[2 * j + 1], a3 = w1w[2 * j + 1];
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                const unsigned int xw[4] = {cur.x[i].x, cur.x[i].y, cur.x[i].z, cur.x[i].w};
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(xw[2 * j]), "r"(xw[2 * j + 1]));
            }
        }
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) s_red[warp][lane][i * 4 + q] = acc[i][q];
    }
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < DG_TC_WARPS; w++) v += s_red[w][lane][i * 4 + q];
            // C fragment: q 0..1 -> output row g, 2..3 -> g+8; column 2c + (q&1)
            // of tile i is activation row 8i + 2c + (q&1).
            const unsigned int n = (q < 2) ? r0 : r1;
            const unsigned int m = 8u * i + 2u * c + (q & 1u);
            if (n < N && m < M) C[(unsigned long long)m * out_stride + n] = __float2bfloat16(v);
        }
    }
}

extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE)
dense_gemv_bf16_tc16(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int out_stride
) {
    dense_gemv_bf16_tc_impl<2>(A, B, C, M, N, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE)
dense_gemv_bf16_tc32(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int out_stride
) {
    dense_gemv_bf16_tc_impl<4>(A, B, C, M, N, K, out_stride);
}
