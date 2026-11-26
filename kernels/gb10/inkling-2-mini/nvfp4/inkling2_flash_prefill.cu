// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 prefill attention, one sequence's rows over its paged BF16 KV cache: a CTA
// takes one query head and 64 query tokens, its 4 warps 16 rows each, two CTAs an SM.
// Key tiles of 64, newest first; K and V tiles alternate in single buffers (V(t) loads
// while S = QK^T runs on K(t), K(t - 1) while PV runs on V(t)), rows XOR-swizzled for
// ldmatrix. Heads run fastest in the grid, so a KV head's query heads read its tiles from
// L2. The layout and the bias's sheared two-at-a-time reads follow the inkling_attn kernel
// of knapcio/Inkling-2-Mini-1x-DGX-Spark (MIT).
// Math of inkling2_prefill_attn (common/prefill_paged_compute.cuh with
// INKLING2_PREFILL_BIAS / INKLING2_ROW_SINKS): score = q.k / sqrt(d) + bias for
// q - k < rel_extent, causal and sliding-window masks, online softmax in fp32 with P
// rounded to BF16 for the PV MMA, one sink logit per (row, head) in the normaliser. Key
// tiles wholly outside a warp's window or causal range are skipped; only tiles on the
// diagonal or the window's edge, or with rows past the chunk, test keys. The bias is
// BF16(sum_j rel[row, head, j] * rel_proj[j, q - k]); its inputs are BF16 values, so the
// tensor cores take them as they are: per key tile a warp's 16 rows x the 79 distances the
// tile spans (10 m16n8k16 MMAs, fp32 accumulate), rounded to BF16 into the warp's shared
// memory (0 past rel_extent) and read back along the diagonals two at a time.
// Rounding differs from the 3-head, 12-warp, 32-key-tile kernel this replaces in the tile
// order and size, one rounding fewer in scale-plus-bias (an fma) and exp2 flushing
// denormal results. Measured 2026-11-26 against it on one GB10, 8,192 rows: sliding-window
// visits 3.09 -> 2.28 ms, a global visit after 24,576 keys 53.5 -> 41.3 ms, with output
// error against an FP64 reference no larger; 1K- and 8K-token prompts' teacher-forced NLL
// unchanged per document (median), the categories' means within build-to-build swings.
// With 32-key tiles, oldest first and the old arithmetic this code reproduces the old
// kernel bit for bit.
// The output leaves gated: BF16(BF16(o) * sigmoid(gate)), inkling2_gate_mul's arithmetic.
//
// grid (num_q_heads * ceil(q_len / 64)), block 128, dynamic shared memory FP_SMEM bytes.
// HDIM 128, d_rel 16, num_q_heads a multiple of num_kv_heads, block_size a power of two.

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip): mma.sync / cp.async / ldmatrix PTX. Never launched under
// cfg!(atlas_scale): inkling2/attn_ops.rs takes inkling2_prefill_attn instead.
#include "../../common/atlas_pdl.cuh"
extern "C" __global__ void inkling2_flash_prefill() { atlas_pdl_enter(); __builtin_trap(); }
#else
#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define FP_BQ 64        // query tokens a CTA
#define FP_BC 64        // keys a tile
#define FP_HD 128
#define FP_WARPS (FP_BQ / 16)
#define FP_THREADS (FP_WARPS * 32)
#define FP_D_REL 16
#define FP_BD 80                // distances a warp's bias tile spans (16 + FP_BC - 1, plus the odd-row shift)
#define FP_BP 88                // its BF16 row pitch (conflict-free staging)
#define FP_SMEM (FP_WARPS * 16 * FP_BP * 2)   // dynamic; inkling2/attn_ops.rs FLASH_SMEM

__device__ __forceinline__ void fp_cp16(unsigned int s, const void* gmem, bool pred) {
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void fp_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int fp_pack(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(const unsigned int*)&v;
}

// 2^x, flushing denormal results: __expf without fast-math takes a denormal-safe path, a
// compare, a select and a multiply more a score
__device__ __forceinline__ float fp_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
#define FP_LOG2E 1.4426950408889634f

// a K or V row's 16-byte chunk c sits at chunk c ^ (row & 7): ldmatrix rows conflict-free
__device__ __forceinline__ unsigned int fp_at(unsigned int base, unsigned int row, unsigned int chunk) {
    return base + (row * FP_HD + ((chunk ^ (row & 7)) * 8)) * 2;
}

extern "C" __global__ void __launch_bounds__(FP_THREADS, 2) inkling2_flash_prefill(
    const __nv_bfloat16* __restrict__ Q,         // [q_len, num_q_heads * 128]
    const __nv_bfloat16* __restrict__ K_cache,   // [blocks, block_size, num_kv_heads, 128]
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,               // [q_len, num_q_heads * 128], gated
    const __nv_bfloat16* __restrict__ gate,      // [q_len, num_q_heads * 128]
    const int* __restrict__ block_table,         // the sequence's table
    const __nv_bfloat16* __restrict__ rel,       // [q_len, num_q_heads, FP_D_REL]
    const uint2* __restrict__ rel_proj_lanes,    // [rel_extent + 1, 4] (weights.rs bias_lanes): row d,
                                                 // lane quad q: BF16 proj (2q, d), (2q+1, d), (2q+8, d),
                                                 // (2q+9, d); row rel_extent zero
    const __nv_bfloat16* __restrict__ sinks,     // [q_len, num_q_heads]
    const unsigned int q_len,
    const unsigned int q_offset,                 // absolute position of query row 0
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int sliding_window,           // 0 = full attention
    const unsigned int rel_extent,
    const float inv_sqrt_d
) {
    atlas_pdl_enter();
    __shared__ __align__(128) __nv_bfloat16 Ks[FP_BC * FP_HD];
    __shared__ __align__(128) __nv_bfloat16 Vs[FP_BC * FP_HD];
    extern __shared__ __align__(16) __nv_bfloat16 fp_bias[];   // [FP_WARPS][16][FP_BP]
    // heads fastest (a KV head's query heads read its tiles from L2), the last query
    // tiles (the most keys) first
    const unsigned int lin = gridDim.x - 1 - blockIdx.x;
    const unsigned int head = lin % num_q_heads, q0 = (lin / num_q_heads) * FP_BQ;
    const unsigned int kvh = head / (num_q_heads / num_kv_heads);
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int g8 = lane / 4, t2 = (lane % 4) * 2;
    const unsigned int r_base = q0 + warp * 16;                    // this warp's first row
    const unsigned int r0 = r_base + g8, r1 = r0 + 8;              // this thread's two rows
    const unsigned int qa0 = q_offset + r0, qa1 = q_offset + r1;    // their absolute positions
    const unsigned int kv_len = q_offset + q_len;
    const unsigned int wq = q_offset + r_base;                     // the warp's first absolute row
    const bool warp_rows = r_base < q_len;

    // the CTA's key range: causal end, window start, in whole tiles
    const unsigned int k_end = q_offset + min(q0 + FP_BQ, q_len);
    unsigned int k_begin = 0;
    if (sliding_window > 0 && q_offset + q0 + 1 > sliding_window) k_begin = q_offset + q0 + 1 - sliding_window;
    const int t_lo = (int)(k_begin / FP_BC), t_hi = (int)((k_end + FP_BC - 1) / FP_BC);   // [t_lo, t_hi)

    // Q fragments of this warp's 16 rows, 8 k16 steps of 4 registers
    unsigned int qf[FP_HD / 16][4];
    {
        const unsigned long long qs = (unsigned long long)num_q_heads * FP_HD;
        const __nv_bfloat16* p0 = Q + (unsigned long long)min(r0, q_len - 1) * qs + head * FP_HD;
        const __nv_bfloat16* p1 = Q + (unsigned long long)min(r1, q_len - 1) * qs + head * FP_HD;
#pragma unroll
        for (int k = 0; k < FP_HD / 16; k++) {
            qf[k][0] = *(const unsigned int*)(p0 + k * 16 + t2);
            qf[k][1] = *(const unsigned int*)(p1 + k * 16 + t2);
            qf[k][2] = *(const unsigned int*)(p0 + k * 16 + t2 + 8);
            qf[k][3] = *(const unsigned int*)(p1 + k * 16 + t2 + 8);
        }
    }
    // this warp's rows' relative-position features: the bias MMA's A fragment
    unsigned int ra[4];
    {
        const __nv_bfloat16* p0 = rel + ((unsigned long long)min(r0, q_len - 1) * num_q_heads + head) * FP_D_REL;
        const __nv_bfloat16* p1 = rel + ((unsigned long long)min(r1, q_len - 1) * num_q_heads + head) * FP_D_REL;
        ra[0] = *(const unsigned int*)(p0 + t2);
        ra[1] = *(const unsigned int*)(p1 + t2);
        ra[2] = *(const unsigned int*)(p0 + t2 + 8);
        ra[3] = *(const unsigned int*)(p1 + t2 + 8);
    }

    const unsigned long long page = (unsigned long long)block_size * num_kv_heads * FP_HD;
    const unsigned long long rowp = (unsigned long long)num_kv_heads * FP_HD;
    const unsigned int ks = (unsigned int)__cvta_generic_to_shared(Ks), vs = (unsigned int)__cvta_generic_to_shared(Vs);
    // 64 keys x 128 dims of K or V: 1024 chunks of 16 bytes, 8 a thread; keys outside
    // [k_begin, kv_len) are zero-filled. The block size is a power of two (attn_ops.rs), so
    // keys are located by shifts: a division took an eighth of the kernel's time.
    const unsigned int bs_shift = __ffs(block_size) - 1, bs_mask = block_size - 1;
    auto issue = [&](unsigned int dst, const __nv_bfloat16* src, int t) {
        const unsigned int chunk = tid % 16;
#pragma unroll
        for (int i = 0; i < FP_BC * 16 / FP_THREADS; i++) {
            const unsigned int row = tid / 16 + i * (FP_THREADS / 16);
            const unsigned int pos = (unsigned int)t * FP_BC + row;
            const bool ok = pos >= k_begin && pos < kv_len;
            const unsigned long long off = ok ? (unsigned long long)block_table[pos >> bs_shift] * page
                                                    + (unsigned long long)(pos & bs_mask) * rowp + kvh * FP_HD + chunk * 8
                                              : 0ull;
            fp_cp16(fp_at(dst, row, chunk), src + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[FP_HD / 8][4];
#pragma unroll
    for (int i = 0; i < FP_HD / 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    __nv_bfloat16* tb = fp_bias + warp * 16 * FP_BP;
    const unsigned int par = g8 & 1;   // odd rows staged one column right: their sheared pairs stay 4-byte aligned

    if (t_hi > t_lo) issue(ks, K_cache, t_hi - 1);
    for (int t = t_hi - 1; t >= t_lo; t--) {
        const unsigned int kt = (unsigned int)t * FP_BC;
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();   // K(t) visible; every warp is done with V(t + 1)
        issue(vs, V_cache, t);

        // this warp's rows see keys [wq + 15 - (window - 1), wq + 15]
        const bool active = warp_rows && kt <= wq + 15
                            && (sliding_window == 0 || kt + FP_BC + sliding_window > wq + 1);
        float s[FP_BC / 8][4];
        if (active) {
            // S = Q K^T over this tile: 8 n8 tiles of keys
#pragma unroll
            for (int n = 0; n < FP_BC / 8; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < FP_HD / 16; k++) {
#pragma unroll
                for (int np = 0; np < FP_BC / 16; np++) {
                    const unsigned int key = 8 * (2 * np + (lane >> 4)) + (lane & 7);
                    const unsigned int chunk = 2 * k + ((lane >> 3) & 1);
                    unsigned int b0, b1, b2, b3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(fp_at(ks, key, chunk)));
                    fp_mma(s[2 * np], qf[k], b0, b1);
                    fp_mma(s[2 * np + 1], qf[k], b2, b3);
                }
            }
            // bias[row a][key b] at distance d = wq + a - (kt + b) = d_lo + (a + FP_BC - 1 - b):
            // B column i is rel_proj row d_lo + i (the zero row outside 0 .. rel_extent), one
            // 8-byte load a lane
            const int d_lo = (int)wq - (int)kt - (FP_BC - 1);
            const bool biased = d_lo < (int)rel_extent;
            if (biased) {
#pragma unroll
                for (int nb = 0; nb < FP_BD / 8; nb++) {
                    const int d = d_lo + nb * 8 + (int)g8;
                    const unsigned int row = d >= 0 && d < (int)rel_extent ? (unsigned int)d : rel_extent;
                    const uint2 b = __ldg(rel_proj_lanes + row * 4 + t2 / 2);
                    const unsigned int b0 = b.x, b1 = b.y;
                    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    fp_mma(c, ra, b0, b1);
                    const unsigned int col = nb * 8 + t2 + par;   // <= 80 < FP_BP
                    tb[g8 * FP_BP + col] = __float2bfloat16(c[0]);
                    tb[g8 * FP_BP + col + 1] = __float2bfloat16(c[1]);
                    tb[(g8 + 8) * FP_BP + col] = __float2bfloat16(c[2]);
                    tb[(g8 + 8) * FP_BP + col + 1] = __float2bfloat16(c[3]);
                }
                __syncwarp();
            }
            // scale, then the bias (staged as exactly 0 past its reach), then the masks: only
            // tiles on the diagonal or the window's edge, or with rows past the chunk, test keys.
            // Uniform branches around whole loops: per-score branches had the compiler copy
            // the softmax for every path, doubling the kernel's code.
            if (biased) {
#pragma unroll
                for (int n = 0; n < FP_BC / 8; n++) {
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        // keys (c, c + 1) of row a sit at columns (a + 63 - c, a + 62 - c), shifted by par
                        const unsigned int a = g8 + 8 * h, c = n * 8 + t2;
                        const unsigned int w = *(const unsigned int*)(tb + a * FP_BP + a + (FP_BC - 2) - c + par);
                        s[n][2 * h] = fmaf(s[n][2 * h], inv_sqrt_d, __uint_as_float(w & 0xffff0000u));
                        s[n][2 * h + 1] = fmaf(s[n][2 * h + 1], inv_sqrt_d, __uint_as_float(w << 16));
                    }
                }
            } else {
#pragma unroll
                for (int n = 0; n < FP_BC / 8; n++)
#pragma unroll
                    for (int j = 0; j < 4; j++) s[n][j] *= inv_sqrt_d;
            }
            const bool edge = kt + FP_BC - 1 > wq || (sliding_window > 0 && kt + sliding_window < wq + 16)
                              || r_base + 15 >= q_len;
            if (edge) {
#pragma unroll
                for (int n = 0; n < FP_BC / 8; n++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const unsigned int qa = j < 2 ? qa0 : qa1, r = j < 2 ? r0 : r1;
                        const unsigned int key = kt + n * 8 + t2 + (j & 1);
                        const bool live = r < q_len && key <= qa && (sliding_window == 0 || qa - key < sliding_window);
                        if (!live) s[n][j] = -INFINITY;
                    }
                }
            }
            float tmax0 = -INFINITY, tmax1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < FP_BC / 8; n++) {
                tmax0 = fmaxf(tmax0, fmaxf(s[n][0], s[n][1]));
                tmax1 = fmaxf(tmax1, fmaxf(s[n][2], s[n][3]));
            }
            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffff, tmax0, 1));
            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffff, tmax0, 2));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffff, tmax1, 1));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffff, tmax1, 2));
            // a row with no visible key yet keeps m = -inf: exponents relative to 0 there
            // (every term 0)
            const float nm0 = fmaxf(m0, tmax0), nm1 = fmaxf(m1, tmax1);
            const float u0 = nm0 == -INFINITY ? 0.0f : nm0, u1 = nm1 == -INFINITY ? 0.0f : nm1;
            const float e0 = fp_ex2((m0 - u0) * FP_LOG2E), e1 = fp_ex2((m1 - u1) * FP_LOG2E);
            m0 = nm0;
            m1 = nm1;
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int n = 0; n < FP_BC / 8; n++) {
                s[n][0] = fp_ex2((s[n][0] - u0) * FP_LOG2E);
                s[n][1] = fp_ex2((s[n][1] - u0) * FP_LOG2E);
                s[n][2] = fp_ex2((s[n][2] - u1) * FP_LOG2E);
                s[n][3] = fp_ex2((s[n][3] - u1) * FP_LOG2E);
                ps0 += s[n][0] + s[n][1];
                ps1 += s[n][2] + s[n][3];
            }
            ps0 += __shfl_xor_sync(0xffffffff, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffff, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffff, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffff, ps1, 2);
            l0 = l0 * e0 + ps0;
            l1 = l1 * e1 + ps1;
#pragma unroll
            for (int i = 0; i < FP_HD / 8; i++) {
                acc[i][0] *= e0;
                acc[i][1] *= e0;
                acc[i][2] *= e1;
                acc[i][3] *= e1;
            }
        }

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();   // V(t) visible; every warp is done with K(t)
        if (t - 1 >= t_lo) issue(ks, K_cache, t - 1);

        if (active) {
            // O += P V: P (BF16) from the score fragments, V^T fragments by ldmatrix.trans
#pragma unroll
            for (int kk = 0; kk < FP_BC / 16; kk++) {
                const unsigned int pa[4] = {fp_pack(s[2 * kk][0], s[2 * kk][1]), fp_pack(s[2 * kk][2], s[2 * kk][3]),
                                            fp_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                            fp_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
                for (int dp = 0; dp < FP_HD / 16; dp++) {
                    const unsigned int i4 = lane >> 3;
                    const unsigned int key = 16 * kk + 8 * (i4 & 1) + (lane & 7);
                    const unsigned int chunk = 2 * dp + (i4 >> 1);
                    unsigned int v0, v1, v2, v3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(fp_at(vs, key, chunk)));
                    fp_mma(acc[2 * dp], pa, v0, v1);
                    fp_mma(acc[2 * dp + 1], pa, v2, v3);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!warp_rows) return;

    // the sink: one extra logit with a zero value joins the normaliser
    const float sg0 = __bfloat162float(sinks[(unsigned long long)min(r0, q_len - 1) * num_q_heads + head]);
    const float sg1 = __bfloat162float(sinks[(unsigned long long)min(r1, q_len - 1) * num_q_heads + head]);
    {
        const float n0 = fmaxf(m0, sg0), n1 = fmaxf(m1, sg1);
        const float e0 = __expf(m0 - n0), e1 = __expf(m1 - n1);
        l0 = l0 * e0 + __expf(sg0 - n0);
        l1 = l1 * e1 + __expf(sg1 - n1);
#pragma unroll
        for (int i = 0; i < FP_HD / 8; i++) {
            acc[i][0] *= e0;
            acc[i][1] *= e0;
            acc[i][2] *= e1;
            acc[i][3] *= e1;
        }
    }
    // the attention output BF16(acc / l), then the output gate as inkling2_gate_mul applies
    // it: BF16(o * sigmoid(g))
    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    const unsigned long long os = (unsigned long long)num_q_heads * FP_HD;
    auto gated = [&](float a, float b, float inv, unsigned long long at) {
        const __nv_bfloat162 o = __floats2bfloat162_rn(a * inv, b * inv);
        const __nv_bfloat162 g = *(const __nv_bfloat162*)&gate[at];
        *(__nv_bfloat162*)&O[at] = __floats2bfloat162_rn(__low2float(o) / (1.0f + __expf(-__low2float(g))),
                                                          __high2float(o) / (1.0f + __expf(-__high2float(g))));
    };
#pragma unroll
    for (int d = 0; d < FP_HD / 8; d++) {
        const unsigned int col = head * FP_HD + d * 8 + t2;
        if (r0 < q_len) gated(acc[d][0], acc[d][1], inv0, (unsigned long long)r0 * os + col);
        if (r1 < q_len) gated(acc[d][2], acc[d][3], inv1, (unsigned long long)r1 * os + col);
    }
}
#endif  // AMD
