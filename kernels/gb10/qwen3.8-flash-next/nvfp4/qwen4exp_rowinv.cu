// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) mHC prefill collapse, down + injection
// projections, ROW-INVARIANT (ATLAS_QWEN4EXP_PREFILL_ROWINV=1).
//
// WHY. The prefill collapse's skinny projections (down N = rank = 320 and
// injection N = hc = 4, both K = hc*H = 10240) went to cuBLASLt's first
// heuristic pick, whose split-K changes with the slab height M: a row got
// other bits when other rows shared its launch. This kernel is the decode
// collapse's tensor-core down (`qwen4exp_hc_mma.cu` hc_mma_down) on the
// prefill's BF16 `normed`:
//
//   * a row's token is its own mma column (n), and an mma column never sees
//     another column, so a row's bits do not depend on its neighbours or on
//     where it sits; rows past T read a clamped copy of row T-1 and are
//     never stored;
//   * K is split over the HCR_CL CTAs of a cluster (fixed slices), each
//     slice one in-order k-chain in the fixed permuted order of hc_mma_down
//     (lane q takes physical k 8q..8q+7 of a 32-k chunk), and the slices are
//     summed in rank order through distributed shared memory: no atomics, no
//     global partials, nothing that depends on M;
//   * more than 96 rows run as groups of 96 on blockIdx.z, each group the
//     same body on its own rows (the group size is launch geometry: a
//     column's chain does not depend on how many columns share it).
//
// Outputs feed the prefill collapse's fused up + mix (`hc_up_mix_bf16_nt`):
// low = bf16(silu(v / hc)) [T, rank], inj_pre = bf16(v) [T, hc].
//
// Grid (ceil(ceil(rows / 16) / HCR_WM), HCR_CL, ceil(T / 96)), block
// 32 * HCR_WM, rows = rank + hc (rank alone for the head). The host checks
// K % (32 * HCR_CL * HCR_U) == 0.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include "../../common/atlas_pdl.cuh"

#ifndef HCR_WM
#define HCR_WM 7u  // m16 weight tiles a CTA (rank + hc = 324 rows = 21 tiles = 3 x 7)
#endif
#define HCR_CL 8u  // cluster CTAs splitting K (portable max)
#define HCR_U 4u   // weight ring, 32-k chunks
#define HCR_MAX_T 96u  // rows a group (12 n8 token tiles)

__device__ __forceinline__ void hcr_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Weights: read-only, no L1 allocation.
__device__ __forceinline__ void hcr_ldw(const __nv_bfloat16* p, unsigned (&w)[4]) {
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p));
}

// The predecessor's `normed`, as a volatile load (kept below the PDL wait).
__device__ __forceinline__ uint4 hcr_ldn(const __nv_bfloat16* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float hcr_silu(float v) { return v / (1.0f + __expf(-v)); }

// DN: normed ring depth in 32-k chunks (load scheduling only).
template <unsigned NT, unsigned DN>
__device__ __forceinline__ void hcr_down(
    const __nv_bfloat16* __restrict__ normed, const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ inject_w, __nv_bfloat16* __restrict__ low_out,
    __nv_bfloat16* __restrict__ inj_out, const unsigned K, const unsigned hc, const unsigned rank,
    const unsigned T, float* s_red
) {
    constexpr unsigned WM = HCR_WM, CL = HCR_CL, U = HCR_U;
    constexpr unsigned E = WM * 16u * NT * 8u;  // CTA partial: [WM*16 rows][NT*8 tokens]
    static_assert(DN >= 1 && U % DN == 0, "normed ring must tile the weight ring");

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const unsigned crank = cluster.block_rank();
    const unsigned lane = threadIdx.x & 31u, wm = threadIdx.x >> 5;
    const unsigned g = lane >> 2, q = lane & 3u;
    const unsigned rows = rank + (inject_w != nullptr ? hc : 0u);
    const unsigned rcta = blockIdx.x * WM * 16u;
    const unsigned row0 = rcta + wm * 16u;
    const unsigned kw = K / CL;
    const unsigned kbeg = crank * kw + 8u * q;
    auto wrow = [&](unsigned r) -> const __nv_bfloat16* {
        return r < rank ? down_w + (size_t)r * K
             : r < rows ? inject_w + (size_t)(r - rank) * K
                        : down_w;  // padding row: read, never stored
    };
    const __nv_bfloat16* wa = wrow(row0 + g) + kbeg;
    const __nv_bfloat16* wb = wrow(row0 + g + 8u) + kbeg;
    const __nv_bfloat16* nx[NT];
    #pragma unroll
    for (unsigned nt = 0; nt < NT; ++nt) nx[nt] = normed + (size_t)min(nt * 8u + g, T - 1u) * K + kbeg;
    const unsigned nc = kw / 32u;

    float acc[NT][4];
    #pragma unroll
    for (unsigned nt = 0; nt < NT; ++nt) {
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) acc[nt][c] = 0.0f;
    }
    unsigned wr[U][2][4];
    uint4 nr[DN][NT];
    #pragma unroll
    for (unsigned u = 0; u < U; ++u) {
        hcr_ldw(wa + (size_t)u * 32u, wr[u][0]);
        hcr_ldw(wb + (size_t)u * 32u, wr[u][1]);
    }
    #pragma unroll
    for (unsigned d = 0; d < DN; ++d) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) nr[d][nt] = hcr_ldn(nx[nt] + d * 32u);
    }
    // One 32-k chunk: physical k 8q..8q+3 are the A/B slots {2q, 2q+1,
    // 2q+8, 2q+9} of the first m16n8k16, 8q+4..8q+7 of the second (the same
    // k <-> slot map on A and B).
    auto step = [&](const unsigned u, const unsigned c, const bool rw, const bool rn) {
        const unsigned a0[4] = {wr[u][0][0], wr[u][1][0], wr[u][0][1], wr[u][1][1]};
        const unsigned a1[4] = {wr[u][0][2], wr[u][1][2], wr[u][0][3], wr[u][1][3]};
        if (rw) {
            hcr_ldw(wa + (size_t)(c + U) * 32u, wr[u][0]);
            hcr_ldw(wb + (size_t)(c + U) * 32u, wr[u][1]);
        }
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            const uint4 b = nr[u % DN][nt];
            if (rn) nr[u % DN][nt] = hcr_ldn(nx[nt] + (c + DN) * 32u);
            hcr_mma(acc[nt], a0, b.x, b.y);
            hcr_mma(acc[nt], a1, b.z, b.w);
        }
    };
    unsigned c0 = 0;
    for (; c0 + U < nc; c0 += U) {
        #pragma unroll
        for (unsigned u = 0; u < U; ++u) step(u, c0 + u, true, true);
    }
    #pragma unroll
    for (unsigned u = 0; u < U; ++u) step(u, c0 + u, false, u + DN < U);

    // This lane's (row, token) of the CTA partial.
    auto at = [&](unsigned nt, unsigned c) {
        return (wm * 16u + g + (c >> 1) * 8u) * (NT * 8u) + nt * 8u + 2u * q + (c & 1u);
    };
    #pragma unroll
    for (unsigned nt = 0; nt < NT; ++nt) {
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) s_red[at(nt, c)] = acc[nt][c];
    }
    // Fixed-order K reduction over the cluster ranks 0, 1, ...
    cluster.sync();
    const float inv_hc = 1.0f / (float)hc;
    for (unsigned e = crank * blockDim.x + threadIdx.x; e < E; e += CL * blockDim.x) {
        float x = cluster.map_shared_rank(s_red, 0)[e];
        #pragma unroll
        for (unsigned k = 1; k < CL; ++k) x += cluster.map_shared_rank(s_red, k)[e];
        const unsigned r = rcta + e / (NT * 8u), t = e % (NT * 8u);
        if (t < T && r < rows) {
            if (r < rank) low_out[(size_t)t * rank + r] = __float2bfloat16(hcr_silu(x * inv_hc));
            else inj_out[(size_t)t * hc + (r - rank)] = __float2bfloat16(x);
        }
    }
    cluster.sync();  // peers' shared memory stays live until every read is done
}

extern "C" __global__ void __cluster_dims__(1, HCR_CL, 1) __launch_bounds__(32 * HCR_WM) hc_rowinv_down(
    const __nv_bfloat16* __restrict__ normed,   // [T, hc*H] BF16
    const __nv_bfloat16* __restrict__ down_w,   // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w, // [hc, hc*H] or null (the head)
    __nv_bfloat16* __restrict__ low_out,        // [T, rank]
    __nv_bfloat16* __restrict__ inj_out,        // [T, hc] (unused for the head)
    const unsigned hidden_size, const unsigned hc, const unsigned rank, const unsigned T
) {
    atlas_pdl_enter();
    __shared__ float s_red[HCR_WM * 16u * HCR_MAX_T];
    const unsigned K = hc * hidden_size;
    // Row group blockIdx.z: rows [96 z, 96 z + 96).
    const unsigned g0 = blockIdx.z * HCR_MAX_T;
    const unsigned Tg = min(HCR_MAX_T, T - g0);
    normed += (size_t)g0 * K;
    low_out += (size_t)g0 * rank;
    if (inj_out != nullptr) inj_out += (size_t)g0 * hc;
#define HCR_DOWN(NT_, DN_) hcr_down<NT_, DN_>(normed, down_w, inject_w, low_out, inj_out, K, hc, rank, Tg, s_red)
    switch ((Tg + 7u) / 8u) {
    case 1: HCR_DOWN(1, 2); break;
    case 2: HCR_DOWN(2, 2); break;
    case 3: HCR_DOWN(3, 2); break;
    case 4: HCR_DOWN(4, 2); break;
    case 5: HCR_DOWN(5, 2); break;
    case 6: HCR_DOWN(6, 2); break;
    case 7: HCR_DOWN(7, 2); break;
    case 8: HCR_DOWN(8, 2); break;
    case 9: HCR_DOWN(9, 1); break;
    case 10: HCR_DOWN(10, 1); break;
    case 11: HCR_DOWN(11, 1); break;
    default: HCR_DOWN(12, 1); break;
    }
#undef HCR_DOWN
}
