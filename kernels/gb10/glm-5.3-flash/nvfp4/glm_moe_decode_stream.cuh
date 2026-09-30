// SPDX-License-Identifier: AGPL-3.0-only

// Stream-loaded twins of the M16 verify-decode kernels
// (ATLAS_GLM_MOE_DECODE_STREAM). Included by moe_w4a16_grouped_gemm.cu after
// glm_moe_decode_m16.cuh, whose tiles, MMA stage and epilogue they call.
//
// The M16 kernels are bound by DRAM, and on GB10 a weight byte read through
// an L1-allocating load (cp.async.ca, or a plain ld) costs about 8% more
// DRAM time than one read with a streaming load: the M16 cp.async pipeline
// streams the routed experts' tables at 203-220 GB/s, the same tables read
// in 64 KB chunks with streaming loads at 245-255 GB/s, and the M16 kernel
// with its cp.async replaced by plain loads is no faster than M16, with
// __ldcs (or ld.global.nc.L1::no_allocate) 7-16% faster
// (scripts/moe-decode-bench). So here the block's 256 threads read the next
// K stage with streaming loads into registers, all issued before any store,
// and store them where pqd_issue's cp.async would have left them (rows past
// the expert's, and ZSKIP's dead weight rows, as zeros), then run
// pqd_mma_stage on the stage that landed. Two stage buffers (40 KB, two
// blocks per SM) in place of M16's four.
//
// Shared memory holds the same bytes before each MMA, and the MMAs and the
// epilogue are the M16 functions, so the outputs are those of the M16
// kernels bit for bit (moe_decode_bench gates every variant against
// production).
//
// MI row slabs of 16: MI = 2 takes up to 32 rows per expert (owner batches
// of two to four streams, which M16 leaves to the M64 K128W grid), each slab
// its own pqd_mma_stage over the same weight tile.
//
// Grid and arguments as the M16 kernels; block (PQS_THREADS, 1, 1).
#pragma once

#define PQS_THREADS 256
// 16-byte pieces of a stage per thread: 1024 weight, 128 scale and up to 128
// activation pieces.
#define PQS_PIECES 5

// Thread t's pieces of K stage kb: the shared bytes of pqd_issue.
template<bool GATE_UP, int MI>
__device__ __forceinline__ void pqs_load(
    unsigned int t, unsigned char (*sA)[PQ2_AP], unsigned char (*sAs)[PQ2_KS / GROUP_SIZE],
    PqwB& sB, PqwS& sS, const int* sTok,
    const unsigned char* __restrict__ A_packed, const unsigned char* __restrict__ A_scale,
    const unsigned char* B_expert, const unsigned char* S_expert,
    const unsigned char* U_expert, const unsigned char* US_expert,
    unsigned int M_eff, unsigned int cta_n, unsigned int N, unsigned int K, unsigned int kb,
    const unsigned char* live
) {
    constexpr unsigned int NC = PQW_NT / 16;                      // 16-byte chunks per row
    constexpr unsigned int NB = PQ2_KP * NC;                      // weight pieces
    constexpr unsigned int NS = (PQ2_KS / GROUP_SIZE) * NC;       // block-scale pieces
    constexpr unsigned int NA = PQD_M * MI * (PQ2_KP / 16);       // activation pieces
    static_assert(NB + NS + NA <= PQS_PIECES * PQS_THREADS, "stage pieces exceed the block");
    const uint4 zero = make_uint4(0, 0, 0, 0);
    uint4 v[PQS_PIECES];
    uint4* dst[PQS_PIECES];
    #pragma unroll
    for (int r = 0; r < PQS_PIECES; ++r) {
        const unsigned int p = t + r * PQS_THREADS;
        v[r] = zero;
        dst[r] = nullptr;
        if (p < NB) {
            const unsigned int kp = p / NC, c = p % NC;
            dst[r] = (uint4*)&sB[kp][(c ^ (kp & 7)) << 4];
            if (!live || live[kb / 2 + kp])
                v[r] = __ldcs((const uint4*)&(pqw_chunk_up<GATE_UP>(c) ? U_expert : B_expert)[
                    (unsigned long long)(kb / 2 + kp) * N + pqw_chunk_col<GATE_UP>(cta_n, c)]);
        } else if (p < NB + NS) {
            const unsigned int g = (p - NB) / NC, c = (p - NB) % NC;
            dst[r] = (uint4*)&sS[g][(c ^ g) << 4];
            v[r] = __ldcs((const uint4*)&(pqw_chunk_up<GATE_UP>(c) ? US_expert : S_expert)[
                (unsigned long long)(kb / GROUP_SIZE + g) * N + pqw_chunk_col<GATE_UP>(cta_n, c)]);
        } else if (p < NB + NS + NA) {
            const unsigned int row = (p - NB - NS) >> 2, col = ((p - NB - NS) & 3) << 4;
            dst[r] = (uint4*)&sA[row][col];
            if (row < M_eff)
                v[r] = *(const uint4*)&A_packed[(unsigned long long)(unsigned int)sTok[row] * (K / 2) + kb / 2 + col];
        }
    }
    #pragma unroll
    for (int r = 0; r < PQS_PIECES; ++r)
        if (dst[r]) *dst[r] = v[r];
    if (t < PQD_M * MI)
        *(unsigned long long*)&sAs[t][0] = t < M_eff
            ? *(const unsigned long long*)&A_scale[(unsigned long long)(unsigned int)sTok[t] * (K / GROUP_SIZE) + kb / GROUP_SIZE]
            : 0ull;
}

template<bool GATE_UP, bool ZSKIP, int MI>
__device__ __forceinline__ void pqs_impl(
    PQ2_ARGS,
    const unsigned int* __restrict__ worklist,
    const PqwGateUp up
) {
    atlas_pdl_enter();
    if ((int)blockIdx.y >= (int)worklist[0]) return;
    const unsigned int expert_id = worklist[4 + blockIdx.y * 2];
    // One row tile per expert: a second M64 tile means more than 64 rows.
    if (expert_id >= num_experts || worklist[5 + blockIdx.y * 2] != 0) return;
    const unsigned int cta_m = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - (int)cta_m;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (M_expert <= 0 || B_expert == 0) return;
    const unsigned int t = threadIdx.x;
    if (M_expert > PQD_M * MI) {
        // More rows than the slabs the host selects these kernels for: NaN
        // over them from the downs, as the M16 downs.
        if constexpr (!GATE_UP) {
            const __nv_bfloat16 nan = __float2bfloat16(__int_as_float(0x7fc00000));
            for (int r = 0; r < M_expert; ++r)
                C[(unsigned long long)(cta_m + r) * N + blockIdx.x * PQW_NT + t] = nan;
        }
        return;
    }
    const float scale2 = scale2_vals[expert_id];
    const unsigned char* U_expert = GATE_UP ? (const unsigned char*)up.packed_ptrs[expert_id] : nullptr;
    const unsigned char* US_expert = GATE_UP ? (const unsigned char*)up.scale_ptrs[expert_id] : nullptr;
    const unsigned int cta_n = blockIdx.x * (GATE_UP ? PQW_NT / 2 : PQW_NT);
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) unsigned char sA[2][PQD_M * MI][PQ2_AP];
    __shared__ __align__(16) unsigned char sAs[2][PQD_M * MI][PQ2_KS / GROUP_SIZE];
    __shared__ __align__(16) PqwB sB[2];
    __shared__ __align__(16) PqwS sS[2];
    __shared__ int sTok[PQD_M * MI];

    if (t < PQD_M * MI)
        sTok[t] = (sorted_token_ids && (int)t < M_expert) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    __syncthreads();

    // ZSKIP: the M16 zskip kernel's mask of the weight rows to load.
    __shared__ unsigned char sLive[ZSKIP ? PQD_ZSKIP_KP : 1];
    const bool zskip = ZSKIP && K / 2 <= PQD_ZSKIP_KP;
    if (zskip) {
        for (unsigned int kp = t; kp < K / 2; kp += PQS_THREADS) {
            unsigned int any = 0;
            for (int r = 0; r < M_expert; ++r)
                any |= A_packed[(unsigned long long)(unsigned int)sTok[r] * (K / 2) + kp] & 0x77u;
            sLive[kp] = any != 0;
        }
        __syncthreads();
    }

    auto load = [&](int buf, unsigned int kb) {
        pqs_load<GATE_UP, MI>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
            B_expert, S_expert, U_expert, US_expert, (unsigned int)M_expert, cta_n, N, K, kb,
            zskip ? sLive : nullptr);
    };

    PqdAcc acc[MI];
    #pragma unroll
    for (int mi = 0; mi < MI; mi++)
        #pragma unroll
        for (int i = 0; i < 4; i++) acc[mi][i][0] = acc[mi][i][1] = acc[mi][i][2] = acc[mi][i][3] = 0.0f;

    const unsigned int stages = K / PQ2_KS;
    load(0, 0);
    __syncthreads();
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st & 1;
        // Stage st is in buffer buf; the other one's MMAs ended before the
        // last barrier.
        if (st + 1 < stages) load(buf ^ 1, (st + 1) * PQ2_KS);
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
            pqd_mma_stage<GATE_UP>(acc[mi], *(const PqdA*)&sA[buf][mi * PQD_M], *(const PqdAs*)&sAs[buf][mi * PQD_M],
                sB[buf], sS[buf], warp_id, lane_id);
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < MI; mi++)
        pqd_epilogue<GATE_UP>(acc[mi], warp_id, lane_id, expert_id, scale2, cta_m + mi * PQD_M,
            M_expert - mi * PQD_M, cta_n, N, C, up);
}

// Down, zero-skipping down and fused gate/up over one row slab (up to 16 rows
// per expert) and over two (up to 32): arguments as the M16 kernels.
#define PQS_DOWN_ARGS PQ2_ARGS, const unsigned int* __restrict__ worklist
#define PQS_GATE_UP_ARGS \
    PQ2_ARGS, const unsigned int* __restrict__ worklist, \
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs, \
    const float* __restrict__ up_scale2_vals, unsigned char* __restrict__ out_packed, unsigned char* __restrict__ out_scale
#define PQS_TABLES A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, \
    expert_offsets, sorted_token_ids, num_experts, N, K, worklist
#define PQS_UP PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale}

extern "C" __global__ void __launch_bounds__(PQS_THREADS) glm_moe_decode_m16s_k128w(PQS_DOWN_ARGS) {
    pqs_impl<false, false, 1>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS) glm_moe_decode_m16s_k128w_zskip(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 1>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS) glm_moe_decode_m16s_gate_up_silu_k128w(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 1>(PQS_TABLES, PQS_UP);
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS) glm_moe_decode_m32s_k128w(PQS_DOWN_ARGS) {
    pqs_impl<false, false, 2>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS) glm_moe_decode_m32s_k128w_zskip(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 2>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS) glm_moe_decode_m32s_gate_up_silu_k128w(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 2>(PQS_TABLES, PQS_UP);
}

#undef PQS_DOWN_ARGS
#undef PQS_GATE_UP_ARGS
#undef PQS_TABLES
#undef PQS_UP
