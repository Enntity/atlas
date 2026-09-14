// SPDX-License-Identifier: AGPL-3.0-only
// Generated from frozen production source; never substitute for its baseline.
// SPDX-License-Identifier: AGPL-3.0-only
// Fused epilogue: decode pre-packed BF16 gate pairs, BF16-round up*up_scale,
// apply SiLU(gate)*up, NVFP4-quantize per 16-element group with e4m3 scales.
__device__ __forceinline__ void atlas_fused_epilogue(
    const unsigned gates[16][2], const float acc[16][4], float up_scale,
    unsigned char* packed, unsigned char* scales,
    unsigned cta_m, unsigned cta_n, unsigned cta_m_local,
    int M_expert, unsigned N)
{
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int gid  = lane >> 2;          // 0..7  (row within warp's 16)
    const int tid  = lane & 3;           // 0..3  (quad sub-lane)
    const unsigned row0 = cta_m + (unsigned)(warp * 16 + gid);
    const unsigned row1 = row0 + 8u;
    const bool ok0 = ((int)(warp * 16 + gid) + (int)cta_m_local) < M_expert;
    const bool ok1 = ((int)(warp * 16 + gid + 8) + (int)cta_m_local) < M_expert;
    const float SWIGLU_LIMIT = 10.0f;
    const unsigned lane_mask = 0xffffffffu;

#pragma unroll
    for (int nt = 0; nt < 16; nt += 2) {
        // ---- decode + compute 4 values per row (BF16 round of up before silu) ----
        float vals[4];
        {
            const unsigned g01 = gates[nt][0];
            const unsigned g23 = gates[nt][1];
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned gi = (i == 0) ? (g01 & 0xFFFFu)
                                  : (i == 1) ? (g01 >> 16)
                                  : (i == 2) ? (g23 & 0xFFFFu)
                                             : (g23 >> 16);
                float g = __bfloat162float(__ushort_as_bfloat16((unsigned short)gi));
                float u = __bfloat162float(__float2bfloat16(acc[nt][i] * up_scale));
                g = fminf(g, SWIGLU_LIMIT);
                u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
                const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
                const __nv_bfloat16 r16 = __float2bfloat16(g * sigmoid_g * u);
                v[i] = __bfloat162float(r16);
            }
            vals[0] = v[0]; vals[1] = v[1]; vals[2] = v[2]; vals[3] = v[3];
        }
        // also decode nt+1
        float vals2[4];
        {
            const unsigned g01 = gates[nt + 1][0];
            const unsigned g23 = gates[nt + 1][1];
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned gi = (i == 0) ? (g01 & 0xFFFFu)
                                  : (i == 1) ? (g01 >> 16)
                                  : (i == 2) ? (g23 & 0xFFFFu)
                                             : (g23 >> 16);
                float g = __bfloat162float(__ushort_as_bfloat16((unsigned short)gi));
                float u = __bfloat162float(__float2bfloat16(acc[nt + 1][i] * up_scale));
                g = fminf(g, SWIGLU_LIMIT);
                u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
                const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
                const __nv_bfloat16 r16 = __float2bfloat16(g * sigmoid_g * u);
                v[i] = __bfloat162float(r16);
            }
            vals2[0] = v[0]; vals2[1] = v[1]; vals2[2] = v[2]; vals2[3] = v[3];
        }

        // x[0,1] from nt, x[2,3] from nt+1 (row0 for i<2, row1 for i>=2)
        // For each row: own 4 values are the two pairs.
        // row0: nhalf = {vals[0], vals[1]} (nt pair), plus next group's pair {vals2[0], vals2[1]}
        // row1: {vals[2], vals[3]}, {vals2[2], vals2[3]}
        float xr0[4] = { vals[0], vals[1], vals2[0], vals2[1] };
        float xr1[4] = { vals[2], vals[3], vals2[2], vals2[3] };

        // ---- per-row maxabs, then butterfly across quads (lanes 0..3 same row) ----
        float m0 = fmaxf(fmaxf(fabsf(xr0[0]), fabsf(xr0[1])), fmaxf(fabsf(xr0[2]), fabsf(xr0[3])));
        float m1 = fmaxf(fmaxf(fabsf(xr1[0]), fabsf(xr1[1])), fmaxf(fabsf(xr1[2]), fabsf(xr1[3])));
        m0 = fmaxf(m0, __shfl_xor_sync(lane_mask, m0, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(lane_mask, m1, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(lane_mask, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(lane_mask, m1, 2));

        // ---- decoded scale / inverse ----
        const unsigned char fp8_0 = silu_nvfp4_float_to_e4m3(m0 / 6.0f);
        const unsigned char fp8_1 = silu_nvfp4_float_to_e4m3(m1 / 6.0f);
        const unsigned sbase = row0 * (N / 16u) + (cta_n + (unsigned)nt * 8u) / 16u;
        const unsigned sbase1 = row1 * (N / 16u) + (cta_n + (unsigned)nt * 8u) / 16u;
        if (tid == 0) {
            if (ok0) scales[sbase] = fp8_0;
            if (ok1) scales[sbase1] = fp8_1;
        }
        const unsigned exp0 = (fp8_0 >> 3) & 0xF, man0 = fp8_0 & 0x7;
        const unsigned exp1 = (fp8_1 >> 3) & 0xF, man1 = fp8_1 & 0x7;
        float dec0, dec1;
        if (exp0 == 0) dec0 = (float)man0 * 0.001953125f;
        else if (exp0 == 15 && man0 == 7) dec0 = 0.0f;
        else dec0 = __uint_as_float((exp0 + 120u) << 23 | (man0 << 20));
        if (exp1 == 0) dec1 = (float)man1 * 0.001953125f;
        else if (exp1 == 15 && man1 == 7) dec1 = 0.0f;
        else dec1 = __uint_as_float((exp1 + 120u) << 23 | (man1 << 20));
        const float inv0 = dec0 > 0.0f ? 1.0f / dec0 : 0.0f;
        const float inv1 = dec1 > 0.0f ? 1.0f / dec1 : 0.0f;

        // ---- pack 4 values -> 2 bytes per row ----
        const unsigned pbase0 = row0 * (N / 2u) + (cta_n + (unsigned)nt * 8u) / 2u + (unsigned)tid;
        const unsigned pbase1 = row1 * (N / 2u) + (cta_n + (unsigned)nt * 8u) / 2u + (unsigned)tid;
        const unsigned int q00 = silu_nvfp4_quantize_e2m1(xr0[0] * inv0);
        const unsigned int q01 = silu_nvfp4_quantize_e2m1(xr0[1] * inv0);
        const unsigned int q02 = silu_nvfp4_quantize_e2m1(xr0[2] * inv0);
        const unsigned int q03 = silu_nvfp4_quantize_e2m1(xr0[3] * inv0);
        const unsigned int q10 = silu_nvfp4_quantize_e2m1(xr1[0] * inv1);
        const unsigned int q11 = silu_nvfp4_quantize_e2m1(xr1[1] * inv1);
        const unsigned int q12 = silu_nvfp4_quantize_e2m1(xr1[2] * inv1);
        const unsigned int q13 = silu_nvfp4_quantize_e2m1(xr1[3] * inv1);
        const unsigned char b0 = (unsigned char)((q01 << 4) | q00);
        const unsigned char b1 = (unsigned char)((q03 << 4) | q02);
        const unsigned char c0 = (unsigned char)((q11 << 4) | q10);
        const unsigned char c1 = (unsigned char)((q13 << 4) | q12);
        if (ok0) { packed[pbase0] = b0; packed[pbase0 + 4u] = b1; }
        if (ok1) { packed[pbase1] = c0; packed[pbase1 + 4u] = c1; }
    }
}

template<bool PQ_VEC_SCALES>
__device__ __forceinline__ void atlas_dev_moe_fused_impl(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned long long* __restrict__ U_packed_ptrs,
    const unsigned long long* __restrict__ U_scale_ptrs,
    const float* __restrict__ U_scale2_vals,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    unsigned int work_expert_id,
    unsigned int work_m_tile,
    unsigned int work_n_tile
) {
    const unsigned int expert_id = work_expert_id;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = work_m_tile * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = work_n_tile * N_TILE_LG;
    if (!B_packed_ptrs[expert_id] || !U_packed_ptrs[expert_id]) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is already compact FP4. Keep each row 16-byte aligned because the
    // loader issues 16-byte cp.async operations for both halves of the row.
    // (A 36-byte stride faults as CUDA_ERROR_MISALIGNED_ADDRESS on row 1.)
    __shared__ unsigned char smem_Ap_pq[2][M_TILE][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char smem_As_pq[2][M_TILE][K_STEP_T64 / GROUP_SIZE];
    // B checkpoint layout is [K/2,N]; load it coalesced, then transpose the
    // 32-byte K run required by the native block-scaled MMA.
    __shared__ unsigned char smem_BpT_pq[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bp_pq[N_TILE_LG][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char smem_Bs_pq[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG];
    __shared__ int smem_tok_pq[M_TILE];

    if (threadIdx.x < M_TILE) {
        const int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_pq[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_pq[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    // Exactly rounded gate results: 64 BF16 values in32 register words/lane.
    unsigned gate_values[16][2];
    #pragma unroll 1
    for (int projection = 0; projection < 2; ++projection) {
    const unsigned char* B_expert = (const unsigned char*)(projection
        ? U_packed_ptrs[expert_id] : B_packed_ptrs[expert_id]);
    const unsigned char* S_expert = (const unsigned char*)(projection
        ? U_scale_ptrs[expert_id] : B_scale_ptrs[expert_id]);
    const float scale2 = projection ? U_scale2_vals[expert_id] : scale2_vals[expert_id];
    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int num_groups = K / GROUP_SIZE;

    #define PQ4_ISSUE_LOADS(buf, kb) do { \
        { \
            /* 128 threads x 16 B = 64 rows x 32 packed K bytes. */ \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int col = (threadIdx.x & 1) << 4; \
            bool valid = (cta_m_local + row) < M_eff && ((kb) + col * 2 + 31 < K); \
            unsigned int a_row = (unsigned int)smem_tok_pq[row]; \
            moe_cp_async_pred_16(&smem_Ap_pq[(buf)][row][col], \
                &A_packed[(unsigned long long)a_row * (K / 2) + (kb) / 2 + col], \
                valid); \
        } \
        if constexpr (PQ_VEC_SCALES) { \
            /* One aligned 4-byte transaction loads all four A scales/row. */ \
            if (threadIdx.x < M_TILE) { \
                unsigned int row = threadIdx.x; \
                bool valid = (cta_m_local + row) < M_eff; \
                unsigned int a_row = (unsigned int)smem_tok_pq[row]; \
                moe_cp_async_pred_4(&smem_As_pq[(buf)][row][0], \
                    &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) \
                        + (kb) / GROUP_SIZE], valid); \
            } \
        } else { \
            /* Conservative scalar scale loader. */ \
            _Pragma("unroll") \
            for (int job = 0; job < 2; job++) { \
                unsigned int jid = threadIdx.x + job * 128; \
                unsigned int row = jid >> 2; \
                unsigned int grp = jid & 3; \
                bool valid = (cta_m_local + row) < M_eff; \
                unsigned int a_row = (unsigned int)smem_tok_pq[row]; \
                smem_As_pq[(buf)][row][grp] = valid \
                    ? A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) \
                        + (kb) / GROUP_SIZE + grp] \
                    : 0; \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                moe_cp_async_pred_16(&smem_BpT_pq[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
            } \
        } \
        if constexpr (PQ_VEC_SCALES) { \
            /* 32 aligned 16-byte transactions replace 512 scalar copies. */ \
            if (threadIdx.x < 32) { \
                unsigned int g = threadIdx.x >> 3; \
                unsigned int ns = (threadIdx.x & 7) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + g; \
                unsigned int gns = cta_n + ns; \
                moe_cp_async_pred_16(&smem_Bs_pq[(buf)][g][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N) && (sg < num_groups)); \
            } \
        } else { \
            unsigned int g = threadIdx.x >> 5; \
            unsigned int nn = threadIdx.x & 31; \
            unsigned int sg = (kb) / GROUP_SIZE + g; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int n_cur = rnd * 32 + nn; \
                unsigned int gns = cta_n + n_cur; \
                bool valid = (gns < N) && (sg < num_groups); \
                smem_Bs_pq[(buf)][g][n_cur] = valid \
                    ? S_expert[(unsigned long long)sg * N + gns] : 0; \
            } \
        } \
    } while(0)

    #define PQ4_TRANSPOSE(buf) do { \
        unsigned int my_n = threadIdx.x; \
        _Pragma("unroll") \
        for (int q = 0; q < (K_STEP_T64 / 2) / 4; q++) { \
            unsigned int w = (unsigned int)smem_BpT_pq[(buf)][q * 4 + 0][my_n] \
                | ((unsigned int)smem_BpT_pq[(buf)][q * 4 + 1][my_n] << 8) \
                | ((unsigned int)smem_BpT_pq[(buf)][q * 4 + 2][my_n] << 16) \
                | ((unsigned int)smem_BpT_pq[(buf)][q * 4 + 3][my_n] << 24); \
            *(unsigned int*)&smem_Bp_pq[my_n][q * 4] = w; \
        } \
    } while(0)

    #define PQ4_FRAG(P, ROW, KK) (*(const unsigned int*)&(P)[(ROW)][(KK) / 2])
    #define PQ4_COMPUTE_MMA(a_buf, b_buf) do { \
        unsigned int ra = warp_m_offset + group_id; \
        unsigned int a0 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra,     tid * 8); \
        unsigned int a1 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra + 8, tid * 8); \
        unsigned int a2 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra,     32 + tid * 8); \
        unsigned int a3 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra + 8, 32 + tid * 8); \
        unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2); \
        unsigned int sfa = (unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][0] \
            | ((unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][1] << 8) \
            | ((unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][2] << 16) \
            | ((unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][3] << 24); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = PQ4_FRAG(smem_Bp_pq, nc, tid * 8); \
            unsigned int b1 = PQ4_FRAG(smem_Bp_pq, nc, 32 + tid * 8); \
            unsigned int sfn = nt * 8 + (lane_id >> 2); \
            unsigned int sfb = (unsigned int)smem_Bs_pq[(b_buf)][0][sfn] \
                | ((unsigned int)smem_Bs_pq[(b_buf)][1][sfn] << 8) \
                | ((unsigned int)smem_Bs_pq[(b_buf)][2][sfn] << 16) \
                | ((unsigned int)smem_Bs_pq[(b_buf)][3][sfn] << 24); \
            unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]), \
                 "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb),"h"(bidB),"h"(tidB_)); \
        } \
    } while(0)

    PQ4_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    PQ4_TRANSPOSE(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        PQ4_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        PQ4_COMPUTE_MMA(cur, cur);
        moe_cp_async_wait_all();
        __syncthreads();
        PQ4_TRANSPOSE(nxt);
        __syncthreads();
        cur = nxt;
    }
    PQ4_COMPUTE_MMA(cur, cur);

    #undef PQ4_ISSUE_LOADS
    #undef PQ4_TRANSPOSE
    #undef PQ4_FRAG
    #undef PQ4_COMPUTE_MMA

    if (projection == 0) {
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            gate_values[nt][0] = (unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][0] * scale2))
                | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][1] * scale2)) << 16);
            gate_values[nt][1] = (unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][2] * scale2))
                | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][3] * scale2)) << 16);
        }
    } else {
        atlas_fused_epilogue(gate_values, acc, scale2, packed_out, scale_out,
            cta_m, cta_n, cta_m_local, M_expert, N);
    }
    // Do not overwrite shared MMA operands until every warp finished this pass.
    __syncthreads();
    }
}

extern "C" __global__ void atlas_dev_moe_fused_gate_up(
    const unsigned char* A, const unsigned char* As,
    const unsigned long long* G, const unsigned long long* Gs, const float* G2,
    const unsigned long long* U, const unsigned long long* Us, const float* U2,
    unsigned char* packed, unsigned char* scales, const int* offsets,
    const int* ids, unsigned experts, unsigned N, unsigned K) {
    atlas_dev_moe_fused_impl<true>(A, As, G, Gs, G2, U, Us, U2,
        packed, scales, offsets, ids, experts, N, K, blockIdx.z, blockIdx.y, blockIdx.x);
}
