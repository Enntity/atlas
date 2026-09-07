// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstddef>
#ifdef __CUDACC__
#define GLM_FP8_HD __host__ __device__
#else
#define GLM_FP8_HD
#endif
namespace glm_shared_fp8 {
GLM_FP8_HD constexpr size_t transpose_index(unsigned row,unsigned col,unsigned rows,unsigned cols) {
    (void)cols;
    return size_t(col)*rows+row;
}
GLM_FP8_HD constexpr unsigned output_col(unsigned warp,unsigned nt,unsigned offset) {
    return warp*32+nt*8+offset;
}
GLM_FP8_HD constexpr bool geometry(unsigned m,unsigned n,unsigned k) {
    return m<=16&&n>0&&n<=4096&&n%128==0&&k>0&&k<=4096&&k%32==0;
}
}
#undef GLM_FP8_HD

#ifdef __CUDACC__
// Standalone specialization of included production fp8_gemm_t: only A extent,
// uniform shape guard, and four warp-N partitions differ from its K32 pipeline.
extern "C" __global__ void glm_shared_fp8_m16(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    if(blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.y || blockIdx.z
        || !glm_shared_fp8::geometry(M,N,K) || M==0) return;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = 0;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][16][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16) + B (FP8, pre-dequanted) via cp.async
    #define FP8_LOADS(buf, kb) do { \
        if(threadIdx.x < 64) { \
            unsigned int row = threadIdx.x >> 2; \
            unsigned int col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + col; \
            unsigned int gr = row < M ? row : 0; \
            cp_async_pred_16(&smem_A[(buf)][row][col], \
                &A[(unsigned long long)gr * K + gc], \
                (row < M) && (gc + 7 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA — identical to w4a16_gemm_t COMPUTE_MMA
    #define FP8_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 4; nt++) { \
            unsigned int nc = glm_shared_fp8::output_col(warp_id, nt, group_id); \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    // Prolog: load first tile, wait, no dequant needed
    FP8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE(cur, cur);

    #undef FP8_LOADS
    #undef FP8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        unsigned int c0 = cta_n + glm_shared_fp8::output_col(warp_id, nt, tid*2);
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

#endif
