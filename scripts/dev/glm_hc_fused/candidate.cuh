// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cuda_bf16.h>
#include <mma.h>

// Standalone screen only. Preserve FP32 highway arithmetic; TF32/RMS reduction
// order differs. The initial DeepSeek producer draft was rejected before build.
extern "C" __global__ void glm_hc_post_mix_rms_tf32(
    const __nv_bfloat16* block, const float* residual, const float* old_post,
    const float* old_comb, const float* next_fn, float* highway_out,
    float* raw_mix_out, float* inv_rms_out, unsigned M, float norm_eps) {
    using namespace nvcuda::wmma;
    constexpr unsigned H = 4096, D = 16384, TM = 32, TN = 32;
    const unsigned tid = threadIdx.x, warp = tid / 32;
    const unsigned lr = tid / 8, d0 = (tid % 8) * 4;
    const unsigned base = blockIdx.x * TM, row = base + lr;
    const bool valid = row < M, consumer = warp < 4;
    // Exact same pointer is supported for residual and highway_out. A thread
    // reads all four streams at a coordinate before writing any of them.
    __shared__ __align__(32) float a[TM][132];
    __shared__ __align__(32) float b[TN][132];
    __shared__ __align__(32) float raw[TM][TN];
    __shared__ float post[TM][4], comb[TM][16];
    if (tid < TM) {
        const unsigned token = base + tid;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            post[tid][j] = token < M ? old_post[token * 4 + j] : 0.f;
        #pragma unroll
        for (unsigned j = 0; j < 16; ++j)
            comb[tid][j] = token < M ? old_comb[token * 16 + j] : 0.f;
    }
    fragment<accumulator, 16, 16, 8, float> accum;
    if (consumer) fill_fragment(accum, 0.f);
    float sumsq = 0.f;
    __syncthreads();
    for (unsigned db = 0; db < H; db += 32) {
        #pragma unroll
        for (unsigned q = 0; q < 4; ++q) {
            const unsigned d = d0 + q;
            float rv[4];
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                rv[i] = valid ? residual[size_t(row) * D + i * H + db + d] : 0.f;
            const float x = valid ? __bfloat162float(block[size_t(row) * H + db + d]) : 0.f;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                float v = post[lr][j] * x;
                #pragma unroll
                for (unsigned i = 0; i < 4; ++i) v += comb[lr][i * 4 + j] * rv[i];
                if (valid) highway_out[size_t(row) * D + j * H + db + d] = v;
                sumsq += v * v;
                a[lr][j * 32 + d] = __float_to_tf32(v);
            }
        }
        for (unsigned index = tid; index < TN * 128; index += 256) {
            const unsigned n = index / 128, k = index % 128;
            const unsigned j = k / 32, d = k % 32;
            b[n][k] = n < 24 ? __float_to_tf32(next_fn[size_t(n) * D + j * H + db + d]) : 0.f;
        }
        __syncthreads();
        if (consumer) {
            fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> fa;
            fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> fb;
            #pragma unroll
            for (unsigned k = 0; k < 128; k += 8) {
                load_matrix_sync(fa, &a[(warp / 2) * 16][k], 132);
                load_matrix_sync(fb, &b[(warp % 2) * 16][k], 132);
                mma_sync(accum, fa, fb, accum);
            }
        }
        // Every consumer must finish before producer warps overwrite a/b.
        __syncthreads();
    }
    #pragma unroll
    for (unsigned offset = 1; offset <= 4; offset *= 2)
        sumsq += __shfl_xor_sync(0xffffffffu, sumsq, offset);
    if ((tid % 8) == 0 && valid)
        inv_rms_out[row] = rsqrtf(sumsq / float(D) + norm_eps);
    if (consumer)
        store_matrix_sync(&raw[(warp / 2) * 16][(warp % 2) * 16], accum, TN, mem_row_major);
    __syncthreads();
    // Eight threads per row write disjoint raw-mix coordinates.
    if (valid)
        for (unsigned n = tid % 8; n < 24; n += 8)
            raw_mix_out[size_t(row) * 24 + n] = raw[lr][n];
}
