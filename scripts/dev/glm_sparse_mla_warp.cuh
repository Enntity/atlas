// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>

// Rejected experiment: approximately 20% slower than the production head8
// kernel at large prefill shapes on GB10. Kept with its standalone benchmark
// to avoid repeating this candidate; never registered in the serving engine.
// Eight-head sparse MLA with one warp per head. Query and output elements
// remain in registers while all heads share each eight-token K/V tile.
// The per-score lane mapping, shuffle tree, and online-softmax recurrence
// match glm_sparse_mla_prefill_bf16_head8.
extern "C" __global__ void glm_sparse_mla_prefill_bf16_head8_warp(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int index_width,
    unsigned int cache_block_size,
    float inv_sqrt_d) {
    constexpr unsigned int HEADS = 8;
    constexpr unsigned int ITEMS = 8;
    constexpr unsigned int DIM = 512;
    constexpr unsigned int ELEMENTS = DIM / 32;
    const unsigned int row = blockIdx.y;
    const unsigned int head_base = blockIdx.x * HEADS;
    const unsigned int tid = threadIdx.x;
    // These conditions are uniform across the entire CTA. Partial head groups
    // must retain their inactive warps until the last block-wide barrier.
    if (row >= rows || head_base >= num_heads || head_dim != DIM || blockDim.x != 256)
        return;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int head = head_base + warp;
    const bool active_head = head < num_heads;
    const int* indices = token_indices + (unsigned long long)row * index_width;
    const unsigned long long cache_block_stride =
        (unsigned long long)cache_block_size * DIM;

    // Convert each cached element once while staging it. Float shared rows
    // provide one full-width bank lane per thread and reuse that conversion
    // across all eight query heads.
    __shared__ float keys[ITEMS][DIM];
    __shared__ float values[ITEMS][DIM];
    float q[ELEMENTS];
    float out[ELEMENTS];
#pragma unroll
    for (unsigned int d = 0; d < ELEMENTS; ++d) {
        q[d] = active_head
            ? __bfloat162float(query[((unsigned long long)row * num_heads + head) * DIM
                + lane + d * 32])
            : 0.0f;
        out[d] = 0.0f;
    }
    // Only lane zero updates the softmax state. Its results are broadcast to
    // the complete warp before each output update.
    float running_max = -INFINITY;
    float running_denom = 0.0f;
    for (unsigned int tile = 0; tile < index_width; tile += ITEMS) {
#pragma unroll
        for (unsigned int i = 0; i < ITEMS; ++i) {
            const unsigned int selected = tile + i;
            const int token = selected < index_width ? indices[selected] : -1;
            const __nv_bfloat16* k = nullptr;
            const __nv_bfloat16* v = nullptr;
            if (token >= 0) {
                const unsigned int logical_block = (unsigned int)token / cache_block_size;
                const unsigned int block_offset = (unsigned int)token % cache_block_size;
                const unsigned int physical_block = block_table[logical_block];
                const unsigned long long offset =
                    (unsigned long long)physical_block * cache_block_stride
                    + (unsigned long long)block_offset * DIM;
                k = k_cache + offset;
                v = v_cache + offset;
            }
#pragma unroll
            for (unsigned int d = 0; d < DIM; d += 256) {
                keys[i][tid + d] = token >= 0 ? __bfloat162float(k[tid + d]) : 0.0f;
                values[i][tid + d] = token >= 0 ? __bfloat162float(v[tid + d]) : 0.0f;
            }
        }
        __syncthreads();

        // Keep eight independent dot-product chains live together. Serially
        // completing one token's dot leaves the FP32 pipeline dependent on a
        // single accumulator despite the eight-way tile reuse.
        float partial[ITEMS];
#pragma unroll
        for (unsigned int i = 0; i < ITEMS; ++i)
            partial[i] = 0.0f;
#pragma unroll
        for (unsigned int d = 0; d < ELEMENTS; ++d) {
#pragma unroll
            for (unsigned int i = 0; i < ITEMS; ++i)
                partial[i] += q[d] * keys[i][lane + d * 32];
        }
        float scores[ITEMS];
#pragma unroll
        for (unsigned int i = 0; i < ITEMS; ++i) {
            for (unsigned int offset = 16; offset; offset >>= 1)
                partial[i] += __shfl_down_sync(0xffffffff, partial[i], offset);
            if (lane == 0) {
                const unsigned int selected = tile + i;
                const int token = selected < index_width ? indices[selected] : -1;
                scores[i] = token >= 0 && active_head ? partial[i] * inv_sqrt_d : -INFINITY;
            }
        }

        float alpha = 0.0f;
        float betas[ITEMS];
        if (lane == 0) {
            float tile_max = -INFINITY;
#pragma unroll
            for (unsigned int i = 0; i < ITEMS; ++i)
                tile_max = fmaxf(tile_max, scores[i]);
            const float next_max = fmaxf(running_max, tile_max);
            alpha = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
            float tile_sum = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < ITEMS; ++i) {
                const float beta = scores[i] == -INFINITY
                    ? 0.0f : expf(scores[i] - next_max);
                betas[i] = beta;
                tile_sum += beta;
            }
            running_denom = running_denom * alpha + tile_sum;
            running_max = next_max;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
#pragma unroll
        for (unsigned int d = 0; d < ELEMENTS; ++d)
            out[d] *= alpha;
#pragma unroll
        for (unsigned int i = 0; i < ITEMS; ++i) {
            const float beta = __shfl_sync(0xffffffff, lane == 0 ? betas[i] : 0.0f, 0);
#pragma unroll
            for (unsigned int d = 0; d < ELEMENTS; ++d)
                out[d] += beta * values[i][lane + d * 32];
        }
        __syncthreads();
    }

    const float denom = __shfl_sync(0xffffffff, running_denom, 0);
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    if (active_head) {
        __nv_bfloat16* dst = output
            + ((unsigned long long)row * num_heads + head) * DIM;
#pragma unroll
        for (unsigned int d = 0; d < ELEMENTS; ++d)
            dst[lane + d * 32] = __float2bfloat16(out[d] * inv_denom);
    }
}
