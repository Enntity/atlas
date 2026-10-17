// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cuda_bf16.h>

// Shared paged scoring and reduction order; arithmetic policy is explicit.
__device__ __forceinline__ float glm_index_score_weight(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
__device__ __forceinline__ float glm_index_score_weight(float value) { return value; }

template<typename Weight, bool normalize>
__device__ __forceinline__ void glm_index_logits_impl(
    const __nv_bfloat16* __restrict__ query,
    const Weight* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    const unsigned int row = blockIdx.y;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int pool_id = blockIdx.x * 8 + warp;
    if (row >= rows || pool_id >= logits_stride) return;
    const unsigned int pool_count = seq_len / pool_size;
    if (pool_id >= pool_count) {
        if (lane == 0) logits[(unsigned long long)row * logits_stride + pool_id] = -INFINITY;
        return;
    }

    const unsigned int raw_pos = pool_id * pool_size;
    const unsigned int logical_block = raw_pos / cache_block_size;
    const unsigned int raw_offset = raw_pos % cache_block_size;
    const unsigned int physical_block = block_table[logical_block];
    const unsigned int pool_offset = raw_offset / pool_size;
    const __nv_bfloat16* key = (const __nv_bfloat16*)((const char*)index_cache
        + (unsigned long long)physical_block * index_block_stride_bytes)
        + (unsigned long long)pool_offset * head_dim;
    const __nv_bfloat16* q = query
        + (unsigned long long)row * index_heads * head_dim;
    const Weight* w = weights + (unsigned long long)row * index_heads;
    float score = 0.0f;
    for (unsigned int head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            dot += __bfloat162float(q[(unsigned long long)head * head_dim + d])
                 * __bfloat162float(key[d]);
        }
        for (unsigned int offset = 16; offset; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0) score += glm_index_score_weight(w[head]) * fmaxf(dot, 0.0f);
    }
    if (lane == 0) {
        if constexpr (normalize)
            logits[(unsigned long long)row * logits_stride + pool_id] =
                score * rsqrtf((float)(head_dim * index_heads));
        else
            logits[(unsigned long long)row * logits_stride + pool_id] = score;
    }
}

