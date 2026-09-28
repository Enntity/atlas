// SPDX-License-Identifier: AGPL-3.0-only

// Experimental BF16 tensor-core semantic scorer. This is deliberately separate
// from glm_indexer.cu: FP32 tensor-core dot products have a different reduction
// order from the scalar scorer and must pass top-K and model-quality checks
// before runtime dispatch is enabled.
//
// Both entry points have the glm_index_logits_bf16_row8 argument ABI. Launch:
//   row8:        grid=(ceil(logits_stride/16), ceil(rows/8)), block=(256,1,1)
//   row8_pool32: grid=(ceil(logits_stride/32), ceil(rows/8)), block=(256,1,1)
// No dynamic shared memory, scratch allocation, or special shared-memory opt-in
// is needed. Static shared memory is 12,800 / 25,600 bytes respectively.
// Requirements: SM80+, 32-byte aligned query, heads=32, head_dim=128,
// pool_size=4, cache_block_size a positive multiple of 4. Cache page/table and
// output sizing follow the scalar scorer. Unsupported geometry does no work;
// the caller must validate these requirements before selecting this kernel.

#include <cuda_bf16.h>
#include <math.h>
#include <mma.h>

namespace {

template <unsigned int KeyTiles>
__device__ __forceinline__ void glm_index_logits_wmma_impl(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    namespace wmma = nvcuda::wmma;
    constexpr unsigned int key_count = KeyTiles * 16;
    // A 16-element skew reduces shared-memory bank conflicts while retaining
    // WMMA's 32-byte matrix-base and 16-byte leading-dimension alignment.
    constexpr unsigned int key_stride = 128 + 16;
    __shared__ __align__(32) __nv_bfloat16 pooled_keys[key_count * key_stride];
    __shared__ __align__(32) float head_dots[8 * KeyTiles * 16 * 16];

    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        index_heads != 32 || head_dim != 128 || pool_size != 4 ||
        cache_block_size == 0 || cache_block_size % 4 != 0) return;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int pool_base = blockIdx.x * key_count;
    const unsigned int row_base = blockIdx.y * 8;
    if (row_base >= rows || pool_base >= logits_stride) return;

    // Use the latest row in this CTA, so wholly future tiles never access
    // pages beyond its causal extent. Partial row tiles still reach the one
    // block-wide barrier before their inactive warps return.
    const unsigned int tile_rows = rows - row_base < 8 ? rows - row_base : 8;
    const unsigned int max_pool_count = (seq_len_start + row_base + tile_rows) / 4;
    if (pool_base >= max_pool_count) {
        for (unsigned int item = threadIdx.x; item < tile_rows * key_count;
             item += blockDim.x) {
            const unsigned int pool_id = pool_base + item % key_count;
            if (pool_id < logits_stride) {
                logits[(unsigned long long)(row_base + item / key_count) *
                           logits_stride + pool_id] = -INFINITY;
            }
        }
        return;
    }
    for (unsigned int item = threadIdx.x; item < key_count * 128;
         item += blockDim.x) {
        const unsigned int pool = item / 128;
        const unsigned int d = item % 128;
        const unsigned int pool_id = pool_base + pool;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (pool_id < logits_stride && pool_id < max_pool_count) {
            const unsigned int raw_pos = pool_id * 4;
            const unsigned int physical_block = block_table[raw_pos / cache_block_size];
            const unsigned int cache_pool = (raw_pos % cache_block_size) / 4;
            const __nv_bfloat16* key =
                (const __nv_bfloat16*)((const char*)index_cache +
                    (unsigned long long)physical_block * index_block_stride_bytes) +
                (unsigned long long)cache_pool * 128;
            value = key[d];
        }
        pooled_keys[pool * key_stride + d] = value;
    }
    __syncthreads();

    const unsigned int row = row_base + warp;
    if (row >= rows) return;
    const unsigned int pool_count = (seq_len_start + row + 1) / 4;
    const __nv_bfloat16* q = query + (unsigned long long)row * 32 * 128;
    // Each lane holds one head weight; warp broadcasts preserve head order.
    const float head_weight = __bfloat162float(weights[(unsigned long long)row * 32 + lane]);
    float* warp_dots = head_dots + warp * KeyTiles * 256;
    const unsigned int pool = lane % key_count;
    const unsigned int dot_column = pool % 16;
    const unsigned int dot_tile_offset = (pool / 16) * 256;
    float score = 0.0f;

#pragma unroll
    for (unsigned int head_base = 0; head_base < 32; head_base += 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[KeyTiles];
#pragma unroll
        for (unsigned int tile = 0; tile < KeyTiles; ++tile) {
            wmma::fill_fragment(accum[tile], 0.0f);
        }
#pragma unroll
        for (unsigned int d = 0; d < 128; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16,
                           __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, q + head_base * 128 + d, 128);
#pragma unroll
            for (unsigned int tile = 0; tile < KeyTiles; ++tile) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, pooled_keys + tile * 16 * key_stride + d,
                                       key_stride);
                wmma::mma_sync(accum[tile], a, b, accum[tile]);
            }
        }
#pragma unroll
        for (unsigned int tile = 0; tile < KeyTiles; ++tile) {
            wmma::store_matrix_sync(warp_dots + tile * 256, accum[tile], 16,
                                    wmma::mem_row_major);
        }
        __syncwarp();
        // Materialize only this warp's 16-head tile in shared memory. ReLU is
        // applied independently to each head before its weight is multiplied.
#pragma unroll
        for (unsigned int head = 0; head < 16; ++head) {
            const float weight = __shfl_sync(0xffffffff, head_weight, head_base + head);
            const float dot = warp_dots[dot_tile_offset + head * 16 + dot_column];
            score += weight * fmaxf(dot, 0.0f);
        }
        // A subsequent store must not overwrite dots still read by a peer.
        __syncwarp();
    }
    if (lane < key_count) {
        const unsigned int pool_id = pool_base + pool;
        if (pool_id < logits_stride) {
            // sqrt(128 * 32) = 64 exactly.
            logits[(unsigned long long)row * logits_stride + pool_id] =
                pool_id < pool_count ? score * (1.0f / 64.0f) : -INFINITY;
        }
    }
}

} // namespace

extern "C" __global__ void glm_index_logits_bf16_wmma_row8(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    glm_index_logits_wmma_impl<1>(query, weights, index_cache, logits, block_table,
        rows, seq_len_start, logits_stride, index_heads, head_dim, pool_size,
        cache_block_size, index_block_stride_bytes);
}

extern "C" __global__ void glm_index_logits_bf16_wmma_row8_pool32(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    glm_index_logits_wmma_impl<2>(query, weights, index_cache, logits, block_table,
        rows, seq_len_start, logits_stride, index_heads, head_dim, pool_size,
        cache_block_size, index_block_stride_bytes);
}
