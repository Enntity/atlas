// SPDX-License-Identifier: AGPL-3.0-only
// GLM token-sharded latent storage (ATLAS_GLM_KV_SHARD=1): the device half of
// the ownership rule in crates/spark-runtime/src/kv_cache/latent_shard.rs —
// physical block `b` is stored by rank `b % world` at local slot `b / world`.
// The ranks' ids differ, but the allocator gives the block at logical index
// `l` an id with `b % world == l % world`, so the ranks agree on owners.
// Keep the two in step.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Cache-write slot remap: global slot -> this rank's local slot, or -1 when
// another rank owns the block (every GLM latent writer skips slot < 0).
// Grid: ceil(n / 256), block 256.
extern "C" __global__ void glm_kv_shard_map_slots(
    const long long* __restrict__ slots,
    long long* __restrict__ local,
    unsigned int n,
    unsigned int block_size,
    unsigned int rank,
    unsigned int world) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long slot = slots[i];
    if (slot < 0) {
        local[i] = -1;
        return;
    }
    const long long bs = (long long)block_size;
    const long long block = slot / bs;
    local[i] = (block % (long long)world == (long long)rank)
        ? (block / (long long)world) * bs + slot % bs
        : -1;
}

// Selected-ID remap for this rank's partial attention: each logical token id
// becomes its local token id (read through an identity block table over the
// local pool) when this rank owns the token's block, else -1, which every
// GLM sparse kernel skips. `in == nullptr` generates causal ids instead: row
// `r` keeps tokens [0, causal_start + r + 1).
// Grid: (ceil(width / 256), rows), block 256.
extern "C" __global__ void glm_kv_shard_localize(
    const int* __restrict__ in,
    int* __restrict__ out,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int width,
    unsigned int block_size,
    unsigned int rank,
    unsigned int world,
    unsigned int causal_start) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int r = blockIdx.y;
    if (c >= width || r >= rows) return;
    const size_t i = (size_t)r * width + c;
    int token;
    if (in != nullptr) {
        token = in[i];
    } else {
        token = c < causal_start + r + 1u ? (int)c : -1;
    }
    int local = -1;
    if (token >= 0) {
        const unsigned int t = (unsigned int)token;
        const unsigned int block = block_table[t / block_size];
        if (block % world == rank) {
            local = (int)((block / world) * block_size + t % block_size);
        }
    }
    out[i] = local;
}

// ATLAS_GLM_KV_SHARD_COMPACT=1: glm_kv_shard_localize with each row's owned
// ids packed to the front in their selected order (a stable partition; the
// rest of the row is -1) and their number in `counts[row]`, so the split
// attention kernels walk only the tokens this rank stores instead of a row
// that is half -1. One CTA of 256 threads per row; thread `t` owns the
// `ceil(width / 256)` consecutive ids from `t * ceil(width / 256)`.
// `out` must not alias `in`. Grid: rows, block 256; width <= 256 * 16.
#define GLM_KV_SHARD_COMPACT_CHUNK 16u
extern "C" __global__ void glm_kv_shard_localize_compact(
    const int* __restrict__ in,
    int* __restrict__ out,
    unsigned int* __restrict__ counts,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int width,
    unsigned int block_size,
    unsigned int rank,
    unsigned int world,
    unsigned int causal_start) {
    __shared__ unsigned int s_kept[256];
    const unsigned int r = blockIdx.x;
    const unsigned int t = threadIdx.x;
    const unsigned int chunk = (width + 255u) / 256u;
    if (r >= rows || blockDim.x != 256u || chunk > GLM_KV_SHARD_COMPACT_CHUNK) return;
    const unsigned int begin = min(t * chunk, width);
    const unsigned int end = min(begin + chunk, width);
    const size_t row = (size_t)r * width;
    int local[GLM_KV_SHARD_COMPACT_CHUNK];
    unsigned int kept = 0;
    for (unsigned int c = begin; c < end; ++c) {
        const int token = in != nullptr ? in[row + c] : (c < causal_start + r + 1u ? (int)c : -1);
        if (token < 0) continue;
        const unsigned int id = (unsigned int)token;
        const unsigned int block = block_table[id / block_size];
        if (block % world == rank) {
            local[kept++] = (int)((block / world) * block_size + id % block_size);
        }
    }
    s_kept[t] = kept;
    __syncthreads();
    unsigned int before = 0, total = 0;
    for (unsigned int i = 0; i < 256u; ++i) {
        if (i == t) before = total;
        total += s_kept[i];
    }
    for (unsigned int k = 0; k < kept; ++k) out[row + before + k] = local[k];
    // The ids this thread dropped land after every kept id, in order.
    const unsigned int dropped = begin - before;
    for (unsigned int k = 0; k < (end - begin) - kept; ++k) out[row + total + dropped + k] = -1;
    if (t == 0) counts[r] = total;
}

// Block gather/scatter: dst block `dst_idx[i]` (or `i` when null) receives
// src block `src_idx[i]` (or `i` when null), for i < n. A block is
// `block_vecs` 16-byte vectors. Grid: n (never 0), block 256.
extern "C" __global__ void glm_kv_shard_copy_blocks(
    const uint4* __restrict__ src,
    const unsigned int* __restrict__ src_idx,
    uint4* __restrict__ dst,
    const unsigned int* __restrict__ dst_idx,
    unsigned int n,
    unsigned int block_vecs) {
    const unsigned int i = blockIdx.x;
    if (i >= n) return;
    const size_t s = (size_t)(src_idx != nullptr ? src_idx[i] : i) * block_vecs;
    const size_t d = (size_t)(dst_idx != nullptr ? dst_idx[i] : i) * block_vecs;
    for (unsigned int v = threadIdx.x; v < block_vecs; v += blockDim.x) {
        dst[d + v] = src[s + v];
    }
}
