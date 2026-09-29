// SPDX-License-Identifier: AGPL-3.0-only
// GLM token-sharded latent storage (ATLAS_GLM_KV_SHARD=1): the device half of
// the ownership rule in crates/spark-runtime/src/kv_cache/latent_shard.rs —
// physical block `b` is stored by rank `b % world` at local slot `b / world`.
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
