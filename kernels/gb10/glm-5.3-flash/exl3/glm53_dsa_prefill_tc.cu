// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3 TP2 sparse-MLA prefill for GB10. The fixed geometry lets QK share
// every gathered latent across sixteen heads and XV share it across all 32
// heads. BF16 tensor-core MMA keeps the cache and softmax numerics above the
// FP8 FlashInfer path while removing the scalar one-warp-per-head rereads.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;
using bf16 = __nv_bfloat16;

constexpr unsigned int HEADS = 32;
constexpr unsigned int HEAD_TILE = 16;
constexpr unsigned int LATENT = 512;
constexpr unsigned int CAUSAL_WIDTH = 2048;
constexpr unsigned int SELECTED_WIDTH = 2051;
constexpr unsigned int TC_WIDTH = 2064;
constexpr unsigned int SCORE_WARPS = 4;
constexpr unsigned int KEY_TILE = 16;
constexpr unsigned int VALUE_TILE = 128;

using MatrixA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                               bf16, wmma::row_major>;
using MatrixBCol = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                  bf16, wmma::col_major>;
using MatrixBRow = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                  bf16, wmma::row_major>;
using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (unsigned int mask = 16; mask != 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (unsigned int mask = 16; mask != 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__device__ __forceinline__ unsigned int sequence_for_token(
    unsigned int token, const int* cu_seqlens, unsigned int num_sequences) {
  unsigned int sequence = 0;
  while (sequence + 1 < num_sequences &&
         token >= static_cast<unsigned int>(cu_seqlens[sequence + 1])) {
    ++sequence;
  }
  return sequence;
}

}  // namespace

extern "C" __global__ void glm53_dsa_prefill_tc_scores(
    const bf16* __restrict__ query,
    const int* __restrict__ selected_indices,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const int* __restrict__ cu_seqlens,
    float* __restrict__ scores,
    bf16* __restrict__ weights,
    unsigned int total_tokens,
    unsigned int num_sequences,
    float attention_scale) {
  const unsigned int head_start = blockIdx.x * HEAD_TILE;
  const unsigned int token = blockIdx.y;
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (head_start >= HEADS || token >= total_tokens) return;

  const unsigned int sequence =
      sequence_for_token(token, cu_seqlens, num_sequences);
  const auto* latent = reinterpret_cast<const bf16*>(
      sequence_state_ptrs[static_cast<unsigned long long>(sequence) * 5]);
  const int* selected = selected_indices +
      static_cast<unsigned long long>(token) * SELECTED_WIDTH;
  // The 8-query causal kernel reuses each latent row across the short-prefix
  // half of the prompt. Tensor cores take over only once the learned sparse
  // selector has more than 2,048 visible positions to prune.
  if (selected[CAUSAL_WIDTH] < 0) return;
  const bf16* token_query = query +
      static_cast<unsigned long long>(token) * HEADS * LATENT +
      head_start * LATENT;

  extern __shared__ unsigned char smem_raw[];
  bf16* query_shared = reinterpret_cast<bf16*>(smem_raw);
  bf16* key_shared = query_shared + HEAD_TILE * LATENT;
  float* tile_shared = reinterpret_cast<float*>(
      key_shared + SCORE_WARPS * KEY_TILE * LATENT);

  for (unsigned int i = threadIdx.x; i < HEAD_TILE * LATENT;
       i += blockDim.x) {
    query_shared[i] = token_query[i];
  }
  __syncthreads();

  for (unsigned int rank_base = warp * KEY_TILE; rank_base < TC_WIDTH;
       rank_base += SCORE_WARPS * KEY_TILE) {
    bf16* warp_keys = key_shared + warp * KEY_TILE * LATENT;
    for (unsigned int i = lane; i < KEY_TILE * LATENT; i += 32) {
      const unsigned int key_row = i / LATENT;
      const unsigned int channel = i % LATENT;
      const unsigned int rank = rank_base + key_row;
      const int position = rank < SELECTED_WIDTH ? selected[rank] : -1;
      warp_keys[i] = position >= 0
          ? latent[static_cast<unsigned long long>(position) * LATENT + channel]
          : __float2bfloat16(0.0f);
    }
    __syncwarp();

    Accumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
    for (unsigned int k = 0; k < LATENT; k += 16) {
      MatrixA q_fragment;
      MatrixBCol k_fragment;
      wmma::load_matrix_sync(q_fragment, query_shared + k, LATENT);
      wmma::load_matrix_sync(k_fragment, warp_keys + k, LATENT);
      wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
    }
    float* warp_tile = tile_shared + warp * HEAD_TILE * KEY_TILE;
    wmma::store_matrix_sync(warp_tile, accumulator, KEY_TILE,
                            wmma::mem_row_major);
    __syncwarp();

    for (unsigned int i = lane; i < HEAD_TILE * KEY_TILE; i += 32) {
      const unsigned int head = i / KEY_TILE;
      const unsigned int rank = rank_base + i % KEY_TILE;
      const unsigned long long destination =
          (static_cast<unsigned long long>(token) * HEADS +
           head_start + head) * TC_WIDTH + rank;
      scores[destination] =
          rank < SELECTED_WIDTH && selected[rank] >= 0
              ? warp_tile[i] * attention_scale
              : -FLT_MAX;
    }
    __syncwarp();
  }
  __syncthreads();

  for (unsigned int local_head = warp; local_head < HEAD_TILE;
       local_head += SCORE_WARPS) {
    const unsigned long long base =
        (static_cast<unsigned long long>(token) * HEADS +
         head_start + local_head) * TC_WIDTH;
    float maximum = -FLT_MAX;
    for (unsigned int rank = lane; rank < TC_WIDTH; rank += 32) {
      maximum = fmaxf(maximum, scores[base + rank]);
    }
    maximum = warp_max(maximum);
    float denominator = 0.0f;
    for (unsigned int rank = lane; rank < TC_WIDTH; rank += 32) {
      const float score = scores[base + rank];
      denominator += score == -FLT_MAX ? 0.0f : expf(score - maximum);
    }
    denominator = warp_sum(denominator);
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (unsigned int rank = lane; rank < TC_WIDTH; rank += 32) {
      const float score = scores[base + rank];
      weights[base + rank] = __float2bfloat16_rn(
          score == -FLT_MAX ? 0.0f : expf(score - maximum) * inverse);
    }
  }
}

extern "C" __global__ void glm53_dsa_prefill_tc_values(
    const bf16* __restrict__ weights,
    const int* __restrict__ selected_indices,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const int* __restrict__ cu_seqlens,
    bf16* __restrict__ output,
    unsigned int total_tokens,
    unsigned int num_sequences) {
  const unsigned int output_start = blockIdx.x * VALUE_TILE;
  const unsigned int token = blockIdx.y;
  const unsigned int warp = threadIdx.x >> 5;
  if (output_start >= LATENT || token >= total_tokens || warp >= 2) return;

  const unsigned int sequence =
      sequence_for_token(token, cu_seqlens, num_sequences);
  const auto* latent = reinterpret_cast<const bf16*>(
      sequence_state_ptrs[static_cast<unsigned long long>(sequence) * 5]);
  const int* selected = selected_indices +
      static_cast<unsigned long long>(token) * SELECTED_WIDTH;
  if (selected[CAUSAL_WIDTH] < 0) return;
  extern __shared__ unsigned char smem_raw[];
  bf16* value_shared = reinterpret_cast<bf16*>(smem_raw);
  float* output_shared = reinterpret_cast<float*>(
      value_shared + KEY_TILE * VALUE_TILE);

  Accumulator accumulator[VALUE_TILE / 16];
#pragma unroll
  for (unsigned int tile = 0; tile < VALUE_TILE / 16; ++tile) {
    wmma::fill_fragment(accumulator[tile], 0.0f);
  }

  const bf16* token_weights = weights +
      static_cast<unsigned long long>(token) * HEADS * TC_WIDTH;
  for (unsigned int rank_base = 0; rank_base < TC_WIDTH;
       rank_base += KEY_TILE) {
    for (unsigned int i = threadIdx.x; i < KEY_TILE * VALUE_TILE;
         i += blockDim.x) {
      const unsigned int key_row = i / VALUE_TILE;
      const unsigned int channel = i % VALUE_TILE;
      const unsigned int rank = rank_base + key_row;
      const int position = rank < SELECTED_WIDTH ? selected[rank] : -1;
      value_shared[i] = position >= 0
          ? latent[static_cast<unsigned long long>(position) * LATENT +
                   output_start + channel]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();

    MatrixA weight_fragment;
    wmma::load_matrix_sync(
        weight_fragment,
        token_weights + warp * HEAD_TILE * TC_WIDTH + rank_base,
        TC_WIDTH);
#pragma unroll
    for (unsigned int tile = 0; tile < VALUE_TILE / 16; ++tile) {
      MatrixBRow value_fragment;
      wmma::load_matrix_sync(value_fragment, value_shared + tile * 16,
                             VALUE_TILE);
      wmma::mma_sync(accumulator[tile], weight_fragment, value_fragment,
                     accumulator[tile]);
    }
    __syncthreads();
  }

  float* warp_output = output_shared + warp * HEAD_TILE * VALUE_TILE;
#pragma unroll
  for (unsigned int tile = 0; tile < VALUE_TILE / 16; ++tile) {
    wmma::store_matrix_sync(warp_output + tile * 16, accumulator[tile],
                            VALUE_TILE, wmma::mem_row_major);
  }
  __syncthreads();
  for (unsigned int i = threadIdx.x; i < HEADS * VALUE_TILE;
       i += blockDim.x) {
    const unsigned int head = i / VALUE_TILE;
    const unsigned int channel = i % VALUE_TILE;
    output[(static_cast<unsigned long long>(token) * HEADS + head) * LATENT +
           output_start + channel] = __float2bfloat16_rn(output_shared[i]);
  }
}
