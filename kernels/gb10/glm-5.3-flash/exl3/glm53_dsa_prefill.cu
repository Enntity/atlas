// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <float.h>

namespace {

constexpr unsigned int INDEX_DIM = 128;
constexpr unsigned int INDEX_HEADS = 32;
constexpr unsigned int KPOOL = 4;
constexpr unsigned int TOP_POOLS = 512;
constexpr unsigned int OUTPUT_WIDTH = TOP_POOLS * KPOOL + KPOOL - 1;
constexpr unsigned int MLA_DIM = 512;
constexpr unsigned int MLA_WARPS = 8;

__device__ __forceinline__ float warp_sum(float value) {
  for (unsigned int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ bool better(float left_score,
                                       int left_index,
                                       float right_score,
                                       int right_index) {
  return left_score > right_score ||
         (left_score == right_score && left_index < right_index);
}

__device__ __forceinline__ unsigned int visible_pools(
    const int* metadata, int position, unsigned int max_pools) {
  if (metadata[1] == 0 || position < metadata[0]) return 0;
  const unsigned int seen = static_cast<unsigned int>(position - metadata[0] + 1);
  return min(min(seen / KPOOL, static_cast<unsigned int>(metadata[2])),
             max_pools);
}

__device__ __forceinline__ unsigned int sequence_for_token(
    unsigned int token,
    const int* cu_seqlens,
    unsigned int num_sequences) {
  unsigned int sequence = 0;
  while (sequence + 1 < num_sequences &&
         token >= static_cast<unsigned int>(cu_seqlens[sequence + 1])) {
    ++sequence;
  }
  return sequence;
}

}  // namespace

// Score every prompt query against only the complete pools visible at that
// query's absolute position. The whole latent/index state may already contain
// the chunk: the position-derived pool ceiling preserves exact causality.
extern "C" __global__ void glm53_dsa_score_prefill(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ head_weights,
    const int* __restrict__ positions,
    const unsigned char* __restrict__ valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const int* __restrict__ cu_seqlens,
    float* __restrict__ scores,
    unsigned int max_pools,
    unsigned int total_tokens,
    unsigned int num_sequences) {
  const unsigned int pool = blockIdx.x;
  const unsigned int token = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  if (token >= total_tokens || pool >= max_pools) return;

  const unsigned int sequence =
      sequence_for_token(token, cu_seqlens, num_sequences);
  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* pooled = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 1]);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[ptr_base + 4]);
  const unsigned long long score_index =
      static_cast<unsigned long long>(token) * max_pools + pool;
  const unsigned int pool_count =
      valid[token] == 0 ? 0 : visible_pools(metadata, positions[token], max_pools);
  if (pool >= pool_count) {
    if (tid == 0) scores[score_index] = -FLT_MAX;
    return;
  }

  const unsigned int lane = tid & 31;
  const unsigned int warp = tid >> 5;
  const unsigned long long pool_base =
      static_cast<unsigned long long>(pool) * INDEX_DIM;
  float key_values[4];
#pragma unroll
  for (unsigned int part = 0; part < 4; ++part) {
    key_values[part] =
        __bfloat162float(pooled[pool_base + lane + part * 32]);
  }

  float warp_score = 0.0f;
  for (unsigned int head = warp; head < INDEX_HEADS; head += 4) {
    const unsigned long long query_base =
        (static_cast<unsigned long long>(token) * INDEX_HEADS + head) *
        INDEX_DIM;
    float dot = 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < 4; ++part) {
      dot += __bfloat162float(query[query_base + lane + part * 32]) *
             key_values[part];
    }
    dot = warp_sum(dot);
    if (lane == 0) {
      warp_score +=
          __bfloat162float(head_weights[token * INDEX_HEADS + head]) *
          fmaxf(dot * 0.08838834764831843f, 0.0f) *
          0.1767766952966369f;
    }
  }

  __shared__ float partial[4];
  if (lane == 0) partial[warp] = warp_score;
  __syncthreads();
  if (tid == 0) {
    scores[score_index] = partial[0] + partial[1] + partial[2] + partial[3];
  }
}

// Select pools independently for every prompt token and append that token's
// visible incomplete tail. One block owns one query, so all queries run in
// parallel while each query retains the decode kernel's deterministic order.
extern "C" __global__ void glm53_dsa_topk_expand_prefill(
    float* __restrict__ scores,
    const int* __restrict__ positions,
    const unsigned char* __restrict__ valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const int* __restrict__ cu_seqlens,
    int* __restrict__ output,
    unsigned int max_pools,
    unsigned int total_tokens,
    unsigned int num_sequences) {
  const unsigned int token = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (token >= total_tokens) return;

  const unsigned int sequence =
      sequence_for_token(token, cu_seqlens, num_sequences);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[static_cast<unsigned long long>(sequence) * 5 + 4]);
  const unsigned long long output_base =
      static_cast<unsigned long long>(token) * OUTPUT_WIDTH;
  for (unsigned int index = tid; index < OUTPUT_WIDTH; index += blockDim.x) {
    output[output_base + index] = -1;
  }
  __syncthreads();
  if (valid[token] == 0 || positions[token] < metadata[0]) return;

  const unsigned int pool_count =
      visible_pools(metadata, positions[token], max_pools);
  const unsigned int select_count = min(pool_count, TOP_POOLS);

  // Before 2,048 complete tokens every visible pool fits in the official
  // top-512 set. Ranking cannot change membership, so emit the complete
  // causal range directly and avoid an O(pool_count^2) selection sort. The
  // sparse MLA reduction is mathematically permutation-invariant.
  if (pool_count <= TOP_POOLS) {
    for (unsigned int index = tid; index < pool_count * KPOOL;
         index += blockDim.x) {
      output[output_base + index] = metadata[0] + index;
    }
    const unsigned int seen =
        static_cast<unsigned int>(positions[token] - metadata[0] + 1);
    const unsigned int tail_len = seen % KPOOL;
    for (unsigned int lane = tid; lane < tail_len; lane += blockDim.x) {
      output[output_base + pool_count * KPOOL + lane] =
          metadata[0] + pool_count * KPOOL + lane;
    }
    return;
  }

  __shared__ float candidate_scores[256];
  __shared__ int candidate_indices[256];
  __shared__ int selected[TOP_POOLS];
  float* const token_scores =
      scores + static_cast<unsigned long long>(token) * max_pools;

  for (unsigned int pick = 0; pick < select_count; ++pick) {
    float best_score = -FLT_MAX;
    int best_index = 0x7fffffff;
    for (unsigned int pool = tid; pool < pool_count; pool += blockDim.x) {
      const float score = token_scores[pool];
      if (better(score, static_cast<int>(pool), best_score, best_index)) {
        best_score = score;
        best_index = static_cast<int>(pool);
      }
    }
    candidate_scores[tid] = best_score;
    candidate_indices[tid] = best_index;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
      if (tid < stride && better(candidate_scores[tid + stride],
                                 candidate_indices[tid + stride],
                                 candidate_scores[tid],
                                 candidate_indices[tid])) {
        candidate_scores[tid] = candidate_scores[tid + stride];
        candidate_indices[tid] = candidate_indices[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      selected[pick] = candidate_indices[0];
      token_scores[candidate_indices[0]] = -FLT_MAX;
    }
    __syncthreads();
  }

  for (unsigned int index = tid; index < select_count * KPOOL;
       index += blockDim.x) {
    output[output_base + index] =
        metadata[0] + selected[index / KPOOL] * KPOOL + index % KPOOL;
  }
  const unsigned int seen =
      static_cast<unsigned int>(positions[token] - metadata[0] + 1);
  const unsigned int tail_len = seen % KPOOL;
  for (unsigned int lane = tid; lane < tail_len; lane += blockDim.x) {
    output[output_base + select_count * KPOOL + lane] =
        metadata[0] + pool_count * KPOOL + lane;
  }
}

// Causal sparse absorbed-MLA for every prompt token. Selection rows already
// contain only positions visible to their query, so latent-cache entries from
// later in the chunk are unreachable.
extern "C" __global__ void glm53_dsa_sparse_mla_prefill(
    const __nv_bfloat16* __restrict__ absorbed_query,
    const int* __restrict__ selected_indices,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const int* __restrict__ cu_seqlens,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_heads,
    unsigned int total_tokens,
    unsigned int num_sequences,
    float attention_scale) {
  const unsigned int head = blockIdx.x;
  const unsigned int token = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp = tid >> 5;
  const unsigned int lane = tid & 31;
  if (head >= num_heads || token >= total_tokens) return;

  const unsigned int sequence =
      sequence_for_token(token, cu_seqlens, num_sequences);
  const auto* latent = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[static_cast<unsigned long long>(sequence) * 5]);
  const auto* selected = selected_indices +
      static_cast<unsigned long long>(token) * OUTPUT_WIDTH;
  const unsigned long long query_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_DIM;
  const unsigned int channel_base = lane * (MLA_DIM / 32);

  float maximum = -FLT_MAX;
  float denominator = 0.0f;
  float accumulator[MLA_DIM / 32];
#pragma unroll
  for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
    accumulator[part] = 0.0f;
  }
  for (unsigned int rank = warp; rank < OUTPUT_WIDTH; rank += MLA_WARPS) {
    const int position = selected[rank];
    if (position < 0) continue;
    const unsigned long long latent_base =
        static_cast<unsigned long long>(position) * MLA_DIM + channel_base;
    float dot = 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      dot += __bfloat162float(absorbed_query[query_base + channel_base + part]) *
             __bfloat162float(latent[latent_base + part]);
    }
    dot = __shfl_sync(0xffffffff, warp_sum(dot), 0);
    const float score = dot * attention_scale;
    const float next_maximum = fmaxf(maximum, score);
    const float old_scale = expf(maximum - next_maximum);
    const float new_scale = expf(score - next_maximum);
    denominator = denominator * old_scale + new_scale;
#pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      accumulator[part] = accumulator[part] * old_scale +
          new_scale * __bfloat162float(latent[latent_base + part]);
    }
    maximum = next_maximum;
  }

  __shared__ float warp_maximum[MLA_WARPS];
  __shared__ float warp_denominator[MLA_WARPS];
  __shared__ float warp_output[MLA_WARPS][MLA_DIM];
  if (lane == 0) {
    warp_maximum[warp] = maximum;
    warp_denominator[warp] = denominator;
  }
#pragma unroll
  for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
    warp_output[warp][channel_base + part] = accumulator[part];
  }
  __syncthreads();
  for (unsigned int stride = MLA_WARPS / 2; stride != 0; stride >>= 1) {
    if (warp < stride) {
      const unsigned int other = warp + stride;
      const float merged_maximum =
          fmaxf(warp_maximum[warp], warp_maximum[other]);
      const float left_scale = expf(warp_maximum[warp] - merged_maximum);
      const float right_scale = expf(warp_maximum[other] - merged_maximum);
      if (lane == 0) {
        warp_denominator[warp] = warp_denominator[warp] * left_scale +
                                 warp_denominator[other] * right_scale;
        warp_maximum[warp] = merged_maximum;
      }
#pragma unroll
      for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
        const unsigned int channel = channel_base + part;
        warp_output[warp][channel] =
            warp_output[warp][channel] * left_scale +
            warp_output[other][channel] * right_scale;
      }
    }
    __syncthreads();
  }
  if (warp == 0) {
    const float inverse = warp_denominator[0] > 0.0f
                              ? 1.0f / warp_denominator[0]
                              : 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      output[query_base + channel_base + part] = __float2bfloat16_rn(
          warp_output[0][channel_base + part] * inverse);
    }
  }
}
