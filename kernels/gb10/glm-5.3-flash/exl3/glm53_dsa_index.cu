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

__device__ __forceinline__ float bf16_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

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

}  // namespace

// Append packed chunks of preprojected BF16 index keys and compression gates.
// One block owns one sequence, so the four-token tail transition is atomic.
// State pointers are u64[sequence, 5]: latent cache, pooled keys, tail keys,
// tail gates, metadata. Metadata is int32[first_valid, valid_count,
// pool_count, tail_len].
extern "C" __global__ void glm53_dsa_pool_append(
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ gates,
    const __nv_bfloat16* __restrict__ ape,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ positions,
    const unsigned char* __restrict__ valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    unsigned int num_sequences) {
  const unsigned int sequence = blockIdx.x;
  const unsigned int channel = threadIdx.x;
  if (sequence >= num_sequences || channel >= INDEX_DIM) return;

  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  auto* pooled = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 1]);
  auto* tail_keys = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 2]);
  auto* tail_gates = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 3]);
  auto* metadata = reinterpret_cast<int*>(sequence_state_ptrs[ptr_base + 4]);

  for (int token = cu_seqlens[sequence];
       token < cu_seqlens[sequence + 1]; ++token) {
    if (valid[token] != 0) {
      const int tail_len = metadata[3];
      const unsigned long long token_base =
          static_cast<unsigned long long>(token) * INDEX_DIM;
      if (tail_len < static_cast<int>(KPOOL - 1)) {
        const unsigned long long tail_index =
            static_cast<unsigned long long>(tail_len) * INDEX_DIM + channel;
        tail_keys[tail_index] = keys[token_base + channel];
        tail_gates[tail_index] = gates[token_base + channel];
      } else {
        float logits[KPOOL];
        float maximum = -FLT_MAX;
        for (unsigned int lane = 0; lane < KPOOL; ++lane) {
          const float gate = lane == KPOOL - 1
                                 ? __bfloat162float(gates[token_base + channel])
                                 : __bfloat162float(
                                       tail_gates[lane * INDEX_DIM + channel]);
          logits[lane] =
              gate + __bfloat162float(ape[lane * INDEX_DIM + channel]);
          maximum = fmaxf(maximum, logits[lane]);
        }
        float denominator = 0.0f;
        for (unsigned int lane = 0; lane < KPOOL; ++lane) {
          logits[lane] = expf(logits[lane] - maximum);
          denominator += logits[lane];
        }
        float sum = 0.0f;
        for (unsigned int lane = 0; lane < KPOOL; ++lane) {
          const float probability = bf16_round(logits[lane] / denominator);
          const float key = lane == KPOOL - 1
                                ? __bfloat162float(keys[token_base + channel])
                                : __bfloat162float(
                                      tail_keys[lane * INDEX_DIM + channel]);
          sum += bf16_round(probability * key);
        }
        pooled[static_cast<unsigned long long>(metadata[2]) * INDEX_DIM +
               channel] = __float2bfloat16_rn(sum);
      }
      __syncthreads();
      if (channel == 0) {
        if (metadata[1] == 0) metadata[0] = positions[token];
        ++metadata[1];
        if (tail_len == static_cast<int>(KPOOL - 1)) {
          ++metadata[2];
          metadata[3] = 0;
        } else {
          metadata[3] = tail_len + 1;
        }
      }
    }
    __syncthreads();
  }
}

// Score every complete, visible pool for one decode query. Queries and the
// learned projection output are BF16. Dot products, the learned head-weight
// combination, and the official 1/sqrt(32) scaling are FP32, matching the
// pinned Transformers implementation without a separate BF16->FP32 launch.
extern "C" __global__ void glm53_dsa_score(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ head_weights,
    const unsigned char* __restrict__ query_valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    float* __restrict__ scores,
    unsigned int max_pools,
    unsigned int num_sequences) {
  const unsigned int pool = blockIdx.x;
  const unsigned int sequence = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  if (sequence >= num_sequences || pool >= max_pools) return;

  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* pooled = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 1]);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[ptr_base + 4]);
  const unsigned long long score_index =
      static_cast<unsigned long long>(sequence) * max_pools + pool;
  if (query_valid[sequence] == 0 || pool >= static_cast<unsigned int>(metadata[2])) {
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
        (static_cast<unsigned long long>(sequence) * INDEX_HEADS + head) *
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
                    __bfloat162float(head_weights[sequence * INDEX_HEADS + head]) *
                    fmaxf(dot * 0.08838834764831843f, 0.0f) *
                    0.1767766952966369f;
    }
  }

  __shared__ float partial[4];
  if (lane == 0) partial[warp] = warp_score;
  __syncthreads();
  if (tid == 0) scores[score_index] = partial[0] + partial[1] + partial[2] + partial[3];
}

// Select the highest-scoring 512 complete pools, expand them to 2048 raw
// positions, append the current 0..3-token tail, and pad to int32[2051].
// `scores` is scratch and is consumed by setting selected entries to -FLT_MAX.
extern "C" __global__ void glm53_dsa_topk_expand_decode(
    float* __restrict__ scores,
    const unsigned char* __restrict__ query_valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    int* __restrict__ output,
    unsigned int max_pools,
    unsigned int num_sequences) {
  const unsigned int sequence = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (sequence >= num_sequences) return;

  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[ptr_base + 4]);
  const unsigned int pool_count =
      min(static_cast<unsigned int>(metadata[2]), max_pools);
  const unsigned int select_count = min(pool_count, TOP_POOLS);
  const unsigned long long output_base =
      static_cast<unsigned long long>(sequence) * OUTPUT_WIDTH;
  for (unsigned int index = tid; index < OUTPUT_WIDTH; index += blockDim.x) {
    output[output_base + index] = -1;
  }
  __syncthreads();
  if (query_valid[sequence] == 0) return;

  // With no more than 512 complete pools, top-k membership is the full
  // visible set. Emit it directly instead of ranking every pool only to keep
  // all of them. This is the common short/medium-context decode path.
  if (pool_count <= TOP_POOLS) {
    for (unsigned int index = tid; index < pool_count * KPOOL;
         index += blockDim.x) {
      output[output_base + index] = metadata[0] + index;
    }
    for (unsigned int lane = tid;
         lane < static_cast<unsigned int>(metadata[3]); lane += blockDim.x) {
      output[output_base + pool_count * KPOOL + lane] =
          metadata[0] + pool_count * KPOOL + lane;
    }
    return;
  }

  __shared__ float candidate_scores[256];
  __shared__ int candidate_indices[256];
  __shared__ int selected[TOP_POOLS];
  float* const sequence_scores = scores +
      static_cast<unsigned long long>(sequence) * max_pools;

  for (unsigned int pick = 0; pick < select_count; ++pick) {
    float best_score = -FLT_MAX;
    int best_index = 0x7fffffff;
    for (unsigned int pool = tid; pool < pool_count; pool += blockDim.x) {
      const float score = sequence_scores[pool];
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
      sequence_scores[candidate_indices[0]] = -FLT_MAX;
    }
    __syncthreads();
  }

  for (unsigned int index = tid; index < select_count * KPOOL;
       index += blockDim.x) {
    const unsigned int pool_rank = index / KPOOL;
    const unsigned int lane = index % KPOOL;
    output[output_base + index] =
        metadata[0] + selected[pool_rank] * KPOOL + lane;
  }
  for (unsigned int lane = tid; lane < static_cast<unsigned int>(metadata[3]);
       lane += blockDim.x) {
    output[output_base + select_count * KPOOL + lane] =
        metadata[0] + metadata[2] * KPOOL + lane;
  }
}

// Sparse absorbed-MLA decode over the selected raw token IDs. The normalized
// 512-wide latent is both the absorbed key and the latent value; callers apply
// the per-head W_UV projection after this kernel.
extern "C" __global__ void glm53_dsa_sparse_mla_decode(
    const __nv_bfloat16* __restrict__ absorbed_query,
    const int* __restrict__ selected_indices,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_heads,
    unsigned int num_sequences,
    float attention_scale) {
  const unsigned int head = blockIdx.x;
  const unsigned int sequence = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp = tid >> 5;
  const unsigned int lane = tid & 31;
  if (head >= num_heads || sequence >= num_sequences) return;

  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* latent = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[ptr_base]);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[ptr_base + 4]);
  const auto* selected = selected_indices +
      static_cast<unsigned long long>(sequence) * OUTPUT_WIDTH;
  const unsigned long long query_base =
      (static_cast<unsigned long long>(sequence) * num_heads + head) * MLA_DIM;
  const unsigned int channel_base = lane * (MLA_DIM / 32);

  float maximum = -FLT_MAX;
  float denominator = 0.0f;
  float accumulator[MLA_DIM / 32];
  #pragma unroll
  for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
    accumulator[part] = 0.0f;
  }

  // The selection buffer has fixed 2,051-token capacity for graph stability,
  // but short contexts contain only the complete pools plus the live tail.
  // Scanning the padded -1 suffix costs more than the real attention on GB10.
  const unsigned int selected_count =
      min(static_cast<unsigned int>(metadata[2]), TOP_POOLS) * KPOOL +
      static_cast<unsigned int>(metadata[3]);
  for (unsigned int rank = warp; rank < selected_count; rank += MLA_WARPS) {
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
    dot = warp_sum(dot);
    dot = __shfl_sync(0xffffffff, dot, 0);
    const float score = dot * attention_scale;
    const float next_maximum = fmaxf(maximum, score);
    const float old_scale = expf(maximum - next_maximum);
    const float new_scale = expf(score - next_maximum);
    denominator = denominator * old_scale + new_scale;
    #pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      accumulator[part] =
          accumulator[part] * old_scale +
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
    const unsigned long long output_base = query_base;
    const float inverse = warp_denominator[0] > 0.0f
                              ? 1.0f / warp_denominator[0]
                              : 0.0f;
    #pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      output[output_base + channel_base + part] =
          __float2bfloat16_rn(warp_output[0][channel_base + part] * inverse);
    }
  }
}
