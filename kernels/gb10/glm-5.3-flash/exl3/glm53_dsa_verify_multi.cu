// SPDX-License-Identifier: AGPL-3.0-only

// Depth-major sparse-MLA verification for the fixed GLM-5.3 DFlash engine.
// Input/output rows remain sequence-major [N,K,...], but each launch advances
// one causal depth across all N sequences. This replaces N*K one-sequence
// launches with K N-wide launches without changing per-sequence state order.

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

}  // namespace

extern "C" __global__ void glm53_dsa_verify_multi_latent_append(
    const __nv_bfloat16* __restrict__ latent,
    const int* __restrict__ positions,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    unsigned int rows_per_sequence,
    unsigned int num_sequences,
    unsigned int latent_capacity) {
  const unsigned int sequence = blockIdx.x / rows_per_sequence;
  const unsigned int token_depth = blockIdx.x % rows_per_sequence;
  if (sequence >= num_sequences) return;
  const unsigned int row = sequence * rows_per_sequence + token_depth;
  const int position = positions[row];
  if (position < 0 || static_cast<unsigned int>(position) >= latent_capacity) {
    return;
  }
  auto* destination = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[static_cast<unsigned long long>(sequence) * 5]);
  const unsigned long long source_base =
      static_cast<unsigned long long>(row) * MLA_DIM;
  const unsigned long long destination_base =
      static_cast<unsigned long long>(position) * MLA_DIM;
  for (unsigned int channel = threadIdx.x; channel < MLA_DIM;
       channel += blockDim.x) {
    destination[destination_base + channel] = latent[source_base + channel];
  }
}

extern "C" __global__ void glm53_dsa_verify_multi_pool_append(
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ gates,
    const __nv_bfloat16* __restrict__ ape,
    const int* __restrict__ positions,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    unsigned int rows_per_sequence,
    unsigned int num_sequences,
    unsigned int token_depth) {
  const unsigned int sequence = blockIdx.x;
  const unsigned int channel = threadIdx.x;
  if (sequence >= num_sequences || channel >= INDEX_DIM ||
      token_depth >= rows_per_sequence) {
    return;
  }
  const unsigned int row = sequence * rows_per_sequence + token_depth;
  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  auto* pooled = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 1]);
  auto* tail_keys = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 2]);
  auto* tail_gates = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 3]);
  auto* metadata = reinterpret_cast<int*>(sequence_state_ptrs[ptr_base + 4]);
  const int tail_len = metadata[3];
  const unsigned long long row_base =
      static_cast<unsigned long long>(row) * INDEX_DIM;
  if (tail_len < static_cast<int>(KPOOL - 1)) {
    const unsigned long long tail_index =
        static_cast<unsigned long long>(tail_len) * INDEX_DIM + channel;
    tail_keys[tail_index] = keys[row_base + channel];
    tail_gates[tail_index] = gates[row_base + channel];
  } else {
    float logits[KPOOL];
    float maximum = -FLT_MAX;
    #pragma unroll
    for (unsigned int lane = 0; lane < KPOOL; ++lane) {
      const float gate = lane == KPOOL - 1
                             ? __bfloat162float(gates[row_base + channel])
                             : __bfloat162float(
                                   tail_gates[lane * INDEX_DIM + channel]);
      logits[lane] = gate + __bfloat162float(ape[lane * INDEX_DIM + channel]);
      maximum = fmaxf(maximum, logits[lane]);
    }
    float denominator = 0.0f;
    #pragma unroll
    for (unsigned int lane = 0; lane < KPOOL; ++lane) {
      logits[lane] = expf(logits[lane] - maximum);
      denominator += logits[lane];
    }
    float sum = 0.0f;
    #pragma unroll
    for (unsigned int lane = 0; lane < KPOOL; ++lane) {
      const float probability = bf16_round(logits[lane] / denominator);
      const float key = lane == KPOOL - 1
                            ? __bfloat162float(keys[row_base + channel])
                            : __bfloat162float(
                                  tail_keys[lane * INDEX_DIM + channel]);
      sum += bf16_round(probability * key);
    }
    pooled[static_cast<unsigned long long>(metadata[2]) * INDEX_DIM + channel] =
        __float2bfloat16_rn(sum);
  }
  __syncthreads();
  if (channel == 0) {
    if (metadata[1] == 0) metadata[0] = positions[row];
    ++metadata[1];
    if (tail_len == static_cast<int>(KPOOL - 1)) {
      ++metadata[2];
      metadata[3] = 0;
    } else {
      metadata[3] = tail_len + 1;
    }
  }
  __syncthreads();

  // Every non-final causal depth is a legal rollback point. Snapshot the
  // updated incomplete-pool state while this CTA still owns it, instead of
  // launching a separate copy kernel after sparse attention.
  if (token_depth + 1 < rows_per_sequence) {
    const unsigned long long snapshot =
        static_cast<unsigned long long>(num_sequences) * 5 +
        (static_cast<unsigned long long>(token_depth) * num_sequences + sequence) * 3;
    auto* snapshot_keys = reinterpret_cast<__nv_bfloat16*>(
        sequence_state_ptrs[snapshot]);
    auto* snapshot_gates = reinterpret_cast<__nv_bfloat16*>(
        sequence_state_ptrs[snapshot + 1]);
    auto* snapshot_metadata = reinterpret_cast<int*>(
        sequence_state_ptrs[snapshot + 2]);
    for (unsigned int element = channel; element < (KPOOL - 1) * INDEX_DIM;
         element += blockDim.x) {
      snapshot_keys[element] = tail_keys[element];
      snapshot_gates[element] = tail_gates[element];
    }
    if (channel < 4) snapshot_metadata[channel] = metadata[channel];
  }
}

extern "C" __global__ void glm53_dsa_verify_multi_score(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ head_weights,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    float* __restrict__ scores,
    unsigned int max_pools,
    unsigned int rows_per_sequence,
    unsigned int num_sequences,
    unsigned int token_depth) {
  const unsigned int pool = blockIdx.x;
  const unsigned int sequence = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  if (sequence >= num_sequences || pool >= max_pools ||
      token_depth >= rows_per_sequence) {
    return;
  }
  const unsigned int row = sequence * rows_per_sequence + token_depth;
  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* pooled = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[ptr_base + 1]);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[ptr_base + 4]);
  const unsigned long long score_index =
      static_cast<unsigned long long>(sequence) * max_pools + pool;
  if (pool >= static_cast<unsigned int>(metadata[2])) {
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
        (static_cast<unsigned long long>(row) * INDEX_HEADS + head) * INDEX_DIM;
    float dot = 0.0f;
    #pragma unroll
    for (unsigned int part = 0; part < 4; ++part) {
      dot += __bfloat162float(query[query_base + lane + part * 32]) *
             key_values[part];
    }
    dot = warp_sum(dot);
    if (lane == 0) {
      warp_score +=
          __bfloat162float(head_weights[row * INDEX_HEADS + head]) *
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

extern "C" __global__ void glm53_dsa_verify_multi_sparse_mla(
    const __nv_bfloat16* __restrict__ absorbed_query,
    const int* __restrict__ selected_indices,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_heads,
    unsigned int rows_per_sequence,
    unsigned int num_sequences,
    unsigned int token_depth,
    float attention_scale,
    unsigned int direct_selection) {
  const unsigned int head = blockIdx.x;
  const unsigned int sequence = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp = tid >> 5;
  const unsigned int lane = tid & 31;
  if (head >= num_heads || sequence >= num_sequences ||
      token_depth >= rows_per_sequence) {
    return;
  }
  const unsigned int row = sequence * rows_per_sequence + token_depth;
  const unsigned long long ptr_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* latent = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[ptr_base]);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[ptr_base + 4]);
  const auto* selected = selected_indices +
      static_cast<unsigned long long>(sequence) * OUTPUT_WIDTH;
  const unsigned long long query_base =
      (static_cast<unsigned long long>(row) * num_heads + head) * MLA_DIM;
  const unsigned int channel_base = lane * (MLA_DIM / 32);
  float maximum = -FLT_MAX;
  float denominator = 0.0f;
  float accumulator[MLA_DIM / 32] = {};
  // Preserve the fixed graph/workspace shape while bounding work to the
  // request's actual visible tokens. The top-k buffer's remaining entries are
  // padding and were the dominant cost for short speculative sequences.
  const unsigned int selected_count =
      min(static_cast<unsigned int>(metadata[2]), TOP_POOLS) * KPOOL +
      static_cast<unsigned int>(metadata[3]);
  for (unsigned int rank = warp; rank < selected_count; rank += MLA_WARPS) {
    const int position = direct_selection != 0
        ? metadata[0] + static_cast<int>(rank)
        : selected[rank];
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

extern "C" __global__ void glm53_dsa_verify_multi_snapshot_tail(
    const unsigned long long* __restrict__ state_ptrs,
    unsigned int num_sequences,
    unsigned int snapshot_depth,
    unsigned int tail_elements) {
  const unsigned int sequence = blockIdx.x;
  if (sequence >= num_sequences) return;
  const unsigned long long current =
      static_cast<unsigned long long>(sequence) * 5;
  const unsigned long long snapshot =
      static_cast<unsigned long long>(num_sequences) * 5 +
      (static_cast<unsigned long long>(snapshot_depth) * num_sequences + sequence) * 3;
  const auto* source_keys = reinterpret_cast<const __nv_bfloat16*>(state_ptrs[current + 2]);
  const auto* source_gates = reinterpret_cast<const __nv_bfloat16*>(state_ptrs[current + 3]);
  const auto* source_metadata = reinterpret_cast<const int*>(state_ptrs[current + 4]);
  auto* target_keys = reinterpret_cast<__nv_bfloat16*>(state_ptrs[snapshot]);
  auto* target_gates = reinterpret_cast<__nv_bfloat16*>(state_ptrs[snapshot + 1]);
  auto* target_metadata = reinterpret_cast<int*>(state_ptrs[snapshot + 2]);
  for (unsigned int index = threadIdx.x; index < tail_elements;
       index += blockDim.x) {
    target_keys[index] = source_keys[index];
    target_gates[index] = source_gates[index];
  }
  if (threadIdx.x < 4) target_metadata[threadIdx.x] = source_metadata[threadIdx.x];
}
