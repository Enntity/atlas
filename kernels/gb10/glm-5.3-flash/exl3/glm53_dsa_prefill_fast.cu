// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <float.h>

namespace {

constexpr unsigned int MLA_DIM = 512;
constexpr unsigned int TOP_POOLS = 512;
constexpr unsigned int KPOOL = 4;
constexpr unsigned int CAUSAL_WIDTH = TOP_POOLS * KPOOL;
constexpr unsigned int OUTPUT_WIDTH = TOP_POOLS * KPOOL + KPOOL - 1;
constexpr unsigned int QUERY_TILE = 8;

__device__ __forceinline__ float warp_sum(float value) {
  for (unsigned int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
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

// Exact causal MLA for the common <=2,048-token prefix. One block owns eight
// queries from the same sequence and one head. Every 512-wide latent row is
// loaded once into shared memory, then reused by all eight query warps. This
// removes the prior eight-warp reduction and cuts latent-cache traffic by up
// to 8x without changing cache precision or attention mathematics.
extern "C" __global__ void glm53_dsa_causal_mla_prefill(
    const __nv_bfloat16* __restrict__ absorbed_query,
    const int* __restrict__ positions,
    const unsigned char* __restrict__ valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const int* __restrict__ cu_seqlens,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_heads,
    unsigned int total_tokens,
    unsigned int num_sequences,
    float attention_scale) {
  const unsigned int head = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int tiles_per_sequence =
      (total_tokens + QUERY_TILE - 1) / QUERY_TILE;
  const unsigned int sequence = blockIdx.y / tiles_per_sequence;
  const unsigned int tile = blockIdx.y % tiles_per_sequence;
  if (head >= num_heads || sequence >= num_sequences) return;

  const unsigned int sequence_start =
      static_cast<unsigned int>(cu_seqlens[sequence]);
  const unsigned int sequence_end =
      static_cast<unsigned int>(cu_seqlens[sequence + 1]);
  const unsigned int token = sequence_start + tile * QUERY_TILE + warp;
  const bool active = token < sequence_end && token < total_tokens &&
                      valid[token] != 0;
  const unsigned long long state_base =
      static_cast<unsigned long long>(sequence) * 5;
  const auto* latent = reinterpret_cast<const __nv_bfloat16*>(
      sequence_state_ptrs[state_base]);
  const auto* metadata = reinterpret_cast<const int*>(
      sequence_state_ptrs[state_base + 4]);

  const int query_position = active ? positions[token] : -1;
  const bool within_causal_tile =
      query_position >= metadata[0] &&
      static_cast<unsigned int>(query_position - metadata[0]) < CAUSAL_WIDTH;
  const unsigned long long query_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_DIM;
  const unsigned int channel_base = lane * (MLA_DIM / 32);
  float query_values[MLA_DIM / 32];
  float accumulator[MLA_DIM / 32];
#pragma unroll
  for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
    query_values[part] = active
        ? __bfloat162float(absorbed_query[query_base + channel_base + part])
        : 0.0f;
    accumulator[part] = 0.0f;
  }

  __shared__ int maximum_position;
  __shared__ __nv_bfloat16 latent_row[MLA_DIM];
  if (tid == 0) maximum_position = metadata[0] - 1;
  __syncthreads();
  if (lane == 0 && active) atomicMax(&maximum_position, query_position);
  __syncthreads();

  float maximum = -FLT_MAX;
  float denominator = 0.0f;
  for (int key_position = metadata[0]; key_position <= maximum_position;
       ++key_position) {
    const unsigned long long latent_base =
        static_cast<unsigned long long>(key_position) * MLA_DIM;
    for (unsigned int channel = tid; channel < MLA_DIM;
         channel += blockDim.x) {
      latent_row[channel] = latent[latent_base + channel];
    }
    __syncthreads();
    if (active && within_causal_tile && key_position <= query_position) {
      float dot = 0.0f;
#pragma unroll
      for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
        dot += query_values[part] *
               __bfloat162float(latent_row[channel_base + part]);
      }
      const float score =
          __shfl_sync(0xffffffff, warp_sum(dot), 0) * attention_scale;
      const float next_maximum = fmaxf(maximum, score);
      const float old_scale = expf(maximum - next_maximum);
      const float new_scale = expf(score - next_maximum);
      denominator = denominator * old_scale + new_scale;
#pragma unroll
      for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
        accumulator[part] = accumulator[part] * old_scale +
            new_scale * __bfloat162float(latent_row[channel_base + part]);
      }
      maximum = next_maximum;
    }
    __syncthreads();
  }

  if (active && within_causal_tile) {
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      output[query_base + channel_base + part] =
          __float2bfloat16_rn(accumulator[part] * inverse);
    }
  }
}

// General sparse fallback after the learned index begins pruning pools. Each
// warp owns one (query, head), retaining that query's independently ranked
// positions while eliminating the old 16-KiB cross-warp output merge.
extern "C" __global__ void glm53_dsa_sparse_mla_prefill_warp(
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
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int token = blockIdx.y * QUERY_TILE + warp;
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
  float query_values[MLA_DIM / 32];
  float accumulator[MLA_DIM / 32];
#pragma unroll
  for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
    query_values[part] =
        __bfloat162float(absorbed_query[query_base + channel_base + part]);
    accumulator[part] = 0.0f;
  }

  float maximum = -FLT_MAX;
  float denominator = 0.0f;
  for (unsigned int rank = 0; rank < OUTPUT_WIDTH; ++rank) {
    const int position = selected[rank];
    if (position < 0) break;
    const unsigned long long latent_base =
        static_cast<unsigned long long>(position) * MLA_DIM + channel_base;
    float dot = 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
      dot += query_values[part] *
             __bfloat162float(latent[latent_base + part]);
    }
    const float score =
        __shfl_sync(0xffffffff, warp_sum(dot), 0) * attention_scale;
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

  const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
  for (unsigned int part = 0; part < MLA_DIM / 32; ++part) {
    output[query_base + channel_base + part] =
        __float2bfloat16_rn(accumulator[part] * inverse);
  }
}
