// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned int KDA_DIM = 128;
constexpr unsigned int CONV_HISTORY = 3;

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ __nv_bfloat16 conv_silu(
    float* __restrict__ history,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16 input) {
  const float current = __bfloat162float(input);
  const float convolved = history[0] * __bfloat162float(weight[0]) +
                          history[1] * __bfloat162float(weight[1]) +
                          history[2] * __bfloat162float(weight[2]) +
                          current * __bfloat162float(weight[3]);
  history[0] = history[1];
  history[1] = history[2];
  history[2] = current;
  return __float2bfloat16_rn(convolved * sigmoid(convolved));
}

}  // namespace

// Apply the three causal depthwise conv4 + SiLU projections in place across a
// packed varlen chunk. One block owns one logical (sequence,head), while the
// device pointer table resolves its fragmented persistent conv-state slot.
extern "C" __global__ void glm53_kda_conv_silu_chunk(
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_weight,
    const __nv_bfloat16* __restrict__ key_weight,
    const __nv_bfloat16* __restrict__ value_weight,
    const std::int64_t* __restrict__ cu_seqlens,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    unsigned int num_sequences,
    unsigned int num_heads) {
  const unsigned int sequence_head = blockIdx.x;
  const unsigned int channel = threadIdx.x;
  if (num_heads == 0 || channel >= KDA_DIM) return;
  const unsigned int sequence = sequence_head / num_heads;
  const unsigned int head = sequence_head % num_heads;
  if (sequence >= num_sequences) return;

  const unsigned long long pointer_base =
      static_cast<unsigned long long>(sequence) * 4;
  auto* query_history =
      reinterpret_cast<float*>(sequence_state_ptrs[pointer_base + 1]);
  auto* key_history =
      reinterpret_cast<float*>(sequence_state_ptrs[pointer_base + 2]);
  auto* value_history =
      reinterpret_cast<float*>(sequence_state_ptrs[pointer_base + 3]);
  const unsigned long long history =
      (static_cast<unsigned long long>(head) * KDA_DIM + channel) *
      CONV_HISTORY;
  const unsigned long long weight =
      (static_cast<unsigned long long>(head) * KDA_DIM + channel) * 4;

  for (std::int64_t token = cu_seqlens[sequence];
       token < cu_seqlens[sequence + 1]; ++token) {
    const unsigned long long element =
        (static_cast<unsigned long long>(token) * num_heads + head) * KDA_DIM +
        channel;
    query[element] = conv_silu(query_history + history,
                               query_weight + weight, query[element]);
    key[element] =
        conv_silu(key_history + history, key_weight + weight, key[element]);
    value[element] = conv_silu(value_history + history,
                               value_weight + weight, value[element]);
  }
}

// FlashKDA's beta TMA descriptor is contiguous [head,total_tokens], unlike all
// other packed activations. Beta remains a raw logit; FlashKDA fuses sigmoid.
extern "C" __global__ void glm53_kda_beta_transpose(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_tokens,
    unsigned int num_heads) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = total_tokens * num_heads;
  if (index >= elements || num_heads == 0) return;
  const unsigned int token = index / num_heads;
  const unsigned int head = index % num_heads;
  output[static_cast<unsigned long long>(head) * total_tokens + token] =
      input[index];
}

// Strict-FP32 gated RMSNorm after FlashKDA. One block owns one token/head and
// returns BF16 at the same boundary as the pinned Transformers implementation.
extern "C" __global__ void glm53_kda_gated_norm_chunk(
    const __nv_bfloat16* __restrict__ recurrent_output,
    const __nv_bfloat16* __restrict__ output_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int token_heads,
    float norm_epsilon) {
  const unsigned int token_head = blockIdx.x;
  const unsigned int channel = threadIdx.x;
  if (token_head >= token_heads || channel >= KDA_DIM) return;

  __shared__ float reduction[KDA_DIM];
  const unsigned long long element =
      static_cast<unsigned long long>(token_head) * KDA_DIM + channel;
  const float core = __bfloat162float(recurrent_output[element]);
  reduction[channel] = core * core;
  __syncthreads();
  for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
    if (channel < stride) reduction[channel] += reduction[channel + stride];
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(KDA_DIM) + norm_epsilon);
  const float gate = sigmoid(__bfloat162float(output_gate[element]));
  const float weight = __bfloat162float(norm_weight[channel]);
  output[element] = __float2bfloat16_rn(core * inverse_rms * weight * gate);
}
