// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr unsigned int VKDA_DIM = 128;
constexpr unsigned int VKDA_STATE_SIZE = VKDA_DIM * VKDA_DIM;
constexpr unsigned int VKDA_CONV_HISTORY = 3;
constexpr unsigned int VKDA_VALUE_TILE = 8;
constexpr unsigned int VKDA_VALUE_TILES = VKDA_DIM / VKDA_VALUE_TILE;

__device__ __forceinline__ float tiled_sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float tiled_bf16_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float tiled_conv_silu(
    float* __restrict__ history,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16 input) {
  const float current = __bfloat162float(input);
  const float conv = history[0] * __bfloat162float(weight[0]) +
                     history[1] * __bfloat162float(weight[1]) +
                     history[2] * __bfloat162float(weight[2]) +
                     current * __bfloat162float(weight[3]);
  history[0] = history[1];
  history[1] = history[2];
  history[2] = current;
  return tiled_bf16_round(conv * tiled_sigmoid(conv));
}

__device__ __forceinline__ float warp_sum(float value) {
  for (unsigned int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

}  // namespace

// Stage one advances the three short-convolution states and writes their BF16
// outputs back over the projection buffers. One block owns a sequence/head, so
// token order and snapshot images remain exact while the recurrent matrix is
// left to the much wider value-tiled stage below.
extern "C" __global__ __launch_bounds__(128, 2) void
glm53_kda_verify_prepare(
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_conv_weight,
    const __nv_bfloat16* __restrict__ key_conv_weight,
    const __nv_bfloat16* __restrict__ value_conv_weight,
    const unsigned long long* __restrict__ state_images,
    unsigned int num_tokens,
    unsigned int num_sequences,
    unsigned int snapshot_count,
    unsigned int num_heads) {
  const unsigned int sequence = blockIdx.x / num_heads;
  const unsigned int model_head = blockIdx.x % num_heads;
  const unsigned int channel = threadIdx.x;
  if (sequence >= num_sequences || model_head >= num_heads ||
      channel >= VKDA_DIM) {
    return;
  }

  const unsigned long long image_base =
      static_cast<unsigned long long>(sequence) * (snapshot_count + 1) * 4;
  float* const query_state =
      reinterpret_cast<float*>(state_images[image_base + 1]);
  float* const key_state =
      reinterpret_cast<float*>(state_images[image_base + 2]);
  float* const value_state =
      reinterpret_cast<float*>(state_images[image_base + 3]);
  const unsigned long long model_channel =
      static_cast<unsigned long long>(model_head) * VKDA_DIM + channel;
  const unsigned long long history = model_channel * VKDA_CONV_HISTORY;
  const unsigned long long weight = model_channel * 4;

  for (unsigned int token = 0; token < num_tokens; ++token) {
    const unsigned long long activation =
        ((static_cast<unsigned long long>(sequence) * num_tokens + token) *
             num_heads +
         model_head) *
            VKDA_DIM +
        channel;
    query[activation] = __float2bfloat16_rn(tiled_conv_silu(
        query_state + history, query_conv_weight + weight, query[activation]));
    key[activation] = __float2bfloat16_rn(tiled_conv_silu(
        key_state + history, key_conv_weight + weight, key[activation]));
    value[activation] = __float2bfloat16_rn(tiled_conv_silu(
        value_state + history, value_conv_weight + weight, value[activation]));

    if (token < snapshot_count) {
      const unsigned long long image =
          image_base + static_cast<unsigned long long>(token + 1) * 4;
      auto* snapshot_query = reinterpret_cast<float*>(state_images[image + 1]);
      auto* snapshot_key = reinterpret_cast<float*>(state_images[image + 2]);
      auto* snapshot_value = reinterpret_cast<float*>(state_images[image + 3]);
      const unsigned long long state_channel = history;
      snapshot_query[state_channel] = query_state[state_channel];
      snapshot_query[state_channel + 1] = query_state[state_channel + 1];
      snapshot_query[state_channel + 2] = query_state[state_channel + 2];
      snapshot_key[state_channel] = key_state[state_channel];
      snapshot_key[state_channel + 1] = key_state[state_channel + 1];
      snapshot_key[state_channel + 2] = key_state[state_channel + 2];
      snapshot_value[state_channel] = value_state[state_channel];
      snapshot_value[state_channel + 1] = value_state[state_channel + 1];
      snapshot_value[state_channel + 2] = value_state[state_channel + 2];
    }
  }
}

// Stage two owns eight contiguous value rows of one sequence/head state matrix.
// All 128 threads traverse K contiguously, turning the old strided one-block
// matrix walk into sixteen independently schedulable, coalesced tiles. Tokens
// remain serial inside a tile, so speculative state images retain causal order.
extern "C" __global__ __launch_bounds__(128, 2) void
glm53_kda_verify_recurrent_tiled(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const unsigned long long* __restrict__ state_images,
    const __nv_bfloat16* __restrict__ forget_projection,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ beta_logit,
    __nv_bfloat16* __restrict__ core_output,
    unsigned int num_tokens,
    unsigned int num_sequences,
    unsigned int snapshot_count,
    unsigned int num_heads) {
  const unsigned int group = blockIdx.x / VKDA_VALUE_TILES;
  const unsigned int value_tile = blockIdx.x % VKDA_VALUE_TILES;
  const unsigned int sequence = group / num_heads;
  const unsigned int model_head = group % num_heads;
  const unsigned int key_index = threadIdx.x;
  if (sequence >= num_sequences || model_head >= num_heads ||
      key_index >= VKDA_DIM) {
    return;
  }

  __shared__ float warp_q_sq[4];
  __shared__ float warp_k_sq[4];
  __shared__ float warp_qk[4];
  __shared__ float warp_memory[4][VKDA_VALUE_TILE];
  __shared__ float warp_q_state[4][VKDA_VALUE_TILE];
  __shared__ float inv_q;
  __shared__ float inv_k;
  __shared__ float qk_dot;
  __shared__ float delta[VKDA_VALUE_TILE];
  __shared__ float core[VKDA_VALUE_TILE];

  const unsigned int lane = key_index & 31;
  const unsigned int warp = key_index >> 5;
  const unsigned int value_start = value_tile * VKDA_VALUE_TILE;
  const unsigned long long image_base =
      static_cast<unsigned long long>(sequence) * (snapshot_count + 1) * 4;
  float* const state = reinterpret_cast<float*>(state_images[image_base]);
  const unsigned long long state_head =
      static_cast<unsigned long long>(model_head) * VKDA_STATE_SIZE;
  const unsigned long long model_channel =
      static_cast<unsigned long long>(model_head) * VKDA_DIM + key_index;

  for (unsigned int token = 0; token < num_tokens; ++token) {
    const unsigned long long vector_base =
        ((static_cast<unsigned long long>(sequence) * num_tokens + token) *
             num_heads +
         model_head) *
        VKDA_DIM;
    const float q_raw = __bfloat162float(query[vector_base + key_index]);
    const float k_raw = __bfloat162float(key[vector_base + key_index]);
    float q_sq = warp_sum(q_raw * q_raw);
    float k_sq = warp_sum(k_raw * k_raw);
    if (lane == 0) {
      warp_q_sq[warp] = q_sq;
      warp_k_sq[warp] = k_sq;
    }
    __syncthreads();
    if (key_index == 0) {
      float q_total = 0.0f;
      float k_total = 0.0f;
#pragma unroll
      for (unsigned int item = 0; item < 4; ++item) {
        q_total += warp_q_sq[item];
        k_total += warp_k_sq[item];
      }
      inv_q = rsqrtf(q_total + 1.0e-6f) * 0.08838834764831843f;
      inv_k = rsqrtf(k_total + 1.0e-6f);
    }
    __syncthreads();

    const float q_norm = q_raw * inv_q;
    const float k_norm = k_raw * inv_k;
    float qk = warp_sum(q_norm * k_norm);
    if (lane == 0) warp_qk[warp] = qk;
    __syncthreads();
    if (key_index == 0) {
      qk_dot = warp_qk[0] + warp_qk[1] + warp_qk[2] + warp_qk[3];
    }
    __syncthreads();

    const float forget =
        __bfloat162float(forget_projection[vector_base + key_index]);
    const float log_decay =
        -5.0f * tiled_sigmoid(expf(a_log[model_head]) *
                              (forget + dt_bias[model_channel]));
    const float decay = expf(log_decay);
    float memory[VKDA_VALUE_TILE];
    float q_state[VKDA_VALUE_TILE];
#pragma unroll
    for (unsigned int item = 0; item < VKDA_VALUE_TILE; ++item) {
      const unsigned int value_index = value_start + item;
      const unsigned long long index =
          state_head + static_cast<unsigned long long>(value_index) * VKDA_DIM +
          key_index;
      const float decayed = state[index] * decay;
      state[index] = decayed;
      memory[item] = warp_sum(k_norm * decayed);
      q_state[item] = warp_sum(q_norm * decayed);
      if (lane == 0) {
        warp_memory[warp][item] = memory[item];
        warp_q_state[warp][item] = q_state[item];
      }
    }
    __syncthreads();

    if (key_index < VKDA_VALUE_TILE) {
      const unsigned int item = key_index;
      const unsigned int value_index = value_start + item;
      const float memory_total = warp_memory[0][item] + warp_memory[1][item] +
                                 warp_memory[2][item] + warp_memory[3][item];
      const float q_state_total =
          warp_q_state[0][item] + warp_q_state[1][item] +
          warp_q_state[2][item] + warp_q_state[3][item];
      const float beta = tiled_bf16_round(tiled_sigmoid(__bfloat162float(
          beta_logit[(static_cast<unsigned long long>(sequence) * num_tokens +
                      token) *
                         num_heads +
                     model_head])));
      const float v = __bfloat162float(value[vector_base + value_index]);
      delta[item] = beta * (v - memory_total);
      core[item] = tiled_bf16_round(q_state_total + qk_dot * delta[item]);
      core_output[vector_base + value_index] = __float2bfloat16_rn(core[item]);
    }
    __syncthreads();

#pragma unroll
    for (unsigned int item = 0; item < VKDA_VALUE_TILE; ++item) {
      const unsigned int value_index = value_start + item;
      const unsigned long long index =
          state_head + static_cast<unsigned long long>(value_index) * VKDA_DIM +
          key_index;
      state[index] += k_norm * delta[item];
      if (token < snapshot_count) {
        const unsigned long long image =
            image_base + static_cast<unsigned long long>(token + 1) * 4;
        auto* snapshot = reinterpret_cast<float*>(state_images[image]);
        snapshot[index] = state[index];
      }
    }
    __syncthreads();
  }
}

// Stage three restores the checkpoint's BF16 recurrent-output boundary, then
// performs the same gated RMSNorm as the original monolithic verifier.
extern "C" __global__ __launch_bounds__(128, 2) void glm53_kda_verify_norm(
    __nv_bfloat16* __restrict__ core_output,
    const __nv_bfloat16* __restrict__ output_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    unsigned int token_heads,
    float norm_epsilon) {
  const unsigned int token_head = blockIdx.x;
  const unsigned int channel = threadIdx.x;
  if (token_head >= token_heads || channel >= VKDA_DIM) return;

  __shared__ float warp_sum_sq[4];
  __shared__ float rms;
  const unsigned long long index =
      static_cast<unsigned long long>(token_head) * VKDA_DIM + channel;
  const float core = __bfloat162float(core_output[index]);
  float sum_sq = warp_sum(core * core);
  if ((channel & 31) == 0) warp_sum_sq[channel >> 5] = sum_sq;
  __syncthreads();
  if (channel == 0) {
    const float total =
        warp_sum_sq[0] + warp_sum_sq[1] + warp_sum_sq[2] + warp_sum_sq[3];
    rms = rsqrtf(total / VKDA_DIM + norm_epsilon);
  }
  __syncthreads();
  const float gate = tiled_sigmoid(__bfloat162float(output_gate[index]));
  core_output[index] = __float2bfloat16_rn(
      core * rms * __bfloat162float(norm_weight[channel]) * gate);
}
