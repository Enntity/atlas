// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr unsigned int KDA_DIM = 128;
constexpr unsigned int KDA_STATE_SIZE = KDA_DIM * KDA_DIM;
constexpr unsigned int KDA_CONV_HISTORY = 3;

__device__ __forceinline__ float glm53_sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float glm53_bf16_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float glm53_conv_silu(
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
  return glm53_bf16_round(conv * glm53_sigmoid(conv));
}

}  // namespace

// One block owns one (sequence, head) pair. Q/K/V are the post-convolution
// BF16 activations. log_decay and beta are the FP32 outputs of GLM's forget
// and input gates, respectively. State remains FP32 across tokens.
//
// The output is BF16 because the pinned Transformers reference casts the
// recurrent result back to the Q dtype before the gated RMSNorm.
extern "C" __global__ __launch_bounds__(256, 1) void glm53_kda_decode(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_heads) {
  const unsigned int head = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (head >= batch_heads) return;

  __shared__ float q_norm[KDA_DIM];
  __shared__ float k_norm[KDA_DIM];
  __shared__ float reduce_q[KDA_DIM];
  __shared__ float reduce_k[KDA_DIM];
  __shared__ float delta[KDA_DIM];
  __shared__ float qk_products[KDA_DIM];

  const unsigned long long vector_base =
      static_cast<unsigned long long>(head) * KDA_DIM;
  const unsigned long long state_base =
      static_cast<unsigned long long>(head) * KDA_STATE_SIZE;

  if (tid < KDA_DIM) {
    const float q = __bfloat162float(query[vector_base + tid]);
    const float k = __bfloat162float(key[vector_base + tid]);
    q_norm[tid] = q;
    k_norm[tid] = k;
    reduce_q[tid] = q * q;
    reduce_k[tid] = k * k;
  }
  __syncthreads();

  for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
    if (tid < stride) {
      reduce_q[tid] += reduce_q[tid + stride];
      reduce_k[tid] += reduce_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid < KDA_DIM) {
    const float q_divisor = sqrtf(reduce_q[0] + 1.0e-6f);
    const float k_divisor = sqrtf(reduce_k[0] + 1.0e-6f);
    q_norm[tid] = q_norm[tid] / q_divisor *
                  0.08838834764831843f;  // 1 / sqrt(128)
    k_norm[tid] /= k_divisor;
    qk_products[tid] = q_norm[tid] * k_norm[tid];
  }
  __syncthreads();

  for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
    if (tid < stride) qk_products[tid] += qk_products[tid + stride];
    __syncthreads();
  }

  float q_state = 0.0f;
  if (tid < KDA_DIM) {
    float memory = 0.0f;
    for (unsigned int k = 0; k < KDA_DIM; ++k) {
      const unsigned long long index = state_base + tid * KDA_DIM + k;
      const float decayed = state[index] * expf(log_decay[vector_base + k]);
      state[index] = decayed;
      memory += k_norm[k] * decayed;
      q_state += q_norm[k] * decayed;
    }
    const float v = __bfloat162float(value[vector_base + tid]);
    delta[tid] = beta[head] * (v - memory);
  }
  __syncthreads();

  for (unsigned int index = tid; index < KDA_STATE_SIZE;
       index += blockDim.x) {
    const unsigned int v = index / KDA_DIM;
    const unsigned int k = index % KDA_DIM;
    state[state_base + index] += k_norm[k] * delta[v];
  }

  if (tid < KDA_DIM) {
    const float result = q_state + qk_products[0] * delta[tid];
    output[vector_base + tid] = __float2bfloat16_rn(result);
  }
}

// Complete post-projection GLM KDA decode operator. One block owns one
// (sequence, head) pair. Model weights are indexed by `head % num_heads`, so a
// contiguous state pool can batch sequences without duplicating weights.
//
// This intentionally preserves the pinned BF16 boundaries: convolution/SiLU,
// beta, and recurrent output are rounded to BF16 at the same points as the
// Transformers reference before subsequent FP32 math.
extern "C" __global__ __launch_bounds__(256, 1) void
glm53_kda_decode_fused_conv_gate_norm(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_conv_weight,
    const __nv_bfloat16* __restrict__ key_conv_weight,
    const __nv_bfloat16* __restrict__ value_conv_weight,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    const __nv_bfloat16* __restrict__ forget_projection,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ beta_logit,
    const __nv_bfloat16* __restrict__ output_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_heads,
    unsigned int num_heads,
    float norm_epsilon) {
  const unsigned int head = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (head >= batch_heads || num_heads == 0) return;

  __shared__ float q_norm[KDA_DIM];
  __shared__ float k_norm[KDA_DIM];
  __shared__ float reduce_q[KDA_DIM];
  __shared__ float reduce_k[KDA_DIM];
  __shared__ float decay[KDA_DIM];
  __shared__ float delta[KDA_DIM];
  __shared__ float qk_products[KDA_DIM];
  __shared__ float beta_value;

  const unsigned int model_head = head % num_heads;
  const unsigned int sequence = head / num_heads;
  const unsigned long long vector_base =
      static_cast<unsigned long long>(head) * KDA_DIM;
  const unsigned long long model_vector_base =
      static_cast<unsigned long long>(model_head) * KDA_DIM;
  const unsigned long long model_conv_base = model_vector_base * 4;
  const unsigned long long sequence_ptr_base =
      static_cast<unsigned long long>(sequence) * 4;
  float* const state = reinterpret_cast<float*>(
      sequence_state_ptrs[sequence_ptr_base]);
  float* const query_conv_state = reinterpret_cast<float*>(
      sequence_state_ptrs[sequence_ptr_base + 1]);
  float* const key_conv_state = reinterpret_cast<float*>(
      sequence_state_ptrs[sequence_ptr_base + 2]);
  float* const value_conv_state = reinterpret_cast<float*>(
      sequence_state_ptrs[sequence_ptr_base + 3]);
  const unsigned long long conv_state_base =
      static_cast<unsigned long long>(model_head) * KDA_DIM *
      KDA_CONV_HISTORY;
  const unsigned long long state_base =
      static_cast<unsigned long long>(model_head) * KDA_STATE_SIZE;

  if (tid < KDA_DIM) {
    const unsigned long long channel = vector_base + tid;
    const unsigned long long model_channel = model_vector_base + tid;
    const unsigned long long history = conv_state_base + tid * KDA_CONV_HISTORY;
    const unsigned long long weight = model_conv_base + tid * 4;
    const float q = glm53_conv_silu(query_conv_state + history,
                                    query_conv_weight + weight, query[channel]);
    const float k = glm53_conv_silu(key_conv_state + history,
                                    key_conv_weight + weight, key[channel]);
    const float v = glm53_conv_silu(value_conv_state + history,
                                    value_conv_weight + weight, value[channel]);
    q_norm[tid] = q;
    k_norm[tid] = k;
    delta[tid] = v;
    reduce_q[tid] = q * q;
    reduce_k[tid] = k * k;

    const float forget = __bfloat162float(forget_projection[channel]);
    const float rate = expf(a_log[model_head]);
    const float log_decay =
        -5.0f * glm53_sigmoid(rate * (forget + dt_bias[model_channel]));
    decay[tid] = expf(log_decay);
  }
  if (tid == 0) {
    beta_value = glm53_bf16_round(
        glm53_sigmoid(__bfloat162float(beta_logit[head])));
  }
  __syncthreads();

  for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
    if (tid < stride) {
      reduce_q[tid] += reduce_q[tid + stride];
      reduce_k[tid] += reduce_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid < KDA_DIM) {
    const float q_divisor = sqrtf(reduce_q[0] + 1.0e-6f);
    const float k_divisor = sqrtf(reduce_k[0] + 1.0e-6f);
    q_norm[tid] = q_norm[tid] / q_divisor * 0.08838834764831843f;
    k_norm[tid] /= k_divisor;
    qk_products[tid] = q_norm[tid] * k_norm[tid];
  }
  __syncthreads();

  for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
    if (tid < stride) qk_products[tid] += qk_products[tid + stride];
    __syncthreads();
  }

  float q_state = 0.0f;
  if (tid < KDA_DIM) {
    float memory = 0.0f;
    for (unsigned int k = 0; k < KDA_DIM; ++k) {
      const unsigned long long index = state_base + tid * KDA_DIM + k;
      const float decayed = state[index] * decay[k];
      state[index] = decayed;
      memory += k_norm[k] * decayed;
      q_state += q_norm[k] * decayed;
    }
    delta[tid] = beta_value * (delta[tid] - memory);
  }
  __syncthreads();

  for (unsigned int index = tid; index < KDA_STATE_SIZE;
       index += blockDim.x) {
    const unsigned int v = index / KDA_DIM;
    const unsigned int k = index % KDA_DIM;
    state[state_base + index] += k_norm[k] * delta[v];
  }

  if (tid < KDA_DIM) {
    const float core =
        glm53_bf16_round(q_state + qk_products[0] * delta[tid]);
    q_norm[tid] = core;
    reduce_q[tid] = core * core;
  }
  __syncthreads();

  for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
    if (tid < stride) reduce_q[tid] += reduce_q[tid + stride];
    __syncthreads();
  }

  if (tid < KDA_DIM) {
    const float rms = rsqrtf(reduce_q[0] / KDA_DIM + norm_epsilon);
    const float gate =
        glm53_sigmoid(__bfloat162float(output_gate[vector_base + tid]));
    const float weight = __bfloat162float(norm_weight[tid]);
    output[vector_base + tid] =
        __float2bfloat16_rn(q_norm[tid] * rms * weight * gate);
  }
}

// Target verification for consecutive rows of one logical sequence. Dense
// projections arrive as one [K, hidden] batch, but a single block owns each
// model head and advances its recurrent/conv state through K in causal order.
// This preserves exact recurrence without re-reading every layer weight K
// times. After each rollback-reachable row the block writes its slice into a
// fixed-address state image; state_images is u64[1 + snapshot_count][4].
extern "C" __global__ __launch_bounds__(256, 1) void
glm53_kda_verify_fused_conv_gate_norm(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ query_conv_weight,
    const __nv_bfloat16* __restrict__ key_conv_weight,
    const __nv_bfloat16* __restrict__ value_conv_weight,
    const unsigned long long* __restrict__ state_images,
    const __nv_bfloat16* __restrict__ forget_projection,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ beta_logit,
    const __nv_bfloat16* __restrict__ output_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int num_sequences,
    unsigned int snapshot_count,
    unsigned int num_heads,
    float norm_epsilon) {
  const unsigned int sequence = blockIdx.x / num_heads;
  const unsigned int model_head = blockIdx.x % num_heads;
  const unsigned int tid = threadIdx.x;
  if (sequence >= num_sequences || model_head >= num_heads || num_tokens == 0) return;

  __shared__ float q_norm[KDA_DIM];
  __shared__ float k_norm[KDA_DIM];
  __shared__ float reduce_q[KDA_DIM];
  __shared__ float reduce_k[KDA_DIM];
  __shared__ float decay[KDA_DIM];
  __shared__ float delta[KDA_DIM];
  __shared__ float qk_products[KDA_DIM];
  __shared__ float beta_value;

  const unsigned long long image_base =
      static_cast<unsigned long long>(sequence) * (snapshot_count + 1) * 4;
  float* const state = reinterpret_cast<float*>(state_images[image_base]);
  float* const query_conv_state = reinterpret_cast<float*>(state_images[image_base + 1]);
  float* const key_conv_state = reinterpret_cast<float*>(state_images[image_base + 2]);
  float* const value_conv_state = reinterpret_cast<float*>(state_images[image_base + 3]);
  const unsigned long long model_vector_base =
      static_cast<unsigned long long>(model_head) * KDA_DIM;
  const unsigned long long model_conv_base = model_vector_base * 4;
  const unsigned long long conv_state_base =
      static_cast<unsigned long long>(model_head) * KDA_DIM *
      KDA_CONV_HISTORY;
  const unsigned long long state_base =
      static_cast<unsigned long long>(model_head) * KDA_STATE_SIZE;

  for (unsigned int token = 0; token < num_tokens; ++token) {
    const unsigned long long vector_base =
        ((static_cast<unsigned long long>(sequence) * num_tokens + token) * num_heads + model_head) *
        KDA_DIM;
    if (tid < KDA_DIM) {
      const unsigned long long channel = vector_base + tid;
      const unsigned long long model_channel = model_vector_base + tid;
      const unsigned long long history =
          conv_state_base + tid * KDA_CONV_HISTORY;
      const unsigned long long weight = model_conv_base + tid * 4;
      const float q =
          glm53_conv_silu(query_conv_state + history,
                          query_conv_weight + weight, query[channel]);
      const float k =
          glm53_conv_silu(key_conv_state + history,
                          key_conv_weight + weight, key[channel]);
      const float v =
          glm53_conv_silu(value_conv_state + history,
                          value_conv_weight + weight, value[channel]);
      q_norm[tid] = q;
      k_norm[tid] = k;
      delta[tid] = v;
      reduce_q[tid] = q * q;
      reduce_k[tid] = k * k;

      const float forget = __bfloat162float(forget_projection[channel]);
      const float rate = expf(a_log[model_head]);
      const float log_decay =
          -5.0f * glm53_sigmoid(rate * (forget + dt_bias[model_channel]));
      decay[tid] = expf(log_decay);
    }
    if (tid == 0) {
      beta_value = glm53_bf16_round(
          glm53_sigmoid(__bfloat162float(
              beta_logit[(static_cast<unsigned long long>(sequence) *
                              num_tokens +
                          token) *
                             num_heads +
                         model_head])));
    }
    __syncthreads();

    for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
      if (tid < stride) {
        reduce_q[tid] += reduce_q[tid + stride];
        reduce_k[tid] += reduce_k[tid + stride];
      }
      __syncthreads();
    }

    if (tid < KDA_DIM) {
      const float q_divisor = sqrtf(reduce_q[0] + 1.0e-6f);
      const float k_divisor = sqrtf(reduce_k[0] + 1.0e-6f);
      q_norm[tid] = q_norm[tid] / q_divisor * 0.08838834764831843f;
      k_norm[tid] /= k_divisor;
      qk_products[tid] = q_norm[tid] * k_norm[tid];
    }
    __syncthreads();

    for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
      if (tid < stride) qk_products[tid] += qk_products[tid + stride];
      __syncthreads();
    }

    float q_state = 0.0f;
    if (tid < KDA_DIM) {
      float memory = 0.0f;
      for (unsigned int k = 0; k < KDA_DIM; ++k) {
        const unsigned long long index = state_base + tid * KDA_DIM + k;
        const float decayed = state[index] * decay[k];
        state[index] = decayed;
        memory += k_norm[k] * decayed;
        q_state += q_norm[k] * decayed;
      }
      delta[tid] = beta_value * (delta[tid] - memory);
    }
    __syncthreads();

    for (unsigned int index = tid; index < KDA_STATE_SIZE;
         index += blockDim.x) {
      const unsigned int v = index / KDA_DIM;
      const unsigned int k = index % KDA_DIM;
      state[state_base + index] += k_norm[k] * delta[v];
    }

    if (tid < KDA_DIM) {
      const float core =
          glm53_bf16_round(q_state + qk_products[0] * delta[tid]);
      q_norm[tid] = core;
      reduce_q[tid] = core * core;
    }
    __syncthreads();

    for (unsigned int stride = KDA_DIM / 2; stride != 0; stride >>= 1) {
      if (tid < stride) reduce_q[tid] += reduce_q[tid + stride];
      __syncthreads();
    }

    if (tid < KDA_DIM) {
      const float rms = rsqrtf(reduce_q[0] / KDA_DIM + norm_epsilon);
      const float gate =
          glm53_sigmoid(__bfloat162float(output_gate[vector_base + tid]));
      const float weight = __bfloat162float(norm_weight[tid]);
      output[vector_base + tid] =
          __float2bfloat16_rn(q_norm[tid] * rms * weight * gate);
    }
    __syncthreads();

    if (token < snapshot_count) {
      const unsigned long long image =
          image_base + static_cast<unsigned long long>(token + 1) * 4;
      auto* snapshot_state =
          reinterpret_cast<float*>(state_images[image]);
      auto* snapshot_q =
          reinterpret_cast<float*>(state_images[image + 1]);
      auto* snapshot_k =
          reinterpret_cast<float*>(state_images[image + 2]);
      auto* snapshot_v =
          reinterpret_cast<float*>(state_images[image + 3]);
      for (unsigned int index = tid; index < KDA_STATE_SIZE;
           index += blockDim.x) {
        snapshot_state[state_base + index] = state[state_base + index];
      }
      for (unsigned int index = tid;
           index < KDA_DIM * KDA_CONV_HISTORY; index += blockDim.x) {
        snapshot_q[conv_state_base + index] =
            query_conv_state[conv_state_base + index];
        snapshot_k[conv_state_base + index] =
            key_conv_state[conv_state_base + index];
        snapshot_v[conv_state_base + index] =
            value_conv_state[conv_state_base + index];
      }
    }
    __syncthreads();
  }
}
