// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3 declares moe_router_dtype=float32. The gate GEMM therefore writes
// FP32 logits and this kernel keeps them FP32 through sigmoid, correction-bias
// selection, normalization, and scaling.

#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
constexpr unsigned int kThreads = 256;
constexpr unsigned int kMaxExperts = 288;
constexpr unsigned int kMaxTopK = 8;
constexpr unsigned int kWarps = kThreads / 32;
}  // namespace

extern "C" __global__ void glm53_moe_topk_sigmoid_batched_f32(
    const float* __restrict__ gate_logits,
    const float* __restrict__ bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor) {
  if (num_experts > kMaxExperts || top_k > kMaxTopK) return;

  __shared__ float sigmoid_scores[kMaxExperts];
  __shared__ float selection_scores[kMaxExperts];
  __shared__ float top_values[kMaxTopK];
  __shared__ unsigned int top_indices[kMaxTopK];
  __shared__ float warp_values[kWarps];
  __shared__ unsigned int warp_indices[kWarps];

  const unsigned int token = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp = tid / 32;
  const unsigned int lane = tid % 32;
  const float* token_logits = gate_logits + token * num_experts;

  for (unsigned int expert = tid; expert < num_experts; expert += kThreads) {
    const float score = 1.0f / (1.0f + __expf(-token_logits[expert]));
    sigmoid_scores[expert] = score;
    selection_scores[expert] = score + bias[expert];
  }
  __syncthreads();

  for (unsigned int rank = 0; rank < top_k; ++rank) {
    float local_value = -CUDART_INF_F;
    unsigned int local_index = 0;
    for (unsigned int expert = tid; expert < num_experts; expert += kThreads) {
      const float candidate = selection_scores[expert];
      if (candidate > local_value ||
          (candidate == local_value && expert < local_index)) {
        local_value = candidate;
        local_index = expert;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value =
          __shfl_down_sync(0xffffffff, local_value, offset);
      const unsigned int other_index =
          __shfl_down_sync(0xffffffff, local_index, offset);
      if (other_value > local_value ||
          (other_value == local_value && other_index < local_index)) {
        local_value = other_value;
        local_index = other_index;
      }
    }
    if (lane == 0) {
      warp_values[warp] = local_value;
      warp_indices[warp] = local_index;
    }
    __syncthreads();
    if (tid == 0) {
      float best_value = warp_values[0];
      unsigned int best_index = warp_indices[0];
      for (unsigned int candidate = 1; candidate < kWarps; ++candidate) {
        if (warp_values[candidate] > best_value ||
            (warp_values[candidate] == best_value &&
             warp_indices[candidate] < best_index)) {
          best_value = warp_values[candidate];
          best_index = warp_indices[candidate];
        }
      }
      top_indices[rank] = best_index;
      selection_scores[best_index] = -CUDART_INF_F;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float sum = 0.0f;
    for (unsigned int rank = 0; rank < top_k; ++rank) {
      top_values[rank] = sigmoid_scores[top_indices[rank]];
      sum += top_values[rank];
    }
    for (unsigned int rank = 0; rank < top_k; ++rank) {
      const float weight = normalize && sum > 1.0e-20f
                               ? top_values[rank] / sum
                               : top_values[rank];
      expert_indices[token * top_k + rank] = top_indices[rank];
      expert_weights[token * top_k + rank] = weight * scaling_factor;
    }
  }
}
