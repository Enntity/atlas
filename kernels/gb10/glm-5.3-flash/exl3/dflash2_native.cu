// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

extern "C" __global__ void dflash2_dynamic_conv2_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* dynamic,
    const __nv_bfloat16* base, __nv_bfloat16* output,
    unsigned rows, unsigned sequence_rows, unsigned hidden, unsigned groups,
    unsigned stage) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * hidden) return;
  const unsigned row = idx / hidden;
  const unsigned local_row = row % sequence_rows;
  const unsigned channel = idx - row * hidden;
  const unsigned group = channel / (hidden / groups);
  float result = 0.0f;
#pragma unroll
  for (unsigned offset = 0; offset < 2; ++offset) {
    if (offset > local_row) continue;
    const unsigned dyn_idx = ((row * 2 + stage) * 2 + offset) * groups + group;
    const unsigned base_idx = (stage * 2 + offset) * hidden + channel;
    const float coefficient = __bfloat162float(base[base_idx])
                            + __bfloat162float(dynamic[dyn_idx]);
    result += coefficient * __bfloat162float(input[(row - offset) * hidden + channel]);
  }
  output[idx] = __float2bfloat16_rn(result);
}

extern "C" __global__ void dflash2_select_path16_bf16(
    const __nv_bfloat16* logits, const __nv_bfloat16* selector_hidden,
    const __nv_bfloat16* predecessor_codebook,
    const __nv_bfloat16* successor_codebook, unsigned* output,
    unsigned* candidate_ids, float* edge_scores,
    unsigned rows, unsigned vocab, unsigned rank, const unsigned* anchor_token) {
  constexpr unsigned TOPK = 16;
  constexpr unsigned THREADS = 256;
  __shared__ float candidates_value[THREADS * TOPK];
  __shared__ unsigned candidates_token[THREADS * TOPK];
  unsigned previous_index = 0;
  for (unsigned row = 0; row < rows; ++row) {
    float local_value[TOPK];
    unsigned local_token[TOPK];
#pragma unroll
    for (unsigned i = 0; i < TOPK; ++i) {
      local_value[i] = -FLT_MAX;
      local_token[i] = 0;
    }
    for (unsigned token = threadIdx.x; token < vocab; token += blockDim.x) {
      float value = __bfloat162float(logits[row * vocab + token]);
      if (!isfinite(value) || value <= local_value[TOPK - 1]) continue;
      unsigned insert = TOPK - 1;
      while (insert > 0 && value > local_value[insert - 1]) {
        local_value[insert] = local_value[insert - 1];
        local_token[insert] = local_token[insert - 1];
        --insert;
      }
      local_value[insert] = value;
      local_token[insert] = token;
    }
#pragma unroll
    for (unsigned i = 0; i < TOPK; ++i) {
      const unsigned at = threadIdx.x * TOPK + i;
      candidates_value[at] = local_value[i];
      candidates_token[at] = local_token[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned top_token[TOPK];
      for (unsigned selected = 0; selected < TOPK; ++selected) {
        unsigned best = 0;
        float best_value = -FLT_MAX;
        for (unsigned i = 0; i < THREADS * TOPK; ++i) {
          if (candidates_value[i] > best_value) {
            best_value = candidates_value[i];
            best = i;
          }
        }
        top_token[selected] = candidates_token[best];
        candidate_ids[row * TOPK + selected] = top_token[selected];
        candidates_value[best] = -FLT_MAX;
      }
    }
    __syncthreads();

    // Materialize the exact DFlash2 edge-score tensor [prev=16, curr=16].
    // Row 0 is anchored by the observed token, so every predecessor lane is
    // intentionally identical.  Later rows use the previous row's candidate
    // IDs.  The host reference sampler walks this tensor probabilistically
    // and retains the realized 16-way row for standard rejection sampling.
    const unsigned pair = threadIdx.x;
    if (pair < TOPK * TOPK) {
      const unsigned prev_idx = pair / TOPK;
      const unsigned curr_idx = pair % TOPK;
      const unsigned pred_token = row == 0
          ? *anchor_token
          : candidate_ids[(row - 1) * TOPK + prev_idx];
      const unsigned successor = candidate_ids[row * TOPK + curr_idx];
      float edge = 0.0f;
      for (unsigned r = 0; r < rank; ++r) {
        edge += __bfloat162float(predecessor_codebook[pred_token * rank + r])
              * __bfloat162float(selector_hidden[row * rank + r])
              * __bfloat162float(successor_codebook[successor * rank + r]);
      }
      edge_scores[(row * TOPK + prev_idx) * TOPK + curr_idx] =
          __bfloat162float(logits[row * vocab + successor]) + edge;
    }
    __syncthreads();

    // Preserve the old greedy token output for temperature-zero requests and
    // as a safe fallback if the host-side probabilistic walk is unavailable.
    if (threadIdx.x == 0) {
      unsigned best_token = candidate_ids[row * TOPK];
      unsigned best_index = 0;
      float best_score = -FLT_MAX;
      const unsigned prev_index = row == 0 ? 0 : previous_index;
      for (unsigned i = 0; i < TOPK; ++i) {
        const float score = edge_scores[(row * TOPK + prev_index) * TOPK + i];
        if (score > best_score) {
          best_score = score;
          best_token = candidate_ids[row * TOPK + i];
          best_index = i;
        }
      }
      output[row] = best_token;
      previous_index = best_index;
    }
    __syncthreads();
  }
}
