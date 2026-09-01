// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr unsigned int INDEX_DIM = 128;
constexpr unsigned int MLA_QK_DIM = 256;
constexpr unsigned int MLA_V_DIM = 256;
constexpr unsigned int MLA_LATENT = 512;
constexpr unsigned int MLA_ROWS_PER_HEAD = MLA_QK_DIM + MLA_V_DIM;
constexpr unsigned int FP8_BLOCK = 128;
constexpr unsigned int FP8_SCALE_COLS = MLA_LATENT / FP8_BLOCK;

__device__ __forceinline__ float fp8_to_float(unsigned char bits) {
  __nv_fp8_e4m3 value;
  *reinterpret_cast<unsigned char*>(&value) = bits;
  return static_cast<float>(value);
}

}  // namespace

// Copy normalized MLA latents into fragmented per-sequence persistent slots.
// Each token is one block; cu_seqlens maps its packed row to a logical
// sequence, and positions supplies the absolute cache row inside that slot.
extern "C" __global__ void glm53_dsa_latent_append(
    const __nv_bfloat16* __restrict__ latent,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ positions,
    const unsigned char* __restrict__ valid,
    const unsigned long long* __restrict__ sequence_state_ptrs,
    unsigned int total_tokens,
    unsigned int num_sequences,
    unsigned int latent_capacity) {
  const unsigned int token = blockIdx.x;
  if (token >= total_tokens || valid[token] == 0) return;

  unsigned int sequence = 0;
  while (sequence + 1 < num_sequences &&
         token >= static_cast<unsigned int>(cu_seqlens[sequence + 1])) {
    ++sequence;
  }
  const int position = positions[token];
  if (position < 0 || static_cast<unsigned int>(position) >= latent_capacity) {
    return;
  }
  auto* destination = reinterpret_cast<__nv_bfloat16*>(
      sequence_state_ptrs[static_cast<unsigned long long>(sequence) * 5]);
  const unsigned long long source_base =
      static_cast<unsigned long long>(token) * MLA_LATENT;
  const unsigned long long destination_base =
      static_cast<unsigned long long>(position) * MLA_LATENT;
  for (unsigned int channel = threadIdx.x; channel < MLA_LATENT;
       channel += blockDim.x) {
    destination[destination_base + channel] = latent[source_base + channel];
  }
}

// PyTorch-compatible affine LayerNorm for the indexer's 128-wide key path.
// Input/output are BF16, reductions and affine math are FP32.
extern "C" __global__ void glm53_dsa_index_layernorm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    float epsilon) {
  const unsigned int row = blockIdx.x;
  const unsigned int channel = threadIdx.x;
  if (row >= rows || channel >= INDEX_DIM) return;

  __shared__ float values[INDEX_DIM];
  __shared__ float reduce[INDEX_DIM];
  const unsigned long long index =
      static_cast<unsigned long long>(row) * INDEX_DIM + channel;
  const float value = __bfloat162float(input[index]);
  values[channel] = value;
  reduce[channel] = value;
  __syncthreads();
  for (unsigned int stride = INDEX_DIM / 2; stride != 0; stride >>= 1) {
    if (channel < stride) reduce[channel] += reduce[channel + stride];
    __syncthreads();
  }
  const float mean = reduce[0] / INDEX_DIM;
  reduce[channel] = (value - mean) * (value - mean);
  __syncthreads();
  for (unsigned int stride = INDEX_DIM / 2; stride != 0; stride >>= 1) {
    if (channel < stride) reduce[channel] += reduce[channel + stride];
    __syncthreads();
  }
  const float inverse = rsqrtf(reduce[0] / INDEX_DIM + epsilon);
  const float result = (values[channel] - mean) * inverse *
                           __bfloat162float(weight[channel]) +
                       __bfloat162float(bias[channel]);
  output[index] = __float2bfloat16_rn(result);
}

// Absorb the per-head 256-wide query into the normalized 512-wide MLA latent
// space: q_abs[h] = q[h] @ W_K[h]. W_K is the first 256 rows of each
// head-major 512-row slice of checkpoint kv_b_proj.
extern "C" __global__ void glm53_dsa_absorb_query_fp8(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ kv_b_weight,
    const float* __restrict__ kv_b_scale,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int num_heads) {
  const unsigned int head = blockIdx.x;
  const unsigned int token = blockIdx.y;
  const unsigned int lane = threadIdx.x;
  if (head >= num_heads || token >= num_tokens) return;

  const unsigned int weight_row_base = head * MLA_ROWS_PER_HEAD;
  const unsigned long long query_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_QK_DIM;
  const unsigned long long output_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_LATENT;
  for (unsigned int column = lane; column < MLA_LATENT;
       column += blockDim.x) {
    float sum = 0.0f;
    for (unsigned int row = 0; row < MLA_QK_DIM; ++row) {
      const unsigned int weight_row = weight_row_base + row;
      const unsigned long long weight_index =
          static_cast<unsigned long long>(weight_row) * MLA_LATENT + column;
      const unsigned long long scale_index =
          static_cast<unsigned long long>(weight_row / FP8_BLOCK) *
              FP8_SCALE_COLS +
          column / FP8_BLOCK;
      sum += __bfloat162float(query[query_base + row]) *
             fp8_to_float(kv_b_weight[weight_index]) * kv_b_scale[scale_index];
    }
    output[output_base + column] = __float2bfloat16_rn(sum);
  }
}

// EXL3 checkpoints quantize only routed experts. Their DSA kv_b projection
// remains BF16, so it must not be reinterpreted as an FP8 byte matrix.
extern "C" __global__ void glm53_dsa_absorb_query_bf16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ kv_b_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int num_heads) {
  const unsigned int head = blockIdx.x;
  const unsigned int token = blockIdx.y;
  const unsigned int lane = threadIdx.x;
  if (head >= num_heads || token >= num_tokens) return;

  const unsigned int weight_row_base = head * MLA_ROWS_PER_HEAD;
  const unsigned long long query_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_QK_DIM;
  const unsigned long long output_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_LATENT;
  for (unsigned int column = lane; column < MLA_LATENT;
       column += blockDim.x) {
    float sum = 0.0f;
    for (unsigned int row = 0; row < MLA_QK_DIM; ++row) {
      const unsigned long long weight_index =
          static_cast<unsigned long long>(weight_row_base + row) * MLA_LATENT +
          column;
      sum += __bfloat162float(query[query_base + row]) *
             __bfloat162float(kv_b_weight[weight_index]);
    }
    output[output_base + column] = __float2bfloat16_rn(sum);
  }
}

// Expand the sparse-attention latent result through the value half of
// kv_b_proj. The value rows are the second 256 rows of each head-major slice.
extern "C" __global__ void glm53_dsa_expand_value_fp8(
    const __nv_bfloat16* __restrict__ latent,
    const unsigned char* __restrict__ kv_b_weight,
    const float* __restrict__ kv_b_scale,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int num_heads) {
  const unsigned int head = blockIdx.x;
  const unsigned int token = blockIdx.y;
  const unsigned int channel = threadIdx.x;
  if (head >= num_heads || token >= num_tokens || channel >= MLA_V_DIM) return;

  const unsigned int weight_row =
      head * MLA_ROWS_PER_HEAD + MLA_QK_DIM + channel;
  const unsigned long long latent_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_LATENT;
  float sum = 0.0f;
  for (unsigned int column = 0; column < MLA_LATENT; ++column) {
    const unsigned long long weight_index =
        static_cast<unsigned long long>(weight_row) * MLA_LATENT + column;
    const unsigned long long scale_index =
        static_cast<unsigned long long>(weight_row / FP8_BLOCK) *
            FP8_SCALE_COLS +
        column / FP8_BLOCK;
    sum += __bfloat162float(latent[latent_base + column]) *
           fp8_to_float(kv_b_weight[weight_index]) * kv_b_scale[scale_index];
  }
  const unsigned long long output_index =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_V_DIM +
      channel;
  output[output_index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void glm53_dsa_expand_value_bf16(
    const __nv_bfloat16* __restrict__ latent,
    const __nv_bfloat16* __restrict__ kv_b_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int num_heads) {
  const unsigned int head = blockIdx.x;
  const unsigned int token = blockIdx.y;
  const unsigned int channel = threadIdx.x;
  if (head >= num_heads || token >= num_tokens || channel >= MLA_V_DIM) return;

  const unsigned int weight_row =
      head * MLA_ROWS_PER_HEAD + MLA_QK_DIM + channel;
  const unsigned long long latent_base =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_LATENT;
  float sum = 0.0f;
  for (unsigned int column = 0; column < MLA_LATENT; ++column) {
    const unsigned long long weight_index =
        static_cast<unsigned long long>(weight_row) * MLA_LATENT + column;
    sum += __bfloat162float(latent[latent_base + column]) *
           __bfloat162float(kv_b_weight[weight_index]);
  }
  const unsigned long long output_index =
      (static_cast<unsigned long long>(token) * num_heads + head) * MLA_V_DIM +
      channel;
  output[output_index] = __float2bfloat16_rn(sum);
}
