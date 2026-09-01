// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// GLM-5.3's final HyperHead is an unweighted arithmetic mean. mHC streams
// remain FP32 inside Atlas so the 45-layer highway does not quantize its
// residual on every block; the model head rounds once into BF16.
extern "C" __global__ void glm53_hc_mean(
    const float* __restrict__ streams,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int hc_mult) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long elements =
      static_cast<unsigned long long>(num_tokens) * hidden_size;
  if (index >= elements || hc_mult == 0) return;
  const unsigned int token = index / hidden_size;
  const unsigned int channel = index % hidden_size;
  const unsigned long long token_base =
      static_cast<unsigned long long>(token) * hc_mult * hidden_size;
  float sum = 0.0f;
  for (unsigned int stream = 0; stream < hc_mult; ++stream) {
    sum += streams[token_base +
                   static_cast<unsigned long long>(stream) * hidden_size +
                   channel];
  }
  output[index] = __float2bfloat16_rn(sum / static_cast<float>(hc_mult));
}
