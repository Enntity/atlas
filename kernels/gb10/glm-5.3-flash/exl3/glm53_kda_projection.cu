// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

// This is deliberately an appliance kernel, not a shape-generic fallback.
// The model loader rejects anything except the official GLM TP2 geometry.
constexpr unsigned int KDA_LOCAL_WIDTH = 4096;
constexpr unsigned int KDA_LOCAL_HEADS = 32;
constexpr unsigned int KDA_LOW_RANK = 128;
constexpr unsigned int KDA_MERGED_WIDTH =
    3 * KDA_LOCAL_WIDTH + KDA_LOCAL_HEADS + 2 * KDA_LOW_RANK;

}  // namespace

// The merged GEMM emits token-major [q,k,v,beta,forget_a,gate_a]. Existing
// recurrence kernels consume matrix-major row sets. One block performs the
// bandwidth-only split for a token while the next projection can be queued.
extern "C" __global__ void glm53_kda_split_merged_bf16(
    const __nv_bfloat16* __restrict__ merged,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ forget_a,
    __nv_bfloat16* __restrict__ gate_a,
    unsigned int rows) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;

  const unsigned long long source =
      static_cast<unsigned long long>(row) * KDA_MERGED_WIDTH;
  const unsigned long long wide_destination =
      static_cast<unsigned long long>(row) * KDA_LOCAL_WIDTH;
  for (unsigned int channel = threadIdx.x; channel < KDA_LOCAL_WIDTH;
       channel += blockDim.x) {
    query[wide_destination + channel] = merged[source + channel];
    key[wide_destination + channel] =
        merged[source + KDA_LOCAL_WIDTH + channel];
    value[wide_destination + channel] =
        merged[source + 2 * KDA_LOCAL_WIDTH + channel];
  }
  if (threadIdx.x < KDA_LOCAL_HEADS) {
    beta[static_cast<unsigned long long>(row) * KDA_LOCAL_HEADS + threadIdx.x] =
        merged[source + 3 * KDA_LOCAL_WIDTH + threadIdx.x];
  }
  if (threadIdx.x < KDA_LOW_RANK) {
    const unsigned long long destination =
        static_cast<unsigned long long>(row) * KDA_LOW_RANK + threadIdx.x;
    forget_a[destination] =
        merged[source + 3 * KDA_LOCAL_WIDTH + KDA_LOCAL_HEADS + threadIdx.x];
    gate_a[destination] =
        merged[source + 3 * KDA_LOCAL_WIDTH + KDA_LOCAL_HEADS + KDA_LOW_RANK +
               threadIdx.x];
  }
}
