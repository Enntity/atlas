// SPDX-License-Identifier: AGPL-3.0-only

// Atlas raw-pointer integration for the MIT-licensed ExLlamaV3 EXL3 kernel.
// The specialized kernel body is retained under exl3_vendor/ with its license.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MOE_TILESIZE_N 256
#include "exl3_vendor/quant/exl3_moe_kernel.cuh"

// Atlas activations are BF16; ExLlamaV3's Hadamard/GEMM pipeline consumes
// IEEE FP16. This conversion is explicit so no pointer is ever reinterpreted.
extern "C" __global__ void glm53_exl3_bf16_to_fp16(
    const __nv_bfloat16* __restrict__ input,
    half* __restrict__ output,
    unsigned int elements
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = __float2half(__bfloat162float(input[i]));
}

// Count and compact only this EP rank's routes. The fused EXL3 kernel expects
// int64 bincounts and token ids, whereas Atlas's generic MoE sorter uses i32
// prefix offsets. Producing the native ABI directly also ensures remote routes
// have zero counts and can never dereference null expert pointers.
//
// Grid: [1,1,1], block: [256,1,1]. Supports the GLM contract (288 experts).
extern "C" __global__ void glm53_exl3_prepare_routes(
    const unsigned int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int64_t* __restrict__ expert_count,
    int4* __restrict__ fat_descriptors,
    int64_t* __restrict__ token_sorted,
    half* __restrict__ weight_sorted,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int num_experts,
    unsigned int local_expert_start,
    unsigned int local_expert_end,
    unsigned int fat_cap
) {
    __shared__ unsigned int counts[288];
    __shared__ unsigned int offsets[289];
    __shared__ unsigned int cursors[288];

    if (num_experts != 288) return;
    for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        counts[e] = 0;
        cursors[e] = 0;
    }
    __syncthreads();

    const unsigned int routes = num_tokens * topk;
    for (unsigned int i = threadIdx.x; i < routes; i += blockDim.x) {
        const unsigned int e = topk_ids[i];
        if (e >= local_expert_start && e < local_expert_end) atomicAdd(&counts[e], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        offsets[0] = 0;
        for (unsigned int e = 0; e < num_experts; ++e) {
            expert_count[e] = static_cast<int64_t>(counts[e]);
            offsets[e + 1] = offsets[e] + counts[e];
        }
        unsigned int fat_count = 0;
        for (unsigned int e = 0; e < num_experts; ++e) {
            if (counts[e] > fat_cap && fat_count < 288) {
                fat_descriptors[fat_count++] = make_int4(
                    static_cast<int>(e),
                    static_cast<int>(offsets[e]),
                    static_cast<int>(counts[e]),
                    0);
            }
        }
        expert_count[num_experts] = static_cast<int64_t>(fat_count);
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < routes; i += blockDim.x) {
        const unsigned int e = topk_ids[i];
        if (e < local_expert_start || e >= local_expert_end) continue;
        const unsigned int pos = offsets[e] + atomicAdd(&cursors[e], 1);
        token_sorted[pos] = static_cast<int64_t>(i / topk);
        weight_sorted[pos] = __float2half(topk_weights[i]);
    }
}

extern "C" __global__ void glm53_exl3_fp32_to_bf16(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int elements
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = __float2bfloat16(input[i]);
}
