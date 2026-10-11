// SPDX-License-Identifier: AGPL-3.0-only

// Explicit packed routed-expert arithmetic. Thin/F32 fat selection depends on
// per-expert occupancy, never total request rows. Host validates all geometry,
// pointers, integer bitrates, positive thin_cap and the clamp policy.
#include <math.h>
#include "exl3_common.cuh"
#include "exl3_expert.cuh"

namespace {
__device__ __forceinline__ float half_round(float value) {
    return __half2float(__float2half_rn(value));
}
__device__ __forceinline__ bool valid_expert(int expert, unsigned int experts) {
    return expert >= 0 && static_cast<unsigned int>(expert) < experts;
}
__device__ __forceinline__ float thin_activation(float gate, float up, float limit, unsigned int clamp) {
    const __half2 g = __float2half2_rn(gate);
    const __half2 reciprocal = h2rcp(__hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(g))));
    __half2 activated = __hmul2(g, reciprocal);
    __half2 u = __float2half2_rn(up);
    if (clamp) {
        const __half2 high = __float2half2_rn(limit);
        u = __hmin2(__hmax2(u, __float2half2_rn(-limit)), high);
        activated = __hmin2(activated, high);
    }
    return __low2float(__hmul2(activated, u));
}
}

// Grid(ceil(experts/128),1,1), block(128,1,1). Fully overwrites counts.
extern "C" __global__ void exl3_moe_route_counts(
    const int* __restrict__ ids, unsigned int* __restrict__ counts,
    unsigned int slots, unsigned int experts) {
    const unsigned int expert = blockIdx.x * 128 + threadIdx.x;
    if (expert >= experts) return;
    unsigned int count = 0;
    for (unsigned int slot = 0; slot < slots; ++slot) count += ids[slot] == static_cast<int>(expert);
    counts[expert] = count;
}

// Same argument order and launch as legacy indirect input transform.
// expanded=0 for gate/up; expanded=1 accepts BF16 slot inputs if requested.
extern "C" __global__ void exl3_moe_input_hadamard_f16(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, float* __restrict__ xh,
    unsigned int rows, unsigned int k, unsigned int topk, unsigned int expanded, unsigned int experts) {
    const unsigned int slot = blockIdx.y, feature = blockIdx.x * 128 + threadIdx.x;
    if (slot >= static_cast<unsigned long long>(rows) * topk || feature >= k) return;
    const unsigned long long out_index = static_cast<unsigned long long>(slot) * k + feature;
    const int expert = ids[slot];
    if (!valid_expert(expert, experts)) { xh[out_index] = __int_as_float(0x7fc00000); return; }
    const __half* suh = reinterpret_cast<const __half*>(table[expert].suh);
    const unsigned long long in_index = static_cast<unsigned long long>(expanded ? slot : slot / topk) * k + feature;
    const __half product = __hmul(__float2half_rn(__bfloat162float(x[in_index])), suh[feature]);
    __shared__ float scratch[128];
    xh[out_index] = half_round(hadamard128(__half2float(product), scratch));
}

// Grid(ceil(n/128),rows*topk,1), block(128,1,1). Native on-demand decode.
// Scratch always F32; thin outputs are explicitly FP16-representable F32.
// This scalar accumulation does not claim the reference MMA/split-K order.
extern "C" __global__ void exl3_moe_mul1_matmul_precision(
    const float* __restrict__ xh, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, float* __restrict__ yh,
    const unsigned int* __restrict__ counts,
    unsigned int rows, unsigned int n, unsigned int k, unsigned int topk,
    unsigned int experts, unsigned int thin_cap) {
    const unsigned int slot = blockIdx.y, output = blockIdx.x * 128 + threadIdx.x;
    if (slot >= static_cast<unsigned long long>(rows) * topk || output >= n) return;
    const unsigned long long index = static_cast<unsigned long long>(slot) * n + output;
    const int expert = ids[slot];
    if (!valid_expert(expert, experts)) { yh[index] = __int_as_float(0x7fc00000); return; }
    const Exl3Expert descriptor = table[expert];
    const unsigned int* packed = reinterpret_cast<const unsigned int*>(descriptor.trellis);
    float sum = 0.0f;
    for (unsigned int input = 0; input < k; ++input) {
        const unsigned long long tile = static_cast<unsigned long long>(input / 16) * (n / 16) + output / 16;
        const float weight = mul1(tile_state(packed + tile * 8 * descriptor.bits, input & 15, output & 15, descriptor.bits));
        sum = __fadd_rn(sum, __fmul_rn(xh[static_cast<unsigned long long>(slot) * k + input], weight));
    }
    yh[index] = counts[expert] <= thin_cap ? half_round(sum) : sum;
}

// Grid(n/128,rows*topk,1), block(128,1,1). Gate/up original basis outputs.
extern "C" __global__ void exl3_moe_output_hadamard_precision(
    const float* __restrict__ yh, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, float* __restrict__ output,
    const unsigned int* __restrict__ counts,
    unsigned int rows, unsigned int n, unsigned int topk, unsigned int experts, unsigned int thin_cap) {
    const unsigned int slot = blockIdx.y, feature = blockIdx.x * 128 + threadIdx.x;
    if (slot >= static_cast<unsigned long long>(rows) * topk || feature >= n) return;
    const unsigned long long index = static_cast<unsigned long long>(slot) * n + feature;
    const int expert = ids[slot];
    if (!valid_expert(expert, experts)) { output[index] = __int_as_float(0x7fc00000); return; }
    __shared__ float scratch[128];
    const __half* svh = reinterpret_cast<const __half*>(table[expert].svh);
    const float value = hadamard128(yh[index], scratch);
    output[index] = counts[expert] <= thin_cap
        ? __half2float(__hmul(__float2half_rn(value), svh[feature]))
        : __fmul_rn(value, __half2float(svh[feature]));
}

// Grid(k/128,rows*topk,1), block(128,1,1). Output may exactly alias gate.
// Thin uses half2 SiLU/post-SiLU clamp; grouped fat clamps before F32 SiLU.
extern "C" __global__ void exl3_moe_activate_down_hadamard(
    const float* gate, const float* __restrict__ up, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ down_table, float* output,
    const unsigned int* __restrict__ counts,
    unsigned int rows, unsigned int k, unsigned int topk, unsigned int experts,
    unsigned int thin_cap, float limit, unsigned int clamp) {
    const unsigned int slot = blockIdx.y, feature = blockIdx.x * 128 + threadIdx.x;
    if (slot >= static_cast<unsigned long long>(rows) * topk || feature >= k) return;
    const unsigned long long index = static_cast<unsigned long long>(slot) * k + feature;
    const int expert = ids[slot];
    if (!valid_expert(expert, experts)) { output[index] = __int_as_float(0x7fc00000); return; }
    float g = gate[index], u = up[index], activation;
    if (counts[expert] <= thin_cap) {
        activation = thin_activation(g, u, limit, clamp);
    } else {
        if (clamp) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        const float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, static_cast<float>(exp(-static_cast<double>(g)))));
        activation = __fmul_rn(__fmul_rn(sigmoid, g), u);
    }
    const __half* suh = reinterpret_cast<const __half*>(down_table[expert].suh);
    const __half product = __hmul(__float2half_rn(activation), suh[feature]);
    __shared__ float scratch[128];
    output[index] = half_round(hadamard128(__half2float(product), scratch));
}

// Grid(n/128,rows*topk,1), block(128,1,1). No per-expert BF16 rounding.
extern "C" __global__ void exl3_moe_down_weighted_output(
    const float* __restrict__ yh, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, const float* __restrict__ weights,
    float* __restrict__ output, const unsigned int* __restrict__ counts,
    unsigned int rows, unsigned int n, unsigned int topk, unsigned int experts, unsigned int thin_cap) {
    const unsigned int slot = blockIdx.y, feature = blockIdx.x * 128 + threadIdx.x;
    if (slot >= static_cast<unsigned long long>(rows) * topk || feature >= n) return;
    const unsigned long long index = static_cast<unsigned long long>(slot) * n + feature;
    const int expert = ids[slot];
    if (!valid_expert(expert, experts)) { output[index] = __int_as_float(0x7fc00000); return; }
    __shared__ float scratch[128];
    const float weight = half_round(weights[slot]);
    const __half* svh = reinterpret_cast<const __half*>(table[expert].svh);
    if (counts[expert] <= thin_cap) {
        const float value = hadamard128_scaled(yh[index], scratch, __fmul_rn(H128_NORM, weight));
        output[index] = __fmul_rn(value, __half2float(svh[feature]));
    } else {
        const float value = __fmul_rn(hadamard128(yh[index], scratch), __half2float(svh[feature]));
        output[index] = __fmul_rn(value, weight);
    }
}

// Grid(ceil(rows*hidden/256),1,1), block(256,1,1). Routed sum rounds BF16
// before shared BF16 addition, then local combined BF16 gets one TP sum.
// round_accum=1 is the bf16_routed_accum statistical-gate control: the
// running routed sum is rounded to BF16 after every slot addition. With
// round_accum=0 the arithmetic is identical to the reference policy.
extern "C" __global__ void exl3_moe_combine_f32(
    const float* __restrict__ weighted_output, const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows, unsigned int hidden, unsigned int topk, unsigned int has_shared,
    unsigned int round_accum) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned long long>(rows) * hidden) return;
    const unsigned long long row = index / hidden;
    const unsigned int feature = index % hidden;
    float sum = 0.0f;
    for (unsigned int slot = 0; slot < topk; ++slot) {
        sum = __fadd_rn(sum, weighted_output[(row * topk + slot) * hidden + feature]);
        if (round_accum) sum = __bfloat162float(__float2bfloat16_rn(sum));
    }
    sum = __bfloat162float(__float2bfloat16_rn(sum));
    if (has_shared) sum = __fadd_rn(sum, __bfloat162float(shared[index]));
    output[index] = __float2bfloat16_rn(sum);
}
