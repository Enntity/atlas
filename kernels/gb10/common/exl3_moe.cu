// SPDX-License-Identifier: AGPL-3.0-only

// Native indirect EXL3 execution for routed expert slots. Host validates
// descriptor ownership/bitrate, slot geometry and IDs produced by the router.
// No CPU route readback, per-expert launch loop or dense decoded weights.
#include "exl3_common.cuh"
#include "exl3_expert.cuh"

// Grid(K/128, rows*topk, 1), block(128,1,1).
extern "C" __global__ void exl3_moe_input_hadamard(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    float* __restrict__ xh,
    unsigned int rows, unsigned int k, unsigned int topk, unsigned int expanded, unsigned int experts
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk || blockIdx.x >= k / 128u) return;
    __shared__ float scratch[128];
    const int expert = ids[slot];
    const bool valid = expert >= 0 && static_cast<unsigned int>(expert) < experts;
    const Exl3Expert descriptor = valid ? table[expert] : Exl3Expert{};
    if (!valid) {
        xh[static_cast<unsigned long long>(slot) * k + blockIdx.x * 128u + threadIdx.x] = __int_as_float(0x7fc00000);
        return;
    }
    const __half* suh = reinterpret_cast<const __half*>(descriptor.suh);
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned int input_row = expanded ? slot : slot / topk;
    const unsigned long long input_index = static_cast<unsigned long long>(input_row) * k + feature;
    const float value = __fmul_rn(__bfloat162float(x[input_index]), __half2float(suh[feature]));
    xh[static_cast<unsigned long long>(slot) * k + feature] = hadamard128(value, scratch);
}

// Grid(ceil(N/128), rows*topk,1), block(128,1,1).
extern "C" __global__ void exl3_moe_mul1_matmul(
    const float* __restrict__ xh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    float* __restrict__ yh,
    unsigned int rows, unsigned int n, unsigned int k, unsigned int topk, unsigned int experts
) {
    const unsigned int slot = blockIdx.y;
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    if (slot >= static_cast<unsigned long long>(rows) * topk || output >= n) return;
    const int expert = ids[slot];
    const bool valid = expert >= 0 && static_cast<unsigned int>(expert) < experts;
    const Exl3Expert descriptor = valid ? table[expert] : Exl3Expert{};
    if (!valid) {
        yh[static_cast<unsigned long long>(slot) * n + output] = __int_as_float(0x7fc00000);
        return;
    }
    const unsigned int* trellis = reinterpret_cast<const unsigned int*>(descriptor.trellis);
    const unsigned long long input_base = static_cast<unsigned long long>(slot) * k;
    float sum = 0.0f;
    for (unsigned int input = 0; input < k; ++input) {
        const unsigned long long tile = static_cast<unsigned long long>(input / 16u)
            * (n / 16u) + output / 16u;
        const unsigned int* words = trellis + tile * (8u * descriptor.bits);
        const float weight = mul1(tile_state(words, input & 15u, output & 15u, descriptor.bits));
        sum = __fadd_rn(sum, __fmul_rn(xh[input_base + input], weight));
    }
    yh[static_cast<unsigned long long>(slot) * n + output] = sum;
}

// Grid(N/128, rows*topk,1), block(128,1,1).
extern "C" __global__ void exl3_moe_output_hadamard(
    const float* __restrict__ yh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows, unsigned int n, unsigned int topk, unsigned int experts
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= static_cast<unsigned long long>(rows) * topk || blockIdx.x >= n / 128u) return;
    __shared__ float scratch[128];
    const int expert = ids[slot];
    const bool valid = expert >= 0 && static_cast<unsigned int>(expert) < experts;
    const Exl3Expert descriptor = valid ? table[expert] : Exl3Expert{};
    if (!valid) {
        output[static_cast<unsigned long long>(slot) * n + blockIdx.x * 128u + threadIdx.x] = __float2bfloat16_rn(__int_as_float(0x7fc00000));
        return;
    }
    const __half* svh = reinterpret_cast<const __half*>(descriptor.svh);
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = static_cast<unsigned long long>(slot) * n + feature;
    output[index] = __float2bfloat16_rn(__fmul_rn(hadamard128(yh[index], scratch), __half2float(svh[feature])));
}

// Combine rank-local expert outputs and shared FFN before ONE TP reduction.
// Grid(ceil(rows*hidden/256),1,1), block(256,1,1). Shared may be absent only
// when has_shared=0. Output must not alias shared or the expert slot buffers.
extern "C" __global__ void exl3_moe_combine(
    const __nv_bfloat16* __restrict__ expert_output,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows, unsigned int hidden, unsigned int topk, unsigned int has_shared
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned long long>(rows) * hidden) return;
    const unsigned long long row = index / hidden;
    const unsigned int feature = index % hidden;
    float sum = 0.0f;
    for (unsigned int slot = 0; slot < topk; ++slot) {
        const unsigned long long route = row * topk + slot;
        sum = __fadd_rn(sum, __fmul_rn(weights[route], __bfloat162float(expert_output[route * hidden + feature])));
    }
    if (has_shared) sum = __fadd_rn(sum, __bfloat162float(shared[index]));
    output[index] = __float2bfloat16_rn(sum);
}

// F32 router weights [experts,hidden] from the original FP16 checkpoint.
// Grid(ceil(experts/128),rows,1),block(128,1,1).
extern "C" __global__ void exl3_router(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ gate,
    float* __restrict__ logits,
    unsigned int rows, unsigned int experts, unsigned int hidden
) {
    const unsigned int expert = blockIdx.x * 128u + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (expert >= experts || row >= rows) return;
    float sum = 0.0f;
    for (unsigned int feature = 0; feature < hidden; ++feature) {
        const float activation = __bfloat162float(input[static_cast<unsigned long long>(row) * hidden + feature]);
        sum = __fadd_rn(sum, __fmul_rn(activation, gate[static_cast<unsigned long long>(expert) * hidden + feature]));
    }
    logits[static_cast<unsigned long long>(row) * experts + expert] = sum;
}

// Host validates 1<=topk<=experts<=512, finite bias/scale, normalize=0|1,
// and the explicit one-group routing policy. Grid(rows,1,1),block(256,1,1).
// Correction bias affects selection only. Lower expert ID wins exact ties.
extern "C" __global__ void exl3_topk_sigmoid(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ ids,
    float* __restrict__ weights,
    unsigned int rows, unsigned int experts, unsigned int topk,
    unsigned int normalize, float scale
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float scores[512];
    __shared__ float selection[512];
    for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float logit = logits[static_cast<unsigned long long>(row) * experts + expert];
        const float score = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-logit)));
        scores[expert] = score;
        selection[expert] = __fadd_rn(score, bias[expert]);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const unsigned long long base = static_cast<unsigned long long>(row) * topk;
    float sum = 0.0f;
    for (unsigned int slot = 0; slot < topk; ++slot) {
        unsigned int best = 0;
        float best_value = __int_as_float(0xff800000);
        for (unsigned int expert = 0; expert < experts; ++expert) {
            if (selection[expert] > best_value) {
                best_value = selection[expert];
                best = expert;
            }
        }
        ids[base + slot] = static_cast<int>(best);
        weights[base + slot] = scores[best];
        sum = __fadd_rn(sum, scores[best]);
        selection[best] = __int_as_float(0xff800000);
    }
    for (unsigned int slot = 0; slot < topk; ++slot) {
        float value = weights[base + slot];
        if (normalize) value = __fdiv_rn(value, sum);
        weights[base + slot] = __fmul_rn(value, scale);
    }
}

// exl3_topk_sigmoid with a parallel argmax per slot (ATLAS_EXL3_PAR_SMALL=1):
// the block computes the scores, then warp 0 runs each slot's argmax with
// shuffles. Selection equals the serial scan: the first strictly greater value
// wins, so ties go to the lowest index, NaN/-inf never win, and expert 0 is the
// fallback. Weights, the sum and normalization stay serial in slot order.
extern "C" __global__ void exl3_topk_sigmoid_par(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ ids,
    float* __restrict__ weights,
    unsigned int rows, unsigned int experts, unsigned int topk,
    unsigned int normalize, float scale
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float scores[512];
    __shared__ float selection[512];
    for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float logit = logits[static_cast<unsigned long long>(row) * experts + expert];
        const float score = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-logit)));
        scores[expert] = score;
        selection[expert] = __fadd_rn(score, bias[expert]);
    }
    __syncthreads();
    if (threadIdx.x >= 32) return;
    const unsigned int lane = threadIdx.x;
    const unsigned long long base = static_cast<unsigned long long>(row) * topk;
    float chosen = 0.0f;  // lane `slot` keeps that slot's score
    for (unsigned int slot = 0; slot < topk; ++slot) {
        float value = __int_as_float(0xff800000);
        unsigned int index = 0;
        for (unsigned int expert = lane; expert < experts; expert += 32) {
            if (selection[expert] > value) { value = selection[expert]; index = expert; }
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, value, offset);
            const unsigned int other_id = __shfl_xor_sync(0xffffffffu, index, offset);
            if (other > value || (other == value && other_id < index)) { value = other; index = other_id; }
        }
        if (lane == slot) chosen = scores[index];
        if (lane == 0) {
            ids[base + slot] = static_cast<int>(index);
            selection[index] = __int_as_float(0xff800000);
        }
        __syncwarp();
    }
    float sum = 0.0f;
    for (unsigned int slot = 0; slot < topk; ++slot) sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, chosen, slot));
    if (lane < topk) {
        float value = chosen;
        if (normalize) value = __fdiv_rn(value, sum);
        weights[base + lane] = __fmul_rn(value, scale);
    }
}
