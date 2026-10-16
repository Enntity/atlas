// SPDX-License-Identifier: AGPL-3.0-only
// Same results as exl3_topk_sigmoid_par for finite logits: scores/sigmoid
// unchanged; each slot picks the max selection with the lowest-id tie break,
// now found by warp 0 with shuffles instead of 8 block-wide tree reductions.
extern "C" __global__ void exl3_topk_sigmoid_par_v3(
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
    // Lane L holds selection[L + 32 i] (experts <= 512, host-checked).
    float mine[16];
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        const unsigned int expert = lane + 32u * i;
        mine[i] = expert < experts ? selection[expert] : __int_as_float(0xff800000);
    }
    float chosen = 0.0f;  // lane `slot` keeps that slot's score
    for (unsigned int slot = 0; slot < topk; ++slot) {
        float value = __int_as_float(0xff800000);
        unsigned int index = 0;
#pragma unroll
        for (unsigned int i = 0; i < 16; ++i) {
            if (lane + 32u * i < experts && mine[i] > value) { value = mine[i]; index = lane + 32u * i; }
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, value, offset);
            const unsigned int other_id = __shfl_xor_sync(0xffffffffu, index, offset);
            if (other > value || (other == value && other_id < index)) { value = other; index = other_id; }
        }
        if (lane == slot) chosen = scores[index];
        if (lane == 0) ids[base + slot] = static_cast<int>(index);
#pragma unroll
        for (unsigned int i = 0; i < 16; ++i)
            if (lane + 32u * i == index) mine[i] = __int_as_float(0xff800000);
    }
    float sum = 0.0f;
    for (unsigned int slot = 0; slot < topk; ++slot) sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, chosen, slot));
    if (lane < topk) {
        float value = chosen;
        if (normalize) value = __fdiv_rn(value, sum);
        weights[base + lane] = __fmul_rn(value, scale);
    }
}
