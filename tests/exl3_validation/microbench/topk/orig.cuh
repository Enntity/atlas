// SPDX-License-Identifier: AGPL-3.0-only
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
    __shared__ float best_values[256];
    __shared__ unsigned int best_ids[256];
    for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float logit = logits[static_cast<unsigned long long>(row) * experts + expert];
        const float score = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-logit)));
        scores[expert] = score;
        selection[expert] = __fadd_rn(score, bias[expert]);
    }
    __syncthreads();
    const unsigned long long base = static_cast<unsigned long long>(row) * topk;
    for (unsigned int slot = 0; slot < topk; ++slot) {
        float value = __int_as_float(0xff800000);
        unsigned int index = 0;
        for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
            if (selection[expert] > value) { value = selection[expert]; index = expert; }
        }
        best_values[threadIdx.x] = value;
        best_ids[threadIdx.x] = index;
        __syncthreads();
        for (unsigned int width = blockDim.x / 2; width > 0; width >>= 1) {
            if (threadIdx.x < width) {
                const float other = best_values[threadIdx.x + width];
                const unsigned int other_id = best_ids[threadIdx.x + width];
                const float mine = best_values[threadIdx.x];
                const unsigned int mine_id = best_ids[threadIdx.x];
                if (other > mine || (other == mine && other_id < mine_id)) {
                    best_values[threadIdx.x] = other;
                    best_ids[threadIdx.x] = other_id;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const unsigned int best = best_ids[0];
            ids[base + slot] = static_cast<int>(best);
            weights[base + slot] = scores[best];
            selection[best] = __int_as_float(0xff800000);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    float sum = 0.0f;
    for (unsigned int slot = 0; slot < topk; ++slot) sum = __fadd_rn(sum, weights[base + slot]);
    for (unsigned int slot = 0; slot < topk; ++slot) {
        float value = weights[base + slot];
        if (normalize) value = __fdiv_rn(value, sum);
        weights[base + slot] = __fmul_rn(value, scale);
    }
}
