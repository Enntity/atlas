// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3 uses the same mHC math and FP32 residual highway as DeepSeek-V4,
// but its large hc_fn matrices are stored in BF16. Compile the shared source
// with the checkpoint's exact input type instead of expanding 90 matrices to
// FP32 during model load.
#define HC_FN_TYPE __nv_bfloat16
#include "../../deepseek-v4-flash/nvfp4/hyper_connection.cu"

// Fixed GLM-5.3 tensor-core mHC pre path. The large learned projection is
// issued by the host as one FP32/TF32 GEMM; these two small kernels compute its
// RMS scale and finish the sigmoid/Sinkhorn/collapse around that GEMM.
// Keeping the FP32 residual highway preserves Atlas's long-generation
// stability while replacing the scalar 24-pass dot product.
#define GLM53_HC_HIDDEN 4096
#define GLM53_HC_MULT 4
#define GLM53_HC_DIM (GLM53_HC_HIDDEN * GLM53_HC_MULT)
#define GLM53_HC_MIX ((2 + GLM53_HC_MULT) * GLM53_HC_MULT)

extern "C" __global__ void glm53_hc_pre_sqsum(
    const float* __restrict__ streams,
    float* __restrict__ sqsum
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    float local = 0.f;
    const float* x = streams + (size_t)row * GLM53_HC_DIM;
    for (unsigned int i = tid; i < GLM53_HC_DIM; i += HC_BLOCK) {
        const float value = x[i];
        local += value * value;
    }
    __shared__ float reduction[HC_BLOCK];
    reduction[tid] = local;
    __syncthreads();
    const float total = hc_block_reduce(reduction, tid);
    if (tid == 0) sqsum[row] = total;
}

extern "C" __global__ void glm53_hc_pre_finish_norm(
    const float* __restrict__ streams,
    const float* __restrict__ mix,
    const float* __restrict__ sqsum,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ norm_output,
    float* __restrict__ post,
    float* __restrict__ comb,
    float norm_epsilon
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float pre_weights[GLM53_HC_MULT];
    if (tid == 0) {
        const float inverse_rms =
            rsqrtf(sqsum[row] / (float)GLM53_HC_DIM + 1.0e-5f);
        float matrix[GLM53_HC_MULT * GLM53_HC_MULT];
        for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
            const float pre_logit =
                mix[row * GLM53_HC_MIX + i] * inverse_rms * scale[0] + base[i];
            pre_weights[i] = 1.f / (1.f + expf(-pre_logit)) + 1.0e-6f;
            const float post_logit =
                mix[row * GLM53_HC_MIX + GLM53_HC_MULT + i] * inverse_rms * scale[1] +
                base[GLM53_HC_MULT + i];
            post[row * GLM53_HC_MULT + i] = 2.f / (1.f + expf(-post_logit));
        }
        for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
            float maximum = -1.0e30f;
            for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
                const unsigned int index = i * GLM53_HC_MULT + j;
                matrix[index] =
                    mix[row * GLM53_HC_MIX + 2 * GLM53_HC_MULT + index] * inverse_rms *
                        scale[2] +
                    base[2 * GLM53_HC_MULT + index];
                maximum = fmaxf(maximum, matrix[index]);
            }
            float denominator = 0.f;
            for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
                const unsigned int index = i * GLM53_HC_MULT + j;
                matrix[index] = expf(matrix[index] - maximum);
                denominator += matrix[index];
            }
            for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
                const unsigned int index = i * GLM53_HC_MULT + j;
                matrix[index] = matrix[index] / denominator + 1.0e-6f;
            }
        }
        for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
            float denominator = 1.0e-6f;
            for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                denominator += matrix[i * GLM53_HC_MULT + j];
            }
            for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                matrix[i * GLM53_HC_MULT + j] /= denominator;
            }
        }
        for (unsigned int iteration = 0; iteration < 19; ++iteration) {
            for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                float denominator = 1.0e-6f;
                for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
                    denominator += matrix[i * GLM53_HC_MULT + j];
                }
                for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
                    matrix[i * GLM53_HC_MULT + j] /= denominator;
                }
            }
            for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
                float denominator = 1.0e-6f;
                for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                    denominator += matrix[i * GLM53_HC_MULT + j];
                }
                for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                    matrix[i * GLM53_HC_MULT + j] /= denominator;
                }
            }
        }
        for (unsigned int j = 0; j < GLM53_HC_MULT; ++j) {
            float denominator = 0.f;
            for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                denominator += matrix[i * GLM53_HC_MULT + j];
            }
            const float inverse = denominator > 0.f ? 1.f / denominator : 0.f;
            for (unsigned int i = 0; i < GLM53_HC_MULT; ++i) {
                matrix[i * GLM53_HC_MULT + j] *= inverse;
            }
        }
        for (unsigned int i = 0; i < GLM53_HC_MULT * GLM53_HC_MULT; ++i) {
            comb[row * GLM53_HC_MULT * GLM53_HC_MULT + i] = matrix[i];
        }
    }
    __syncthreads();
    const float* x = streams + (size_t)row * GLM53_HC_DIM;
    float collapsed[GLM53_HC_HIDDEN / HC_BLOCK];
    float square_sum = 0.f;
    unsigned int slot = 0;
    for (unsigned int channel = tid; channel < GLM53_HC_HIDDEN;
         channel += HC_BLOCK, ++slot) {
        float value = 0.f;
        for (unsigned int stream = 0; stream < GLM53_HC_MULT; ++stream) {
            value += pre_weights[stream] * x[stream * GLM53_HC_HIDDEN + channel];
        }
        // Match the old two-kernel path exactly: RMSNorm observes the rounded
        // BF16 collapse, not the pre-rounding FP32 accumulator.
        const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
        output[(size_t)row * GLM53_HC_HIDDEN + channel] = rounded;
        collapsed[slot] = __bfloat162float(rounded);
        square_sum += collapsed[slot] * collapsed[slot];
    }
    if (norm_output == nullptr || norm_weight == nullptr) return;

    __shared__ float norm_reduction[HC_BLOCK];
    __shared__ float inverse_rms;
    norm_reduction[tid] = square_sum;
    __syncthreads();
    const float total = hc_block_reduce(norm_reduction, tid);
    if (tid == 0) {
        inverse_rms = rsqrtf(total / (float)GLM53_HC_HIDDEN + norm_epsilon);
    }
    __syncthreads();
    slot = 0;
    for (unsigned int channel = tid; channel < GLM53_HC_HIDDEN;
         channel += HC_BLOCK, ++slot) {
        const float normalized = collapsed[slot] * inverse_rms *
            __bfloat162float(norm_weight[channel]);
        norm_output[(size_t)row * GLM53_HC_HIDDEN + channel] =
            __float2bfloat16_rn(normalized);
    }
}
