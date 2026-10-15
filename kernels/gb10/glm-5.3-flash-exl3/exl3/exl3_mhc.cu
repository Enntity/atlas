// SPDX-License-Identifier: AGPL-3.0-only

// Native GLM EXL3 mHC numerical boundary. The highway retains its existing
// four-byte arena layout; stored residuals become BF16-representable F32.
// A small fused site measures the incoming unrounded post residual before
// rounding it for the next sublayer. Large sites measure rounded residuals.
// This distinct export preserves all legacy hyper_connection kernels.
#include <cuda_bf16.h>

namespace {
constexpr unsigned int HC = 4;
constexpr unsigned int MIX = 24;
constexpr unsigned int THREADS = 256;

__device__ __forceinline__ float reduce(float* values, unsigned int lane) {
    for (unsigned int width = THREADS / 2; width > 0; width >>= 1) {
        if (lane < width) values[lane] = __fadd_rn(values[lane], values[lane + width]);
        __syncthreads();
    }
    return values[0];
}

// Match the reference's butterfly warp reduction followed by a serial
// sum of warp leaders for the small fused projection. The rounded
// projection remains a separate diagnostic path until its TF32 policy is proven.
__device__ __forceinline__ float projection_reduce(
    float* values, unsigned int lane, unsigned int groups
) {
    for (unsigned int group = 0; group < groups; ++group) {
        float value = values[group * THREADS + lane];
        for (unsigned int offset = 16; offset > 0; offset >>= 1)
            value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
        if ((lane & 31) == 0) values[group * THREADS + lane] = value;
    }
    __syncthreads();
    if (lane == 0) {
        float value = 0.0f;
        for (unsigned int warp = 0; warp < groups * THREADS / 32; ++warp)
            value = __fadd_rn(value, values[warp * 32]);
        values[0] = value;
    }
    __syncthreads();
    const float result = values[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float sum4(float a, float b, float c, float d) {
    return __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
}

__device__ __forceinline__ float sigmoid(float value) {
    // This module explicitly enables contraction for the libdevice expf
    // denominator, matching the installed reference. All reduction and
    // residual operations use explicit rounding intrinsics below.
    return 1.0f / (1.0f + expf(-value));
}
} // namespace

#include "exl3_mhc_tf32.cuh"

// Grid(rows), block(256). All dimensions/policies are validated by the host.
// normed must not alias highway; post/comb are overwritten after raw mixes.
extern "C" __global__ void exl3_mhc_pre_norm(
    float* highway,
    const float* __restrict__ fn,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normed,
    float* __restrict__ post,
    float* __restrict__ comb,
    unsigned int hidden, unsigned int iterations,
    float rms_eps, float norm_eps, float hc_eps, float post_multiplier,
    unsigned int unrounded_mix, unsigned int rounded_splits
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    const unsigned int width = HC * hidden;
    float* residual = highway + static_cast<unsigned long long>(row) * width;
    __shared__ float scratch[2 * THREADS];
    __shared__ float mixes[MIX];
    __shared__ float pre[HC];
    __shared__ float inverse;
    __shared__ float rounded_partials[64 * 25];

    if (rounded_splits == 0 || rounded_splits > 64 || rounded_splits > width / 64) return;
    if (!unrounded_mix) {
        exl3_mhc_tf32::partials(residual, fn, rounded_partials, width, rounded_splits);
        __syncthreads();
        if (lane == 0) {
            float square = 0.0f;
            for (unsigned int split = 0; split < rounded_splits; ++split)
                square = __fadd_rn(square, rounded_partials[split * 25 + 24]);
            inverse = rsqrtf(__fadd_rn(square / static_cast<float>(width), rms_eps));
            for (unsigned int mix = 0; mix < MIX; ++mix) {
                float dot = 0.0f;
                for (unsigned int split = 0; split < rounded_splits; ++split)
                    dot = __fadd_rn(dot, rounded_partials[split * 25 + mix]);
                mixes[mix] = __fmul_rn(dot, inverse);
            }
        }
        __syncthreads();
    } else {
    const unsigned int groups = 1;
    const unsigned int splits = unrounded_mix ? (gridDim.x < 8 && hidden <= 4096 ? 8 : 4) : 1;
    const unsigned int projection_threads = groups * THREADS;
    const unsigned int split_width = unrounded_mix ? hidden / splits : width;
    float total_square = 0.0f;
    for (unsigned int split = 0; split < splits; ++split) {
        for (unsigned int group = 0; group < groups; ++group) {
            float square = 0.0f;
            for (unsigned int feature = group * THREADS + lane; feature < split_width; feature += projection_threads) {
                if (unrounded_mix) {
                    for (unsigned int stream = 0; stream < HC; ++stream) {
                        const float value = residual[stream * hidden + split * split_width + feature];
                        square = __fmaf_rn(value, value, square);
                    }
                } else {
                    const float value = bf16(residual[feature]);
                    square = __fmaf_rn(value, value, square);
                }
            }
            scratch[group * THREADS + lane] = square;
        }
        __syncthreads();
        total_square = __fadd_rn(total_square, (unrounded_mix ? projection_reduce(scratch, lane, groups) : reduce(scratch, lane)));
    }
    if (lane == 0) inverse = rsqrtf(__fadd_rn(total_square / static_cast<float>(width), rms_eps));
    __syncthreads();
    for (unsigned int mix = 0; mix < MIX; ++mix) {
        float total = 0.0f;
        for (unsigned int split = 0; split < splits; ++split) {
            for (unsigned int group = 0; group < groups; ++group) {
                float dot = 0.0f;
                for (unsigned int feature = group * THREADS + lane; feature < split_width; feature += projection_threads) {
                    if (unrounded_mix) {
                        for (unsigned int stream = 0; stream < HC; ++stream) {
                            const unsigned int at = stream * hidden + split * split_width + feature;
                            dot = __fmaf_rn(fn[static_cast<unsigned long long>(mix) * width + at], residual[at], dot);
                        }
                    } else {
                        dot = __fmaf_rn(fn[static_cast<unsigned long long>(mix) * width + feature], bf16(residual[feature]), dot);
                    }
                }
                scratch[group * THREADS + lane] = dot;
            }
            __syncthreads();
            total = __fadd_rn(total, (unrounded_mix ? projection_reduce(scratch, lane, groups) : reduce(scratch, lane)));
        }
        if (lane == 0) mixes[mix] = __fmul_rn(total, inverse);
        __syncthreads();
    }
    }
    if (lane == 0) {
        float matrix[HC * HC];
        for (unsigned int i = 0; i < HC; ++i) {
            pre[i] = __fadd_rn(sigmoid(__fmaf_rn(mixes[i], scale[0], base[i])), hc_eps);
            post[static_cast<unsigned long long>(row) * HC + i] = __fmul_rn(
                sigmoid(__fmaf_rn(mixes[HC + i], scale[1], base[HC + i])), post_multiplier);
            for (unsigned int j = 0; j < HC; ++j) {
                const unsigned int at = i * HC + j;
                matrix[at] = __fmaf_rn(mixes[2 * HC + at], scale[2], base[2 * HC + at]);
            }
            float* values = matrix + i * HC;
            const float maximum = fmaxf(fmaxf(values[0], values[1]), fmaxf(values[2], values[3]));
            for (unsigned int j = 0; j < HC; ++j) values[j] = expf(values[j] - maximum);
            const float denominator = sum4(values[0], values[1], values[2], values[3]);
            for (unsigned int j = 0; j < HC; ++j) values[j] = __fadd_rn(__fdiv_rn(values[j], denominator), hc_eps);
        }
        for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
            if (iteration != 0) {
                for (unsigned int i = 0; i < HC; ++i) {
                    float* values = matrix + i * HC;
                    const float denominator = __fadd_rn(sum4(values[0], values[1], values[2], values[3]), hc_eps);
                    for (unsigned int j = 0; j < HC; ++j) values[j] = __fdiv_rn(values[j], denominator);
                }
            }
            for (unsigned int j = 0; j < HC; ++j) {
                const float denominator = __fadd_rn(sum4(matrix[j], matrix[HC + j], matrix[2 * HC + j], matrix[3 * HC + j]), hc_eps);
                for (unsigned int i = 0; i < HC; ++i) matrix[i * HC + j] = __fdiv_rn(matrix[i * HC + j], denominator);
            }
        }
        for (unsigned int at = 0; at < HC * HC; ++at) comb[static_cast<unsigned long long>(row) * HC * HC + at] = matrix[at];
    }
    __syncthreads();
    // The reference collapses 1024-feature tiles over 64 lanes with
    // sixteen independent squared-sum fragments per lane. Preserve that
    // accumulation and reduction order, including the cross-warp step first.
    float square_slots[16] = {};
    if (lane < 64) {
        for (unsigned int block = 0; block < hidden; block += 1024) {
            for (unsigned int slot = 0; slot < 16; ++slot) {
                const unsigned int feature = block + (slot / 8) * 512 + lane * 8 + slot % 8;
                if (feature >= hidden) continue;
                float collapsed = 0.0f;
                for (unsigned int stream = 0; stream < HC; ++stream) {
                    const unsigned int at = stream * hidden + feature;
                    const float rounded = bf16(residual[at]);
                    residual[at] = rounded;
                    collapsed = __fmaf_rn(pre[stream], rounded, collapsed);
                }
                square_slots[slot] = __fmaf_rn(collapsed, collapsed, square_slots[slot]);
                normed[static_cast<unsigned long long>(row) * hidden + feature] = __float2bfloat16_rn(collapsed);
            }
        }
    }
    float square = 0.0f;
    for (unsigned int slot = 0; slot < 16; ++slot)
        square = __fadd_rn(square, square_slots[(slot & 1) * 8 + slot / 2]);
    scratch[lane] = square;
    __syncthreads();
    float collapsed_sum = 0.0f;
    if (lane < 64) {
        collapsed_sum = __fadd_rn(square, scratch[lane ^ 32]);
        for (unsigned int offset = 16; offset > 0; offset >>= 1)
            collapsed_sum = __fadd_rn(collapsed_sum, __shfl_xor_sync(0xffffffffu, collapsed_sum, offset));
    }
    if (lane == 0) inverse = rsqrtf(__fadd_rn(collapsed_sum / static_cast<float>(hidden), norm_eps));
    __syncthreads();
    for (unsigned int feature = lane; feature < hidden; feature += THREADS) {
        const unsigned long long at = static_cast<unsigned long long>(row) * hidden + feature;
        const float rounded = __bfloat162float(normed[at]);
        normed[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(rounded, inverse), __bfloat162float(norm_weight[feature])));
    }
}

// Compute the fused post residual in F32. Each feature owns all four old
// streams before writing, permitting the existing highway to alias output.
// The next pre site chooses whether to measure this unrounded residual or
// materialize BF16 first. Grid(rows), block(256).
extern "C" __global__ void exl3_mhc_post_unrounded(
    const __nv_bfloat16* __restrict__ block_output,
    const float* highway, const float* __restrict__ post,
    const float* __restrict__ comb, float* output, unsigned int hidden
) {
    const unsigned int row = blockIdx.x;
    const unsigned long long base = static_cast<unsigned long long>(row) * HC * hidden;
    for (unsigned int feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        float old[HC];
        for (unsigned int source = 0; source < HC; ++source) old[source] = bf16(highway[base + source * hidden + feature]);
        const float x = __bfloat162float(block_output[static_cast<unsigned long long>(row) * hidden + feature]);
        for (unsigned int target = 0; target < HC; ++target) {
            // Preserve the installed contraction order: round the first
            // residual product, then fuse the layer-output product into it.
            float value = __fmul_rn(comb[static_cast<unsigned long long>(row) * HC * HC + target], old[0]);
            value = __fmaf_rn(post[static_cast<unsigned long long>(row) * HC + target], x, value);
            for (unsigned int source = 1; source < HC; ++source) {
                value = __fmaf_rn(comb[static_cast<unsigned long long>(row) * HC * HC + source * HC + target], old[source], value);
            }
            output[base + target * hidden + feature] = value;
        }
    }
}

// Final contraction reads the rounded residual that the reference stores.
// Grid(rows), block(256). Input is the unrounded FP32 final post result.
extern "C" __global__ void exl3_mhc_contract(
    const float* __restrict__ highway, __nv_bfloat16* __restrict__ output,
    unsigned int hidden
) {
    const unsigned int row = blockIdx.x;
    for (unsigned int feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        const float* value = highway + static_cast<unsigned long long>(row) * HC * hidden + feature;
        float sum = 0.0f;
        for (unsigned int stream = 0; stream < HC; ++stream) sum = __fadd_rn(sum, bf16(value[stream * hidden]));
        output[static_cast<unsigned long long>(row) * hidden + feature] = __float2bfloat16_rn(__fmul_rn(sum, 0.25f));
    }
}
