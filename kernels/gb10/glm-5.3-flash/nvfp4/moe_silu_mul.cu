// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the deepseek-v4-flash `moe_silu_mul.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../deepseek-v4-flash/nvfp4/moe_silu_mul.cu"

#include "silu_nvfp4_quant.cuh"

// Fused DeepSeek-clamped SiLU·mul + standard token-major NVFP4 quantization.
//
// This is the native-FP4 analogue of common/moe_silu_mul.cu's fused FP8
// helper.  It emits exactly the packed `[M,K/2]` E2M1 + `[M,K/16]` E4M3
// layout consumed by moe_w4a4_grouped_gemm_prequant_t_k64.  The product is
// explicitly rounded through BF16 before the group maximum and FP4 encode,
// matching the old moe_silu_mul -> quantize_bf16_to_nvfp4 pair.
//
// `out_bf16` is nullable.  Base-model serving passes null and avoids the full
// BF16 intermediate; MoE down-projection LoRA may request it because its fold
// consumes the post-SiLU activations.

extern "C" __global__ void silu_mul_quant_nvfp4(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned int M,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* grow = gate + (unsigned long long)row * K;
    const __nv_bfloat16* urow = up + (unsigned long long)row * K;
    unsigned char* prow = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* srow = scale_out + (unsigned long long)row * (K / 16);
    __nv_bfloat16* brow = out_bf16 ? out_bf16 + (unsigned long long)row * K : nullptr;
    const unsigned int groups = K / 16;

    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        float vals[16];
        float group_max = 0.0f;
        const unsigned int base = group * 16;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float r = silu_nvfp4_act(__bfloat162float(grow[base + i]),
                                           __bfloat162float(urow[base + i]));
            if (brow) brow[base + i] = __float2bfloat16(r);
            vals[i] = r;
            group_max = fmaxf(group_max, fabsf(r));
        }

        float inv;
        srow[group] = silu_nvfp4_group_scale(group_max, &inv);
#pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const unsigned int q0 = silu_nvfp4_quantize_e2m1(vals[i] * inv);
            const unsigned int q1 = silu_nvfp4_quantize_e2m1(vals[i + 1] * inv);
            prow[group * 8 + i / 2] = (unsigned char)((q1 << 4) | q0);
        }
    }
}
