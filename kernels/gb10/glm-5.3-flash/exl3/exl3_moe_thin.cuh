// SPDX-License-Identifier: AGPL-3.0-only
// Thin routed-expert activation shared by exl3_moe_precision.cu and the
// cooperative down kernel's fused staging (exl3_coop_gemm.cu): half2 SiLU,
// optional post-SiLU clamp, one F16 product (thin numerics policy).
#pragma once
#include <cuda_fp16.h>

namespace {
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
