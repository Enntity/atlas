// SPDX-License-Identifier: AGPL-3.0-only
// Bounded shared-expert original-basis reconstruction only. Callers own two
// F32[K,N] buffers and BF16[N,K], reconstruct global geometry before TP slicing.
// Unlike packed execution, normalized H coefficients multiply inside each dot.
// Scalar ordered FMA is explicit; qualification against installed Torch GEMM
// and actual loaded shared weights is required, not assumed equivalent.
#include "exl3_common.cuh"

// All exports: grid (ceil(N/128), K, 1), block (128, 1, 1).
// K,N must be H128-aligned; bits must be integer 1..8; K <= 65535.
extern "C" __global__ void exl3_shared_decode(
    const unsigned int* __restrict__ trellis, float* __restrict__ inner,
    unsigned int k, unsigned int n, unsigned int bits
) {
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    const unsigned int input = blockIdx.y;
    if (input >= k || output >= n) return;
    const unsigned long long tile = static_cast<unsigned long long>(input / 16u)
        * (n / 16u) + output / 16u;
    const unsigned int* words = trellis + tile * (8u * bits);
    inner[static_cast<unsigned long long>(input) * n + output] =
        mul1(tile_state(words, input & 15u, output & 15u, bits));
}

extern "C" __global__ void exl3_shared_left_hadamard(
    const float* __restrict__ inner, const __half* __restrict__ suh,
    float* __restrict__ transformed, unsigned int k, unsigned int n
) {
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    const unsigned int input = blockIdx.y;
    if (input >= k || output >= n) return;
    const unsigned int base = input & ~127u;
    float sum = 0.0f;
    for (unsigned int lane = 0; lane < 128u; ++lane) {
        const float coefficient = (__popc((input & 127u) & lane) & 1u)
            ? -H128_NORM : H128_NORM;
        sum = __fmaf_rn(coefficient,
            inner[static_cast<unsigned long long>(base + lane) * n + output], sum);
    }
    transformed[static_cast<unsigned long long>(input) * n + output] =
        __fmul_rn(sum, __half2float(suh[input]));
}

extern "C" __global__ void exl3_shared_right_hadamard(
    const float* __restrict__ transformed, const __half* __restrict__ svh,
    __nv_bfloat16* __restrict__ weight, unsigned int k, unsigned int n
) {
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    const unsigned int input = blockIdx.y;
    if (input >= k || output >= n) return;
    const unsigned int base = output & ~127u;
    float sum = 0.0f;
    for (unsigned int lane = 0; lane < 128u; ++lane) {
        const float coefficient = (__popc((output & 127u) & lane) & 1u)
            ? -H128_NORM : H128_NORM;
        sum = __fmaf_rn(transformed[static_cast<unsigned long long>(input) * n + base + lane],
                        coefficient, sum);
    }
    const float scaled = __fmul_rn(sum, __half2float(svh[output]));
    const float rounded_half = __half2float(__float2half_rn(scaled));
    weight[static_cast<unsigned long long>(output) * k + input] =
        __float2bfloat16_rn(rounded_half);
}
