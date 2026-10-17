// SPDX-License-Identifier: AGPL-3.0-only

// Correctness-first native EXL3 mul1 linear. No dense decoded weight buffer.
// All pointers are row-major; K and N must be nonzero multiples of 128.
// The host validates integer bits in 1..8 and the mul1 marker before launch.
// Original FP16 suh/svh are retained, while transformed activations are F32.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_common.cuh"

// Grid (K/128, rows, 1), block (128, 1, 1).
extern "C" __global__ void exl3_input_hadamard(
    const __nv_bfloat16* __restrict__ x,
    const __half* __restrict__ suh,
    float* __restrict__ xh,
    unsigned int rows, unsigned int k
) {
    if (blockIdx.y >= rows || blockIdx.x >= k / 128u) return;
    __shared__ float scratch[128];
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.y) * k + feature;
    const float value = __fmul_rn(__bfloat162float(x[index]), __half2float(suh[feature]));
    xh[index] = hadamard128(value, scratch);
}

// Explicit packed-reference input precision; original F32 export is unchanged.
// BF16 -> FP16, FP16 scale product, F32 H128, then FP16-representable F32.
// Grid (K/128, rows, 1), block (128, 1, 1).
extern "C" __global__ void exl3_input_hadamard_f16(
    const __nv_bfloat16* __restrict__ x,
    const __half* __restrict__ suh,
    float* __restrict__ xh,
    unsigned int rows, unsigned int k
) {
    if (blockIdx.y >= rows || blockIdx.x >= k / 128u) return;
    __shared__ float scratch[128];
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.y) * k + feature;
    const __half product = __hmul(__float2half_rn(__bfloat162float(x[index])), suh[feature]);
    xh[index] = __half2float(__float2half_rn(hadamard128(__half2float(product), scratch)));
}

// Grid (ceil(N/128), rows, 1), block (128, 1, 1).
// One thread computes one output coefficient, decoding weights on demand.
// Payload shape [K/16, N/16, 8*bits] in little-endian u32 words (I16 pairs).
extern "C" __global__ void exl3_mul1_matmul(
    const float* __restrict__ xh,
    const unsigned int* __restrict__ trellis,
    float* __restrict__ yh,
    unsigned int rows, unsigned int n, unsigned int k, unsigned int bits
) {
    const unsigned int output = blockIdx.x * 128u + threadIdx.x;
    if (blockIdx.y >= rows || output >= n) return;
    const unsigned long long input_base = static_cast<unsigned long long>(blockIdx.y) * k;
    const unsigned int output_tiles = n / 16u;
    float sum = 0.0f;
    for (unsigned int input = 0; input < k; ++input) {
        const unsigned long long tile = static_cast<unsigned long long>(input / 16u)
            * output_tiles + output / 16u;
        const unsigned int* words = trellis + tile * (8u * bits);
        const float weight = mul1(tile_state(words, input & 15u, output & 15u, bits));
        // Separate multiply/add matches the CPU reference summation order;
        // do not silently fuse this into FMA under compiler fast-math flags.
        sum = __fadd_rn(sum, __fmul_rn(xh[input_base + input], weight));
    }
    yh[static_cast<unsigned long long>(blockIdx.y) * n + output] = sum;
}

// Grid (N/128, rows, 1), block (128, 1, 1).
extern "C" __global__ void exl3_output_hadamard(
    const float* __restrict__ yh,
    const __half* __restrict__ svh,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows, unsigned int n
) {
    if (blockIdx.y >= rows || blockIdx.x >= n / 128u) return;
    __shared__ float scratch[128];
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.y) * n + feature;
    const float value = hadamard128(yh[index], scratch);
    output[index] = __float2bfloat16_rn(__fmul_rn(value, __half2float(svh[feature])));
}

// Grid (ceil(count/256), 1, 1), block (256, 1, 1). No launch for count=0.
// Dense/shared BF16 packed activation, alpha=1 and beta=0 explicitly.
// Gate clamps before SiLU; SiLU rounds to BF16 before the F32 up product.
// Reference contract: vLLM 487ecf187 libtorch_stable/activation_kernels.cu.
// output may equal gate; up must remain separate. The routed export below
// retains its independent post-SiLU clamp and rounding policy.
extern "C" __global__ void exl3_silu_mul_dense(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* output,
    unsigned long long count, float limit, unsigned int clamp
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float g = __bfloat162float(gate[index]);
    float u = __bfloat162float(up[index]);
    if (clamp) {
        g = __bfloat162float(__float2bfloat16_rn(fminf(g, limit)));
        u = fmaxf(fminf(u, limit), -limit);
    }
    const float activated = __bfloat162float(__float2bfloat16_rn(
        __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)))));
    // Reference applies up + beta even when beta is +0. This canonicalizes
    // negative-zero up to positive zero before multiplication.
    output[index] = __float2bfloat16_rn(__fmul_rn(activated, __fadd_rn(u, 0.0f)));
}

// Grid (ceil(count/256), 1, 1), block (256, 1, 1). No launch for count=0.
// Clamp policy is explicit and validated by the host. Standard SwiGLU applies
// the upper bound AFTER SiLU, and the symmetric bound to the up projection.
// output may equal gate; up must remain a separate, non-overlapping buffer.
// Formula reference (independent implementation, not copied source):
// github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/
// exllamav3/exllamav3_ext/activation_kernels.cuh, act_mul_kernel_f.
extern "C" __global__ void exl3_silu_mul(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* output,
    unsigned long long count, float limit, unsigned int clamp
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float g = __bfloat162float(gate[index]);
    float u = __bfloat162float(up[index]);
    float activated = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    if (clamp != 0u) {
        activated = fminf(activated, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    output[index] = __float2bfloat16_rn(__fmul_rn(activated, u));
}

// F32 final logits: grid(N/128,rows,1),block(128,1,1).
extern "C" __global__ void exl3_output_hadamard_f32(
    const float* __restrict__ yh, const __half* __restrict__ svh,
    float* __restrict__ output, unsigned int rows, unsigned int n
) {
    if (blockIdx.y >= rows || blockIdx.x >= n / 128u) return;
    __shared__ float scratch[128];
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.y) * n + feature;
    output[index] = __fmul_rn(hadamard128(yh[index], scratch), __half2float(svh[feature]));
}

// Rank-major NCCL gather [world,rows,local_vocab] -> [rows,active_vocab].
// Trim only checkpoint vocabulary padding, after gathering complete H128 blocks.
// Grid(ceil(rows*active_vocab/256),1,1),block(256,1,1).
extern "C" __global__ void exl3_vocab_gather_reorder(
    const float* __restrict__ gathered, float* __restrict__ output,
    unsigned int rows, unsigned int local_vocab, unsigned int active_vocab
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned long long>(rows) * active_vocab) return;
    const unsigned long long row = index / active_vocab;
    const unsigned int vocab = index % active_vocab;
    const unsigned int rank = vocab / local_vocab;
    const unsigned int local = vocab % local_vocab;
    output[index] = gathered[(static_cast<unsigned long long>(rank) * rows + row) * local_vocab + local];
}

// Serving's existing BF16 sampler buffer; raw F32 logits stay available in
// the native head resource for teacher-forced validation.
extern "C" __global__ void exl3_logits_to_bf16(
    const float* __restrict__ input, __nv_bfloat16* __restrict__ output,
    unsigned long long count
) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2bfloat16_rn(input[index]);
}
