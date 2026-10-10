// SPDX-License-Identifier: AGPL-3.0-only

// Independent global-load probe for the rounded mHC projection. This bypasses
// the reference's staging machinery while preserving TF32 MMA fragments,
// contiguous balanced split-K ranges and explicit squared-sum reductions.
#include <cuda_bf16.h>

// Grid(ceil(rows/128), splits), block(256). Outputs are split-major F32.
extern "C" __global__ void native_mhc_tf32_probe(
    const __nv_bfloat16* input, const float* weight,
    float* dots, float* squares,
    unsigned int rows, unsigned int width, unsigned int splits
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int group = lane / 4u;
    const unsigned int offset = lane & 3u;
    const unsigned int row = blockIdx.x * 128u + threadIdx.x / 32u * 16u + group;
    const unsigned int split = blockIdx.y;
    if (width % 64u || splits == 0 || splits > width / 64u) return;
    const unsigned int blocks = width / 64u;
    const unsigned int remainder = blocks % splits;
    const unsigned int count = blocks / splits + (split < remainder);
    const unsigned int start = (split * (blocks / splits) + min(split, remainder)) * 64u;
    float acc[3][4] = {};
    float sq0 = 0.0f, sq1 = 0.0f;
    for (unsigned int k = start; k < start + count * 64u; k += 8u) {
        const unsigned long long at0 = static_cast<unsigned long long>(row) * width + k + offset;
        const unsigned long long at1 = static_cast<unsigned long long>(row + 8u) * width + k + offset;
        const float a0 = row < rows ? __bfloat162float(input[at0]) : 0.0f;
        const float a1 = row + 8u < rows ? __bfloat162float(input[at1]) : 0.0f;
        const float a2 = row < rows ? __bfloat162float(input[at0 + 4u]) : 0.0f;
        const float a3 = row + 8u < rows ? __bfloat162float(input[at1 + 4u]) : 0.0f;
        sq0 = __fadd_rn(sq0, __fadd_rn(__fmul_rn(a0, a0), __fmul_rn(a2, a2)));
        sq1 = __fadd_rn(sq1, __fadd_rn(__fmul_rn(a1, a1), __fmul_rn(a3, a3)));
        for (unsigned int tile = 0; tile < 3; ++tile) {
            const unsigned long long w = static_cast<unsigned long long>(tile * 8u + group) * width + k + offset;
            const unsigned int b0 = __float_as_uint(weight[w]);
            const unsigned int b1 = __float_as_uint(weight[w + 4u]);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[tile][0]), "+f"(acc[tile][1]), "+f"(acc[tile][2]), "+f"(acc[tile][3])
                : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)),
                  "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)), "r"(b0), "r"(b1));
        }
    }
    for (unsigned int step = 2; step > 0; step >>= 1) {
        sq0 = __fadd_rn(sq0, __shfl_xor_sync(0xffffffffu, sq0, step));
        sq1 = __fadd_rn(sq1, __shfl_xor_sync(0xffffffffu, sq1, step));
    }
    const unsigned long long base = static_cast<unsigned long long>(split) * rows;
    if (offset == 0) {
        if (row < rows) squares[base + row] = sq0;
        if (row + 8u < rows) squares[base + row + 8u] = sq1;
    }
    for (unsigned int tile = 0; tile < 3; ++tile) {
        const unsigned int column = tile * 8u + offset * 2u;
        if (row < rows) {
            dots[(base + row) * 24u + column] = acc[tile][0];
            dots[(base + row) * 24u + column + 1u] = acc[tile][1];
        }
        if (row + 8u < rows) {
            dots[(base + row + 8u) * 24u + column] = acc[tile][2];
            dots[(base + row + 8u) * 24u + column + 1u] = acc[tile][3];
        }
    }
}
