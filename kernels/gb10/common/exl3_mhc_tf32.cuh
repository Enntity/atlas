// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Native rounded mHC projection. Each warp owns complete balanced K64 split
// ranges; only row zero of its M16 fragment contains the current residual.
// All lanes supply the TF32 weight fragments. No residual writes are allowed
// until every split has completed and the caller has synchronized the block.
namespace exl3_mhc_tf32 {
// One warp's split. Quantity q (24 mixes, then the squared norm) of this
// split lands at output[split * split_stride + q * quantity_stride].
__device__ __forceinline__ void split_partial(
    const float* residual, const float* weight, float* output,
    unsigned int width, unsigned int splits, unsigned int split,
    unsigned int split_stride, unsigned int quantity_stride
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int group = lane / 4u;
    const unsigned int offset = lane & 3u;
    const unsigned int blocks = width / 64u;
    const unsigned int remainder = blocks % splits;
    {
        const unsigned int count = blocks / splits + (split < remainder);
        const unsigned int start = (split * (blocks / splits) + min(split, remainder)) * 64u;
        float acc[3][4] = {};
        float square = 0.0f;
        for (unsigned int k = start; k < start + count * 64u; k += 8u) {
            const float a0 = group == 0 ? bf16(residual[k + offset]) : 0.0f;
            const float a2 = group == 0 ? bf16(residual[k + offset + 4u]) : 0.0f;
            square = __fadd_rn(square, __fadd_rn(__fmul_rn(a0, a0), __fmul_rn(a2, a2)));
            for (unsigned int tile = 0; tile < 3; ++tile) {
                const unsigned long long w = static_cast<unsigned long long>(tile * 8u + group) * width + k + offset;
                const unsigned int b0 = __float_as_uint(weight[w]);
                const unsigned int b1 = __float_as_uint(weight[w + 4u]);
                const unsigned int zero = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[tile][0]), "+f"(acc[tile][1]), "+f"(acc[tile][2]), "+f"(acc[tile][3])
                    : "r"(__float_as_uint(a0)), "r"(zero),
                      "r"(__float_as_uint(a2)), "r"(zero), "r"(b0), "r"(b1));
            }
        }
        for (unsigned int step = 2; step > 0; step >>= 1)
            square = __fadd_rn(square, __shfl_xor_sync(0xffffffffu, square, step));
        if (lane == 0) output[split * split_stride + 24u * quantity_stride] = square;
        if (group == 0) {
            for (unsigned int tile = 0; tile < 3; ++tile) {
                const unsigned int column = tile * 8u + offset * 2u;
                output[split * split_stride + column * quantity_stride] = acc[tile][0];
                output[split * split_stride + (column + 1u) * quantity_stride] = acc[tile][1];
            }
        }
    }
}

__device__ __forceinline__ void partials(
    const float* residual, const float* weight, float* output,
    unsigned int width, unsigned int splits
) {
    for (unsigned int split = threadIdx.x / 32u; split < splits; split += 8u)
        split_partial(residual, weight, output, width, splits, split, 25u, 1u);
}
} // namespace exl3_mhc_tf32
