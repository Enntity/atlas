// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define GLM_DR_INLINE __host__ __device__ __forceinline__
#else
#define GLM_DR_INLINE inline
#endif
GLM_DR_INLINE uint32_t glm_m16_direct_fragment(
    const unsigned char* stage, unsigned stride, unsigned column, unsigned packed_k
) {
    return uint32_t(stage[packed_k * stride + column])
        | (uint32_t(stage[(packed_k + 1) * stride + column]) << 8)
        | (uint32_t(stage[(packed_k + 2) * stride + column]) << 16)
        | (uint32_t(stage[(packed_k + 3) * stride + column]) << 24);
}
#undef GLM_DR_INLINE
