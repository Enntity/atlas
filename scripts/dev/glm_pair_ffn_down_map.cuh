// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "glm_pair_ffn_api.h"
// Standalone fixed-M10 scheduling only. The native MMA body is unchanged.
namespace pair_ffn {
static_assert(R == 10 && K == 8 && I == 2048 && H == 2 * I, "fixed M10 tile mapping");
constexpr unsigned gu_down_capacity = X * (I / 128);
__host__ __device__ inline bool reused_gu_index(unsigned wid, int total,
                                               unsigned capacity, unsigned& index) {
    if (total < 0 || capacity > gu_down_capacity || unsigned(total) > capacity) return false;
    index = wid / 2;
    return index < unsigned(total);
}
__host__ __device__ inline bool reused_gu_tile(unsigned packed, unsigned half,
                                              unsigned& m, unsigned& n) {
    m = packed >> 6;
    const unsigned gu_n = packed & 63;
    // Unique top8 over ten rows gives one M64 tile per expert, not a generic map.
    if (m != 0 || gu_n >= 16 || half > 1) return false;
    n = 2 * gu_n + half;
    return true;
}
}
