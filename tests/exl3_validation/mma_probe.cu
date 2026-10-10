// SPDX-License-Identifier: AGPL-3.0-only
// Diagnostic entry uses the same native MMA helper as scoped production.
#include "exl3_mma.cuh"
// Grid [ceil(N/32),rows,1], block128. Explicit alternative profiles are
// test-only controls; production accepts only its qualified named profile.
extern "C" __global__ void exl3_mma_probe(
    const float* xh, const unsigned int* packed, float* output,
    unsigned int rows, unsigned int n, unsigned int k, unsigned int bits,
    unsigned int tile_k, unsigned int tile_n, unsigned int slices,
    unsigned int half_partial) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int base_n = blockIdx.x * 32u + (threadIdx.x / 32u) * 8u;
    if (base_n >= n || blockIdx.y >= rows) return;
    float total[4];
    exl3_mma::accumulate(xh, packed, total,
        static_cast<unsigned long long>(blockIdx.y) * k, base_n,
        n, k, bits, tile_k, tile_n, slices, half_partial);
    if (lane < 4u) {
        const unsigned long long index = static_cast<unsigned long long>(blockIdx.y) * n
            + base_n + (lane & 3u) * 2u;
        output[index] = total[0];
        output[index + 1u] = total[1];
    }
}
