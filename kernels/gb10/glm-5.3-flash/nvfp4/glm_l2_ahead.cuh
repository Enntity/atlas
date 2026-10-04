// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// L2 prefetch of the weights the main stream reads next (ATLAS_GLM_L2_AHEAD,
// crates/spark-model/src/layers/ops/l2_ahead.rs). A verify step's target
// forward runs on one stream; while it waits on an all-reduce or runs a
// latency-bound chain (the sparse-MLA indexer, the attention core) DRAM is
// idle. Forked onto a side stream at such a point, this kernel asks for the
// next projections' weights, so their GEMVs find part of them in L2.
//
// Loads, prefetches and bulk prefetches of immutable weights only: nothing is
// stored, so no output of any kernel can change. A region is `rows` rows of
// `row_bytes` bytes, `ld` bytes apart (a whole weight is one row). Every
// address asked for holds a byte of a region, or shares its 16-byte granule
// with one (bulk), so no request leaves the region's pages.
//
// Mechanisms (`mode`), which scripts/dev/glm_l2_ahead_bench.cu compares:
//   L2A_LINES    prefetch.global.L2, one per 128-byte line
//   L2A_SECTORS  prefetch.global.L2, one per 32-byte sector
//   L2A_LAST     prefetch.global.L2::evict_last, one per 32-byte sector
//   L2A_TOUCH    a discarded byte load per 32-byte sector (atlas_l2_touch;
//                the kernel then lasts until the bytes arrive)
//   L2A_BULK     cp.async.bulk.prefetch.L2, one per 4 KiB of a row. The
//                common/ notes on cp.async.bulk corrupting on sm_121 are about
//                bulk copies into shared memory; a bulk prefetch writes
//                nothing, so it can be slow or ignored, never wrong.
//
// Prior art (docs/glm-prior-art.md): jayleaton's L2 prefetch
// (github.com/jayleaton/glm53-tensorfold-spark patches/0460, Apache-2.0), as
// carried in Mia's TensorFold recipe patch 0046-glm-l2-prefetch (Apache-2.0):
// a small kernel on a side stream, forked at the all-gathers and after the
// attention projections, that prefetches the next weights (bulk, per-line
// evict_last or touch). Ours is written for Atlas's launches: regions of
// strided rows, the per-sector variants, and the sites in l2_ahead.rs. No
// code copied.
#include "atlas_pdl_touch.cuh"

#define L2A_LINES 0u
#define L2A_SECTORS 1u
#define L2A_LAST 2u
#define L2A_TOUCH 3u
#define L2A_BULK 4u
#define L2A_THREADS 256
#define L2A_REGIONS 8

struct L2aRegion {
    const unsigned char* p;
    unsigned long long ld;
    unsigned int row_bytes;
    unsigned int rows;
};

__device__ __forceinline__ unsigned int l2a_granule(unsigned int mode) {
    return mode == L2A_LINES ? 128u : (mode == L2A_BULK ? 4096u : 32u);
}

// Thread `t` of `threads` asks for its share of region `r` (fewer than 2^32
// granules: the host splits nothing larger than a few hundred MiB).
__device__ __forceinline__ void l2a_region(
    const L2aRegion r, unsigned int mode, unsigned int t, unsigned int threads
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const unsigned int g = l2a_granule(mode);
    const unsigned int per_row = (r.row_bytes + g - 1u) / g;
    const unsigned int total = r.rows * per_row;
    for (unsigned int i = t; i < total; i += threads) {
        const unsigned int off = (i % per_row) * g;
        const unsigned char* a = r.p + (unsigned long long)(i / per_row) * r.ld + off;
        if (mode == L2A_BULK) {
            const unsigned long long lo = (unsigned long long)a & ~15ull;
            const unsigned long long hi = ((unsigned long long)a + min(g, r.row_bytes - off) + 15ull) & ~15ull;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(lo), "r"((unsigned int)(hi - lo)) : "memory");
        } else if (mode == L2A_TOUCH) {
            atlas_l2_touch(a);
        } else if (mode == L2A_LAST) {
            asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
        } else {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
        }
    }
#endif
}

#define L2A_REGION(i) L2aRegion{p##i, ld##i, bytes##i, rows##i}

// Up to L2A_REGIONS regions (unused ones have rows = 0), asked for in order: each
// thread walks region 0's share, then region 1's, and so on. Plain launch on
// a side stream: not on the PDL list, no wait.
extern "C" __global__ void __launch_bounds__(L2A_THREADS) glm_l2_ahead(
    const unsigned char* __restrict__ p0, unsigned long long ld0, unsigned int bytes0, unsigned int rows0,
    const unsigned char* __restrict__ p1, unsigned long long ld1, unsigned int bytes1, unsigned int rows1,
    const unsigned char* __restrict__ p2, unsigned long long ld2, unsigned int bytes2, unsigned int rows2,
    const unsigned char* __restrict__ p3, unsigned long long ld3, unsigned int bytes3, unsigned int rows3,
    const unsigned char* __restrict__ p4, unsigned long long ld4, unsigned int bytes4, unsigned int rows4,
    const unsigned char* __restrict__ p5, unsigned long long ld5, unsigned int bytes5, unsigned int rows5,
    const unsigned char* __restrict__ p6, unsigned long long ld6, unsigned int bytes6, unsigned int rows6,
    const unsigned char* __restrict__ p7, unsigned long long ld7, unsigned int bytes7, unsigned int rows7,
    unsigned int mode
) {
    const L2aRegion r[L2A_REGIONS] = {L2A_REGION(0), L2A_REGION(1), L2A_REGION(2), L2A_REGION(3),
                                      L2A_REGION(4), L2A_REGION(5), L2A_REGION(6), L2A_REGION(7)};
    const unsigned int threads = gridDim.x * blockDim.x, t = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
    for (int q = 0; q < L2A_REGIONS; q++) l2a_region(r[q], mode, t, threads);
}
