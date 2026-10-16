// SPDX-License-Identifier: AGPL-3.0-only

// BF16 in-place vector addition: dst[i] += src[i]
// Used by 2-rank send/recv all-reduce in NcclBackend.

#include "atlas_pdl.cuh"
#include <cuda_bf16.h>

extern "C" __global__ void bf16_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    atlas_pdl_enter();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __hadd(dst[i], src[i]);
    }
}

// RDMA pair flag kernels (ATLAS_RDMA_KERNEL_FLAGS=1): replace the stream
// memory operations around a decode-sized exchange.
// Publish `value` to a host-mapped flag word after all prior stream work,
// with system-scope release semantics. Grid(1), block(1).
extern "C" __global__ void atlas_flag_store_release(
    unsigned long long* flag, unsigned long long value
) {
    __threadfence_system();
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(flag), "l"(value) : "memory");
}

// bf16_add_inplace that first waits (system-scope acquire) until the
// host-mapped `flag` reaches `expected`; `src` (the RDMA receive slot) is
// then read with cache-volatile loads. Same arithmetic as bf16_add_inplace.
extern "C" __global__ void bf16_add_inplace_wait(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n,
    const unsigned long long* flag,
    unsigned long long expected
) {
    if (threadIdx.x == 0) {
        unsigned long long seen;
        do {
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(seen) : "l"(flag) : "memory");
        } while (seen < expected);
    }
    __syncthreads();
    // Per-thread system-scope acquire orders this thread's src loads after the flag.
    asm volatile("{ .reg .u64 t; ld.acquire.sys.global.u64 t, [%0]; }" :: "l"(flag) : "memory");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned short bits = __ldcv(reinterpret_cast<const unsigned short*>(src) + i);
        dst[i] = __hadd(dst[i], __ushort_as_bfloat16(bits));
    }
}
