// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Programmatic dependent launch (PDL). A kernel launched with the runtime's
// programmatic-serialization attribute (ATLAS_PDL=1, kernels on the runtime's
// PDL list) may be scheduled before its stream predecessor finishes. Entry
// order: let our own successor be scheduled once all our blocks have started,
// then wait until the predecessor's writes are visible. Every block runs this
// before any other work (early exits included), so "kernel N complete"
// still implies "kernel N-1 complete". Without the attribute both are no-ops.
__device__ __forceinline__ void atlas_pdl_enter() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Weight touch before the wait. A kernel launched with PDL usually starts
// while its predecessors (small per-layer kernels) still run, and sits in
// `griddepcontrol.wait` with the memory bus idle. Its weights do not depend on
// the predecessor, so the first `ctas` CTAs (the ones certain to be resident)
// issue one discarded byte load per 32-byte sector of the first `rows` rows of
// each region, pulling them into L2 during the wait; the kernel body then reads
// the same bytes from L2. Loads only: no value reaches the kernel's arithmetic,
// so every output bit is unchanged. `ld` is the byte stride between rows.
// Without the PDL attribute the touch lands right before the body's own loads.
struct AtlasTouch {
    const unsigned char* p;
    unsigned int row_bytes;
    unsigned long long ld;
};

__device__ __forceinline__ void atlas_l2_touch(const unsigned char* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("{\n\t.reg .u32 t;\n\tld.volatile.global.u8 t, [%0];\n\t}" :: "l"(p) : "memory");
#endif
}

// `atlas_pdl_enter()` with the touch of two regions (weights, scales) between
// its halves. `cta` is this CTA's index in the grid.
__device__ __forceinline__ void atlas_pdl_enter_touch(
    const AtlasTouch a, const AtlasTouch b, unsigned int rows, unsigned int cta, unsigned int ctas
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (cta < ctas) {
        const AtlasTouch r[2] = {a, b};
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            const unsigned int per_row = (r[q].row_bytes + 31u) / 32u;
            const unsigned int total = rows * per_row;
            for (unsigned int i = cta * blockDim.x + threadIdx.x; i < total; i += ctas * blockDim.x)
                atlas_l2_touch(r[q].p + (unsigned long long)(i / per_row) * r[q].ld + (i % per_row) * 32u);
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
