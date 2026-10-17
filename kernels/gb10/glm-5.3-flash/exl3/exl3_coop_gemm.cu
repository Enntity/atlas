// SPDX-License-Identifier: AGPL-3.0-only
//
// Cooperative packed EXL3 GEMM (ATLAS_EXL3_COOP_GEMM=1): Atlas entry points
// around the vendored exllamav3 GEMM core (exllamav3_*.cuh, MIT, Copyright (c)
// 2025 Turboderp; see each header for its notice, upstream SHA256 and changes).
// Design and numerics argument: docs/design/exl3-coop-gemm-port.md.
//
// Pipeline (all on one stream, unchanged outer contract of
// Exl3LinearWorkspace): exl3_coop_input_hadamard_f16h -> exl3_coop_gemm_* ->
// Atlas exl3_output_hadamard[_f32]. The input transform is Atlas's
// exl3_input_hadamard_f16 arithmetic with the F16 value stored as __half (the
// existing kernel stores the same F16 value widened to F32). The GEMM writes
// unrounded F32 partials/results (c_fp32 = true) where exl3_mul1_matmul_mma
// writes its F32 output.
#include "exl3_common.cuh"
#include "exl3_expert.cuh"
#include "exllamav3_gemm_inner.cuh"

// Grid (K/128, rows, 1), block (128, 1, 1). Bit-for-bit the F16 value that
// exl3_input_hadamard_f16 stores (as F32) in its scratch, stored as __half.
extern "C" __global__ void exl3_coop_input_hadamard_f16h(
    const __nv_bfloat16* __restrict__ x,
    const __half* __restrict__ suh,
    __half* __restrict__ xh,
    unsigned int rows, unsigned int k
) {
    if (blockIdx.y >= rows || blockIdx.x >= k / 128u) return;
    __shared__ float scratch[128];
    const unsigned int feature = blockIdx.x * 128u + threadIdx.x;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.y) * k + feature;
    const __half product = __hmul(__float2half_rn(__bfloat162float(x[index])), suh[feature]);
    xh[index] = __float2half_rn(hadamard128(__half2float(product), scratch));
}

// Cooperative GEMM, grid (launched_slices, 1, 1), block (256 * TILESIZE_K / 16),
// dynamic shared memory SMEM_MAX, launched with cuLaunchCooperativeKernel.
//   A      [size_m, size_k] F16, row-major (input-transformed activations)
//   B      [size_k/16, size_n/16, 16*bits] EXL3 trellis (uint16)
//   C      [size_m, size_n] F32, row-major; need not be initialized
//   locks  >= ceil(size_m/16) * (size_n/128) zero-initialized ints; every
//          launch leaves them zero again (upstream self-resetting protocol)
//   logical_tiles_n, col_tile_offset, num_slices: the reference split-K
//          partition (logical width / tile_n, logical_offset / tile_n, slices)
//   slice_base: first slice index this grid covers (CTAs whose slice cannot
//          intersect this component's window are not launched)
//
// Rows are processed in 16-row chunks exactly like upstream exl3_gemm_kernel.
// Upstream separates chunks with grid.sync() because it reuses one lock row;
// here each chunk owns its own lock row (locks + chunk * size_n/128), so no
// grid-wide barrier is needed and the arithmetic per chunk is unchanged.
template<EXL3_GEMM_T_ARGS, bool fused_input = false, bool fused_output = false>
__device__ __forceinline__ void exl3_coop_gemm_body(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const int logical_tiles_n,
    const int col_tile_offset,
    const int num_slices,
    const int slice_base,
    // fused_input: A is the raw BF16 activations and suh its input scales
    const half* __restrict__ suh = nullptr,
    // fused_output: BF16 output [size_m, size_n] and output scales; C unused
    __nv_bfloat16* __restrict__ out = nullptr,
    const half* __restrict__ svh = nullptr
) {
    const int slice_index = slice_base + (int) blockIdx.x;
    const int lock_stride = size_n / 128;
    int size_m_ = size_m;
    int chunk = 0;
    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES,
                               false, false, false, false, false, fused_input, fused_input, fused_output>
        (
            fused_input
                ? (const void*) (((const __nv_bfloat16*) A) + (long long) chunk * 16 * size_k)
                : (const void*) (A + (long long) chunk * 16 * size_k),
            B,
            (void*) (C + (long long) chunk * 16 * size_n),
            MIN(size_m_, 16),
            size_k,
            size_n,
            locks + (long long) chunk * lock_stride,
            logical_tiles_n,
            col_tile_offset,
            num_slices,
            slice_index,
            Exl3v3Parts{},
            0,
            0x40000000,
            suh,
            fused_output ? out + (long long) chunk * 16 * size_n : nullptr,
            svh
        );
        size_m_ -= 16;
        chunk++;
        // Shared-memory stages are reused by the next chunk.
        __syncthreads();
    }
}

#define EXL3_COOP_GEMM_ARGS \
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C, \
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks, \
    const int logical_tiles_n, const int col_tile_offset, const int num_slices, const int slice_base

#define EXL3_COOP_GEMM_PASS \
    A, B, C, size_m, size_k, size_n, locks, logical_tiles_n, col_tile_offset, num_slices, slice_base

// mul1 codebook (cb = 2), F32 output. Shapes 2, 3, 4 of the upstream kernel map
// are the ones Atlas's pinned GLM-5.3 profiles select; bits 5 and 6 are the
// checkpoint's bitrates.
#define EXL3_COOP_GEMM_INSTANCE(_bits, _tk, _tn, _shape) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * _tk / 16) \
    void exl3_coop_gemm_b##_bits##_k##_tk##_n##_tn(EXL3_COOP_GEMM_ARGS) \
    { \
        exl3_coop_gemm_body<_bits, true, 2, _shape>(EXL3_COOP_GEMM_PASS); \
    }

EXL3_COOP_GEMM_INSTANCE(5, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMM_INSTANCE(5, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMM_INSTANCE(5, 16, 512, EXL3_GEMM_SHAPE_4)
EXL3_COOP_GEMM_INSTANCE(6, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMM_INSTANCE(6, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMM_INSTANCE(6, 16, 512, EXL3_GEMM_SHAPE_4)

// Fused input transform (ATLAS_EXL3_COOP_FUSED_INPUT=1): same GEMM, A is the
// raw BF16 activations [size_m, size_k] and the transform (inner header note
// 13) is computed into a staged F16 row in the prologue of every CTA, so the
// separate exl3_coop_input_hadamard_f16h launch is not needed. The host uses
// it only when size_m * size_k F16 fit after the reduce buffer in SMEM_MAX.
#define EXL3_COOP_GEMMH_INSTANCE(_bits, _tk, _tn, _shape) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * _tk / 16) \
    void exl3_coop_gemmh_b##_bits##_k##_tk##_n##_tn(EXL3_COOP_GEMM_ARGS, const half* __restrict__ suh) \
    { \
        exl3_coop_gemm_body<_bits, true, 2, _shape, true>(EXL3_COOP_GEMM_PASS, suh); \
    }

EXL3_COOP_GEMMH_INSTANCE(5, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMMH_INSTANCE(5, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMMH_INSTANCE(5, 16, 512, EXL3_GEMM_SHAPE_4)
EXL3_COOP_GEMMH_INSTANCE(6, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMMH_INSTANCE(6, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMMH_INSTANCE(6, 16, 512, EXL3_GEMM_SHAPE_4)

// Fused input AND output transforms (ATLAS_EXL3_COOP_FUSED_OUTPUT=1, with
// fused input): BF16 output written by each column tile's last chunk (inner
// header note 14); no F32 output scratch, no output-transform launch.
#define EXL3_COOP_GEMMF_INSTANCE(_bits, _tk, _tn, _shape) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * _tk / 16) \
    void exl3_coop_gemmf_b##_bits##_k##_tk##_n##_tn(EXL3_COOP_GEMM_ARGS, const half* __restrict__ suh, \
        __nv_bfloat16* __restrict__ out, const half* __restrict__ svh) \
    { \
        exl3_coop_gemm_body<_bits, true, 2, _shape, true, true>(EXL3_COOP_GEMM_PASS, suh, out, svh); \
    }

EXL3_COOP_GEMMF_INSTANCE(5, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMMF_INSTANCE(5, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMMF_INSTANCE(5, 16, 512, EXL3_GEMM_SHAPE_4)
EXL3_COOP_GEMMF_INSTANCE(6, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMMF_INSTANCE(6, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMMF_INSTANCE(6, 16, 512, EXL3_GEMM_SHAPE_4)

// Routed-MoE thin experts (ATLAS_EXL3_COOP_MOE=1). Replaces
// exl3_moe_mul1_matmul_precision_mma[_split] for launches whose every route is
// thin (rows <= thin_cap): per slot (row, expert) an M=1 GEMM with Atlas's thin
// profile tile (32, 256), `num_slices` split-K slices (8) and the running fold
// total rounded to F16 after every add (half_partial). A operand is the F32
// per-slot scratch (F16-representable; rounded to F16 exactly as the MMA
// kernel's pair_bits does). Output is the per-slot F32 row the existing output
// Hadamard / combine consume; no atomics.
//
// Work items are (slot, slice) pairs, ordered slot-major with slices in
// DESCENDING order: item = slot * num_slices + (num_slices - 1 - slice). A
// slice's fold for a column waits (lock-ordered) only on higher slices of the
// same slot, i.e. on lower item indices. CTA b takes items b, b + gridDim.x,
// ... in increasing order, so the lowest unfinished item's owner is always
// executing it and every item it waits on is already finished: no deadlock as
// long as the grid is co-resident (cooperative launch). Which CTA runs a slice
// does not affect arithmetic: each item is exactly one reference slice.
// Locks: one row of n/128 ints per slot (zeroed, self-resetting).
// Grid(min(items, SMs),1,1), block(512,1,1), dynamic smem SMEM_MAX.
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * 32 / 16)
void exl3_coop_moe_thin_b4_k32_n256(
    const float* __restrict__ xh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table,
    float* __restrict__ yh,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    const int items = slots * num_slices;
    const int tiles_n = size_n / 256;
    const int lock_stride = size_n / 128;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int slot = item / num_slices;
        const int slice = num_slices - 1 - item % num_slices;
        const int expert = ids[slot];
        float* y = yh + (long long) slot * size_n;
        const bool valid = expert >= 0 && expert < experts;
        const Exl3Expert* descriptor = valid ? table + expert : nullptr;
        if (!valid || descriptor->bits != 4u || counts[expert] > thin_cap)
        {
            // Same poison the existing MMA kernel writes for an unusable slot.
            if (slice == 0)
                for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                    y[i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_GEMM_SHAPE_3, true, true>
        (
            xh + (long long) slot * size_k,
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            1,
            size_k,
            size_n,
            locks + (long long) slot * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice
        );
        // Shared-memory stages are reused by the next item.
        __syncthreads();
    }
}

// Fused q|k|v (ATLAS_EXL3_COOP_FUSED_QKV=1): one launch over the reference's
// fused width (3 * comp_n columns, logical_n), with each component's trellis
// and F32 output kept in its own allocation (inner header note 8). The grid is
// the profile's full `num_slices`; every slice is a reference slice over the
// fused matrix, so each column's partition and fold are those the per-component
// launches (exl3_coop_gemm_* with col_tile_offset) already reproduce.
// Rows are processed in 16-row passes with one lock row of 3*comp_n/128 ints
// per pass. Grid (num_slices,1,1), block (256*TILESIZE_K/16), smem SMEM_MAX.
#define EXL3_COOP_GEMM3_INSTANCE(_bits, _tk, _tn, _shape) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * _tk / 16) \
    void exl3_coop_gemm3_b##_bits##_k##_tk##_n##_tn( \
        const half* __restrict__ A, \
        const uint16_t* B0, const uint16_t* B1, const uint16_t* B2, \
        float* C0, float* C1, float* C2, \
        const int size_m, const int size_k, const int comp_n, \
        int* __restrict__ locks, const int num_slices) \
    { \
        Exl3v3Parts parts; \
        parts.b[0] = B0; parts.b[1] = B1; parts.b[2] = B2; \
        parts.c[0] = (void*) C0; parts.c[1] = (void*) C1; parts.c[2] = (void*) C2; \
        parts.a[0] = nullptr; parts.a[1] = nullptr; parts.a[2] = nullptr; \
        parts.count = 3; \
        const int logical_tiles_n = 3 * (comp_n / _tn); \
        const int lock_stride = 3 * comp_n / 128; \
        int size_m_ = size_m; \
        int chunk = 0; \
        while (size_m_ > 0) \
        { \
            Exl3v3Parts chunk_parts = parts; \
            for (int i = 0; i < 3; ++i) \
                chunk_parts.c[i] = (void*) (((float*) parts.c[i]) + (long long) chunk * 16 * comp_n); \
            exl3_gemm_kernel_inner<_bits, true, 2, _shape, false, false, true> \
            ( \
                A + (long long) chunk * 16 * size_k, nullptr, nullptr, \
                MIN(size_m_, 16), size_k, comp_n, \
                locks + (long long) chunk * lock_stride, \
                logical_tiles_n, 0, num_slices, (int) blockIdx.x, chunk_parts \
            ); \
            size_m_ -= 16; \
            chunk++; \
            __syncthreads(); \
        } \
    }

EXL3_COOP_GEMM3_INSTANCE(6, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMM3_INSTANCE(6, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMM3_INSTANCE(6, 16, 512, EXL3_GEMM_SHAPE_4)

// gemm3 with the fused input transform (A = raw BF16 input; one transform for
// all three components, as in the unfused path).
#define EXL3_COOP_GEMM3H_INSTANCE(_bits, _tk, _tn, _shape) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * _tk / 16) \
    void exl3_coop_gemm3h_b##_bits##_k##_tk##_n##_tn( \
        const half* __restrict__ A, \
        const uint16_t* B0, const uint16_t* B1, const uint16_t* B2, \
        float* C0, float* C1, float* C2, \
        const int size_m, const int size_k, const int comp_n, \
        int* __restrict__ locks, const int num_slices, const half* __restrict__ suh) \
    { \
        Exl3v3Parts parts; \
        parts.b[0] = B0; parts.b[1] = B1; parts.b[2] = B2; \
        parts.c[0] = (void*) C0; parts.c[1] = (void*) C1; parts.c[2] = (void*) C2; \
        parts.a[0] = nullptr; parts.a[1] = nullptr; parts.a[2] = nullptr; \
        parts.count = 3; \
        const int logical_tiles_n = 3 * (comp_n / _tn); \
        const int lock_stride = 3 * comp_n / 128; \
        int size_m_ = size_m; \
        int chunk = 0; \
        while (size_m_ > 0) \
        { \
            Exl3v3Parts chunk_parts = parts; \
            for (int i = 0; i < 3; ++i) \
                chunk_parts.c[i] = (void*) (((float*) parts.c[i]) + (long long) chunk * 16 * comp_n); \
            exl3_gemm_kernel_inner<_bits, true, 2, _shape, false, false, true, false, false, true, true> \
            ( \
                (const void*) (((const __nv_bfloat16*) A) + (long long) chunk * 16 * size_k), nullptr, nullptr, \
                MIN(size_m_, 16), size_k, comp_n, \
                locks + (long long) chunk * lock_stride, \
                logical_tiles_n, 0, num_slices, (int) blockIdx.x, chunk_parts, 0, 0x40000000, suh \
            ); \
            size_m_ -= 16; \
            chunk++; \
            __syncthreads(); \
        } \
    }

EXL3_COOP_GEMM3H_INSTANCE(6, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMM3H_INSTANCE(6, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMM3H_INSTANCE(6, 16, 512, EXL3_GEMM_SHAPE_4)

// Two-component fused launch with per-component input (inner header note 11):
// the packed shared expert's gate|up (ATLAS_EXL3_SHARED_PACKED=1), whose two
// tensors have their own input scales. Same scheduling as gemm3 over the
// fused width 2 * comp_n; each component reads its own F16 A.
#define EXL3_COOP_GEMM2_INSTANCE(_bits, _tk, _tn, _shape) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * _tk / 16) \
    void exl3_coop_gemm2_b##_bits##_k##_tk##_n##_tn( \
        const half* A0, const half* A1, \
        const uint16_t* B0, const uint16_t* B1, \
        float* C0, float* C1, \
        const int size_m, const int size_k, const int comp_n, \
        int* __restrict__ locks, const int num_slices) \
    { \
        const int logical_tiles_n = 2 * (comp_n / _tn); \
        const int lock_stride = 2 * comp_n / 128; \
        int size_m_ = size_m; \
        int chunk = 0; \
        while (size_m_ > 0) \
        { \
            Exl3v3Parts chunk_parts; \
            chunk_parts.b[0] = B0; chunk_parts.b[1] = B1; chunk_parts.b[2] = nullptr; \
            chunk_parts.c[0] = (void*) (C0 + (long long) chunk * 16 * comp_n); \
            chunk_parts.c[1] = (void*) (C1 + (long long) chunk * 16 * comp_n); \
            chunk_parts.c[2] = nullptr; \
            chunk_parts.a[0] = (const void*) (A0 + (long long) chunk * 16 * size_k); \
            chunk_parts.a[1] = (const void*) (A1 + (long long) chunk * 16 * size_k); \
            chunk_parts.a[2] = nullptr; \
            chunk_parts.count = 2; \
            exl3_gemm_kernel_inner<_bits, true, 2, _shape, false, false, true, false, true> \
            ( \
                nullptr, nullptr, nullptr, \
                MIN(size_m_, 16), size_k, comp_n, \
                locks + (long long) chunk * lock_stride, \
                logical_tiles_n, 0, num_slices, (int) blockIdx.x, chunk_parts \
            ); \
            size_m_ -= 16; \
            chunk++; \
            __syncthreads(); \
        } \
    }

EXL3_COOP_GEMM2_INSTANCE(5, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMM2_INSTANCE(6, 32, 128, EXL3_GEMM_SHAPE_2)
EXL3_COOP_GEMM2_INSTANCE(5, 32, 256, EXL3_GEMM_SHAPE_3)
EXL3_COOP_GEMM2_INSTANCE(6, 32, 256, EXL3_GEMM_SHAPE_3)

// Routed-MoE thin experts, v2 (ATLAS_EXL3_COOP_MOE_V2=1). Same per-slot
// arithmetic as exl3_coop_moe_thin_b4_k32_n256, finer and fused scheduling:
//  - `projections` (1 or 2) matrices per launch sharing ids/counts/geometry
//    (gate and up together: same slots, separate input/output scratch);
//  - work items are single (slice, column) chunks (inner header note 9), so a
//    slice spanning two columns (down) becomes two items.
// Item order: (projection, slot)-major; within a slot, slices descending then
// columns ascending. A chunk's lock wait depends only on higher slices of the
// same column of the same (projection, slot), i.e. on lower item indices;
// CTA b runs items b, b + gridDim.x, ... in order, so (as in v1) the lowest
// unfinished item's owner is always executing it: no deadlock on a
// co-resident grid. Locks: one row of size_n/128 ints per (projection, slot).
// Grid(min(items, SMs),1,1), block(512,1,1), dynamic smem SMEM_MAX.
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * 32 / 16)
void exl3_coop_moe2_thin_b4_k32_n256(
    const float* __restrict__ xh0,
    const float* __restrict__ xh1,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const long long total = (long long) tiles_k * tiles_n;
    // Chunks per slot: every slice contributes one chunk per column it touches.
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int lock_stride = size_n / 128;
    const int items = projections * slots * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int matrix = item / chunks;          // projection * slots + slot
        const int projection = matrix / slots;
        const int slot = matrix - projection * slots;
        // Locate chunk j: slices descending, columns ascending within a slice.
        int j = item - matrix * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = ids[slot];
        float* y = (projection ? yh1 : yh0) + (long long) slot * size_n;
        const Exl3Expert* table = projection ? table1 : table0;
        const bool valid = expert >= 0 && expert < experts;
        const Exl3Expert* descriptor = valid ? table + expert : nullptr;
        if (!valid || descriptor->bits != 4u || counts[expert] > thin_cap)
        {
            // Same poison as v1 / the existing MMA kernel, written once per row.
            if (item - matrix * chunks == 0)
                for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                    y[i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_GEMM_SHAPE_3, true, true>
        (
            (projection ? xh1 : xh0) + (long long) slot * size_k,
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            1,
            size_k,
            size_n,
            locks + (long long) matrix * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1
        );
        // Shared-memory stages are reused by the next item.
        __syncthreads();
    }
}

// Routed-MoE thin experts, v3 (ATLAS_EXL3_COOP_MOE_V3=1): v2's items and
// order with the both_sub thread layout (inner header note 10): 256 threads
// per CTA, each running both k16 sub-blocks, so two CTAs fit per SM. Tile
// 32x256 (the thin profile's), SH_STAGES/FRAG_STAGES are pipelining depths
// only. Dynamic shared memory is the stage buffers only (no reduce buffer):
// SH_STAGES * (1024 + 4096) bytes for bits 4. The host sizes the grid from
// cuOccupancyMaxActiveBlocksPerMultiprocessor * SMs.
// Grid(min(items, occupancy * SMs),1,1), block(256,1,1).
#define EXL3_COOP_MOE3_MIN_BLOCKS 2
#define EXL3_COOP_MOE3_SHAPE 16, 32, 256, 6, 3
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS, EXL3_COOP_MOE3_MIN_BLOCKS)
void exl3_coop_moe3_thin_b4_k32_n256(
    const float* __restrict__ xh0,
    const float* __restrict__ xh1,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const long long total = (long long) tiles_k * tiles_n;
    // Chunks per slot: every slice contributes one chunk per column it touches.
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int lock_stride = size_n / 128;
    const int items = projections * slots * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int matrix = item / chunks;          // projection * slots + slot
        const int projection = matrix / slots;
        const int slot = matrix - projection * slots;
        // Locate chunk j: slices descending, columns ascending within a slice.
        int j = item - matrix * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = ids[slot];
        float* y = (projection ? yh1 : yh0) + (long long) slot * size_n;
        const Exl3Expert* table = projection ? table1 : table0;
        const bool valid = expert >= 0 && expert < experts;
        const Exl3Expert* descriptor = valid ? table + expert : nullptr;
        if (!valid || descriptor->bits != 4u || counts[expert] > thin_cap)
        {
            // Same poison as v1 / the existing MMA kernel, written once per row.
            if (item - matrix * chunks == 0)
                for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                    y[i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_COOP_MOE3_SHAPE, true, true, false, true>
        (
            (projection ? xh1 : xh0) + (long long) slot * size_k,
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            1,
            size_k,
            size_n,
            locks + (long long) matrix * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1
        );
        // Shared-memory stages are reused by the next item.
        __syncthreads();
    }
}

// Routed-MoE thin experts, v6 (ATLAS_EXL3_COOP_MOE_V6=1): v3 (items, order,
// both_sub layout, 2 CTAs/SM) with the A row staged once per item in shared
// memory (inner header note 12). Dynamic smem: v3's stages + size_k F16
// (<= 8192 B for K <= 4096). Arithmetic identical to v1-v3.
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS, EXL3_COOP_MOE3_MIN_BLOCKS)
void exl3_coop_moe6_thin_b4_k32_n256(
    const float* __restrict__ xh0,
    const float* __restrict__ xh1,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const long long total = (long long) tiles_k * tiles_n;
    // Chunks per slot: every slice contributes one chunk per column it touches.
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int lock_stride = size_n / 128;
    const int items = projections * slots * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int matrix = item / chunks;          // projection * slots + slot
        const int projection = matrix / slots;
        const int slot = matrix - projection * slots;
        // Locate chunk j: slices descending, columns ascending within a slice.
        int j = item - matrix * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = ids[slot];
        float* y = (projection ? yh1 : yh0) + (long long) slot * size_n;
        const Exl3Expert* table = projection ? table1 : table0;
        const bool valid = expert >= 0 && expert < experts;
        const Exl3Expert* descriptor = valid ? table + expert : nullptr;
        if (!valid || descriptor->bits != 4u || counts[expert] > thin_cap)
        {
            // Same poison as v1 / the existing MMA kernel, written once per row.
            if (item - matrix * chunks == 0)
                for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                    y[i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_COOP_MOE3_SHAPE, true, true, false, true, false, true>
        (
            (projection ? xh1 : xh0) + (long long) slot * size_k,
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            1,
            size_k,
            size_n,
            locks + (long long) matrix * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1
        );
        // Shared-memory stages are reused by the next item.
        __syncthreads();
    }
}

// Routed-MoE thin experts, v7 (ATLAS_EXL3_COOP_MOE_V7=1), gate|up only: v6
// with the input transform fused into the A staging (inner header note 13):
// A is the raw BF16 layer input row of the slot's token (slot / topk) and the
// expert's input scales, so exl3_moe_input_hadamard_f16 is not launched.
// Identical F16 A values, identical arithmetic otherwise. Down keeps v6.
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS, EXL3_COOP_MOE3_MIN_BLOCKS)
void exl3_coop_moe7_thin_b4_k32_n256(
    const __nv_bfloat16* __restrict__ x,
    const int topk,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const long long total = (long long) tiles_k * tiles_n;
    // Chunks per slot: every slice contributes one chunk per column it touches.
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int lock_stride = size_n / 128;
    const int items = projections * slots * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int matrix = item / chunks;          // projection * slots + slot
        const int projection = matrix / slots;
        const int slot = matrix - projection * slots;
        // Locate chunk j: slices descending, columns ascending within a slice.
        int j = item - matrix * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = ids[slot];
        float* y = (projection ? yh1 : yh0) + (long long) slot * size_n;
        const Exl3Expert* table = projection ? table1 : table0;
        const bool valid = expert >= 0 && expert < experts;
        const Exl3Expert* descriptor = valid ? table + expert : nullptr;
        if (!valid || descriptor->bits != 4u || counts[expert] > thin_cap)
        {
            // Same poison as v1 / the existing MMA kernel, written once per row.
            if (item - matrix * chunks == 0)
                for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                    y[i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_COOP_MOE3_SHAPE, true, true, false, true, false, true, true>
        (
            (const void*) (x + (long long) (slot / topk) * size_k),
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            1,
            size_k,
            size_n,
            locks + (long long) matrix * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1,
            (const half*) descriptor->suh
        );
        // Shared-memory stages are reused by the next item.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Multi-row routed MoE (ATLAS_EXL3_COOP_MOE_MULTIROW=1; speculative verify,
// 2 <= rows <= thin_cap). Slots are grouped by expert so each selected
// expert's weights are streamed once for all its rows (inner header note 15).
// Per-row arithmetic is the thin profile's, exactly as v1-v7: MMA rows are
// independent and the split-K partition depends only on the matrix shape.
// ---------------------------------------------------------------------------

// One block of 256 threads. groups layout (ints): [0] = group count G,
// then expert[cap], start[cap], count[cap], slot[cap] with cap = slots.
// Groups are in ascending expert order; slots ascending within an expert
// (deterministic). Out-of-range ids form no group (top-k ids are valid).
extern "C" __global__ void exl3_moe_group_slots(
    const int* __restrict__ ids, int* __restrict__ groups,
    const int slots, const int experts)
{
    __shared__ int count_s[512];
    __shared__ int gidx_s[512];
    __shared__ int start_s[512];
    if (experts > 512) return;
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
    {
        int c = 0;
        for (int s = 0; s < slots; ++s) c += ids[s] == e;
        count_s[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int g = 0, start = 0;
        for (int e = 0; e < experts; ++e)
        {
            gidx_s[e] = g; start_s[e] = start;
            if (count_s[e]) { ++g; start += count_s[e]; }
        }
        groups[0] = g;
    }
    __syncthreads();
    int* g_expert = groups + 1;
    int* g_start = g_expert + slots;
    int* g_count = g_start + slots;
    int* g_slot = g_count + slots;
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
    {
        if (!count_s[e]) continue;
        const int g = gidx_s[e];
        g_expert[g] = e; g_start[g] = start_s[e]; g_count[g] = count_s[e];
        int j = start_s[e];
        for (int s = 0; s < slots; ++s) if (ids[s] == e) g_slot[j++] = s;
    }
}

// M-tile units for the grouped multi-row routed MoE path
// (ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1). Same slot/expert/start/count arrays as
// exl3_moe_group_slots, but every expert's slots are split into
// ceil(count / m_tile) units of at most m_tile rows (the vendored inner's
// TILESIZE_M) and the units are partitioned by accumulation profile: experts
// whose count is <= thin_cap into the thin region, the rest into the fat one.
// The thin region is exactly what exl3_coop_moe8_multirow_b4_k32_n256 already
// consumes (per-unit count <= thin_cap, rows <= 16, so neither poison fires);
// the fat region is exl3_coop_moe_fat_multirow_b4_k16_n128. Each exported unit
// therefore reproduces, per row, the value
// exl3_moe_mul1_matmul_precision_mma computes for that expert's count.
// One block of 256 threads; O(experts * slots), like exl3_moe_group_slots.
//
// Layout: two regions of `1 + 4 * slots` ints, thin first then fat. Within a
// region: [0] = unit count G, expert[G], start[G] (index into the region's slot
// list), rows[G], then the region's slot list (its experts' slots, expert
// ascending then slot ascending, each expert's units consecutive).
extern "C" __global__ void exl3_moe_group_units(
    const int* __restrict__ ids, int* __restrict__ out,
    const int slots, const int experts, const unsigned int thin_cap,
    const int m_tile)
{
    __shared__ int count_s[512];
    __shared__ int tu_s[512];
    __shared__ int fu_s[512];
    __shared__ int tgidx_s[512];
    __shared__ int tstart_s[512];
    __shared__ int fgidx_s[512];
    __shared__ int fstart_s[512];
    if (experts > 512) return;
    const int region = 1 + 4 * slots;
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
    {
        int c = 0;
        for (int s = 0; s < slots; ++s) c += ids[s] == e;
        count_s[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int gt = 0, gf = 0, st = 0, sf = 0;
        for (int e = 0; e < experts; ++e)
        {
            const int c = count_s[e];
            if (!c) { tu_s[e] = 0; fu_s[e] = 0; continue; }
            const int units = (c + m_tile - 1) / m_tile;
            if (static_cast<unsigned int>(c) <= thin_cap)
            {
                tgidx_s[e] = gt; tstart_s[e] = st; tu_s[e] = units; fu_s[e] = 0;
                gt += units; st += c;
            }
            else
            {
                fgidx_s[e] = gf; fstart_s[e] = sf; fu_s[e] = units; tu_s[e] = 0;
                gf += units; sf += c;
            }
        }
        out[0] = gt;
        out[region] = gf;
    }
    __syncthreads();
    int* t_expert = out + 1;
    int* t_start = t_expert + slots;
    int* t_rows = t_start + slots;
    int* t_slot = t_rows + slots;
    int* f_expert = out + region + 1;
    int* f_start = f_expert + slots;
    int* f_rows = f_start + slots;
    int* f_slot = f_rows + slots;
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
    {
        const int c = count_s[e];
        if (!c) continue;
        const bool thin = static_cast<unsigned int>(c) <= thin_cap;
        int* g_expert = thin ? t_expert : f_expert;
        int* g_start = thin ? t_start : f_start;
        int* g_rows = thin ? t_rows : f_rows;
        int* g_slot = thin ? t_slot : f_slot;
        const int gidx = thin ? tgidx_s[e] : fgidx_s[e];
        const int start = thin ? tstart_s[e] : fstart_s[e];
        const int units = thin ? tu_s[e] : fu_s[e];
        for (int k = 0; k < units; ++k)
        {
            const int left = c - k * m_tile;
            g_expert[gidx + k] = e;
            g_start[gidx + k] = start + k * m_tile;
            g_rows[gidx + k] = left < m_tile ? left : m_tile;
        }
        int j = start;
        for (int s = 0; s < slots; ++s) if (ids[s] == e) g_slot[j++] = s;
    }
}

// Grid (K/128, slots, 1), block (128,1,1). exl3_moe_input_hadamard_f16's
// arithmetic per slot (expanded = 0: input row slot / topk), F16 output.
extern "C" __global__ void exl3_moe_input_hadamard_f16h(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table, __half* __restrict__ xh,
    const unsigned int rows, const unsigned int k, const unsigned int topk,
    const unsigned int experts)
{
    const unsigned int slot = blockIdx.y, feature = blockIdx.x * 128 + threadIdx.x;
    if (slot >= rows * topk || feature >= k) return;
    const unsigned long long out_index = (unsigned long long) slot * k + feature;
    const int expert = ids[slot];
    if (expert < 0 || (unsigned int) expert >= experts) { xh[out_index] = __ushort_as_half(0x7e00); return; }
    const __half* suh = reinterpret_cast<const __half*>(table[expert].suh);
    const unsigned long long in_index = (unsigned long long) (slot / topk) * k + feature;
    const __half product = __hmul(__float2half_rn(__bfloat162float(x[in_index])), suh[feature]);
    __shared__ float scratch[128];
    xh[out_index] = __float2half_rn(hadamard128(__half2float(product), scratch));
}

// F16-representable F32 scratch -> F16 (exact). Grid(ceil(n/256)), block 256.
extern "C" __global__ void exl3_f32_to_f16(
    const float* __restrict__ in, __half* __restrict__ out, const unsigned long long n)
{
    const unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

// Two fragment stages (pipelining depth only) keep the gather variant within
// 128 registers without spills.
#ifndef EXL3_COOP_MOE8_FR
#define EXL3_COOP_MOE8_FR 1
#endif
#ifndef EXL3_COOP_MOE8_MIN_BLOCKS
#define EXL3_COOP_MOE8_MIN_BLOCKS 2
#endif
#define EXL3_COOP_MOE8_SHAPE 16, 32, 256, 6, EXL3_COOP_MOE8_FR
// Grid(occupancy * SMs,1,1) (fixed: the group count is on the device),
// block(256,1,1), smem = v3's (30720 B). Items: (projection, group)-major,
// slices descending, columns ascending; locks: one row of size_n/128 ints per
// (projection, group). A: per-slot F16 rows [slots, size_k]; C: per-slot F32
// rows [slots, size_n] (gathered through each group's slot list).
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS, EXL3_COOP_MOE8_MIN_BLOCKS)
void exl3_coop_moe8_multirow_b4_k32_n256(
    const half* __restrict__ xh0,
    const half* __restrict__ xh1,
    const int* __restrict__ groups,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const long long total = (long long) tiles_k * tiles_n;
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int n_groups = groups[0];
    const int* g_expert = groups + 1;
    const int* g_start = g_expert + slots;
    const int* g_count = g_start + slots;
    const int* g_slot = g_count + slots;
    const int lock_stride = size_n / 128;
    const int items = projections * n_groups * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int matrix = item / chunks;            // projection * n_groups + group
        const int projection = matrix / n_groups;
        const int group = matrix - projection * n_groups;
        int j = item - matrix * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = g_expert[group];
        const int rows = g_count[group];
        const int* row_slots = g_slot + g_start[group];
        float* y = projection ? yh1 : yh0;
        const Exl3Expert* descriptor = (projection ? table1 : table0) + expert;
        if (descriptor->bits != 4u || counts[expert] > thin_cap || rows > 16)
        {
            // Same poison as the per-slot kernels, for every row of the group.
            if (item - matrix * chunks == 0)
                for (int r = 0; r < rows; ++r)
                    for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                        y[(long long) row_slots[r] * size_n + i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_COOP_MOE8_SHAPE, false, true, false, true,
                               false, false, false, false, true>
        (
            (const void*) (projection ? xh1 : xh0),
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            rows,
            size_k,
            size_n,
            locks + (long long) matrix * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1,
            nullptr,
            nullptr,
            nullptr,
            row_slots
        );
        __syncthreads();
    }
}

// Routed-MoE FAT grouped multi-row
// (ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1): the exl3_coop_moe8_multirow schedule
// over the fat units exl3_moe_group_units exports (each <= 16 rows), with the
// fat accumulation profile of exl3_moe_mul1_matmul_precision_mma: tile
// (16, 128) x one split-K slice, running fold left unrounded F32
// (round_partial = false). A slot's row is therefore bit for bit the value the
// per-slot fat path writes for the same inputs. The M-tile chunking lives in
// the unit split, not here: a group with more rows than the M tile is several
// units, each an independent M-tile pass. The only geometry poison kept is the
// genuinely invalid one (a non-4-bit descriptor or a unit wider than the M
// tile), never an expert count.
#ifndef EXL3_COOP_MOE_FAT_SH
#define EXL3_COOP_MOE_FAT_SH 6
#endif
#ifndef EXL3_COOP_MOE_FAT_FR
#define EXL3_COOP_MOE_FAT_FR 1
#endif
#ifndef EXL3_COOP_MOE_FAT_MIN_BLOCKS
#define EXL3_COOP_MOE_FAT_MIN_BLOCKS 2
#endif
// Dynamic shared memory: SH stages of the A tile (16 * 16 halves) and the B
// tile (1 * 8 * 256/16 * 4 uint16). TILEBLOCKS_K == 1, so threadblock_reduce
// never touches the sh_c reduce buffer (same reason as the thin multirow
// kernel).
#define EXL3_COOP_MOE_FAT_SMEM \
    (EXL3_COOP_MOE_FAT_SH * (16 * 16 * 2 + 1 * 8 * 16 * 4 * 2))
#define EXL3_COOP_MOE_FAT_SHAPE 16, 16, 128, EXL3_COOP_MOE_FAT_SH, EXL3_COOP_MOE_FAT_FR

// Grid(occupancy * SMs,1,1) (fixed: the unit count is on the device),
// block(256,1,1), smem = EXL3_COOP_MOE_FAT_SMEM. Items: (projection,
// unit)-major, slices descending then columns ascending; locks: one row of
// size_n/128 ints per (projection, unit). A: per-slot F16 rows [slots,
// size_k]; C: per-slot F32 rows [slots, size_n] (gathered through each unit's
// slot list). `num_slices` is the fat profile's 1.
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS, EXL3_COOP_MOE_FAT_MIN_BLOCKS)
void exl3_coop_moe_fat_multirow_b4_k16_n128(
    const half* __restrict__ xh0,
    const half* __restrict__ xh1,
    const int* __restrict__ groups,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const int num_slices
) {
    const int tiles_k = size_k / 16;
    const int tiles_n = size_n / 128;
    const long long total = (long long) tiles_k * tiles_n;
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int n_groups = groups[0];
    const int* g_expert = groups + 1;
    const int* g_start = g_expert + slots;
    const int* g_count = g_start + slots;
    const int* g_slot = g_count + slots;
    const int lock_stride = size_n / 128;
    const int items = projections * n_groups * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int matrix = item / chunks;            // projection * n_groups + group
        const int projection = matrix / n_groups;
        const int group = matrix - projection * n_groups;
        int j = item - matrix * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = g_expert[group];
        const int rows = g_count[group];
        const int* row_slots = g_slot + g_start[group];
        float* y = projection ? yh1 : yh0;
        const Exl3Expert* descriptor = (projection ? table1 : table0) + expert;
        if (descriptor->bits != 4u || rows > 16 || expert >= experts)
        {
            // Same poison as the per-slot kernels, for every row of the unit.
            if (item - matrix * chunks == 0)
                for (int r = 0; r < rows; ++r)
                    for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                        y[(long long) row_slots[r] * size_n + i] = __int_as_float(0x7fc00000);
            continue;
        }
        exl3_gemm_kernel_inner<4, true, 2, EXL3_COOP_MOE_FAT_SHAPE, false, false, false, false,
                               false, false, false, false, true>
        (
            (const void*) (projection ? xh1 : xh0),
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            rows,
            size_k,
            size_n,
            locks + (long long) matrix * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1,
            nullptr,
            nullptr,
            nullptr,
            row_slots
        );
        __syncthreads();
    }
}

// Routed-MoE thin down projection, v9 (ATLAS_EXL3_COOP_MOE_V9=1): v6's
// items/layout with the A row computed from the slot's gate and up GEMM sums
// (inner header note 16), replacing the gate/up output transforms and the
// activation/down-input transform launches. Output: per-slot F32 rows in yh
// (must not alias the gate/up sums). Grid(min(items, occupancy * SMs)),
// block(256), smem = v6's (stages + size_k F16 row).
extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS, EXL3_COOP_MOE3_MIN_BLOCKS)
void exl3_coop_moe9_down_b4_k32_n256(
    const float* __restrict__ gate_yh,
    const float* __restrict__ up_yh,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ gate_table,
    const Exl3Expert* __restrict__ up_table,
    const Exl3Expert* __restrict__ down_table,
    float* __restrict__ yh,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices,
    const float limit,
    const unsigned int clamp
) {
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const long long total = (long long) tiles_k * tiles_n;
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int lock_stride = size_n / 128;
    const int items = slots * chunks;
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int slot = item / chunks;
        int j = item - slot * chunks;
        int slice = num_slices - 1;
        int column = 0;
        for (; slice >= 0; --slice)
        {
            const long long beg = total * slice / num_slices;
            const long long end = total * (slice + 1) / num_slices;
            if (end <= beg) continue;
            const int first = (int) (beg / tiles_k);
            const int count = (int) ((end - 1) / tiles_k) - first + 1;
            if (j < count) { column = first + j; break; }
            j -= count;
        }
        const int expert = ids[slot];
        float* y = yh + (long long) slot * size_n;
        const bool valid = expert >= 0 && expert < experts;
        const Exl3Expert* descriptor = valid ? down_table + expert : nullptr;
        if (!valid || descriptor->bits != 4u || counts[expert] > thin_cap)
        {
            if (item - slot * chunks == 0)
                for (int i = threadIdx.x; i < size_n; i += blockDim.x)
                    y[i] = __int_as_float(0x7fc00000);
            continue;
        }
        Exl3v3ActSource act;
        act.gate = gate_yh + (long long) slot * size_k;
        act.up = up_yh + (long long) slot * size_k;
        act.gate_svh = (const half*) gate_table[expert].svh;
        act.up_svh = (const half*) up_table[expert].svh;
        act.suh = (const half*) descriptor->suh;
        act.limit = limit;
        act.clamp = clamp;
        exl3_gemm_kernel_inner<4, true, 2, EXL3_COOP_MOE3_SHAPE, true, true, false, true,
                               false, true, false, false, false, true>
        (
            nullptr,
            (const uint16_t*) descriptor->trellis,
            (void*) y,
            1,
            size_k,
            size_n,
            locks + (long long) slot * lock_stride,
            tiles_n,
            0,
            num_slices,
            slice,
            Exl3v3Parts{},
            column,
            column + 1,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            act
        );
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Routed-MoE thin experts, v10 (ATLAS_EXL3_COOP_MOE_V10=1): a streaming
// persistent kernel. Each CTA owns items b, b + gridDim.x, ... (v2 item order:
// (projection, slot)-major, slices descending, columns ascending) and runs ONE
// cp.async pipeline over the concatenation of its items' k-tiles, so the next
// item's B/A tiles are already in flight while the current item finishes and
// folds. Per item the arithmetic is v2-v7's exactly: tile 32x256 (both k16
// sub-blocks per thread, separate accumulators, MMA chain over the item's
// k-tiles in ascending order), acc0 + acc1, +0.0 seed on the column's first
// chunk, lock-ordered fold (reverse k), F16 rounding of the running total.
// A: per-slot F16 rows (exl3_moe_input_hadamard_f16h); C: per-slot F32 rows.
// Grid(min(items, occupancy * SMs)), block(256), smem SH * 5 KB. Defaults
// SH 4 / 3 CTAs per SM (78 registers, no spills): bench gate|up 167 vs 186 us
// (v6), down 89 vs 103 us, bit-exact.
// ---------------------------------------------------------------------------
#ifndef EXL3_MOE10_SH
#define EXL3_MOE10_SH 4
#endif
#ifndef EXL3_MOE10_MIN_BLOCKS
#define EXL3_MOE10_MIN_BLOCKS 3
#endif

struct Exl3Moe10Item
{
    const half* a;            // slot's F16 A row
    const uint16_t* b;        // expert trellis
    float* c;                 // slot's F32 output row
    int* lock_row;            // (projection, slot) lock row
    int column, k0, tiles;    // column tile, first k-tile, k-tiles (0: skip)
    int poison;               // write NaN row (invalid route) when reached
};

__device__ __forceinline__ Exl3Moe10Item exl3_moe10_item(
    const int item, const int chunks, const long long total, const int tiles_k,
    const int num_slices, const int slots, const half* xh0, const half* xh1,
    const int* ids, const Exl3Expert* table0, const Exl3Expert* table1,
    float* yh0, float* yh1, const unsigned int* counts, int* locks,
    const int size_k, const int size_n, const int experts, const unsigned int thin_cap)
{
    Exl3Moe10Item r;
    const int matrix = item / chunks;
    const int projection = matrix / slots;
    const int slot = matrix - projection * slots;
    int j = item - matrix * chunks;
    const int first_chunk = j == 0;
    r.column = 0; r.k0 = 0; r.tiles = 0;
    for (int slice = num_slices - 1; slice >= 0; --slice)
    {
        const long long beg = total * slice / num_slices;
        const long long end = total * (slice + 1) / num_slices;
        if (end <= beg) continue;
        const int first = (int) (beg / tiles_k);
        const int count = (int) ((end - 1) / tiles_k) - first + 1;
        if (j < count)
        {
            r.column = first + j;
            const long long cb = (long long) r.column * tiles_k;
            r.k0 = (int) (MAX(beg, cb) - cb);
            r.tiles = (int) (MIN(end, cb + tiles_k) - cb) - r.k0;
            break;
        }
        j -= count;
    }
    const int expert = ids[slot];
    r.a = (projection ? xh1 : xh0) + (long long) slot * size_k;
    r.c = (projection ? yh1 : yh0) + (long long) slot * size_n;
    r.lock_row = locks + (long long) matrix * (size_n / 128);
    const bool valid = expert >= 0 && expert < experts;
    const Exl3Expert* d = valid ? (projection ? table1 : table0) + expert : nullptr;
    r.poison = 0;
    if (!valid || d->bits != 4u || counts[expert] > thin_cap)
    {
        r.tiles = 0;
        r.poison = first_chunk;
        r.b = nullptr;
    }
    else r.b = (const uint16_t*) d->trellis;
    return r;
}

extern "C" __global__ __launch_bounds__(256, EXL3_MOE10_MIN_BLOCKS)
void exl3_coop_moe10_thin_b4_k32_n256(
    const half* __restrict__ xh0,
    const half* __restrict__ xh1,
    const int* __restrict__ ids,
    const Exl3Expert* __restrict__ table0,
    const Exl3Expert* __restrict__ table1,
    float* __restrict__ yh0,
    float* __restrict__ yh1,
    const unsigned int* __restrict__ counts,
    int* __restrict__ locks,
    const int projections,
    const int slots,
    const int size_k,
    const int size_n,
    const int experts,
    const unsigned int thin_cap,
    const int num_slices
) {
    constexpr int SH = EXL3_MOE10_SH;
    constexpr int STAGE_A = 512;    // halves: 16 x 32 (row 0 used)
    constexpr int STAGE_B = 2048;   // u16: 2 sub-blocks x 16 n-blocks x 64
    extern __shared__ half moe10_shared[];
    half* sh_a = moe10_shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH * STAGE_A);

    const int t = threadIdx.x, warp_id = t / 32, lane_id = t % 32;
    const int tiles_k = size_k / 32;
    const int tiles_n = size_n / 256;
    const int blocks_n = tiles_n * 16;
    const long long total = (long long) tiles_k * tiles_n;
    int chunks = 0;
    for (int s = 0; s < num_slices; ++s)
    {
        const long long beg = total * s / num_slices;
        const long long end = total * (s + 1) / num_slices;
        if (end > beg) chunks += (int) ((end - 1) / tiles_k - beg / tiles_k + 1);
    }
    const int items = projections * slots * chunks;
    const int gl_b_stride_k = blocks_n * 2 * 64;   // u16 per k-tile row pair
    const int gl_b_stride_n = 16 * 64;             // u16 per 256-column tile
    // B tile load mapping (one int4 per thread): 2 rows of 128 int4.
    const int load_b_n = t % 128, load_b_k = t / 128;
    const int load_b_gl = load_b_k * (blocks_n * 64 / 8) + load_b_n;

    auto item_at = [&] (int item) {
        return exl3_moe10_item(item, chunks, total, tiles_k, num_slices, slots, xh0, xh1, ids,
                               table0, table1, yh0, yh1, counts, locks, size_k, size_n, experts, thin_cap);
    };

    // Producer cursor over the CTA's tile stream.
    int p_item = blockIdx.x;
    Exl3Moe10Item p_it = p_item < items ? item_at(p_item) : Exl3Moe10Item{};
    int p_tile = 0;
    auto p_skip_empty = [&] () {
        while (p_item < items && p_tile >= p_it.tiles)
        {
            p_item += gridDim.x; p_tile = 0;
            if (p_item < items) p_it = item_at(p_item);
        }
    };
    p_skip_empty();
    int p_count = 0;   // tiles issued
    auto issue = [&] () {
        if (p_item < items)
        {
            const int stage = p_count % SH;
            const int kt = p_it.k0 + p_tile;
            const int4* gb = (const int4*) (p_it.b + (long long) kt * gl_b_stride_k + (long long) p_it.column * gl_b_stride_n);
            cp_async((int4*) (sh_b + stage * STAGE_B) + t, gb + load_b_gl);
            if (t < 4) cp_async((int4*) (sh_a + stage * STAGE_A) + t, (const int4*) (p_it.a + kt * 32) + t);
            ++p_tile;
            p_skip_empty();
        }
        cp_async_fence();
        ++p_count;
    };

    // Consumer state.
    FragC acc0[4], acc1[4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) { acc0[n] = {}; acc1[n] = {}; }

    for (int i = 0; i < SH - 1; ++i) issue();

    int c_count = 0;
    for (int c_item = blockIdx.x; c_item < items; c_item += gridDim.x)
    {
        const Exl3Moe10Item it = item_at(c_item);
        if (it.poison)
            for (int i = t; i < size_n; i += 256) it.c[i] = __int_as_float(0x7fc00000);
        for (int tile = 0; tile < it.tiles; ++tile, ++c_count)
        {
            cp_async_wait<SH - 2>();
            __syncthreads();
            const int stage = c_count % SH;
            const half* sa = sh_a + stage * STAGE_A;
            const uint16_t* sb = sh_b + stage * STAGE_B;
            FragA fa0, fa1;
            {
                const int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
                const int base_c = lane_id / 16;
                ldsm4(fa0, (const int4*) sa + r * 4 + ((base_c) ^ ((r >> 1) & 3)));
                ldsm4(fa1, (const int4*) sa + r * 4 + ((base_c + 2) ^ ((r >> 1) & 3)));
            }
            FragB fb0[4], fb1[4];
            #pragma unroll
            for (int n2 = 0; n2 < 4; n2 += 2)
            {
                const int sub_n2 = warp_id * 2 + n2 / 2;
                dq_dispatch<4, 2>((const uint32_t*) (sb + (sub_n2) * 64), lane_id << 3, fb0[n2], fb0[n2 + 1]);
                dq_dispatch<4, 2>((const uint32_t*) (sb + (16 + sub_n2) * 64), lane_id << 3, fb1[n2], fb1[n2 + 1]);
            }
            #pragma unroll
            for (int n = 0; n < 4; ++n)
            {
                ptx_mma_m16n8k16(fa0, fb0[n], acc0[n]);
                ptx_mma_m16n8k16(fa1, fb1[n], acc1[n]);
            }
            __syncthreads();   // stage fully consumed before it is refilled
            issue();
        }
        if (it.tiles == 0) continue;
        // Item fold (identical to the inner's reduce for size_m = 1, both_sub).
        #pragma unroll
        for (int n = 0; n < 4; ++n)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc0[n][j] += acc1[n][j];
        }
        const int last_k = it.k0 + it.tiles - 1;
        const int lock_i = tiles_k - last_k - 1;
        const int lock_d = it.tiles;
        int* lock = it.lock_row + it.column;
        barrier_acquire(lock, lock_i);
        const bool first = lock_i == 0;
        const bool last = lock_i + lock_d == tiles_k;
        const int r0 = lane_id / 4, cc = (lane_id % 4) * 2, n0 = warp_id * 4;
        float* cbase = it.c + it.column * 256;
        if (r0 < 1)
        {
            #pragma unroll
            for (int n = 0; n < 4; ++n)
            {
                float* cp = cbase + (n0 + n) * 8 + cc;
                float v0 = acc0[n][0], v1 = acc0[n][1];
                if (first) { v0 = __fadd_rn(v0, 0.0f); v1 = __fadd_rn(v1, 0.0f); }
                else { v0 += cp[0]; v1 += cp[1]; }
                v0 = __half2float(__float2half_rn(v0));
                v1 = __half2float(__float2half_rn(v1));
                cp[0] = v0; cp[1] = v1;
            }
        }
        barrier_release(lock, lock_d, last);
        #pragma unroll
        for (int n = 0; n < 4; ++n) { acc0[n] = {}; acc1[n] = {}; }
    }
    cp_async_wait<0>();
}
