// SPDX-License-Identifier: MIT
//
// Vendored from exllamav3 (exllamav3_ext), adapted for Atlas. Original notice:
//
// MIT License
//
// Copyright (c) 2025 Turboderp
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Upstream: exllamav3_ext/quant/exl3_gemm_inner.cuh (sha256 46aefd5e631ef8bd21b816939d9ed0511decf9ae42241b6ad745a57042b84abf).
//
// Atlas adaptations (every change is marked "ATLAS:" below; everything else is
// upstream code, reformatted only where a removed branch left nothing behind):
//  1. Logical split-K window. Upstream partitions the flattened (k-tile, n-tile)
//     space of the matrix it is given over gridDim.x CTAs. Atlas reproduces a
//     reference that ran fused/wider matrices, so the partition is computed over
//     a LOGICAL width (logical_tiles_n tiles) with CTA index `slice_index` of
//     `num_slices`, then clamped to this physical component's column-tile window
//     [col_tile_offset, col_tile_offset + tiles_n). With logical == physical,
//     col_tile_offset == 0 and slice_index == blockIdx.x, num_slices ==
//     gridDim.x, this is exactly upstream's partition.
//  2. The shmem_out_had (fused output Hadamard) path is removed: Atlas keeps its
//     own output Hadamard kernel, whose rounding differs from had_ff_r_128_inner.
//     This is upstream's shmem_out_had == false instantiation (as used by
//     exl3_mgemm_kernel) with the post_scale argument dropped.
//  3. The sm_86-only FP16-accumulate path (EXL3_GEMM_H_ACC) is removed; upstream
//     compiles it out on every other architecture, including sm_121.
//  4. The first (bottom-most in k) partial of every column tile is added to +0.0f
//     before it is written, matching Atlas's fold `total = 0; total = sum + total`
//     (exl3_mma.cuh accumulate_logical). This only changes a -0.0 partial to
//     +0.0 and is otherwise the identity.
//  5. The deprecated `register` storage class is dropped (ill-formed in C++17),
//     unused locals are removed, and the A/B/C tile helpers keep upstream names.
//  6. Optional F32 A operand (a_fp32): the A tile is read as F32 with plain
//     loads, rounded to F16 (RN) and stored to the same swizzled shared-memory
//     slot the cp.async would fill. Atlas's routed-MoE scratch is F32 and its
//     MMA rounds it to F16 the same way (exl3_mma.cuh pair_bits).
//  7. Optional F16 rounding of the running fold total (round_partial), stored
//     in F32: after every fold add, total = f32(f16_rn(total)). This is Atlas's
//     `half_partial` (accumulate_logical) and the value sequence upstream's
//     c_fp32 == false path produces (half partials, float re-add).
//  8. Optional multi-component B/C (multi): a fused matrix of `parts.count`
//     equal-width components, each stored as its own [K/16, size_n/16, 16*bits]
//     trellis with its own [M, size_n] output, addressed as one matrix of
//     count * size_n columns. Column tile n lives in component n / tiles_n at
//     local tile n % tiles_n (size_n % TILESIZE_N == 0, so no tile straddles
//     components). The split-K partition, lock index and fold are those of the
//     fused width; only the B/C base pointer of a column tile changes.
//  9. Optional column restriction [col_lo, col_hi) (physical column tiles):
//     the CTA runs only the part of its slice that lies in those columns. A
//     slice's work in one column is one independent chunk (own MMA chains,
//     own reduce and lock step: lock_i/lock_d depend only on the chunk's k
//     range), so running a slice's column chunks separately is identical to
//     running the slice. Defaults (0, all) are the unrestricted partition.
// 10. Optional both_sub (TILESIZE_K == 32 only): a 256-thread CTA in which
//     every thread runs BOTH k16 sub-blocks of a tile, keeping two separate
//     accumulator sets (acc0 for sub-block 0, acc1 for sub-block 1) with the
//     same per-sub-block MMA chains as upstream's two 256-thread halves. The
//     threadblock reduce becomes the in-register `acc0 + acc1` (upstream:
//     sub-block 1 stores, sub-block 0 adds: the same F32 add). Loads are
//     already done by threads 0..255 only. Only the thread layout changes.
// 11. Optional per-component A (multi_a, with multi): component `p` reads its
//     own A matrix `parts.a[p]` (components with different input transforms,
//     e.g. separate gate/up tensors). The A tile pointer is re-based when a
//     CTA moves to a new column tile, exactly where B/C are.
// 12. Optional a_smem_row (F32 source, or with note 13): the F32 A rows
//     (size_m x size_k, rounded to F16 RN exactly as note 6) are converted
//     once per call into shared memory (both_sub kernels: their unused reduce
//     buffer; others: right after the reduce buffer, sized by the host), and
//     each pipeline stage copies its A tile from there.
//     Note 6's per-stage synchronous global loads otherwise stall the warp
//     that issues them, and with it the CTA at the next barrier, for one
//     global-load latency per k-tile. The F16 values are identical.
// 13. Optional a_had (with a_smem_row): the staged A rows are computed from
//     the raw BF16 activations with Atlas's input transform instead of read
//     from a precomputed scratch: per element f16(bf16->f32 x) * suh (one F16
//     multiply), then the 128-point butterfly with the same per-element op
//     sequence as exl3_common.cuh hadamard128 (stage d: element i with bit d
//     set takes partner - value, else value + partner, F32 RN), times
//     H128_NORM (F32 RN), rounded to F16 RN. Identical values to
//     exl3_moe_input_hadamard_f16 / exl3_input_hadamard_f16, so the separate
//     input-transform launch is not needed.
// 14. Optional out_had (single component, F32 accumulation): the CTA holding
//     a column tile's LAST chunk stages the final F32 sums of the tile in the
//     reduce buffer and applies Atlas's output transform per 128-column group
//     (same butterfly as note 13, x H128_NORM, x f32(svh), BF16 RN; the op
//     sequence of exl3.cu exl3_output_hadamard), writing BF16 output instead
//     of the F32 sums. Tile widths are whole 128-column groups.
// 15. Optional gather (multi-row MoE): logical row m of A and C is physical
//     row row_slots[m] (A row stride size_k, C row stride size_n), so one
//     expert's weights are streamed once for every slot that selected it.
//     MMA rows are independent, so each row's arithmetic is unchanged.
// 16. Optional a_act (with a_smem_row; routed-MoE down, thin): the staged A
//     row is computed from the gate and up GEMM sums of the slot with the
//     whole chain the separate kernels apply, per 128-feature group and in
//     their per-element op order: gate/up output transform (butterfly,
//     x H128_NORM, F16 product with svh), thin activation (exl3_moe_thin.cuh),
//     down input transform (F16 product with suh, butterfly, x H128_NORM,
//     F16 RN). Replaces exl3_moe_output_hadamard_precision (gate, up) and
//     exl3_moe_activate_down_hadamard.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "exllamav3_util.cuh"
#include "exllamav3_ptx.cuh"
#include "exllamav3_kernel_map.cuh"
#include "exllamav3_dq.cuh"
#include "exl3_moe_thin.cuh"  // ATLAS: header note 16 (Atlas, AGPL-3.0-only)

// ATLAS: notes 13/14. 128-point butterfly of one warp: element i = lane +
// 32 * j lives in v[j]; stage d combines i with i ^ d exactly as
// exl3_common.cuh hadamard128 (bit d set: partner - value, else value +
// partner, F32 RN). Returns before the H128_NORM scale.
__device__ __forceinline__ void exl3v3_h128_butterfly(float (&v)[4], const int lane)
{
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float other = __shfl_xor_sync(0xffffffffu, v[j], d);
            v[j] = (lane & d) ? __fsub_rn(other, v[j]) : __fadd_rn(v[j], other);
        }
    }
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = __fadd_rn(a0, a1); v[1] = __fsub_rn(a0, a1);
        v[2] = __fadd_rn(a2, a3); v[3] = __fsub_rn(a2, a3);
    }
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = __fadd_rn(a0, a2); v[2] = __fsub_rn(a0, a2);
        v[1] = __fadd_rn(a1, a3); v[3] = __fsub_rn(a1, a3);
    }
}
#define EXL3V3_H128_NORM 0.08838834613561630249f

template <bool a_fp32> struct exl3v3_a_elem { using type = half; };
template <> struct exl3v3_a_elem<true> { using type = float; };

// ATLAS: header note 8
#define EXL3V3_MAX_PARTS 3
// ATLAS: header note 16 (a_act only)
struct Exl3v3ActSource
{
    const float* gate;     // [size_m, size_k] F32 gate GEMM sums
    const float* up;       // [size_m, size_k] F32 up GEMM sums
    const half* gate_svh;  // gate output scales (size_k)
    const half* up_svh;    // up output scales (size_k)
    const half* suh;       // down input scales (size_k)
    float limit;
    unsigned int clamp;
};

struct Exl3v3Parts
{
    const uint16_t* b[EXL3V3_MAX_PARTS];
    void* c[EXL3V3_MAX_PARTS];
    const void* a[EXL3V3_MAX_PARTS];  // ATLAS: header note 11 (multi_a only)
    int count;
};

template<EXL3_GEMM_T_ARGS, bool a_fp32 = false, bool round_partial = false, bool multi = false,
         bool both_sub = false, bool multi_a = false, bool a_smem_row = false,
         bool a_had = false, bool out_had = false, bool gather = false, bool a_act = false>
inline __device__
void exl3_gemm_kernel_inner
(
    // ATLAS: F16 (upstream) or F32 (a_fp32, header note 6) row-major A
    const void* __restrict__ A_in,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    // ATLAS: logical split-K window (see header note 1)
    const int logical_tiles_n,
    const int col_tile_offset,
    const int num_slices,
    const int slice_index,
    // ATLAS: header note 8 (ignored unless multi)
    const Exl3v3Parts& parts = Exl3v3Parts{},
    // ATLAS: header note 9
    const int col_lo = 0,
    const int col_hi = 0x40000000,
    // ATLAS: header note 13 (a_had only): input scales, size_k halves
    const half* __restrict__ a_suh = nullptr,
    // ATLAS: header note 14 (out_had only): BF16 output [size_m, size_n] and
    // output scales (size_n halves)
    __nv_bfloat16* __restrict__ out_bf16 = nullptr,
    const half* __restrict__ out_svh = nullptr,
    // ATLAS: header note 15 (gather only): physical row of each logical row
    const int* __restrict__ row_slots = nullptr,
    // ATLAS: header note 16 (a_act only)
    const Exl3v3ActSource& act = Exl3v3ActSource{}
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    // const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halves
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    // ATLAS: shmem_out_had == false (header note 2)
    const int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP;          // in floats

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    // col_swizzled = col ^ ((row >> SHIFT) & MASK)
    const int A_COLS = TILESIZE_K / 8;                                            // int4 columns per row
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M == 16, "Invalid kernel params");                     // strictly assume size_m <= 16
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );

    // ATLAS: header note 6
    using AElem = typename exl3v3_a_elem<a_fp32>::type;
    const AElem* A = (const AElem*) A_in;

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);
    // ATLAS: header note 12 (both_sub never uses sh_c for its reduce)
    // (non-both_sub kernels: after the reduce buffer; the host checks that
    // size_m * size_k F16 fit in the dynamic shared memory it requests)
    static_assert(!a_smem_row || ((a_fp32 || a_had || a_act) && !multi_a), "a_smem_row needs an F32 or fused source");
    half* a_row = both_sub ? (half*) sh_c : (half*) (sh_c + sh_c_size);

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    // ATLAS: header note 10
    static_assert(!both_sub || TILEBLOCKS_K == 2, "both_sub requires TILESIZE_K == 32");
    int sub_k = both_sub ? 0 : threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    // Start and end index of current slice, must span at least one tile
    // ATLAS: upstream is
    //     int num_slices = gridDim.x;
    //     int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    //     int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    // Here the same partition runs over the logical flattened tile space and is
    // clamped to this component's window, then rebased to physical indices.
    const long long logical_tiles = (long long) tiles_k * logical_tiles_n;
    const long long window_beg = (long long) col_tile_offset * tiles_k;
    // ATLAS: header note 8, the window spans every component
    const int window_tiles_n = multi ? parts.count * tiles_n : tiles_n;
    const long long window_end = window_beg + (long long) window_tiles_n * tiles_k;
    const long long logical_beg = logical_tiles * slice_index / num_slices;
    const long long logical_end = logical_tiles * (slice_index + 1) / num_slices;
    // ATLAS: header note 9
    const long long clamp_beg = window_beg + (long long) MAX(col_lo, 0) * tiles_k;
    const long long clamp_end = MIN(window_end, window_beg + (long long) col_hi * tiles_k);
    int slice_beg = (int) (MAX(logical_beg, clamp_beg) - window_beg);
    int slice_end = (int) (MIN(logical_end, clamp_end) - window_beg);
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    // Batch dimension
    const int slice_m = 0;

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    // ATLAS: header note 11
    auto a_base = [&] (int tile_n) -> const AElem*
    {
        if constexpr (multi && multi_a) return (const AElem*) parts.a[tile_n / tiles_n];
        else return A;
    };
    const AElem* gl_a_ptr = a_base(slice0_n) + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        // ATLAS: header note 15
        load_a_gl[i] = (gather ? (m < size_m ? row_slots[m] : 0) : m) * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    // ATLAS: header note 8
    auto b_tile = [&] (int tile_k, int tile_n) -> const uint16_t*
    {
        if constexpr (multi)
        {
            const int part = tile_n / tiles_n;
            return parts.b[part] + tile_k * gl_b_stride_k + (tile_n - part * tiles_n) * gl_b_stride_n;
        }
        else
            return B + tile_k * gl_b_stride_k + tile_n * gl_b_stride_n;
    };
    const uint16_t* gl_b_ptr = b_tile(slice0_k, slice0_n);
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = a_base(slice0_n) + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = b_tile(slice0_k, slice0_n);
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    // ATLAS: header note 8
    auto c_base = [&] (int tile_n) -> void*
    {
        if constexpr (multi)
        {
            const int part = tile_n / tiles_n;
            const int local = tile_n - part * tiles_n;
            if constexpr (c_fp32) return (void*) (((float*) parts.c[part]) + local * gl_c_stride_n);
            else                  return (void*) (((half*) parts.c[part]) + local * gl_c_stride_n);
        }
        else
        {
            if constexpr (c_fp32) return (void*) (((float*) C) + tile_n * gl_c_stride_n);
            else                  return (void*) (((half*) C) + tile_n * gl_c_stride_n);
        }
    };
    half* gl_c_ptr_16 = ((half*) c_base(slice2_n)) + slice_m * gl_c_stride_m;
    float* gl_c_ptr_32 = ((float*) c_base(slice2_n)) + slice_m * gl_c_stride_m;

    // ATLAS: `register` dropped (header note 5); FragC_h removed (header note 3)
    FragA frag_a[FRAG_STAGES];
    FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    FragC frag_c[FRAGS_N_PER_WARP];
    // ATLAS: header note 10, sub-block 1 of a both_sub thread
    FragA frag_a1[both_sub ? FRAG_STAGES : 1];
    FragB frag_b1[both_sub ? FRAG_STAGES : 1][both_sub ? FRAGS_N_PER_WARP : 1];
    FragC frag_c1[both_sub ? FRAGS_N_PER_WARP : 1];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (multi)
            {
                // ATLAS: header note 8 (slice_m == 0)
                if constexpr (c_fp32)
                    gl_c_ptr_32 = (float*) c_base(slice2_n);
                else
                    gl_c_ptr_16 = (half*) c_base(slice2_n);
            }
            else if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix (XOR-swizzled for bank-conflict-free ldmatrix)
            if constexpr (a_smem_row)
            {
                // ATLAS: header notes 12/13, same F16 values from the staged row
                const int4* src = (const int4*) (a_row + (gl_a_ptr - A));
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) sh[load_a_sh[i]] = src[load_a_gl[i]];
                }
            }
            else if constexpr (!a_fp32)
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                }
            }
            else
            {
                // ATLAS: header note 6. load_a_gl indexes 8-element groups.
                const float4* gl = (const float4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i])
                    {
                        const float4 f0 = gl[2 * load_a_gl[i]];
                        const float4 f1 = gl[2 * load_a_gl[i] + 1];
                        half2_uint32 h0(__floats2half2_rn(f0.x, f0.y));
                        half2_uint32 h1(__floats2half2_rn(f0.z, f0.w));
                        half2_uint32 h2(__floats2half2_rn(f1.x, f1.y));
                        half2_uint32 h3(__floats2half2_rn(f1.z, f1.w));
                        sh[load_a_sh[i]] = make_int4((int) h0.as_uint32, (int) h1.as_uint32,
                                                     (int) h2.as_uint32, (int) h3.as_uint32);
                    }
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments (XOR-swizzled shared memory layout)
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[buf], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);
                if constexpr (both_sub)
                {
                    // ATLAS: header note 10, sub-block 1 columns
                    int c1_swizzled = (base_c + 2) ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                    ldsm4(frag_a1[buf], (int4*) sh1_a_ptr + R * A_COLS + c1_swizzled);
                }
            }
        }

        // B fragments
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
            if constexpr (both_sub)
            {
                // ATLAS: header note 10, sub-block 1 trellis tiles
                const uint32_t* shb1 = (const uint32_t*) (sh1_b_ptr + (TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
                dq_dispatch<bits, cb>(shb1, lane_id << 3, frag_b1[buf][n2], frag_b1[buf][n2 + 1]);
            }
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            frag_c[n] = {};
        if constexpr (both_sub)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c1[n] = {};
        }
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[n][j];
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[n][j] += *sh_red++;
                }
            }
        };

        auto store_small = [&] (int i)
        {
            if (sub_k == i && lane_id / 4 < size_m)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    *sh_red++ = frag_c[n][0];
                    *sh_red++ = frag_c[n][1];
                }
            }
            __syncthreads();
        };

        auto add_small = [&] (int i)
        {
            if (sub_k == i && lane_id / 4 < size_m)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    frag_c[n][0] += *sh_red++;
                    frag_c[n][1] += *sh_red++;
                }
            }
        };

        if constexpr (both_sub)
        {
            // ATLAS: header note 10, acc0 + acc1 in registers
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                #pragma unroll
                for (int j = 0; j < 4; ++j) frag_c[n][j] += frag_c1[n][j];
            }
            return;
        }

        if (size_m <= 8)
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store_small(1);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store_small(3);
                add_small(2);
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
        }
        else
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store(1);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store(1);
                add(0);
                store(2);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store(3);
                add(2);
                store(1);
                add(0);
                store(2);
                add(0);
            }
        }
    };

    // ATLAS: write_sum_tile_sh / output_had_sh_gl removed (header note 2)

    // ATLAS: header note 15, physical output row of logical row r (< size_m)
    auto c_row = [&] (int r) -> int { if constexpr (gather) return row_slots[r]; else return r; };

    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + c_row(r0) * size_n + (n0 + n) * 8 + c;
                    frag_c[n][0] += *c_ptr++;
                    frag_c[n][1] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + c_row(r0) * size_n + (n0 + n) * 8 + c);
                    float2 interim = __half22float2(*c_ptr);
                    frag_c[n][0] += interim.x;
                    frag_c[n][1] += interim.y;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + c_row(r1) * size_n + (n0 + n) * 8 + c;
                    frag_c[n][2] += *c_ptr++;
                    frag_c[n][3] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + c_row(r1) * size_n + (n0 + n) * 8 + c);
                    float2 interim = __half22float2(*c_ptr);
                    frag_c[n][2] += interim.x;
                    frag_c[n][3] += interim.y;
                }
            }
        }
    };

    // ATLAS: header note 4. Identity except -0.0f -> +0.0f.
    auto seed_sum_zero = [&]()
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) frag_c[n][j] = __fadd_rn(frag_c[n][j], 0.0f);
        }
    };

    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + c_row(r0) * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][0];
                    *c_ptr++ = frag_c[n][1];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + c_row(r0) * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);
                    *c_ptr = sum;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + c_row(r1) * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][2];
                    *c_ptr++ = frag_c[n][3];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + c_row(r1) * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[n][2], frag_c[n][3]);
                    *c_ptr = sum;
                }
            }
        }
    };

    // Output reduction
    auto reduce = [&] ()
    {
        // ATLAS: EXL3_GEMM_H_ACC fold removed (header note 3)

        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        // ATLAS: header note 4
        if (!sub_k && first)
        {
            seed_sum_zero();
        }

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        if (!sub_k && !first)
        {
            read_sum_gl();
        }

        // ATLAS: header note 7
        if constexpr (round_partial)
        {
            if (!sub_k)
            {
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[n][j] = __half2float(__float2half_rn(frag_c[n][j]));
                }
            }
        }

        // All but last threadblock in column write the intermediate result to global memory
        if (!sub_k && !last)
        {
            write_sum_gl();
        }

        // Last block writes in row-major format
        // ATLAS: shmem_out_had == false (header note 2)
        if constexpr (out_had)
        {
            // ATLAS: header note 14. `last` is uniform across the CTA.
            static_assert(!multi && c_fp32 && !both_sub, "out_had: single F32 component");
            if (last)
            {
                if (!sub_k)
                {
                    const int n0 = warp_id * FRAGS_N_PER_WARP;
                    const int r0 = lane_id / 4, r1 = r0 + 8, c = (lane_id % 4) * 2;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        if (r0 < size_m)
                        {
                            sh_c[r0 * TILESIZE_N + (n0 + n) * 8 + c] = frag_c[n][0];
                            sh_c[r0 * TILESIZE_N + (n0 + n) * 8 + c + 1] = frag_c[n][1];
                        }
                        if (r1 < size_m)
                        {
                            sh_c[r1 * TILESIZE_N + (n0 + n) * 8 + c] = frag_c[n][2];
                            sh_c[r1 * TILESIZE_N + (n0 + n) * 8 + c + 1] = frag_c[n][3];
                        }
                    }
                }
                __syncthreads();
                const int groups = TILESIZE_N / 128;
                const int lane = threadIdx.x % 32;
                for (int g = threadIdx.x / 32; g < size_m * groups; g += blockDim.x / 32)
                {
                    const int row = g / groups;
                    const int base = row * TILESIZE_N + (g % groups) * 128;
                    float v[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = sh_c[base + lane + 32 * j];
                    exl3v3_h128_butterfly(v, lane);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        const int col = slice2_n * TILESIZE_N + (g % groups) * 128 + lane + 32 * j;
                        const float value = __fmul_rn(v[j], EXL3V3_H128_NORM);
                        out_bf16[row * size_n + col] = __float2bfloat16_rn(__fmul_rn(value, __half2float(out_svh[col])));
                    }
                }
            }
        }
        else if (!sub_k && last)
        {
            write_sum_gl();
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
            if constexpr (both_sub)
                ptx_mma_m16n8k16(frag_a1[buf], frag_b1[buf][n], frag_c1[n]);
        }
    };

    // ATLAS: header note 13, fused input transform into the staged rows
    static_assert(!a_had || a_smem_row, "a_had stages through a_smem_row");
    static_assert(!a_act || (a_smem_row && !a_had), "a_act stages through a_smem_row");
    if constexpr (a_act)
    {
        // ATLAS: header note 16
        const int lane = threadIdx.x % 32;
        const int warps = blockDim.x / 32;
        const int chunks = size_m * size_k / 128;
        for (int chunk = threadIdx.x / 32; chunk < chunks; chunk += warps)
        {
            const int base = chunk * 128;
            const int feature0 = base % size_k;
            float g[4], u[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                g[j] = act.gate[base + lane + 32 * j];
                u[j] = act.up[base + lane + 32 * j];
            }
            exl3v3_h128_butterfly(g, lane);
            exl3v3_h128_butterfly(u, lane);
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int f = feature0 + lane + 32 * j;
                const float gate_out = __half2float(__hmul(__float2half_rn(__fmul_rn(g[j], EXL3V3_H128_NORM)), act.gate_svh[f]));
                const float up_out = __half2float(__hmul(__float2half_rn(__fmul_rn(u[j], EXL3V3_H128_NORM)), act.up_svh[f]));
                const float activation = thin_activation(gate_out, up_out, act.limit, act.clamp);
                v[j] = __half2float(__hmul(__float2half_rn(activation), act.suh[f]));
            }
            exl3v3_h128_butterfly(v, lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                a_row[base + lane + 32 * j] = __float2half_rn(__fmul_rn(v[j], EXL3V3_H128_NORM));
        }
        __syncthreads();
    }
    else if constexpr (a_had)
    {
        const __nv_bfloat16* xin = (const __nv_bfloat16*) A_in;
        const int lane = threadIdx.x % 32;
        const int warps = blockDim.x / 32;
        const int chunks = size_m * size_k / 128;
        for (int chunk = threadIdx.x / 32; chunk < chunks; chunk += warps)
        {
            const int base = chunk * 128;
            const int feature0 = base % size_k;
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int i = lane + 32 * j;
                const __half product = __hmul(__float2half_rn(__bfloat162float(xin[base + i])), a_suh[feature0 + i]);
                v[j] = __half2float(product);
            }
            exl3v3_h128_butterfly(v, lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                a_row[base + lane + 32 * j] = __float2half_rn(__fmul_rn(v[j], EXL3V3_H128_NORM));
        }
        __syncthreads();
    }
    // ATLAS: header note 12, stage the A rows once (F32 -> F16 RN)
    else if constexpr (a_smem_row)
    {
        const float4* src = (const float4*) A;
        half2* dst = (half2*) a_row;
        const int quads = size_m * size_k / 4;
        for (int i = threadIdx.x; i < quads; i += blockDim.x)
        {
            const float4 f = src[i];
            dst[2 * i] = __floats2half2_rn(f.x, f.y);
            dst[2 * i + 1] = __floats2half2_rn(f.z, f.w);
        }
        __syncthreads();
    }

    // Start global to shared pipeline
    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE_OLD(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }

    #undef FSTAGE_OLD
    #undef FSTAGE
}
