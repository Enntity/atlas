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
// Upstream: exllamav3_ext/quant/exl3_kernel_map.cuh (sha256 59c49719b489764410a129a73ec3fddfab9eb96bfeedbf365d020ce0a39b26d4)
//       and exllamav3_ext/quant/exl3_gemm_inner.cuh (SMEM_MAX, EXL3_GEMM_BASE_THREADS).
// Atlas adaptation: only the template-argument list, the kernel shape table and
// the launch constants are kept. Host selection/autotune tables are replaced by
// Atlas's pinned numerics profiles (spark-model exl3_numerics.rs).
#pragma once

#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

//                            M       K       N  SH_STAGES  FRAG_STAGES
#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

// Constants
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6
