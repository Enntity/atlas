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
// Upstream: exllamav3_ext/ptx.cuh (sha256 8ceacb1b321af5877a9a3d651e038fb1f44b543c3011e2f814f448830efe27ee).
// Atlas adaptation: only the pieces the packed GEMM core uses are kept, verbatim
// (fragment types, m16n8k16 F32-accumulate MMA, lock-ordered global barrier,
// cp.async, ldmatrix, bitfield helpers). The FP16-accumulate MMA overload (used
// only on sm_86), m8n8k4, group_barrier and memory-op helpers are omitted.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// Tensor core fragments

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;

// m16n8k16 tensor core matmul
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type

// FP16 @ FP16 + FP32 -> FP32
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
}

// Global barrier

__device__ inline void barrier_acquire
(
    int* lock,
    int stage
)
{
    if (threadIdx.x == 0)
    {
        volatile int state = -1;
        do
        {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        }
        while (state != stage);
    }
    __syncthreads();
}

__device__ inline void barrier_release
(
    int* lock,
    int val,
    bool reset
)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            *lock = 0;
            return;
        }
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}

// Load global to shared memory

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Async copy fence, commit all pending async copies

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most n async groups are still pending.

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Load 16x16 matrix fragment from shared memory, directly in tensor core layout

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

// Bitfield stuff

static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t result64;
    asm ("bfe.u64 %0, %1, %2, %3;"
         : "=l"(result64)
         : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
}

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
