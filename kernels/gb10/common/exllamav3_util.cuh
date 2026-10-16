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
// Upstream: exllamav3_ext/util.h (sha256 ba89ac6793bf31cfe123d13baa2e34531a18a24bc3b2ff4add8cf4856b75a1c8)
//       and exllamav3_ext/util.cuh (sha256 1907cea115260db7c3b0de540e375e7733b7ffc9019bec190c1d92abea1d964e).
// Atlas adaptation: only the macros and bit-cast unions the packed GEMM core
// uses. Host-side helpers (cuda_check, cuBLAS, torch) are omitted.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#ifndef CEIL_DIVIDE
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#endif
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};
