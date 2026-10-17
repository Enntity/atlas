// SPDX-License-Identifier: AGPL-3.0-only

// Exact-operation helpers for the EXL3 reference-policy (FLA) KDA kernels.
//
// Every floating-point operation in kda_fla_*.cuh goes through these helpers
// so nvcc can neither contract (mul+add -> fma) nor substitute approximations.
// Each helper names the PTX instruction the pinned Triton reference emits on
// GB10 (docs/design/exl3-kda-reference-core.md §3.6a). Bit-exactness against
// the reference stage oracle depends on matching these instructions and the
// reference's operand order, not on the mathematically equivalent value.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace kda_fla {

// log2(e) as the f32 constant Triton uses for tl.exp: 0x3FB8AA3B.
__device__ __forceinline__ float log2e() { return __int_as_float(0x3FB8AA3B); }

__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }

// ex2.approx.f32 (no flush-to-zero) — tl.exp lowering in conv/gate/decode.
__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// ex2.approx.ftz.f32 — FLA's exp2() in the chunk kernels.
__device__ __forceinline__ float ex2_approx_ftz(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Triton tl.exp(x) = ex2.approx.f32(mul.f32(x, log2e)).
__device__ __forceinline__ float tl_exp(float x) { return ex2_approx(fmul(x, log2e())); }

// div.full.f32 — Triton's default f32 division.
__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float sqrt_approx_ftz(float x) {
    float y;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Inductor's lowering of torch sigmoid (reference `_cast_sigmoid`):
// div.full(1, add(ex2.approx(mul(0 - x, log2e)), 1)).
__device__ __forceinline__ float inductor_sigmoid(float x) {
    return div_full(1.0f, fadd(ex2_approx(fmul(fsub(0.0f, x), log2e())), 1.0f));
}

// cvt.rn.bf16.f32 (round to nearest even).
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// D = A*B + C, m16n8k16, BF16 operands, F32 accumulate.
// a: 4 regs (8 bf16), b: 2 regs (4 bf16), c/d: 4 f32. Standard PTX fragment layout.
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                               const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// D = A*B + C, m16n8k8, TF32 operands fed as raw f32 bits (hardware uses the
// top 19 bits; no cvt.*.tf32 — the reference emits none), F32 accumulate.
// a: 4 regs, b: 2 regs, c/d: 4 f32.
__device__ __forceinline__ void mma_tf32_1688(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                              const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ uint32_t pack_bf16x2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

}  // namespace kda_fla
