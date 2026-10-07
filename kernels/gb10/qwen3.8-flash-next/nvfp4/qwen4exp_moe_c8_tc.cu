// SPDX-License-Identifier: AGPL-3.0-only
//
// The expert units of qwen4exp_moe_c8.cu on TENSOR CORES
// (ATLAS_QWEN4EXP_MOE_TC=1, `layers/moe/forward_rows.rs` and serial decode's
// `forward_row_local`): exactness contract (b) -- a new numerics baseline
// that is ROW-INVARIANT: a row's outputs do not depend on which or how many
// other rows share the launch, and serial decode (1 row) runs the same
// kernels, so speculation stays exact against it.
//
// Numerics: weights w = lut[nibble] * dec_e4m3(scale), EXACT in BF16 (E2M1
// has 2 significant bits, E4M3 4, the product <= 6 and inside BF16's range);
// activations BF16 (gate/up: the input rows as they are; down: the FP32
// SiLU*up rounded to BF16); mma.sync m16n8k16 BF16 with FP32 accumulation;
// the per-tensor scale2 once per output, then one BF16 rounding. Against
// today's FP32 chain (w = lut * (dec * s2) rounded, then a * w products) it
// differs by accumulation order and down's BF16 activation (synthetic bench:
// max |diff| <= 0.8% of a row's max |out|).
//
// Row invariance: a unit's rows are one m16 tile (C8_RMAX = 16), padded with
// zeros; every output runs the same MMA sequence in the same k order whatever
// the tile holds; gate/up's 4 K quarters are summed in a fixed order.
// scripts/dev/qwen4exp_moe_c8_bench.sh tc-units-check compares every row alone
// with waves of 2..64 rows, byte for byte.
//
// K order (any fixed bijection is valid): thread t of a row holds real
// k = kb + 32t + 8j + i (j < 4, i < 8) of a k128 super-block -- one 16-byte
// load of B; for MMA pair j the f16 trick pairs nibbles (i, i + 4), so MMA 0
// takes virtual (2t, 2t+1 | 2t+8, 2t+9) = real (+0, +4 | +1, +5) and MMA 1
// (+2, +6 | +3, +7); A is permuted the same way (PRMT of 16 contiguous bytes).
//
// Grids as qwen4exp_moe_c8.cu: gate/up (640 / 8, units), down (2560 / 64,
// units), block 256, workspace from qwen4exp_moe_c8_tc_plan.
//
// MEASURED (preliminary: GB10, 256-expert EP2 pool, synthetic routing, us
// gate/up + down, fastest of 5) rows pair -> this: 1 row 46+30 -> 50+25,
// 4 rows 176+91 -> 178+88, 16 rows 404+252 -> 363+201, 32 rows at 50 / 70 /
// 90 / 119 unique: 662+422 -> 455+267, 708+450 -> 610+343, 838+506 ->
// 791+439, 1044+567 -> 1032+562. Tried: a smem-staged tile under a CTA
// barrier and warp-private cp.async rings (160-180 GB/s); a 32-output CTA a
// warp per n8 tile (slower at C1).

#include "qwen4exp_moe_c8.cuh"

extern "C" __global__ void __launch_bounds__(1024) qwen4exp_moe_c8_tc_plan(
    const unsigned int* __restrict__ expert_indices, unsigned int* __restrict__ ws,
    unsigned int top_k, unsigned int rows
) {
    c8_plan_body(expert_indices, ws, top_k, rows);
}

__device__ __forceinline__ void tc_mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned tc_bf16x2(float lo, float hi) {
    unsigned d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

// The thread's 8 weights of a k32 block (packed word q, scale byte e) as the
// 4 BF16 pairs (i, i + 4), i < 4: lut * dec(e), exact.
__device__ __forceinline__ void tc_b8(unsigned q, unsigned char e, unsigned (&b)[4]) {
    const float d = c8_dec_e4m3(e) * 16384.0f;  // * 2^14 undoes the f16 trick's scale
#pragma unroll
    for (unsigned s = 0; s < 16; s += 4) {
        const unsigned a = s <= 9 ? q << (9 - s) : q >> (s - 9);
        const unsigned x = (a & 0x0E000E00u) | ((q << (12 - s)) & 0x80008000u);
        const __half2 h = *(const __half2*)&x;
        b[s / 4] = tc_bf16x2(__low2float(h) * d, __high2float(h) * d);
    }
}

// A pairs for one row of a k32 block from its 8 BF16 at k = kb + 8t + i.
__device__ __forceinline__ void tc_a8(uint4 w, unsigned (&a)[4]) {
    a[0] = __byte_perm(w.x, w.z, 0x5410);  // (k0, k4)
    a[1] = __byte_perm(w.x, w.z, 0x7632);  // (k1, k5)
    a[2] = __byte_perm(w.y, w.w, 0x5410);  // (k2, k6)
    a[3] = __byte_perm(w.y, w.w, 0x7632);  // (k3, k7)
}

// One k32 block of an n8 tile: the thread's packed word q and scale byte e
// (row g of the tile), A pairs of rows g and g + 8.
__device__ __forceinline__ void tc_block(float (&acc)[4], unsigned q, unsigned char e,
                                         const unsigned (&a)[4], const unsigned (&a8)[4]) {
    unsigned b[4];
    tc_b8(q, e, b);
    tc_mma(acc, a[0], a8[0], a[1], a8[1], b[0], b[1]);
    tc_mma(acc, a[2], a8[2], a[3], a8[3], b[2], b[3]);
}

// ── B straight to registers, 16 bytes a thread a k128 super-block ──
// Thread t of row g holds real k = kb + 32t + 8j + i (j < 4, i < 8) of a
// super-block: one uint4 of B (word j: k32 MMA pair j), two scale bytes
// (groups 2t, 2t + 1 of the super-block), A rows as 64 contiguous bytes.
// The super-block loop is unrolled so the B loads issue ahead of use.
// Super-blocks [sb0, sb0 + SB) of the tile (all of K: sb0 = 0, SB = K/128).
template <unsigned K, unsigned SB, typename Act>
__device__ __forceinline__ void tc_tile(const unsigned char* __restrict__ B, const unsigned char* __restrict__ Sc,
                                         unsigned lane, float (&acc)[4], Act&& act, unsigned sb0 = 0) {
    constexpr unsigned ROW = K / 2, K16 = K / 16;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned char* b = B + (size_t)g * ROW + 16 * t + 64 * sb0;
    const unsigned char* s = Sc + (size_t)g * K16 + 2 * t + 8 * sb0;
#pragma unroll
    for (unsigned sb = 0; sb < SB; sb++) {
        const uint4 w = *(const uint4*)(b + 64 * sb);
        const unsigned short e = *(const unsigned short*)(s + 8 * sb);
        const unsigned q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            unsigned a[4], a8[4];
            act((sb0 + sb) * 128 + 32 * t + 8 * j, a, a8);
            tc_block(acc, q[j], (unsigned char)(e >> (8 * (j >> 1))), a, a8);
        }
    }
}

// ── down: a CTA = (unit, 64 outputs), warp w the n8 tile 8w, all of K ──
extern "C" __global__ void __launch_bounds__(256) qwen4exp_moe_c8_tc_down(C8_SD_ARGS) {
    constexpr unsigned TILE = 64;
    C8Unit u;
    if (!c8_unit(ws, u)) return;
    const unsigned slots = rows * top_k;
    const unsigned n0 = blockIdx.x * TILE;
    const unsigned char* B = u.shared() ? sh_down_packed : (const unsigned char*)packed_ptrs[u.expert];
    const unsigned char* Sc = u.shared() ? sh_down_scale : (const unsigned char*)scale_ptrs[u.expert];
    const float s2 = u.shared() ? sh_down_s2 : scale2_vals[u.expert];
    __nv_bfloat16* out = u.shared() ? sh_down_out : C;
    if (B == 0) {
        for (unsigned i = threadIdx.x; i < u.n * TILE; i += blockDim.x)
            out[(size_t)c8_slot(ws, u, i / TILE) * C8_H + n0 + i % TILE] = __float2bfloat16(0.0f);
        return;
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3u;
    const bool r0 = g < u.n, r1 = g + 8 < u.n;
    const unsigned q0 = c8_slot(ws, u, r0 ? g : 0), q1 = c8_slot(ws, u, r1 ? g + 8 : 0);
    const float* act_base = act + (u.shared() ? (size_t)slots * C8_I : 0);
    const float* X0 = act_base + (size_t)q0 * C8_I;
    const float* X1 = act_base + (size_t)q1 * C8_I;
    const unsigned nw = n0 + 8 * warp;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    tc_tile<C8_I, C8_I / 128>(B + (size_t)nw * (C8_I / 2), Sc + (size_t)nw * (C8_I / 16), lane, acc,
                   [&](unsigned k, unsigned (&a)[4], unsigned (&a8)[4]) {
                       a[0] = a[1] = a[2] = a[3] = a8[0] = a8[1] = a8[2] = a8[3] = 0;
                       if (r0) {
                           const float4 x = *(const float4*)(X0 + k), y = *(const float4*)(X0 + k + 4);
                           a[0] = tc_bf16x2(x.x, y.x); a[1] = tc_bf16x2(x.y, y.y);
                           a[2] = tc_bf16x2(x.z, y.z); a[3] = tc_bf16x2(x.w, y.w);
                       }
                       if (r1) {
                           const float4 x = *(const float4*)(X1 + k), y = *(const float4*)(X1 + k + 4);
                           a8[0] = tc_bf16x2(x.x, y.x); a8[1] = tc_bf16x2(x.y, y.y);
                           a8[2] = tc_bf16x2(x.z, y.z); a8[3] = tc_bf16x2(x.w, y.w);
                       }
                   });
    const unsigned col = nw + 2 * t;
    if (r0) *(unsigned*)(out + (size_t)q0 * C8_H + col) = tc_bf16x2(acc[0] * s2, acc[1] * s2);
    if (r1) *(unsigned*)(out + (size_t)q1 * C8_H + col) = tc_bf16x2(acc[2] * s2, acc[3] * s2);
}

// ── gate/up + SiLU: a CTA = (unit, 8 outputs) x {gate, up} like the rows
// kernel's grid (640 / 8, units); warp w: projection w & 1, K quarter w >> 1
// (5 super-blocks); the quarters summed in order (fixed: row-invariant).
extern "C" __global__ void __launch_bounds__(256) qwen4exp_moe_c8_tc_gate_up(C8_GU_ARGS) {
    C8Unit u;
    if (!c8_unit(ws, u)) return;
    const unsigned slots = rows * top_k;
    const unsigned n0 = blockIdx.x * 8;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, proj = warp & 1u, kq = warp >> 1;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned char *B, *Sc;
    if (u.shared()) {
        B = proj ? sh_up_packed : sh_gate_packed; Sc = proj ? sh_up_scale : sh_gate_scale;
    } else {
        B = (const unsigned char*)(proj ? up_packed_ptrs : gate_packed_ptrs)[u.expert];
        Sc = (const unsigned char*)(proj ? up_scale_ptrs : gate_scale_ptrs)[u.expert];
    }
    __nv_bfloat16* out = u.shared() ? (proj ? sh_up_out : sh_gate_out) : (proj ? up_out : gate_out);
    if (B == 0) {
        if (out && kq == 0)
            for (unsigned i = lane; i < u.n * 8; i += 32)
                out[(size_t)c8_slot(ws, u, i / 8) * C8_I + n0 + i % 8] = __float2bfloat16(0.0f);
        return;
    }
    __shared__ float s_part[4][2][16][8];
    __shared__ unsigned short s_gu[2][16][8];
    const bool r0 = g < u.n, r1 = g + 8 < u.n;
    const unsigned q0 = c8_slot(ws, u, r0 ? g : 0), q1 = c8_slot(ws, u, r1 ? g + 8 : 0);
    const __nv_bfloat16* A0 = A + (size_t)(u.shared() ? q0 : q0 / top_k) * C8_H;
    const __nv_bfloat16* A1 = A + (size_t)(u.shared() ? q1 : q1 / top_k) * C8_H;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    tc_tile<C8_H, C8_H / 512>(B + (size_t)n0 * (C8_H / 2), Sc + (size_t)n0 * (C8_H / 16), lane, acc,
                               [&](unsigned k, unsigned (&a)[4], unsigned (&a8)[4]) {
                                   const uint4 z = make_uint4(0, 0, 0, 0);
                                   tc_a8(r0 ? *(const uint4*)(A0 + k) : z, a);
                                   tc_a8(r1 ? *(const uint4*)(A1 + k) : z, a8);
                               }, kq * (C8_H / 512));
    s_part[kq][proj][g][2 * t] = acc[0]; s_part[kq][proj][g][2 * t + 1] = acc[1];
    s_part[kq][proj][g + 8][2 * t] = acc[2]; s_part[kq][proj][g + 8][2 * t + 1] = acc[3];
    __syncthreads();
    {
        const unsigned p = threadIdx.x >> 7, m = (threadIdx.x >> 3) & 15u, j = threadIdx.x & 7u;
        const float v = ((s_part[0][p][m][j] + s_part[1][p][m][j]) + s_part[2][p][m][j]) + s_part[3][p][m][j];
        const __nv_bfloat16 o = __float2bfloat16(v * (p ? (u.shared() ? sh_up_s2 : up_scale2_vals[u.expert])
                                                        : (u.shared() ? sh_gate_s2 : gate_scale2_vals[u.expert])));
        s_gu[p][m][j] = __bfloat16_as_ushort(o);
        __nv_bfloat16* po = u.shared() ? (p ? sh_up_out : sh_gate_out) : (p ? up_out : gate_out);
        if (po && m < u.n) po[(size_t)c8_slot(ws, u, m) * C8_I + n0 + j] = o;
    }
    __syncthreads();
    if (threadIdx.x < u.n * 8) {
        const unsigned m = threadIdx.x >> 3, j = threadIdx.x & 7u;
        float gf = __uint_as_float((unsigned)s_gu[0][m][j] << 16);
        float uf = __uint_as_float((unsigned)s_gu[1][m][j] << 16);
        const float SWIGLU_LIMIT = 10.0f;
        if (!u.shared()) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        float* act_base = act + (u.shared() ? (size_t)slots * C8_I : 0);
        act_base[(size_t)c8_slot(ws, u, m) * C8_I + n0 + j] = (gf / (1.0f + __expf(-gf))) * uf;
    }
}
