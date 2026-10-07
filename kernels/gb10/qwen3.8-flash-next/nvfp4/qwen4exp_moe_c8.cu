// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) routed + shared MoE for a launch of verify
// rows (C4/C8: 16-36), each output byte what `qwen4exp_moe_rows.cu` -- so
// serial decode's moe_expert_{gate_up,silu_down}_shared -- writes
// (ATLAS_QWEN4EXP_MOE_UNITS=1, `layers/moe/forward_rows.rs`).
//
// WHY. At 32 rows the rows pair is compute-bound: with every expert in L2 it
// takes ~550 (gate/up) + ~385 us (silu/down) at C8 routing vs ~530 + ~265 us
// of DRAM time for ~70 unique experts (scripts/dev/qwen4exp_moe_c8_bench.cu,
// POOL=3). gate/up decodes every weight per (row, slot); silu/down's lanes
// split K/16 = 40 steps 2:1 and it re-decodes per 2-row chunk.
//
// WHAT. A plan groups the (row, slot) entries into UNITS -- one expert's
// rows, up to C8_RMAX; the shared expert's rows likewise -- heaviest first.
//   gate/up: a CTA = (unit, 8 outputs) x {gate, up}; each warp the rows
//     kernel's two outputs, decoding each step's weights once for RC rows;
//     the epilogue stores SiLU(gate) * up of the BF16 outputs as FP32 (the
//     activation silu/down computes), so down needs no SiLU.
//   down: a CTA = (unit, 64 outputs), tile in shared; 8 lanes an output with
//     5 steps each (the uneven lane chains rebalanced, below).
// Each output's operation sequence is the single-row kernel's (the target
// builds with --fmad=false): chain l accumulates k16 = l, l + 32, ... < K/16
// in order, each step's 8 packed bytes in order, `acc += a_lo * w_lo +
// a_hi * w_hi` with w = lut[nibble] * (dec_e4m3(scale) * s2); the
// shfl_down 16/8/4/2/1 tree over the 32 chains; one BF16 rounding. Remote
// (NULL) experts write zeros.
//
// MEASURED (preliminary: GB10, 256-expert EP2 pool, 32 rows, synthetic
// routing, us gate/up + down) rows pair -> this at 50/70/90/119 unique:
//   675+432 -> 553+389,  750+463 -> 680+443,  876+521 -> 864+541,
//   1089+571 -> 1091+619. C1 verify (4 rows) and C4 (16) neutral to -4%.
// Tried and dropped: a smem-staged gate/up tile (whole, or two cp.async
// groups; -5..15%), persistent CTAs with a 2-4 stage cp.async ring (-10..40%,
// latency-bound compute at 8-16 warps an SM), RC = 8 (spills).
//
// Workspace (u32, qwen4exp_moe_c8_plan): [0] unit count; at C8_WS_UNITS 3
// words a unit {expert | C8_SHARED, start, rows}; at C8_WS_ROWS the
// expert-sorted slot ids (a routed unit's rows are rowlist[start ..], a
// shared unit's tokens start ..). `act`: FP32 [slots + rows, 640], slot q
// at q, shared token t at slots + t.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define C8_RMAX 16u
#define C8_SLOTS_MAX 1024u
#define C8_ROWS_MAX 64u
#define C8_UNITS_MAX (C8_SLOTS_MAX + C8_ROWS_MAX)
#define C8_WS_UNITS 16u
#define C8_WS_ROWS (C8_WS_UNITS + 3u * C8_UNITS_MAX)
#define C8_SHARED 0xFFFFFFFFu
#define C8_H 2560u
#define C8_I 640u

__device__ __forceinline__ float c8_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}

// ── plan: one block of 1024 threads ──
// Entries ranked by (expert, q); a unit head every C8_RMAX entries of one
// expert; shared units of C8_RMAX tokens; units placed by (rows desc, index).
extern "C" __global__ void __launch_bounds__(1024) qwen4exp_moe_c8_plan(
    const unsigned int* __restrict__ expert_indices, unsigned int* __restrict__ ws,
    unsigned int top_k, unsigned int rows
) {
    __shared__ unsigned s_idx[C8_SLOTS_MAX];
    __shared__ unsigned s_e[C8_SLOTS_MAX];
    __shared__ unsigned s_cnt[C8_UNITS_MAX];
    __shared__ unsigned s_units;
    const unsigned slots = rows * top_k;
    if (slots > C8_SLOTS_MAX || rows > C8_ROWS_MAX) __trap();
    const unsigned n_sh = (rows + C8_RMAX - 1) / C8_RMAX;
    const unsigned cands = slots + n_sh;
    if (threadIdx.x == 0) s_units = 0;
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) s_idx[t] = expert_indices[t];
    __syncthreads();
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) {
        const unsigned e = s_idx[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < slots; j++) {
            const unsigned f = s_idx[j];
            rank += (f < e) || (f == e && j < t);
        }
        s_e[rank] = e;
        ws[C8_WS_ROWS + rank] = t;
    }
    __syncthreads();
    // Candidate c < slots: the unit headed at rank c (0 rows: not a head);
    // c >= slots: shared unit c - slots.
    for (unsigned c = threadIdx.x; c < cands; c += blockDim.x) {
        unsigned cnt;
        if (c < slots) {
            const unsigned e = s_e[c];
            unsigned start = c;
            while (start > 0 && s_e[start - 1] == e) start--;
            unsigned end = c + 1;
            while (end < slots && s_e[end] == e) end++;
            cnt = (c - start) % C8_RMAX == 0 ? min(C8_RMAX, end - c) : 0u;
        } else {
            cnt = min(C8_RMAX, rows - (c - slots) * C8_RMAX);
        }
        s_cnt[c] = cnt;
        if (cnt) atomicAdd(&s_units, 1u);
    }
    __syncthreads();
    for (unsigned c = threadIdx.x; c < cands; c += blockDim.x) {
        const unsigned cnt = s_cnt[c];
        if (!cnt) continue;
        unsigned pos = 0;
        for (unsigned d = 0; d < cands; d++) {
            const unsigned o = s_cnt[d];
            pos += (o > cnt) || (o == cnt && d < c);
        }
        unsigned* u = ws + C8_WS_UNITS + 3 * pos;
        if (c < slots) { u[0] = s_e[c]; u[1] = c; }
        else           { u[0] = C8_SHARED; u[1] = (c - slots) * C8_RMAX; }
        u[2] = cnt;
    }
    if (threadIdx.x == 0) ws[0] = s_units;
}

struct C8Unit {
    unsigned expert, start, n;
    __device__ __forceinline__ bool shared() const { return expert == C8_SHARED; }
};
__device__ __forceinline__ bool c8_unit(const unsigned* __restrict__ ws, C8Unit& u) {
    if (blockIdx.y >= ws[0]) return false;
    const unsigned* p = ws + C8_WS_UNITS + 3 * blockIdx.y;
    u.expert = p[0]; u.start = p[1]; u.n = p[2];
    return true;
}
// Unit row i's slot q (routed) or token (shared).
__device__ __forceinline__ unsigned c8_slot(const unsigned* ws, const C8Unit& u, unsigned i) {
    return u.shared() ? u.start + i : ws[C8_WS_ROWS + u.start + i];
}

__device__ __forceinline__ void c8_bf16_row(const __nv_bfloat16* p, unsigned k16, float (&f)[16]) {
    const uint4* q = (const uint4*)p + k16 * 2;
    const uint4 lo = q[0], hi = q[1];
    const unsigned w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int b = 0; b < 8; b++) {
        f[2 * b] = __uint_as_float(w[b] << 16);
        f[2 * b + 1] = __uint_as_float(w[b] & 0xFFFF0000u);
    }
}

__device__ __forceinline__ void c8_f32_row(const float* p, unsigned k16, float (&f)[16]) {
#pragma unroll
    for (int v = 0; v < 4; v++) {
        const float4 x = ((const float4*)p)[k16 * 4 + v];
        f[4 * v] = x.x; f[4 * v + 1] = x.y; f[4 * v + 2] = x.z; f[4 * v + 3] = x.w;
    }
}

#define C8_GU_ARGS                                                             \
    const __nv_bfloat16* __restrict__ A,                                       \
    const unsigned long long* __restrict__ gate_packed_ptrs,                   \
    const unsigned long long* __restrict__ gate_scale_ptrs,                    \
    const float* __restrict__ gate_scale2_vals,                                \
    const unsigned long long* __restrict__ up_packed_ptrs,                     \
    const unsigned long long* __restrict__ up_scale_ptrs,                      \
    const float* __restrict__ up_scale2_vals,                                  \
    const unsigned char* __restrict__ sh_gate_packed,                          \
    const unsigned char* __restrict__ sh_gate_scale, float sh_gate_s2,         \
    const unsigned char* __restrict__ sh_up_packed,                            \
    const unsigned char* __restrict__ sh_up_scale, float sh_up_s2,             \
    const unsigned int* __restrict__ ws,                                       \
    __nv_bfloat16* __restrict__ gate_out, __nv_bfloat16* __restrict__ up_out,  \
    __nv_bfloat16* __restrict__ sh_gate_out,                                   \
    __nv_bfloat16* __restrict__ sh_up_out, float* __restrict__ act,            \
    unsigned int top_k, unsigned int rows
#define C8_SD_ARGS                                                             \
    const float* __restrict__ act,                                             \
    const unsigned long long* __restrict__ packed_ptrs,                        \
    const unsigned long long* __restrict__ scale_ptrs,                         \
    const float* __restrict__ scale2_vals,                                     \
    const unsigned char* __restrict__ sh_down_packed,                          \
    const unsigned char* __restrict__ sh_down_scale, float sh_down_s2,         \
    const unsigned int* __restrict__ ws,                                       \
    __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ sh_down_out,    \
    unsigned int top_k, unsigned int rows

__device__ __forceinline__ void c8_cp16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
// `bytes` (a multiple of 16) from global to shared by the whole CTA: 16-byte
// cp.async when both ends allow, else bytewise. The caller commits and waits.
__device__ __forceinline__ void c8_stage(unsigned char* dst, const unsigned char* src, unsigned bytes) {
    if ((((size_t)src | (size_t)dst) & 15) == 0) {
        for (unsigned i = threadIdx.x; i < bytes / 16; i += blockDim.x) c8_cp16(dst + 16 * i, src + 16 * i);
    } else {
        for (unsigned i = threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
    }
}
// The 16 weights of one packed step, w[i] = lut[nibble i] * sc, without the
// LUT: nibbles i and i + 4 of a 32-bit word become an f16x2 whose halves are
// lut / 2^14 exactly (E2M1's 2 exponent bits and 1 mantissa bit land on
// f16's low exponent bits and top mantissa bit; 0 and 0.5 on f16's
// subnormals), widened exactly to f32 and multiplied by sc * 2^14 (exact:
// a power-of-two scaling). The real product is lut * sc's, so the rounded
// f32 is the LUT form's bit for bit.
__device__ __forceinline__ void c8_dq16(unsigned long long p, float sc, float (&w)[16]) {
    const float scx = sc * 16384.0f;
#pragma unroll
    for (unsigned h = 0; h < 2; h++) {
        const unsigned q = (unsigned)(p >> (32 * h));
#pragma unroll
        for (unsigned s = 0; s < 16; s += 4) {
            const unsigned a = s <= 9 ? q << (9 - s) : q >> (s - 9);
            const unsigned b = q << (12 - s);
            const unsigned x = (a & 0x0E000E00u) | (b & 0x80008000u);
            const __half2 hx = *(const __half2*)&x;
            // nibble s / 4 of the word's low half, and nibble s / 4 + 4.
            w[8 * h + s / 4] = __low2float(hx) * scx;
            w[8 * h + s / 4 + 4] = __high2float(hx) * scx;
        }
    }
}

// ── gate/up + SiLU: grid (640 / 8, units max), block 256 ──
// Warps 0-3 gate, 4-7 up; warp w runs outputs n1 = n0 + 2 (w % 4) and
// n1 + 1 -- the rows kernel's two outputs a warp, lane l the chain
// k16 = l + 32 s, s < 5 -- for every row of the unit, R at a time: each
// step's 16 weights per output are decoded once and serve the R rows. The
// next step's packed words load while this step computes.
template <unsigned R>
__device__ __forceinline__ void c8_gu_chunk(
    const unsigned char* __restrict__ B, const unsigned char* __restrict__ S, float s2,
    const __nv_bfloat16* __restrict__ A, const unsigned (&in_row)[8],
    unsigned n1, unsigned lane, float (&o1)[8], float (&o2)[8]
) {
    constexpr unsigned ROW = C8_H / 2, K16 = C8_H / 16;
    const unsigned char* b1 = B + (size_t)n1 * ROW;
    const unsigned char* t1 = S + (size_t)n1 * K16;
    float acc1[R], acc2[R];
#pragma unroll
    for (unsigned r = 0; r < R; r++) { acc1[r] = 0.0f; acc2[r] = 0.0f; }
    unsigned long long p1 = *(const unsigned long long*)(b1 + lane * 8);
    unsigned long long p2 = *(const unsigned long long*)(b1 + ROW + lane * 8);
    unsigned char e1 = t1[lane], e2 = t1[K16 + lane];
#pragma unroll 1
    for (unsigned k16 = lane; k16 < K16; k16 += 32) {
        const unsigned long long c1 = p1, c2 = p2;
        const float sc1 = c8_dec_e4m3(e1) * s2, sc2 = c8_dec_e4m3(e2) * s2;
        if (k16 + 32 < K16) {
            p1 = *(const unsigned long long*)(b1 + (k16 + 32) * 8);
            p2 = *(const unsigned long long*)(b1 + ROW + (k16 + 32) * 8);
            e1 = t1[k16 + 32]; e2 = t1[K16 + k16 + 32];
        }
        float w1[16], w2[16];
        c8_dq16(c1, sc1, w1);
        c8_dq16(c2, sc2, w2);
#pragma unroll
        for (unsigned r = 0; r < R; r++) {
            float f[16];
            c8_bf16_row(A + (size_t)in_row[r] * C8_H, k16, f);
#pragma unroll
            for (int b = 0; b < 8; b++) {
                acc1[r] += f[2 * b] * w1[2 * b] + f[2 * b + 1] * w1[2 * b + 1];
                acc2[r] += f[2 * b] * w2[2 * b] + f[2 * b + 1] * w2[2 * b + 1];
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; r++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc1[r] += __shfl_down_sync(0xFFFFFFFF, acc1[r], off);
            acc2[r] += __shfl_down_sync(0xFFFFFFFF, acc2[r], off);
        }
        o1[r] = acc1[r]; o2[r] = acc2[r];
    }
}

template <unsigned RC>
__device__ __forceinline__ void c8_gate_up(C8_GU_ARGS) {
    static_assert(RC >= 1 && RC <= 8, "row chunk");
    if (blockDim.x != 256) __trap();
    C8Unit u;
    if (!c8_unit(ws, u)) return;
    const unsigned slots = rows * top_k;
    const unsigned n0 = blockIdx.x * 8;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, proj = warp >> 2;
    const unsigned char *B, *S;
    float s2;
    if (u.shared()) {
        B = proj ? sh_up_packed : sh_gate_packed; S = proj ? sh_up_scale : sh_gate_scale;
        s2 = proj ? sh_up_s2 : sh_gate_s2;
    } else {
        B = (const unsigned char*)(proj ? up_packed_ptrs : gate_packed_ptrs)[u.expert];
        S = (const unsigned char*)(proj ? up_scale_ptrs : gate_scale_ptrs)[u.expert];
        s2 = (proj ? up_scale2_vals : gate_scale2_vals)[u.expert];
    }
    __nv_bfloat16* out = u.shared() ? (proj ? sh_up_out : sh_gate_out) : (proj ? up_out : gate_out);
    if (B == 0) {  // remote expert (or no shared expert): zeros
        if (out)
            for (unsigned i = threadIdx.x & 127u; i < u.n * 8; i += 128)
                out[(size_t)c8_slot(ws, u, i / 8) * C8_I + n0 + i % 8] = __float2bfloat16(0.0f);
        return;
    }
    __shared__ unsigned short s_gu[2][RC][8];
    const unsigned j1 = 2 * (warp & 3u), n1 = n0 + j1;
    const bool clamp = !u.shared();
    float* act_base = act + (u.shared() ? (size_t)slots * C8_I : 0);
#pragma unroll 1
    for (unsigned c0 = 0; c0 < u.n; c0 += RC) {
        const unsigned n = min(RC, u.n - c0);
        unsigned in_row[8], q_row[8];
#pragma unroll
        for (unsigned r = 0; r < 8; r++) {
            q_row[r] = c8_slot(ws, u, c0 + min(r, n - 1));
            in_row[r] = u.shared() ? q_row[r] : q_row[r] / top_k;
        }
        float v1[8], v2[8];
        switch (n) {
#define C8_GU_CASE(R_) \
        case R_: if constexpr (R_ <= RC) c8_gu_chunk<R_>(B, S, s2, A, in_row, n1, lane, v1, v2); break;
        C8_GU_CASE(1) C8_GU_CASE(2) C8_GU_CASE(3) C8_GU_CASE(4)
        C8_GU_CASE(5) C8_GU_CASE(6) C8_GU_CASE(7) C8_GU_CASE(8)
#undef C8_GU_CASE
        }
        if (lane == 0) {
#pragma unroll
            for (unsigned r = 0; r < RC; r++) {
                if (r >= n) break;
                const unsigned short g1 = __bfloat16_as_ushort(__float2bfloat16(v1[r]));
                const unsigned short g2 = __bfloat16_as_ushort(__float2bfloat16(v2[r]));
                s_gu[proj][r][j1] = g1;
                s_gu[proj][r][j1 + 1] = g2;
                if (out) *(unsigned*)(out + (size_t)q_row[r] * C8_I + n1) = (unsigned)g1 | ((unsigned)g2 << 16);
            }
        }
        __syncthreads();
        // The rows kernel's SiLU of the BF16 gate/up, verbatim.
        if (threadIdx.x < n * 8) {
            const unsigned r = threadIdx.x / 8, jj = threadIdx.x % 8;
            float gf = __uint_as_float((unsigned)s_gu[0][r][jj] << 16);
            float uf = __uint_as_float((unsigned)s_gu[1][r][jj] << 16);
            const float SWIGLU_LIMIT = 10.0f;
            if (clamp) {
                gf = fminf(gf, SWIGLU_LIMIT);
                uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
            }
            act_base[(size_t)q_row[r] * C8_I + n0 + jj] = (gf / (1.0f + __expf(-gf))) * uf;
        }
        __syncthreads();
    }
}

// ── down: grid (2560 / 32, units max), block 256 ──
// K/16 = 40 steps an output, so the rows kernel's lane chains are uneven
// (lanes 0-7 two steps, 8-31 one). Here 8 lanes run one output: lane j of
// the group holds chains j (k16 = j, j + 32), j + 8, j + 16, j + 24 (one
// step each) -- five steps a lane -- and the shfl_down tree's first two
// levels are in-thread adds of the same operands: chain c + chain c + 16,
// then c + (c + 8); levels 4/2/1 shuffle within the group. Warp w: outputs
// n0 + 32 g + 4w + group, g < OG. The CTA's 32 OG-output tile is staged in
// shared (cp.async).

// One k16 step of OG outputs (tile rows 32 apart) for R rows.
template <unsigned R, unsigned OG>
__device__ __forceinline__ void c8_dn_step(
    const unsigned char* __restrict__ s_w, const unsigned char* __restrict__ s_s, float s2,
    const float* __restrict__ act, const unsigned (&q_row)[8],
    unsigned k16, float (&x)[R][OG]
) {
    constexpr unsigned ROW = C8_I / 2, K16 = C8_I / 16;
    float w[OG][16];
#pragma unroll
    for (unsigned g = 0; g < OG; g++) {
        const unsigned long long p = *(const unsigned long long*)(s_w + g * 32 * ROW + k16 * 8);
        const float sc = c8_dec_e4m3(s_s[g * 32 * K16 + k16]) * s2;
        c8_dq16(p, sc, w[g]);
    }
#pragma unroll
    for (unsigned r = 0; r < R; r++) {
        float f[16];
        c8_f32_row(act + (size_t)q_row[r] * C8_I, k16, f);
#pragma unroll
        for (unsigned g = 0; g < OG; g++)
#pragma unroll
            for (int b = 0; b < 8; b++) x[r][g] += f[2 * b] * w[g][2 * b] + f[2 * b + 1] * w[g][2 * b + 1];
    }
}

template <unsigned R, unsigned OG>
__device__ __forceinline__ void c8_dn_chunk(
    const unsigned char* __restrict__ s_w, const unsigned char* __restrict__ s_s, float s2,
    const float* __restrict__ act, const unsigned (&q_row)[8],
    unsigned j, float (&o)[8][OG]
) {
    float a[R][OG], bb[R][OG];  // chains j + (j+16), (j+8) + (j+24)
#pragma unroll 1
    for (unsigned i = 0; i < 4; i++) {
        const unsigned c = i == 1 ? 2u : i == 2 ? 1u : i;  // chains j, j+16 | j+8, j+24
        const unsigned k16 = j + 8 * c;
        float x[R][OG];
#pragma unroll
        for (unsigned r = 0; r < R; r++)
#pragma unroll
            for (unsigned g = 0; g < OG; g++) x[r][g] = 0.0f;
        c8_dn_step<R, OG>(s_w, s_s, s2, act, q_row, k16, x);
        if (c == 0) c8_dn_step<R, OG>(s_w, s_s, s2, act, q_row, k16 + 32, x);
#pragma unroll
        for (unsigned r = 0; r < R; r++)
#pragma unroll
            for (unsigned g = 0; g < OG; g++) {
                if (i == 0) a[r][g] = x[r][g];
                else if (i == 1) a[r][g] = a[r][g] + x[r][g];
                else if (i == 2) bb[r][g] = x[r][g];
                else bb[r][g] = bb[r][g] + x[r][g];
            }
    }
#pragma unroll
    for (unsigned r = 0; r < R; r++)
#pragma unroll
        for (unsigned g = 0; g < OG; g++) {
            float v = a[r][g] + bb[r][g];
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off, 8);
            o[r][g] = v;
        }
}

template <unsigned RC, unsigned OG>
__device__ __forceinline__ void c8_down(C8_SD_ARGS) {
    static_assert(RC >= 1 && RC <= 8, "row chunk");
    constexpr unsigned ROW = C8_I / 2, K16 = C8_I / 16, TILE = 32 * OG;
    if (blockDim.x != 256) __trap();
    C8Unit u;
    if (!c8_unit(ws, u)) return;
    const unsigned slots = rows * top_k;
    const unsigned n0 = blockIdx.x * TILE;
    const unsigned char* B = u.shared() ? sh_down_packed : (const unsigned char*)packed_ptrs[u.expert];
    const unsigned char* S = u.shared() ? sh_down_scale : (const unsigned char*)scale_ptrs[u.expert];
    const float s2 = u.shared() ? sh_down_s2 : scale2_vals[u.expert];
    __nv_bfloat16* out = u.shared() ? sh_down_out : C;
    if (B == 0) {
        for (unsigned i = threadIdx.x; i < u.n * TILE; i += blockDim.x)
            out[(size_t)c8_slot(ws, u, i / TILE) * C8_H + n0 + i % TILE] = __float2bfloat16(0.0f);
        return;
    }
    // The tile: [TILE][320] packed, then [TILE][40] scales, each contiguous
    // in global.
    __shared__ __align__(16) unsigned char s_tile[TILE * (ROW + K16)];
    c8_stage(s_tile, B + (size_t)n0 * ROW, TILE * ROW);
    c8_stage(s_tile + TILE * ROW, S + (size_t)n0 * K16, TILE * K16);
    asm volatile("cp.async.commit_group;\n\tcp.async.wait_all;" ::: "memory");
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, j = lane & 7u;
    const unsigned ol = 4 * warp + (lane >> 3);
    const unsigned char* s_w = s_tile + ol * ROW;
    const unsigned char* s_s = s_tile + TILE * ROW + ol * K16;
    const float* act_base = act + (u.shared() ? (size_t)slots * C8_I : 0);
#pragma unroll 1
    for (unsigned c0 = 0; c0 < u.n; c0 += RC) {
        const unsigned nr = min(RC, u.n - c0);
        unsigned q_row[8];
#pragma unroll
        for (unsigned r = 0; r < 8; r++) q_row[r] = c8_slot(ws, u, c0 + min(r, nr - 1));
        float v[8][OG];
        switch (nr) {
#define C8_DN_CASE(R_) \
        case R_: if constexpr (R_ <= RC) c8_dn_chunk<R_, OG>(s_w, s_s, s2, act_base, q_row, j, v); break;
        C8_DN_CASE(1) C8_DN_CASE(2) C8_DN_CASE(3) C8_DN_CASE(4)
        C8_DN_CASE(5) C8_DN_CASE(6) C8_DN_CASE(7) C8_DN_CASE(8)
#undef C8_DN_CASE
        }
        // Group heads (lanes 0, 8, 16, 24) hold outputs n0 + 32 g + 4w + 0..3.
#pragma unroll
        for (unsigned r = 0; r < RC; r++) {
            if (r >= nr) break;
#pragma unroll
            for (unsigned g = 0; g < OG; g++) {
                const unsigned h = __bfloat16_as_ushort(__float2bfloat16(v[r][g]));
                const unsigned lo = h | (__shfl_down_sync(0xFFFFFFFF, h, 8) << 16);
                const unsigned hi = __shfl_down_sync(0xFFFFFFFF, lo, 16);
                if (lane == 0)
                    *(uint2*)(out + (size_t)q_row[r] * C8_H + n0 + 32 * g + 4 * warp) = make_uint2(lo, hi);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3) qwen4exp_moe_c8_gate_up(C8_GU_ARGS) {
    c8_gate_up<4>(A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs,
        up_scale_ptrs, up_scale2_vals, sh_gate_packed, sh_gate_scale, sh_gate_s2, sh_up_packed,
        sh_up_scale, sh_up_s2, ws, gate_out, up_out, sh_gate_out, sh_up_out, act, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(256, 3) qwen4exp_moe_c8_down(C8_SD_ARGS) {
    c8_down<2, 2>(act, packed_ptrs, scale_ptrs, scale2_vals, sh_down_packed, sh_down_scale,
        sh_down_s2, ws, C, sh_down_out, top_k, rows);
}
