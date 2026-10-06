// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) routed + shared MoE GEMVs for several rows
// in ONE launch over the union of their experts, on the ORIGINALS expert
// layout ([N, K/2] packed, [N, K/16] scales)
// (ATLAS_QWEN4EXP_BATCH_FAST=1, `layers/moe/forward_rows.rs`).
//
// qwen4exp_moe_rows_gate_up / qwen4exp_moe_rows_silu_down replace, BIT FOR
// BIT, one launch per row of moe_expert_gate_up_shared /
// moe_expert_silu_down_shared (common/moe_shared_expert_fused.cu), the
// kernels single-sequence decode runs (`MoeLayer::forward`; TP1, and TP2's
// hybrid layout). Every CTA is that kernel's CTA for one (row, slot) -- the
// body is the same, statement for statement, only the row's activation and
// output pointers and the expert id come from a table -- so each output is
// the same IEEE operation sequence (the target builds with --fmad=false).
//
// What changes is the ORDER the CTAs run in. `qwen4exp_moe_rows_plan` sorts
// the launch's (row, slot) entries by expert id, and every row's
// shared-expert CTA follows; CTAs launch in that order (x, the output tile,
// fastest). So the rows that picked one expert, and all rows' shared expert,
// run back to back: the expert's weights stream from DRAM once and the
// co-resident CTAs of the other rows read them from L2 -- the union of the
// rows' experts at the single-row kernels' register footprint and occupancy.
// One launch per projection pair also replaces two per row.
//
// (A one-CTA-per-expert form that keeps an accumulator pair per row was
// built first and measured 0.50-0.85x of the per-row loop on GB10: the extra
// registers cost more occupancy than the shared reads saved, on kernels that
// are memory-latency bound -- scripts/dev/qwen4exp_batch_exact_bench.cu.)
//
// Layouts (rows r < R, slot s < top_k, q = r * top_k + s):
//   A            [R, K]           BF16 activation rows
//   expert_ids   [R * top_k]      u32, each row's own top-k
//   order        [R * top_k]      u32 scratch, written by the plan kernel
//   gate/up_out  [R * top_k, N]   row q = row r's slot s (the per-row kernel's
//                                 [top_k, N] block of row r, stacked)
//   sh_*_out     [R, N]
//   down C       [R * top_k, N], sh_down_out [R, N]
// EP: a NULL packed pointer is a remote expert; its outputs are zero, as the
// per-row kernel writes them.
//
// Grid: plan (1), (128); gate_up (ceil(N/8), R*top_k + R, 2); silu_down
// (ceil(N/8), R*top_k + R, 1) with K*4 bytes of dynamic shared memory.
// Block 128.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QR_BLOCK 128
#define QR_N_PER_BLOCK 4
#define QR_WARP 32
#define QR_GROUP 16

__device__ __constant__ float QR_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// The replaced kernels' atlas_dec_e4m3 on NVIDIA: the verbatim cast.
__device__ __forceinline__ float qr_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}

// order[rank] = q, ranks by (expert id, q): a stable sort of the entries by
// expert. One block; R * top_k is at most a few hundred.
extern "C" __global__ void qwen4exp_moe_rows_plan(
    const unsigned int* __restrict__ expert_indices,
    unsigned int* __restrict__ order,
    unsigned int slots
) {
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) {
        const unsigned e = expert_indices[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < slots; j++) {
            const unsigned f = expert_indices[j];
            rank += (f < e) || (f == e && j < t);
        }
        order[rank] = t;
    }
}

extern "C" __global__ void qwen4exp_moe_rows_gate_up(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ order,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows
) {
    const unsigned slots = rows * top_k;
    const bool is_shared = blockIdx.y >= slots;
    const unsigned q = is_shared ? 0u : order[blockIdx.y];
    const unsigned row = is_shared ? blockIdx.y - slots : q / top_k;
    const unsigned proj = blockIdx.z;
    A += (unsigned long long)row * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    if (is_shared) {
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = ((proj == 0) ? sh_gate_out : sh_up_out) + (unsigned long long)row * N;
            const unsigned n_base = blockIdx.x * (QR_N_PER_BLOCK * 2);
            for (unsigned i = threadIdx.x; i < QR_N_PER_BLOCK * 2 && n_base + i < N; i += QR_BLOCK)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) { B_packed = sh_gate_packed; B_scale = sh_gate_scale; s2 = sh_gate_s2; C = sh_gate_out; }
        else           { B_packed = sh_up_packed;   B_scale = sh_up_scale;   s2 = sh_up_s2;   C = sh_up_out; }
        C += (unsigned long long)row * N;
    } else {
        const unsigned expert_id = expert_indices[q];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        C += (unsigned long long)q * N;
        if (B_packed == 0) {
            const unsigned n_base = blockIdx.x * (QR_N_PER_BLOCK * 2);
            for (unsigned i = threadIdx.x; i < QR_N_PER_BLOCK * 2 && n_base + i < N; i += QR_BLOCK)
                C[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned threads_per_out = QR_BLOCK / QR_N_PER_BLOCK;
    const unsigned local_out = threadIdx.x / threads_per_out;
    const unsigned lane = threadIdx.x % threads_per_out;
    const unsigned n1 = blockIdx.x * (QR_N_PER_BLOCK * 2) + local_out * 2;
    const unsigned n2 = n1 + 1;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QR_E2M1_LUT[threadIdx.x];
    __syncthreads();
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    const unsigned half_K = K / 2;
    const unsigned num_groups = K / QR_GROUP;
    const unsigned K16 = K / 16;

    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        const unsigned sg = base_k / QR_GROUP;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = qr_dec_e4m3(sb1) * s2;
        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? qr_dec_e4m3(sb2) * s2 : 0.0f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }
#pragma unroll
    for (int offset = QR_WARP / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);
    if (have_n2) {
#pragma unroll
        for (int offset = QR_WARP / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

extern "C" __global__ void qwen4exp_moe_rows_silu_down(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const unsigned int* __restrict__ order,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows
) {
    const unsigned slots = rows * top_k;
    const bool is_shared = blockIdx.y >= slots;
    const unsigned q = is_shared ? 0u : order[blockIdx.y];
    const unsigned row = is_shared ? blockIdx.y - slots : q / top_k;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    __nv_bfloat16* out;
    if (is_shared) {
        out = sh_down_out + (unsigned long long)row * N;
        if (sh_down_packed == 0) {
            const unsigned n_base = blockIdx.x * (QR_N_PER_BLOCK * 2);
            for (unsigned i = threadIdx.x; i < QR_N_PER_BLOCK * 2 && n_base + i < N; i += QR_BLOCK)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)row * K;
        u_ptr = sh_up_in + (unsigned long long)row * K;
    } else {
        const unsigned expert_id = expert_indices[q];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)q * K;
        u_ptr = up_out + (unsigned long long)q * K;
        out = C + (unsigned long long)q * N;
        if (B_packed == 0) {
            const unsigned n_base = blockIdx.x * (QR_N_PER_BLOCK * 2);
            for (unsigned i = threadIdx.x; i < QR_N_PER_BLOCK * 2 && n_base + i < N; i += QR_BLOCK)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned threads_per_out = QR_BLOCK / QR_N_PER_BLOCK;
    const unsigned local_out = threadIdx.x / threads_per_out;
    const unsigned lane = threadIdx.x % threads_per_out;
    const unsigned n1 = blockIdx.x * (QR_N_PER_BLOCK * 2) + local_out * 2;
    const unsigned n2 = n1 + 1;
    const bool have_n2 = (n2 < N);
    const unsigned half_K = K / 2;
    const unsigned num_groups = K / QR_GROUP;
    const unsigned K16 = K / 16;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QR_E2M1_LUT[threadIdx.x];
    // The replaced kernel's expression and clamp verbatim (see its note on
    // why the clamp skips the shared expert).
    const float SWIGLU_LIMIT = 10.0f;
    for (unsigned i = threadIdx.x; i < K; i += QR_BLOCK) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        if (!is_shared) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();
    if (n1 >= N) return;

    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned base_k = k16 * 16;
        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        const unsigned sg = base_k / QR_GROUP;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = qr_dec_e4m3(sb1) * s2;
        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? qr_dec_e4m3(sb2) * s2 : 0.0f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }
#pragma unroll
    for (int offset = QR_WARP / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);
    if (have_n2) {
#pragma unroll
        for (int offset = QR_WARP / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}
