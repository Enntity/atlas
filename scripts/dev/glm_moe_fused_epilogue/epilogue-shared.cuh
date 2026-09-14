// SPDX-License-Identifier: AGPL-3.0-only
// Fused epilogue: decode pre-packed BF16 gate pairs, load pre-rounded BF16 up
// tiles, apply SiLU(gate)*up, NVFP4-quantize per 16-element group with e4m3 scales.
__device__ __forceinline__ void atlas_fused_epilogue(
    const unsigned gates[16][2], const __nv_bfloat16* up_tile,
    unsigned char* packed, unsigned char* scales,
    unsigned cta_m, unsigned cta_n, unsigned cta_m_local,
    int M_expert, unsigned N)
{
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int gid  = lane >> 2;          // 0..7  (row within warp's 16)
    const int tid  = lane & 3;           // 0..3  (quad sub-lane)
    const unsigned row0 = cta_m + (unsigned)(warp * 16 + gid);
    const unsigned row1 = row0 + 8u;
    const bool ok0 = ((int)(warp * 16 + gid) + (int)cta_m_local) < M_expert;
    const bool ok1 = ((int)(warp * 16 + gid + 8) + (int)cta_m_local) < M_expert;
    const float SWIGLU_LIMIT = 10.0f;
    const unsigned lane_mask = 0xffffffffu;

#pragma unroll
    for (int nt = 0; nt < 16; nt += 2) {
        // ---- decode + compute 4 values per row (BF16 round of up before silu) ----
        float vals[4];
        {
            const unsigned g01 = gates[nt][0];
            const unsigned g23 = gates[nt][1];
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned gi = (i == 0) ? (g01 & 0xFFFFu)
                                  : (i == 1) ? (g01 >> 16)
                                  : (i == 2) ? (g23 & 0xFFFFu)
                                             : (g23 >> 16);
                float g = __bfloat162float(__ushort_as_bfloat16((unsigned short)gi));
                float u = __bfloat162float(up_tile[(warp * 16 + gid + (i >= 2 ? 8 : 0)) * 128 + nt * 8 + tid * 2 + (i & 1)]);
                g = fminf(g, SWIGLU_LIMIT);
                u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
                const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
                const __nv_bfloat16 r16 = __float2bfloat16(g * sigmoid_g * u);
                v[i] = __bfloat162float(r16);
            }
            vals[0] = v[0]; vals[1] = v[1]; vals[2] = v[2]; vals[3] = v[3];
        }
        // also decode nt+1
        float vals2[4];
        {
            const unsigned g01 = gates[nt + 1][0];
            const unsigned g23 = gates[nt + 1][1];
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned gi = (i == 0) ? (g01 & 0xFFFFu)
                                  : (i == 1) ? (g01 >> 16)
                                  : (i == 2) ? (g23 & 0xFFFFu)
                                             : (g23 >> 16);
                float g = __bfloat162float(__ushort_as_bfloat16((unsigned short)gi));
                float u = __bfloat162float(up_tile[(warp * 16 + gid + (i >= 2 ? 8 : 0)) * 128 + (nt + 1) * 8 + tid * 2 + (i & 1)]);
                g = fminf(g, SWIGLU_LIMIT);
                u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
                const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
                const __nv_bfloat16 r16 = __float2bfloat16(g * sigmoid_g * u);
                v[i] = __bfloat162float(r16);
            }
            vals2[0] = v[0]; vals2[1] = v[1]; vals2[2] = v[2]; vals2[3] = v[3];
        }

        // x[0,1] from nt, x[2,3] from nt+1 (row0 for i<2, row1 for i>=2)
        // For each row: own 4 values are the two pairs.
        // row0: nhalf = {vals[0], vals[1]} (nt pair), plus next group's pair {vals2[0], vals2[1]}
        // row1: {vals[2], vals[3]}, {vals2[2], vals2[3]}
        float xr0[4] = { vals[0], vals[1], vals2[0], vals2[1] };
        float xr1[4] = { vals[2], vals[3], vals2[2], vals2[3] };

        // ---- per-row maxabs, then butterfly across quads (lanes 0..3 same row) ----
        float m0 = fmaxf(fmaxf(fabsf(xr0[0]), fabsf(xr0[1])), fmaxf(fabsf(xr0[2]), fabsf(xr0[3])));
        float m1 = fmaxf(fmaxf(fabsf(xr1[0]), fabsf(xr1[1])), fmaxf(fabsf(xr1[2]), fabsf(xr1[3])));
        m0 = fmaxf(m0, __shfl_xor_sync(lane_mask, m0, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(lane_mask, m1, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(lane_mask, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(lane_mask, m1, 2));

        // ---- decoded scale / inverse ----
        const unsigned char fp8_0 = silu_nvfp4_float_to_e4m3(m0 / 6.0f);
        const unsigned char fp8_1 = silu_nvfp4_float_to_e4m3(m1 / 6.0f);
        const unsigned sbase = row0 * (N / 16u) + (cta_n + (unsigned)nt * 8u) / 16u;
        const unsigned sbase1 = row1 * (N / 16u) + (cta_n + (unsigned)nt * 8u) / 16u;
        if (tid == 0) {
            if (ok0) scales[sbase] = fp8_0;
            if (ok1) scales[sbase1] = fp8_1;
        }
        const unsigned exp0 = (fp8_0 >> 3) & 0xF, man0 = fp8_0 & 0x7;
        const unsigned exp1 = (fp8_1 >> 3) & 0xF, man1 = fp8_1 & 0x7;
        float dec0, dec1;
        if (exp0 == 0) dec0 = (float)man0 * 0.001953125f;
        else if (exp0 == 15 && man0 == 7) dec0 = 0.0f;
        else dec0 = __uint_as_float((exp0 + 120u) << 23 | (man0 << 20));
        if (exp1 == 0) dec1 = (float)man1 * 0.001953125f;
        else if (exp1 == 15 && man1 == 7) dec1 = 0.0f;
        else dec1 = __uint_as_float((exp1 + 120u) << 23 | (man1 << 20));
        const float inv0 = dec0 > 0.0f ? 1.0f / dec0 : 0.0f;
        const float inv1 = dec1 > 0.0f ? 1.0f / dec1 : 0.0f;

        // ---- pack 4 values -> 2 bytes per row ----
        const unsigned pbase0 = row0 * (N / 2u) + (cta_n + (unsigned)nt * 8u) / 2u + (unsigned)tid;
        const unsigned pbase1 = row1 * (N / 2u) + (cta_n + (unsigned)nt * 8u) / 2u + (unsigned)tid;
        const unsigned int q00 = silu_nvfp4_quantize_e2m1(xr0[0] * inv0);
        const unsigned int q01 = silu_nvfp4_quantize_e2m1(xr0[1] * inv0);
        const unsigned int q02 = silu_nvfp4_quantize_e2m1(xr0[2] * inv0);
        const unsigned int q03 = silu_nvfp4_quantize_e2m1(xr0[3] * inv0);
        const unsigned int q10 = silu_nvfp4_quantize_e2m1(xr1[0] * inv1);
        const unsigned int q11 = silu_nvfp4_quantize_e2m1(xr1[1] * inv1);
        const unsigned int q12 = silu_nvfp4_quantize_e2m1(xr1[2] * inv1);
        const unsigned int q13 = silu_nvfp4_quantize_e2m1(xr1[3] * inv1);
        const unsigned char b0 = (unsigned char)((q01 << 4) | q00);
        const unsigned char b1 = (unsigned char)((q03 << 4) | q02);
        const unsigned char c0 = (unsigned char)((q11 << 4) | q10);
        const unsigned char c1 = (unsigned char)((q13 << 4) | q12);
        if (ok0) { packed[pbase0] = b0; packed[pbase0 + 4u] = b1; }
        if (ok1) { packed[pbase1] = c0; packed[pbase1 + 4u] = c1; }
    }
}
