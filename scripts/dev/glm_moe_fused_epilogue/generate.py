# SPDX-License-Identifier: AGPL-3.0-only
"""Derive one standalone fused CTA from the frozen production MMA body."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
source=here.parents[2]/'kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='bad2afc59183a4b46fd094864ef88ae428517a23dee0d00f1d9e87de7503b33b'
s=source.read_text()
a=s.index('template<bool PQ_VEC_SCALES>\n__device__ __forceinline__ void moe_w4a4_grouped_gemm_prequant_t_k64_impl(')
b=s.index('\n#define PQ4_PREQUANT_ARGS',a)
body=s[a:b].replace('moe_w4a4_grouped_gemm_prequant_t_k64_impl','atlas_dev_moe_fused_impl')
body=body.replace('    __nv_bfloat16* __restrict__ C,','''    const unsigned long long* __restrict__ U_packed_ptrs,
    const unsigned long long* __restrict__ U_scale_ptrs,
    const float* __restrict__ U_scale2_vals,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,''',1)
a=body.index('    const unsigned char* B_expert =')
b=body.index('    const unsigned int warp_id',a)
body=body[:a]+'''    if (!B_packed_ptrs[expert_id] || !U_packed_ptrs[expert_id]) return;

'''+body[b:]
a=body.index('    float acc[16][4];')
body=body[:a]+'''    // Exactly rounded gate results: 64 BF16 values in32 register words/lane.
    unsigned gate_values[16][2];
    #pragma unroll 1
    for (int projection = 0; projection < 2; ++projection) {
    const unsigned char* B_expert = (const unsigned char*)(projection
        ? U_packed_ptrs[expert_id] : B_packed_ptrs[expert_id]);
    const unsigned char* S_expert = (const unsigned char*)(projection
        ? U_scale_ptrs[expert_id] : B_scale_ptrs[expert_id]);
    const float scale2 = projection ? U_scale2_vals[expert_id] : scale2_vals[expert_id];
'''+body[a:]
a=body.index('    #pragma unroll\n    for (int nt = 0; nt < 16; nt++)',body.index('    #undef PQ4_COMPUTE_MMA'))
body=body[:a]+'''    if (projection == 0) {
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            gate_values[nt][0] = (unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][0] * scale2))
                | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][1] * scale2)) << 16);
            gate_values[nt][1] = (unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][2] * scale2))
                | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(acc[nt][3] * scale2)) << 16);
        }
    } else {
        atlas_fused_epilogue(gate_values, acc, scale2, packed_out, scale_out,
            cta_m, cta_n, cta_m_local, M_expert, N);
    }
    // Do not overwrite shared MMA operands until every warp finished this pass.
    __syncthreads();
    }
}
'''
wrapper='''
extern "C" __global__ void atlas_dev_moe_fused_gate_up(
    const unsigned char* A, const unsigned char* As,
    const unsigned long long* G, const unsigned long long* Gs, const float* G2,
    const unsigned long long* U, const unsigned long long* Us, const float* U2,
    unsigned char* packed, unsigned char* scales, const int* offsets,
    const int* ids, unsigned experts, unsigned N, unsigned K) {
    atlas_dev_moe_fused_impl<true>(A, As, G, Gs, G2, U, Us, U2,
        packed, scales, offsets, ids, experts, N, K, blockIdx.z, blockIdx.y, blockIdx.x);
}
'''
out='// SPDX-License-Identifier: AGPL-3.0-only\n// Generated from frozen production source; never substitute for its baseline.\n'+(here/'epilogue.cuh').read_text()+'\n'+body+wrapper
(here/'generated.cuh').write_text(out)
print(hashlib.sha256(out.encode()).hexdigest())
