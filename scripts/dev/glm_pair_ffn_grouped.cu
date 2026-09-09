// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_pair_ffn_api.h"
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
namespace pair_ffn {
void gate_up(const unsigned char* a,const unsigned char* as,Table g,Table u,Bf* go,Bf* uo,
             const int* off,const int* tok,const unsigned* list,const int* total,
             unsigned rows,bool vector,cudaStream_t s) {
    if(rows==5) {
        // Literal native OFF: K5_COMPACT_MOE=0/FUSED_COMPACT_GATE_UP=0.
        // Two dense expert-grid launches, same native FP4 MMA, no worklist.
        for(unsigned p=0;p<2;++p) {
            Table t=p?u:g; Bf* out=p?uo:go;
            if(vector)moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dim3(I/128,1,E),128,0,s>>>(
                a,as,t.packed,t.scale,t.scale2,out,off,tok,E,I,H);
            else moe_w4a4_grouped_gemm_prequant_t_k64<<<dim3(I/128,1,E),128,0,s>>>(
                a,as,t.packed,t.scale,t.scale2,out,off,tok,E,I,H);
        }
        return;
    }
    const unsigned cap=rows*K*(I/128);
    if(vector) moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up<<<dim3(cap,2),128,0,s>>>(
        a,as,g.packed,g.scale,g.scale2,go,u.packed,u.scale,u.scale2,uo,off,tok,E,I,H,list,total,cap);
    else moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up<<<dim3(cap,2),128,0,s>>>(
        a,as,g.packed,g.scale,g.scale2,go,u.packed,u.scale,u.scale2,uo,off,tok,E,I,H,list,total,cap);
}
void down(const unsigned char* a,const unsigned char* as,Table d,Bf* out,const int* off,
          bool vector,cudaStream_t s) {
    // Literal production native prequant DOWN: dense M64 grid, not compact-down.
    // Unique top8 gives <=10 rows/expert, so one M64 tile; no offset D2H in compute.
    if(vector) moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dim3(H/128,1,E),128,0,s>>>(
        a,as,d.packed,d.scale,d.scale2,out,off,nullptr,E,H,I);
    else moe_w4a4_grouped_gemm_prequant_t_k64<<<dim3(H/128,1,E),128,0,s>>>(
        a,as,d.packed,d.scale,d.scale2,out,off,nullptr,E,H,I);
}
}
