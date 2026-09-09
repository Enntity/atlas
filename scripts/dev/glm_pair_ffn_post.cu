// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_pair_ffn_api.h"
#include "../../kernels/gb10/common/bf16_add.cu"
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu"
// Avoid including moe_permute twice: its real kernel is defined by route TU.
extern "C" __global__ void moe_batched_blend(__nv_bfloat16*,const __nv_bfloat16*,
    const __nv_bfloat16*,const __nv_bfloat16*,unsigned,unsigned);
namespace pair_ffn {
void finish(Bf* local,const Bf* peer,const Bf* shared,const Bf* norm,const float* residual,
            const float* post,const float* comb,float* out,unsigned rows,bool joint,cudaStream_t s) {
    bf16_add_inplace<<<(rows*H+255)/256,256,0,s>>>(local,peer,rows*H);
    if(joint) {
        moe_batched_blend<<<rows,256,0,s>>>(local,shared,norm,nullptr,H,rows);
        hc_post<<<rows,256,0,s>>>(local,residual,post,comb,out,H,HC);
    } else {
        // Existing K5 FUSED_MOE_HC=1 control: same explicit BF16 blend round trip.
        hc_post_moe_blend<<<rows,256,0,s>>>(local,shared,norm,nullptr,residual,post,comb,out,H,HC);
    }
}
}
