// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_pair_ffn_api.h"
#include "../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu"
// GLM's actual inherited target is this clamped shadow, not common SiLU.
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu"
namespace pair_ffn {
void quant(const Bf* a,unsigned char* p,unsigned char* sc,unsigned rows,unsigned cols,cudaStream_t s) {
    quantize_bf16_to_nvfp4<<<rows,128,0,s>>>(a,p,sc,1.0f,rows,cols);
}
void silu_quant(const Bf* g,const Bf* u,unsigned char* p,unsigned char* sc,unsigned rows,cudaStream_t s) {
    silu_mul_quant_nvfp4<<<rows*K,128,0,s>>>(g,u,p,sc,nullptr,rows*K,I);
}
void activation(Bf* g,const Bf* u,unsigned rows,cudaStream_t s) {
    moe_silu_mul<<<(rows*I+255)/256,256,0,s>>>(g,u,g,rows*I);
}
}
