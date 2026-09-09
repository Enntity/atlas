// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_pair_ffn_api.h"
// Name-only isolation of the production constant, also defined by GEMV's TU.
#define E2M1_LUT pair_fixture_gemm_lut
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu"
#undef E2M1_LUT
namespace pair_ffn {
void shared_t(const Bf* a,Weight w,Bf* out,unsigned n,unsigned k,cudaStream_t s) {
    w4a16_gemm_t<<<dim3(n/128,1),128,0,s>>>(a,w.packed,w.scale,w.scale2,out,5,n,k,n);
}
}
