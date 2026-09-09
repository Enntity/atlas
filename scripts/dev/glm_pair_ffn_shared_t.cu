// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_pair_ffn_api.h"
#include <cstdlib>
// Name-only isolation of the production constant, also defined by GEMV's TU.
#define E2M1_LUT pair_fixture_gemm_lut
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu"
#undef E2M1_LUT
namespace pair_ffn {
void shared_t(const Bf* a,Weight w,Bf* out,unsigned rows,unsigned n,unsigned k,cudaStream_t s) {
    if((rows!=5&&rows!=10&&rows!=15&&rows!=20)||!((n==I&&k==H)||(n==H&&k==I)))std::abort();
    w4a16_gemm_t<<<dim3(n/128,1),128,0,s>>>(a,w.packed,w.scale,w.scale2,out,rows,n,k,n);
}
}
