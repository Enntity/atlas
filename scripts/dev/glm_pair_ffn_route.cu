// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_pair_ffn_api.h"
#include "../../kernels/gb10/common/dense_gemm_bf16.cu"
#include "../../kernels/gb10/common/moe_topk_sigmoid.cu"
#include "../../kernels/gb10/common/moe_permute.cu"
namespace pair_ffn {
void route(const Bf* a,const Bf* w,const float* bias,Bf* logits,unsigned* ids,float* weights,
           unsigned rows,cudaStream_t s) {
    // Literal qualified K5 router and the Joint generic order-preserving router.
    if(rows==5) dense_gemm_bf16_router_m5<<<E/16,dim3(16,5),0,s>>>(a,w,logits,5,E,H);
    else dense_gemm_bf16_router<<<dim3((E+63)/64,(rows+15)/16),dim3(16,16),0,s>>>(a,w,logits,rows,E,H);
    moe_topk_sigmoid_batched<<<rows,256,0,s>>>(logits,bias,ids,weights,E,K,1,1.0f);
}
void sort(const unsigned* ids,int* tok,int* exp,int* off,int* inv,unsigned rows,cudaStream_t s) {
    moe_sort_by_expert<<<1,256,0,s>>>(ids,tok,exp,off,inv,rows*K,E,K);
}
void work(const int* off,Table t,unsigned* list,int* total,cudaStream_t s) {
    moe_build_tile_worklist<<<1,256,0,s>>>(off,t.packed,list,total,E,I/128,64);
}
void unpermute(const Bf* expert,Bf* out,const int* inv,const unsigned* ids,const float* weights,
               unsigned rows,unsigned rank,cudaStream_t s) {
    moe_unpermute_reduce_indexed_ep<<<rows,256,0,s>>>(expert,out,inv,
        reinterpret_cast<const int*>(ids),weights,H,rows,K,rank*144,(rank+1)*144);
}
}
