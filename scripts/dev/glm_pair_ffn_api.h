// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace pair_ffn {
using Bf = __nv_bfloat16;
using U64 = unsigned long long;
constexpr unsigned H=4096, I=2048, E=288, R=10, K=8, X=R*K, HC=4;
struct Table { U64 *packed, *scale; float* scale2; };
struct Weight { const unsigned char *packed, *scale; float scale2; };
void route(const Bf*,const Bf*,const float*,Bf*,unsigned*,float*,unsigned,cudaStream_t);
void sort(const unsigned*,int*,int*,int*,int*,unsigned,cudaStream_t);
void work(const int*,Table,unsigned*,int*,cudaStream_t);
void quant(const Bf*,unsigned char*,unsigned char*,unsigned,unsigned,cudaStream_t);
void gate_up(const unsigned char*,const unsigned char*,Table,Table,Bf*,Bf*,const int*,
             const int*,const unsigned*,const int*,unsigned,bool,cudaStream_t);
void silu_quant(const Bf*,const Bf*,unsigned char*,unsigned char*,unsigned,cudaStream_t);
void down(const unsigned char*,const unsigned char*,Table,Bf*,const int*,bool,cudaStream_t);
void shared_t(const Bf*,Weight,Bf*,unsigned,unsigned,cudaStream_t);
void activation(Bf*,const Bf*,unsigned,cudaStream_t);
void unpermute(const Bf*,Bf*,const int*,const unsigned*,const float*,unsigned,unsigned,cudaStream_t);
void finish(Bf*,const Bf*,const Bf*,const Bf*,const float*,const float*,const float*,float*,
            unsigned,bool,cudaStream_t);
}
