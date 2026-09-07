// SPDX-License-Identifier: AGPL-3.0-only
// Test-only BF16-input compatibility; include after production scalar oracle.
#pragma once

#define BTDEC_ARGS \
 const __nv_bfloat16* __restrict__ A, \
 const unsigned long long* __restrict__ gp,const unsigned long long* __restrict__ gs,const float* __restrict__ gf,__nv_bfloat16* __restrict__ go, \
 const unsigned long long* __restrict__ up,const unsigned long long* __restrict__ us,const float* __restrict__ uf,__nv_bfloat16* __restrict__ uo, \
 const unsigned int* __restrict__ ids, \
 const unsigned char* __restrict__ sgp,const unsigned char* __restrict__ sgs,float sgf,__nv_bfloat16* __restrict__ sgo, \
 const unsigned char* __restrict__ sup,const unsigned char* __restrict__ sus,float suf,__nv_bfloat16* __restrict__ suo, \
 unsigned int N,unsigned int K,unsigned int top_k
#define BTDEC_PASS A,gp,gs,gf,go,up,us,uf,uo,ids,sgp,sgs,sgf,sgo,sup,sus,suf,suo,N,K,top_k

template<unsigned ROWS,bool STAGE>
__device__ __forceinline__ void btile_decode_impl(BTDEC_ARGS){
    // Host requires exact N2048/K4096; retain dynamic arithmetic dimensions.
    if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||N%128||K%64||!top_k)return;
    const unsigned y=blockIdx.y,proj=blockIdx.z;
    if(y>=ROWS*(top_k+1)||proj>=2)return;
    const bool shared=y>=ROWS*top_k;
    const unsigned token=shared?y-ROWS*top_k:y/top_k;
    const unsigned n=blockIdx.x*32+threadIdx.x;
    if(n>=N)return; // N multiple128 makes the complete CTA inactive.
    const __nv_bfloat16* A_token=A+(unsigned long long)token*K;
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;__nv_bfloat16* C;unsigned long long c_offset;
    if(shared){
        B_packed=proj?sup:sgp;B_scale=proj?sus:sgs;s2=proj?suf:sgf;
        C=proj?suo:sgo;c_offset=(unsigned long long)token*N;
    }else{
        const unsigned expert=ids[y];
        B_packed=(const unsigned char*)(proj?up[expert]:gp[expert]);
        B_scale=(const unsigned char*)(proj?us[expert]:gs[expert]);
        s2=proj?uf[expert]:gf[expert];C=proj?uo:go;c_offset=(unsigned long long)y*N;
    }
    if(!B_packed){C[c_offset+n]=__float2bfloat16(0.0f);return;}
    __shared__ float btdec_lut[16];
    // 9 words/row avoid the 8-way bank alias of an unpadded32-byte row.
    __shared__ unsigned btdec_stage[STAGE?32:1][STAGE?9:1];
    if(threadIdx.x<16)btdec_lut[threadIdx.x]=E2M1_LUT_T[threadIdx.x];
    __syncthreads();
    const unsigned num_groups=K/16;
    float acc=0.0f;
    for(unsigned sg=0;sg<num_groups;++sg){
        if constexpr(STAGE){
            if(!shared&&sg%4==0){
                // All32 lanes finished prior stage before any overwrites.
                __syncthreads();
                const unsigned long long base=((unsigned long long)(n/128)*(K/64)+sg/4)*4096+(n%128/32)*1024;
                #pragma unroll
                for(unsigned round=0;round<2;++round){
                    const unsigned vi=round*32+threadIdx.x,row=vi/2,col=(vi%2)*4;
                    const uint4 v=((const uint4*)(B_packed+base))[vi];
                    btdec_stage[row][col]=v.x;btdec_stage[row][col+1]=v.y;
                    btdec_stage[row][col+2]=v.z;btdec_stage[row][col+3]=v.w;
                }
                __syncthreads();
            }
        }
        unsigned char sb=B_scale[(unsigned long long)sg*N+n];
        float sc=atlas_dec_e4m3(sb)*s2;
        const unsigned kh_base=sg*8;
        #pragma unroll
        for(unsigned kh_off=0;kh_off<8;++kh_off){
            unsigned k_half=kh_base+kh_off;
            unsigned char byte;
            if(shared)byte=B_packed[(unsigned long long)k_half*N+n];
            else if constexpr(STAGE)byte=((const unsigned char*)btdec_stage[threadIdx.x])[k_half%32];
            else byte=B_packed[(((unsigned long long)(n/128)*(K/64)+k_half/32)*128+n%128)*32+k_half%32];
            float a_lo=__bfloat162float(A_token[k_half*2]);
            float a_hi=__bfloat162float(A_token[k_half*2+1]);
            float w_lo=btdec_lut[byte&0xFu]*sc;
            float w_hi=btdec_lut[(byte>>4)&0xFu]*sc;
            acc+=a_lo*w_lo+a_hi*w_hi;
        }
    }
    C[c_offset+n]=__float2bfloat16(acc);
}

#define BTDEC_EXPORT(ROWS) \
extern "C" __global__ void glm_btile_decode_direct##ROWS(BTDEC_ARGS){btile_decode_impl<ROWS,false>(BTDEC_PASS);} \
extern "C" __global__ void glm_btile_decode_stage##ROWS(BTDEC_ARGS){btile_decode_impl<ROWS,true>(BTDEC_PASS);}
BTDEC_EXPORT(1)
BTDEC_EXPORT(2)
BTDEC_EXPORT(3)
#undef BTDEC_EXPORT
#undef BTDEC_PASS
#undef BTDEC_ARGS
