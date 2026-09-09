// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "glm_pair_ffn_api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
namespace pair_ffn {
inline void require(bool ok,const char* text) {
    if(!ok) { std::fprintf(stderr,"FAIL: %s\n",text); std::exit(2); }
}
inline void checked(cudaError_t e,const char* where) {
    if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s: %s\n",where,cudaGetErrorString(e)); std::exit(1); }
}
#define PCHECK(x) ::pair_ffn::checked((x),#x)
constexpr size_t limit=192ULL*1024*1024, packed=size_t(H)*I/2, scales=size_t(H)*I/16;
inline size_t live=0,peak=0;
inline bool fits(size_t count,size_t element,size_t used) {
    return element && used<=limit && count<=(limit-256)/element
        && count*element+256<=limit-used;
}
// Same 128-byte guarded allocation scheme as bench_glm_moe_{gate_up_m16,down_cost}.
// All payloads, tables and both shared layouts are real allocations counted here.
template<class T> struct Buffer {
    T *allocation,*ptr; size_t count,bytes;
    explicit Buffer(size_t n):count(n) {
        require(fits(n,sizeof(T),live),"192MiB allocation budget/overflow");
        bytes=n*sizeof(T)+256;
        PCHECK(cudaMalloc(&allocation,bytes)); live+=bytes; peak=std::max(peak,live);
        PCHECK(cudaMemset(allocation,0xa5,bytes)); ptr=allocation+128/sizeof(T);
    }
    ~Buffer() { cudaFree(allocation); live-=bytes; }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<T>& v,size_t start=0) {
        require(start<=count && v.size()<=count-start,"upload bounds");
        PCHECK(cudaMemcpy(ptr+start,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> read(size_t n=0,size_t start=0) const {
        if(!n)n=count;
        require(start<=count && n<=count-start,"read bounds"); std::vector<T> v(n);
        PCHECK(cudaMemcpy(v.data(),ptr+start,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
    }
    void guards() const {
        std::array<unsigned char,128> a,b;
        PCHECK(cudaMemcpy(a.data(),allocation,128,cudaMemcpyDeviceToHost));
        PCHECK(cudaMemcpy(b.data(),ptr+count,128,cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<128;++i) require(a[i]==0xa5 && b[i]==0xa5,"allocation canary");
    }
};
inline float f32(Bf x) { return __bfloat162float(x); }
inline Bf bf(float x) { return __float2bfloat16(x); }
inline unsigned char code(size_t i,unsigned seed) {
    unsigned x=unsigned(i)^(seed*0x9e3779b9u); x^=x>>16; x*=0x7feb352du; x^=x>>15;
    return static_cast<unsigned char>(x^(x>>8));
}
inline unsigned char scale_code(size_t i,unsigned seed) { return 0x18+8*((i+seed)%3); }
inline float tensor_scale(unsigned seed) { return std::ldexp(1.0f,int(seed%3)-2); }
inline double e2m1(unsigned c) {
    const double lut[]={0,.5,1,1.5,2,3,4,6}; return (c&8?-1:1)*lut[c&7];
}
inline double e4m3(unsigned c) {
    require(c<127,"finite positive FP8 scale"); unsigned e=c>>3,m=c&7;
    return e?std::ldexp(1.0+double(m)/8,int(e)-7):std::ldexp(double(m),-9);
}
struct Tables {
    Buffer<U64> p{E},s{E}; Buffer<float> f{E};
    Table view() { return {p.ptr,s.ptr,f.ptr}; }
    void guards() { p.guards();s.guards();f.guards(); }
};
struct Weights {
    Buffer<unsigned char> routed_p{24*packed},routed_s{24*scales};
    Buffer<unsigned char> shared_p{3*packed},shared_s{3*scales},shared_tp{3*packed},shared_ts{3*scales};
    Buffer<Bf> router{E*H}; Buffer<float> bias{E};
    std::array<std::array<std::unique_ptr<Tables>,3>,2> tables;
    std::array<unsigned,8> ids{};
    static unsigned seed(unsigned projection,unsigned weight) { return 31+projection*8+weight; }
    Weights() {
        for(auto& rank:tables) for(auto& table:rank) table=std::make_unique<Tables>();
        // Every expert has its OWN complete packed/scales projection; no alias replicas.
        for(unsigned p=0;p<3;++p)for(unsigned w=0;w<8;++w) initialize(p,w,false);
        for(unsigned p=0;p<3;++p) initialize(p,0,true);
        std::vector<Bf> r(E*H,bf(0));
        for(unsigned e=0;e<E;++e)for(unsigned k=0;k<64;++k)
            r[e*H+k]=bf(float(int((e*7+k*3)%17)-8)/128);
        router.upload(r);
    }
    void initialize(unsigned projection,unsigned weight,bool shared) {
        const unsigned n=projection==2?H:I,k=projection==2?I:H;
        const unsigned salt=shared?100+projection:seed(projection,weight);
        std::vector<unsigned char> p(packed),s(scales),pt(packed),st(scales);
        for(unsigned row=0;row<n;++row) {
            for(unsigned j=0;j<k/2;++j) {
                const size_t index=size_t(row)*(k/2)+j;
                p[index]=code(index,salt); pt[size_t(j)*n+row]=p[index];
            }
            for(unsigned j=0;j<k/16;++j) {
                const size_t index=size_t(row)*(k/16)+j;
                s[index]=scale_code(index,salt); st[size_t(j)*n+row]=s[index];
            }
        }
        if(shared) {
            shared_p.upload(p,projection*packed); shared_s.upload(s,projection*scales);
            shared_tp.upload(pt,projection*packed);shared_ts.upload(st,projection*scales);
        } else {
            routed_p.upload(pt,(projection*8+weight)*packed);
            routed_s.upload(st,(projection*8+weight)*scales);
        }
    }
    Weight shared(unsigned p,bool transposed) {
        return {(transposed?shared_tp.ptr:shared_p.ptr)+p*packed,
            (transposed?shared_ts.ptr:shared_s.ptr)+p*scales,tensor_scale(100+p)};
    }
    void select(unsigned test) {
        const std::array<unsigned,8> edge={0,17,142,143,144,145,286,287};
        for(unsigned w=0;w<8;++w) ids[w]=test==0?edge[w]:(test==1?w:280+w);
        std::vector<float> bias_host(E,-2.0f);
        for(auto e:ids)bias_host[e]=2.0f; bias.upload(bias_host);
        for(unsigned rank=0;rank<2;++rank)for(unsigned p=0;p<3;++p) {
            std::vector<U64> a(E),b(E);std::vector<float> f(E,1);
            for(unsigned w=0;w<8;++w) if(ids[w]/144==rank) {
                a[ids[w]]=reinterpret_cast<U64>(routed_p.ptr+(p*8+w)*packed);
                b[ids[w]]=reinterpret_cast<U64>(routed_s.ptr+(p*8+w)*scales);
                f[ids[w]]=tensor_scale(seed(p,w));
            }
            tables[rank][p]->p.upload(a);tables[rank][p]->s.upload(b);tables[rank][p]->f.upload(f);
        }
    }
    void guards() {
        routed_p.guards();routed_s.guards();shared_p.guards();shared_s.guards();shared_tp.guards();shared_ts.guards();
        router.guards();bias.guards();for(auto& rank:tables)for(auto& t:rank)t->guards();
    }
};
struct Scratch {
    Buffer<Bf> logits,gate,up,down,sg,su;
    Buffer<unsigned char> ap,as,dp,ds;
    Buffer<unsigned> ids,list;Buffer<float> coeff;
    Buffer<int> tok,exp,off{E+1},inv,total{1};
    explicit Scratch(unsigned rows):logits(rows*E),gate(rows*K*I),up(rows*K*I),
        down(rows*K*H),sg(rows*I),su(rows*I),ap(rows*H/2),as(rows*H/16),
        dp(rows*K*I/2),ds(rows*K*I/16),ids(rows*K),list(rows*K*(I/128)*2),
        coeff(rows*K),tok(rows*K),exp(rows*K),inv(rows*K) {}
    void poison(cudaStream_t s) {
        // BF16 0xffff is NaN. Remote rows must never enter unpermute/reduction.
        for(auto* b:{&gate,&up,&down})PCHECK(cudaMemsetAsync(b->ptr,0xff,b->count*sizeof(Bf),s));
    }
    void guards() {
        logits.guards();gate.guards();up.guards();down.guards();sg.guards();su.guards();
        ap.guards();as.guards();dp.guards();ds.guards();ids.guards();list.guards();coeff.guards();
        tok.guards();exp.guards();off.guards();inv.guards();total.guards();
    }
};
struct Fixture {
    const unsigned rows;
    Weights w; Scratch b; cudaStream_t stream;
    Buffer<Bf> input,rank0,rank1,shared;
    Buffer<float> residual,post,comb,highway;
    std::vector<Bf> host_input;std::vector<float> host_residual,host_post,host_comb;
    explicit Fixture(unsigned count=R):rows(count),b(count),input(count*H),rank0(count*H),
        rank1(count*H),shared(count*H),residual(count*HC*H),post(count*HC),
        comb(count*HC*HC),highway(count*HC*H) {
        require(count==10||count==15||count==20,"fixed owner-count envelope");
        PCHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    ~Fixture() { cudaStreamDestroy(stream); }
    void inputs(bool reversed) {
        host_input.assign(rows*H,bf(0));host_residual.resize(rows*HC*H);
        host_post.resize(rows*HC);host_comb.resize(rows*HC*HC);
        for(unsigned t=0;t<rows;++t) {
            const unsigned source=reversed?(rows/5-1-t/5)*5+t%5:t;
            for(unsigned k=0;k<H;++k)host_input[t*H+k]=bf(float(int((source*13+k*7)%31)-15)/32);
            for(unsigned j=0;j<HC;++j) {
                host_post[t*HC+j]=float(j+1)/8;
                for(unsigned i=0;i<HC;++i)host_comb[t*HC*HC+i*HC+j]=i==j?.625f:.125f;
                for(unsigned h=0;h<H;++h)host_residual[(t*HC+j)*H+h]=float(int((source*11+j*7+h)%37)-18)/64;
            }
        }
        input.upload(host_input);residual.upload(host_residual);post.upload(host_post);comb.upload(host_comb);
    }
    void guards() {
        w.guards();b.guards();input.guards();rank0.guards();rank1.guards();shared.guards();
        residual.guards();post.guards();comb.guards();highway.guards();
    }
};
}
