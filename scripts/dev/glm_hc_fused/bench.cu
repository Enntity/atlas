// SPDX-License-Identifier: AGPL-3.0-only
// Standalone synthetic chain. Oracle adapted from bench_glm_hc_tuned.cu;
// cuBLASLt descriptors mirror spark-runtime/src/cublaslt.rs TF32 dispatch.
// nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a --fmad=false bench.cu -lcublasLt -o bench
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifndef ATLAS_HOST_ONLY
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu"
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/glm_hc_prefill_vec.cu"
#include "candidate.cuh"
#include "finalizer.cuh"
#endif
constexpr unsigned H=4096,D=4*H,MIX=24,ITERS=20;
constexpr float EPS=1e-6f;
static void require(bool ok,const char* msg){if(!ok){std::fprintf(stderr,"FAIL: %s\n",msg);std::exit(2);}}
static uint16_t bf16(float f){uint32_t u;std::memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return uint16_t(u>>16);}
static float f32(uint16_t b){uint32_t u=uint32_t(b)<<16;float x;std::memcpy(&x,&u,4);return x;}
// Exact host operand rounding from hc_tf32_heuristics_r9.cu, both tie modes.
static float tf32(float f,bool even){uint32_t x;std::memcpy(&x,&f,4);x=(x+(even?0xfffu+((x>>13)&1u):0x1000u))&0xffffe000u;std::memcpy(&f,&x,4);return f;}
struct Fixture {
    unsigned rows;std::vector<float> residual,post,comb,fn,scale,base;std::vector<uint16_t> block;
    Fixture(unsigned m):rows(m),residual(size_t(m)*D),post(size_t(m)*4),comb(size_t(m)*16),
        fn(size_t(MIX)*D),scale{.17f,-.23f,.31f},base(MIX),block(size_t(m)*H){
        std::mt19937 rng(9121+m);std::uniform_real_distribution<float> r(-1,1);
        for(auto& x:residual)x=r(rng)*1.7f;
        for(auto& x:block)x=bf16(r(rng)*1.2f);
        for(auto& x:post)x=r(rng)+1;
        for(auto& x:fn)x=r(rng)*.015f;
        for(auto& x:base)x=r(rng)*.7f;
        // Exercise exact TF32 operand ties and saturated gates without making
        // the entire mixer insensitive to its raw inputs.
        for(size_t i=0;i<fn.size();i+=97){uint32_t u;std::memcpy(&u,&fn[i],4);u=(u&0xffffe000u)|0x1000u;std::memcpy(&fn[i],&u,4);}
        base[1]=24.f;base[6]=-24.f;base[8]=18.f;
        for(unsigned t=0;t<m;++t)for(unsigned j=0;j<4;++j){
            float sum=0;for(unsigned i=0;i<4;++i){auto& v=comb[size_t(t)*16+i*4+j];v=r(rng)+1.01f;sum+=v;}
            for(unsigned i=0;i<4;++i)comb[size_t(t)*16+i*4+j]/=sum;
        }
        if(m>1){std::fill(residual.begin(),residual.begin()+D,0);std::fill(block.begin(),block.begin()+H,0);}
    }
};
struct Expected {std::vector<double> highway,raw,inv,y,post,comb;};
static void finalize_reference(const float* highway,const float* raw,double inv,
                               const Fixture& f,double* y,double* post,double* comb){
    double pre[4];
    for(int i=0;i<4;++i){
        pre[i]=1/(1+std::exp(-(double(raw[i])*inv*f.scale[0]+f.base[i])))+EPS;
        post[i]=2/(1+std::exp(-(double(raw[4+i])*inv*f.scale[1]+f.base[4+i])));
    }
    for(int i=0;i<16;++i)comb[i]=double(raw[8+i])*inv*f.scale[2]+f.base[8+i];
    for(int i=0;i<4;++i){
        double mx=-INFINITY,sum=0;
        for(int j=0;j<4;++j)mx=std::max(mx,comb[i*4+j]);
        for(int j=0;j<4;++j){comb[i*4+j]=std::exp(comb[i*4+j]-mx);sum+=comb[i*4+j];}
        for(int j=0;j<4;++j)comb[i*4+j]=comb[i*4+j]/sum+EPS;
    }
    auto columns=[&](double eps){for(int j=0;j<4;++j){double sum=eps;for(int i=0;i<4;++i)sum+=comb[i*4+j];for(int i=0;i<4;++i)comb[i*4+j]/=sum;}};
    columns(EPS);
    for(unsigned iter=1;iter<ITERS;++iter){
        for(int i=0;i<4;++i){double sum=EPS;for(int j=0;j<4;++j)sum+=comb[i*4+j];for(int j=0;j<4;++j)comb[i*4+j]/=sum;}
        columns(EPS);
    }
    columns(0);
    for(unsigned d=0;d<H;++d){y[d]=0;for(int i=0;i<4;++i)y[d]+=pre[i]*highway[i*H+d];}
}
static Expected oracle(const Fixture& f){
    Expected o;o.highway.resize(f.residual.size());o.raw.resize(size_t(f.rows)*MIX);o.inv.resize(f.rows);
    for(unsigned t=0;t<f.rows;++t){
        for(unsigned j=0;j<4;++j)for(unsigned d=0;d<H;++d){
            double v=double(f.post[t*4+j])*f32(f.block[size_t(t)*H+d]);
            for(unsigned i=0;i<4;++i)v+=double(f.comb[t*16+i*4+j])*f.residual[size_t(t)*D+i*H+d];
            o.highway[size_t(t)*D+j*H+d]=v;
        }
    }
    return o;
}
static void host_tests(){
    require(f32(bf16(1.5f))==1.5f,"BF16 host encoding");Fixture f(3);auto o=oracle(f);
    for(unsigned d=0;d<D;++d)require(o.highway[d]==0,"zero highway row");
    std::vector<float> x(D,0),raw(MIX,0);std::array<double,H> y{};double p[4],c[16];
    finalize_reference(x.data(),raw.data(),1000,f,y.data(),p,c);
    for(double v:y)require(v==0,"zero collapsed row");
    for(int j=0;j<4;++j){double sum=0;for(int i=0;i<4;++i)sum+=c[i*4+j];require(std::abs(sum-1)<1e-12,"Sinkhorn column normalization");}
    std::puts("PASS host: BF16, zero highway and independent Sinkhorn oracle");
}
#ifndef ATLAS_HOST_ONLY
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
#define LT(x) do{cublasStatus_t e=(x);if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLASLt %d status%d\n",__LINE__,int(e));std::exit(1);}}while(0)
constexpr size_t GiB=size_t(1)<<30;static size_t live=0,peak=0;
template<class T>struct Buffer{
    T *raw,*p;size_t n,bytes;
    Buffer(size_t count):n(count),bytes(count*sizeof(T)+512){
        require(bytes<=2*GiB&&live<=2*GiB-bytes,"2GiB fixture allocation cap");CHECK(cudaMalloc(&raw,bytes));
        p=reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(raw)+256);CHECK(cudaMemset(raw,0xa5,bytes));live+=bytes;peak=std::max(peak,live);
    }
    ~Buffer(){cudaFree(raw);live-=bytes;}Buffer(const Buffer&)=delete;
    void upload(const std::vector<T>& v){require(v.size()==n,"upload extent");CHECK(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> v(n);CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
    void poison(int x,cudaStream_t s){CHECK(cudaMemsetAsync(p,x,n*sizeof(T),s));}
    void guards(){std::array<unsigned char,256>a,b;CHECK(cudaMemcpy(a.data(),raw,256,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),p+n,256,cudaMemcpyDeviceToHost));for(int i=0;i<256;++i)require(a[i]==0xa5&&b[i]==0xa5,"redzone overwritten");}
};
struct RawMix{
    cublasLtHandle_t h;Buffer<unsigned char> workspace{64*1024*1024};
    cublasLtMatmulDesc_t desc{};cublasLtMatrixLayout_t a{},b{},d{};cublasLtMatmulPreference_t pref{};
    cublasLtMatmulHeuristicResult_t result{};unsigned rows=0;
    RawMix(){LT(cublasLtCreate(&h));}
    ~RawMix(){cublasLtMatmulPreferenceDestroy(pref);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(d);cublasLtMatmulDescDestroy(desc);cublasLtDestroy(h);}
    // Same first heuristic as runtime, cached outside timing to avoid attributing
    // descriptor/algorithm setup savings to fusion.
    void prepare(unsigned m){
        require(rows==0,"one descriptor/heuristic setup per fixture");rows=m;
        LT(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F_FAST_TF32,CUDA_R_32F));
        cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
        LT(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
        LT(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
        LT(cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,D,MIX,D));LT(cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,D,m,D));LT(cublasLtMatrixLayoutCreate(&d,CUDA_R_32F,MIX,m,MIX));
        LT(cublasLtMatmulPreferenceCreate(&pref));size_t cap=workspace.n;
        LT(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cap,sizeof(cap)));
        int returned=0;
        LT(cublasLtMatmulAlgoGetHeuristic(h,desc,a,b,d,d,pref,1,&result,&returned));
        require(returned==1&&result.state==CUBLAS_STATUS_SUCCESS&&result.workspaceSize<=cap,"valid bounded TF32 heuristic");
    }
    void run(const float* x,const float* w,float* out,unsigned m,cudaStream_t s){
        require(m==rows,"cached heuristic shape unchanged");
        float alpha=1,beta=0;
        LT(cublasLtMatmul(h,desc,&alpha,w,a,x,b,&beta,out,d,out,d,&result.algo,workspace.p,workspace.n,s));
    }
};
struct Error{double max_abs=0,sum2=0,ref2=0,ref_max=0;size_t count=0;
    void add(double x,double y,double atol,double rtol,const char* name){
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite output");double e=std::abs(x-y);max_abs=std::max(max_abs,e);ref_max=std::max(ref_max,std::abs(y));sum2+=e*e;ref2+=y*y;++count;
        if(e>atol+rtol*std::abs(y)){std::fprintf(stderr,"FAIL %s x=%.12g ref=%.12g error=%.9g\n",name,x,y,e);std::exit(2);}
    }
    double relative(){return std::sqrt(sum2/std::max(ref2,1e-24));}
    bool tf32_valid(){return relative()<1e-3&&max_abs<2e-3*std::max(1.,ref_max);}
};
static double run_case(unsigned m,bool inplace,bool timed){
    Fixture f(m);size_t highway=size_t(m)*D;
    RawMix lt;Buffer<float> original(highway),r0(highway),r1(highway),o0(highway),o1(highway);
    Buffer<float> post(f.post.size()),comb(f.comb.size()),fn(f.fn.size()),scale(3),base(24),raw0(size_t(m)*24),raw1(size_t(m)*24),inv(m);
    Buffer<uint16_t> block(f.block.size()),y0(size_t(m)*H),y1(size_t(m)*H);
    Buffer<float> p0(size_t(m)*4),p1(size_t(m)*4),c0(size_t(m)*16),c1(size_t(m)*16);
    original.upload(f.residual);post.upload(f.post);comb.upload(f.comb);fn.upload(f.fn);scale.upload(f.scale);base.upload(f.base);block.upload(f.block);
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));require(free>=4*GiB,"retain4GiB free");
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    lt.prepare(m);
    auto* h0=inplace?r0.p:o0.p;auto* h1=inplace?r1.p:o1.p;
    auto reset=[&](bool candidate){
        CHECK(cudaMemcpyAsync(candidate?r1.p:r0.p,original.p,highway*4,cudaMemcpyDeviceToDevice,stream));
        if(candidate){o1.poison(0x5a,stream);raw1.poison(0xff,stream);inv.poison(0xff,stream);y1.poison(0xff,stream);p1.poison(0xff,stream);c1.poison(0xff,stream);}
        else{o0.poison(0xa5,stream);raw0.poison(0xff,stream);y0.poison(0xff,stream);p0.poison(0xff,stream);c0.poison(0xff,stream);}
    };
    auto launch=[&](bool candidate){
        if(candidate){
            glm_hc_post_mix_rms_tf32<<<(m+31)/32,256,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(block.p),r1.p,post.p,comb.p,fn.p,h1,raw1.p,inv.p,m,EPS);CHECK(cudaGetLastError());
            atlas_dev_hc_pre_from_raw_and_rsqrt<<<m,256,0,stream>>>(h1,raw1.p,inv.p,scale.p,base.p,reinterpret_cast<__nv_bfloat16*>(y1.p),p1.p,c1.p,H,4,ITERS,EPS,EPS);
        }else{
            hc_post<<<m,256,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(block.p),r0.p,post.p,comb.p,h0,H,4);CHECK(cudaGetLastError());
            lt.run(h0,fn.p,raw0.p,m,stream);
            glm_hc_pre_from_raw_mix_vec<<<m,256,0,stream>>>(h0,raw0.p,scale.p,base.p,reinterpret_cast<__nv_bfloat16*>(y0.p),p0.p,c0.p,H,4,ITERS,EPS,EPS);
        }CHECK(cudaGetLastError());
    };
    reset(false);reset(true);launch(false);launch(true);CHECK(cudaStreamSynchronize(stream));
    auto x0=inplace?r0.read():o0.read(),x1=inplace?r1.read():o1.read();
    require(std::memcmp(x0.data(),x1.data(),highway*4)==0,"FP32 highway must match all bits");
    for(float x:x0)require(std::isfinite(x),"finite highway");
    auto a=raw0.read(),b=raw1.read(),iv=inv.read(),pa=p0.read(),pb=p1.read(),ca=c0.read(),cb=c1.read();auto ya=y0.read(),yb=y1.read();
    Error rawerr,inverr,yerr,perr,cerr;
    for(size_t i=0;i<a.size();++i)rawerr.add(b[i],a[i],INFINITY,0,"rawmix A/B");
    require(rawerr.tf32_valid(),"rawmix relativeL2/maxabs gate");
    for(unsigned t=0;t<m;++t){double ss=0;for(unsigned d=0;d<D;++d){double x=x0[size_t(t)*D+d];ss+=x*x;}double ref=1/std::sqrt(ss/D+EPS);inverr.add(iv[t],ref,0,2e-6,"inverse RMS oracle");}
    // Independent FP64 dots over explicitly rounded TF32 operands, including
    // representative full-size rows and the final partial CTA row.
    Error tf[2][2];std::vector<unsigned> sample{0,m>1?1u:0u,m/3,m/2,m-1};
    if(m<=33){sample.clear();for(unsigned t=0;t<m;++t)sample.push_back(t);}
    std::sort(sample.begin(),sample.end());sample.erase(std::unique(sample.begin(),sample.end()),sample.end());
    for(unsigned t:sample)for(unsigned j=0;j<MIX;++j){
        double dots[2]{};
        for(unsigned d=0;d<D;++d)for(int even=0;even<2;++even)
            dots[even]+=double(tf32(x0[size_t(t)*D+d],even))*tf32(f.fn[size_t(j)*D+d],even);
        for(int arm=0;arm<2;++arm)for(int even=0;even<2;++even)
            tf[arm][even].add((arm?b:a)[t*MIX+j],dots[even],INFINITY,0,"TF32 CPU dot");
    }
    for(int arm=0;arm<2;++arm){
        require(tf[arm][0].tf32_valid()||tf[arm][1].tf32_valid(),"independent TF32 RNA/RNE oracle");
        std::printf("PASS TF32 FP64 dots arm=%d dots=%zu RNA_rel=%g RNA_max=%g RNE_rel=%g RNE_max=%g\n",arm,tf[arm][0].count,tf[arm][0].relative(),tf[arm][0].max_abs,tf[arm][1].relative(),tf[arm][1].max_abs);
    }
    // NEW candidate-only chain tolerances; historical exact-raw gates below stay strict.
    for(size_t i=0;i<ya.size();++i)yerr.add(f32(yb[i]),f32(ya[i]),.004,.008,"collapsed A/B");
    for(size_t i=0;i<pa.size();++i)perr.add(pb[i],pa[i],.001,0,"post A/B");
    for(size_t i=0;i<ca.size();++i)cerr.add(cb[i],ca[i],.001,0,"comb A/B");
    if(m<=33){
        auto ref=oracle(f);Error he,ye[2],pe[2],ce[2];
        for(size_t i=0;i<highway;++i)he.add(x0[i],ref.highway[i],2e-6,1e-6,"FP64 highway");
        for(unsigned t=0;t<m;++t){
            double ss=0;for(unsigned d=0;d<D;++d){double v=x0[size_t(t)*D+d];ss+=v*v;}
            for(int arm=0;arm<2;++arm){
                std::array<double,H> yy;double pp[4],cc[16];
                finalize_reference(x0.data()+size_t(t)*D,(arm?b:a).data()+t*MIX,arm?iv[t]:1/std::sqrt(ss/D+EPS),f,yy.data(),pp,cc);
                for(unsigned d=0;d<H;++d)ye[arm].add(f32((arm?yb:ya)[size_t(t)*H+d]),yy[d],.002,.004,"own-raw FP64 collapsed");
                for(int j=0;j<4;++j)pe[arm].add((arm?pb:pa)[t*4+j],pp[j],2e-5,0,"own-raw FP64 post");
                for(int j=0;j<16;++j)ce[arm].add((arm?cb:ca)[t*16+j],cc[j],2e-5,0,"own-raw FP64 comb");
            }
        }
        std::printf("PASS independent FP64 highway+finalizers rows=%u inplace=%d highway_max=%g\n",m,inplace,he.max_abs);
    }
    auto guards=[&]{original.guards();r0.guards();r1.guards();o0.guards();o1.guards();post.guards();comb.guards();fn.guards();scale.guards();base.guards();raw0.guards();raw1.guards();inv.guards();block.guards();y0.guards();y1.guards();p0.guards();p1.guards();c0.guards();c1.guards();lt.workspace.guards();};
    guards();std::printf("PASS numeric rows=%u inplace=%d raw_max=%g raw_rrmse=%g inv_max=%g y_max=%g post_max=%g comb_max=%g peak_MiB=%.3f\n",m,inplace,rawerr.max_abs,rawerr.relative(),inverr.max_abs,yerr.max_abs,perr.max_abs,cerr.max_abs,double(peak)/(1<<20));
    double gain=0;
    if(timed){
        cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));std::vector<float> times[2];
        for(int pair=0;pair<17;++pair)for(int j=0;j<2;++j){int arm=(pair+j)&1;reset(arm);CHECK(cudaEventRecord(start,stream));launch(arm);CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));if(pair>=2)times[arm].push_back(ms);}
        for(int arm=0;arm<2;++arm){std::printf("samples_ms arm%d=",arm);for(float x:times[arm])std::printf("%.6f,",x);std::puts("");std::sort(times[arm].begin(),times[arm].end());}
        gain=times[0][7]/times[1][7];std::printf("CHAIN rows=%u inplace=%d baseline_ms=%.6f fused_ms=%.6f speedup=%.6f\n",m,inplace,times[0][7],times[1][7],gain);CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));guards();
    }
    CHECK(cudaStreamDestroy(stream));return gain;
}
#endif
int main(int argc,char** argv){
    require(argc==2&&(!std::strcmp(argv[1],"--host-test")||!std::strcmp(argv[1],"--run")),"usage: bench --host-test|--run");host_tests();if(!std::strcmp(argv[1],"--host-test"))return 0;
#ifdef ATLAS_HOST_ONLY
    require(false,"host-only executable cannot launch CUDA");
#else
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));require(free>=6*GiB,"need6GiB free (2GiB fixture cap +4GiB reserve)");
    for(unsigned m:{1,3,33})for(bool inplace:{false,true})run_case(m,inplace,false);
    for(unsigned m:{4096,3519})for(bool inplace:{true,false})if(run_case(m,inplace,true)<1.8){std::puts("REJECT full-chain speedup<1.8; skip remaining fixtures");return 3;}
    std::puts("PASS bounded HC chain screen; cross-layer integration remains unqualified");
#endif
}
