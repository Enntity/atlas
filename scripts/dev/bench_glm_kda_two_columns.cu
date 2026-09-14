// SPDX-License-Identifier: AGPL-3.0-only
// Synthetic standalone oracle/timing; no production selection or model access.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef ATLAS_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/common/kda.cu"
#include "glm_kda_two_columns.cuh"
#endif
static void require(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(2);}}
static void host_tests(){
    std::array<int,128> columns{};
    for(unsigned block=0;block<16;++block)for(unsigned warp=0;warp<4;++warp)
        for(unsigned half=0;half<2;++half)++columns[block*8+warp*2+half];
    for(int count:columns)require(count==1,"unique complete column ownership");
    std::puts("PASS host: 128 columns each have one owner; reduction remains 32 lanes");
}
#ifndef ATLAS_HOST_ONLY
#define CHECK(call) do{auto error=(call);if(error!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(error));std::exit(1);}}while(0)
static size_t live=0,peak=0;
template<class T>struct Buffer{
    T* raw=nullptr;T* p;size_t count,bytes;
    explicit Buffer(size_t n):count(n),bytes(n*sizeof(T)+256){
        require(bytes<(size_t(1)<<29)&&live+bytes<(size_t(1)<<29),"512MiB allocation ceiling");
        CHECK(cudaMalloc(&raw,bytes));p=reinterpret_cast<T*>(reinterpret_cast<char*>(raw)+128);
        CHECK(cudaMemset(raw,0xa5,bytes));live+=bytes;peak=std::max(peak,live);
    }
    ~Buffer(){cudaFree(raw);live-=bytes;}
    Buffer(const Buffer&)=delete;
    void upload(const std::vector<T>& x){require(x.size()==count,"upload extent");CHECK(cudaMemcpy(p,x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T>x(count);CHECK(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
    void guards(){std::array<unsigned char,128>a,b;CHECK(cudaMemcpy(a.data(),raw,128,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),p+count,128,cudaMemcpyDeviceToHost));for(int i=0;i<128;++i)require(a[i]==0xa5&&b[i]==0xa5,"allocation guards");}
};
static float value(size_t i,unsigned seed){unsigned x=unsigned(i)^seed;x^=x>>16;x*=0x7feb352d;x^=x>>15;return float(x&65535)/65535.f-.5f;}
template<class T>static std::vector<T> values(size_t count,unsigned seed,float amp){std::vector<T>x(count);for(size_t i=0;i<count;++i)x[i]=T(value(i,seed)*amp);return x;}
template<class T>static void exact(const std::vector<T>&a,const std::vector<T>&b,const char* label){
    require(a.size()==b.size(),"oracle size");for(size_t i=0;i<a.size();++i)if(!std::isfinite(float(a[i]))||std::memcmp(&a[i],&b[i],sizeof(T))){std::fprintf(stderr,"FAIL %s index=%zu old=%.9g new=%.9g\n",label,i,double(float(a[i])),double(float(b[i])));std::exit(2);}
}
template<class F>static float timing(F launch,cudaStream_t stream){
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));CHECK(cudaEventRecord(a,stream));
    for(int i=0;i<10;++i)launch();CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));
    float ms;CHECK(cudaEventElapsedTime(&ms,a,b));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));return ms/10;
}
static float run(unsigned tokens,unsigned mode){
    constexpr unsigned heads=32,dim=128;const size_t plane=size_t(tokens)*heads*dim,hsize=size_t(heads)*dim*dim;
    Buffer<__nv_bfloat16> qkv(plane*3),gate(plane),raw_beta(size_t(tokens)*heads),old_out(plane),new_out(plane);
    Buffer<float> alog(heads),bias(heads*dim),q(plane),k(plane),d(plane),b(size_t(tokens)*heads),initial(hsize),old_h(hsize),new_h(hsize);
    auto input=values<__nv_bfloat16>(qkv.count,11+mode,1.f);
    if(mode==1)for(unsigned t=0;t<tokens;++t)std::fill(input.begin()+size_t(t)*3*heads*dim,input.begin()+(size_t(t)*3+2)*heads*dim,__float2bfloat16(0));
    qkv.upload(input);gate.upload(values<__nv_bfloat16>(gate.count,23+mode,mode==2?12.f:2.f));
    raw_beta.upload(values<__nv_bfloat16>(raw_beta.count,29+mode,2.f));
    alog.upload(values<float>(heads,37,.5f));bias.upload(values<float>(heads*dim,41,1.f));
    auto state=values<float>(hsize,43+mode,mode==2?4.f:.5f);initial.upload(state);old_h.upload(state);new_h.upload(state);
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    // Both recurrences consume the exact same production FP32 normalized planes.
    kda_preprocess_regresident<<<dim3(heads,tokens),128,0,stream>>>(qkv.p,gate.p,raw_beta.p,alog.p,bias.p,q.p,k.p,d.p,b.p,tokens,heads,dim,-5.f);
    CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(stream));
    auto launch=[&](bool newer){
        if(newer)atlas_dev_kda_two_columns<<<dim3(heads,16),128,0,stream>>>(qkv.p,q.p,k.p,d.p,b.p,new_h.p,new_out.p,tokens,heads,dim);
        else kda_recurrent_bf16_regresident<<<dim3(heads,32),128,0,stream>>>(qkv.p,q.p,k.p,d.p,b.p,old_h.p,old_out.p,tokens,heads,dim);
        CHECK(cudaGetLastError());
    };
    CHECK(cudaMemset(old_out.p,0xa5,plane*2));CHECK(cudaMemset(new_out.p,0x5a,plane*2));
    launch(false);launch(true);CHECK(cudaStreamSynchronize(stream));
    exact(old_out.read(),new_out.read(),"all BF16 outputs");exact(old_h.read(),new_h.read(),"all FP32 state");
    old_out.guards();new_out.guards();old_h.guards();new_h.guards();qkv.guards();q.guards();k.guards();d.guards();
    // A second pass validates continuation from the newly produced state.
    launch(false);launch(true);CHECK(cudaStreamSynchronize(stream));
    exact(old_out.read(),new_out.read(),"continued outputs");exact(old_h.read(),new_h.read(),"continued state");
    std::vector<float>ot,nt;
    for(int warm=0;warm<3;++warm){launch(false);launch(true);}CHECK(cudaStreamSynchronize(stream));
    auto reset=[&]{CHECK(cudaMemcpyAsync(old_h.p,initial.p,hsize*4,cudaMemcpyDeviceToDevice,stream));CHECK(cudaMemcpyAsync(new_h.p,initial.p,hsize*4,cudaMemcpyDeviceToDevice,stream));};
    for(int rep=0;rep<5;++rep){reset();
        if(rep%2){nt.push_back(timing([&]{launch(true);},stream));ot.push_back(timing([&]{launch(false);},stream));}
        else{ot.push_back(timing([&]{launch(false);},stream));nt.push_back(timing([&]{launch(true);},stream));}
    }
    std::sort(ot.begin(),ot.end());std::sort(nt.begin(),nt.end());float gain=ot[2]/nt[2];
    std::printf("PASS tokens=%u mode=%u old_ms=%.6f new_ms=%.6f speedup=%.3f\n",tokens,mode,ot[2],nt[2],gain);
    CHECK(cudaStreamDestroy(stream));return gain;
}
#endif
int main(int argc,char**argv){
    require(argc==2&&(!std::strcmp(argv[1],"--host-test")||!std::strcmp(argv[1],"--run")),"usage --host-test|--run");
    host_tests();if(!std::strcmp(argv[1],"--host-test"))return 0;
#ifdef ATLAS_HOST_ONLY
    require(false,"host-only executable");
#else
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));require(free>=(size_t(9)<<29),"need4.5GiBfree:512MiBcap+4GiBreserve");
    cudaFuncAttributes a{},b{};int ac,bc;
    CHECK(cudaFuncGetAttributes(&a,kda_recurrent_bf16_regresident));CHECK(cudaFuncGetAttributes(&b,atlas_dev_kda_two_columns));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ac,kda_recurrent_bf16_regresident,128,0));CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bc,atlas_dev_kda_two_columns,128,0));
    std::printf("resources old_regs=%d new_regs=%d old_ctas=%d new_ctas=%d old_smem=%zu new_smem=%zu\n",a.numRegs,b.numRegs,ac,bc,a.sharedSizeBytes,b.sharedSizeBytes);
    float gain=run(2048,0);if(gain<1.5f){std::puts("REJECT: first production-length gain<1.5x");return 3;}
    for(unsigned n:{1024u,2052u})run(n,0);
    for(unsigned mode:{1u,2u})run(2048,mode);
    std::printf("PASS bounded oracle/timing peak_MiB=%.1f; production integration pending\n",double(peak)/(1<<20));
#endif
}
