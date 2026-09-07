// SPDX-License-Identifier: AGPL-3.0-only
// Standalone only: see glm_router_bn4_plan.md. No model/device work in HOST_ONLY.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "glm_router_bn4.cuh"
namespace r = glm_router_bn4;
static void require(bool ok,const char* message) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(2);}
}
constexpr size_t budget=16ULL*1024*1024, guard=128;
static bool timing_requested(int argc,char** argv) {
    require(argc==1 || (argc==2 && std::strcmp(argv[1],"--timing")==0),
            "usage: bench_glm_router_bn4 [--timing]");
    return argc==2;
}
static bool allocation_fits(size_t count,size_t element,size_t live) {
    return element && live<=budget && count<=(budget-2*guard)/element
        && count*element+2*guard<=budget-live;
}
static void host_tests() {
    std::array<unsigned,r::rows*r::columns> outputs{};
    for(unsigned tile=0;tile<r::columns/r::bn;++tile)
        for(unsigned lane=0;lane<r::threads;++lane) if(r::owns_output(lane)) {
            require(r::row(lane)<r::rows,"output row bounds");
            ++outputs[r::row(lane)*r::columns+tile*r::bn+r::column(lane)];
        }
    for(unsigned owners:outputs)require(owners==1,"each router output has exactly one owner");
    for(unsigned tile_rows:{r::rows,r::bn}) {
        std::vector<unsigned> hits(tile_rows*r::bk);
        for(unsigned lane=0;lane<r::threads;++lane)
            for(unsigned chunk=lane;chunk<tile_rows*r::bk/4;chunk+=r::threads)
                for(unsigned j=0;j<4;++j)
                    ++hits[r::load_row(chunk)*r::bk+r::load_k(chunk)+j];
        for(unsigned hit:hits)require(hit==1,"shared tile load coverage");
    }
    unsigned expected=0;
    for(unsigned kb=0;kb<r::width;kb+=r::bk)
        for(unsigned k=0;k<r::bk;++k)require(kb+k==expected++,"increasing K order");
    require(expected==4096,"full K traversal");
    require(r::geometry(5,288,4096,32,1,1),"valid geometry");
    require(!r::geometry(4,288,4096,32,1,1),"reject wrong M");
    require(!r::geometry(5,287,4096,32,1,1),"reject N tail");
    require(!r::geometry(5,288,4095,32,1,1),"reject K tail");
    require(!r::geometry(5,288,4096,16,2,1),"reject block shape");
    require(allocation_fits(1,2,budget-258),"exact budget boundary");
    require(!allocation_fits(1,2,budget-257),"budget overrun");
    require(!allocation_fits(SIZE_MAX,2,0),"count overflow");
    require(!allocation_fits(1,0,0),"zero element");
    require(!allocation_fits(1,2,SIZE_MAX),"live overflow");
    std::puts("PASS CPU output ownership/tile loads/K order/allocation bounds");
}
#ifdef ATLAS_ROUTER_HOST_ONLY
int main(int argc,char** argv){bool timing=timing_requested(argc,argv);host_tests();
    std::printf("CPU CLI mode: %s\n",timing?"timing requested":"correctness only");}
#else
#include <cuda_runtime.h>
#include "../../kernels/gb10/common/dense_gemm_bf16.cu"
#define CHECK(call) do {auto e=(call);if(e!=cudaSuccess){ \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
static size_t live=0,peak=0;
struct Buffer {
    unsigned char* allocation=nullptr;
    uint16_t* ptr=nullptr;
    size_t count,bytes;
    explicit Buffer(size_t n):count(n),bytes(0) {
        require(allocation_fits(n,2,live),"16MiB device allocation budget");
        bytes=n*2+2*guard;
        CHECK(cudaMalloc(&allocation,bytes));live+=bytes;peak=std::max(peak,live);
        CHECK(cudaMemset(allocation,0xa5,bytes));ptr=(uint16_t*)(allocation+guard);
    }
    ~Buffer(){cudaFree(allocation);live-=bytes;}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<uint16_t>& x) {
        require(x.size()==count,"upload extent");
        CHECK(cudaMemcpy(ptr,x.data(),count*2,cudaMemcpyHostToDevice));
    }
    std::vector<uint16_t> read()const {
        std::vector<uint16_t> x(count);CHECK(cudaMemcpy(x.data(),ptr,count*2,cudaMemcpyDeviceToHost));return x;
    }
    void check_guards()const {
        unsigned char lo[guard],hi[guard];
        CHECK(cudaMemcpy(lo,allocation,guard,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hi,allocation+guard+count*2,guard,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(lo[i]==0xa5&&hi[i]==0xa5,"device canary");
    }
};
static uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static uint16_t bf16(float x) {
    uint32_t u;std::memcpy(&u,&x,4);return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
static float value(uint16_t b){uint32_t u=uint32_t(b)<<16;float x;std::memcpy(&x,&u,4);return x;}
static float sample(unsigned i,unsigned salt){return (int(hash(i+salt)&65535)-32768)/32768.0f;}
static void fixture(std::vector<uint16_t>& a,std::vector<uint16_t>& b,unsigned profile,unsigned epoch) {
    for(unsigned row=0;row<r::rows;++row)for(unsigned k=0;k<r::width;++k) {
        float x=sample(row*r::width+(profile==2?k/2:k),123+epoch*991);
        if(profile==1)x=std::ldexp(x,int((k+row)%13)-6);
        a[row*r::width+k]=bf16(x*(row+1)/3.0f);
    }
    for(unsigned col=0;col<r::columns;++col)for(unsigned k=0;k<r::width;++k) {
        unsigned c=profile==3?col/2:col;
        float x=sample(c*r::width+(profile==2?k/2:k),321+epoch*811);
        if(profile==1)x=std::ldexp(x,int((k+col)%11)-8);
        if(profile==2 && k%2)x=-x;
        if(profile==4)x=0;
        uint16_t bits=bf16(x);
        if(profile==3 && col%2 && k==col)bits^=1; // One BF16-ulp near-tie perturbation.
        b[col*r::width+k]=bits;
    }
}
static void exact(const std::vector<uint16_t>& a,const std::vector<uint16_t>& b,const char* label) {
    require(a.size()==b.size(),"comparison extent");
    for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]) {
        std::fprintf(stderr,"FAIL %s i=%zu got=%04x expected=%04x\n",label,i,a[i],b[i]);std::exit(2);
    }
}
static void valid_reference(const std::vector<uint16_t>& out) {
    require(out.size()==r::rows*r::columns,"complete reference extent");
    for(uint16_t bits:out)
        require(bits!=0x7f7f && std::isfinite(value(bits)),"reference finite/nonpoison at every output");
}
static void cpu_columns(const std::vector<uint16_t>& a,const std::vector<uint16_t>& b,
                        const std::vector<uint16_t>& out) {
    for(unsigned row=0;row<r::rows;++row)for(unsigned col:{0u,1u,15u,16u,143u,144u,286u,287u}) {
        float sum=0;double precise=0,magnitude=0;
        for(unsigned k=0;k<r::width;++k) {
            float av=value(a[row*r::width+k]),bv=value(b[col*r::width+k]);
            volatile float product=av*bv; // Materialize product: never host FMA.
            sum=sum+product;
            precise+=double(av)*bv;magnitude+=std::abs(double(av)*bv);
        }
        require(std::isfinite(sum)&&std::abs(double(sum)-precise)<=1e-7+0.0005*magnitude,"CPU double sum bound");
        if(out[row*r::columns+col]!=bf16(sum)) {
            std::fprintf(stderr,"FAIL CPU column row=%u col=%u\n",row,col);std::exit(2);
        }
    }
}
static void launch(bool candidate,const Buffer& a,const Buffer& b,Buffer& out,cudaStream_t stream) {
    auto ap=(const __nv_bfloat16*)a.ptr;auto bp=(const __nv_bfloat16*)b.ptr;auto cp=(__nv_bfloat16*)out.ptr;
    if(candidate)glm_router_m5_bn4<<<72,32,0,stream>>>(ap,bp,cp,5,288,4096);
    else dense_gemm_bf16_router_m5<<<18,dim3(16,5),0,stream>>>(ap,bp,cp,5,288,4096);
    CHECK(cudaGetLastError());
}
static void poison(Buffer& x,cudaStream_t stream){CHECK(cudaMemsetAsync(x.ptr,0x7f,x.count*2,stream));}
static float time_one(bool candidate,const Buffer& a,const Buffer& b,Buffer& out,cudaStream_t stream) {
    cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start,stream));
    for(unsigned i=0;i<50;++i)launch(candidate,a,b,out,stream);
    CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
    float ms;CHECK(cudaEventElapsedTime(&ms,start,end));CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));
    return ms*1000/50;
}
int main(int argc,char** argv) {
    const bool timing=timing_requested(argc,argv);
    host_tests();
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Buffer a(r::rows*r::width),b(r::columns*r::width),old(r::rows*r::columns),candidate(old.count);
    std::vector<uint16_t> ah(a.count),bh(b.count);
    const size_t expected=(a.count+b.count+old.count+candidate.count)*2+8*guard;
    require(live==expected&&live<=budget,"exact live device accounting");
    std::printf("Device bytes=%zu limit=%zu (matrix allocations plus all guards)\n",live,budget);
    cudaGraph_t graph;cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    launch(false,a,b,old,stream);launch(true,a,b,candidate,stream);
    CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(unsigned profile=0;profile<5;++profile)for(unsigned epoch=0;epoch<2;++epoch) {
        fixture(ah,bh,profile,epoch);a.upload(ah);b.upload(bh);
        poison(old,stream);poison(candidate,stream);
        launch(false,a,b,old,stream);launch(true,a,b,candidate,stream);CHECK(cudaStreamSynchronize(stream));
        const auto reference=old.read();valid_reference(reference);
        exact(candidate.read(),reference,"eager production M5");cpu_columns(ah,bh,reference);
        poison(old,stream);poison(candidate,stream);CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
        exact(old.read(),reference,"refreshed graph original");exact(candidate.read(),reference,"refreshed graph BN4");
        exact(a.read(),ah,"immutable A");exact(b.read(),bh,"immutable B");
        for(const Buffer* buffer:{&a,&b,&old,&candidate})buffer->check_guards();
        std::printf("PASS profile=%u epoch=%u all1440 logits BITEXACT +CPUcolumns +graphrefresh +guards\n",profile,epoch);
    }
    // Invalid geometry must return uniformly without touching the output.
    poison(candidate,stream);
    glm_router_m5_bn4<<<72,32,0,stream>>>((__nv_bfloat16*)a.ptr,(__nv_bfloat16*)b.ptr,(__nv_bfloat16*)candidate.ptr,4,288,4096);
    CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(stream));
    exact(candidate.read(),std::vector<uint16_t>(candidate.count,0x7f7f),"invalid geometry untouched");
    if(timing) {
    // Time a nonzero random profile, never the preceding all-zero fixture.
    fixture(ah,bh,0,19);a.upload(ah);b.upload(bh);
    launch(false,a,b,old,stream);launch(true,a,b,candidate,stream);CHECK(cudaStreamSynchronize(stream));
    const auto reference=old.read();valid_reference(reference);
    cpu_columns(ah,bh,reference);exact(candidate.read(),reference,"timing fixture");
    std::vector<float> old_us,new_us;
    for(unsigned round=0;round<7;++round) {
        float times[2];
        for(unsigned step=0;step<2;++step) {
            const bool use_candidate=(round+step)%2;Buffer& out=use_candidate?candidate:old;
            poison(out,stream);times[use_candidate]=time_one(use_candidate,a,b,out,stream);
        }
        exact(old.read(),reference,"posttiming original");exact(candidate.read(),reference,"posttiming BN4");
        exact(a.read(),ah,"posttiming immutable A");exact(b.read(),bh,"posttiming immutable B");
        for(const Buffer* buffer:{&a,&b,&old,&candidate})buffer->check_guards();
        old_us.push_back(times[0]);new_us.push_back(times[1]);
        std::printf("TIME round=%u original_us=%.3f bn4_us=%.3f\n",round,times[0],times[1]);
    }
    std::sort(old_us.begin(),old_us.end());std::sort(new_us.begin(),new_us.end());
    std::printf("PASS median original_us=%.3f bn4_us=%.3f speedup=%.4f peak_device_bytes=%zu\n",old_us[3],new_us[3],old_us[3]/new_us[3],peak);
    } else {
        std::printf("PASS correctness only; timing disabled; peak_device_bytes=%zu\n",peak);
    }
    CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
}
#endif
