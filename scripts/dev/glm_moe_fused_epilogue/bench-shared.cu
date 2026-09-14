// SPDX-License-Identifier: AGPL-3.0-only
// Adapted from Atlas dd0ffd15 scripts/dev/glm_moe_m128/bench.cu.
// Synthetic full-working-set eligible-chain screen, not a full FFN benchmark.
// Common input prequantization is outside timing; weights are not model-derived.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef ATLAS_HOST_ONLY
#include <cuda_runtime.h>
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu"
#include "generated-shared.cuh"
#endif

static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(2); }
}
static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b;
    return x ^ (x >> 16);
}
struct Routing {
    std::array<int,289> offsets{};
    std::vector<int> ids;
    int max_rows=0;
};
static Routing routing(int rows, bool skew) {
    std::array<std::vector<int>,288> groups;
    for (int t=0;t<rows;++t) {
        std::array<bool,288> seen{};
        for (int r=0;r<8;++r) {
            unsigned e=(skew && r==0)?unsigned(t%8):mix(t*19+r*4001)%288;
            while(seen[e]) e=(e+1)%288;
            seen[e]=true; groups[e].push_back(t);
        }
    }
    Routing out;
    for(int e=0;e<288;++e) {
        out.max_rows=std::max(out.max_rows,int(groups[e].size()));
        out.ids.insert(out.ids.end(),groups[e].begin(),groups[e].end());
        out.offsets[e+1]=int(out.ids.size());
    }
    return out;
}
static void host_tests() {
    for(int rows:{4096,4100}) for(bool skew:{false,true}) {
        auto r=routing(rows,skew); std::vector<int> counts(rows);
        require(r.ids.size()==size_t(rows*8),"top8 extent");
        for(int e=0;e<288;++e) {
            std::vector<bool> seen(rows);
            for(int p=r.offsets[e];p<r.offsets[e+1];++p) {
                int t=r.ids[p]; require(t>=0&&t<rows&&!seen[t],"unique expert/token route");
                seen[t]=true; ++counts[t];
            }
        }
        for(int n:counts) require(n==8,"exactly8 experts per token");
        require(r.offsets[144]>0&&r.offsets[144]<rows*8,"both EP ownership halves populated");
    }
    std::vector<int> outputs(64*128);
    for(int th=0;th<128;++th) {
        int warp=th/32,lane=th%32,group=lane>>2,tid=lane&3;
        for(int nt=0;nt<16;++nt) for(int rr:{0,8}) for(int cc:{0,1})
            ++outputs[(warp*16+group+rr)*128+nt*8+tid*2+cc];
    }
    for(int n:outputs) require(n==1,"original MMA epilogue covers each element once");
    require(2048%128==0&&128%16==0,"FP4 scale groups do not cross output tiles");
    std::puts("PASS host: 4 routing fixtures and original epilogue ownership");
}

#ifndef ATLAS_HOST_ONLY
#define CHECK(call) do { cudaError_t err=(call); if(err!=cudaSuccess) { \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(err));std::exit(1); } } while(0)
constexpr size_t GiB=size_t(1)<<30;
static size_t live=0,peak=0;
template<class T> struct Buffer {
    T *raw=nullptr,*p=nullptr; size_t count,bytes;
    explicit Buffer(size_t n):count(n),bytes(n*sizeof(T)+256) {
        require(bytes<=3*GiB&&live<=3*GiB-bytes,"3GiB fixture allocation ceiling");
        CHECK(cudaMalloc(&raw,bytes)); p=reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(raw)+128);
        CHECK(cudaMemset(raw,0xa5,bytes)); live+=bytes;peak=std::max(peak,live);
    }
    ~Buffer(){cudaFree(raw);live-=bytes;}
    Buffer(const Buffer&)=delete;
    void upload(const T* x){CHECK(cudaMemcpy(p,x,count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read() const {
        std::vector<T> x(count);CHECK(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return x;
    }
    void poison(int value,cudaStream_t stream){CHECK(cudaMemsetAsync(p,value,count*sizeof(T),stream));}
    void guards() const {
        std::array<unsigned char,128> a,b;
        CHECK(cudaMemcpy(a.data(),raw,128,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(),p+count,128,cudaMemcpyDeviceToHost));
        for(int i=0;i<128;++i)require(a[i]==0xa5&&b[i]==0xa5,"allocation redzone overwritten");
    }
};
__global__ void fill(unsigned char* out,size_t n,bool scale,unsigned seed) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        unsigned x=unsigned(i)^seed;x^=x>>16;x*=0x7feb352d;x^=x>>15;
        out[i]=scale?static_cast<unsigned char>(0x18+x%24):static_cast<unsigned char>(x);
    }
}
struct Weight {
    Buffer<unsigned char> w,s;
    Buffer<unsigned long long> wp,sp;
    Buffer<float> scale2;
    Weight(unsigned n,unsigned k,unsigned seed):w(size_t(144)*n*k/2),s(size_t(144)*n*k/16),wp(288),sp(288),scale2(288) {
        std::array<unsigned long long,288> a{},b{};std::array<float,288> c{};
        for(int e=0;e<144;++e) {
            a[e]=reinterpret_cast<unsigned long long>(w.p+size_t(e)*n*k/2);
            b[e]=reinterpret_cast<unsigned long long>(s.p+size_t(e)*n*k/16);
            c[e]=.31f+float((e+seed)%17)*.07f;
        }
        wp.upload(a.data());sp.upload(b.data());scale2.upload(c.data());
        fill<<<4096,256>>>(w.p,w.count,false,seed);
        fill<<<4096,256>>>(s.p,s.count,true,seed+1009);CHECK(cudaGetLastError());
    }
    void guards(){w.guards();s.guards();wp.guards();sp.guards();scale2.guards();}
};
static void resources() {
    cudaFuncAttributes a{},b{};int ac=0,bc=0;
    CHECK(cudaFuncGetAttributes(&a,moe_w4a4_grouped_gemm_prequant_t_k64_vecscale));
    CHECK(cudaFuncGetAttributes(&b,atlas_dev_moe_fused_gate_up));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ac,moe_w4a4_grouped_gemm_prequant_t_k64_vecscale,128,0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bc,atlas_dev_moe_fused_gate_up,128,0));
    std::printf("resources baseline_regs=%d candidate_regs=%d baseline_smem=%zu candidate_smem=%zu baseline_ctas=%d candidate_ctas=%d\n",
        a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,ac,bc);
}
static double run_case(int rows,bool skew) {
    constexpr unsigned H=4096,N=2048;
    auto route=routing(rows,skew);const size_t expanded=size_t(rows)*8;
    const size_t packed_bytes=expanded*N/2,scale_bytes=expanded*N/16;
    Weight gate(N,H,97),up(N,H,8171),down(H,N,60013);
    Buffer<unsigned char> a(size_t(rows)*H/2),as(size_t(rows)*H/16);
    Buffer<int> offsets(289),ids(expanded);
    Buffer<__nv_bfloat16> gate_out(expanded*N),up_out(expanded*N);
    Buffer<unsigned char> staged(packed_bytes+scale_bytes),base_p(packed_bytes+scale_bytes),new_p(packed_bytes+scale_bytes);
    Buffer<__nv_bfloat16> base_out(expanded*H),new_out(expanded*H);
    offsets.upload(route.offsets.data());ids.upload(route.ids.data());
    fill<<<1024,256>>>(a.p,a.count,false,29);fill<<<1024,256>>>(as.p,as.count,true,51);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));require(free>=4*GiB,"retain4GiB free after allocations");
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    dim3 gu_grid(N/128,(route.max_rows+63)/64,288),d_grid(H/128,(route.max_rows+63)/64,288);
    auto reset=[&](bool newer) {
        if(newer){new_p.poison(0x5a,stream);new_out.poison(0x5a,stream);}
        else {gate_out.poison(0xa5,stream);up_out.poison(0xa5,stream);base_p.poison(0xa5,stream);base_out.poison(0xa5,stream);staged.poison(0xa5,stream);}
    };
    auto launch=[&](bool newer) {
        auto* packed=newer?new_p.p:base_p.p;
        if(newer) {
            atlas_dev_moe_fused_gate_up<<<gu_grid,128,0,stream>>>(a.p,as.p,
                gate.wp.p,gate.sp.p,gate.scale2.p,up.wp.p,up.sp.p,up.scale2.p,
                packed,packed+packed_bytes,offsets.p,ids.p,288,N,H);
            CHECK(cudaGetLastError());
        } else {
            for(int proj=0;proj<2;++proj) {
                auto& w=proj?up:gate;auto* out=proj?up_out.p:gate_out.p;
                moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<gu_grid,128,0,stream>>>(
                    a.p,as.p,w.wp.p,w.sp.p,w.scale2.p,out,offsets.p,ids.p,288,N,H);
                CHECK(cudaGetLastError());
            }
            silu_mul_quant_nvfp4<<<expanded,128,0,stream>>>(gate_out.p,up_out.p,staged.p,staged.p+packed_bytes,nullptr,unsigned(expanded),N);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpyAsync(packed,staged.p,packed_bytes+scale_bytes,cudaMemcpyDeviceToDevice,stream));
        }
        moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<d_grid,128,0,stream>>>(
            packed,packed+packed_bytes,down.wp.p,down.sp.p,down.scale2.p,
            newer?new_out.p:base_out.p,offsets.p,nullptr,288,H,N);
        CHECK(cudaGetLastError());
    };
    reset(false);reset(true);launch(false);launch(true);CHECK(cudaStreamSynchronize(stream));
    {
        auto b=base_p.read(),c=new_p.read();size_t checked=0;
        for(int e=0;e<288;++e)for(int r=route.offsets[e];r<route.offsets[e+1];++r) {
            for(int part=0;part<2;++part) {
                size_t stride=part?N/16:N/2,start=(part?packed_bytes:0)+size_t(r)*stride;
                for(size_t j=0;j<stride;++j) {
                    if(e<144) {require(b[start+j]==c[start+j],"local packed FP4/scale byte mismatch");++checked;}
                    else require(c[start+j]==0x5a,"candidate remote packed/scale row written");
                }
            }
        }
        std::printf("PASS packed bytes+scales rows=%d skew=%d compared=%zu\n",rows,skew,checked);
    }
    {
        auto b=base_out.read(),c=new_out.read();size_t checked=0;
        auto* bp=reinterpret_cast<unsigned short*>(b.data());auto* cp=reinterpret_cast<unsigned short*>(c.data());
        for(int e=0;e<288;++e)for(size_t i=size_t(route.offsets[e])*H;i<size_t(route.offsets[e+1])*H;++i) {
            if(e<144) {
                if(bp[i]!=cp[i]||(bp[i]&0x7f80)==0x7f80) {
                    std::fprintf(stderr,"FAIL downstream expert=%d element=%zu old=%04x new=%04x\n",e,i,bp[i],cp[i]);std::exit(2);
                }
                ++checked;
            } else require(bp[i]==0xa5a5&&cp[i]==0x5a5a,"remote downstream row written");
        }
        std::printf("PASS downstream BF16 allbits rows=%d skew=%d compared=%zu\n",rows,skew,checked);
    }
    auto guards=[&] {
        gate.guards();up.guards();down.guards();a.guards();as.guards();offsets.guards();ids.guards();
        gate_out.guards();up_out.guards();staged.guards();base_p.guards();new_p.guards();base_out.guards();new_out.guards();
    };
    guards();std::vector<float> bt,ct;
    cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
    for(int pair=0;pair<17;++pair)for(int arm=0;arm<2;++arm) {
        bool newer=((pair+arm)&1)!=0;
        reset(newer); // Same-stream reset completes before start event; excluded from timing.
        CHECK(cudaEventRecord(start,stream));launch(newer);CHECK(cudaEventRecord(stop,stream));
        CHECK(cudaEventSynchronize(stop));float ms;CHECK(cudaEventElapsedTime(&ms,start,stop));
        if(pair>=2)(newer?ct:bt).push_back(ms);
    }
    require(bt.size()==15&&ct.size()==15,"15 interleaved timing pairs after2 warmup pairs");
    auto samples=[](const char* name,const std::vector<float>& xs) {
        std::printf("samples_ms %s=",name);for(float x:xs)std::printf("%.6f,",x);std::puts("");
    };
    samples("baseline",bt);samples("fused",ct);
    std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());guards();
    double gain=bt[7]/ct[7];
    std::printf("CHAIN rows=%d skew=%d max_expert_rows=%d local_routes=%d baseline_ms=%.6f fused_ms=%.6f speedup=%.6f peak_MiB=%.3f\n",
        rows,skew,route.max_rows,route.offsets[144],bt[7],ct[7],gain,double(peak)/(1<<20));
    CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));CHECK(cudaStreamDestroy(stream));return gain;
}
#endif
int main(int argc,char** argv) {
    require(argc==2&&(!std::strcmp(argv[1],"--host-test")||!std::strcmp(argv[1],"--run")),"usage: bench --host-test|--run");
    host_tests();if(!std::strcmp(argv[1],"--host-test"))return 0;
#ifdef ATLAS_HOST_ONLY
    require(false,"host-only executable cannot run CUDA");
#else
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));
    require(free>=7*GiB,"need7GiB free before fixture (3GiB cap +4GiB reserve)");resources();
    for(auto fixture:{std::pair<int,bool>{4096,false},{4096,true},{4100,false},{4100,true}}) {
        double gain=run_case(fixture.first,fixture.second);
        if(gain<1.2){std::puts("REJECT eligible-chain gain <1.2; skip remaining fixtures");return 3;}
    }
    std::puts("PASS bounded eligible-chain screen; full FFN/model integration remains unqualified");
#endif
}
