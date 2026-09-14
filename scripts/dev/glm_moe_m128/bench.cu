// SPDX-License-Identifier: AGPL-3.0-only
// Standalone only. Deterministic synthetic routed inputs; no checkpoint access.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#ifndef ATLAS_HOST_ONLY
#include <cuda_runtime.h>
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "candidate.cuh"
#endif

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
struct Routing {
    std::array<int, 289> offsets{};
    std::vector<int> ids;
    int max_rows = 0;
};
static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b;
    return x ^ (x >> 16);
}
static Routing routing(int rows, bool skew, int first = 0) {
    std::array<std::vector<int>, 288> groups;
    for (int t = 0; t < rows; ++t) {
        int token = t + first;
        std::array<bool, 288> seen{};
        for (int r = 0; r < 8; ++r) {
            unsigned e = (skew && r == 0) ? unsigned(token % 8) : mix(token * 19 + r * 4001) % 288;
            while (seen[e]) e = (e + 1) % 288;
            seen[e] = true; groups[e].push_back(t);
        }
    }
    Routing out;
    for (int e = 0; e < 288; ++e) {
        out.max_rows = std::max(out.max_rows, int(groups[e].size()));
        out.ids.insert(out.ids.end(), groups[e].begin(), groups[e].end());
        out.offsets[e + 1] = int(out.ids.size());
    }
    return out;
}
static void host_tests() {
    for (int rows : {2048, 4096}) for (bool skew : {false, true}) {
        auto r = routing(rows, skew);
        require(r.ids.size() == size_t(rows * 8), "top8 extent");
        std::vector<int> counts(rows);
        for (int e = 0; e < 288; ++e) {
            std::vector<bool> seen(rows);
            for (int p = r.offsets[e]; p < r.offsets[e + 1]; ++p) {
                int t = r.ids[p];
                require(t >= 0 && t < rows && !seen[t], "expert unique valid tokens");
                seen[t] = true; ++counts[t];
            }
        }
        require(std::all_of(counts.begin(), counts.end(), [](int n) { return n == 8; }), "eight unique experts per token");
    }
    for (bool skew : {false,true}) {
        auto a=routing(2048,skew),b=routing(2048,skew,2048),c=routing(4096,skew);
        for(int e=0;e<288;++e)
            require(c.offsets[e+1]-c.offsets[e] == a.offsets[e+1]-a.offsets[e]+b.offsets[e+1]-b.offsets[e],"split route histogram");
    }
    // Exhaustively check CTA ownership for the widened loads and output stores.
    std::vector<int> ap(128*32),as(128*4),bp(32*128),bs(4*128),out(128*128);
    for(int t=0;t<256;++t) {
        for(int j=0;j<16;++j) ++ap[(t>>1)*32+(t&1)*16+j];
        if(t<128) {
            for(int j=0;j<4;++j) ++as[t*4+j];
            for(int r=0;r<2;++r)for(int j=0;j<16;++j)
                ++bp[(r*16+(t>>3))*128+(t&7)*16+j];
        }
        if(t<32)for(int j=0;j<16;++j) ++bs[(t>>3)*128+(t&7)*16+j];
        int warp=t/32,lane=t%32,tid=lane&3,group=lane>>2;
        for(int nt=0;nt<16;++nt)for(int rr:{0,8})for(int cc:{0,1})
            ++out[(warp*16+group+rr)*128+nt*8+tid*2+cc];
    }
    for(auto* v:{&ap,&as,&bp,&bs,&out})
        require(std::all_of(v->begin(),v->end(),[](int n){return n==1;}),"unique complete CTA writes");
    std::puts("PASS host: routing/split histograms and exhaustive A/B/scale/output CTA ownership");
}

#ifndef ATLAS_HOST_ONLY
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
static size_t live = 0, peak = 0;
constexpr size_t GiB = size_t(1) << 30;
template<class T> struct Buffer {
    T* raw = nullptr; T* p; size_t count, bytes;
    explicit Buffer(size_t n):count(n),bytes(n*sizeof(T)+256) {
        require(bytes <= 3*GiB && live <= 3*GiB-bytes, "3GiB benchmark allocation ceiling");
        CHECK(cudaMalloc(&raw, bytes)); p = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(raw)+128);
        CHECK(cudaMemset(raw, 0xa5, bytes)); live += bytes; peak = std::max(peak, live);
    }
    ~Buffer() { cudaFree(raw); live -= bytes; }
    Buffer(const Buffer&) = delete;
    void upload(const T* data) { CHECK(cudaMemcpy(p, data, count*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> read() const {
        std::vector<T> out(count); CHECK(cudaMemcpy(out.data(), p, count*sizeof(T), cudaMemcpyDeviceToHost)); return out;
    }
    void guards() const {
        std::array<unsigned char,128> a,b;
        CHECK(cudaMemcpy(a.data(), raw, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), p+count, 128, cudaMemcpyDeviceToHost));
        for (int i=0;i<128;++i) require(a[i]==0xa5 && b[i]==0xa5, "device guard overwritten");
    }
};
__global__ void fill(unsigned char* out, size_t bytes, bool scale, unsigned seed) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<bytes; i+=size_t(gridDim.x)*blockDim.x) {
        unsigned x=unsigned(i)^seed; x^=x>>16; x*=0x7feb352d; x^=x>>15;
        out[i]=scale ? static_cast<unsigned char>(0x20+(x%24)) : static_cast<unsigned char>(x);
    }
}
template<class F> static float event_time(F launch, cudaStream_t stream) {
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a,stream));
    for(int i=0;i<10;++i) launch();
    CHECK(cudaEventRecord(b,stream)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); return ms/10;
}
static void resources() {
    cudaFuncAttributes old{}, candidate{}; int old_ctas=0, new_ctas=0;
    CHECK(cudaFuncGetAttributes(&old, moe_w4a4_grouped_gemm_prequant_t_k64_vecscale));
    CHECK(cudaFuncGetAttributes(&candidate, atlas_dev_moe_m128));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&old_ctas,moe_w4a4_grouped_gemm_prequant_t_k64_vecscale,128,0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&new_ctas,atlas_dev_moe_m128,256,0));
    std::printf("resources old_smem=%zu new_smem=%zu old_regs=%d new_regs=%d old_ctas=%d new_ctas=%d\n",
        old.sharedSizeBytes,candidate.sharedSizeBytes,old.numRegs,candidate.numRegs,old_ctas,new_ctas);
}
struct Timing { float old_ms, new_ms; };
static Timing run_case(int rows, bool skew, bool down, int first=0) {
    const unsigned n=down?4096:2048, k=down?2048:4096;
    const size_t wb=size_t(n)*k/2, sb=size_t(n)*k/16;
    auto route=routing(rows,skew,first);
    const int expanded=rows*8, a_rows=down?expanded:rows;
    // 144 different expert matrices exceed L2 by a wide margin. Half the
    // global pointer table is NULL, matching one real EP2 rank.
    Buffer<unsigned char> w(144*wb), s(144*sb), a(size_t(a_rows)*k/2), as(size_t(a_rows)*k/16);
    Buffer<unsigned long long> wp(288),sp(288);
    Buffer<float> scale2(288);
    Buffer<int> offsets(289), ids(expanded);
    Buffer<__nv_bfloat16> reference(size_t(expanded)*n), candidate(size_t(expanded)*n);
    std::array<unsigned long long,288> wptr{},sptr{};
    std::array<float,288> scales{};
    for(int e=0;e<144;++e) {
        wptr[e]=reinterpret_cast<unsigned long long>(w.p+e*wb);
        sptr[e]=reinterpret_cast<unsigned long long>(s.p+e*sb);
        scales[e]=0.31f+float(e%17)*0.07f;
    }
    wp.upload(wptr.data());sp.upload(sptr.data());scale2.upload(scales.data());
    offsets.upload(route.offsets.data());ids.upload(route.ids.data());
    fill<<<4096,256>>>(w.p,w.count,false,97);fill<<<4096,256>>>(s.p,s.count,true,13);
    fill<<<1024,256>>>(a.p,a.count,false,29);fill<<<1024,256>>>(as.p,as.count,true,51);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    dim3 grid((n+127)/128,(route.max_rows+63)/64,288);
    dim3 wide((n+127)/128,(route.max_rows+127)/128,288);
    auto launch=[&](bool newer) {
        if(newer) atlas_dev_moe_m128<<<wide,256,0,stream>>>(a.p,as.p,wp.p,sp.p,scale2.p,candidate.p,offsets.p,down?nullptr:ids.p,288,n,k);
        else moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<grid,128,0,stream>>>(a.p,as.p,wp.p,sp.p,scale2.p,reference.p,offsets.p,down?nullptr:ids.p,288,n,k);
        CHECK(cudaGetLastError());
    };
    // Distinct poison prevents a matching missing local store from passing.
    CHECK(cudaMemset(reference.p,0xa5,reference.count*2));
    CHECK(cudaMemset(candidate.p,0x5a,candidate.count*2));
    launch(false);launch(true);CHECK(cudaStreamSynchronize(stream));
    auto x=reference.read(),y=candidate.read();
    auto* xb=reinterpret_cast<unsigned short*>(x.data()); auto* yb=reinterpret_cast<unsigned short*>(y.data());
    for(int e=0;e<288;++e) for(size_t i=size_t(route.offsets[e])*n;i<size_t(route.offsets[e+1])*n;++i) {
        if(e<144) {
            if(xb[i]!=yb[i] || (xb[i]&0x7f80)==0x7f80) {
                std::fprintf(stderr,"FAIL oracle rows=%d skew=%d down=%d expert=%d element=%zu old=%04x new=%04x\n",rows,skew,down,e,i,xb[i],yb[i]);std::exit(2);
            }
        } else require(xb[i]==0xa5a5 && yb[i]==0x5a5a,"remote row written");
    }
    reference.guards();candidate.guards();w.guards();s.guards();a.guards();as.guards();
    for(int i=0;i<5;++i){launch(false);launch(true);} CHECK(cudaStreamSynchronize(stream));
    std::vector<float> old_times,new_times;
    for(int r=0;r<5;++r) {
        if(r%2) {new_times.push_back(event_time([&]{launch(true);},stream));old_times.push_back(event_time([&]{launch(false);},stream));}
        else {old_times.push_back(event_time([&]{launch(false);},stream));new_times.push_back(event_time([&]{launch(true);},stream));}
    }
    std::sort(old_times.begin(),old_times.end());std::sort(new_times.begin(),new_times.end());
    float gain=old_times[2]/new_times[2];
    std::printf("PASS oracle rows=%d skew=%d down=%d first=%d max_expert_rows=%d old_ms=%.6f new_ms=%.6f speedup=%.3f\n",rows,skew,down,first,route.max_rows,old_times[2],new_times[2],gain);
    CHECK(cudaStreamDestroy(stream)); return {old_times[2],new_times[2]};
}
#endif

int main(int argc,char** argv) {
    require(argc==2 && (!std::strcmp(argv[1],"--host-test") || !std::strcmp(argv[1],"--run")),"usage: bench --host-test|--run");
    host_tests();if(!std::strcmp(argv[1],"--host-test"))return 0;
#ifdef ATLAS_HOST_ONLY
    require(false,"host-only binary cannot run CUDA");
#else
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));
    require(free>=7*GiB,"need >=7GiB free (3GiB cap +4GiB reserve)");
    resources();
    float minimum=1000;
    for(bool skew:{false,true}) {
        float weighted_old=0,weighted_new=0;
        for(bool down:{false,true}) {
            auto lo=run_case(2048,skew,down,0);
            auto hi=run_case(2048,skew,down,2048);
            auto whole=run_case(4096,skew,down,0);
            float effective=(lo.old_ms+hi.old_ms)/whole.new_ms;
            float chunk_only=(lo.old_ms+hi.old_ms)/whole.old_ms;
            float direct=whole.old_ms/whole.new_ms;
            std::printf("THROUGHPUT skew=%d down=%d effective_4096_M128_vs_two_2048_M64=%.3f chunk_only_M64=%.3f direct_4096_M128=%.3f\n",skew,down,effective,chunk_only,direct);
            int projections=down?1:2;
            weighted_old+=projections*(lo.old_ms+hi.old_ms);
            weighted_new+=projections*whole.new_ms;
            minimum=std::min(minimum,direct);
        }
        float effective=weighted_old/weighted_new;
        std::printf("COMBINED skew=%d 2gate_plus_down_effective_gain=%.3f\n",skew,effective);
        if(effective<1.5f) {std::puts("REJECT: combined effective throughput <1.5; stop");return 3;}
    }
    std::printf("PASS bounded timing min_direct_4096_gain=%.3f peak_MiB=%.1f; integration pending\n",minimum,double(peak)/(1<<20));
#endif
}
