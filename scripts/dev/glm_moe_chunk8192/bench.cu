// SPDX-License-Identifier: AGPL-3.0-only
// Standalone chunk-size canary. Production M64 only; synthetic data, no checkpoint.
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
#endif

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b;
    return x ^ (x >> 16);
}
struct Routing {
    std::array<int, 289> offsets{};
    std::vector<int> ids;
    int max_rows = 0;
};
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
struct Fixture {
    int rows;
    Routing whole, lo, hi;
    std::vector<int> split_to_whole, whole_to_split, split_ids;
    Fixture(int count, bool skew): rows(count), whole(routing(count, skew)),
        lo(routing(4096, skew)), hi(routing(count - 4096, skew, 4096)),
        split_to_whole(count * 8, -1), whole_to_split(count * 8, -1) {
        require(count == 8192 || count == 8196, "bounded 8K fixture");
        split_ids = lo.ids; split_ids.insert(split_ids.end(), hi.ids.begin(), hi.ids.end());
        for (int e = 0; e < 288; ++e) {
            int cursor = whole.offsets[e];
            for (int part = 0; part < 2; ++part) {
                const auto& split = part ? hi : lo;
                int first = part ? 4096 : 0, base = part ? int(lo.ids.size()) : 0;
                for (int p = split.offsets[e]; p < split.offsets[e + 1]; ++p, ++cursor) {
                    require(cursor < whole.offsets[e + 1], "split route extent");
                    require(whole.ids[cursor] == first + split.ids[p], "same expert and GLOBAL token");
                    split_to_whole[base + p] = cursor;
                    require(whole_to_split[cursor] == -1, "unique route permutation");
                    whole_to_split[cursor] = base + p;
                }
            }
            require(cursor == whole.offsets[e + 1], "complete expert route permutation");
        }
    }
};
static void host_tests() {
    for (int rows : {8192, 8196}) for (bool skew : {false, true}) {
        Fixture f(rows, skew);
        require(f.whole.ids.size() == size_t(rows * 8), "top8 extent");
        std::vector<int> counts(rows), route_seen(rows * 8);
        for (int e = 0; e < 288; ++e) {
            std::vector<bool> seen(rows);
            require(f.whole.offsets[e + 1] > f.whole.offsets[e], "all288 experts exercised");
            for (int p = f.whole.offsets[e]; p < f.whole.offsets[e + 1]; ++p) {
                int t = f.whole.ids[p];
                require(t >= 0 && t < rows && !seen[t], "unique valid expert tokens");
                seen[t] = true; ++counts[t];
                int split = f.whole_to_split[p];
                require(split >= 0 && split < rows * 8 && f.split_to_whole[split] == p, "inverse route permutation");
                ++route_seen[split];
                // Gate inputs use the actual global activation row, never a
                // regenerated local row. Down inputs gather this same canonical route.
                int first = split >= 4096 * 8 ? 4096 : 0;
                int local = f.split_ids[split];
                require(first + local == t, "global activation row preserved");
                if (first) require(first + local != local, "second chunk cannot reset activation rows");
            }
        }
        require(std::all_of(counts.begin(), counts.end(), [](int n) { return n == 8; }), "eight unique experts per token");
        require(std::all_of(route_seen.begin(), route_seen.end(), [](int n) { return n == 1; }), "split mapping covers every route exactly once");
    }
    std::puts("PASS host:8192/8196 uniform/skew, top8,144 local+144 remote experts, exact GLOBAL activation/route permutation");
}

#ifndef ATLAS_HOST_ONLY
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
constexpr size_t GiB = size_t(1) << 30;
static size_t live = 0, peak = 0;
template<class T> struct Buffer {
    T* raw = nullptr; T* p; size_t count, bytes;
    explicit Buffer(size_t n):count(n),bytes(n*sizeof(T)+256) {
        require(bytes <= 3*GiB && live <= 3*GiB-bytes, "3GiB device allocation ceiling");
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
__global__ void gather_rows(const unsigned char* canonical, unsigned char* split,
                            const int* split_to_whole, size_t bytes, unsigned row_bytes) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<bytes; i+=size_t(gridDim.x)*blockDim.x) {
        size_t row=i/row_bytes, col=i%row_bytes;
        split[i]=canonical[size_t(split_to_whole[row])*row_bytes+col];
    }
}
template<class F> static float event_time(F launch, cudaStream_t stream) {
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a,stream));
    for(int i=0;i<5;++i) launch();
    CHECK(cudaEventRecord(b,stream)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); return ms/5;
}
static void resources() {
    cudaFuncAttributes a{}; int ctas=0;
    CHECK(cudaFuncGetAttributes(&a, moe_w4a4_grouped_gemm_prequant_t_k64_vecscale));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas,moe_w4a4_grouped_gemm_prequant_t_k64_vecscale,128,0));
    std::printf("production_M64 resources smem=%zu regs=%d CTAs_per_SM=%d\n",a.sharedSizeBytes,a.numRegs,ctas);
}
struct Timing { float split_ms, whole_ms; };
static Timing run_case(int rows, bool skew, bool down) {
    Fixture f(rows,skew);
    const unsigned n=down?4096:2048, k=down?2048:4096;
    const size_t wb=size_t(n)*k/2, sb=size_t(n)*k/16;
    const int expanded=rows*8, a_rows=down?expanded:rows, second_base=4096*8;
    Buffer<unsigned char> w(144*wb),s(144*sb),a(size_t(a_rows)*k/2),as(size_t(a_rows)*k/16);
    Buffer<unsigned char> split_a(down?a.count:0),split_as(down?as.count:0);
    Buffer<unsigned long long> wp(288),sp(288); Buffer<float> scale2(288);
    Buffer<int> off_whole(289),off_lo(289),off_hi(289),ids_whole(expanded),ids_split(expanded),map(expanded);
    Buffer<__nv_bfloat16> whole(size_t(expanded)*n),split(size_t(expanded)*n);
    std::array<unsigned long long,288> wptr{},sptr{}; std::array<float,288> scales{};
    for(int e=0;e<144;++e) {
        wptr[e]=reinterpret_cast<unsigned long long>(w.p+e*wb);
        sptr[e]=reinterpret_cast<unsigned long long>(s.p+e*sb);
        scales[e]=0.31f+float(e%17)*0.07f;
    }
    wp.upload(wptr.data());sp.upload(sptr.data());scale2.upload(scales.data());
    off_whole.upload(f.whole.offsets.data());off_lo.upload(f.lo.offsets.data());off_hi.upload(f.hi.offsets.data());
    ids_whole.upload(f.whole.ids.data());ids_split.upload(f.split_ids.data());map.upload(f.split_to_whole.data());
    fill<<<4096,256>>>(w.p,w.count,false,97);fill<<<4096,256>>>(s.p,s.count,true,13);
    // Generate canonical activations ONCE. All split inputs are views/copies
    // of these exact bytes; no second fill with reset local row indices.
    fill<<<1024,256>>>(a.p,a.count,false,29);fill<<<1024,256>>>(as.p,as.count,true,51);
    if(down) {
        gather_rows<<<1024,256>>>(a.p,split_a.p,map.p,a.count,k/2);
        gather_rows<<<1024,256>>>(as.p,split_as.p,map.p,as.count,k/16);
    }
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto launch_whole=[&] {
        dim3 grid((n+127)/128,(f.whole.max_rows+63)/64,288);
        moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<grid,128,0,stream>>>(
            a.p,as.p,wp.p,sp.p,scale2.p,whole.p,off_whole.p,down?nullptr:ids_whole.p,288,n,k);
        CHECK(cudaGetLastError());
    };
    auto launch_split=[&] {
        for(int part=0;part<2;++part) {
            int token_base=part?4096:0,route_base=part?second_base:0;
            int max_rows=part?f.hi.max_rows:f.lo.max_rows;
            auto* ap=down?split_a.p+size_t(route_base)*k/2:a.p+size_t(token_base)*k/2;
            auto* asp=down?split_as.p+size_t(route_base)*k/16:as.p+size_t(token_base)*k/16;
            dim3 grid((n+127)/128,(max_rows+63)/64,288);
            moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<grid,128,0,stream>>>(
                ap,asp,wp.p,sp.p,scale2.p,split.p+size_t(route_base)*n,
                part?off_hi.p:off_lo.p,down?nullptr:ids_split.p+route_base,288,n,k);
            CHECK(cudaGetLastError());
        }
    };
    CHECK(cudaMemset(whole.p,0xa5,whole.count*2));CHECK(cudaMemset(split.p,0x5a,split.count*2));
    launch_whole();launch_split();CHECK(cudaStreamSynchronize(stream));
    {
        auto x=whole.read(),y=split.read();
        const auto* xb=reinterpret_cast<const unsigned short*>(x.data());
        const auto* yb=reinterpret_cast<const unsigned short*>(y.data());
        size_t checked=0;
        for(int e=0;e<288;++e) for(int p=f.whole.offsets[e];p<f.whole.offsets[e+1];++p) {
            size_t xi=size_t(p)*n,yi=size_t(f.whole_to_split[p])*n;
            for(unsigned col=0;col<n;++col) {
                unsigned short a_bits=xb[xi+col],b_bits=yb[yi+col];
                if(e<144) {
                    if(a_bits!=b_bits || (a_bits&0x7f80)==0x7f80) {
                        std::fprintf(stderr,"FAIL oracle rows=%d skew=%d down=%d expert=%d GLOBALtoken=%d col=%u whole=%04x split=%04x\n",
                            rows,skew,down,e,f.whole.ids[p],col,a_bits,b_bits);std::exit(2);
                    }
                    ++checked;
                } else require(a_bits==0xa5a5 && b_bits==0x5a5a,"remote row written");
            }
        }
        require(checked>0,"oracle checked local output");
    }
    whole.guards();split.guards();w.guards();s.guards();a.guards();as.guards();split_a.guards();split_as.guards();
    wp.guards();sp.guards();scale2.guards();off_whole.guards();off_lo.guards();off_hi.guards();ids_whole.guards();ids_split.guards();map.guards();
    for(int i=0;i<3;++i){launch_whole();launch_split();}CHECK(cudaStreamSynchronize(stream));
    std::vector<float> split_times,whole_times;
    for(int r=0;r<5;++r) {
        if(r%2) {whole_times.push_back(event_time(launch_whole,stream));split_times.push_back(event_time(launch_split,stream));}
        else {split_times.push_back(event_time(launch_split,stream));whole_times.push_back(event_time(launch_whole,stream));}
    }
    std::sort(split_times.begin(),split_times.end());std::sort(whole_times.begin(),whole_times.end());
    std::printf("PASS bitwise oracle rows=%d split=4096+%d skew=%d down=%d max_expert_rows=%d split_M64_ms=%.6f whole_M64_ms=%.6f gain=%.3f\n",
        rows,rows-4096,skew,down,f.whole.max_rows,split_times[2],whole_times[2],split_times[2]/whole_times[2]);
    CHECK(cudaStreamDestroy(stream));return {split_times[2],whole_times[2]};
}
#endif

int main(int argc,char** argv) {
    require(argc>=2 && argc<=3 && (!std::strcmp(argv[1],"--host-test") || !std::strcmp(argv[1],"--run")),"usage: bench --host-test | --run [minimum_gain]");
    host_tests();if(!std::strcmp(argv[1],"--host-test"))return 0;
#ifdef ATLAS_HOST_ONLY
    require(false,"host-only binary cannot run CUDA");
#else
    float minimum_gain=argc==3?std::strtof(argv[2],nullptr):1.15f;
    require(std::isfinite(minimum_gain) && minimum_gain>=1.0f,"finite minimum_gain>=1");
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));
    require(free>=9*GiB,"need9GiB free:3GiB device cap, host oracle copies and4GiB reserve");
    resources();
    float worst=1000;
    for(int rows:{8192,8196}) for(bool skew:{false,true}) {
        auto gate=run_case(rows,skew,false),down=run_case(rows,skew,true);
        float split=2*gate.split_ms+down.split_ms,whole=2*gate.whole_ms+down.whole_ms,gain=split/whole;
        worst=std::min(worst,gain);
        std::printf("COMBINED rows=%d skew=%d 2gate_plus_down split_ms=%.6f whole_ms=%.6f gain=%.3f threshold=%.3f\n",rows,skew,split,whole,gain,minimum_gain);
        if(gain<minimum_gain) {std::puts("REJECT: weighted chunk gain below threshold; fail fast");return 3;}
    }
    std::printf("PASS macro canary min_weighted_gain=%.3f peak_device_MiB=%.1f; engine integration pending\n",worst,double(peak)/(1<<20));
#endif
}
