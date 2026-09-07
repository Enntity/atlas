// SPDX-License-Identifier: AGPL-3.0-only

// Standalone only; see glm_moe_down_cost_plan.md. Never loads a model.
// nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_down_cost.cu -o /tmp/bench-glm-moe-down-cost
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#ifndef ATLAS_MOE_DOWN_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../kernels/gb10/common/moe_permute.cu"
#include "glm_moe_m16n128.cuh"
#endif

#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
#ifndef ATLAS_MOE_TEST_ROWS
#define ATLAS_MOE_TEST_ROWS 4
#endif
static_assert(ATLAS_MOE_TEST_ROWS == 4 || ATLAS_MOE_TEST_ROWS == 5, "fixture supports C4 or K5 only");
constexpr unsigned max_rows = ATLAS_MOE_TEST_ROWS;
constexpr unsigned dn = 4096, dk = 2048, experts = 288, routes = 8 * max_rows, weights = 8;
constexpr size_t packed_weight = size_t(dn) * dk / 2, scale_weight = size_t(dn) * dk / 16;
constexpr size_t memory_limit = 64ULL * 1024 * 1024;
static size_t device_live = 0, device_peak = 0;
static bool allocation_fits(size_t count, size_t element, size_t live) {
    return element && count <= (memory_limit - 256) / element
        && live <= memory_limit - (count * element + 256);
}
#ifndef ATLAS_MOE_DOWN_HOST_ONLY
template<class T> struct Buffer {
    T* allocation; T* ptr; size_t count, bytes;
    explicit Buffer(size_t n) : count(n) {
        require(allocation_fits(n, sizeof(T), device_live), "64MiB device budget/overflow");
        bytes = n * sizeof(T) + 256;
        CHECK(cudaMalloc(&allocation, bytes));
        device_live += bytes; device_peak = std::max(device_peak, device_live);
        CHECK(cudaMemset(allocation, 0xa5, bytes));
        ptr = allocation + 128 / sizeof(T);
    }
    ~Buffer() { cudaFree(allocation); device_live -= bytes; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& data, size_t offset = 0) {
        require(offset <= count && data.size() <= count - offset, "upload bounds");
        CHECK(cudaMemcpy(ptr + offset, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read(size_t offset = 0, size_t length = 0) const {
        if (!length) length = count;
        require(offset <= count && length <= count - offset, "read bounds");
        std::vector<T> out(length);
        CHECK(cudaMemcpy(out.data(), ptr + offset, length * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
    void guards() const {
        unsigned char a[128], b[128];
        CHECK(cudaMemcpy(a, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i) require(a[i] == 0xa5 && b[i] == 0xa5, "allocation guard");
    }
};
#endif
template<class T> static void exact(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    require(a.size() == b.size(), "comparison extent");
    for (size_t i = 0; i < a.size(); ++i) if (std::memcmp(&a[i], &b[i], sizeof(T))) {
        std::fprintf(stderr, "FAIL: %s element=%zu\n", label, i); std::exit(2);
    }
}
// All scale values are dyadic: exact FP32 accumulation at this bounded K.
static unsigned char packed_code(size_t index, unsigned seed) {
    unsigned x = unsigned(index) ^ (seed * 0x9e3779b9u);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    return static_cast<unsigned char>(x ^ (x >> 8));
}
static unsigned char scale_code(size_t index, unsigned seed) {
    return static_cast<unsigned char>(0x20 + 8 * ((index + seed) % 3));
}
static double fp4(unsigned value) {
    const double positive[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return (value & 8 ? -1 : 1) * positive[value & 7];
}
static double scale(unsigned char code) { return std::ldexp(1.0, int(code >> 3) - 7); }
static unsigned short bf16_bits(double value) {
    float f = float(value); unsigned bits; std::memcpy(&bits, &f, 4);
    return static_cast<unsigned short>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
struct Map {
    const char* name;
    std::array<int, experts> counts{};
    std::array<int, experts> local;
    explicit Map(const char* label) : name(label) { local.fill(-1); }
};
static bool valid_map(const Map& map) {
    unsigned total = 0; std::array<bool, weights> used{};
    for (unsigned e = 0; e < experts; ++e) {
        if (map.counts[e] < 0 || map.counts[e] > int(max_rows) || map.local[e] < -1 || map.local[e] >= int(weights)) return false;
        total += unsigned(map.counts[e]);
        if (map.counts[e] && map.local[e] >= 0) {
            if (used[map.local[e]]) return false;
            used[map.local[e]] = true;
        }
    }
    return total <= routes;
}
static std::vector<int> offsets_for(const Map& map) {
    require(valid_map(map), "invalid map cannot reach CUDA");
    std::vector<int> offsets(experts + 1);
    for (unsigned e = 0; e < experts; ++e) offsets[e + 1] = offsets[e] + map.counts[e];
    return offsets;
}
static std::vector<unsigned> expected_work(const Map& map) {
    require(valid_map(map), "worklist host validation");
    std::vector<unsigned> result;
    for (unsigned e = 0; e < experts; ++e) if (map.counts[e] && map.local[e] >= 0)
        for (unsigned nt = 0; nt < dn / 128; ++nt) { result.push_back(e); result.push_back(nt); }
    require(result.size() <= routes * (dn / 128) * 2, "worklist capacity");
    return result; // Every supported expert has one M64 tile, so m_tile=0.
}
static std::vector<Map> fixtures() {
    Map first("first8"), edge("boundary-mixed"), skew("single-local"), empty("all-empty"), remote("remote-only"), varied("varied-rows-permuted");
    for (unsigned e = 0; e < 8; ++e) {
        first.counts[e] = remote.counts[e] = int(max_rows); first.local[e] = int(e);
        varied.counts[e] = int(e % max_rows) + 1; varied.local[e] = int(7 - e);
    }
    const unsigned ids[] = {0, 17, 142, 143, 144, 145, 286, 287};
    for (unsigned i = 0; i < 8; ++i) { edge.counts[ids[i]] = int(max_rows); edge.local[ids[i]] = i % 2 ? -1 : int(i); }
    skew.counts[287] = int(max_rows); skew.local[287] = 7;
    return {first, edge, skew, empty, remote, varied};
}
static void host_tests() {
    // Independent coordinate check for the candidate's four N32 warp slices.
    std::array<unsigned,16*128> written{};
    for(unsigned warp=0;warp<4;++warp)for(unsigned lane=0;lane<32;++lane)
        for(unsigned nt=0;nt<4;++nt) {
            const unsigned row=lane/4, col=warp*32+nt*8+(lane%4)*2;
            require(row+8<16 && col+1<128,"M16 output coordinate bounds");
            ++written[row*128+col];++written[row*128+col+1];
            ++written[(row+8)*128+col];++written[(row+8)*128+col+1];
        }
    require(std::all_of(written.begin(),written.end(),[](unsigned n){return n==1;}),
            "M16 warp partition must cover each output exactly once");
    auto maps = fixtures();
    for (const auto& map : maps) require(valid_map(map), "positive map fixture");
    auto work = expected_work(maps[0]);
    require(work.size() == 8 * 32 * 2 && work[0] == 0 && work[1] == 0
            && work[work.size()-2] == 7 && work.back() == 31, "independent down map covers32 tiles");
    require(expected_work(maps[3]).empty() && expected_work(maps[4]).empty(), "empty/remote maps");
    Map invalid = maps[0]; invalid.counts[0] = int(max_rows) + 1; require(!valid_map(invalid), "reject expert row overflow");
    invalid = maps[0]; invalid.counts[8] = 1; require(!valid_map(invalid), "reject total route overflow");
    invalid = maps[0]; invalid.local[1] = 0; require(!valid_map(invalid), "reject live weight alias");
    invalid = maps[0]; invalid.local[1] = 8; require(!valid_map(invalid), "reject weight index");
    require(allocation_fits(1, 4, 0) && !allocation_fits(std::numeric_limits<size_t>::max(), 4, 0)
            && !allocation_fits(1, 4, memory_limit), "checked allocation gate");
    require(offsets_for(maps[0])[8] == int(routes) && offsets_for(maps[0]).back() == int(routes), "trailing empty offsets");
    std::printf("PASS host map/capacity/aliasing tests (no CUDA calls)\n");
}

int main(int argc, char** argv) {
    bool host_only=false, timing=false, m16=false;
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"--host-test") && !host_only) host_only=true;
        else if(!std::strcmp(argv[i],"--timing") && !timing) timing=true;
        else if(!std::strcmp(argv[i],"--m16") && !m16) m16=true;
        else require(false,"usage: bench-glm-moe-down-cost [--host-test|[--timing] [--m16]]");
    }
    require(!host_only || (!timing && !m16),"--host-test is standalone");
    host_tests(); if (host_only) return 0;
#ifdef ATLAS_MOE_DOWN_HOST_ONLY
    require(false, "CPU-only build supports --host-test only");
#else
    Buffer<unsigned char> bp(weights * packed_weight), bs(weights * scale_weight);
    Buffer<unsigned char> ap(size_t(routes) * dk / 2), as(size_t(routes) * dk / 16);
    Buffer<__nv_bfloat16> full(size_t(routes) * dn), trimmed(full.count), compact(full.count);
    Buffer<unsigned long long> bptr(experts), sptr(experts);
    Buffer<float> tensor_scale(experts);
    Buffer<int> eoff(experts + 1), total(1);
    Buffer<unsigned> work(routes * (dn / 128) * 2);
    for (unsigned w = 0; w < weights; ++w) {
        std::vector<unsigned char> data(packed_weight);
        for (size_t i = 0; i < data.size(); ++i) data[i] = packed_code(i, w + 11);
        bp.upload(data, w * packed_weight);
        data.resize(scale_weight);
        for (size_t i = 0; i < data.size(); ++i) data[i] = scale_code(i, w);
        bs.upload(data, w * scale_weight);
    }
    std::vector<unsigned char> host_a(ap.count), host_s(as.count);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = packed_code(i, 99);
    for (size_t i = 0; i < host_s.size(); ++i) host_s[i] = scale_code(i, 3);
    ap.upload(host_a); as.upload(host_s);
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto builder = [&]() {
        moe_build_tile_worklist<<<1,256,0,stream>>>(eoff.ptr, bptr.ptr, work.ptr, total.ptr, experts, dn / 128, 64);
        CHECK(cudaGetLastError());
    };
    auto dense = [&](unsigned count, __nv_bfloat16* out, bool vec = true) {
        if (vec) moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dim3(dn/128,1,count),128,0,stream>>>(
            ap.ptr, as.ptr, bptr.ptr, sptr.ptr, tensor_scale.ptr, out, eoff.ptr, nullptr, count, dn, dk);
        else moe_w4a4_grouped_gemm_prequant_t_k64<<<dim3(dn/128,1,count),128,0,stream>>>(
            ap.ptr, as.ptr, bptr.ptr, sptr.ptr, tensor_scale.ptr, out, eoff.ptr, nullptr, count, dn, dk);
        CHECK(cudaGetLastError());
    };
    auto compact_launch = [&](bool vec = true) {
        if (vec) moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact<<<routes*(dn/128),128,0,stream>>>(
            ap.ptr, as.ptr, bptr.ptr, sptr.ptr, tensor_scale.ptr, compact.ptr, eoff.ptr, nullptr,
            experts, dn, dk, work.ptr, total.ptr, routes*(dn/128));
        else moe_w4a4_grouped_gemm_prequant_t_k64_compact<<<routes*(dn/128),128,0,stream>>>(
            ap.ptr, as.ptr, bptr.ptr, sptr.ptr, tensor_scale.ptr, compact.ptr, eoff.ptr, nullptr,
            experts, dn, dk, work.ptr, total.ptr, routes*(dn/128));
        CHECK(cudaGetLastError());
    };
    auto m16_launch = [&](bool vec = true) {
        if(vec) glm_moe_down_m16n128_vecscale<<<dim3(dn/128,1,experts),128,0,stream>>>(
            ap.ptr,as.ptr,bptr.ptr,sptr.ptr,tensor_scale.ptr,trimmed.ptr,eoff.ptr,nullptr,experts,dn,dk);
        else glm_moe_down_m16n128<<<dim3(dn/128,1,experts),128,0,stream>>>(
            ap.ptr,as.ptr,bptr.ptr,sptr.ptr,tensor_scale.ptr,trimmed.ptr,eoff.ptr,nullptr,experts,dn,dk);
        CHECK(cudaGetLastError());
    };
    cudaGraph_t graph; cudaGraphExec_t executable;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    dense(experts, full.ptr); builder(); compact_launch();
    if(m16) m16_launch();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    std::printf("explicit_device_bytes=%zu hard_cap=%zu K=%u N=%u synthetic8weights\n", device_peak, memory_limit, dk, dn);
    std::printf("M16_candidate=%s max_rows_per_expert=%u route_capacity=%u\n",m16?"on":"off",max_rows,routes);
    const auto maps = fixtures();
    // Timing never replaces a gate: first finish all eager and graph cases.
    // A third pass refreshes/revalidates each fixture before its timing interval.
    for (unsigned round = 0; round < (timing ? 3u : 2u); ++round) for (const auto& map : maps) {
        const auto offsets = offsets_for(map); const auto expected = expected_work(map);
        std::vector<unsigned long long> hp(experts), hs(experts); std::vector<float> sf(experts, 1.0f);
        for (unsigned e = 0; e < experts; ++e) if (map.local[e] >= 0) {
            const unsigned w = unsigned(map.local[e]);
            hp[e] = reinterpret_cast<unsigned long long>(bp.ptr + w * packed_weight);
            hs[e] = reinterpret_cast<unsigned long long>(bs.ptr + w * scale_weight);
            sf[e] = std::ldexp(1.0f, int(w % 3) - 1);
        }
        eoff.upload(offsets); bptr.upload(hp); sptr.upload(hs); tensor_scale.upload(sf);
        const bool can_trim = std::all_of(map.counts.begin()+8, map.counts.end(), [](int n){return n==0;});
        for (auto* out : {&full,&trimmed,&compact}) CHECK(cudaMemsetAsync(out->ptr,0x5a,out->count*2,stream));
        CHECK(cudaMemsetAsync(work.ptr,0xa5,work.count*4,stream));
        CHECK(cudaMemsetAsync(total.ptr,0xa5,4,stream));
        if (round) CHECK(cudaGraphLaunch(executable,stream));
        else { dense(experts,full.ptr); builder(); compact_launch(); if(m16)m16_launch(); }
        CHECK(cudaStreamSynchronize(stream));
        require(total.read()[0] == int(expected.size()/2), "device work count vs independent host");
        const auto actual_work = work.read();
        for (size_t i=0;i<actual_work.size();++i)
            require(actual_work[i] == (i<expected.size()?expected[i]:0xa5a5a5a5u), "worklist entry/tail poison");
        const auto reference = full.read(); exact(reference,compact.read(),"dense/compact vecscale");
        if(m16) {exact(reference,trimmed.read(),"production dense/M16 vecscale");trimmed.guards();}
        if (can_trim) { dense(8,trimmed.ptr); CHECK(cudaStreamSynchronize(stream)); exact(reference,trimmed.read(),"dense288/dense8"); }
        // Both scale loaders must retain the same complete output and remote poison.
        // Re-poison: correct earlier vecscale writes must not hide omitted scalar writes.
        CHECK(cudaMemsetAsync(trimmed.ptr,0x5a,trimmed.count*2,stream));
        CHECK(cudaMemsetAsync(compact.ptr,0x5a,compact.count*2,stream));
        dense(experts,trimmed.ptr,false); compact_launch(false); CHECK(cudaStreamSynchronize(stream));
        trimmed.guards(); compact.guards();
        exact(reference,trimmed.read(),"dense scalar/vecscale"); exact(reference,compact.read(),"compact scalar/vecscale");
        if(m16) {
            CHECK(cudaMemsetAsync(trimmed.ptr,0x5a,trimmed.count*2,stream));
            m16_launch(false);CHECK(cudaStreamSynchronize(stream));trimmed.guards();
            exact(reference,trimmed.read(),"production dense/M16 scalar scale");
        }
        for (unsigned row=0;row<routes;++row) {
            unsigned e=0; while (e<experts && offsets[e+1]<=int(row)) ++e;
            const bool local = e<experts && map.local[e]>=0;
            for (unsigned col=0;col<dn;++col) {
                unsigned short bits; std::memcpy(&bits,&reference[size_t(row)*dn+col],2);
                if (!local) require(bits==0x5a5a,"remote/unused output modified");
                else require(std::isfinite(float(reference[size_t(row)*dn+col])),"nonfinite local output");
            }
            if (!local) continue;
            const unsigned w=unsigned(map.local[e]);
            for (unsigned col : {0u,1u,63u,127u,128u,2047u,4095u}) {
                double sum=0;
                for (unsigned k=0;k<dk;++k) {
                    const unsigned ac=host_a[size_t(row)*(dk/2)+k/2];
                    const unsigned bc=packed_code(size_t(k/2)*dn+col,w+11);
                    sum += fp4((ac >> (4*(k%2))) & 15) * scale(host_s[size_t(row)*(dk/16)+k/16])
                        * fp4((bc >> (4*(k%2))) & 15) * scale(scale_code(size_t(k/16)*dn+col,w));
                }
                unsigned short bits; std::memcpy(&bits,&reference[size_t(row)*dn+col],2);
                if (bits != bf16_bits(sum*sf[e])) {
                    std::fprintf(stderr,"FAIL CPU oracle map=%s row=%u col=%u got=%04x expected=%04x\n",map.name,row,col,bits,bf16_bits(sum*sf[e]));
                    std::exit(2);
                }
            }
        }
        exact(offsets,eoff.read(),"immutable offsets"); exact(hp,bptr.read(),"immutable B pointers");
        exact(hs,sptr.read(),"immutable scale pointers"); exact(sf,tensor_scale.read(),"immutable scale2");
        exact(host_a,ap.read(),"immutable packed A"); exact(host_s,as.read(),"immutable A scales");
        bp.guards();bs.guards();ap.guards();as.guards();full.guards();trimmed.guards();compact.guards();
        bptr.guards();sptr.guards();tensor_scale.guards();eoff.guards();total.guards();work.guards();
        std::printf("PASS map=%s mode=%s full_output_BITEXACT CPU_columns remote_poison worklist_guards\n",map.name,round?"graph":"eager");
        if (timing && round == 2 && can_trim) {
            auto variant = [&](unsigned v) { if(v==0)dense(8,trimmed.ptr); else if(v==1)dense(experts,full.ptr);
                else if(v==2)compact_launch(); else if(v==3)builder(); else if(v==4){builder();compact_launch();} else m16_launch(); };
            const unsigned variants=m16?6:5;
            for(unsigned i=0;i<5;++i)for(unsigned v=0;v<variants;++v)variant(v);
            cudaEvent_t begin,end; CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
            std::vector<float> samples[6];
            for(unsigned trial=0;trial<5;++trial)for(unsigned order=0;order<variants;++order){
                const unsigned v=(trial+order)%variants;
                CHECK(cudaEventRecord(begin,stream));for(unsigned i=0;i<100;++i)variant(v);
                CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
                float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));samples[v].push_back(ms*10.0f);
            }
            const char* names[]={"dense8","dense288","compact_only","builder_only","builder_plus_compact","m16_dense288"};
            for(unsigned v=0;v<variants;++v){std::sort(samples[v].begin(),samples[v].end());
                std::printf("TIMING map=%s path=%s us=%.3f eager_events median5x100 interleaved setup_excluded\n",map.name,names[v],samples[v][2]);}
            CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));
        }
    }
    for(unsigned w=0;w<weights;++w){
        auto data=bp.read(w*packed_weight,packed_weight);
        for(size_t i=0;i<data.size();++i)require(data[i]==packed_code(i,w+11),"weight mutation");
        data=bs.read(w*scale_weight,scale_weight);
        for(size_t i=0;i<data.size();++i)require(data[i]==scale_code(i,w),"weight scale mutation");
    }
    CHECK(cudaGraphExecDestroy(executable));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
    std::printf("PASS completed mode=%s peak_explicit_device_bytes=%zu no_production_promotion\n",timing?"timing":"correctness",device_peak);
#endif
}
