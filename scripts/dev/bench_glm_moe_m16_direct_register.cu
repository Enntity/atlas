// SPDX-License-Identifier: AGPL-3.0-only
// Standalone original-T M16 direct-register fixture; see glm_moe_m16_direct_register_plan.md.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "glm_moe_m16_direct_fragment.h"
#ifndef ATLAS_MOE_DOWN_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../kernels/gb10/common/moe_permute.cu"
#include "glm_moe_m16_direct_register.cuh"
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
constexpr unsigned dn = 2048, dk = 4096, experts = 288, routes = 8 * max_rows, weights = 4;
constexpr size_t packed_weight = size_t(dn) * dk / 2, scale_weight = size_t(dn) * dk / 16;
constexpr size_t memory_limit = 64ULL * 1024 * 1024;
static size_t device_live = 0, device_peak = 0;
static bool allocation_fits(size_t count, size_t element, size_t live) {
    return element && count <= (memory_limit - 256) / element
        && live <= memory_limit - (count * element + 256);
}
static size_t fixture_budget(){
    const size_t sizes[]={2*weights*packed_weight,2*weights*scale_weight,
        size_t(max_rows)*dk/2,size_t(max_rows)*dk/16,
        size_t(routes)*dn*2,size_t(routes)*dn*2,size_t(routes)*dn*2,
        size_t(routes)*dn*2,size_t(routes)*dn*2,size_t(routes)*dn*2,
        experts*8,experts*8,experts*8,experts*8,experts*4,experts*4,
        (experts+1)*4,routes*4,4,routes*(dn/128)*2*4};
    size_t live=0;
    for(size_t bytes:sizes){require(allocation_fits(bytes,1,live),"exact guarded fixture budget");live+=bytes+256;}
    return live;
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
    ~Buffer() { CHECK(cudaFree(allocation)); device_live -= bytes; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& data, size_t offset = 0, cudaStream_t stream = 0) {
        require(offset <= count && data.size() <= count - offset, "upload bounds");
        CHECK(cudaMemcpyAsync(ptr + offset, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
        // Explicitly complete this source's upload before callers resize or
        // destroy their pageable vector. Request metadata uses the work stream.
        CHECK(cudaStreamSynchronize(stream));
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

struct Case {
    const char* name;
    std::array<int,experts> count{}, local;
    std::vector<int> ids;
    explicit Case(const char* n):name(n){local.fill(-1);}
};
static bool valid(const Case& c) {
    unsigned total=0;std::array<bool,weights> seen{};
    for(unsigned e=0;e<experts;++e){
        if(c.count[e]<0||c.count[e]>int(max_rows)||c.local[e]<-1||c.local[e]>=int(weights))return false;
        if(c.count[e]&&c.local[e]>=0){if(seen[c.local[e]])return false;seen[c.local[e]]=true;}
        std::array<bool,max_rows> tokens{};
        for(int j=0;j<c.count[e];++j){
            if(total>=c.ids.size())return false;
            int id=c.ids[total++];
            if(id<0||id>=int(max_rows)||tokens[id])return false;
            tokens[id]=true;
        }
    }
    return total==c.ids.size()&&total<=routes;
}
static void fill_ids(Case& c,unsigned salt){
    c.ids.clear();
    for(unsigned e=0;e<experts;++e)for(int j=0;j<c.count[e];++j)
        c.ids.push_back(int((max_rows-1-unsigned(j)+e+salt)%max_rows));
}
static std::vector<Case> cases(){
    Case dense("four-local-four-remote"),edge("boundaries"),varied("varied-gather"),empty("empty"),remote("remote-only"),partial("partial-all-local");
    unsigned ids[]={0,17,142,143,144,145,286,287};
    for(unsigned i=0;i<8;++i){
        dense.count[i]=edge.count[ids[i]]=remote.count[i]=int(max_rows);
        if(i<4){dense.local[i]=int(i);edge.local[ids[2*i]]=int(3-i);partial.count[i]=int(max_rows);partial.local[i]=int(i);}
        varied.count[ids[i]]=int(i%max_rows)+1;
        if(i%2==0)varied.local[ids[i]]=int(i/2);
    }
    std::vector<Case> all={dense,edge,varied,empty,remote,partial};
    const char* population_names[]={"population1","population2","population3","population4"};
    for(unsigned population=1;population<=4;++population){
        Case value(population_names[population-1]);
        for(unsigned i=0;i<8;++i){value.count[ids[i]]=int(population);if(i%2==0)value.local[ids[i]]=int(i/2);}
        all.push_back(value);
    }
    for(unsigned i=0;i<all.size();++i)fill_ids(all[i],i);
    return all;
}
static std::vector<int> offsets(const Case& c){
    require(valid(c),"invalid map/gather cannot reach CUDA");std::vector<int> out(experts+1);
    for(unsigned e=0;e<experts;++e)out[e+1]=out[e]+c.count[e];return out;
}
static std::vector<unsigned> expected_work(const Case& c){
    require(valid(c),"invalid work map");std::vector<unsigned> out;
    for(unsigned e=0;e<experts;++e)if(c.count[e]&&c.local[e]>=0)
        for(unsigned n=0;n<dn/128;++n){out.push_back(e);out.push_back(n);}
    require(out.size()<=routes*(dn/128)*2,"work bound");return out;
}
static void fragment_tests(){
    std::array<unsigned char,2*32*144> stages{},snapshot{};
    std::array<unsigned char,128*48> materialized{};
    std::array<unsigned char,32*128> seen{};
    for(unsigned pattern=0;pattern<3;++pattern){
        for(size_t i=0;i<stages.size();++i)
            stages[i]=static_cast<unsigned char>((i*(17+pattern*6))^(i>>3)^(pattern*71));
        snapshot=stages;
        for(unsigned buffer=0;buffer<2;++buffer){
            auto* source=stages.data()+buffer*32*144;
            materialized.fill(0xa5);seen.fill(0);
            // Independent old software-transpose output; byte assignments only.
            for(unsigned col=0;col<128;++col)for(unsigned kp=0;kp<32;++kp)
                materialized[col*48+kp]=source[kp*144+col];
            for(unsigned thread=0;thread<128;++thread)for(unsigned nt=0;nt<4;++nt)
                for(unsigned word=0;word<2;++word){
                    const unsigned lane=thread%32,tid=lane%4;
                    const unsigned col=(thread/32)*32+nt*8+lane/4;
                    const unsigned kp=word*16+tid*4;
                    uint32_t reference;std::memcpy(&reference,&materialized[col*48+kp],4);
                    require(glm_m16_direct_fragment(source,144,col,kp)==reference,"actual direct fragment vs materialized transpose bits");
                    for(unsigned byte=0;byte<4;++byte)++seen[(kp+byte)*128+col];
                }
            for(auto count:seen)require(count==1,"every active B-stage byte consumed exactly once");
            for(unsigned col=0;col<128;++col)for(unsigned pad=32;pad<48;++pad)
                require(materialized[col*48+pad]==0xa5,"materialized padding untouched");
        }
        require(stages==snapshot,"both source stages and padded columns immutable");
    }
    // Single-bit probes independently fix little-endian byte/bit positions.
    stages.fill(0);for(unsigned buffer=0;buffer<2;++buffer){
        auto* source=stages.data()+buffer*32*144;
        for(unsigned col=0;col<128;++col)for(unsigned kp=0;kp<32;++kp)
            for(unsigned bit=0;bit<8;++bit){
                source[kp*144+col]=static_cast<unsigned char>(1u<<bit);
                const unsigned start=(kp/4)*4;
                require(glm_m16_direct_fragment(source,144,col,start)==(uint32_t(1)<<((kp%4)*8+bit)),"isolated B byte/bit mapping");
                require(glm_m16_direct_fragment(source,144,col,start^16)==0,"other B fragment word unaffected");
                source[kp*144+col]=0;
            }
    }
    std::puts("PASS actual_direct_fragment both_stage_buffers all128_lanes four_N_fragments both_words full4096_byte_ownership isolated_bits immutable_padding");
}

static void host_tests(){
    fragment_tests();
    auto all=cases();for(const auto& c:all)require(valid(c),"valid gathered fixture");
    require(expected_work(all[0]).size()==4*16*2,"gate/up16 Ntiles");
    require(expected_work(all[3]).empty()&&expected_work(all[4]).empty(),"remote/empty skips");
    auto c=all[0];c.ids[0]=int(max_rows);require(!valid(c),"reject out-of-range gather");
    c=all[0];c.ids[1]=c.ids[0];require(!valid(c),"reject duplicate within expert");
    c=all[0];c.local[1]=0;require(!valid(c),"reject live pair alias");
    c=all[0];c.count[0]=int(max_rows)+1;require(!valid(c),"reject M overflow");
    c=all[0];c.ids.pop_back();require(!valid(c),"reject truncated gather");
    require(allocation_fits(1,4,0)&&!allocation_fits(std::numeric_limits<size_t>::max(),4,0),"allocation overflow");
    require(fixture_budget()==(max_rows==4?38566408:38766376),"exact20-allocation footprint");
    std::array<bool,6> populations{};
    for(const auto& value:all){
        const auto eo=offsets(value);const auto work=expected_work(value);
        std::vector<unsigned> coverage(routes);
        for(size_t i=0;i<work.size();i+=2)if(work[i+1]==0){
            unsigned expert=work[i];populations[value.count[expert]]=true;
            for(int row=eo[expert];row<eo[expert+1];++row)++coverage[row];
        }
        for(unsigned row=0;row<routes;++row){unsigned expert=0;while(expert<experts&&eo[expert+1]<=int(row))++expert;
            require(coverage[row]==unsigned(expert<experts&&value.local[expert]>=0),"all local rows/remote gaps covered");}
    }
    for(unsigned population=1;population<=max_rows;++population)require(populations[population],"all populations1..C4/K5 tested");
    std::printf("PASS host gathered rows=%u populations1_to_%u maps/bounds/aliasing device_bytes=%zu\n",max_rows,max_rows,fixture_budget());
}

int main(int argc,char** argv){
    const bool host=argc==2&&!std::strcmp(argv[1],"--host-test");
    const bool timing=argc==2&&!std::strcmp(argv[1],"--timing");
    require(argc==1||host||timing,"usage: bench-glm-moe-m16-direct-register [--host-test|--timing]");
    host_tests();if(host)return 0;
#ifdef ATLAS_MOE_DOWN_HOST_ONLY
    require(false,"CPU-only build supports --host-test only");
#else
    Buffer<unsigned char> bp(2*weights*packed_weight),bs(2*weights*scale_weight);
    Buffer<unsigned char> ap(size_t(max_rows)*dk/2),as(size_t(max_rows)*dk/16);
    Buffer<__nv_bfloat16> old_gate(size_t(routes)*dn),old_up(old_gate.count),m16_gate(old_gate.count),m16_up(old_gate.count),direct_gate(old_gate.count),direct_up(old_gate.count);
    Buffer<unsigned long long> gp(experts),gs(experts),up(experts),us(experts);
    Buffer<float> gsf(experts),usf(experts);
    Buffer<int> off(experts+1),gather(routes),total(1);
    Buffer<unsigned> work(routes*(dn/128)*2);
    CHECK(cudaDeviceSynchronize()); // complete default-stream guard initialization
    require(device_live==fixture_budget(),"actual allocation sum matches CPU budget");
    auto check_guards=[&](){
        bp.guards();bs.guards();ap.guards();as.guards();old_gate.guards();old_up.guards();
        m16_gate.guards();m16_up.guards();direct_gate.guards();direct_up.guards();
        gp.guards();gs.guards();up.guards();us.guards();gsf.guards();usf.guards();
        off.guards();gather.guards();total.guards();work.guards();
    };
    auto check_weights=[&](){
        for(unsigned w=0;w<2*weights;++w){
            auto bytes=bp.read(w*packed_weight,packed_weight);
            for(size_t i=0;i<bytes.size();++i)require(bytes[i]==packed_code(i,w+31),"original T packed weight changed");
            bytes=bs.read(w*scale_weight,scale_weight);
            for(size_t i=0;i<bytes.size();++i)require(bytes[i]==scale_code(i,w),"original T weight scale changed");
        }
    };
    for(unsigned w=0;w<2*weights;++w){
        std::vector<unsigned char> bytes(packed_weight);
        for(size_t i=0;i<bytes.size();++i)bytes[i]=packed_code(i,w+31);bp.upload(bytes,w*packed_weight);
        bytes.resize(scale_weight);for(size_t i=0;i<bytes.size();++i)bytes[i]=scale_code(i,w);bs.upload(bytes,w*scale_weight);
    }
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto build=[&](){moe_build_tile_worklist<<<1,256,0,stream>>>(off.ptr,gp.ptr,work.ptr,total.ptr,experts,dn/128,64);CHECK(cudaGetLastError());};
    auto launch=[&](unsigned variant,bool vec=true){
        require(variant<3,"kernel variant");
        const dim3 grid(routes*(dn/128),2);
        auto* gate_out=variant==0?old_gate.ptr:variant==1?m16_gate.ptr:direct_gate.ptr;
        auto* up_out=variant==0?old_up.ptr:variant==1?m16_up.ptr:direct_up.ptr;
#define GU_ARGS ap.ptr,as.ptr,gp.ptr,gs.ptr,gsf.ptr,gate_out,up.ptr,us.ptr,usf.ptr,up_out,off.ptr,gather.ptr,experts,dn,dk,work.ptr,total.ptr,routes*(dn/128)
        if(variant==0){
            if(vec)moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up<<<grid,128,0,stream>>>(GU_ARGS);
            else moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up<<<grid,128,0,stream>>>(GU_ARGS);
        }else if(variant==1){
            if(vec)glm_moe_gate_up_m16n128_vecscale<<<grid,128,0,stream>>>(GU_ARGS);
            else glm_moe_gate_up_m16n128<<<grid,128,0,stream>>>(GU_ARGS);
        }else{
            if(vec)glm_moe_gate_up_m16_direct_vecscale<<<grid,128,0,stream>>>(GU_ARGS);
            else glm_moe_gate_up_m16_direct<<<grid,128,0,stream>>>(GU_ARGS);
        }
#undef GU_ARGS
        CHECK(cudaGetLastError());
    };
    cudaGraph_t graph[2];cudaGraphExec_t exec[2];
    for(unsigned vec=0;vec<2;++vec){
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));build();
        for(unsigned variant=0;variant<3;++variant)launch(variant,vec);
        CHECK(cudaStreamEndCapture(stream,&graph[vec]));CHECK(cudaGraphInstantiate(&exec[vec],graph[vec],nullptr,nullptr,0));
    }
    std::printf("device_bytes=%zu cap=%zu max_rows=%u local_pairs=%u N=%u K=%u\n",device_peak,memory_limit,max_rows,weights,dn,dk);
    auto all=cases();
    for(unsigned round=0;round<(timing?3u:2u);++round)for(unsigned ci=0;ci<all.size();++ci){
        auto c=all[ci];fill_ids(c,round+ci);const auto eo=offsets(c);const auto ew=expected_work(c);
        std::vector<int> ids(routes,-1);std::copy(c.ids.begin(),c.ids.end(),ids.begin());
        std::vector<unsigned long long> hgp(experts),hgs(experts),hup(experts),hus(experts);
        std::vector<float> hgf(experts,1),huf(experts,1);
        for(unsigned e=0;e<experts;++e)if(c.local[e]>=0){
            unsigned g=unsigned(c.local[e]),u=g+weights;
            hgp[e]=reinterpret_cast<unsigned long long>(bp.ptr+g*packed_weight);hgs[e]=reinterpret_cast<unsigned long long>(bs.ptr+g*scale_weight);
            hup[e]=reinterpret_cast<unsigned long long>(bp.ptr+u*packed_weight);hus[e]=reinterpret_cast<unsigned long long>(bs.ptr+u*scale_weight);
            hgf[e]=std::ldexp(1.0f,int(g%3)-1);huf[e]=std::ldexp(1.0f,int(u%3)-1);
        }
        std::vector<unsigned char> ha(ap.count),hs(as.count);
        for(size_t i=0;i<ha.size();++i)ha[i]=packed_code(i,101+ci+round);
        unsigned zero_source=0;
        for(unsigned e=0;e<experts;++e)if(c.count[e]&&c.local[e]>=0){zero_source=unsigned(ids[eo[e]]);break;}
        std::fill(ha.begin()+zero_source*(dk/2),ha.begin()+(zero_source+1)*(dk/2),0);
        for(size_t i=0;i<hs.size();++i)hs[i]=scale_code(i,ci+round);
        ap.upload(ha,0,stream);as.upload(hs,0,stream);off.upload(eo,0,stream);gather.upload(ids,0,stream);
        gp.upload(hgp,0,stream);gs.upload(hgs,0,stream);up.upload(hup,0,stream);us.upload(hus,0,stream);gsf.upload(hgf,0,stream);usf.upload(huf,0,stream);
        auto poison=[&](){for(auto* out:{&old_gate,&old_up,&m16_gate,&m16_up,&direct_gate,&direct_up})CHECK(cudaMemsetAsync(out->ptr,0x5a,out->count*2,stream));};
        poison();CHECK(cudaMemsetAsync(work.ptr,0xa5,work.count*4,stream));CHECK(cudaMemsetAsync(total.ptr,0xa5,4,stream));
        if(round)CHECK(cudaGraphLaunch(exec[1],stream));else{build();for(unsigned v=0;v<3;++v)launch(v);}
        CHECK(cudaStreamSynchronize(stream));
        require(total.read()[0]==int(ew.size()/2),"independent work count");auto aw=work.read();
        for(size_t i=0;i<aw.size();++i)require(aw[i]==(i<ew.size()?ew[i]:0xa5a5a5a5u),"worklist entry/tail");
        const auto rg=old_gate.read(),ru=old_up.read();exact(rg,m16_gate.read(),"gate M64/M16 vecscale");exact(ru,m16_up.read(),"up M64/M16 vecscale");
        exact(rg,direct_gate.read(),"gate M64/direct-register vecscale");exact(ru,direct_up.read(),"up M64/direct-register vecscale");
        poison();if(round)CHECK(cudaGraphLaunch(exec[0],stream));else{build();for(unsigned v=0;v<3;++v)launch(v,false);}CHECK(cudaStreamSynchronize(stream));
        exact(rg,old_gate.read(),"gate production scalar");exact(ru,old_up.read(),"up production scalar");
        exact(rg,m16_gate.read(),"gate M16 scalar");exact(ru,m16_up.read(),"up M16 scalar");
        exact(rg,direct_gate.read(),"gate direct-register scalar");exact(ru,direct_up.read(),"up direct-register scalar");
        for(unsigned row=0;row<routes;++row){
            unsigned e=0;while(e<experts&&eo[e+1]<=int(row))++e;bool local=e<experts&&c.local[e]>=0;
            for(unsigned which=0;which<2;++which){const auto& out=which?ru:rg;
                for(unsigned col=0;col<dn;++col){unsigned short bits;std::memcpy(&bits,&out[size_t(row)*dn+col],2);
                    if(!local)require(bits==0x5a5a,"remote/unused output");else{
                        require(std::isfinite(float(out[size_t(row)*dn+col]))&&bits!=0x5a5a,"nonfinite/unwritten output");
                        if(unsigned(ids[row])==zero_source)require((bits&0x7fff)==0,"complete consumed zero row");}}
                if(!local)continue;unsigned w=unsigned(c.local[e])+which*weights,src=unsigned(ids[row]);
                for(unsigned col:{0u,1u,31u,127u,128u,1023u,2047u}){double sum=0;
                    for(unsigned k=0;k<dk;++k){unsigned a=ha[size_t(src)*(dk/2)+k/2],b=packed_code(size_t(k/2)*dn+col,w+31);
                        sum+=fp4((a>>(4*(k%2)))&15)*scale(hs[size_t(src)*(dk/16)+k/16])
                            *fp4((b>>(4*(k%2)))&15)*scale(scale_code(size_t(k/16)*dn+col,w));}
                    unsigned short bits;std::memcpy(&bits,&out[size_t(row)*dn+col],2);
                    require(bits==bf16_bits(sum*(which?huf[e]:hgf[e])),"CPU gathered projection oracle");}
            }
        }
        auto check_immutable=[&](){
            exact(ha,ap.read(),"A unchanged");exact(hs,as.read(),"A scales unchanged");exact(eo,off.read(),"offsets unchanged");exact(ids,gather.read(),"gather unchanged");
            exact(hgp,gp.read(),"gate ptrs");exact(hgs,gs.read(),"gate scales ptrs");exact(hup,up.read(),"up ptrs");exact(hus,us.read(),"up scales ptrs");
            exact(hgf,gsf.read(),"gate scale2");exact(huf,usf.read(),"up scale2");
            require(total.read()[0]==int(ew.size()/2),"work count unchanged");exact(aw,work.read(),"worklist/tail unchanged");
        };
        check_immutable();check_guards();
        std::printf("PASS case=%s mode=%s M64_M16_direct_register_ALL_OUTPUTS_BITEXACT both_scales CPU_gather_columns consumed_zero guards\n",c.name,round?"graph":"eager");
        if(timing&&round==2){
            auto run=[&](unsigned v){if(v>=3)build();launch(v%3);};
            for(unsigned i=0;i<5;++i)for(unsigned v=0;v<6;++v)run(v);
            cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float> samples[6];
            for(unsigned trial=0;trial<5;++trial)for(unsigned order=0;order<6;++order){unsigned v=(trial+order)%6;
                CHECK(cudaEventRecord(a,stream));for(unsigned i=0;i<100;++i)run(v);CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));
                float ms;CHECK(cudaEventElapsedTime(&ms,a,b));samples[v].push_back(ms*10);}
            const char* names[]={"M64_fused","M16_fused","direct_register_M16","builder_M64","builder_M16","builder_direct_register_M16"};
            for(unsigned v=0;v<6;++v){std::sort(samples[v].begin(),samples[v].end());std::printf("TIMING case=%s rows=%u path=%s us=%.3f eager_events median5x100 interleaved\n",c.name,max_rows,names[v],samples[v][2]);}
            CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
            exact(rg,old_gate.read(),"post-timing M64 gate");exact(ru,old_up.read(),"post-timing M64 up");
            exact(rg,m16_gate.read(),"post-timing M16 gate");exact(ru,m16_up.read(),"post-timing M16 up");
            exact(rg,direct_gate.read(),"post-timing direct gate");exact(ru,direct_up.read(),"post-timing direct up");
            check_immutable();check_weights();check_guards();
        }
    }
    check_weights();check_guards();
    for(unsigned vec=0;vec<2;++vec){CHECK(cudaGraphExecDestroy(exec[vec]));CHECK(cudaGraphDestroy(graph[vec]));}CHECK(cudaStreamDestroy(stream));
    std::printf("PASS fused gate/up complete device_peak=%zu no_promotion\n",device_peak);
#endif
}
