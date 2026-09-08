// SPDX-License-Identifier: AGPL-3.0-only
// Standalone B-tile experiment; see glm_moe_btile_m64_plan.md.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "glm_moe_btile_m64_bounds.h"
#ifndef ATLAS_MOE_DOWN_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../kernels/gb10/common/moe_permute.cu"
#include "glm_moe_btile_m64.cuh"
#endif

#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
constexpr unsigned max_rows = glm_btile_m64_max_rows;
constexpr unsigned dn = 2048, dk = 4096, experts = 288, routes = max_rows + 64, weights = 2;
// Gathered A has 1024 true token rows. No-gather A is an explicitly expanded
// route-major tensor and therefore needs all 1088 rows, including remote rows.
constexpr unsigned source_capacity = routes;
constexpr unsigned max_tiles = weights * ((max_rows + 63) / 64) * (dn / 128);
constexpr size_t packed_weight = size_t(dn) * dk / 2, scale_weight = size_t(dn) * dk / 16;
constexpr size_t memory_limit = 64ULL * 1024 * 1024;
static size_t device_live = 0, device_peak = 0;
static bool allocation_fits(size_t count, size_t element, size_t live) {
    return element && count <= (memory_limit - 256) / element
        && live <= memory_limit - (count * element + 256);
}
static size_t fixture_budget(){
    // Mirror explicit allocations including their individual 256-byte guards.
    const size_t sizes[]={2*weights*packed_weight,2*weights*packed_weight,
        2*weights*scale_weight,size_t(source_capacity)*dk/2,size_t(source_capacity)*dk/16,
        size_t(routes)*dn*2,size_t(routes)*dn*2,size_t(routes)*dn*2,size_t(routes)*dn*2,
        experts*8,experts*8,experts*8,experts*8,experts*8,experts*8,
        experts*4,experts*4,(experts+1)*4,routes*4,4,max_tiles*2*4};
    size_t live=0;for(size_t bytes:sizes){require(allocation_fits(bytes,1,live),"fixture memory budget");live+=bytes+256;}
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
    const unsigned counts[]={15,16,17,63,64,65,127,128,129,130,148,255,256,257,1023,1024,1,2,3,4,5};
    const char* names[]={"M15","M16","M17","M63","M64","M65","M127","M128","M129", "M130","M148","M255","M256","M257","M1023","M1024","M1","M2","M3","M4","M5"};
    std::vector<Case> all;
    for(unsigned c=0;c<21;++c){
        Case value(names[c]); unsigned ids[]={0,17,142,143,144,145,286,287};
        for(unsigned i=0;i<8;++i)value.count[ids[i]]=int(c<9||c>=16?counts[c]:8);
        if(c>=9&&c<16){value.count[17]=int(counts[c]);value.count[287]=16;}
        // Nonzero starts and pointer swaps exercise both packed M and N fields.
        value.local[17]=int(c%2); value.local[287]=int(1-c%2);
        all.push_back(value);
    }
    Case empty("empty"),remote("remote-only");
    remote.count[0]=int(max_rows);remote.count[287]=64;
    all.push_back(empty);all.push_back(remote);
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
        for(unsigned mt=0;mt<(unsigned(c.count[e])+63)/64;++mt)
            for(unsigned n=0;n<dn/128;++n){out.push_back(e);out.push_back((mt<<6)|n);}
    require(out.size()<=max_tiles*2,"work bound");return out;
}

static bool valid_b_shape(unsigned n,unsigned k,size_t bytes){
    return n==dn&&k==dk&&bytes==packed_weight;
}
// Byte-exact [K/2,N] -> [N/128,K/64,128,32], with no nibble conversion.
static size_t tiled_offset(unsigned n,unsigned kp){
    return (((size_t(n/128)*(dk/64)+kp/32)*128+n%128)*32)+kp%32;
}
static size_t original_offset(size_t tiled){
    size_t tile=tiled/4096,inner=tiled%4096;
    return ((tile%64)*32+inner%32)*dn+(tile/64)*128+inner/32;
}
static std::vector<unsigned char> pack_b(const std::vector<unsigned char>& original){
    require(valid_b_shape(dn,dk,original.size()),"pack shape/extent");
    std::vector<unsigned char> tiled(original.size());size_t dst=0;
    for(unsigned nt=0;nt<dn/128;++nt)for(unsigned kt=0;kt<dk/64;++kt)
        for(unsigned n=0;n<128;++n)for(unsigned b=0;b<32;++b)
            tiled[dst++]=original[size_t(kt*32+b)*dn+nt*128+n];
    return tiled;
}
static void packing_tests(){
    require(valid_b_shape(dn,dk,packed_weight),"valid B shape");
    require(!valid_b_shape(4096,2048,packed_weight)&&!valid_b_shape(dn,dk,packed_weight-1),"reject B shape/extent");
    require(tiled_offset(0,0)==0&&tiled_offset(1,0)==32&&tiled_offset(127,31)==4095,"B tile row mapping");
    require(tiled_offset(0,32)==4096&&tiled_offset(128,0)==64*4096,"B K/N tile mapping");
    std::vector<unsigned char> seen(packed_weight),original(packed_weight),roundtrip(packed_weight);
    for(unsigned kp=0;kp<dk/2;++kp)for(unsigned n=0;n<dn;++n){
        size_t src=size_t(kp)*dn+n,dst=tiled_offset(n,kp);
        require(dst<packed_weight&&!seen[dst],"B packing bijection");seen[dst]=1;
        require(original_offset(dst)==src,"independent inverse mapping");
        original[src]=packed_code(src,701);
    }
    auto tiled=pack_b(original);
    for(size_t dst=0;dst<tiled.size();++dst){
        require(seen[dst]==1&&tiled[dst]==original[original_offset(dst)],"packed byte/nibble preserved");
        roundtrip[original_offset(dst)]=tiled[dst];
    }
    exact(original,roundtrip,"full B pack/unpack");
}

static void host_tests(){
    packing_tests();
    auto all=cases();for(const auto& c:all)require(valid(c),"valid gathered fixture");
    require(expected_work(all[0]).size()==weights*16*2,"gate/up16 Ntiles");
    require(expected_work(all[21]).empty()&&expected_work(all[22]).empty(),"remote/empty skips");
    auto c=all[0];c.ids[0]=int(max_rows);require(!valid(c),"reject out-of-range gather");
    c=all[0];c.ids[1]=c.ids[0];require(!valid(c),"reject duplicate within expert");
    c=all[0];c.local[287]=c.local[17];require(!valid(c),"reject live pair alias");
    c=all[0];c.count[0]=int(max_rows)+1;require(!valid(c),"reject M overflow");
    c=all[0];c.ids.pop_back();require(!valid(c),"reject truncated gather");
    for(unsigned i=0;i<21;++i){
        unsigned m=unsigned(all[i].count[17]);auto work=expected_work(all[i]);
        unsigned other=unsigned(all[i].count[287]);
        require(work.size()==((m+63)/64+(other+63)/64)*16*2,"multiple M tile count");
        for(size_t j=0;j<work.size();j+=2)
            require(glm_btile_m64_work_valid(all[i].count[work[j]],work[j+1]>>6,work[j+1]&63),"packed M/N work fields");
        const auto eo=offsets(all[i]);std::vector<unsigned> writes(routes);
        for(size_t j=0;j<work.size();j+=2){
            if((work[j+1]&63)!=0)continue; // one N tile proves row partition
            unsigned e=work[j],mt=work[j+1]>>6;
            for(unsigned warp=0;warp<4;++warp)for(unsigned group=0;group<8;++group)
                for(unsigned half=0;half<2;++half){
                    unsigned local=mt*64+warp*16+group+half*8;
                    if(local<unsigned(all[i].count[e]))++writes[eo[e]+local];
                }
        }
        for(unsigned row=0;row<routes;++row){
            unsigned e=0;while(e<experts&&eo[e+1]<=int(row))++e;
            require(writes[row]==unsigned(e<experts&&all[i].local[e]>=0),"M64 tail row ownership");
            if(e<experts){
                require(unsigned(all[i].ids[row])<max_rows,"gather uses real token rows only");
                require(row<source_capacity,"no-gather uses explicit route-major source extent");
            }
        }
    }
    require(expected_work(all[5])[32+1]==64,"M65 second tile encoding");
    require(allocation_fits(1,4,0)&&!allocation_fits(std::numeric_limits<size_t>::max(),4,0),"allocation overflow");
    require(!allocation_fits(1,1,memory_limit)&&!allocation_fits(1,1,std::numeric_limits<size_t>::max()),"live allocation cap");
    require(fixture_budget()==56015240,"explicit guarded fixture footprint");
    require(offsets(all[15]).back()==int(routes),"largest concentrated route extent");
    std::printf("PASS host source_rows=%u route_capacity=%u M1_2_3_4_5_15_16_17_63_64_65_127_128_129_130_148_255_256_257_1023_1024 gathered_and_route_major_rowownership exhaustive_B_bijection_roundtrip bounds aliasing device_bytes=%zu\n",max_rows,routes,fixture_budget());
}

int main(int argc,char** argv){
    const bool host=argc==2&&!std::strcmp(argv[1],"--host-test");
    const bool timing=argc==2&&!std::strcmp(argv[1],"--timing");
    require(argc==1||host||timing,"usage: bench-glm-moe-btile-m64 [--host-test|--timing]");
    host_tests();if(host)return 0;
#ifdef ATLAS_MOE_DOWN_HOST_ONLY
    require(false,"CPU-only build supports --host-test only");
#else
    Buffer<unsigned char> bp(2*weights*packed_weight),bt(bp.count),bs(2*weights*scale_weight);
    Buffer<unsigned char> ap(size_t(source_capacity)*dk/2),as(size_t(source_capacity)*dk/16);
    Buffer<__nv_bfloat16> old_gate(size_t(routes)*dn),old_up(old_gate.count),tile_gate(old_gate.count),tile_up(old_gate.count);
    Buffer<unsigned long long> gp(experts),gs(experts),up(experts),us(experts),tgp(experts),tup(experts);
    Buffer<float> gsf(experts),usf(experts);
    Buffer<int> off(experts+1),gather(routes),total(1);
    Buffer<unsigned> work(max_tiles*2);
    require(device_live==fixture_budget(),"actual allocation accounting matches CPU budget");
    auto check_guards=[&](){
        bp.guards();bt.guards();bs.guards();ap.guards();as.guards();old_gate.guards();old_up.guards();tile_gate.guards();tile_up.guards();
        tgp.guards();tup.guards();gp.guards();gs.guards();up.guards();us.guards();gsf.guards();usf.guards();off.guards();gather.guards();total.guards();work.guards();
    };
    for(unsigned w=0;w<2*weights;++w){
        std::vector<unsigned char> bytes(packed_weight);
        for(size_t i=0;i<bytes.size();++i)bytes[i]=packed_code(i,w+31);bp.upload(bytes,w*packed_weight);bt.upload(pack_b(bytes),w*packed_weight);
        bytes.resize(scale_weight);for(size_t i=0;i<bytes.size();++i)bytes[i]=scale_code(i,w);bs.upload(bytes,w*scale_weight);
    }
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto build=[&](){moe_build_tile_worklist<<<1,256,0,stream>>>(off.ptr,gp.ptr,work.ptr,total.ptr,experts,dn/128,64);CHECK(cudaGetLastError());};
    auto launch=[&](unsigned variant,bool vec,unsigned abi,bool gathered){
        require(variant<2&&abi<3,"kernel variant/ABI");
        const dim3 grid(max_tiles,2);
        const int* token_ids=gathered?gather.ptr:nullptr;
        // All exports share the production ABI; only B pointers change for tile.
#define GU_ARGS(GP,UP,GOUT,UOUT) ap.ptr,as.ptr,GP,gs.ptr,gsf.ptr,GOUT,UP,us.ptr,usf.ptr,UOUT,off.ptr,token_ids,experts,dn,dk,work.ptr,total.ptr,max_tiles
        if(abi==0&&variant==0){
            if(vec)moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up<<<grid,128,0,stream>>>(GU_ARGS(gp.ptr,up.ptr,old_gate.ptr,old_up.ptr));
            else moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up<<<grid,128,0,stream>>>(GU_ARGS(gp.ptr,up.ptr,old_gate.ptr,old_up.ptr));
        }else if(abi==0){
            if(vec)glm_moe_gate_up_btile_m64_vecscale<<<grid,128,0,stream>>>(GU_ARGS(tgp.ptr,tup.ptr,tile_gate.ptr,tile_up.ptr));
            else glm_moe_gate_up_btile_m64<<<grid,128,0,stream>>>(GU_ARGS(tgp.ptr,tup.ptr,tile_gate.ptr,tile_up.ptr));
        }
#undef GU_ARGS
        if(abi!=0)for(unsigned projection=0;projection<2;++projection){
            auto* packed=variant?(projection?tup.ptr:tgp.ptr):(projection?up.ptr:gp.ptr);
            auto* scales=projection?us.ptr:gs.ptr;
            auto* scale2=projection?usf.ptr:gsf.ptr;
            auto* output=variant?(projection?tile_up.ptr:tile_gate.ptr):(projection?old_up.ptr:old_gate.ptr);
#define P_ARGS ap.ptr,as.ptr,packed,scales,scale2,output,off.ptr,token_ids,experts,dn,dk
#define C_ARGS P_ARGS,work.ptr,total.ptr,max_tiles
            if(abi==1){
                if(variant==0){
                    if(vec)moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact<<<max_tiles,128,0,stream>>>(C_ARGS);
                    else moe_w4a4_grouped_gemm_prequant_t_k64_compact<<<max_tiles,128,0,stream>>>(C_ARGS);
                }else{
                    if(vec)glm_moe_btile_m64_vecscale_compact<<<max_tiles,128,0,stream>>>(C_ARGS);
                    else glm_moe_btile_m64_compact<<<max_tiles,128,0,stream>>>(C_ARGS);
                }
            }else{
                const dim3 dense(dn/128,(max_rows+63)/64,experts);
                if(variant==0){
                    if(vec)moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dense,128,0,stream>>>(P_ARGS);
                    else moe_w4a4_grouped_gemm_prequant_t_k64<<<dense,128,0,stream>>>(P_ARGS);
                }else{
                    if(vec)glm_moe_btile_m64_vecscale_dense<<<dense,128,0,stream>>>(P_ARGS);
                    else glm_moe_btile_m64_dense<<<dense,128,0,stream>>>(P_ARGS);
                }
            }
#undef P_ARGS
#undef C_ARGS
            CHECK(cudaGetLastError());
        }
        CHECK(cudaGetLastError());
    };
    cudaGraph_t graph[2][3][2];cudaGraphExec_t exec[2][3][2];
    for(unsigned gathered=0;gathered<2;++gathered)for(unsigned abi=0;abi<3;++abi)for(unsigned vec=0;vec<2;++vec){
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));build();launch(0,vec,abi,gathered);launch(1,vec,abi,gathered);
        CHECK(cudaStreamEndCapture(stream,&graph[gathered][abi][vec]));
        CHECK(cudaGraphInstantiate(&exec[gathered][abi][vec],graph[gathered][abi][vec],nullptr,nullptr,0));
    }
    std::printf("device_bytes=%zu cap=%zu max_rows=%u local_pairs=%u N=%u K=%u\n",device_peak,memory_limit,max_rows,weights,dn,dk);
    auto all=cases();
    for(unsigned round=0;round<(timing?3u:2u);++round)for(unsigned ci=0;ci<all.size();++ci)for(unsigned gathered=0;gathered<2;++gathered){
        auto c=all[ci];fill_ids(c,round+ci);const auto eo=offsets(c);const auto ew=expected_work(c);
        const int max_expert_rows=*std::max_element(c.count.begin(),c.count.end());
        std::vector<int> ids(routes,-1);std::copy(c.ids.begin(),c.ids.end(),ids.begin());
        std::vector<unsigned long long> hgp(experts),hgs(experts),hup(experts),hus(experts),htgp(experts),htup(experts);
        std::vector<float> hgf(experts,1),huf(experts,1);
        for(unsigned e=0;e<experts;++e)if(c.local[e]>=0){
            unsigned g=unsigned(c.local[e]),u=g+weights;
            hgp[e]=reinterpret_cast<unsigned long long>(bp.ptr+g*packed_weight);hgs[e]=reinterpret_cast<unsigned long long>(bs.ptr+g*scale_weight);
            hup[e]=reinterpret_cast<unsigned long long>(bp.ptr+u*packed_weight);hus[e]=reinterpret_cast<unsigned long long>(bs.ptr+u*scale_weight);
            htgp[e]=reinterpret_cast<unsigned long long>(bt.ptr+g*packed_weight);htup[e]=reinterpret_cast<unsigned long long>(bt.ptr+u*packed_weight);
            hgf[e]=std::ldexp(1.0f,int(g%3)-1);huf[e]=std::ldexp(1.0f,int(u%3)-1);
        }
        std::vector<unsigned char> ha(ap.count),hs(as.count);
        for(size_t i=0;i<ha.size();++i)ha[i]=packed_code(i,101+ci+round);
        // Guarantee a consumed exact-zero source row whenever local work exists.
        unsigned zero_src=0;
        for(unsigned e=0;e<experts;++e)if(c.count[e]&&c.local[e]>=0){
            zero_src=gathered?unsigned(ids[eo[e]]):unsigned(eo[e]);break;
        }
        std::fill(ha.begin()+size_t(zero_src)*(dk/2),ha.begin()+size_t(zero_src+1)*(dk/2),0);
        for(size_t i=0;i<hs.size();++i)hs[i]=scale_code(i,ci+round);
        if(gathered){
            std::fill(ha.begin()+size_t(max_rows)*dk/2,ha.end(),0xa5);
            std::fill(hs.begin()+size_t(max_rows)*dk/16,hs.end(),0xa5);
        }
        ap.upload(ha);as.upload(hs);off.upload(eo);gather.upload(ids);gp.upload(hgp);gs.upload(hgs);up.upload(hup);us.upload(hus);gsf.upload(hgf);usf.upload(huf);tgp.upload(htgp);tup.upload(htup);
        auto poison=[&](){for(auto* out:{&old_gate,&old_up,&tile_gate,&tile_up})CHECK(cudaMemsetAsync(out->ptr,0x5a,out->count*2,stream));};
        poison();CHECK(cudaMemsetAsync(work.ptr,0xa5,work.count*4,stream));CHECK(cudaMemsetAsync(total.ptr,0xa5,4,stream));
        if(round)CHECK(cudaGraphLaunch(exec[gathered][0][1],stream));else{build();launch(0,true,0,gathered);launch(1,true,0,gathered);}
        CHECK(cudaStreamSynchronize(stream));
        require(total.read()[0]==int(ew.size()/2),"independent work count");auto aw=work.read();
        for(size_t i=0;i<aw.size();++i)require(aw[i]==(i<ew.size()?ew[i]:0xa5a5a5a5u),"worklist entry/tail");
        const auto rg=old_gate.read(),ru=old_up.read();
        exact(rg,tile_gate.read(),"gate M64/Btile vecscale");exact(ru,tile_up.read(),"up M64/Btile vecscale");
        for(unsigned abi=0;abi<3;++abi)for(unsigned vec=0;vec<2;++vec){
            poison();
            if(round)CHECK(cudaGraphLaunch(exec[gathered][abi][vec],stream));
            else{build();launch(0,vec,abi,gathered);launch(1,vec,abi,gathered);}
            CHECK(cudaStreamSynchronize(stream));
            exact(rg,old_gate.read(),"gate production ABI/scales");exact(ru,old_up.read(),"up production ABI/scales");
            exact(rg,tile_gate.read(),"gate Btile ABI/scales");exact(ru,tile_up.read(),"up Btile ABI/scales");
        }
        for(unsigned row=0;row<routes;++row){
            unsigned e=0;while(e<experts&&eo[e+1]<=int(row))++e;bool local=e<experts&&c.local[e]>=0;
            for(unsigned which=0;which<2;++which){const auto& out=which?ru:rg;
                for(unsigned col=0;col<dn;++col){unsigned short bits;std::memcpy(&bits,&out[size_t(row)*dn+col],2);
                    if(!local)require(bits==0x5a5a,"remote/unused output");else{
                        require(std::isfinite(float(out[size_t(row)*dn+col]))&&bits!=0x5a5a,"nonfinite/unwritten local output");
                        if((gathered?unsigned(ids[row]):row)==zero_src)require((bits&0x7fff)==0,"complete consumed zero source row");
                    }}
                if(!local)continue;unsigned w=unsigned(c.local[e])+which*weights,src=gathered?unsigned(ids[row]):row;
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
            exact(htgp,tgp.read(),"tile gate ptrs");exact(htup,tup.read(),"tile up ptrs");
            exact(hgf,gsf.read(),"gate scale2");exact(huf,usf.read(),"up scale2");
            require(total.read()[0]==int(ew.size()/2),"work count unchanged");
            exact(aw,work.read(),"worklist and tail unchanged");
        };
        check_immutable();
        check_guards();
        std::printf("PASS case=%s mode=%s gathered=%u dense_separate_compact_fused_both_scales M64_BtileM64_both_outputs_BITEXACT CPU_columns zero_source guards\n",c.name,round?"graph":"eager",gathered);
        if(timing&&round==2){
            auto run=[&](unsigned v){if(v>=2)build();launch(v%2,true,0,gathered);};
            for(unsigned i=0;i<5;++i)for(unsigned v=0;v<4;++v)run(v);
            cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float> samples[4];
            for(unsigned trial=0;trial<5;++trial)for(unsigned order=0;order<4;++order){unsigned v=(trial+order)%4;
                CHECK(cudaEventRecord(a,stream));for(unsigned i=0;i<100;++i)run(v);CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));
                float ms;CHECK(cudaEventElapsedTime(&ms,a,b));samples[v].push_back(ms*10);}
            const char* names[]={"M64_fused","Btile_M64_fused","builder_M64","builder_Btile_M64"};
            for(unsigned v=0;v<4;++v){std::sort(samples[v].begin(),samples[v].end());std::printf("TIMING case=%s gathered=%u source_rows=%u max_expert_rows=%d path=%s us=%.3f eager_events median5x100 interleaved\n",c.name,gathered,gathered?max_rows:source_capacity,max_expert_rows,names[v],samples[v][2]);}
            CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
            exact(rg,old_gate.read(),"post-timing production gate");exact(ru,old_up.read(),"post-timing production up");
            exact(rg,tile_gate.read(),"post-timing Btile gate");exact(ru,tile_up.read(),"post-timing Btile up");
            check_immutable();
            check_guards();
        }
    }
    for(unsigned w=0;w<2*weights;++w){auto bytes=bp.read(w*packed_weight,packed_weight);for(size_t i=0;i<bytes.size();++i)require(bytes[i]==packed_code(i,w+31),"weight changed");
        bytes=bt.read(w*packed_weight,packed_weight);for(size_t i=0;i<bytes.size();++i)require(bytes[i]==packed_code(original_offset(i),w+31),"tiled weight changed");
        bytes=bs.read(w*scale_weight,scale_weight);for(size_t i=0;i<bytes.size();++i)require(bytes[i]==scale_code(i,w),"weight scale changed");}
    check_guards();
    for(unsigned gathered=0;gathered<2;++gathered)for(unsigned abi=0;abi<3;++abi)for(unsigned vec=0;vec<2;++vec){
        CHECK(cudaGraphExecDestroy(exec[gathered][abi][vec]));CHECK(cudaGraphDestroy(graph[gathered][abi][vec]));
    }
    CHECK(cudaStreamDestroy(stream));
    std::printf("PASS Btile M64 full fixture complete device_peak=%zu no_promotion\n",device_peak);
#endif
}
