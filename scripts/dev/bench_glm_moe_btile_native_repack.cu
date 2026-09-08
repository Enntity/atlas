// SPDX-License-Identifier: AGPL-3.0-only
// Standalone byte-only prerequisite; see glm_moe_btile_native_repack_plan.md.
#include "glm_moe_btile_native_layout.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef ATLAS_MOE_DOWN_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/common/transpose_u8.cu"
#undef TILE
#include "glm_moe_btile_native_repack.cuh"
#endif
using namespace glm_native_btile;

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(2); }
}
constexpr size_t chunk_bytes = 65536;
constexpr size_t host_limit = 32ULL << 20, device_limit = 64ULL << 20;
constexpr size_t host_budget = 4*packed_bytes + 3*scale_bytes + chunk_bytes;
constexpr size_t device_budget = 3*packed_bytes + 2*scale_bytes + 5*256;
static size_t host_live=0, host_peak=0;
static bool fits(size_t bytes,size_t live,size_t limit) {
    return bytes<=limit && live<=limit-bytes;
}
struct HostBuffer {
    std::vector<unsigned char> data;
    explicit HostBuffer(size_t bytes) {
        require(fits(bytes,host_live,host_limit),"host tensor-buffer budget");
        data.resize(bytes);host_live+=bytes;host_peak=std::max(host_peak,host_live);
    }
    ~HostBuffer(){host_live-=data.size();}
    HostBuffer(const HostBuffer&)=delete;
    HostBuffer& operator=(const HostBuffer&)=delete;
};
struct HostFixture {
    HostBuffer native{packed_bytes}, scales{scale_bytes}, transposed{packed_bytes},
        transposed_scales{scale_bytes}, tiled{packed_bytes}, seen{packed_bytes},
        seen_scales{scale_bytes}, readback{chunk_bytes};
};
static unsigned char pattern(size_t index,unsigned seed) {
    if(seed==0)return static_cast<unsigned char>(index);
    uint32_t x=uint32_t(index)^(seed*0x9e3779b9u);
    x^=x>>16;x*=0x7feb352du;x^=x>>15;
    return static_cast<unsigned char>(x^(x>>8));
}
static void prepare_and_check_cpu(HostFixture& h,unsigned seed) {
    for(size_t i=0;i<packed_bytes;++i)h.native.data[i]=pattern(i,seed);
    for(size_t i=0;i<scale_bytes;++i)h.scales.data[i]=pattern(i,seed);
    std::fill(h.transposed.data.begin(),h.transposed.data.end(),0xa5);
    std::fill(h.transposed_scales.data.begin(),h.transposed_scales.data.end(),0xa5);
    std::fill(h.tiled.data.begin(),h.tiled.data.end(),0xa5);
    std::fill(h.seen.data.begin(),h.seen.data.end(),0);
    std::fill(h.seen_scales.data.begin(),h.seen_scales.data.end(),0);
    // Independent native -> T transpose, exactly the existing production byte contract.
    for(unsigned row=0;row<n;++row) {
        for(unsigned col=0;col<k/2;++col)
            h.transposed.data[size_t(col)*n+row]=h.native.data[size_t(row)*(k/2)+col];
        for(unsigned col=0;col<k/group;++col) {
            size_t dst=size_t(col)*n+row,src=size_t(row)*(k/group)+col;
            require(dst<scale_bytes&&!h.seen_scales.data[dst],"scale transpose bijection");
            h.seen_scales.data[dst]=1;h.transposed_scales.data[dst]=h.scales.data[src];
            require((dst%n)*(k/group)+dst/n==src,"scale transpose inverse");
        }
    }
    // Independent existing T -> tile traversal. Do not use the new native helper.
    size_t dst=0;
    for(unsigned nt=0;nt<n/128;++nt)for(unsigned kt=0;kt<k/64;++kt)
        for(unsigned row=0;row<128;++row)for(unsigned byte=0;byte<32;++byte)
            h.tiled.data[dst++]=h.transposed.data[size_t(kt*32+byte)*n+nt*128+row];
    require(dst==packed_bytes,"complete T-to-tile reference");
    for(size_t out=0;out<packed_bytes;++out) {
        size_t src=native_source_for_tile(out);
        require(src<packed_bytes&&!h.seen.data[src],"native packed permutation bijection");
        h.seen.data[src]=1;
        require(tile_for_native(src)==out,"native packed permutation inverse");
        require(h.native.data[src]==h.tiled.data[out],"native vs independent native-T-tile byte oracle");
    }
    for(size_t src=0;src<packed_bytes;++src)
        require(h.seen.data[src]==1&&native_source_for_tile(tile_for_native(src))==src,"full packed inverse roundtrip");
    for(size_t out=0;out<scale_bytes;++out)
        require(h.seen_scales.data[out]==1&&h.transposed_scales.data[out]==h.scales.data[(out%n)*(k/group)+out/n],"full scale byte oracle");
}
static void boundary_tests() {
    Projection p{{0x10000000,packed_bytes},{0x20000000,scale_bytes},0x3f400000};
    Span scratch{0x30000000,packed_bytes};
    require(valid_repack(n,k,group,p,scratch),"valid exact native projection");
    require(!valid_repack(k,n,group,p,scratch)&&!valid_repack(n,k,32,p,scratch),"shape/group refusal");
    for(unsigned fault=0;fault<12;++fault) {
        auto bad=p;auto temp=scratch;
        switch(fault) {
            case 0:bad.packed.address=0;break;
            case 1:bad.packed.address++;break;
            case 2:bad.packed.bytes--;break;
            case 3:bad.packed.bytes++;break;
            case 4:bad.scales.address=0;break;
            case 5:bad.scales.bytes--;break;
            case 6:bad.scales.address=bad.packed.address;break;
            case 7:temp.address=bad.packed.address+16;break;
            case 8:temp.bytes--;break;
            case 9:temp.address=std::numeric_limits<uint64_t>::max()-15;break;
            case 10:bad.scalar_bits=0x7fc00000;break;
            case 11:bad.scales.address++;break;
        }
        require(!valid_repack(n,k,group,bad,temp),"invalid span/scalar refused before CUDA");
    }
    require(host_budget==18415616&&device_budget==13632768,"explicit buffer budgets");
    require(!fits(std::numeric_limits<size_t>::max(),0,host_limit),"budget overflow refusal");
    require(!fits(1,host_limit,host_limit)&&!fits(1,std::numeric_limits<size_t>::max(),device_limit),"live budget refusal");
}

#ifndef ATLAS_MOE_DOWN_HOST_ONLY
#define CHECK(call) do { const auto status=(call); if(status!=cudaSuccess) { \
    std::fprintf(stderr,"CUDA failure %s:%d: %s; aborting fixture\n",__FILE__,__LINE__,cudaGetErrorString(status)); \
    std::exit(1); } } while(0)
static size_t device_live=0,device_peak=0,allocations=0;
struct DeviceBuffer {
    unsigned char* allocation=nullptr;
    unsigned char* ptr=nullptr;
    size_t bytes;
    explicit DeviceBuffer(size_t count):bytes(count) {
        require(count>0&&count<=device_limit-256&&fits(count+256,device_live,device_limit),"guarded device budget");
        CHECK(cudaMalloc(&allocation,count+256));ptr=allocation+128;
        device_live+=count+256;device_peak=std::max(device_peak,device_live);++allocations;
        CHECK(cudaMemset(allocation,0xa5,count+256));
    }
    ~DeviceBuffer(){CHECK(cudaFree(allocation));device_live-=bytes+256;--allocations;}
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    Span span()const{return {reinterpret_cast<uint64_t>(ptr),bytes};}
    void upload(const HostBuffer& input,cudaStream_t stream){
        require(input.data.size()==bytes,"upload exact extent");
        // HostFixture owns these unchanged bytes through the repack's stream
        // synchronization. Explicit same-stream order avoids default-stream
        // assumptions about pageable H2D staging versus DMA completion.
        CHECK(cudaMemcpyAsync(ptr,input.data.data(),bytes,cudaMemcpyHostToDevice,stream));
    }
    void guards()const{
        unsigned char guard[256];
        CHECK(cudaMemcpy(guard,allocation,128,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(guard+128,ptr+bytes,128,cudaMemcpyDeviceToHost));
        for(auto value:guard)require(value==0xa5,"GPU allocation guard");
    }
};

static void repack(Projection& projection,Span scratch,unsigned char poison,cudaStream_t stream) {
    // Every boundary check precedes the first mutation. A CUDA error exits the
    // entire fixture; never continue with a partially transformed projection.
    require(valid_repack(n,k,group,projection,scratch),"native repack boundary before CUDA");
    auto* packed=reinterpret_cast<unsigned char*>(projection.packed.address);
    auto* scales=reinterpret_cast<unsigned char*>(projection.scales.address);
    auto* temporary=reinterpret_cast<unsigned char*>(scratch.address);
    CHECK(cudaMemcpyAsync(temporary,packed,packed_bytes,cudaMemcpyDeviceToDevice,stream));
    // Diagnostic-only poisoning AFTER preserving the source. Run complementary
    // poisons: byte data has no reserved NaN sentinel, so one poison can equal
    // an expected byte. A missing write cannot match both complementary runs.
    CHECK(cudaMemsetAsync(packed,poison,packed_bytes,stream));
    glm_native_to_btile_u8<<<(packed_bytes+255)/256,256,0,stream>>>(temporary,packed,n,k);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream)); // scratch cannot be reused before completion
    CHECK(cudaMemcpyAsync(temporary,scales,scale_bytes,cudaMemcpyDeviceToDevice,stream));
    CHECK(cudaMemsetAsync(scales,poison,scale_bytes,stream));
    // Actual production transpose_u8: [N,K/16] -> [K/16,N], into the same owner.
    transpose_u8<<<dim3((k/group+31)/32,(n+31)/32),dim3(32,8),0,stream>>>(temporary,scales,n,k/group);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));
}

template<class Expected>
static void compare_all(const DeviceBuffer& device,HostBuffer& chunk,Expected expected,const char* label) {
    for(size_t offset=0;offset<device.bytes;offset+=chunk.data.size()) {
        const size_t length=std::min(chunk.data.size(),device.bytes-offset);
        CHECK(cudaMemcpy(chunk.data.data(),device.ptr+offset,length,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<length;++i)if(chunk.data[i]!=expected(offset+i)) {
            std::fprintf(stderr,"FAIL: %s byte=%zu actual=%02x expected=%02x\n",label,offset+i,unsigned(chunk.data[i]),unsigned(expected(offset+i)));
            std::exit(2);
        }
    }
}

static void native_tests(HostFixture& h) {
    DeviceBuffer packed(packed_bytes),scales(scale_bytes),scratch(packed_bytes),
        reference_packed(packed_bytes),reference_scales(scale_bytes);
    // Constructors initialize guards on the default stream. Complete that
    // initialization before any nonblocking-stream upload can overwrite data.
    CHECK(cudaDeviceSynchronize());
    require(device_live==device_budget&&allocations==5,"actual guarded device accounting");
    const std::array<Span,5> spans={packed.span(),scales.span(),scratch.span(),reference_packed.span(),reference_scales.span()};
    for(size_t i=0;i<spans.size();++i) {
        require(valid_span(spans[i],spans[i].bytes),"all allocation spans valid");
        for(size_t j=0;j<i;++j)require(disjoint(spans[i],spans[j]),"immutable reference/work buffers disjoint");
    }
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(unsigned seed=0;seed<3;++seed)for(unsigned char poison:{0xa5,0x5a}) {
        prepare_and_check_cpu(h,seed);
        packed.upload(h.native,stream);scales.upload(h.scales,stream);
        reference_packed.upload(h.native,stream);reference_scales.upload(h.scales,stream);
        Projection projection{packed.span(),scales.span(),0x3f400000u};
        const auto before=projection;const size_t before_live=device_live,before_allocations=allocations;
        repack(projection,scratch.span(),poison,stream);
        require(projection.packed.address==before.packed.address&&projection.packed.bytes==before.packed.bytes
            &&projection.scales.address==before.scales.address&&projection.scales.bytes==before.scales.bytes
            &&projection.scalar_bits==before.scalar_bits,"owner addresses/extents/scalar bits unchanged");
        require(device_live==before_live&&allocations==before_allocations,"no allocation inside repack");
        compare_all(packed,h.readback,[&](size_t i){return h.tiled.data[i];},"full native-to-T-to-tile reference");
        compare_all(packed,h.readback,[&](size_t i){return h.native.data[native_source_for_tile(i)];},"full direct native reference");
        compare_all(scales,h.readback,[&](size_t i){return h.transposed_scales.data[i];},"full scale transpose reference");
        compare_all(reference_packed,h.readback,[&](size_t i){return h.native.data[i];},"immutable native packed reference");
        compare_all(reference_scales,h.readback,[&](size_t i){return h.scales.data[i];},"immutable native scale reference");
        compare_all(scratch,h.readback,[&](size_t i){return i<scale_bytes?h.scales.data[i]:h.native.data[i];},"scratch scale prefix and untouched packed tail");
        for(const auto* buffer:{&packed,&scales,&scratch,&reference_packed,&reference_scales})buffer->guards();
        std::printf("PASS native_repack pattern=%u poison=%02x ALL_PACKED_SCALE_BYTES independent_T_tile inverse scratch_reuse immutable_references fixed_owners_scalar guards\n",seed,unsigned(poison));
    }
    CHECK(cudaStreamDestroy(stream));
    std::printf("PASS native_repack device_bytes=%zu host_payload=%zu allocations=%zu NO_TIMING_NO_MODEL_REPACK\n",device_peak,host_peak,allocations);
}
#endif

int main(int argc,char**argv) {
    const bool host=argc==2&&!std::strcmp(argv[1],"--host-test");
    require(argc==1||host,"usage: bench-glm-moe-btile-native-repack [--host-test]");
    boundary_tests();HostFixture h;
    require(host_live==host_budget,"actual host payload budget");
    for(unsigned seed=0;seed<3;++seed)prepare_and_check_cpu(h,seed);
    std::printf("PASS CPU native_packed_bijection_inverse_ALL_BYTES native_T_tile_equivalence scale_transpose_ALL_BYTES patterns=3 host_payload=%zu planned_device=%zu\n",host_peak,device_budget);
    if(host)return 0;
#ifdef ATLAS_MOE_DOWN_HOST_ONLY
    require(false,"CPU-only build supports --host-test only");
#else
    native_tests(h);
    require(device_live==0&&allocations==0,"all fixture device owners freed");
#endif
}
