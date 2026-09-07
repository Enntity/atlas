// SPDX-License-Identifier: AGPL-3.0-only
// Standalone only: see glm_shared_fp8_plan.md. No production dispatch.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#ifndef ATLAS_SHARED_FP8_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu"
#endif
#include "glm_shared_fp8.cuh"
static void require(bool ok,const char* message) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(2);}
}
constexpr size_t memory_limit=32ULL*1024*1024;
constexpr unsigned max_rows=16;
static size_t device_live=0,device_peak=0;
static bool allocation_fits(size_t n,size_t width,size_t live) {
    return width&&n<=(memory_limit-256)/width&&live<=memory_limit-(n*width+256);
}
struct Shape{const char* name;unsigned n,k,seed;float scale2;};
static const std::array<Shape,3> shapes={{{"gate",2048,4096,31,.75f},
    {"up",2048,4096,73,.375f},{"down",4096,2048,117,.625f}}};
static size_t fixture_budget(const Shape& s){
    const size_t sizes[]={size_t(max_rows)*s.k*2,
        size_t(s.n)*s.k/2,size_t(s.n)*s.k/2,size_t(s.n)*s.k/16,size_t(s.n)*s.k/16,
        size_t(s.n)*s.k,size_t(max_rows)*s.n*2,size_t(max_rows)*s.n*2,size_t(max_rows)*s.n*2};
    size_t live=0;for(size_t bytes:sizes){require(allocation_fits(bytes,1,live),"explicit32MiB fixture budget");live+=bytes+256;}return live;
}
static float from_bf16(unsigned short bits) {
    unsigned wide=unsigned(bits)<<16; float value; std::memcpy(&value,&wide,4); return value;
}
static unsigned short bf16_bits(double value) {
    float f=float(value); unsigned bits; std::memcpy(&bits,&f,4);
    return static_cast<unsigned short>((bits+0x7fff+((bits>>16)&1))>>16);
}
static bool within_one_bf16_ulp(unsigned short a,unsigned short b) {
    const int aa=(a&0x8000)?0x8000-int(a&0x7fff):0x8000+int(a);
    const int bb=(b&0x8000)?0x8000-int(b&0x7fff):0x8000+int(b);
    return std::abs(aa-bb)<=1;
}
static double e4m3(unsigned code) {
    unsigned magnitude=code&127, exponent=magnitude>>3, fraction=magnitude&7;
    if(magnitude==127)return std::numeric_limits<double>::quiet_NaN();
    double value=exponent?std::ldexp(1.0+fraction/8.0,int(exponent)-7):std::ldexp(double(fraction),-9);
    return code&128?-value:value;
}
// Independent exhaustive E4M3 nearest/ties-to-even/satfinite reference.
static unsigned char encode_e4m3(double x) {
    const bool sign=std::signbit(x);double magnitude=std::fabs(x);
    unsigned best=0;double error=std::numeric_limits<double>::infinity();
    for(unsigned c=0;c<127;++c){double d=std::fabs(magnitude-e4m3(c));
        if(d<error||(d==error&&!(c&1))){best=c;error=d;}}
    return static_cast<unsigned char>(best|(sign?128:0));
}
static double round_e4m3(double x) { return e4m3(encode_e4m3(x)); }
static unsigned char packed_code(size_t i,unsigned seed) {
    unsigned x=unsigned(i)^(seed*0x9e3779b9u);x^=x>>16;x*=0x7feb352du;x^=x>>15;
    return static_cast<unsigned char>(x^(x>>8));
}
static unsigned char scale_code(size_t i,unsigned seed) {
    return static_cast<unsigned char>(0x20+((i+seed)%16));
}
static double fp4(unsigned c) {
    const double p[]={0,.5,1,1.5,2,3,4,6};return (c&8?-1:1)*p[c&7];
}
template<class T> static void exact(const std::vector<T>& a,const std::vector<T>& b,const char* label) {
    require(a.size()==b.size(),"comparison extent");
    for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(T))){
        std::fprintf(stderr,"FAIL: %s element=%zu\n",label,i);std::exit(2);}
}

static std::vector<unsigned char> transpose(const std::vector<unsigned char>& original,unsigned rows,unsigned cols){
    require(rows&&cols&&original.size()==size_t(rows)*cols,"transpose extent");
    std::vector<unsigned char> result(original.size());
    for(unsigned row=0;row<rows;++row)for(unsigned col=0;col<cols;++col)
        result[glm_shared_fp8::transpose_index(row,col,rows,cols)]=original[size_t(row)*cols+col];
    return result;
}
static std::vector<unsigned char> expected_fp8(const Shape& s,
    const std::vector<unsigned char>& packed,const std::vector<unsigned char>& scales) {
    require(packed.size()==size_t(s.n)*s.k/2&&scales.size()==size_t(s.n)*s.k/16,"CPU predecode source extents");
    std::array<std::array<unsigned char,16>,256> table{};
    for(unsigned sb=0;sb<256;++sb)if((sb&127)!=127){
        const float sv=float(e4m3(sb))*s.scale2;
        for(unsigned code=0;code<16;++code)table[sb][code]=encode_e4m3(float(fp4(code))*sv);
    }
    std::vector<unsigned char> expected(size_t(s.n)*s.k);
    for(unsigned n=0;n<s.n;++n)for(unsigned kp=0;kp<s.k/2;++kp){
        unsigned sb=scales[size_t(n)*(s.k/16)+kp/8];require((sb&127)!=127,"no NaN scale in oracle");
        unsigned byte=packed[size_t(n)*(s.k/2)+kp];
        expected[size_t(n)*s.k+2*kp]=table[sb][byte&15];
        expected[size_t(n)*s.k+2*kp+1]=table[sb][byte>>4];
    }
    return expected;
}
static void host_tests(){
    require(glm_shared_fp8::transpose_index(1,0,128,32)==1,"original-to-T column stride");
    for(const auto& s:shapes){
        for(unsigned divisor:{2u,16u}){
            unsigned rows=s.n,cols=s.k/divisor;
            std::vector<unsigned char> original(size_t(rows)*cols),seen(original.size());
            for(size_t j=0;j<original.size();++j)original[j]=packed_code(j,s.seed);
            auto t=transpose(original,rows,cols);
            for(unsigned row=0;row<rows;++row)for(unsigned col=0;col<cols;++col){
                size_t dst=glm_shared_fp8::transpose_index(row,col,rows,cols);
                require(dst<t.size()&&!seen[dst],"transpose bijection");seen[dst]=1;
                require(t[size_t(col)*rows+row]==original[size_t(row)*cols+col],"independent transpose oracle");
            }
            exact(original,transpose(t,cols,rows),"full transpose roundtrip");
        }
        require(glm_shared_fp8::geometry(1,s.n,s.k)&&glm_shared_fp8::geometry(16,s.n,s.k),"supported exact shapes");
        require(fixture_budget(s)<memory_limit,"guarded fixture fits");
    }
    for(unsigned m=0;m<=16;++m){std::vector<unsigned> seen(16*128);
        for(unsigned warp=0;warp<4;++warp)for(unsigned lane=0;lane<32;++lane)
            for(unsigned nt=0;nt<4;++nt)for(unsigned half=0;half<2;++half)for(unsigned side=0;side<2;++side){
                unsigned row=lane/4+half*8,col=glm_shared_fp8::output_col(warp,nt,(lane%4)*2+side);
                require(row<16&&col<128,"M16 output bounds");if(row<m)++seen[row*128+col];}
        for(unsigned row=0;row<16;++row)for(unsigned col=0;col<128;++col)
            require(seen[row*128+col]==unsigned(row<m),"M16 full output ownership");
    }
    std::vector<unsigned> a(16*32),b(128*32);
    for(unsigned tid=0;tid<128;++tid){
        if(tid<64)for(unsigned j=0;j<8;++j)++a[(tid/4)*32+(tid%4)*8+j];
        for(unsigned j=0;j<32;++j)++b[tid*32+j];}
    for(auto* values:{&a,&b})for(unsigned n:*values)require(n==1,"M16 FP8 shared copies");
    require(!glm_shared_fp8::geometry(17,2048,4096)&&!glm_shared_fp8::geometry(5,129,4096)
        &&!glm_shared_fp8::geometry(5,2048,4095)&&!glm_shared_fp8::geometry(5,8192,32),"unsupported geometry");
    require(encode_e4m3(1.0625)==0x38&&encode_e4m3(1.1875)==0x3a&&encode_e4m3(-1000)==0xfe,"E4M3 ties saturation");
    require(encode_e4m3(0.0)==0&&encode_e4m3(-0.0)==0x80,"FP8 signed zero preserved");
    require(encode_e4m3(std::ldexp(1.0,-10))==0&&encode_e4m3(3*std::ldexp(1.0,-10))==2,"subnormal tie rounding");
    const Shape tiny={"CPU_decode",2,32,0,.75f};
    std::vector<unsigned char> packed(32,0x80),scales={0x38,0x38,0x7e,0x01};
    packed[0]=0x21;packed[1]=0xf8;std::fill(packed.begin()+16,packed.end(),0x77);
    auto decoded=expected_fp8(tiny,packed,scales);
    require(decoded[0]==0x2c&&decoded[1]==0x34&&decoded[2]==0x80&&decoded[3]==0xc9,"both nibbles scale2 and signed zero");
    for(unsigned k=4;k<32;++k)require(decoded[k]==(k%2?0x80:0),"all zero nibbles preserve sign");
    for(unsigned k=32;k<64;++k)require(decoded[k]==(k<48?0x7e:4),"scale-group boundary saturation and subnormal tie");
    require(within_one_bf16_ulp(0x3f80,0x3f81)&&!within_one_bf16_ulp(0x3f80,0x3f82),"CPU comparison exact1ULP");
    require(!allocation_fits(SIZE_MAX,2,0)&&!allocation_fits(1,2,SIZE_MAX)
        &&!allocation_fits(1,2,memory_limit),"allocation overflow");
    std::printf("PASS CPU packed/scales transpose full_bijection_roundtrip M0to16_ownership copies E4M3_RNE cap32MiB budgets=%zu/%zu/%zu\n",
        fixture_budget(shapes[0]),fixture_budget(shapes[1]),fixture_budget(shapes[2]));
}

#ifndef ATLAS_SHARED_FP8_HOST_ONLY
#define CHECK(call) do { auto error=(call); if(error!=cudaSuccess){ \
    std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(error));std::exit(1);}}while(0)
template<class T> struct Buffer {
    T* allocation; T* ptr; size_t count,bytes;
    explicit Buffer(size_t n):count(n){
        require(allocation_fits(n,sizeof(T),device_live),"32MiB explicit allocation/overflow cap");
        bytes=n*sizeof(T)+256;CHECK(cudaMalloc(&allocation,bytes));
        device_live+=bytes;device_peak=std::max(device_peak,device_live);
        CHECK(cudaMemset(allocation,0xa5,bytes));ptr=allocation+128/sizeof(T);
    }
    ~Buffer(){cudaFree(allocation);device_live-=bytes;}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<T>& h){require(h.size()==count,"upload extent");
        CHECK(cudaMemcpy(ptr,h.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T> h(count);
        CHECK(cudaMemcpy(h.data(),ptr,count*sizeof(T),cudaMemcpyDeviceToHost));return h;}
    void guards()const{unsigned char a[128],b[128];
        CHECK(cudaMemcpy(a,allocation,128,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b,ptr+count,128,cudaMemcpyDeviceToHost));
        for(unsigned j=0;j<128;++j)require(a[j]==0xa5&&b[j]==0xa5,"allocation guard");}
};

static void cpu_columns(const Shape& s,unsigned m,const std::vector<unsigned short>& a,
    const std::vector<unsigned char>& expected,const std::vector<unsigned short>& output) {
    std::vector<double> qa(size_t(m)*s.k);std::array<double,65536> lut;
    lut.fill(std::numeric_limits<double>::quiet_NaN());
    for(size_t i=0;i<qa.size();++i){unsigned bits=a[i];if(std::isnan(lut[bits]))lut[bits]=round_e4m3(from_bf16(a[i]));qa[i]=lut[bits];}
    for(unsigned col:{0u,1u,2u,31u,32u,127u,128u,s.n-1})for(unsigned row=0;row<m;++row){
        double sum=0;for(unsigned k=0;k<s.k;++k)sum+=qa[size_t(row)*s.k+k]*e4m3(expected[size_t(col)*s.k+k]);
        auto want=bf16_bits(sum),got=output[size_t(row)*s.n+col];
        if(!within_one_bf16_ulp(got,want)){std::fprintf(stderr,"FAIL CPU shape=%s M=%u row=%u col=%u got=%g want=%g\n",
            s.name,m,row,col,from_bf16(got),from_bf16(want));std::exit(2);}
    }
}
static void run_shape(const Shape& s,bool timing) {
    require(glm_shared_fp8::geometry(max_rows,s.n,s.k),"bounded projection geometry");
    Buffer<unsigned short> a(size_t(max_rows)*s.k);
    Buffer<unsigned char> bp(size_t(s.n)*s.k/2),bt(bp.count),bs(size_t(s.n)*s.k/16),bst(bs.count),bf8(size_t(s.n)*s.k);
    Buffer<unsigned short> original(size_t(max_rows)*s.n),predecoded(original.count),m16(original.count);
    require(device_live==fixture_budget(s),"actual explicit device accounting");
    auto guards=[&](){a.guards();bp.guards();bt.guards();bs.guards();bst.guards();bf8.guards();
        original.guards();predecoded.guards();m16.guards();};
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(unsigned m:{0u,1u,4u,5u,16u}){
        auto launch=[&](unsigned variant){
            require(variant<3,"projection selection");
            auto* ap=reinterpret_cast<__nv_bfloat16*>(a.ptr);const dim3 grid(s.n/128,1);
            if(variant==0)w4a16_gemm_t<<<grid,128,0,stream>>>(ap,bt.ptr,bst.ptr,s.scale2,
                reinterpret_cast<__nv_bfloat16*>(original.ptr),m,s.n,s.k,s.n);
            else if(variant==1)fp8_gemm_t<<<grid,128,0,stream>>>(ap,bf8.ptr,
                reinterpret_cast<__nv_bfloat16*>(predecoded.ptr),m,s.n,s.k);
            else glm_shared_fp8_m16<<<grid,128,0,stream>>>(ap,bf8.ptr,
                reinterpret_cast<__nv_bfloat16*>(m16.ptr),m,s.n,s.k);
            CHECK(cudaGetLastError());
        };
        cudaGraph_t graph;cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(unsigned v=0;v<3;++v)launch(v);
        CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        std::vector<unsigned short> previous;
        for(unsigned epoch=0;epoch<2;++epoch){
            std::vector<unsigned char> hp(bp.count),hs(bs.count);
            for(size_t i=0;i<hp.size();++i)hp[i]=packed_code(i,s.seed+epoch*19);
            for(size_t i=0;i<hs.size();++i)hs[i]=scale_code(i,s.seed+epoch);
            // Exercise saturation, subnormal scales and signed zero on complete columns.
            std::fill(hs.begin(),hs.begin()+s.k/16,0x7e);
            std::fill(hs.begin()+s.k/16,hs.begin()+2*s.k/16,0x01);
            std::fill(hs.begin()+2*s.k/16,hs.begin()+3*s.k/16,0x00);
            auto ht=transpose(hp,s.n,s.k/2),hst=transpose(hs,s.n,s.k/16);
            auto expected=expected_fp8(s,hp,hs);
            std::vector<unsigned short> ha(a.count,0x7fc1);
            for(unsigned row=0;row<m;++row)for(unsigned k=0;k<s.k;++k){
                unsigned source=epoch?m-1-row:row;int v=int(packed_code(size_t(source)*s.k+k,s.seed+epoch+211))-128;
                ha[size_t(row)*s.k+k]=bf16_bits(double(v)/128.0);}
            if(m>1)std::fill(ha.begin()+(epoch%m)*s.k,ha.begin()+(epoch%m+1)*s.k,0);
            a.upload(ha);bp.upload(hp);bt.upload(ht);bs.upload(hs);bst.upload(hst);
            // Only explicit setup mutates the cached FP8 weights; no predecode in graphs/timing.
            CHECK(cudaMemsetAsync(bf8.ptr,0xff,bf8.count,stream));
            cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
            CHECK(cudaEventRecord(begin,stream));
            predequant_nvfp4_to_fp8<<<unsigned((bp.count+255)/256),256,0,stream>>>(bp.ptr,bs.ptr,s.scale2,bf8.ptr,s.n,s.k);
            CHECK(cudaGetLastError());CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
            float setup_ms;CHECK(cudaEventElapsedTime(&setup_ms,begin,end));
            CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));
            exact(expected,bf8.read(),"every predecoded E4M3 byte vs independent CPU");
            auto poison=[&](){for(auto* out:{&original,&predecoded,&m16})CHECK(cudaMemsetAsync(out->ptr,0x5a,out->count*2,stream));};
            poison();for(unsigned v=0;v<3;++v)launch(v);CHECK(cudaStreamSynchronize(stream));
            const auto reference=original.read();
            auto check_outputs=[&](){
                exact(reference,original.read(),"original projection unchanged");
                exact(reference,predecoded.read(),"existing FP8 full bit equality");
                exact(reference,m16.read(),"M16 FP8 full bit equality");
                for(unsigned row=0;row<max_rows;++row)for(unsigned col=0;col<s.n;++col){
                    unsigned short bits=reference[size_t(row)*s.n+col];
                    if(row>=m)require(bits==0x5a5a,"inactive output tail");
                    else {require(bits!=0x5a5a&&std::isfinite(from_bf16(bits)),"fresh finite output");
                        if((m>1&&row==epoch%m)||col==2)require(from_bf16(bits)==0,"exact zero row/weight column");}}
            };
            auto immutable=[&](){exact(ha,a.read(),"A immutable including NaN padding");
                exact(hp,bp.read(),"original weights immutable");exact(ht,bt.read(),"T weights immutable");
                exact(hs,bs.read(),"original scales immutable");exact(hst,bst.read(),"T scales immutable");
                exact(expected,bf8.read(),"cached FP8 weights immutable");guards();};
            check_outputs();immutable();cpu_columns(s,m,ha,expected,reference);
            poison();CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));check_outputs();immutable();
            if(epoch&&m)require(reference!=previous,"refreshed graph fixture differs");previous=reference;
            std::printf("PASS FP8cache shape=%s M=%u epoch=%u all_predecode_bytes full3way_BITEXACT CPUcolumns graphrefresh immutable tails setup_us=%.3f\n",
                s.name,m,epoch,setup_ms*1000);
            if(timing&&epoch==1&&(m==4||m==5)){
                for(unsigned i=0;i<5;++i)for(unsigned v=0;v<3;++v)launch(v);
                cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));std::vector<float> samples[3];
                for(unsigned trial=0;trial<5;++trial)for(unsigned order=0;order<3;++order){
                    unsigned v=(trial+order)%3;CHECK(cudaEventRecord(start,stream));
                    for(unsigned i=0;i<100;++i)launch(v);CHECK(cudaEventRecord(stop,stream));CHECK(cudaEventSynchronize(stop));
                    float ms;CHECK(cudaEventElapsedTime(&ms,start,stop));samples[v].push_back(ms*10);
                }
                const char* names[]={"original_W4A16_M64","predecoded_FP8_M64","predecoded_FP8_M16"};
                for(unsigned v=0;v<3;++v){auto& values=samples[v];std::sort(values.begin(),values.end());
                    std::printf("TIMING shape=%s M=%u path=%s us=%.3f eager_events median5x100 interleaved excludes_predecode hot_weights_not_fullmodel\n",
                        s.name,m,names[v],values[2]);}
                CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));CHECK(cudaStreamSynchronize(stream));
                check_outputs();immutable();
            }
        }
        CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
    }
    guards();CHECK(cudaStreamDestroy(stream));
}
#endif

int main(int argc,char** argv){
    const bool host=argc==2&&!std::strcmp(argv[1],"--host-test");
    const bool timing=argc==2&&!std::strcmp(argv[1],"--timing");
    require(argc==1||host||timing,"usage: bench-glm-shared-fp8 [--host-test|--timing]");
    host_tests();if(host)return 0;
#ifdef ATLAS_SHARED_FP8_HOST_ONLY
    require(false,"CPU-only build supports --host-test only");
#else
    for(const auto& s:shapes)run_shape(s,false);
    if(timing)for(const auto& s:shapes)run_shape(s,true);
    require(device_live==0,"all explicit allocations released");
    std::printf("PASS standalone FP8 shared cache peak_device=%zu cap=%zu no_production_promotion\n",device_peak,memory_limit);
#endif
}
