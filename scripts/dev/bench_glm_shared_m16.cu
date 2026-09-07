// SPDX-License-Identifier: AGPL-3.0-only
// Standalone only: see glm_shared_m16_plan.md. Root owns all GPU execution.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#ifndef ATLAS_SHARED_M16_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu"
#endif
#include "glm_shared_m16.cuh"

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
constexpr size_t memory_limit = 32ULL * 1024 * 1024;
constexpr unsigned max_rows = 16;
static size_t device_live = 0, device_peak = 0;
static bool allocation_fits(size_t n, size_t width, size_t live) {
    return width && n <= (memory_limit - 256) / width
        && live <= memory_limit - (n * width + 256);
}
struct Shape { const char* name; unsigned n, k, ldb; };
static const std::array<Shape, 3> shapes = {{{"gate_up",2048,4096,2048},
    {"down",4096,2048,4096},{"padded_tail",129,64,256}}};
static size_t fixture_budget(const Shape& s) {
    const size_t sizes[] = {size_t(max_rows)*s.k*2,
        size_t(s.k/2)*s.ldb, size_t(s.k/2)*s.ldb,
        size_t(s.k/16)*s.ldb, size_t(s.k/16)*s.ldb,
        size_t(max_rows)*s.n*2,size_t(max_rows)*s.n*2,
        size_t(max_rows)*s.n*2,size_t(max_rows)*s.n*2};
    size_t live=0;
    for(size_t bytes:sizes){require(allocation_fits(bytes,1,live),"fixture bounded allocation");live+=bytes+256;}
    return live;
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
static double round_e4m3(double x) {
    const bool sign=std::signbit(x);double magnitude=std::fabs(x);
    unsigned best=0;double error=std::numeric_limits<double>::infinity();
    for(unsigned c=0;c<127;++c){double d=std::fabs(magnitude-e4m3(c));
        if(d<error||(d==error&&!(c&1))){best=c;error=d;}}
    return e4m3(best|(sign?128:0));
}
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
static void host_tests() {
    for(unsigned m=0;m<=16;++m){
        std::vector<unsigned> seen(16*128);
        for(unsigned warp=0;warp<4;++warp)for(unsigned lane=0;lane<32;++lane)
            for(unsigned nt=0;nt<4;++nt)for(unsigned half=0;half<2;++half)for(unsigned side=0;side<2;++side){
                unsigned r=glm_shared_m16_row(lane/4,half),c=glm_shared_m16_col(warp,nt,(lane%4)*2+side);
                require(r<16&&c<128,"M16 output bounds");if(r<m)++seen[r*128+c];}
        for(unsigned r=0;r<16;++r)for(unsigned c=0;c<128;++c)
            require(seen[r*128+c]==unsigned(r<m),"four warps partition N exactly once");
    }
    std::vector<unsigned> a(16*32),b(16*128),scales(2*128);
    for(unsigned tid=0;tid<128;++tid){
        if(tid<64)for(unsigned j=0;j<8;++j)++a[(tid/4)*32+(tid%4)*8+j];
        for(unsigned j=0;j<16;++j){++b[(tid/8)*128+(tid%8)*16+j];
            if(tid/8<2)++scales[(tid/8)*128+(tid%8)*16+j];}}
    for(const auto* v:{&a,&b,&scales})for(unsigned n:*v)require(n==1,"A/B/scale copy coverage");
    for(const auto& s:shapes){
        require(glm_shared_m16_shape(0,s.n,s.k,s.ldb)&&glm_shared_m16_shape(16,s.n,s.k,s.ldb),"supported shape");
        for(unsigned nt=0;nt<(s.n+127)/128;++nt)for(unsigned kb=0;kb<s.k;kb+=32)
            for(unsigned tid=0;tid<128;++tid){
                size_t bp=size_t(kb/2+tid/8)*s.ldb+nt*128+(tid%8)*16;
                require(bp+15<size_t(s.k/2)*s.ldb,"padded B load within allocation");
                if(tid/8<2){size_t bs=size_t(kb/16+tid/8)*s.ldb+nt*128+(tid%8)*16;
                    require(bs+15<size_t(s.k/16)*s.ldb,"scale load within allocation");}}
        require(fixture_budget(s)<memory_limit,"32MiB budget");
    }
    require(!glm_shared_m16_shape(17,2048,4096,2048)&&!glm_shared_m16_shape(5,0,4096,2048),"reject width/N");
    require(!glm_shared_m16_shape(5,2048,4095,2048)&&!glm_shared_m16_shape(5,129,64,129),"reject K/LDB alignment");
    require(!glm_shared_m16_shape(5,2048,4096,128),"reject short LDB");
    require(!glm_shared_m16_shape(5,2048,8192,2048)&&!glm_shared_m16_shape(5,8192,32,8192),"bounded standalone geometry");
    require(allocation_fits(1,2,0)&&!allocation_fits(std::numeric_limits<size_t>::max(),2,0),"allocation size overflow");
    require(!allocation_fits(1,1,memory_limit)&&!allocation_fits(1,1,std::numeric_limits<size_t>::max()),"allocation live overflow");
    require(round_e4m3(1.0625)==1.0&&round_e4m3(1.1875)==1.25&&round_e4m3(-1000)==-448,"E4M3 ties/saturation");
    require(round_e4m3(std::ldexp(1.0,-10))==0&&round_e4m3(3*std::ldexp(1.0,-10))==std::ldexp(1.0,-8),"E4M3 subnormal ties");
    require(bf16_bits(1.0)==0x3f80&&from_bf16(0xbf80)==-1.0,"BF16 reference");
    require(within_one_bf16_ulp(0x3f80,0x3f81)&&!within_one_bf16_ulp(0x3f80,0x3f82)
        &&within_one_bf16_ulp(0,0x8000)&&within_one_bf16_ulp(0xbf80,0xbf81),"CPU strict one-ULP bound");
    std::printf("PASS host ownership_M0to16 copy_coverage padded_LDB bounds E4M3_RNE budgets=%zu/%zu/%zu cap=%zu\n",
        fixture_budget(shapes[0]),fixture_budget(shapes[1]),fixture_budget(shapes[2]),memory_limit);
}

#ifndef ATLAS_SHARED_M16_HOST_ONLY
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
    const std::vector<unsigned char>& b,const std::vector<unsigned char>& scales,float s2,
    const std::vector<unsigned short>& output) {
    std::vector<double> qa(size_t(m)*s.k);
    std::array<double,65536> rounded;rounded.fill(std::numeric_limits<double>::quiet_NaN());
    for(size_t i=0;i<qa.size();++i){unsigned bits=a[i];
        if(std::isnan(rounded[bits]))rounded[bits]=round_e4m3(from_bf16(a[i]));qa[i]=rounded[bits];}
    std::vector<unsigned> columns={0,1,31,32,127,128,s.n-1};
    columns.erase(std::remove_if(columns.begin(),columns.end(),[&](unsigned c){return c>=s.n;}),columns.end());
    std::sort(columns.begin(),columns.end());columns.erase(std::unique(columns.begin(),columns.end()),columns.end());
    for(unsigned col:columns){std::vector<double> qb(s.k);
        for(unsigned k=0;k<s.k;++k){unsigned packed=b[size_t(k/2)*s.ldb+col];
            // Original arithmetic: scale E4M3→FP32 * scale2, then LUT * scale.
            float scale_value=float(e4m3(scales[size_t(k/16)*s.ldb+col]))*s2;
            float weight=float(fp4((packed>>(4*(k%2)))&15))*scale_value;
            qb[k]=round_e4m3(weight);}
        for(unsigned row=0;row<m;++row){double sum=0;
            for(unsigned k=0;k<s.k;++k)sum+=qa[size_t(row)*s.k+k]*qb[k];
            auto got=output[size_t(row)*s.n+col],want=bf16_bits(sum);
            if(!within_one_bf16_ulp(got,want)){std::fprintf(stderr,"CPU mismatch shape=%s M=%u row=%u col=%u got=%g want=%g\n",s.name,m,row,col,from_bf16(got),from_bf16(want));require(false,"CPU E4M3 projection oracle");}}
    }
}
static void run_shape(const Shape& s,bool timing) {
    require(glm_shared_m16_shape(max_rows,s.n,s.k,s.ldb),"shape preflight");
    Buffer<unsigned short> a(size_t(max_rows)*s.k);
    Buffer<unsigned char> b0(size_t(s.k/2)*s.ldb),b1(b0.count),bs0(size_t(s.k/16)*s.ldb),bs1(bs0.count);
    Buffer<unsigned short> old0(size_t(max_rows)*s.n),old1(old0.count),tile0(old0.count),tile1(old0.count);
    require(device_live==fixture_budget(s),"actual CPU budget matches allocations");
    auto guards=[&](){a.guards();b0.guards();b1.guards();bs0.guards();bs1.guards();
        old0.guards();old1.guards();tile0.guards();tile1.guards();};
    std::vector<unsigned char> hb[2],hs[2];
    for(unsigned w=0;w<2;++w){hb[w].resize(b0.count);hs[w].resize(bs0.count);
        for(size_t j=0;j<hb[w].size();++j)hb[w][j]=packed_code(j,31+w);}
    b0.upload(hb[0]);b1.upload(hb[1]);
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(unsigned m:{0u,1u,4u,5u,15u,16u}){
        const float scale2[2]={0.75f,0.375f};
        auto launch=[&](unsigned variant,unsigned w){
            require(variant<2&&w<2,"kernel selection");
            auto* b=w?b1.ptr:b0.ptr;auto* bs=w?bs1.ptr:bs0.ptr;
            auto* out=variant?(w?tile1.ptr:tile0.ptr):(w?old1.ptr:old0.ptr);
            const dim3 grid((s.n+127)/128,1);
            if(variant==0)w4a16_gemm_t<<<grid,128,0,stream>>>(
                reinterpret_cast<__nv_bfloat16*>(a.ptr),b,bs,scale2[w],reinterpret_cast<__nv_bfloat16*>(out),m,s.n,s.k,s.ldb);
            else glm_shared_w4a16_m16<<<grid,128,0,stream>>>(
                reinterpret_cast<__nv_bfloat16*>(a.ptr),b,bs,scale2[w],reinterpret_cast<__nv_bfloat16*>(out),m,s.n,s.k,s.ldb);
            CHECK(cudaGetLastError());
        };
        cudaGraph_t graph;cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(unsigned v=0;v<2;++v)for(unsigned w=0;w<2;++w)launch(v,w);
        CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        std::vector<unsigned short> previous[2];
        for(unsigned round=0;round<2;++round){
            std::vector<unsigned short> ha(a.count,0x7fc1); // NaN padding must never enter a live row.
            for(unsigned row=0;row<m;++row)for(unsigned k=0;k<s.k;++k){
                unsigned source=round?m-1-row:row; // replay reverses row identities and changes values
                int value=int(packed_code(size_t(source)*s.k+k,101+round+m))-128;
                ha[size_t(row)*s.k+k]=bf16_bits(double(value)/128.0);}
            if(m>1)std::fill(ha.begin()+(round%m)*s.k,ha.begin()+(round%m+1)*s.k,0);
            for(unsigned w=0;w<2;++w)for(size_t j=0;j<hs[w].size();++j)hs[w][j]=scale_code(j,w+round+m);
            a.upload(ha);bs0.upload(hs[0]);bs1.upload(hs[1]);
            for(auto* out:{&old0,&old1,&tile0,&tile1})CHECK(cudaMemsetAsync(out->ptr,0x5a,out->count*2,stream));
            if(round)CHECK(cudaGraphLaunch(exec,stream));else for(unsigned v=0;v<2;++v)for(unsigned w=0;w<2;++w)launch(v,w);
            CHECK(cudaStreamSynchronize(stream));
            const std::vector<unsigned short> reference[]={old0.read(),old1.read()};
            for(unsigned w=0;w<2;++w){
                if(round&&m)require(reference[w]!=previous[w],"graph replay consumed refreshed values");
                previous[w]=reference[w];}
            auto check_outputs=[&](){
                exact(reference[0],old0.read(),"old gate/down fresh output");exact(reference[1],old1.read(),"old up/down fresh output");
                exact(reference[0],tile0.read(),"M16 gate/down bit equality");exact(reference[1],tile1.read(),"M16 up/down bit equality");
                for(unsigned w=0;w<2;++w)for(unsigned row=0;row<max_rows;++row)for(unsigned col=0;col<s.n;++col){
                    unsigned short bits=reference[w][size_t(row)*s.n+col];
                    if(row>=m)require(bits==0x5a5a,"inactive output tail unchanged");
                    else{require(bits!=0x5a5a&&std::isfinite(from_bf16(bits)),"finite freshly written projection");
                        if(m>1&&row==round%m)require(from_bf16(bits)==0,"exact zero source row");}}
            };
            auto immutable=[&](){exact(ha,a.read(),"A including NaN padding unchanged");
                exact(hb[0],b0.read(),"B0 unchanged");exact(hb[1],b1.read(),"B1 unchanged");
                exact(hs[0],bs0.read(),"scale0 unchanged");exact(hs[1],bs1.read(),"scale1 unchanged");guards();};
            check_outputs();immutable();
            for(unsigned w=0;w<2;++w)cpu_columns(s,m,ha,hb[w],hs[w],scale2[w],reference[w]);
            std::printf("PASS shared shape=%s M=%u mode=%s full_BITEXACT CPU_columns zero tails immutable guards\n",s.name,m,round?"graph_refreshed":"eager");
            if(timing&&round==1&&(m==4||m==5)){
                for(unsigned i=0;i<5;++i)for(unsigned v=0;v<2;++v)for(unsigned w=0;w<2;++w)launch(v,w);
                cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
                std::vector<float> samples[2][2];
                for(unsigned trial=0;trial<5;++trial)for(unsigned order=0;order<4;++order){
                    unsigned variant=(trial+order)%4,v=variant/2,w=variant%2;
                    CHECK(cudaEventRecord(begin,stream));for(unsigned i=0;i<100;++i)launch(v,w);
                    CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
                    float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));samples[v][w].push_back(ms*10);}
                for(unsigned v=0;v<2;++v)for(unsigned w=0;w<2;++w){auto& values=samples[v][w];std::sort(values.begin(),values.end());
                    std::printf("TIMING shape=%s M=%u projection=%u variant=%s us=%.3f eager_events median5x100 interleaved\n",s.name,m,w,v?"M16_Nwarp":"original_M64",values[2]);}
                CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamSynchronize(stream));
                check_outputs();immutable();
            }
        }
        CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
    }
    guards();CHECK(cudaStreamDestroy(stream));
}
#endif

int main(int argc,char** argv) {
    const bool host=argc==2&&!std::strcmp(argv[1],"--host-test");
    const bool timing=argc==2&&!std::strcmp(argv[1],"--timing");
    require(argc==1||host||timing,"usage: bench-glm-shared-m16 [--host-test|--timing]");
    host_tests();if(host)return 0;
#ifdef ATLAS_SHARED_M16_HOST_ONLY
    require(false,"CPU-only build accepts --host-test only");
#else
    // No performance sample is allowed before all three shapes pass every gate.
    for(const auto& s:shapes)run_shape(s,false);
    if(timing)for(const auto& s:shapes)run_shape(s,true);
    require(device_live==0,"fixture allocations released");
    std::printf("PASS shared M16 standalone complete device_peak=%zu cap=%zu no_production_promotion\n",device_peak,memory_limit);
#endif
}
