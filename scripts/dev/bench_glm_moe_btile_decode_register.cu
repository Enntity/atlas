// SPDX-License-Identifier: AGPL-3.0-only
// Standalone BF16-input B-tile compatibility; see glm_moe_btile_decode_register_plan.md.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#ifndef ATLAS_BTILE_DECODE_HOST_ONLY
#include <cuda_runtime.h>
#include "../../kernels/gb10/common/moe_shared_expert_fused_t.cu"
#undef BLOCK_SIZE
#undef GROUP_SIZE
// These standalone TUs each declare their private E4M3 helper. Rename only
// helper linkage while retaining the exact production kernel implementations.
#define atlas_dec_e4m3 btdec_oracle_e4m3_2
#include "../../kernels/gb10/common/moe_shared_expert_fused_batch2_t.cu"
#undef atlas_dec_e4m3
#undef BLOCK_SIZE
#undef GROUP_SIZE
#define atlas_dec_e4m3 btdec_oracle_e4m3_3
#include "../../kernels/gb10/common/moe_shared_expert_fused_batch3_t.cu"
#undef atlas_dec_e4m3
#undef BLOCK_SIZE
#undef GROUP_SIZE
#include "glm_moe_btile_decode_register.cuh"
#endif

static void require(bool yes,const char* why){if(!yes){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(2);}}
constexpr unsigned DN=2048,DK=4096,NE=288,TK=8,MAXR=3;
constexpr size_t WB=size_t(DN)*DK/2,SB=size_t(DN)*DK/16,CAP=64ULL*1024*1024;
static size_t tile_offset(unsigned n,unsigned kp){return (((size_t(n/128)*(DK/64)+kp/32)*128+n%128)*32)+kp%32;}
static size_t original_offset(size_t dst){size_t t=dst/4096,i=dst%4096;return ((t%64)*32+i%32)*DN+(t/64)*128+i/32;}
static unsigned char code(size_t i,unsigned seed){unsigned x=unsigned(i)^seed*0x9e3779b9u;x^=x>>16;x*=0x7feb352du;x^=x>>15;return static_cast<unsigned char>(x^(x>>8));}
static unsigned char scode(size_t i,unsigned seed){return static_cast<unsigned char>(0x20+8*((i+seed)%3));}
static bool fits(size_t n,size_t elem,size_t live){return elem&&n<=(CAP-256)/elem&&live<=CAP-(n*elem+256);}
static size_t fixture_budget(){
    const size_t sizes[]={6*WB,4*WB,6*SB,MAXR*DK*2,
        MAXR*TK*DN*2,MAXR*TK*DN*2,MAXR*DN*2,MAXR*DN*2,
        MAXR*TK*DN*2,MAXR*TK*DN*2,MAXR*DN*2,MAXR*DN*2,
        MAXR*TK*DN*2,MAXR*TK*DN*2,MAXR*DN*2,MAXR*DN*2,
        NE*8,NE*8,NE*8,NE*8,NE*8,NE*8,NE*4,NE*4,MAXR*TK*4};
    size_t used=0;for(size_t bytes:sizes){require(fits(bytes,1,used),"fixture allocation budget");used+=bytes+256;}return used;
}
static bool valid_routes(const std::vector<unsigned>& ids,unsigned rows){
    if(rows<1||rows>MAXR||ids.size()!=rows*TK)return false;
    for(unsigned r=0;r<rows;++r){std::array<bool,NE> seen{};for(unsigned j=0;j<TK;++j){unsigned e=ids[r*TK+j];if(e>=NE||seen[e])return false;seen[e]=true;}}
    return true;
}
static std::vector<unsigned char> pack(const std::vector<unsigned char>& src){
    require(src.size()==WB,"pack extent");std::vector<unsigned char> dst(WB);size_t p=0;
    for(unsigned nt=0;nt<DN/128;++nt)for(unsigned kt=0;kt<DK/64;++kt)
        for(unsigned n=0;n<128;++n)for(unsigned b=0;b<32;++b)dst[p++]=src[size_t(kt*32+b)*DN+nt*128+n];
    return dst;
}
static unsigned char word_byte(unsigned lo,unsigned hi,unsigned j){return static_cast<unsigned char>((j<4?lo:hi)>>(8*(j%4)));}
static void register_byte_tests(){
    require(word_byte(0x76543210u,0xfedcba98u,0)==0x10&&word_byte(0x76543210u,0xfedcba98u,7)==0xfe,"packed word byte extraction");
    std::vector<unsigned char> src(WB);for(size_t i=0;i<WB;++i)src[i]=code(i,13);auto tiled=pack(src);
    for(unsigned n=0;n<DN;++n)for(unsigned kb=0;kb<DK/2;kb+=32){
        unsigned words[8];std::memcpy(words,tiled.data()+tile_offset(n,kb),32);
        for(unsigned group=0;group<4;++group)for(unsigned j=0;j<8;++j){
            unsigned lo,hi;std::memcpy(&lo,tiled.data()+tile_offset(n,kb+group*8),4);std::memcpy(&hi,tiled.data()+tile_offset(n,kb+group*8+4),4);
            unsigned char expected=src[size_t(kb+group*8+j)*DN+n];
            require(word_byte(lo,hi,j)==expected,"word register decode");
            require(word_byte(words[group*2],words[group*2+1],j)==expected,"vec register group decode");
        }
    }
}
static void host_tests(){
    register_byte_tests();
    require(tile_offset(1,0)==32&&tile_offset(0,32)==4096&&tile_offset(128,0)==262144,"B tile corners");
    std::vector<unsigned char> seen(WB),src(WB);for(size_t i=0;i<WB;++i)src[i]=code(i,13);
    auto dst=pack(src);
    for(unsigned kp=0;kp<DK/2;++kp)for(unsigned n=0;n<DN;++n){size_t p=tile_offset(n,kp);require(p<WB&&!seen[p],"B bijection");seen[p]=1;require(original_offset(p)==size_t(kp)*DN+n&&dst[p]==src[size_t(kp)*DN+n],"pack inverse bytes");}
    // Simulate both cooperative 16-byte load rounds and per-lane consumption.
    for(unsigned nb=0;nb<DN;nb+=32)for(unsigned kb=0;kb<DK/2;kb+=32){
        unsigned char stage[32][36]{};size_t base=tile_offset(nb,kb);
        for(unsigned round=0;round<2;++round)for(unsigned lane=0;lane<32;++lane){unsigned linear=(round*32+lane)*16;std::memcpy(stage[linear/32]+linear%32,&dst[base+linear],16);}
        for(unsigned lane=0;lane<32;++lane)for(unsigned b=0;b<32;++b)require(stage[lane][b]==src[size_t(kb+b)*DN+nb+lane],"cooperative stage slice");
    }
    for(unsigned r=1;r<=3;++r){std::vector<unsigned> ids(r*TK);for(unsigned i=0;i<ids.size();++i)ids[i]=i%TK;require(valid_routes(ids,r),"valid routes");ids[0]=NE;require(!valid_routes(ids,r),"reject expert bound");ids[0]=ids[1];require(!valid_routes(ids,r),"reject duplicate within token");}
    require(!valid_routes({},0)&&!valid_routes({},4)&&fits(1,4,0)&&!fits(std::numeric_limits<size_t>::max(),4,0),"shape and budget rejection");
    require(fixture_budget()==45799520,"exact guarded device footprint");
    std::puts("PASS host Btile decode word_vec_bytes exhaustive_pack staged_slice routes budget");
}

#ifndef ATLAS_BTILE_DECODE_HOST_ONLY
#define CK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
static size_t live=0,peak=0;
template<class T> struct Buffer{
    T* base;T* ptr;size_t count,bytes;
    Buffer(size_t n):count(n),bytes(n*sizeof(T)+256){require(fits(n,sizeof(T),live),"64MiB allocation cap");CK(cudaMalloc(&base,bytes));ptr=base+128/sizeof(T);live+=bytes;peak=std::max(peak,live);CK(cudaMemset(base,0xa5,bytes));}
    ~Buffer(){cudaFree(base);live-=bytes;}
    Buffer(const Buffer&)=delete;
    void upload(const std::vector<T>& data,size_t offset=0){require(offset<=count&&data.size()<=count-offset,"upload bounds");CK(cudaMemcpy(ptr+offset,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(size_t offset=0,size_t n=0)const{if(!n)n=count;require(offset<=count&&n<=count-offset,"read bounds");std::vector<T> out(n);CK(cudaMemcpy(out.data(),ptr+offset,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}
    void guards()const{unsigned char a[128],b[128];CK(cudaMemcpy(a,base,128,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b,ptr+count,128,cudaMemcpyDeviceToHost));for(unsigned i=0;i<128;++i)require(a[i]==0xa5&&b[i]==0xa5,"allocation canary");}
};
template<class T> static void exact(const std::vector<T>& a,const std::vector<T>& b,const char* label){require(a.size()==b.size(),"exact extent");for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(T))){std::fprintf(stderr,"FAIL %s index=%zu\n",label,i);std::exit(2);}}
static unsigned short bf16(double d){float f=float(d);unsigned b;std::memcpy(&b,&f,4);return (b+0x7fff+((b>>16)&1))>>16;}
static float from_bf16(unsigned short h){unsigned b=unsigned(h)<<16;float f;std::memcpy(&f,&b,4);return f;}
static double fp4(unsigned x){const double v[]={0,.5,1,1.5,2,3,4,6};return (x&8?-1:1)*v[x&7];}
static double fscale(unsigned char s){return std::ldexp(1.0,int(s>>3)-7);}
static void gpu_tests(bool timing){
    Buffer<unsigned char> orig(6*WB),tile(4*WB),scales(6*SB);
    Buffer<unsigned short> input(MAXR*DK);
    constexpr size_t routed=MAXR*TK*DN,shared=MAXR*DN;
    Buffer<unsigned short> out[12]={{routed},{routed},{shared},{shared},{routed},{routed},{shared},{shared},{routed},{routed},{shared},{shared}};
    Buffer<unsigned long long> gp(NE),up(NE),gs(NE),us(NE),tp(NE),tu(NE);
    Buffer<float> gf(NE),uf(NE);Buffer<unsigned> ids(MAXR*TK);
    auto guards=[&](){orig.guards();tile.guards();scales.guards();input.guards();for(auto& b:out)b.guards();gp.guards();up.guards();gs.guards();us.guards();tp.guards();tu.guards();gf.guards();uf.guards();ids.guards();};
    for(unsigned w=0;w<6;++w){std::vector<unsigned char> b(WB);for(size_t i=0;i<WB;++i)b[i]=code(i,w+31);orig.upload(b,w*WB);if(w<4)tile.upload(pack(b),w*WB);b.resize(SB);for(size_t i=0;i<SB;++i)b[i]=scode(i,w);scales.upload(b,w*SB);}
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    require(live==fixture_budget(),"actual allocation matches CPU accounting");
    std::printf("device_bytes=%zu cap=%zu original_tiled_routed_pairs=2 shared_original_pair=1\n",peak,CAP);
    for(unsigned rows=1;rows<=MAXR;++rows)for(unsigned ci=0;ci<8;++ci){
        // Shared stays original; exercise all four present/absent masks.
        unsigned mask=ci%4;
        auto launch=[&](unsigned v){
            require(v<3,"variant");dim3 grid(DN/32,rows*(TK+1),2);
            auto cv=[](unsigned short* p){return reinterpret_cast<__nv_bfloat16*>(p);};
#define DARGS(G,U) cv(input.ptr),G,gs.ptr,gf.ptr,cv(out[v*4].ptr),U,us.ptr,uf.ptr,cv(out[v*4+1].ptr),ids.ptr,mask&1?orig.ptr+4*WB:nullptr,mask&1?scales.ptr+4*SB:nullptr,0.5f,cv(out[v*4+2].ptr),mask&2?orig.ptr+5*WB:nullptr,mask&2?scales.ptr+5*SB:nullptr,2.0f,cv(out[v*4+3].ptr),DN,DK,TK
            if(v==0){
                if(rows==1)moe_expert_gate_up_shared_t<<<grid,32,0,stream>>>(DARGS(gp.ptr,up.ptr));
                else if(rows==2)moe_expert_gate_up_shared_batch2_t<<<grid,32,0,stream>>>(DARGS(gp.ptr,up.ptr));
                else moe_expert_gate_up_shared_batch3_t<<<grid,32,0,stream>>>(DARGS(gp.ptr,up.ptr));
            }else if(v==1){
                if(rows==1)glm_btile_decode_word1<<<grid,32,0,stream>>>(DARGS(tp.ptr,tu.ptr));
                else if(rows==2)glm_btile_decode_word2<<<grid,32,0,stream>>>(DARGS(tp.ptr,tu.ptr));
                else glm_btile_decode_word3<<<grid,32,0,stream>>>(DARGS(tp.ptr,tu.ptr));
            }else{
                if(rows==1)glm_btile_decode_vec1<<<grid,32,0,stream>>>(DARGS(tp.ptr,tu.ptr));
                else if(rows==2)glm_btile_decode_vec2<<<grid,32,0,stream>>>(DARGS(tp.ptr,tu.ptr));
                else glm_btile_decode_vec3<<<grid,32,0,stream>>>(DARGS(tp.ptr,tu.ptr));
            }
#undef DARGS
            CK(cudaGetLastError());
        };
        cudaGraph_t graph;cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(unsigned v=0;v<3;++v)launch(v);
        CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(unsigned round=0;round<2;++round){
            std::array<int,NE> local;local.fill(-1);
            unsigned e0=ci%2?17:0,e1=ci%2?287:1;
            if(ci!=2){local[e0]=round%2;local[e1]=1-int(round%2);}
            const unsigned choices[]={e0,e1,142,143,144,145,286,37};
            std::vector<unsigned> hi(MAXR*TK,NE);
            for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<TK;++j)hi[r*TK+j]=choices[(j+r+round)%TK];
            require(valid_routes(std::vector<unsigned>(hi.begin(),hi.begin()+rows*TK),rows),"route guard before CUDA");
            std::vector<unsigned long long> hgp(NE),hup(NE),hgs(NE),hus(NE),htp(NE),htu(NE);
            std::vector<float> hgf(NE,1),huf(NE,1);
            for(unsigned e=0;e<NE;++e)if(local[e]>=0){unsigned g=unsigned(local[e]),u=g+2;
                hgp[e]=(unsigned long long)(orig.ptr+g*WB);hup[e]=(unsigned long long)(orig.ptr+u*WB);
                htp[e]=(unsigned long long)(tile.ptr+g*WB);htu[e]=(unsigned long long)(tile.ptr+u*WB);
                hgs[e]=(unsigned long long)(scales.ptr+g*SB);hus[e]=(unsigned long long)(scales.ptr+u*SB);
                hgf[e]=std::ldexp(1.0f,int(g)-1);huf[e]=std::ldexp(1.0f,int(u)-2);
            }
            std::vector<unsigned short> ha(MAXR*DK);
            for(unsigned r=0;r<MAXR;++r)for(unsigned k=0;k<DK;++k){int v=int(code(k+size_t((r+round)%MAXR)*DK,101+ci)%33)-16;ha[size_t(r)*DK+k]=bf16(ci==3?0:double(v)/16);}
            gp.upload(hgp);up.upload(hup);gs.upload(hgs);us.upload(hus);tp.upload(htp);tu.upload(htu);gf.upload(hgf);uf.upload(huf);ids.upload(hi);input.upload(ha);
            for(auto& b:out)CK(cudaMemsetAsync(b.ptr,0x5a,b.count*2,stream));
            if(round)CK(cudaGraphLaunch(exec,stream));else for(unsigned v=0;v<3;++v)launch(v);
            CK(cudaStreamSynchronize(stream));
            std::array<std::vector<unsigned short>,4> reference;
            for(unsigned p=0;p<4;++p)reference[p]=out[p].read();
            auto compare=[&](){for(unsigned v=0;v<3;++v)for(unsigned p=0;p<4;++p)exact(reference[p],out[v*4+p].read(),"complete routed/shared outputs");};
            auto immutable=[&](){exact(ha,input.read(),"BF16 input");exact(hi,ids.read(),"route IDs");exact(hgp,gp.read(),"gate ptrs");exact(hup,up.read(),"up ptrs");exact(hgs,gs.read(),"gate scales");exact(hus,us.read(),"up scales");exact(htp,tp.read(),"tile gate ptrs");exact(htu,tu.read(),"tile up ptrs");exact(hgf,gf.read(),"gate scale2");exact(huf,uf.read(),"up scale2");};
            compare();
            for(unsigned p=0;p<4;++p){bool sh=p>=2;unsigned proj=p%2,nrows=sh?rows:rows*TK,maxrows=sh?MAXR:MAXR*TK;
                for(unsigned row=0;row<maxrows;++row){bool active=row<nrows;int w=-1;double factor=1;
                    if(active){if(sh){if(mask&(1u<<proj)){w=4+proj;factor=proj?2:.5;}}else{unsigned e=hi[row];if(local[e]>=0){w=local[e]+int(proj*2);factor=proj?huf[e]:hgf[e];}}}
                    for(unsigned col=0;col<DN;++col){auto bits=reference[p][size_t(row)*DN+col];if(!active)require(bits==0x5a5a,"inactive width tail");else if(w<0)require(bits==0,"remote/absent shared exact zero");else require(std::isfinite(from_bf16(bits)),"finite output");}
                    if(w<0)continue;unsigned token=sh?row:row/TK;
                    for(unsigned col:{0u,1u,31u,32u,127u,128u,1023u,2047u}){double sum=0;
                        for(unsigned k=0;k<DK;++k){unsigned b=code(size_t(k/2)*DN+col,unsigned(w)+31);sum+=double(from_bf16(ha[size_t(token)*DK+k]))*fp4((b>>(4*(k%2)))&15)*fscale(scode(size_t(k/16)*DN+col,unsigned(w)))*factor;}
                        require(reference[p][size_t(row)*DN+col]==bf16(sum),"CPU BF16-input ordered oracle columns");
                    }
                }
            }
            immutable();guards();
            std::printf("PASS rows=%u case=%u shared_mask=%u mode=%s word_vec_full_BITEXACT CPU_columns immutable canaries\n",rows,ci,mask,round?"graph":"eager");
            if(timing&&round){
                for(unsigned i=0;i<3;++i)for(unsigned v=0;v<3;++v)launch(v);
                cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float> samples[3];
                for(unsigned trial=0;trial<5;++trial)for(unsigned order=0;order<3;++order){unsigned v=(trial+order)%3;CK(cudaEventRecord(a,stream));for(unsigned i=0;i<30;++i)launch(v);CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));samples[v].push_back(ms*1000/30);}
                const char* names[]={"production_T","Btile_word","Btile_vec"};
                for(unsigned v=0;v<3;++v){std::sort(samples[v].begin(),samples[v].end());std::printf("TIMING rows=%u case=%u mask=%u path=%s us=%.3f median5x30 interleaved_eager\n",rows,ci,mask,names[v],samples[v][2]);}
                CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));compare();immutable();guards();
            }
        }
        CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
    }
    for(unsigned w=0;w<6;++w){auto b=orig.read(w*WB,WB);for(size_t i=0;i<WB;++i)require(b[i]==code(i,w+31),"original B immutable");if(w<4){b=tile.read(w*WB,WB);for(size_t i=0;i<WB;++i)require(b[i]==code(original_offset(i),w+31),"tiled B immutable");}b=scales.read(w*SB,SB);for(size_t i=0;i<SB;++i)require(b[i]==scode(i,w),"B scale immutable");}
    guards();CK(cudaStreamDestroy(stream));std::printf("PASS complete decode compatibility peak=%zu no_promotion\n",peak);
}
#endif

int main(int argc,char** argv){
    bool host=argc==2&&!std::strcmp(argv[1],"--host-test"),timing=argc==2&&!std::strcmp(argv[1],"--timing");
    require(argc==1||host||timing,"usage: bench-glm-moe-btile-decode [--host-test|--timing]");host_tests();if(host)return 0;
#ifdef ATLAS_BTILE_DECODE_HOST_ONLY
    require(false,"CPU build supports --host-test only");
#else
    gpu_tests(timing);
#endif
}
