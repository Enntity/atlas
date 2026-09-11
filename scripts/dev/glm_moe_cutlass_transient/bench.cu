// SPDX-License-Identifier: AGPL-3.0-only
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
#include <cuda_bf16.h>
#endif
#include "common.cuh"
#ifndef ATLAS_HOST_ONLY
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu"
#ifdef ATLAS_CUTLASS_COOPERATIVE_K256
#include "collective_cooperative.cuh"
#else
#include "collective.cuh"
#endif
#include "staging.cuh"

static double fp4(unsigned q) {
    constexpr double lut[8]={0,.5,1,1.5,2,3,4,6};
    return (q&8)?-lut[q&7]:lut[q&7];
}
static double fp8(unsigned b) {
    int e=(b>>3)&15,m=b&7;
    require(!(e==15 && m==7),"finite scale encoding");
    double v=e?std::ldexp(1.0+m/8.0,e-7):m/512.0;
    return b&128?-v:v;
}
static unsigned weight_byte(size_t i,unsigned seed,bool scale) {
    unsigned x=unsigned(i)^seed;x^=x>>16;x*=0x7feb352d;x^=x>>15;
    return scale?0x20+x%24:x&255;
}
static void independent(const std::vector<__nv_bfloat16>& x,
    const std::vector<__nv_bfloat16>& y,const Routing& r,
    const unsigned char* packed,const unsigned char* scales,int n,int k,bool down,
    unsigned seed,const std::array<float,288>& scale2) {
    double worst=0;
    for(int test=0;test<96;++test) {
        int e=(test*41)%144, row=r.offsets[e]+(test*13)%(r.offsets[e+1]-r.offsets[e]);
        int tok=down?row:r.ids[row],col=(test*101+7)%n;
        std::vector<unsigned char> a(k/2),s(k/16);
        CHECK(cudaMemcpy(a.data(),packed+size_t(tok)*k/2,a.size(),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(s.data(),scales+size_t(tok)*k/16,s.size(),cudaMemcpyDeviceToHost));
        double sum=0,abs_sum=0;
        for(int j=0;j<k;++j) {
            unsigned aq=(a[j/2]>>(4*(j&1)))&15;
            unsigned b=weight_byte(size_t(e)*n*k/2+size_t(j/2)*n+col,seed,false);
            unsigned bq=(b>>(4*(j&1)))&15;
            unsigned bs=weight_byte(size_t(e)*n*k/16+size_t(j/16)*n+col,seed+1,true);
            double term=fp4(aq)*fp8(s[j/16])*fp4(bq)*fp8(bs);
            sum+=term;abs_sum+=std::abs(term);
        }
        sum*=scale2[e];abs_sum*=scale2[e];
        int exponent=0;std::frexp(std::abs(sum),&exponent);
        double half_ulp=std::ldexp(1.0,exponent-9);
        double bound=half_ulp+8e-6*abs_sum+1e-5;
        for(double value:{double(__bfloat162float(x[size_t(row)*n+col])),double(__bfloat162float(y[size_t(row)*n+col]))}) {
            worst=std::max(worst,std::abs(value-sum));
            if(std::abs(value-sum)>bound) {
                std::fprintf(stderr,"independent e=%d row=%d col=%d got=%.9g expected=%.9g bound=%.9g\n",e,row,col,value,sum,bound);
                require(false,"FP64 dot oracle");
            }
        }
    }
    std::printf("PASS independent96 FP64 quantized dots max_abs_error=%.9g\n",worst);
}
struct Result {float baseline,cutlass,quant,stage,sfb,core;};
static Result run_case(bool skew,int projection) {
    constexpr int rows=4100;
    bool down=projection==2;unsigned seed=97+projection*1009;
    int n=down?4096:2048,k=down?2048:4096,expanded=rows*8,a_rows=down?expanded:rows;
    size_t wb=size_t(n)*k/2,sb=size_t(n)*k/16;
    Routing r=routing(rows,skew);
    Buffer<unsigned char> wt(144*wb),st(144*sb),wn(144*wb),sn(144*sb),sfb(144*sb);
    Buffer<__nv_bfloat16> input(size_t(a_rows)*k),old(size_t(expanded)*n),out(size_t(expanded)*n);
    Buffer<unsigned char> a(size_t(a_rows)*k/2),as(size_t(a_rows)*k/16),workspace(512ull<<20);
    Buffer<unsigned long long> wptr(288),sptr(288);Buffer<float> globals(288);
    Buffer<int> offsets(289),ids(expanded);
    Buffer<unsigned> operand_errors(1);
    std::array<unsigned long long,288> wtp{},stp{},wnp{},sfp{};
    std::array<float,288> scale2{};
    for(int e=0;e<144;++e) {
        wtp[e]=(unsigned long long)(wt.p+e*wb);stp[e]=(unsigned long long)(st.p+e*sb);
        wnp[e]=(unsigned long long)(wn.p+e*wb);sfp[e]=(unsigned long long)(sfb.p+e*sb);
        scale2[e]=.31f+float(e%17)*.07f;
    }
    cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    StagePlan plan(workspace.p,workspace.count,r,n,k);
    require(plan.sfb_stride==sb,"actual SFB extent equals allocated stride");
    size_t required=0;
    require(launch_projection(plan.p,wnp.data(),sfp.data(),scale2.data(),out.p,n,k,
        workspace.p,plan.p.cursor,workspace.count,stream,100000,true,&required)==0,"dry workspace plan and can_implement");
    require(required<=workspace.count,"all staging and GEMM capacity prevalidated");
    require(required>plan.p.cursor,"nonempty GEMM metadata/workspace extent");
    require(launch_projection(plan.p,wnp.data(),sfp.data(),scale2.data(),out.p,n,k,
        workspace.p,plan.p.cursor,required-1,stream,100000,true,nullptr)==-2,"one-byte-short workspace rejected before writes");
    {
        auto untouched=workspace.read();
        require(std::all_of(untouched.begin(),untouched.end(),[](unsigned char x){return x==0xa5;}),
            "full512MiB workspace untouched after both dry planning calls");
    }
    std::printf("plan projection=%d skew=%d local_experts=144 max_expert_rows=%d staging_bytes=%zu workspace_required=%zu cap=%zu SFB=%zu live_MiB=%.3f\n",
        projection,skew,r.max_rows,plan.p.cursor,required,workspace.count,sfb.count,live/1048576.0);
    // No candidate workspace writes or kernels precede the validated plan.
    plan.upload(stream);wptr.upload(wtp.data());sptr.upload(stp.data());globals.upload(scale2.data());
    offsets.upload(r.offsets.data());ids.upload(r.ids.data());
    fill<<<4096,256,0,stream>>>(wt.p,wt.count,false,seed);fill<<<4096,256,0,stream>>>(st.p,st.count,true,seed+1);
    fill_bf16<<<1024,256,0,stream>>>(input.p,input.count,29);
    native_from_t<<<dim3((wb+255)/256,1,144),256,0,stream>>>(wt.p,wn.p,n,k/2);
    native_from_t<<<dim3((sb+255)/256,1,144),256,0,stream>>>(st.p,sn.p,n,k/16);
    CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(stream));
    auto quant=[&] {quantize_bf16_to_nvfp4<<<a_rows,256,0,stream>>>(input.p,a.p,as.p,1.0f,a_rows,k);CHECK(cudaGetLastError());};
    auto astage=[&] {plan.stage(a.p,as.p,down?nullptr:ids.p,offsets.p,r.max_rows,k,stream);};
    auto bstage=[&] {stage_sfb<<<dim3((sb+255)/256,1,144),256,0,stream>>>(sn.p,sfb.p,n,k,sb);CHECK(cudaGetLastError());};
    auto baseline=[&] {
        quant();moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dim3((n+127)/128,(r.max_rows+63)/64,288),128,0,stream>>>(
            a.p,as.p,wptr.p,sptr.p,globals.p,old.p,offsets.p,down?nullptr:ids.p,288,n,k);CHECK(cudaGetLastError());
    };
    auto core=[&] {
        require(launch_projection(plan.p,wnp.data(),sfp.data(),scale2.data(),out.p,n,k,
            workspace.p,plan.p.cursor,workspace.count,stream,100000,false,nullptr)==0,"CUTLASS launch");
        CHECK(cudaGetLastError());
    };
    auto candidate=[&] {quant();astage();bstage();core();};
    CHECK(cudaMemset(old.p,0xa5,old.count*2));CHECK(cudaMemset(out.p,0x5a,out.count*2));
    baseline();candidate();CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemset(operand_errors.p,0,sizeof(unsigned)));
    size_t verify_extent=std::max(wb,size_t(r.max_rows)*k/2);
    verify_operands<<<dim3((verify_extent+255)/256,1,144),256,0,stream>>>(a.p,as.p,wt.p,st.p,wn.p,sn.p,sfb.p,
        (unsigned char* const*)plan.p.dA,(unsigned char* const*)plan.p.dSFA,offsets.p,down?nullptr:ids.p,n,k,sb,operand_errors.p);
    CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(stream));
    require(operand_errors.read()[0]==0,"all quantized operand bytes and scales identical");
    std::puts("PASS exhaustive A/SFA/B/SFB byte identity and one-byte-short workspace rejection");
    {
        auto x=old.read(),y=out.read();double err2=0,ref2=0,maxerr=0;size_t changed=0,count=0;
        auto* xb=(unsigned short*)x.data();auto* yb=(unsigned short*)y.data();
        for(int e=0;e<288;++e) for(size_t i=size_t(r.offsets[e])*n;i<size_t(r.offsets[e+1])*n;++i) {
            if(e>=144) {require(xb[i]==0xa5a5 && yb[i]==0x5a5a,"remote writes");continue;}
            double xx=__bfloat162float(x[i]),yy=__bfloat162float(y[i]),diff=std::abs(xx-yy);
            require(std::isfinite(xx)&&std::isfinite(yy),"finite outputs");
            require(diff<=.008*std::max({std::abs(xx),std::abs(yy),1.0}),"per-element tight reduction bound");
            err2+=diff*diff;ref2+=xx*xx;maxerr=std::max(maxerr,diff);changed+=xb[i]!=yb[i];++count;
        }
        double rel=std::sqrt(err2/std::max(ref2,1e-30));require(rel<=.004,"relative L2 <=0.004");
        std::printf("PASS outputs count=%zu changed_bits=%zu relative_L2=%.9g max_abs=%.9g\n",count,changed,rel,maxerr);
        independent(x,y,r,a.p,as.p,n,k,down,seed,scale2);
    }
    for(int i=0;i<3;++i) {baseline();candidate();}CHECK(cudaStreamSynchronize(stream));
    std::array<float,5> bt{},ct{},qt{},at{},sf{},co{};
    for(int t=0;t<5;++t) {
        if(t&1) {ct[t]=event_time(candidate,stream);bt[t]=event_time(baseline,stream);}
        else {bt[t]=event_time(baseline,stream);ct[t]=event_time(candidate,stream);}
        qt[t]=event_time(quant,stream);at[t]=event_time(astage,stream);
        sf[t]=event_time(bstage,stream);co[t]=event_time(core,stream);
    }
    auto median=[](auto x){std::sort(x.begin(),x.end());return x[2];};
    Result result{median(bt),median(ct),median(qt),median(at),median(sf),median(co)};
    std::printf("timing projection=%d skew=%d baseline_inclusive_ms=%.6f cutlass_inclusive_ms=%.6f speedup=%.6f quant_ms=%.6f A_gather_SFA_ms=%.6f\n",
        projection,skew,result.baseline,result.cutlass,result.baseline/result.cutlass,result.quant,result.stage);
    std::printf("components projection=%d skew=%d SFB_pack_ms=%.6f CUTLASS_core_with_metadata_ms=%.6f\n",
        projection,skew,result.sfb,result.core);
    wt.guards();st.guards();wn.guards();sn.guards();sfb.guards();input.guards();a.guards();as.guards();workspace.guards();old.guards();out.guards();
    CHECK(cudaStreamDestroy(stream));return result;
}
#endif
int main(int argc,char** argv) {
    for(bool skew:{false,true}) {
        auto r=routing(4100,skew);require(r.ids.size()==32800,"4100 top8 routes");
        for(int e=0;e<144;++e) require(r.offsets[e+1]>r.offsets[e],"full144 active experts");
    }
    if(argc==2 && std::strcmp(argv[1],"--host-test")==0) {std::puts("PASS host4100 top8 full144 routing");return 0;}
#ifndef ATLAS_HOST_ONLY
    bool components=argc==2 && std::strcmp(argv[1],"--components")==0;
    bool uniform=argc==2 && std::strcmp(argv[1],"--uniform")==0;
    require(components || uniform || (argc==2 && std::strcmp(argv[1],"--run")==0),"explicit --run, --uniform or --components required");
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));
    require(free>=7*GiB,"7GiB free guard retains4GiB beyond3GiB fixture ceiling");
    cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
    require(prop.major==12 && prop.minor==1,"GB10 sm121 only");
    std::printf("device=%s free_MiB=%.3f\n",prop.name,free/1048576.0);
    for(bool skew:{false,true}) {
        auto gate=run_case(skew,0),up=run_case(skew,1),down=run_case(skew,2);
        // All3 projection-inclusive times. The actual gate/up pair reuses its
        // A quant/gather. Remove exactly one measured common preparation.
        float old=gate.baseline+up.baseline+down.baseline-up.quant;
        float candidate=gate.cutlass+up.cutlass+down.cutlass-up.quant-up.stage;
        std::printf("COMBINED skew=%d baseline_ms=%.6f cutlass_ms=%.6f speedup=%.6f peak_MiB=%.3f\n",skew,old,candidate,old/candidate,peak/1048576.0);
        std::printf("COMPONENT_SUM SFB_pack_ms=%.6f CUTLASS_core_with_metadata_ms=%.6f required_saving_for1p5_ms=%.6f\n",
            gate.sfb+up.sfb+down.sfb,gate.core+up.core+down.core,candidate-old/1.5f);
        if(components) {std::puts("DONE uniform-only component diagnostic; prior performance decision unchanged");return 0;}
        if(old/candidate<1.5f) {std::puts("REJECT below1.5x fullthreeprojection threshold");return 3;}
        if(uniform) {std::puts("PASS uniform-only1.5x gate; skew and model qualification pending");return 0;}
    }
    std::puts("PASS performance>=1.5x uniform and skew; model qualification remains pending");return 0;
#else
    return 2;
#endif
}
