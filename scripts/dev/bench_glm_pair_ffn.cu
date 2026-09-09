// SPDX-License-Identifier: AGPL-3.0-only
// Fixed native-T GLM FFN experiment, not a Model/attention/NCCL/quality test.
// Build (root only): nvcc -O3 --fmad=false -std=c++17
//   -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_pair_ffn.cu
//   scripts/dev/glm_pair_ffn_{route,grouped,activation,shared_t,post}.cu -o BENCH
// Explicit comparison thresholds required; zero/zero is the initial strict probe.
// Run: BENCH --atol 0 --rtol 0 --repeat 0
// Reused-GU down ONLY: append --compare joint-down (zero/zero mandatory).
// This compares Joint dense down against Joint reused-list down, not TwoK5.
// Shared-width ONLY: append --compare joint-shared (zero/zero mandatory).
// BOTH arms use Joint dense down: two generic-T M5 shared chains vs one M10.
// Wider traversal: append --compare owner-batch --owners 3..8 (strict zero/zero).
// Control is qualified JointShared M10 chunks plus K5 tail; candidate M15..M40.
// Example: BENCH --atol 0 --rtol 0 --repeat 0 --compare owner-batch --owners 4
// No kernel source changes. M35/M40 have three router M16 tiles; other inherited
// GEMMs keep M64 row tiling. Max guarded device payload is 156625800 bytes;
// actual counted allocations must still satisfy the unchanged 192MiB cap.
// Timing allowed only AFTER all six restored-input cases pass the explicit gate.
#include "glm_pair_ffn_run.cuh"
#include "glm_pair_ffn_down_map.cuh"
#include <cerrno>
#include <string>
using namespace pair_ffn;
struct Options { double atol,rtol; unsigned repeat; bool reused_down,shared10,wide; unsigned owners=2; };
static double number(const char* s) {
    require(s&&*s&&*s!='-'&&*s!='+'&&*s!=' ',"unsigned finite decimal required");
    char* end=nullptr;errno=0;double n=std::strtod(s,&end);
    require(!errno&&end!=s&&!*end&&std::isfinite(n)&&n>=0,"finite number/end/overflow");return n;
}
static Options options(int argc,char** argv) {
    require(argc==7||argc==9||argc==11,"usage: bench_glm_pair_ffn --atol VALUE --rtol VALUE --repeat 0..100 [--compare joint-down|joint-shared|owner-batch] [--owners 3..8]");
    Options out{};bool a=false,r=false,n=false,c=false,owners=false;
    for(int i=1;i<argc;i+=2) {
        const std::string key=argv[i];
        if(key=="--compare"&&!c) {
            const std::string mode=argv[i+1];
            require(mode=="joint-down"||mode=="joint-shared"||mode=="owner-batch","unknown explicit comparison");
            c=true;out.reused_down=mode=="joint-down";out.shared10=mode=="joint-shared";
            out.wide=mode=="owner-batch";continue;
        }
        if(key=="--owners"&&!owners) {
            const std::string count=argv[i+1];
            require(count.size()==1&&count[0]>='3'&&count[0]<='8',"owners must be a single digit 3..8");
            owners=true;out.owners=unsigned(count[0]-'0');continue;
        }
        const double value=number(argv[i+1]);
        if(key=="--atol"&&!a) {a=true;out.atol=value;}
        else if(key=="--rtol"&&!r) {r=true;out.rtol=value;}
        else if(key=="--repeat"&&!n) {
            require(value<=100&&std::floor(value)==value,"repeat bound/integer");n=true;out.repeat=unsigned(value);
        } else require(false,"unknown/duplicate option");
    }
    require(a&&r&&n,"every bound must be explicit");
    require(owners==out.wide,"owner-batch requires explicit owners; other modes retain two owners");
    require(!(out.reused_down||out.shared10||out.wide)||(out.atol==0&&out.rtol==0),"isolated comparison requires strict zero/zero plus bit equality");
    return out;
}
static void down_map_checks() {
    unsigned index=0,m=0,n=0;
    require(!reused_gu_index(0,0,gu_down_capacity,index),"zero list count");
    require(!reused_gu_index(0,1,0,index),"zero capacity");
    require(!reused_gu_index(0,-1,gu_down_capacity,index),"negative count");
    require(!reused_gu_index(0,2,1,index),"count exceeds capacity");
    require(!reused_gu_index(0,1,gu_down_capacity+1,index),"capacity exceeds fixed allocation");
    std::array<bool,gu_down_capacity*2> visited{};
    for(unsigned wid=0;wid<gu_down_capacity*2;++wid) {
        require(reused_gu_index(wid,gu_down_capacity,gu_down_capacity,index),"full list capacity");
        require(index==wid/2&&reused_gu_tile(index%16,wid&1,m,n),"16-to-32 N mapping");
        const unsigned tile=(index/16)*32+n;
        require(m==0&&n<32&&tile==wid&&!visited[tile],"unique exact down tile");visited[tile]=true;
    }
    require(!reused_gu_index(gu_down_capacity*2,gu_down_capacity,gu_down_capacity,index),"tail CTA refuses before list read");
    require(!reused_gu_tile(16,0,m,n)&&!reused_gu_tile(64,0,m,n)
        &&!reused_gu_tile(0,2,m,n),"reject non-M10/malformed tile");
    std::puts("PASS host shared CTA mapping zero/full/capacity/16-to-32 bounds");
}
template<class T> static double value(T v) { return double(v); }
template<> double value(Bf v) { return f32(v); }
template<class T> static bool metrics(const std::vector<T>& reference,const std::vector<T>& candidate,
                                     const char* name,const Options& o) {
    require(reference.size()==candidate.size()&&!reference.empty(),"metrics extent");
    double max_abs=0,diff2=0,ref2=0,new2=0,dot=0;size_t unequal=0,outside=0;
    for(size_t i=0;i<reference.size();++i) {
        const double a=value(reference[i]),b=value(candidate[i]);require(std::isfinite(a)&&std::isfinite(b),"finite output required");
        const double d=std::abs(a-b);max_abs=std::max(max_abs,d);diff2+=d*d;ref2+=a*a;new2+=b*b;dot+=a*b;
        unequal+=std::memcmp(&reference[i],&candidate[i],sizeof(T))!=0;outside+=d>o.atol+o.rtol*std::abs(a);
    }
    const double relative_rms=ref2?std::sqrt(diff2/ref2):(diff2?INFINITY:0);
    const double cosine=ref2&&new2?dot/std::sqrt(ref2*new2):(ref2==new2?1:0);
    std::printf("metrics=%s elements=%zu unequal=%zu max_abs=%.9g relative_rms=%.9g cosine=%.12g outside=%zu atol=%.9g rtol=%.9g\n",
        name,reference.size(),unequal,max_abs,relative_rms,cosine,outside,o.atol,o.rtol);
    return outside==0;
}
static void rank_equal(const RankSnapshot& a,const RankSnapshot& b) {
    for(const auto* x:{&a.gate,&a.up,&a.down,&a.routed,&b.gate,&b.up,&b.down,&b.routed})
        for(auto v:*x)require(std::isfinite(f32(v)),"finite canonical routed intermediate");
    equal(a.logits,b.logits,"router BF16");equal(a.ids,b.ids,"actual top8 IDs");equal(a.coeff,b.coeff,"actual top8 coefficients");
    equal(a.ap,b.ap,"quantized input bytes");equal(a.as,b.as,"quantized input scales");
    equal(a.gate,b.gate,"canonical routed gate");equal(a.up,b.up,"canonical routed up");
    equal(a.dp,b.dp,"canonical fused clamped SiLU quantized bytes");equal(a.ds,b.ds,"canonical fused SiLU scales");
    equal(a.down,b.down,"canonical native down");equal(a.routed,b.routed,"rank-local unpermute");
}
// Independent finite E4M3 RN-ties-even/satfinite conversion: enumerate the
// positive representable values, not the production CUDA conversion helper.
static double fp8(double x) {
    require(std::isfinite(x),"finite shared host conversion");
    const double a=std::min(std::abs(x),448.0);
    unsigned lo=0,hi=126;
    while(lo<hi) {const unsigned mid=(lo+hi)/2;if(e4m3(mid)<a)lo=mid+1;else hi=mid;}
    unsigned nearest=lo;
    if(lo) {
        const double left=a-e4m3(lo-1),right=e4m3(lo)-a;
        if(left<right||(left==right&&((lo-1)&1)==0))nearest=lo-1;
    }
    return std::copysign(e4m3(nearest),x);
}
static void shared_oracle(Fixture& f,const SharedSnapshot& s,unsigned rank) {
    require(s.rows.size()==f.rows&&s.input.size()==f.rows*H&&s.gate.size()==f.rows*I
        &&s.up.size()==f.rows*I&&s.activated.size()==f.rows*I&&s.down.size()==f.rows*H,"full shared snapshot extents");
    equal(s.input,f.host_input,"actual shared input canonical owner rows");
    for(unsigned row=0;row<f.rows;++row)require(s.rows[row]==row,"canonical five-row owner mapping");
    for(const auto* a:{&s.input,&s.gate,&s.up,&s.activated,&s.down})
        for(auto v:*a)require(std::isfinite(f32(v)),"finite full shared intermediate");
    // Sample every owner's EVERY row, including rows32..39 for M35/M40.
    // Gate/up/down use actual input (down uses captured post-SiLU). The oracle
    // covers full K at N boundaries; it is not an exhaustive full-weight proof.
    for(unsigned p=0;p<3;++p) {
        const unsigned k=p==2?I:H,n=p==2?H:I,salt=100+p;
        const auto& input=p==2?s.activated:s.input;
        const auto& output=p==0?s.gate:(p==1?s.up:s.down);
        for(unsigned row=0;row<f.rows;++row)for(unsigned col:{0u,127u,128u,n-1}) {
            double sum=0;
            for(unsigned j=0;j<k;++j) {
                const unsigned packed_value=code(size_t(col)*(k/2)+j/2,salt);
                const double weight=e2m1((packed_value>>(4*(j%2)))&15)
                    *e4m3(scale_code(size_t(col)*(k/16)+j/16,salt))*tensor_scale(salt);
                sum+=fp8(f32(input[size_t(row)*k+j]))*fp8(weight);
            }
            const Bf expected=bf(float(sum)),got=output[size_t(row)*n+col];
            if(f32(expected)!=f32(got)) {
                std::fprintf(stderr,"FAIL shared host dot rank=%u owner=%u row=%u projection=%u col=%u expected=%g got=%g\n",
                    rank,row/5,row%5,p,col,f32(expected),f32(got));std::exit(2);
            }
        }
    }
}
static void shared_equal(const SharedSnapshot& a,const SharedSnapshot& b,unsigned rank) {
    equal(a.rows,b.rows,"shared row map");equal(a.input,b.input,"shared input");
    const std::array<const std::vector<Bf>*,4> left={&a.gate,&a.up,&a.activated,&a.down};
    const std::array<const std::vector<Bf>*,4> right={&b.gate,&b.up,&b.activated,&b.down};
    const char* names[]={"gate-preactivation","up-preactivation","post-SiLU","down"};
    for(unsigned p=0;p<4;++p) {
        const unsigned cols=p==3?H:I;
        require(left[p]->size()==a.rows.size()*cols&&right[p]->size()==a.rows.size()*cols,"shared exact tensor extents");
        for(size_t i=0;i<left[p]->size();++i)if(std::memcmp(&(*left[p])[i],&(*right[p])[i],sizeof(Bf))) {
            std::fprintf(stderr,"FAIL shared exact stage=%s rank=%u owner=%zu row=%zu col=%zu old=%g new=%g\n",
                names[p],rank,i/cols/5,i/cols%5,i%cols,f32((*left[p])[i]),f32((*right[p])[i]));std::exit(2);
        }
        if(a.rows.size()==R)std::printf("PASS shared exact stage=%s rank=%u owners=0,1 elements=%zu unequal=0\n",names[p],rank,left[p]->size());
        else std::printf("PASS shared exact stage=%s rank=%u owners=%zu elements=%zu unequal=0\n",names[p],rank,a.rows.size()/5,left[p]->size());
    }
}
static void independent_post(Fixture& f,const Result& result) {
    // Host oracle for actual BF16 rank sum, replicated shared-once and FP32 mHC.
    // This verifies the control fused and Joint unfused tails independently.
    for(unsigned t=0;t<f.rows;++t)for(unsigned h=0;h<H;++h) {
        const size_t x=t*H+h;
        const Bf reduced=bf(f32(result.rank[0].routed[x])+f32(result.rank[1].routed[x]));
        const Bf blended=bf(f32(reduced)+f32(result.shared[x]));
        for(unsigned j=0;j<HC;++j) {
            float expected=f.host_post[t*HC+j]*f32(blended);
            for(unsigned i=0;i<HC;++i)expected+=f.host_comb[t*HC*HC+i*HC+j]*f.host_residual[(t*HC+i)*H+h];
            require(result.highway[(t*HC+j)*H+h]==expected,"independent BF16 rank-add/shared/mHC oracle");
        }
    }
    for(unsigned rank=0;rank<2;++rank)for(unsigned t=0;t<f.rows;++t)for(unsigned e=0;e<E;++e) {
        float acc=0;
        for(unsigned k=0;k<64;++k)acc+=f32(f.host_input[t*H+k])*float(int((e*7+k*3)%17)-8)/128;
        require(f32(result.rank[rank].logits[t*E+e])==f32(bf(acc)),"independent sparse-weight full router oracle");
    }
}
static float timed(Fixture& f,bool joint,bool vector,unsigned repeats,bool reused_down=false,bool shared10=false,unsigned chunk_rows=0) {
    cudaEvent_t begin,end;PCHECK(cudaEventCreate(&begin));PCHECK(cudaEventCreate(&end));
    // This is a serialized two-rank arithmetic surrogate on ONE device, no NCCL.
    PCHECK(cudaEventRecord(begin,f.stream));
    for(unsigned i=0;i<repeats;++i)(void)run(f,joint,vector,false,reused_down,shared10,false,chunk_rows);
    PCHECK(cudaEventRecord(end,f.stream));PCHECK(cudaEventSynchronize(end));float ms=0;
    PCHECK(cudaEventElapsedTime(&ms,begin,end));PCHECK(cudaEventDestroy(begin));PCHECK(cudaEventDestroy(end));
    return ms/repeats;
}
int main(int argc,char** argv) {
    const Options o=options(argc,argv);
    const bool old_joint=o.reused_down||o.shared10||o.wide;
    const bool shared_checks=o.shared10||o.wide;
    if(shared_checks)require(fp8(1.0625)==1&&fp8(1.1875)==1.25&&fp8(1.0/1024)==0
        &&fp8(3.0/1024)==1.0/256&&fp8(-900)==-448,"host FP8 tie/subnormal/saturation controls");
    if(o.reused_down)down_map_checks();
    require(fits(1,4,0)&&!fits(std::numeric_limits<size_t>::max(),8,0)&&!fits(1,4,limit),"allocation accounting negatives");
    Fixture f(o.owners*5);std::printf("device_live=%zu device_peak=%zu cap=%zu H=%u I=%u experts=%u topk=%u rows=%u\n",live,peak,limit,H,I,E,K,f.rows);
    std::printf("scope=router-to-mHC arithmetic; actual NCCL/attention/token-quality excluded; no timed D2H\n");
    if(o.wide)std::printf("comparison=JointShared-M10-chunks-plus-K5-tail-vs-M%u; owners=%u; BOTH genericT-shared+dense-M64-down; strict every-row intermediate/post equality; no serving/NCCL claim\n",f.rows,o.owners);
    else if(o.reused_down)std::puts("comparison=Joint-dense-down-vs-Joint-reused-GU-down; identical M10 route/GU/shared/post; down CTAs9216->2560; no extra builder/allocation; fixed M64 arithmetic");
    else if(o.shared10)std::puts("comparison=Joint-shared-two-genericT-M5-vs-one-genericT-M10; BOTH Joint dense-down; same route/GU/down/post; full shared intermediate bits+sampled host dots+owner-row maps; no extra device allocation");
    else std::printf("control=twoM5 grouped denseGU scalar-or-vector prequantdown shared-generic-T FUSED_MOE_HC1; joint=M10 compact-fusedGU+two-genericT-K5-shared; norm_topk=1 route_scale=1\n");
    bool numerical=true;
    // Every case uses eight distinct full-size tensors, ten unique-per-token routes;
    // mixed boundary IDs and all-local/all-remote masks, normal/reversed owners.
    for(unsigned test=0;test<3;++test)for(unsigned reverse=0;reverse<2;++reverse) {
        f.w.select(test);f.inputs(reverse!=0);
        for(bool vector:{false,true}) {
            std::printf("case=%u reverse=%u vector=%u\n",test,reverse,unsigned(vector));
            auto old=run(f,old_joint,vector,true,false,o.wide,shared_checks,o.wide?10:0);
            auto joint=run(f,true,vector,true,o.reused_down,shared_checks,shared_checks);
            for(unsigned rank=0;rank<2;++rank)rank_equal(old.rank[rank],joint.rank[rank]);
            independent_post(f,old);independent_post(f,joint);
            if(o.wide)std::printf("PASS every-row router/post oracle owners=%u rows=0..%u router_tiles=%u including_rows32plus=%u\n",
                o.owners,f.rows-1,(f.rows+15)/16,unsigned(f.rows>32));
            if(shared_checks) {
                for(unsigned rank=0;rank<2;++rank) {
                    shared_oracle(f,old.shared_rank[rank],rank);shared_oracle(f,joint.shared_rank[rank],rank);
                    shared_equal(old.shared_rank[rank],joint.shared_rank[rank],rank);
                }
                shared_equal(old.shared_rank[0],old.shared_rank[1],1);
                shared_equal(joint.shared_rank[0],joint.shared_rank[1],1);
                equal(joint.shared_rank[1].down,joint.shared,"final shared captures actual last replicated rank");
            }
            if(old_joint) {
                equal(old.shared,joint.shared,"isolated Joint exact shared");
                equal(old.highway,joint.highway,"isolated Joint exact final mHC");
            }
            numerical=metrics(old.shared,joint.shared,
                o.wide?"shared-M10-chunks-vs-wide":(o.shared10?"shared-Joint-genericT-M5-vs-M10":(o.reused_down?"shared-Joint-dense-vs-reused":"shared-genericT-control-vs-joint")),o)&&numerical;
            numerical=metrics(old.highway,joint.highway,"complete-mHC",o)&&numerical;
            // A repeated restored-input run must not depend on dead scratch contents.
            auto again=run(f,true,vector,true,o.reused_down,shared_checks,shared_checks);
            if(old_joint)for(unsigned rank=0;rank<2;++rank)rank_equal(joint.rank[rank],again.rank[rank]);
            if(shared_checks)for(unsigned rank=0;rank<2;++rank)shared_equal(joint.shared_rank[rank],again.shared_rank[rank],rank);
            equal(joint.shared,again.shared,"restored-input shared repeat");equal(joint.highway,again.highway,"restored-input mHC repeat");
        }
    }
    require(numerical,"explicit shared/final tolerance gate failed; no timing claim");
    if(o.repeat) {
        f.w.select(0);f.inputs(false);
        for(bool vector:{false,true}) {
            for(unsigned i=0;i<3;++i){(void)run(f,old_joint,vector,false,false,o.wide,false,o.wide?10:0);(void)run(f,true,vector,false,o.reused_down,shared_checks);}
            for(unsigned order=0;order<2;++order) {
                float old=0,joint=0;
                if(!order){old=timed(f,old_joint,vector,o.repeat,false,o.wide,o.wide?10:0);joint=timed(f,true,vector,o.repeat,o.reused_down,shared_checks);}
                else {joint=timed(f,true,vector,o.repeat,o.reused_down,shared_checks);old=timed(f,old_joint,vector,o.repeat,false,o.wide,o.wide?10:0);}
                if(o.wide)std::printf("single_gpu_serialized_two_rank_arithmetic comparison=M10-chunks-vs-M%u owners=%u vector=%u order=%u repeats=%u chunks_ms=%.6f wide_ms=%.6f ratio=%.6f BOTH_dense_down=1 timed_D2H=0\n",
                    f.rows,o.owners,unsigned(vector),order,o.repeat,old,joint,old/joint);
                else if(o.reused_down)std::printf("single_gpu_serialized_two_rank_arithmetic comparison=Joint-dense-vs-reused-down vector=%u order=%u repeats=%u dense_ms=%.6f reused_ms=%.6f ratio=%.6f timed_D2H=0\n",
                    unsigned(vector),order,o.repeat,old,joint,old/joint);
                else if(o.shared10)std::printf("single_gpu_serialized_two_rank_arithmetic comparison=Joint-shared-M5-vs-M10 vector=%u order=%u repeats=%u sharedM5_ms=%.6f sharedM10_ms=%.6f ratio=%.6f BOTH_Joint_dense_down=1 timed_D2H=0\n",
                    unsigned(vector),order,o.repeat,old,joint,old/joint);
                else std::printf("single_gpu_serialized_two_rank_arithmetic vector=%u order=%u repeats=%u twoM5_ms=%.6f M10_ms=%.6f ratio=%.6f collective_model_only=2x40960_vs_1x81920 timed_D2H=0\n",
                    unsigned(vector),order,o.repeat,old,joint,old/joint);
            }
        }
    }
    PCHECK(cudaStreamSynchronize(f.stream));f.guards();
    std::puts("PASS fixed-input FFN arithmetic gates; not full-model correctness or distributed throughput");
}
