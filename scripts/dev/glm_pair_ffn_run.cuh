// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "glm_pair_ffn_fixture.cuh"
namespace pair_ffn {
template<class T> inline void append(std::vector<T>& dst,const std::vector<T>& src) {
    dst.insert(dst.end(),src.begin(),src.end());
}
template<class T> inline void equal(const std::vector<T>& a,const std::vector<T>& b,const char* name) {
    require(a.size()==b.size(),"comparison extent");
    for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(T))) {
        std::fprintf(stderr,"FAIL exact %s index=%zu\n",name,i);std::exit(2);
    }
}
struct RankSnapshot {
    std::vector<Bf> logits,gate,up,down,routed;
    std::vector<unsigned> ids;std::vector<float> coeff;
    std::vector<unsigned char> ap,as,dp,ds;
};
struct Result {
    std::array<RankSnapshot,2> rank;
    std::vector<Bf> shared; std::vector<float> highway;
};
// Canonicalize actual atomic scatter by token_to_perm; never assume expert-sort order.
template<class T> inline std::vector<T> canonical(const std::vector<T>& src,
        const std::vector<int>& inv,const std::vector<unsigned>& ids,unsigned cols,unsigned rank) {
    std::vector<T> out(inv.size()*cols,T{});
    for(size_t slot=0;slot<inv.size();++slot)if(ids[slot]/144==rank)
        std::copy_n(src.begin()+size_t(inv[slot])*cols,cols,out.begin()+slot*cols);
    return out;
}
inline void capture(Fixture& f,RankSnapshot& out,unsigned rows,unsigned base,unsigned rank) {
    PCHECK(cudaStreamSynchronize(f.stream));auto& b=f.b;
    const auto ids=b.ids.read(rows*K);const auto inv=b.inv.read(rows*K);
    const auto tok=b.tok.read(rows*K),exp=b.exp.read(rows*K),off=b.off.read();
    std::vector<bool> visited(rows*K,false);std::array<unsigned,E> counts{};
    for(unsigned t=0;t<rows;++t) {
        std::array<bool,E> unique{};
        for(unsigned j=0;j<K;++j) {
            const auto slot=t*K+j,e=ids[slot];require(e<E&&!unique[e],"unique actual top8");unique[e]=true;
            require(std::find(f.w.ids.begin(),f.w.ids.end(),e)!=f.w.ids.end(),"actual router selected allocated expert");
            ++counts[e];require(inv[slot]>=0&&unsigned(inv[slot])<rows*K&&!visited[inv[slot]],"sort inverse bijection");
            visited[inv[slot]]=true;require(tok[inv[slot]]==int(t)&&exp[inv[slot]]==int(e),"actual sort mapping");
        }
    }
    unsigned sum=0,local=0;
    for(unsigned e=0;e<E;++e) {
        require(counts[e]<=rows&&off[e]==int(sum),"per-expert padding/offset bound");sum+=counts[e];
        if(counts[e]&&e/144==rank)++local;
    }
    require(sum==rows*K&&off[E]==int(sum),"full expanded extent");
    const auto total=rows==R?b.total.read()[0]:0;
    if(rows==R)require(total==int(local*I/128),"actual compact work count");
    if(total) {
        auto list=b.list.read(size_t(total)*2);std::vector<unsigned> expected;
        for(unsigned e=0;e<E;++e)if(counts[e]&&e/144==rank)for(unsigned n=0;n<I/128;++n){expected.push_back(e);expected.push_back(n);}
        equal(list,expected,"worklist skips remote, one padded M64 per expert");
    }
    auto gate=b.gate.read(rows*K*I),up=b.up.read(rows*K*I),down=b.down.read(rows*K*H);
    for(unsigned row=0;row<rows*K;++row)if(unsigned(exp[row])/144!=rank)for(unsigned h=0;h<H;++h) {
        unsigned short bits;std::memcpy(&bits,&down[size_t(row)*H+h],2);
        require(bits==0xffff,"remote down poison must remain unwritten");
    }
    const auto ap=b.ap.read(rows*H/2),as=b.as.read(rows*H/16),dp=b.dp.read(rows*K*I/2),ds=b.ds.read(rows*K*I/16);
    // Sampled independent dequantized FP4 CPU oracle, actual quantized inputs.
    // Dyadic test weights/scale2 keep the bounded accumulation exactly representable.
    for(unsigned t=0;t<rows;++t)for(unsigned j=0;j<K;++j) {
        const unsigned slot=t*K+j,e=ids[slot];if(e/144!=rank)continue;
        const unsigned w=unsigned(std::find(f.w.ids.begin(),f.w.ids.end(),e)-f.w.ids.begin());
        const unsigned row=unsigned(inv[slot]);
        for(unsigned p=0;p<3;++p)for(unsigned col: {0u,127u,(p==2?H:I)-1}) {
            const unsigned width=p==2?I:H,salt=Weights::seed(p,w);
            double sum_value=0;
            for(unsigned k=0;k<width;++k) {
                const auto& aq=p==2?dp:ap;const auto& aqscale=p==2?ds:as;
                const unsigned arow=p==2?row:t;
                const unsigned a_byte=aq[size_t(arow)*(width/2)+k/2];
                const double av=e2m1((a_byte>>(4*(k%2)))&15)*e4m3(aqscale[size_t(arow)*(width/16)+k/16]);
                const unsigned wb=code(size_t(col)*(width/2)+k/2,salt);
                const double wv=e2m1((wb>>(4*(k%2)))&15)*e4m3(scale_code(size_t(col)*(width/16)+k/16,salt));
                sum_value+=av*wv;
            }
            const Bf expected=bf(float(sum_value*tensor_scale(salt)));
            const auto& actual=p==0?gate:(p==1?up:down);
            const Bf got=actual[size_t(row)*(p==2?H:I)+col];
            require(std::isfinite(f32(got)),"finite sampled routed projection");
            if(f32(got)!=f32(expected)) {
                std::fprintf(stderr,"CPU routed oracle rank=%u token=%u expert=%u p=%u col=%u expected=%g got=%g\n",
                    rank,base+t,e,p,col,f32(expected),f32(got));std::exit(2);
            }
        }
    }
    append(out.logits,b.logits.read(rows*E));append(out.ids,ids);append(out.coeff,b.coeff.read(rows*K));
    append(out.ap,ap);append(out.as,as);
    append(out.gate,canonical(gate,inv,ids,I,rank));append(out.up,canonical(up,inv,ids,I,rank));
    append(out.down,canonical(down,inv,ids,H,rank));
    append(out.dp,canonical(dp,inv,ids,I/2,rank));append(out.ds,canonical(ds,inv,ids,I/16,rank));
    append(out.routed,(rank?f.rank1:f.rank0).read(rows*H,base*H));
}
inline Result run(Fixture& f,bool joint,bool vector,bool diagnostic) {
    Result result;auto& b=f.b;const unsigned width=joint?R:5;
    for(unsigned base=0;base<R;base+=width) {
        for(unsigned rank=0;rank<2;++rank) {
            // Each simulated rank executes its own replicated shared work, as serving does.
            for(unsigned first=0;first<width;first+=5) {
                const Bf* a=f.input.ptr+(base+first)*H;
                Bf* g=b.sg.ptr+first*I;Bf* u=b.su.ptr+first*I;Bf* out=f.shared.ptr+(base+first)*H;
                // Preserve the exact qualified shared-T arithmetic in both modes.
                shared_t(a,f.w.shared(0,true),g,I,H,f.stream);
                shared_t(a,f.w.shared(1,true),u,I,H,f.stream);
                activation(g,u,5,f.stream);
                shared_t(g,f.w.shared(2,true),out,H,I,f.stream);
            }
            if(diagnostic)b.poison(f.stream); // outside timing; no numerical work on local rows depends on poison.
            route(f.input.ptr+base*H,f.w.router.ptr,f.w.bias.ptr,b.logits.ptr,b.ids.ptr,b.coeff.ptr,width,f.stream);
            sort(b.ids.ptr,b.tok.ptr,b.exp.ptr,b.off.ptr,b.inv.ptr,width,f.stream);
            auto g=f.w.tables[rank][0]->view(),u=f.w.tables[rank][1]->view(),d=f.w.tables[rank][2]->view();
            if(joint)work(b.off.ptr,g,b.list.ptr,b.total.ptr,f.stream);
            quant(f.input.ptr+base*H,b.ap.ptr,b.as.ptr,width,H,f.stream);
            gate_up(b.ap.ptr,b.as.ptr,g,u,b.gate.ptr,b.up.ptr,b.off.ptr,b.tok.ptr,b.list.ptr,b.total.ptr,width,vector,f.stream);
            silu_quant(b.gate.ptr,b.up.ptr,b.dp.ptr,b.ds.ptr,width,f.stream);
            down(b.dp.ptr,b.ds.ptr,d,b.down.ptr,b.off.ptr,vector,f.stream);
            unpermute(b.down.ptr,(rank?f.rank1:f.rank0).ptr+base*H,b.inv.ptr,b.ids.ptr,b.coeff.ptr,width,rank,f.stream);
            PCHECK(cudaGetLastError());
            if(diagnostic)capture(f,result.rank[rank],width,base,rank);
        }
        finish(f.rank0.ptr+base*H,f.rank1.ptr+base*H,f.shared.ptr+base*H,f.input.ptr+base*H,
            f.residual.ptr+base*HC*H,f.post.ptr+base*HC,f.comb.ptr+base*HC*HC,f.highway.ptr+base*HC*H,width,joint,f.stream);
        PCHECK(cudaGetLastError());
    }
    if(diagnostic) {
        PCHECK(cudaStreamSynchronize(f.stream));result.shared=f.shared.read();result.highway=f.highway.read();f.guards();
    }
    return result;
}
}
