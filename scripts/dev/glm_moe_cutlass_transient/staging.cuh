// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Byte-only layout changes. Production quantizer runs before this kernel.
__global__ void stage_a(const unsigned char* packed, const unsigned char* scales,
    const int* ids, const int* offsets, unsigned char* const* dst,
    unsigned char* const* sf, int k) {
    int e=blockIdx.z, row=blockIdx.x, g=threadIdx.x+blockIdx.y*blockDim.x;
    int me=offsets[e+1]-offsets[e];
    if(row>=me || g>=k/16) return;
    int sorted=offsets[e]+row, token=ids?ids[sorted]:sorted;
    auto layout=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(me,1,k,1));
    sf[e][layout(row,g*16,0)]=scales[size_t(token)*(k/16)+g];
    #pragma unroll
    for(int j=0;j<8;++j) dst[e][size_t(row)*(k/2)+g*8+j]=packed[size_t(token)*(k/2)+g*8+j];
}
__global__ void stage_sfb(const unsigned char* native_scales, unsigned char* sfb,
    int n,int k,size_t stride) {
    int e=blockIdx.z;
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(n)*(k/16)) return;
    int col=i/(k/16),g=i%(k/16);
    auto layout=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1,n,k,1));
    sfb[size_t(e)*stride+layout(col,g*16,0)]=native_scales[size_t(e)*n*(k/16)+i];
}
__global__ void native_from_t(const unsigned char* source,unsigned char* dest,
    int n,int width) {
    int e=blockIdx.z;
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(n)*width) dest[size_t(e)*n*width+i]=source[size_t(e)*n*width+(i%width)*n+i/width];
}
__global__ void fill_bf16(__nv_bfloat16* a,size_t count,unsigned seed) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        unsigned x=unsigned(i)^seed;x^=x>>16;x*=0x7feb352d;x^=x>>15;
        a[i]=__float2bfloat16(float(int(x%8193)-4096)/1024.0f);
    }
}

// Exhaustive byte oracle over all active A bytes/scales and native B bytes/SFB.
// This runs outside timing and checks the exact bytes handed to the collective.
__global__ void verify_operands(const unsigned char* a,const unsigned char* as,
    const unsigned char* wt,const unsigned char* st,const unsigned char* wn,
    const unsigned char* sn,const unsigned char* sfb,unsigned char* const* ap,
    unsigned char* const* asp,const int* offsets,const int* ids,int n,int k,
    size_t sfb_stride,unsigned* errors) {
    int e=blockIdx.z;
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    int me=offsets[e+1]-offsets[e];
    auto la=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(me,1,k,1));
    auto lb=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1,n,k,1));
    if(i<size_t(me)*k/2) {
        size_t row=i/(k/2),col=i%(k/2),sorted=offsets[e]+row,tok=ids?ids[sorted]:sorted;
        if(ap[e][i]!=a[tok*k/2+col]) atomicAdd(errors,1u);
    }
    if(i<size_t(me)*k/16) {
        size_t row=i/(k/16),g=i%(k/16),sorted=offsets[e]+row,tok=ids?ids[sorted]:sorted;
        if(asp[e][la(row,g*16,0)]!=as[tok*k/16+g]) atomicAdd(errors,1u);
    }
    if(i<size_t(n)*k/2) {
        size_t col=i/(k/2),g=i%(k/2),base=size_t(e)*n*k/2;
        if(wn[base+i]!=wt[base+g*n+col]) atomicAdd(errors,1u);
    }
    if(i<size_t(n)*k/16) {
        size_t col=i/(k/16),g=i%(k/16),base=size_t(e)*n*k/16;
        unsigned char expected=st[base+g*n+col];
        if(sn[base+i]!=expected || sfb[size_t(e)*sfb_stride+lb(col,g*16,0)]!=expected) atomicAdd(errors,1u);
    }
}

// All offsets and layouts are planned before metadata or kernels touch ws.
// Lifetime is the entire case, including asynchronous launches and timing.
struct StagePlan {
    GroupedAPrep p;
    size_t cap, cursor=0, sfb_stride;
    unsigned char* ws;
    std::vector<const ElementA::DataType*> a_ptrs;
    std::vector<const ElementSF*> sf_ptrs;
    std::vector<StrideA> strides;
    std::vector<LayoutSFA> layouts;
    std::vector<std::pair<size_t,size_t>> regions;
    size_t take(size_t bytes) {
        require(cursor<=cap && bytes<=cap-cursor,"workspace staging capacity");
        size_t old=cursor;cursor=align_up_(cursor+bytes,256);
        require(cursor<=cap,"aligned workspace capacity");
        regions.push_back({old,bytes});return old;
    }
    template<class T> T* reserve(size_t count) {return reinterpret_cast<T*>(ws+take(count*sizeof(T)));}
    StagePlan(unsigned char* w,size_t c,const Routing& route,int n,int k):cap(c),ws(w) {
        sfb_stride=size_t(size(filter_zeros(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1,n,k,1)))));
        for(int e=0;e<144;++e) {
            int me=route.offsets[e+1]-route.offsets[e];
            require(me>0,"fixture must activate every local expert");
            p.eidx.push_back(e);p.ms.push_back(route.offsets[e]);p.me.push_back(me);
            p.host_shapes.push_back(ProblemShape{me,n,k});
            a_ptrs.push_back(reinterpret_cast<const ElementA::DataType*>(reserve<unsigned char>(size_t(me)*k/2)));
            auto layout=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(me,n,k,1));
            layouts.push_back(layout);
            sf_ptrs.push_back(reinterpret_cast<const ElementSF*>(reserve<unsigned char>(size_t(size(filter_zeros(layout))))));
            strides.push_back(cutlass::make_cute_packed_stride(StrideA{},{me,k,1}));
        }
        p.G=144;
        p.dShapes=reserve<ProblemShape>(144);p.dA=reserve<const ElementA::DataType*>(144);
        p.dSFA=reserve<const ElementSF*>(144);p.dsA=reserve<StrideA>(144);p.dlSFA=reserve<LayoutSFA>(144);
        p.cursor=cursor;
    }
    void upload(cudaStream_t stream) {
        CHECK(cudaMemcpyAsync(p.dShapes,p.host_shapes.data(),144*sizeof(ProblemShape),cudaMemcpyHostToDevice,stream));
        CHECK(cudaMemcpyAsync(p.dA,a_ptrs.data(),144*sizeof(void*),cudaMemcpyHostToDevice,stream));
        CHECK(cudaMemcpyAsync(p.dSFA,sf_ptrs.data(),144*sizeof(void*),cudaMemcpyHostToDevice,stream));
        CHECK(cudaMemcpyAsync(p.dsA,strides.data(),144*sizeof(StrideA),cudaMemcpyHostToDevice,stream));
        CHECK(cudaMemcpyAsync(p.dlSFA,layouts.data(),144*sizeof(LayoutSFA),cudaMemcpyHostToDevice,stream));
    }
    void stage(const unsigned char* a,const unsigned char* as,const int* ids,
        const int* offsets,int max_rows,int k,cudaStream_t stream) {
        stage_a<<<dim3(max_rows,(k/16+255)/256,144),256,0,stream>>>(a,as,ids,offsets,
            (unsigned char* const*)p.dA,(unsigned char* const*)p.dSFA,k);
        CHECK(cudaGetLastError());
    }
};
