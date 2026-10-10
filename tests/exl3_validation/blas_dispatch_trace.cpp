// SPDX-License-Identifier: AGPL-3.0-only
// Diagnostic interposition only. No CUDA math, configuration, or tensor edits.
#include <dlfcn.h>
#include <cstdint>
#include <cstddef>

namespace {
thread_local bool active = false;
thread_local std::uint32_t counts[2] = {};
thread_local const char* paths[2] = {};
thread_local int gemm_args[13] = {};
void record(void* symbol, unsigned int index) {
    if (!active) return;
    ++counts[index];
    Dl_info info{};
    if (dladdr(symbol, &info)) paths[index] = info.dli_fname;
}
}
extern "C" void atlas_blas_trace_begin() {
    counts[0] = counts[1] = 0; paths[0] = paths[1] = nullptr; active = true;
}
extern "C" std::uint64_t atlas_blas_trace_end() {
    active = false;
    return counts[0] | (static_cast<std::uint64_t>(counts[1]) << 32);
}
extern "C" const char* atlas_blas_trace_binding(unsigned int index) {
    return index < 2 ? paths[index] : nullptr;
}
extern "C" int atlas_blas_trace_gemm_argument(unsigned int index) {
    return index < 13 ? gemm_args[index] : -1;
}
extern "C" int cublasGemmEx(void* handle, int ta, int tb, int m, int n, int k,
    const void* alpha, const void* a, int at, int lda, const void* b, int bt,
    int ldb, const void* beta, void* out, int ct, int ldc, int compute, int algo) {
    using Fn = int (*)(void*,int,int,int,int,int,const void*,const void*,int,int,
        const void*,int,int,const void*,void*,int,int,int,int);
    static void* symbol = dlsym(RTLD_NEXT, "cublasGemmEx");
    if (!symbol) return 13; // Explicit prerequisite failure, no substitute math.
    record(symbol, 0);
    if (active) {
        const int values[13] = {ta,tb,m,n,k,at,bt,ct,lda,ldb,ldc,compute,algo};
        for (unsigned int i = 0; i < 13; ++i) gemm_args[i] = values[i];
    }
    return reinterpret_cast<Fn>(symbol)(handle,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,
                                      beta,out,ct,ldc,compute,algo);
}
extern "C" int cublasLtMatmul(void* handle, void* desc, const void* alpha,
    const void* a, void* la, const void* b, void* lb, const void* beta,
    const void* c, void* lc, void* out, void* ld, const void* algo,
    void* workspace, std::size_t workspace_bytes, void* stream) {
    using Fn = int (*)(void*,void*,const void*,const void*,void*,const void*,void*,
        const void*,const void*,void*,void*,void*,const void*,void*,std::size_t,void*);
    static void* symbol = dlsym(RTLD_NEXT, "cublasLtMatmul");
    if (!symbol) return 13;
    record(symbol, 1);
    return reinterpret_cast<Fn>(symbol)(handle,desc,alpha,a,la,b,lb,beta,c,lc,out,ld,
                                      algo,workspace,workspace_bytes,stream);
}
