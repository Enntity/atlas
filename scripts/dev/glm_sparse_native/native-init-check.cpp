// SPDX-License-Identifier: AGPL-3.0-only
#include "atlas-glm-sparse-native.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <initializer_list>
int main() {
    auto rc=cudaFree(nullptr);
    if(rc) { std::printf("{\"stage\":\"context\",\"status\":%d}\n",int(rc)); return 1; }
    if(atlas_glm_sparse_native_version()!=2) return 2;
    for(uint32_t heads : {16u, 32u}) for(int pass=0;pass<2;++pass) {
        size_t before=0,after=0,total=0;
        if(cudaMemGetInfo(&before,&total)) return 3;
        auto start=std::chrono::steady_clock::now();
        int status=atlas_glm_sparse_native_init(heads);
        auto end=std::chrono::steady_clock::now();
        if(cudaMemGetInfo(&after,&total)) return 4;
        double ms=std::chrono::duration<double,std::milli>(end-start).count();
        std::printf("{\"stage\":\"native_init\",\"heads\":%u,\"pass\":%d,\"status\":%d,\"ms\":%.6f,\"free_before\":%zu,\"free_after\":%zu}\n",heads,pass,status,ms,before,after);
        if(status) { std::fprintf(stderr,"%s\n",cudaGetErrorString(cudaError_t(status))); return 5; }
    }
    return 0;
}
