// SPDX-License-Identifier: AGPL-3.0-only
// DFlash2 drafter projection microbenchmark: the 7 per-layer projections of
// GLM-5.3-Flash-DFlash2 (hidden 4096, q 32x128, kv 8x128, intermediate 12288)
// at M rows, for one weight format per build:
//   -DFMT_BF16   dense_gemv_bf16_batchm (current drafter path, M <= 8);
//                TC=1: dense_gemv_bf16_tc16 (tensor-core BF16 tier)
//   -DFMT_MXFP8  mxfp8_gemv_tc8/16 (E4M3 + E8M0 per 32; ATLAS_DFLASH_MXFP8)
//   -DFMT_NVFP4  w4a16_gemv_tc8/16 (NVFP4; ATLAS_DFLASH_NVFP4_TC)
// Also the fused q|k|v (N=6144) and gate|up (N=24576) shapes. Weights rotate
// over 6 copies so every launch streams from DRAM. Env: ITERS, M.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#if defined(FMT_BF16)
#include "dense_gemv_bf16_batchm.cu"
#elif defined(FMT_MXFP8)
#include "mxfp8_gemv.cu"
#elif defined(FMT_NVFP4)
#include "w4a16_gemv.cu"
#endif
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct Shape { const char* name; unsigned n, k; };

int main() {
    const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 200;
    const unsigned M = getenv("M") ? atoi(getenv("M")) : 8;
    const int copies = 6;
    Shape shapes[] = {{"q", 4096, 4096}, {"k", 1024, 4096}, {"v", 1024, 4096}, {"o", 4096, 4096},
                      {"gate", 12288, 4096}, {"up", 12288, 4096}, {"down", 4096, 12288},
                      {"qkv fused", 6144, 4096}, {"gate|up fused", 24576, 4096}};
    __nv_bfloat16 *a, *c; CK(cudaMalloc(&a, 32 * 12288 * 2)); CK(cudaMalloc(&c, 32 * 24576 * 2));
    CK(cudaMemset(a, 0x3c, 32 * 12288 * 2));
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    double layer_us = 0, layer_mb = 0;
    for (int si = 0; si < 9; ++si) {
        const Shape& s = shapes[si];
#if defined(FMT_BF16)
        const size_t wbytes = (size_t) s.n * s.k * 2, sbytes = 0;
        const char* fmt = "bf16";
#elif defined(FMT_MXFP8)
        const size_t wbytes = (size_t) s.n * s.k, sbytes = (size_t) s.n * s.k / 32;
        const char* fmt = "mxfp8";
#else
        const size_t wbytes = (size_t) s.n * s.k / 2, sbytes = (size_t) s.n * s.k / 16;
        const char* fmt = "nvfp4";
#endif
        std::vector<void*> w(copies), sc(copies);
        for (int i = 0; i < copies; ++i) {
            CK(cudaMalloc(&w[i], wbytes)); CK(cudaMemset(w[i], 0x11, wbytes));
            if (sbytes) { CK(cudaMalloc(&sc[i], sbytes)); CK(cudaMemset(sc[i], 0x38, sbytes)); }
        }
        auto run = [&](int i) {
            const int j = i % copies;
#if defined(FMT_BF16)
            if (getenv("TC")) dense_gemv_bf16_tc16<<<(s.n + 15) / 16, 256>>>(a, (const __nv_bfloat16*) w[j], c, M, s.n, s.k, s.n);
            else dense_gemv_bf16_batchm<<<(s.n + 3) / 4, 256>>>(a, (const __nv_bfloat16*) w[j], c, M, s.n, s.k, s.n);
#elif defined(FMT_MXFP8)
            if (M <= 8) mxfp8_gemv_tc8<<<(s.n + 15) / 16, 256>>>(a, (const unsigned char*) w[j], (const unsigned char*) sc[j], c, M, s.n, s.k, s.n);
            else mxfp8_gemv_tc16<<<(s.n + 15) / 16, 256>>>(a, (const unsigned char*) w[j], (const unsigned char*) sc[j], c, M, s.n, s.k, s.n);
#else
            if (M <= 8) w4a16_gemv_tc8<<<(s.n + 15) / 16, 256>>>(a, (const unsigned char*) w[j], (const unsigned char*) sc[j], 1.0f, c, M, s.n, s.k);
            else w4a16_gemv_tc16<<<(s.n + 15) / 16, 256>>>(a, (const unsigned char*) w[j], (const unsigned char*) sc[j], 1.0f, c, M, s.n, s.k);
#endif
        };
        for (int i = 0; i < 10; ++i) run(i);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaEventRecord(ea));
        for (int i = 0; i < iters; ++i) run(i);
        CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
        float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
        const double us = ms * 1e3 / iters, mb = (wbytes + sbytes) / 1e6;
        printf("%-6s M=%u %-14s N=%5u K=%5u  %7.1f us  %6.1f MB  %6.1f GB/s\n", fmt, M, s.name, s.n, s.k, us, mb, mb / us * 1e3);
        if (si < 7) { layer_us += us; layer_mb += mb; }
        for (int i = 0; i < copies; ++i) { CK(cudaFree(w[i])); if (sbytes) CK(cudaFree(sc[i])); }
    }
    printf("%-6s M=%u per layer (7 separate projections) %7.1f us %6.1f MB; x5 layers = %.2f ms\n", "", M, layer_us, layer_mb, layer_us * 5 / 1e3);
    return 0;
}
