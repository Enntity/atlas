// SPDX-License-Identifier: AGPL-3.0-only
// Microbenchmark: exl3_mhc_pre_norm_split vs hoisted-load variant (bit-exact check).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "exl3_mhc.cu"
#include "v2.cuh"
#include "v3.cuh"
#include "v4.cuh"
__global__ void empty_k() {}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static float frand() {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((rng >> 40) / float(1 << 24)) * 2.0f - 1.0f;
}

int main() {
    const unsigned hidden = 4096, width = 4 * hidden; const unsigned iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 20; const unsigned splits = 8, rsplits = 8;
    const int layers = 90, reps = 20;
    std::vector<float> hw(width), hfn(24 * width), hscale(3), hbase(24);
    std::vector<__nv_bfloat16> hnw(hidden);
    for (auto& x : hw) x = frand() * 4.0f;
    for (auto& x : hfn) x = frand() * 0.02f;
    for (auto& x : hscale) x = 0.5f + frand() * 0.4f;
    for (auto& x : hbase) x = frand();
    for (auto& x : hnw) x = __float2bfloat16(1.0f + frand() * 0.5f);
    float *w, *fn, *sc, *bs, *post, *comb;
    __nv_bfloat16 *nw, *normed;
    CK(cudaMalloc(&w, width * 4 * layers)); CK(cudaMalloc(&fn, hfn.size() * 4 * layers));
    CK(cudaMalloc(&sc, 12)); CK(cudaMalloc(&bs, 96)); CK(cudaMalloc(&post, 16)); CK(cudaMalloc(&comb, 64));
    CK(cudaMalloc(&nw, hidden * 2)); CK(cudaMalloc(&normed, hidden * 2 * layers));
    for (int l = 0; l < layers; ++l) CK(cudaMemcpy(fn + hfn.size() * l, hfn.data(), hfn.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sc, hscale.data(), 12, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bs, hbase.data(), 96, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(nw, hnw.data(), hidden * 2, cudaMemcpyHostToDevice));
    auto reset = [&]() { for (int l = 0; l < layers; ++l) CK(cudaMemcpy(w + width * l, hw.data(), width * 4, cudaMemcpyHostToDevice)); };
    auto partials = [&](int l) {
        exl3_mhc_pre_partials<<<dim3(25, splits, 1), 256>>>(w + width * l, fn + hfn.size() * l, normed + hidden * l, hidden, splits);
    };
    auto finish = [&](int v, int l) {
        if (v == 0)
            exl3_mhc_pre_norm_split<<<1, 256>>>(w + width * l, fn, sc, bs, nw, normed + hidden * l, post, comb, hidden, iters,
                                                1e-6f, 1e-6f, 1e-6f, 2.0f, splits, rsplits);
        else if (v == 2)
            exl3_mhc_pre_norm_split_v4<<<1, 256>>>(w + width * l, fn, sc, bs, nw, normed + hidden * l, post, comb, hidden, iters,
                                                   1e-6f, 1e-6f, 1e-6f, 2.0f, splits, rsplits);
        else
            exl3_mhc_pre_norm_split_v2<<<1, 256>>>(w + width * l, fn, sc, bs, nw, normed + hidden * l, post, comb, hidden, iters,
                                                   1e-6f, 1e-6f, 1e-6f, 2.0f, splits, rsplits);
    };
    std::vector<float> rw(width), rpost(4), rcomb(16), xw(width), xpost(4), xcomb(16);
    std::vector<__nv_bfloat16> rn(hidden), xn(hidden);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float base_ms = 0;
    for (int v = -1; v < 3; ++v) {
        reset();
        if (v >= 0 && v < 2) {
            partials(0); finish(v, 0); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            auto& W = v ? xw : rw; auto& N = v ? xn : rn; auto& P = v ? xpost : rpost; auto& C = v ? xcomb : rcomb;
            CK(cudaMemcpy(W.data(), w, width * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(N.data(), normed, hidden * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(P.data(), post, 16, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(C.data(), comb, 64, cudaMemcpyDeviceToHost));
        }
        CK(cudaEventRecord(a));
        for (int r = 0; r < reps; ++r)
            for (int l = 0; l < layers; ++l) { partials(l); if (v >= 0) finish(v, l); }
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const float us = ms * 1e3f / (reps * layers);
        if (v < 0) { base_ms = us; printf("partials only        %7.2f us/call\n", us); continue; }
        printf("%s finish  %7.2f us/call (incl partials %7.2f)\n", v == 2 ? "v4  " : v ? "v2  " : "orig", us - base_ms, us);
    }
    const bool same = !memcmp(rw.data(), xw.data(), width * 4) && !memcmp(rn.data(), xn.data(), hidden * 2) &&
                      !memcmp(rpost.data(), xpost.data(), 16) && !memcmp(rcomb.data(), xcomb.data(), 64);
    {
        unsigned long long* td; CK(cudaMalloc(&td, 16 * 8)); unsigned long long th[16];
        double acc[9] = {};
        for (int r = 0; r < 50; ++r) {
            partials(r % layers);
            exl3_mhc_pre_norm_split_v3<<<1, 256>>>(w + width * (r % layers), fn, sc, bs, nw, normed + hidden * (r % layers), post, comb, hidden, iters, 1e-6f, 1e-6f, 1e-6f, 2.0f, splits, rsplits, td);
            CK(cudaDeviceSynchronize()); CK(cudaMemcpy(th, td, 16 * 8, cudaMemcpyDeviceToHost));
            for (int i = 1; i < 9; ++i) acc[i] += (th[i] - th[i - 1]) / 50.0;
        }
        const char* nm[] = {"", "entry loads", "fold", "inv/mixes/pre", "sinkhorn+sync?", "collapse", "sqsum", "reduce/inv", "normalize"};
        for (int i = 1; i < 9; ++i) printf("  stage %-16s %7.0f ns\n", nm[i], acc[i]);
        CK(cudaEventRecord(a));
        for (int r = 0; r < reps * layers; ++r) empty_k<<<1, 256>>>();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        printf("empty launch %7.2f us\n", ms * 1e3f / (reps * layers));
    }
    printf("v2 vs orig: %s\n", same ? "BIT-EXACT" : "MISMATCH");
    return 0;
}
