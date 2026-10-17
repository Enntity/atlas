// SPDX-License-Identifier: AGPL-3.0-only
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "orig.cuh"
#include "v2.cuh"
#include "v3.cuh"
#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("%d %s\n", __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)
static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static float frand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return ((rng >> 40) / float(1 << 24)) * 2.0f - 1.0f; }
int main() {
    const unsigned E = 288, K = 8; const int trials = 2000;
    float *lg, *bs, *w0, *w1, *w2; int *i0, *i1, *i2;
    CK(cudaMalloc(&lg, E * 4)); CK(cudaMalloc(&bs, E * 4)); CK(cudaMalloc(&w0, K * 4)); CK(cudaMalloc(&w1, K * 4));
    CK(cudaMalloc(&i0, K * 4)); CK(cudaMalloc(&w2, K * 4)); CK(cudaMalloc(&i2, K * 4)); CK(cudaMalloc(&i1, K * 4));
    std::vector<float> hl(E), hb(E), a(K), b(K); std::vector<int> ia(K), ib(K);
    int bad = 0, bad3 = 0;
    for (int t = 0; t < trials; ++t) {
        const bool ties = t % 2;
        for (unsigned e = 0; e < E; ++e) {
            hl[e] = ties ? std::round(frand() * 8) / 4 : frand() * 6;
            hb[e] = ties ? std::round(frand() * 2) / 8 : frand() * 0.5f;
        }
        CK(cudaMemcpy(lg, hl.data(), E * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bs, hb.data(), E * 4, cudaMemcpyHostToDevice));
        const unsigned norm = t % 3 != 0;
        exl3_topk_sigmoid_par<<<1, 256>>>(lg, bs, i0, w0, 1, E, K, norm, 2.5f);
        exl3_topk_sigmoid_par_v2<<<1, 256>>>(lg, bs, i1, w1, 1, E, K, norm, 2.5f);
        exl3_topk_sigmoid_par_v3<<<1, 256>>>(lg, bs, i2, w2, 1, E, K, norm, 2.5f);
        { std::vector<float> c(K); std::vector<int> ic(K); CK(cudaMemcpy(c.data(), w2, K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ic.data(), i2, K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a.data(), w0, K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ia.data(), i0, K * 4, cudaMemcpyDeviceToHost)); if (memcmp(a.data(), c.data(), K * 4) || memcmp(ia.data(), ic.data(), K * 4)) ++bad3; }
        CK(cudaMemcpy(a.data(), w0, K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), w1, K * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ia.data(), i0, K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ib.data(), i1, K * 4, cudaMemcpyDeviceToHost));
        if (memcmp(a.data(), b.data(), K * 4) || memcmp(ia.data(), ib.data(), K * 4)) ++bad;
    }
    printf("mismatching trials: v2 %d v3 %d / %d\n", bad, bad3, trials);
    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    for (int v = 0; v < 3; ++v) {
        CK(cudaEventRecord(s));
        for (int r = 0; r < 2000; ++r) {
            if (v == 0) exl3_topk_sigmoid_par<<<1, 256>>>(lg, bs, i0, w0, 1, E, K, 1, 2.5f);
            else if (v == 1) exl3_topk_sigmoid_par_v2<<<1, 256>>>(lg, bs, i1, w1, 1, E, K, 1, 2.5f);
            else exl3_topk_sigmoid_par_v3<<<1, 256>>>(lg, bs, i2, w2, 1, E, K, 1, 2.5f);
        }
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        float ms; CK(cudaEventElapsedTime(&ms, s, e));
        printf("%s %7.2f us/call\n", v == 2 ? "v3  " : v ? "v2  " : "orig", ms * 1e3f / 2000);
    }
}
