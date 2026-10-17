// SPDX-License-Identifier: AGPL-3.0-only
// Rounded site: exl3_mhc_pre_norm(unrounded_mix=0) vs partials_rounded + pre_norm_split.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "exl3_mhc.cu"
#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("%d %s\n", __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)
static unsigned long long rng = 0x1234567ull;
static float frand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return ((rng >> 40) / float(1 << 24)) * 2.0f - 1.0f; }
int main() {
    const unsigned hidden = 4096, width = 4 * hidden, iters = 20, rs = 48;
    std::vector<float> hw(width), hfn(24 * width), hs(3), hb(24);
    std::vector<__nv_bfloat16> hnw(hidden);
    for (auto& x : hw) x = frand() * 4.0f;
    for (auto& x : hfn) x = frand() * 0.02f;
    for (auto& x : hs) x = 0.5f + frand() * 0.4f;
    for (auto& x : hb) x = frand();
    for (auto& x : hnw) x = __float2bfloat16(1.0f + frand() * 0.5f);
    float *w, *fn, *sc, *bs, *post, *comb; __nv_bfloat16 *nw, *normed;
    CK(cudaMalloc(&w, width * 4)); CK(cudaMalloc(&fn, hfn.size() * 4)); CK(cudaMalloc(&sc, 12)); CK(cudaMalloc(&bs, 96));
    CK(cudaMalloc(&post, 16)); CK(cudaMalloc(&comb, 64)); CK(cudaMalloc(&nw, hidden * 2)); CK(cudaMalloc(&normed, hidden * 2));
    CK(cudaMemcpy(fn, hfn.data(), hfn.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc, hs.data(), 12, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bs, hb.data(), 96, cudaMemcpyHostToDevice)); CK(cudaMemcpy(nw, hnw.data(), hidden * 2, cudaMemcpyHostToDevice));
    std::vector<float> W[2], P[2], C[2]; std::vector<__nv_bfloat16> N[2];
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int v = 0; v < 2; ++v) {
        auto run = [&]() {
            if (v == 0)
                exl3_mhc_pre_norm<<<1, 256>>>(w, fn, sc, bs, nw, normed, post, comb, hidden, iters, 1e-6f, 1e-6f, 1e-6f, 2.0f, 0, rs);
            else {
                exl3_mhc_pre_partials_rounded<<<dim3(rs, 1), 32>>>(w, fn, normed, hidden, rs);
                exl3_mhc_pre_norm_split<<<1, 256>>>(w, fn, sc, bs, nw, normed, post, comb, hidden, iters, 1e-6f, 1e-6f, 1e-6f, 2.0f, rs, rs);
            }
        };
        CK(cudaMemcpy(w, hw.data(), width * 4, cudaMemcpyHostToDevice));
        run(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        W[v].resize(width); P[v].resize(4); C[v].resize(16); N[v].resize(hidden);
        CK(cudaMemcpy(W[v].data(), w, width * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(P[v].data(), post, 16, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C[v].data(), comb, 64, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(N[v].data(), normed, hidden * 2, cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(a)); for (int r = 0; r < 200; ++r) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); printf("%s %8.2f us/call\n", v ? "split  " : "rounded", ms * 1e3f / 200);
    }
    printf("highway %s normed %s post %s comb %s\n", memcmp(W[0].data(), W[1].data(), width * 4) ? "DIFF" : "same",
           memcmp(N[0].data(), N[1].data(), hidden * 2) ? "DIFF" : "same", memcmp(P[0].data(), P[1].data(), 16) ? "DIFF" : "same",
           memcmp(C[0].data(), C[1].data(), 64) ? "DIFF" : "same");
}
