// SPDX-License-Identifier: AGPL-3.0-only
// Multi-row routed MoE microbenchmark (speculative verify, r rows x top-8):
// per-slot v6 (weights streamed once per slot) vs expert-grouped v8
// (weights streamed once per unique expert), bit-exact per-slot check,
// unique experts, bytes moved and achieved GB/s. Env: ITERS, POOL (experts
// the rows draw from; smaller = more overlap between rows).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "exl3_coop_gemm.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static inline unsigned long long next() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

struct Proj { int k, n; Exl3Expert* table; };
static Proj make_proj(int k, int n, int experts) {
    Proj p{k, n, nullptr};
    size_t bytes = (size_t) (k / 16) * (n / 16) * 128;
    std::vector<unsigned long long> h(bytes / 8);
    std::vector<Exl3Expert> table(experts);
    for (int e = 0; e < experts; ++e) {
        for (auto& x : h) x = next();
        void* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
        table[e] = Exl3Expert{(unsigned long long) d, 0, 0, 4u, 0u};
    }
    CK(cudaMalloc(&p.table, sizeof(Exl3Expert) * experts));
    CK(cudaMemcpy(p.table, table.data(), sizeof(Exl3Expert) * experts, cudaMemcpyHostToDevice));
    return p;
}

int main() {
    const int experts = 256, topk = 8, slices = 8;
    const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 100;
    const int pool = getenv("POOL") ? atoi(getenv("POOL")) : 256;
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    Proj gate = make_proj(4096, 1024, experts), up = make_proj(4096, 1024, experts), down = make_proj(1024, 4096, experts);
    const int max_slots = 8 * topk;
    float *xf0, *xf1, *xfd; half *xh0, *xh1, *xhd;
    std::vector<float> hf((size_t) max_slots * 4096);
    for (auto& x : hf) x = __half2float(__float2half_rn(((next() >> 40) / float(1 << 24)) * 2.0f - 1.0f));
    std::vector<half> hh(hf.size()); for (size_t i = 0; i < hf.size(); ++i) hh[i] = __float2half_rn(hf[i]);
    for (float** p : {&xf0, &xf1, &xfd}) { CK(cudaMalloc(p, hf.size() * 4)); CK(cudaMemcpy(*p, hf.data(), hf.size() * 4, cudaMemcpyHostToDevice)); }
    for (half** p : {&xh0, &xh1, &xhd}) { CK(cudaMalloc(p, hh.size() * 2)); CK(cudaMemcpy(*p, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice)); }
    float *y0, *y1, *yd; for (float** p : {&y0, &y1, &yd}) CK(cudaMalloc(p, (size_t) max_slots * 4096 * 4));
    int* locks; CK(cudaMalloc(&locks, 2 * max_slots * 32 * 4)); CK(cudaMemset(locks, 0, 2 * max_slots * 32 * 4));
    unsigned* counts; CK(cudaMalloc(&counts, experts * 4));
    int* ids; CK(cudaMalloc(&ids, max_slots * 4));
    int* groups; CK(cudaMalloc(&groups, (1 + 4 * max_slots) * 4));
    void* f6 = (void*) exl3_coop_moe6_thin_b4_k32_n256; void* f8 = (void*) exl3_coop_moe8_multirow_b4_k32_n256;
    const int sm6 = 6 * 5120 + 8192, sm8 = 6 * 5120;
    CK(cudaFuncSetAttribute(f6, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CK(cudaFuncSetAttribute(f8, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int occ6, occ8; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ6, f6, 256, sm6)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, f8, 256, sm8));
    printf("v6 occupancy %d/SM, v8 occupancy %d/SM, pool %d experts\n", occ6, occ8, pool);
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    for (int r : {1, 2, 4, 8}) {
        const int slots = r * topk;
        // Each row draws topk distinct experts from the first `pool` experts.
        std::vector<int> hid(slots); std::vector<unsigned> hc(experts, 0);
        for (int row = 0; row < r; ++row) {
            std::vector<int> pick;
            while ((int) pick.size() < topk) { int e = (int) (next() % pool); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
            for (int j = 0; j < topk; ++j) { hid[row * topk + j] = pick[j]; hc[pick[j]]++; }
        }
        int unique = 0; for (unsigned c : hc) unique += c > 0;
        CK(cudaMemcpy(ids, hid.data(), slots * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(counts, hc.data(), experts * 4, cudaMemcpyHostToDevice));
        auto chunks_of = [&](int k, int n) { int tk = k / 32, total = tk * (n / 256), c = 0; for (int i = 0; i < slices; ++i) { int b = total * i / slices, e = total * (i + 1) / slices; if (e > b) c += (e - 1) / tk - b / tk + 1; } return c; };
        auto v6 = [&](int proj) {
            bool gu = proj == 2; const Proj& A = gu ? gate : down; const Proj& B = gu ? up : down;
            const float* x0 = gu ? xf0 : xfd; const float* x1 = gu ? xf1 : xfd; float* o0 = gu ? y0 : yd; float* o1 = gu ? y1 : yd;
            int k = A.k, n = A.n, e = experts, p = proj, s = slots, sl = slices; unsigned cap = 32;
            int items = p * s * chunks_of(k, n), grid = std::min(items, occ6 * sms);
            void* args[] = {&x0, &x1, &ids, (void*) &A.table, (void*) &B.table, &o0, &o1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
            CK(cudaLaunchCooperativeKernel(f6, grid, 256, args, sm6, 0));
        };
        auto v8 = [&](int proj, bool group) {
            bool gu = proj == 2; const Proj& A = gu ? gate : down; const Proj& B = gu ? up : down;
            const half* x0 = gu ? xh0 : xhd; const half* x1 = gu ? xh1 : xhd; float* o0 = gu ? y0 : yd; float* o1 = gu ? y1 : yd;
            int k = A.k, n = A.n, e = experts, p = proj, s = slots, sl = slices; unsigned cap = 32;
            if (group) exl3_moe_group_slots<<<1, 256>>>(ids, groups, slots, experts);
            int grid = occ8 * sms;
            void* args[] = {&x0, &x1, &groups, (void*) &A.table, (void*) &B.table, &o0, &o1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
            CK(cudaLaunchCooperativeKernel(f8, grid, 256, args, sm8, 0));
        };
        // Bit-exact per-slot check.
        std::vector<float> a0(slots * 1024), a1(slots * 1024), ad(slots * 4096), b0(a0.size()), b1(a1.size()), bd(ad.size());
        v6(2); v6(1); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a0.data(), y0, a0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a1.data(), y1, a1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ad.data(), yd, ad.size() * 4, cudaMemcpyDeviceToHost));
        for (float* p : {y0, y1, yd}) CK(cudaMemset(p, 0xff, (size_t) max_slots * 4096 * 4));
        v8(2, true); v8(1, false); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(b0.data(), y0, b0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b1.data(), y1, b1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bd.data(), yd, bd.size() * 4, cudaMemcpyDeviceToHost));
        bool same = !memcmp(a0.data(), b0.data(), a0.size() * 4) && !memcmp(a1.data(), b1.data(), a1.size() * 4) && !memcmp(ad.data(), bd.data(), ad.size() * 4);
        for (int mode = 0; mode < 2; ++mode) {
            for (int i = 0; i < 5; ++i) { if (mode) { v8(2, true); v8(1, false); } else { v6(2); v6(1); } }
            CK(cudaEventRecord(ea));
            for (int i = 0; i < iters; ++i) { if (mode) { v8(2, true); v8(1, false); } else { v6(2); v6(1); } }
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb)); double us = ms * 1e3 / iters;
            const double mb_slot = 3.0 * slots * 2.097152, mb_uniq = 3.0 * unique * 2.097152;
            double mb = mode ? mb_uniq : mb_slot;
            printf("r=%d slots %2d unique %2d  %-22s %8.1f us  %6.1f MB  %6.1f GB/s%s\n", r, slots, unique,
                   mode ? "v8 grouped (+group)" : "v6 per-slot", us, mb, mb / us * 1e3, mode ? (same ? "  bit-exact YES" : "  bit-exact NO") : "");
        }
    }
    // Dense coop GEMM vs rows (K4096 x N4096, 6-bit, 48 slices): weights are
    // streamed once per launch for all rows; checks the staging-free path.
    {
        const int K = 4096, N = 4096;
        size_t tb = (size_t) (K / 16) * (N / 16) * 16 * 6 * 2;
        std::vector<unsigned long long> ht(tb / 8); for (auto& x : ht) x = next();
        void* trellis; CK(cudaMalloc(&trellis, tb)); CK(cudaMemcpy(trellis, ht.data(), tb, cudaMemcpyHostToDevice));
        half* a; CK(cudaMalloc(&a, 16 * K * 2)); CK(cudaMemcpy(a, hh.data(), 16 * K * 2, cudaMemcpyHostToDevice));
        float* c; CK(cudaMalloc(&c, 16 * N * 4));
        int* lk; CK(cudaMalloc(&lk, 2 * 32 * 4)); CK(cudaMemset(lk, 0, 2 * 32 * 4));
        void* fs[3] = {(void*) exl3_coop_gemm_b6_k32_n128, (void*) exl3_coop_gemm_b6_k32_n256, (void*) exl3_coop_gemm_b6_k16_n512};
        int tks[3] = {32, 32, 16}, tns[3] = {128, 256, 512};
        for (int v = 0; v < 3; ++v) {
            CK(cudaFuncSetAttribute(fs[v], cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
            for (int rows : {1, 2, 4, 8}) {
                int m = rows, k = K, n = N, ltn = N / tns[v], off = 0, ns = 48, base = 0; const uint16_t* bt = (const uint16_t*) trellis;
                void* args[] = {&a, &bt, &c, &m, &k, &n, &lk, &ltn, &off, &ns, &base};
                for (int i = 0; i < 5; ++i) CK(cudaLaunchCooperativeKernel(fs[v], 48, 256 * tks[v] / 16, args, 90 * 1024, 0));
                CK(cudaEventRecord(ea));
                for (int i = 0; i < iters; ++i) CK(cudaLaunchCooperativeKernel(fs[v], 48, 256 * tks[v] / 16, args, 90 * 1024, 0));
                CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
                float ms; CK(cudaEventElapsedTime(&ms, ea, eb)); double us = ms * 1e3 / iters, mb = tb / 1e6;
                printf("dense K4096xN4096 b6 tile (%d,%d) rows %d  %7.1f us  %5.1f MB  %6.1f GB/s\n", tks[v], tns[v], rows, us, mb, mb / us * 1e3);
            }
        }
    }
    return 0;
}
