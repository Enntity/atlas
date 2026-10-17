// SPDX-License-Identifier: AGPL-3.0-only
// Microbenchmark: routed-MoE thin coop kernels v2/v3/v6 (+ bit-exact check),
// a read-only roofline, and a side-stream overlap probe (SIDE_MB=...).
// Build in the builder container (no GPU), run on the Spark host:
//   nvcc -O3 -arch=sm_121a --fmad=false --expt-relaxed-constexpr \
//     -I<repo>/kernels/gb10/glm-5.3-flash/exl3 -o bench bench.cu
// Env: ITERS, SLOTS, GRIDCAP, SIDE_MB, SIDE_GRID.
// Decode geometry per rank: gate|up K4096->N1024 x2 projections, down
// K1024->N4096, 8 slots, 4-bit mul1 trellis, thin profile 32x256 x 8 slices.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "exl3_coop_gemm.cu"
#include "exl3_moe_precision.cu"
#include "exl3.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static inline unsigned long long next() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

// Roofline: read every byte of the selected experts' trellis once.
__global__ void read_kernel(const Exl3Expert* table, const int* ids, int slots, size_t bytes, unsigned long long* sink) {
    unsigned long long acc = 0;
    const size_t vecs = bytes / 16;
    for (int s = 0; s < slots; ++s) {
        const int4* p = (const int4*) table[ids[s]].trellis;
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < vecs; i += (size_t) gridDim.x * blockDim.x) {
            int4 v = __ldcg(p + i); acc += (unsigned) (v.x ^ v.y ^ v.z ^ v.w);
        }
    }
    if (acc == 0x123456789ull) *sink = acc;
}
struct Proj { int k, n; Exl3Expert* table; std::vector<void*> trellis; };

static Proj make_proj(int k, int n, int experts) {
    Proj p{k, n, nullptr, {}};
    size_t bytes = (size_t) k / 16 * (n / 16) * 16 * 4 * 2 / 2 * 1; // 16*bits u16 per 16x16 tile = 64 u16 = 128 B
    bytes = (size_t) (k / 16) * (n / 16) * 128;
    std::vector<unsigned long long> h(bytes / 8);
    std::vector<Exl3Expert> table(experts);
    for (int e = 0; e < experts; ++e) {
        for (auto& x : h) x = next();
        void* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
        p.trellis.push_back(d);
        std::vector<__half> hs(k);
        for (auto& x : hs) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
        void* suh; CK(cudaMalloc(&suh, k * 2)); CK(cudaMemcpy(suh, hs.data(), k * 2, cudaMemcpyHostToDevice));
        std::vector<__half> hv(n);
        for (auto& x : hv) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
        void* svh; CK(cudaMalloc(&svh, n * 2)); CK(cudaMemcpy(svh, hv.data(), n * 2, cudaMemcpyHostToDevice));
        table[e] = Exl3Expert{(unsigned long long) d, (unsigned long long) suh, (unsigned long long) svh, 4u, 0u};
    }
    CK(cudaMalloc(&p.table, sizeof(Exl3Expert) * experts));
    CK(cudaMemcpy(p.table, table.data(), sizeof(Exl3Expert) * experts, cudaMemcpyHostToDevice));
    return p;
}

static float* make_xh(int slots, int k) {
    std::vector<float> h((size_t) slots * k);
    for (auto& x : h) x = __half2float(__float2half_rn(((next() >> 40) / float(1 << 24)) * 2.0f - 1.0f));
    float* d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

struct Variant { const char* name; void* fn; int block; int smem; int cap; bool balanced; };

int main() {
    const int experts = 64, slots = getenv("SLOTS") ? atoi(getenv("SLOTS")) : 8, slices = 8, iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 200;
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    Proj gate = make_proj(4096, 1024, experts), up = make_proj(4096, 1024, experts), down = make_proj(1024, 4096, experts);
    float *xg = make_xh(slots, 4096), *xu = make_xh(slots, 4096), *xd = make_xh(slots, 1024);
    float *y0, *y1, *yd; CK(cudaMalloc(&y0, slots * 4096 * 4)); CK(cudaMalloc(&y1, slots * 4096 * 4)); CK(cudaMalloc(&yd, slots * 4096 * 4));
    int* locks; CK(cudaMalloc(&locks, 2 * slots * 32 * 4)); CK(cudaMemset(locks, 0, 2 * slots * 32 * 4));
    unsigned* counts; std::vector<unsigned> hc(experts, 1);
    CK(cudaMalloc(&counts, experts * 4)); CK(cudaMemcpy(counts, hc.data(), experts * 4, cudaMemcpyHostToDevice));
    const int rot = experts / 8;
    int* ids; CK(cudaMalloc(&ids, rot * slots * 4));
    std::vector<int> hid(rot * slots);
    for (int r = 0; r < rot; ++r) for (int s = 0; s < slots; ++s) hid[r * slots + s] = (r * 8 + s * 7) % experts;
    CK(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice));

    Variant vs[] = {
        {"v2", (void*) exl3_coop_moe2_thin_b4_k32_n256, 512, 90 * 1024, 0, false},
        {"v3", (void*) exl3_coop_moe3_thin_b4_k32_n256, 256, 6 * 5120, 0, false},
        {"v6", (void*) exl3_coop_moe6_thin_b4_k32_n256, 256, 6 * 5120 + 8192, 0, false},
    };
    for (auto& v : vs) {
        CK(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem));
        CK(cudaFuncSetAttribute(v.fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, v.fn, v.block, v.smem));
        v.cap = occ * sms;
        printf("%-6s block %d smem %d occupancy %d/SM cap %d\n", v.name, v.block, v.smem, occ, v.cap);
    }
    auto launch = [&](Variant& v, int proj, int r) {
        bool gu = proj == 2;
        const Proj& A = gu ? gate : down; const Proj& B = gu ? up : down;
        const float* x0 = gu ? xg : xd; const float* x1 = gu ? xu : xd;
        float* o0 = gu ? y0 : yd; float* o1 = gu ? y1 : yd;
        int* idp = ids + r * slots; int k = A.k, n = A.n, e = experts; unsigned cap = 32; int sl = slices, s = slots, p = proj;
        int tiles_k = k / 32, tiles_n = n / 256, total = tiles_k * tiles_n, chunks = 0;
        for (int i = 0; i < sl; ++i) { int b = total * i / sl, en = total * (i + 1) / sl; if (en > b) chunks += (en - 1) / tiles_k - b / tiles_k + 1; }
        int items = p * s * chunks;
        int gcap = v.cap; if (getenv("GRIDCAP") && v.name[1] != '2') { int g = atoi(getenv("GRIDCAP")); gcap = g < v.cap ? g : v.cap; }
        int grid = items < gcap ? items : gcap;
        void* args[] = {&x0, &x1, &idp, (void*) &A.table, (void*) &B.table, &o0, &o1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
        CK(cudaLaunchCooperativeKernel(v.fn, grid, v.block, args, v.smem, 0));
    };
    // Bit-exact check vs v2.
    std::vector<float> ref0(slots * 1024), ref1(slots * 1024), refd(slots * 4096), t0(slots * 1024), t1(slots * 1024), td(slots * 4096);
    for (int vi = 0; vi < 3; ++vi) {
        CK(cudaMemset(y0, 0xff, slots * 4096 * 4)); CK(cudaMemset(y1, 0xff, slots * 4096 * 4)); CK(cudaMemset(yd, 0xff, slots * 4096 * 4));
        launch(vs[vi], 2, 1); launch(vs[vi], 1, 1); CK(cudaDeviceSynchronize());
        auto& a = vi ? t0 : ref0; auto& b = vi ? t1 : ref1; auto& c = vi ? td : refd;
        CK(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.data(), yd, c.size() * 4, cudaMemcpyDeviceToHost));
        if (vi) {
            bool same = !memcmp(a.data(), ref0.data(), a.size() * 4) && !memcmp(b.data(), ref1.data(), b.size() * 4) && !memcmp(c.data(), refd.data(), c.size() * 4);
            printf("%-6s bit-exact vs v2: %s\n", vs[vi].name, same ? "YES" : "NO");
        }
    }
    cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
    const double gu_mb = 2.0 * slots * 4096 * 1024 / 2 / 1e6, dn_mb = 1.0 * slots * 4096 * 1024 / 2 / 1e6;
    for (auto& v : vs) {
        for (int proj : {2, 1}) {
            for (int i = 0; i < 10; ++i) launch(v, proj, i % rot);
            CK(cudaEventRecord(ea));
            for (int i = 0; i < iters; ++i) launch(v, proj, i % rot);
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            double us = ms * 1e3 / iters, mb = proj == 2 ? gu_mb : dn_mb;
            printf("%-6s %-7s %8.1f us  %6.1f GB/s\n", v.name, proj == 2 ? "gate|up" : "down", us, mb / us * 1e3);
        }
    }
    {
        unsigned long long* sink; CK(cudaMalloc(&sink, 8));
        for (int grid : {96, 192, 384, 768}) for (int proj : {2, 1}) {
            const Proj& A = proj == 2 ? gate : down; size_t bytes = (size_t) (A.k / 16) * (A.n / 16) * 128;
            auto run = [&](int i) {
                read_kernel<<<grid, 256>>>(A.table, ids + (i % rot) * slots, slots, bytes, sink);
                if (proj == 2) read_kernel<<<grid, 256>>>(up.table, ids + (i % rot) * slots, slots, bytes, sink);
            };
            for (int i = 0; i < 10; ++i) run(i);
            CK(cudaEventRecord(ea)); for (int i = 0; i < iters; ++i) run(i); CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb)); double us = ms * 1e3 / iters, mb = proj == 2 ? gu_mb : dn_mb;
            printf("read   grid %4d %-7s %8.1f us  %6.1f GB/s\n", grid, proj == 2 ? "gate|up" : "down", us, mb / us * 1e3);
        }
    }
    // Concurrency probe: routed v3 (gate|up then down) on s1 while a
    // memory-bound side kernel streams SIDE_MB on s2 (proxy for the shared
    // expert). Compares wall time against the two run back to back.
    if (getenv("SIDE_MB")) {
        const size_t side_bytes = (size_t) (atof(getenv("SIDE_MB")) * 1e6);
        void* side; CK(cudaMalloc(&side, side_bytes * 8));
        unsigned long long* sink; CK(cudaMalloc(&sink, 8));
        Exl3Expert hside[8]; for (int i = 0; i < 8; ++i) hside[i] = Exl3Expert{(unsigned long long) ((char*) side + side_bytes * i), 0, 0, 4u, 0u};
        Exl3Expert* dside; CK(cudaMalloc(&dside, sizeof(hside))); CK(cudaMemcpy(dside, hside, sizeof(hside), cudaMemcpyHostToDevice));
        int* zero_ids; CK(cudaMalloc(&zero_ids, 4)); CK(cudaMemset(zero_ids, 0, 4));
        cudaStream_t s1, s2; CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        cudaEvent_t fork, join; CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        int sg = getenv("SIDE_GRID") ? atoi(getenv("SIDE_GRID")) : 96;
        Variant& v = vs[1];
        auto moe = [&](cudaStream_t st, int i) {
            for (int proj : {2, 1}) {
                bool gu = proj == 2; const Proj& A = gu ? gate : down; const Proj& B = gu ? up : down;
                const float* x0 = gu ? xg : xd; const float* x1 = gu ? xu : xd; float* o0 = gu ? y0 : yd; float* o1 = gu ? y1 : yd;
                int* idp = ids + (i % rot) * slots; int k = A.k, n = A.n, e = experts; unsigned cap = 32; int sl = slices, s = slots, p = proj;
                int items = p * s * (gu ? 8 : 16); int gcap = v.cap; if (getenv("GRIDCAP")) gcap = atoi(getenv("GRIDCAP"));
                int grid = items < gcap ? items : gcap;
                void* args[] = {&x0, &x1, &idp, (void*) &A.table, (void*) &B.table, &o0, &o1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
                CK(cudaLaunchCooperativeKernel(v.fn, grid, v.block, args, v.smem, st));
            }
        };
        auto sidek = [&](cudaStream_t st, int i) { read_kernel<<<sg, 256, 0, st>>>(dside, zero_ids, 1, side_bytes, sink); (void) i; };
        for (int mode = 0; mode < 3; ++mode) {
            CK(cudaDeviceSynchronize()); CK(cudaEventRecord(ea, s1));
            for (int i = 0; i < iters; ++i) {
                if (mode == 0) { moe(s1, i); }
                else if (mode == 1) { moe(s1, i); sidek(s1, i); }
                else { CK(cudaEventRecord(fork, s1)); CK(cudaStreamWaitEvent(s2, fork)); sidek(s2, i); moe(s1, i); CK(cudaEventRecord(join, s2)); CK(cudaStreamWaitEvent(s1, join)); }
            }
            CK(cudaEventRecord(eb, s1)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            printf("concurrency %-22s %8.1f us/iter\n", mode == 0 ? "moe only" : mode == 1 ? "moe then side (serial)" : "side || moe", ms * 1e3 / iters);
        }
    }
    // v7 (fused input transform, gate|up) vs Atlas input transform + v6.
    {
        std::vector<__nv_bfloat16> hx(4096);
        for (auto& x : hx) x = __float2bfloat16(((next() >> 40) / float(1 << 24)) * 4.0f - 2.0f);
        __nv_bfloat16* dx; CK(cudaMalloc(&dx, 4096 * 2)); CK(cudaMemcpy(dx, hx.data(), 4096 * 2, cudaMemcpyHostToDevice));
        void* f7 = (void*) exl3_coop_moe7_thin_b4_k32_n256; void* f6 = (void*) exl3_coop_moe6_thin_b4_k32_n256;
        const int sm6 = 6 * 5120 + 8192;
        CK(cudaFuncSetAttribute(f7, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CK(cudaFuncSetAttribute(f6, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        const int topk = slots;  // one token
        auto had = [&](int r) {
            int* idp = ids + (r % rot) * slots;
            exl3_moe_input_hadamard_f16<<<dim3(32, slots), 128>>>(dx, idp, gate.table, xg, 1, 4096, topk, 0, experts);
            exl3_moe_input_hadamard_f16<<<dim3(32, slots), 128>>>(dx, idp, up.table, xu, 1, 4096, topk, 0, experts);
        };
        auto run = [&](void* fn, int r) {
            int* idp = ids + (r % rot) * slots; int k = 4096, n = 1024, e = experts, p = 2, s = slots, sl = slices, tk = topk;
            unsigned cap = 32; int grid = 96;
            if (fn == f6) {
                void* args[] = {&xg, &xu, &idp, (void*) &gate.table, (void*) &up.table, &y0, &y1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
                CK(cudaLaunchCooperativeKernel(fn, grid, 256, args, sm6, 0));
            } else {
                void* args[] = {&dx, &tk, &idp, (void*) &gate.table, (void*) &up.table, &y0, &y1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
                CK(cudaLaunchCooperativeKernel(fn, grid, 256, args, sm6, 0));
            }
        };
        std::vector<float> a0(slots * 1024), a1(slots * 1024), b0(slots * 1024), b1(slots * 1024);
        had(3); run(f6, 3); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a0.data(), y0, a0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a1.data(), y1, a1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemset(y0, 0xff, slots * 4096 * 4)); CK(cudaMemset(y1, 0xff, slots * 4096 * 4));
        run(f7, 3); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(b0.data(), y0, b0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b1.data(), y1, b1.size() * 4, cudaMemcpyDeviceToHost));
        bool same = !memcmp(a0.data(), b0.data(), a0.size() * 4) && !memcmp(a1.data(), b1.data(), a1.size() * 4);
        printf("v7     bit-exact vs Atlas input transform + v6: %s\n", same ? "YES" : "NO");
        for (int mode = 0; mode < 2; ++mode) {
            for (int i = 0; i < 10; ++i) { if (!mode) had(i); run(mode ? f7 : f6, i); }
            CK(cudaEventRecord(ea));
            for (int i = 0; i < iters; ++i) { if (!mode) had(i); run(mode ? f7 : f6, i); }
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            printf("%-34s %8.1f us\n", mode ? "v7 gate|up (fused transform)" : "2x input transform + v6 gate|up", ms * 1e3 / iters);
        }
    }
    // v10 streaming persistent kernel vs v6 (same items, same arithmetic).
    {
        void* f6 = (void*) exl3_coop_moe6_thin_b4_k32_n256; void* f10 = (void*) exl3_coop_moe10_thin_b4_k32_n256;
        const int sm6 = 6 * 5120 + 8192, sm10 = EXL3_MOE10_SH * 5120;
        CK(cudaFuncSetAttribute(f10, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        int occ10; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ10, f10, 256, sm10));
        int cap10 = occ10 * sms; if (getenv("V10CAP")) cap10 = std::min(cap10, atoi(getenv("V10CAP")));
        printf("v10    occupancy %d/SM cap %d\n", occ10, cap10);
        // F16 copies of the F32 (F16-representable) per-slot inputs
        auto to_half = [&](const float* src, size_t n) {
            std::vector<float> h(n); CK(cudaMemcpy(h.data(), src, n * 4, cudaMemcpyDeviceToHost));
            std::vector<__half> hh(n); for (size_t i = 0; i < n; ++i) hh[i] = __float2half_rn(h[i]);
            __half* d; CK(cudaMalloc(&d, n * 2)); CK(cudaMemcpy(d, hh.data(), n * 2, cudaMemcpyHostToDevice)); return d; };
        __half *hg = to_half(xg, slots * 4096), *hu = to_half(xu, slots * 4096), *hd = to_half(xd, slots * 1024);
        auto launch = [&](void* fn, int proj, int r) {
            bool gu = proj == 2; const Proj& A = gu ? gate : down; const Proj& B = gu ? up : down;
            float* o0 = gu ? y0 : yd; float* o1 = gu ? y1 : yd;
            int* idp = ids + (r % rot) * slots; int k = A.k, n = A.n, e = experts, p = proj, s = slots, sl = slices; unsigned cap = 32;
            int tkk = k / 32, total = tkk * (n / 256), ch = 0; for (int i = 0; i < sl; ++i) { int b = total * i / sl, en = total * (i + 1) / sl; if (en > b) ch += (en - 1) / tkk - b / tkk + 1; }
            int items = p * s * ch;
            if (fn == f6) {
                const float* x0 = gu ? xg : xd; const float* x1 = gu ? xu : xd;
                void* args[] = {&x0, &x1, &idp, (void*) &A.table, (void*) &B.table, &o0, &o1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
                CK(cudaLaunchCooperativeKernel(fn, std::min(items, 96), 256, args, sm6, 0));
            } else {
                const __half* x0 = gu ? hg : hd; const __half* x1 = gu ? hu : hd;
                void* args[] = {&x0, &x1, &idp, (void*) &A.table, (void*) &B.table, &o0, &o1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
                CK(cudaLaunchCooperativeKernel(fn, std::min(items, cap10), 256, args, sm10, 0));
            }
        };
        std::vector<float> a0(slots * 1024), a1(slots * 1024), ad(slots * 4096), b0(a0.size()), b1(a1.size()), bd(ad.size());
        launch(f6, 2, 1); launch(f6, 1, 1); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a0.data(), y0, a0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a1.data(), y1, a1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ad.data(), yd, ad.size() * 4, cudaMemcpyDeviceToHost));
        for (float* q : {y0, y1, yd}) CK(cudaMemset(q, 0xff, slots * 4096 * 4));
        launch(f10, 2, 1); launch(f10, 1, 1); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(b0.data(), y0, b0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b1.data(), y1, b1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bd.data(), yd, bd.size() * 4, cudaMemcpyDeviceToHost));
        printf("v10    bit-exact vs v6: gate|up %s down %s\n",
               memcmp(a0.data(), b0.data(), a0.size() * 4) || memcmp(a1.data(), b1.data(), a1.size() * 4) ? "NO" : "YES",
               memcmp(ad.data(), bd.data(), ad.size() * 4) ? "NO" : "YES");
        for (void* fn : {f6, f10}) for (int proj : {2, 1}) {
            for (int i = 0; i < 10; ++i) launch(fn, proj, i);
            CK(cudaEventRecord(ea)); for (int i = 0; i < iters; ++i) launch(fn, proj, i);
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb)); float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            double us = ms * 1e3 / iters, mb = proj == 2 ? gu_mb : dn_mb;
            printf("%-6s %-7s %8.1f us  %6.1f GB/s\n", fn == f6 ? "v6" : "v10", proj == 2 ? "gate|up" : "down", us, mb / us * 1e3);
        }
    }

    // v9 down (fused gate/up output transforms + activation + down input
    // transform) vs Atlas's separate kernels + v6 down.
    for (unsigned clamp : {0u, 1u}) {
        const float limit = 7.0f; int r = 2;
        int* idp = ids + r * slots;
        void* f6 = (void*) exl3_coop_moe6_thin_b4_k32_n256; void* f9 = (void*) exl3_coop_moe9_down_b4_k32_n256;
        const int sm6 = 6 * 5120 + 8192;
        CK(cudaFuncSetAttribute(f9, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        // gate/up GEMM sums (v6) as the shared starting point
        { int k = 4096, n = 1024, e = experts, p = 2, s = slots, sl = slices; unsigned cap = 32;
          void* args[] = {&xg, &xu, &idp, (void*) &gate.table, (void*) &up.table, &y0, &y1, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
          CK(cudaLaunchCooperativeKernel(f6, 96, 256, args, sm6, 0)); }
        float *g1, *u1, *xa, *yref, *y9; for (float** q : {&g1, &u1, &xa}) CK(cudaMalloc(q, slots * 1024 * 4));
        CK(cudaMalloc(&yref, slots * 4096 * 4)); CK(cudaMalloc(&y9, slots * 4096 * 4));
        auto reference = [&]() {
            exl3_moe_output_hadamard_precision<<<dim3(8, slots), 128>>>(y0, idp, gate.table, g1, counts, 1, 1024, slots, experts, 32);
            exl3_moe_output_hadamard_precision<<<dim3(8, slots), 128>>>(y1, idp, up.table, u1, counts, 1, 1024, slots, experts, 32);
            exl3_moe_activate_down_hadamard<<<dim3(8, slots), 128>>>(g1, u1, idp, down.table, xa, counts, 1, 1024, slots, experts, 32, limit, clamp);
            int k = 1024, n = 4096, e = experts, p = 1, s = slots, sl = slices; unsigned cap = 32;
            void* args[] = {&xa, &xa, &idp, (void*) &down.table, (void*) &down.table, &yref, &yref, &counts, &locks, &p, &s, &k, &n, &e, &cap, &sl};
            CK(cudaLaunchCooperativeKernel(f6, 96, 256, args, sm6, 0));
        };
        auto fused = [&]() {
            int k = 1024, n = 4096, e = experts, s = slots, sl = slices; unsigned cap = 32; float lim = limit; unsigned cl = clamp;
            void* args[] = {&y0, &y1, &idp, (void*) &gate.table, (void*) &up.table, (void*) &down.table, &y9, &counts, &locks, &s, &k, &n, &e, &cap, &sl, &lim, &cl};
            CK(cudaLaunchCooperativeKernel(f9, 96, 256, args, sm6, 0));
        };
        reference(); fused(); CK(cudaDeviceSynchronize());
        std::vector<float> a(slots * 4096), b(slots * 4096);
        CK(cudaMemcpy(a.data(), yref, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), y9, b.size() * 4, cudaMemcpyDeviceToHost));
        printf("v9 down clamp %u bit-exact vs 2x output H + activate/down H + v6 down: %s\n", clamp, memcmp(a.data(), b.data(), a.size() * 4) ? "NO" : "YES");
        for (int mode = 0; mode < 2; ++mode) {
            for (int i = 0; i < 5; ++i) { if (mode) fused(); else reference(); }
            CK(cudaEventRecord(ea)); for (int i = 0; i < iters; ++i) { if (mode) fused(); else reference(); }
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb)); float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            printf("%-46s %8.1f us\n", mode ? "v9 fused down" : "2x out H + act/down H + v6 down", ms * 1e3 / iters);
        }
    }

    // Dense coop: fused input transform (gemmh) vs f16h transform + gemm.
    for (int variant = 0; variant < 3; ++variant) {
        const int K = 4096, N = 4096, bits = 6, tk = variant == 2 ? 16 : 32, tn = variant == 0 ? 128 : variant == 1 ? 256 : 512;
        void* fg = variant == 0 ? (void*) exl3_coop_gemm_b6_k32_n128 : variant == 1 ? (void*) exl3_coop_gemm_b6_k32_n256 : (void*) exl3_coop_gemm_b6_k16_n512;
        void* fh = variant == 0 ? (void*) exl3_coop_gemmh_b6_k32_n128 : variant == 1 ? (void*) exl3_coop_gemmh_b6_k32_n256 : (void*) exl3_coop_gemmh_b6_k16_n512;
        const int block = 256 * tk / 16, smem = 90 * 1024;
        CK(cudaFuncSetAttribute(fg, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CK(cudaFuncSetAttribute(fh, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        size_t tb = (size_t) (K / 16) * (N / 16) * 16 * bits * 2;
        std::vector<unsigned long long> ht(tb / 8); for (auto& x : ht) x = next();
        void* trellis; CK(cudaMalloc(&trellis, tb)); CK(cudaMemcpy(trellis, ht.data(), tb, cudaMemcpyHostToDevice));
        std::vector<__half> hs(K); for (auto& x : hs) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
        __half* suh; CK(cudaMalloc(&suh, K * 2)); CK(cudaMemcpy(suh, hs.data(), K * 2, cudaMemcpyHostToDevice));
        const int R = 3;
        std::vector<__nv_bfloat16> hx(R * K); for (auto& x : hx) x = __float2bfloat16(((next() >> 40) / float(1 << 24)) * 4.0f - 2.0f);
        __nv_bfloat16* dx; CK(cudaMalloc(&dx, R * K * 2)); CK(cudaMemcpy(dx, hx.data(), R * K * 2, cudaMemcpyHostToDevice));
        __half* ah; CK(cudaMalloc(&ah, R * K * 2));
        float *c0, *c1; CK(cudaMalloc(&c0, R * N * 4)); CK(cudaMalloc(&c1, R * N * 4));
        int* lk; CK(cudaMalloc(&lk, N / 128 * 4)); CK(cudaMemset(lk, 0, N / 128 * 4));
        for (int rows = 1; rows <= R; ++rows) {
            int m = rows, k = K, n = N, ltn = N / tn, off = 0, ns = 48, base = 0;
            const uint16_t* bt = (const uint16_t*) trellis;
            exl3_coop_input_hadamard_f16h<<<dim3(K / 128, rows), 128>>>(dx, suh, ah, rows, K);
            void* a1[] = {&ah, &bt, &c0, &m, &k, &n, &lk, &ltn, &off, &ns, &base};
            CK(cudaLaunchCooperativeKernel(fg, 48, block, a1, smem, 0));
            void* a2[] = {&dx, &bt, &c1, &m, &k, &n, &lk, &ltn, &off, &ns, &base, &suh};
            CK(cudaLaunchCooperativeKernel(fh, 48, block, a2, smem, 0));
            CK(cudaDeviceSynchronize());
            std::vector<float> r0(rows * N), r1(rows * N);
            CK(cudaMemcpy(r0.data(), c0, rows * N * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r1.data(), c1, rows * N * 4, cudaMemcpyDeviceToHost));
            printf("gemmh tile (%d,%d) rows %d bit-exact vs f16h + gemm: %s\n", tk, tn, rows, memcmp(r0.data(), r1.data(), rows * N * 4) ? "NO" : "YES");
            // gemmf (fused input + output) vs f16h + gemm + exl3_output_hadamard.
            {
                void* ff = variant == 0 ? (void*) exl3_coop_gemmf_b6_k32_n128 : variant == 1 ? (void*) exl3_coop_gemmf_b6_k32_n256 : (void*) exl3_coop_gemmf_b6_k16_n512;
                CK(cudaFuncSetAttribute(ff, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
                static __half* svh = nullptr; static __nv_bfloat16 *o0 = nullptr, *o1 = nullptr;
                if (!svh) {
                    std::vector<__half> hv(N); for (auto& x : hv) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
                    CK(cudaMalloc(&svh, N * 2)); CK(cudaMemcpy(svh, hv.data(), N * 2, cudaMemcpyHostToDevice));
                    CK(cudaMalloc(&o0, R * N * 2)); CK(cudaMalloc(&o1, R * N * 2));
                }
                exl3_output_hadamard<<<dim3(N / 128, rows), 128>>>(c0, svh, o0, rows, N);
                void* a3[] = {&dx, &bt, &c1, &m, &k, &n, &lk, &ltn, &off, &ns, &base, &suh, &o1, &svh};
                CK(cudaLaunchCooperativeKernel(ff, 48, block, a3, smem, 0));
                CK(cudaDeviceSynchronize());
                std::vector<__nv_bfloat16> q0(rows * N), q1(rows * N);
                CK(cudaMemcpy(q0.data(), o0, rows * N * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(q1.data(), o1, rows * N * 2, cudaMemcpyDeviceToHost));
                printf("gemmf tile (%d,%d) rows %d bit-exact vs f16h + gemm + output H128: %s\n", tk, tn, rows, memcmp(q0.data(), q1.data(), rows * N * 2) ? "NO" : "YES");
            }
        }
        for (int mode = 0; mode < 3; ++mode) {
            int m = 1, k = K, n = N, ltn = N / tn, off = 0, ns = 48, base = 0; const uint16_t* bt = (const uint16_t*) trellis;
            void* ff = variant == 0 ? (void*) exl3_coop_gemmf_b6_k32_n128 : variant == 1 ? (void*) exl3_coop_gemmf_b6_k32_n256 : (void*) exl3_coop_gemmf_b6_k16_n512;
            static __half* svh2 = nullptr; static __nv_bfloat16* o2 = nullptr;
            if (!svh2) { CK(cudaMalloc(&svh2, N * 2)); CK(cudaMemset(svh2, 0x3c, N * 2)); CK(cudaMalloc(&o2, N * 2)); }
            void* a3[] = {&dx, &bt, &c1, &m, &k, &n, &lk, &ltn, &off, &ns, &base, &suh, &o2, &svh2};
            void* a1[] = {&ah, &bt, &c0, &m, &k, &n, &lk, &ltn, &off, &ns, &base};
            void* a2[] = {&dx, &bt, &c1, &m, &k, &n, &lk, &ltn, &off, &ns, &base, &suh};
            CK(cudaEventRecord(ea));
            for (int i = 0; i < iters; ++i) {
                if (mode == 0) { exl3_coop_input_hadamard_f16h<<<dim3(K / 128, 1), 128>>>(dx, suh, ah, 1, K); CK(cudaLaunchCooperativeKernel(fg, 48, block, a1, smem, 0)); exl3_output_hadamard<<<dim3(N / 128, 1), 128>>>(c0, svh2, o2, 1, N); }
                else if (mode == 1) { CK(cudaLaunchCooperativeKernel(fh, 48, block, a2, smem, 0)); exl3_output_hadamard<<<dim3(N / 128, 1), 128>>>(c1, svh2, o2, 1, N); }
                else CK(cudaLaunchCooperativeKernel(ff, 48, block, a3, smem, 0));
            }
            CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb));
            float ms; CK(cudaEventElapsedTime(&ms, ea, eb));
            printf("dense tile (%d,%d) rows 1 %-34s %8.1f us\n", tk, tn, mode == 0 ? "f16h + gemm + out H128" : mode == 1 ? "gemmh + out H128" : "gemmf (both fused)", ms * 1e3 / iters);
        }
    }
    return 0;
}
