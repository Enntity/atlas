// SPDX-License-Identifier: AGPL-3.0-only
// Standalone answers for ATLAS_GLM_L2_AHEAD: a side-stream kernel
// (glm_l2_ahead, kernels/gb10/glm-5.3-flash/nvfp4/glm_l2_ahead.cuh) that asks
// for the next projections' weights while the verify step's single stream
// waits on an all-reduce or runs a latency-bound chain.
//
// 1. Bitwise: every GEMV the sites feed (KDA q batch3_touch, shared gate/up
//    tc8_pair_touch, shared down tc8_touch K-slice, MLA q_b / o mxfp8_gemv_tc8)
//    gives the same bits with and without a prefetch of its weight in every
//    mode. Must print "bitwise: 0 of N outputs differ".
// 2. Mechanism: after a 128 MB flush, prefetch W MB (lines, sectors, last,
//    touch, bulk, and "persist" = touch under a persisting access-policy
//    window), wait `gap` us, then time a read of the W MB. "res" is the
//    resident fraction (cold - t) / (cold - hot); "pf" the prefetch kernel's
//    own time.
// 3. Survival: prefetch 8 MB, let it land, stream S MB through L2 the way the
//    routed MoE does (cs: streaming loads as the m16s twins; plain loads; cs +
//    prefetch.global.L2 ahead as the _l2pf twins), then read the 8 MB.
// 4. Interference: a chain of ten L2-latency-bound PDL kernels, alone and with
//    a 12 MB prefetch forked beside it.
// 5. Sites: one emulated layer segment per site, real GEMVs in production
//    launch modes, a latency-bound `spin` for the small kernels and a plain
//    spin for the one-shot all-reduce wait; weights cycle over distinct
//    copies behind a 100 MB streamed MoE stand-in, so every read starts cold:
//      f  FFN all-reduce -> next KDA layer: [wait] [35 us chain] q k v
//      a  attention all-reduce -> shared gate/up, down, router: [wait] [35 us]
//      q  sparse-MLA indexer chain (90 us) -> q_b
//      o  after the W_uk absorb: [40 us attention] W_uv (BF16 read) -> o
//    "saved" = base - arm per segment; "/step" scales it by the layers a C1
//    verify step runs it on (f 34, a 42, q 11, o 11). No persist arm here:
//    a policy window covers one region.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_l2_ahead_bench.cu -o l2_ahead_bench
//   ./l2_ahead_bench [reps=7] [ctas=32] [budget_mb=12] [wait_us=10]
// Device memory: ~0.8 GB. Last line "PASS: bitwise 0 of N outputs differ"
// (exit 0) iff the bitwise check passes; the rest is the timing report.
#include "w4a16_gemv.cu"
#include "mxfp8_gemv.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
typedef unsigned char u8;
typedef unsigned long long u64;

// Latency-bound stand-in for the small kernels (dependent loads from a table
// of `mask + 1` entries: 4 KiB stays in L1, 1 MiB goes to L2).
extern "C" __global__ void spin(unsigned int* tab, unsigned int mask, unsigned int iters) {
    atlas_pdl_enter();
    unsigned int x = threadIdx.x + blockIdx.x * 977u;
    for (unsigned int i = 0; i < iters; i++) x = __ldcg(tab + ((x + i) & mask));
    if (x == 0xFFFFFFFFu) tab[0] = 0u;  // never: entries are below the table size
}

// `spin` without the PDL entry, launched plain: the one-shot all-reduce
// kernel, which is not on the PDL list (its successor starts when it ends).
extern "C" __global__ void hold(unsigned int* tab, unsigned int mask, unsigned int iters) {
    unsigned int x = threadIdx.x + blockIdx.x * 977u;
    for (unsigned int i = 0; i < iters; i++) x = __ldcg(tab + ((x + i) & mask));
    if (x == 0xFFFFFFFFu) tab[0] = 0u;
}

// One pass over [p, p + n16 * 16): plain or streaming (`cs`) loads; `ahead16`
// > 0 also prefetches the line that far ahead, every 128 bytes (the _l2pf
// twins' pattern). Stands for the routed MoE, the BF16 W_uv / router reads,
// and probes L2 residency.
extern "C" __global__ void rd(const uint4* p, u64 n16, unsigned int cs, u64 ahead16, unsigned int* sink) {
    atlas_pdl_enter();
    unsigned int acc = 0;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        if (ahead16 && (i & 7) == 0 && i + ahead16 < n16)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i + ahead16));
        const uint4 v = cs ? __ldcs(p + i) : p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads
}

extern "C" __global__ void fill(u8* p, u64 n, unsigned int kind, unsigned int seed) {
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
        unsigned int h = (unsigned int)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        u8 b = (u8)h;
        if (kind == 1) b = 0x20 + (b & 0x1F);              // E4M3 block scales
        if (kind == 2 && (b & 0x7F) == 0x7F) b ^= 1;      // E4M3 values, no NaN
        if (kind == 3) b = 120 + (b & 7);                  // E8M0 scales
        if (kind == 4) b = (i & 1) ? (0x3C + (b & 1)) : b; // BF16 in a sane range
        p[i] = b;
    }
}

template <typename... Args>
static void launch(cudaStream_t s, bool pdl, void (*kern)(Args...), dim3 grid, unsigned int block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = dim3(block, 1, 1); cfg.stream = s;
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

// ── glm_l2_ahead host side (mirrors crates/spark-model/src/layers/ops/l2_ahead.rs) ──
struct Region { const u8* p; u64 ld; unsigned int bytes, rows; };
static Region whole(const void* p, u64 n) { return {(const u8*)p, n, (unsigned int)n, 1u}; }
static u64 size_of(const Region& r) { return (u64)r.bytes * r.rows; }

// The leading `budget` bytes of `rs` in order: whole regions, then whole rows
// of the next one (a one-row region: its first bytes).
static std::vector<Region> budgeted(const std::vector<Region>& rs, u64 budget) {
    std::vector<Region> out;
    for (Region r : rs) {
        if (budget == 0 || out.size() == L2A_REGIONS) break;
        if (size_of(r) > budget) {
            if (r.rows > 1) r.rows = (unsigned int)(budget / r.bytes);
            else r.bytes = (unsigned int)budget;
            if (r.rows == 0) break;
        }
        budget -= size_of(r);
        out.push_back(r);
    }
    return out;
}

enum { M_LINES, M_SECTORS, M_LAST, M_TOUCH, M_BULK, M_PERSIST, M_MODES };
static const char* MODE[M_MODES] = {"lines", "sectors", "last", "touch", "bulk", "persist"};
static u64 g_persist = 0;  // window bytes while a persist arm runs (0: unsupported, plain touch)

static void l2a(cudaStream_t s, int mode, unsigned int ctas, const std::vector<Region>& rs) {
    Region r[L2A_REGIONS] = {};
    for (size_t i = 0; i < rs.size() && i < L2A_REGIONS; i++) r[i] = rs[i];
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(ctas); cfg.blockDim = dim3(L2A_THREADS); cfg.stream = s;
    cudaLaunchAttribute at;
    if (mode == M_PERSIST && g_persist) {  // touch under a persisting window over region 0
        at.id = cudaLaunchAttributeAccessPolicyWindow;
        at.val.accessPolicyWindow.base_ptr = (void*)r[0].p;
        at.val.accessPolicyWindow.num_bytes = std::min<u64>(size_of(r[0]), g_persist);
        at.val.accessPolicyWindow.hitRatio = 1.f;
        at.val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        at.val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cfg.attrs = &at; cfg.numAttrs = 1;
    }
    const unsigned int m = mode == M_PERSIST ? L2A_TOUCH : (unsigned int)mode;
#define R(i) r[i].p, r[i].ld, r[i].bytes, r[i].rows
    CK(cudaLaunchKernelEx(&cfg, glm_l2_ahead, R(0), R(1), R(2), R(3), R(4), R(5), R(6), R(7), m));
#undef R
}

static void persist_begin(int mode) {
    if (mode != M_PERSIST) return;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_persist = std::min<u64>(prop.persistingL2CacheMaxSize, prop.accessPolicyMaxWindowSize);
    // A device that refuses a persisting carve-out runs the arm as plain touch.
    if (g_persist && cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize) != cudaSuccess) {
        (void)cudaGetLastError();
        g_persist = 0;
    }
}
static void persist_end(int mode) {
    if (mode != M_PERSIST || !g_persist) return;
    g_persist = 0;
    CK(cudaDeviceSynchronize());
    CK(cudaCtxResetPersistingL2Cache());
    CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
}

// ── weights ──
static const float SCALE2 = 0.37f;
struct Nv { u8 *w, *s; unsigned int n, k; };   // NVFP4 [n, k]
struct Mx { u8 *w, *s; unsigned int n, k; };   // MXFP8 [n, k]
static u8* dev(u64 n, unsigned int kind, unsigned int seed) {
    u8* p; CK(cudaMalloc(&p, n)); fill<<<1024, 256>>>(p, n, kind, seed); return p;
}
static Nv nv(unsigned int n, unsigned int k, unsigned int seed) {
    return {dev((u64)n * k / 2, 0, seed), dev((u64)n * k / 16, 1, seed + 7), n, k};
}
static Mx mx(unsigned int n, unsigned int k, unsigned int seed) {
    return {dev((u64)n * k, 2, seed), dev((u64)n * k / 32, 3, seed + 7), n, k};
}
// Scales first (small), then values, as l2_ahead.rs orders a weight.
static void push_nv(std::vector<Region>& v, const Nv& w) { v.push_back(whole(w.s, (u64)w.n * w.k / 16)); v.push_back(whole(w.w, (u64)w.n * w.k / 2)); }
static void push_mx(std::vector<Region>& v, const Mx& w) { v.push_back(whole(w.s, (u64)w.n * w.k / 32)); v.push_back(whole(w.w, (u64)w.n * w.k)); }

// One KDA + MoE layer: q/k/v (4096x4096), the shared expert (2048 wide: this
// rank's gate/up rows [1024, 2048) and down columns [1024, 2048)) and the BF16
// router (288 x 4096). One sparse-MLA layer: q_b (8192x1536) and o
// (4096x8192) MXFP8, W_uk / W_uv BF16 (32 heads x 512 x 256).
struct Kda { Nv q, k, v, gate, up, down; u8* router; };
struct Mla { Mx qb, o; u8 *wuk, *wuv; };
static const unsigned int H = 4096, HALF = 1024, ROUTER = 288 * 4096 * 2, WUV = 32 * 512 * 256 * 2;

static unsigned int* g_sink;
static void read(cudaStream_t s, bool pdl, const void* p, u64 bytes, unsigned int cs = 0, u64 ahead = 0) {
    launch(s, pdl, rd, dim3(192), 256u, (const uint4*)p, bytes / 16, cs, ahead / 16, g_sink);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;
    const unsigned int ctas = argc > 2 ? (unsigned int)atoi(argv[2]) : 32u;
    const u64 budget = (argc > 3 ? (u64)atoi(argv[3]) : 12ull) << 20;
    const unsigned int wait_us = argc > 4 ? (unsigned int)atoi(argv[4]) : 10u;
    const int KDA_LAYERS = 6, MLA_LAYERS = 4;
    cudaStream_t s0, s1;
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    cudaEvent_t fork_ev; CK(cudaEventCreateWithFlags(&fork_ev, cudaEventDisableTiming));

    std::mt19937 rng(7);
    std::vector<unsigned int> hTab(1u << 18);
    for (size_t i = 0; i < hTab.size(); i++) hTab[i] = rng() & ((1u << 18) - 1u);
    unsigned int* dTab; CK(cudaMalloc(&dTab, hTab.size() * 4));
    CK(cudaMemcpy(dTab, hTab.data(), hTab.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_sink, 64));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf> hA(32 * 8192);
    for (auto& x : hA) x = __float2bfloat16(nd(rng));
    bf *dA, *dC[2];
    CK(cudaMalloc(&dA, hA.size() * 2)); CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    for (auto& c : dC) CK(cudaMalloc(&c, 32 * 8192 * 2));
    const u64 MOE = 170ull << 20, X = 16ull << 20;
    u8* moe = dev(MOE, 0, 99);   // the routed MoE stand-in
    u8* flushbuf = dev(128ull << 20, 0, 97);
    u8* probe = dev(X, 0, 98);
    std::vector<Kda> kda(KDA_LAYERS);
    std::vector<Mla> mla(MLA_LAYERS);
    for (int l = 0; l < KDA_LAYERS; l++) {
        const unsigned int b = 100u * l;
        kda[l] = {nv(H, H, b), nv(H, H, b + 1), nv(H, H, b + 2), nv(2 * HALF, H, b + 3), nv(2 * HALF, H, b + 4),
                  nv(H, 2 * HALF, b + 5), dev(ROUTER, 4, b + 6)};
    }
    for (int l = 0; l < MLA_LAYERS; l++) {
        const unsigned int b = 5000u + 100u * l;
        mla[l] = {mx(8192, 1536, b), mx(H, 8192, b + 1), dev(WUV, 4, b + 2), dev(WUV, 4, b + 3)};
    }
    CK(cudaDeviceSynchronize());

    // This rank's shared-expert slices (ATLAS_GLM_SHARED_TP_SPLIT): gate/up rows
    // [HALF, 2 HALF) and down's K columns [HALF, 2 HALF) of each row.
    auto gate_rows = [](const Nv& w) { return Nv{w.w + (u64)HALF * H / 2, w.s + (u64)HALF * H / 16, HALF, H}; };
    auto down_regions = [](std::vector<Region>& v, const Nv& d) {
        v.push_back({d.s + HALF / 16, 2 * HALF / 16, HALF / 16, H});
        v.push_back({d.w + HALF / 2, 2 * HALF / 2, HALF / 2, H});
    };
    // The kernels, production launch modes (M = 3 rows: C1 verify).
    const unsigned int M = 3, TOUCH_ROWS_KDA = H, TOUCH_ROWS_SH = HALF;  // 12 MB touch covers whole weights
    auto qkv = [&](cudaStream_t s, const Nv& w, bf* c) {
        launch(s, true, w4a16_gemv_batch3_touch, dim3((w.n + 3) / 4), 256u, (const bf*)dA, (const u8*)w.w,
               (const u8*)w.s, SCALE2, c, M, w.n, w.k, TOUCH_ROWS_KDA, 32u);
    };
    auto pair = [&](cudaStream_t s, const Kda& L, bf* c) {
        const Nv g = gate_rows(L.gate), u = gate_rows(L.up);
        launch(s, true, w4a16_gemv_tc8_pair_touch, dim3(HALF / 16, 1, 2), 256u, (const bf*)dA, (const u8*)g.w,
               (const u8*)g.s, SCALE2, c, (const u8*)u.w, (const u8*)u.s, SCALE2, c + (u64)M * HALF, M, HALF, H,
               TOUCH_ROWS_SH, 32u);
    };
    auto down = [&](cudaStream_t s, const Kda& L, bf* c) {
        launch(s, true, w4a16_gemv_tc8_touch, dim3(H / 16), 256u, (const bf*)dA, (const u8*)(L.down.w + HALF / 2),
               (const u8*)(L.down.s + HALF / 16), SCALE2, c, M, H, HALF, HALF, HALF / 8, H, 32u);
    };
    auto mxgemv = [&](cudaStream_t s, const Mx& w, bf* c) {
        launch(s, true, mxfp8_gemv_tc8, dim3((w.n + 15) / 16), 256u, (const bf*)dA, (const u8*)w.w,
               (const u8*)w.s, c, M, w.n, w.k, w.n);
    };

    // ── 1. Bitwise ──
    size_t diff = 0, checked = 0;
    {
        std::vector<unsigned short> o[2] = {std::vector<unsigned short>(32 * 8192), std::vector<unsigned short>(32 * 8192)};
        auto compare = [&](size_t elems, const std::function<void(bf*)>& gemv, const std::vector<Region>& w) {
            for (int mode = 0; mode < M_MODES; mode++) {
                persist_begin(mode);
                for (int v = 0; v < 2; v++) {
                    CK(cudaMemsetAsync(dC[v], v ? 0xEE : 0x11, 32 * 8192 * 2, s0));
                    read(s0, false, flushbuf, 128ull << 20);
                    if (v) l2a(s0, mode, ctas, budgeted(w, budget));
                    gemv(dC[v]);
                }
                CK(cudaStreamSynchronize(s0));
                persist_end(mode);
                for (int v = 0; v < 2; v++) CK(cudaMemcpy(o[v].data(), dC[v], elems * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < elems; i++, checked++) diff += o[0][i] != o[1][i];
            }
        };
        const Kda& L = kda[1];
        std::vector<Region> wq, wsh, wd, wqb, wo;
        push_nv(wq, L.q); push_nv(wsh, gate_rows(L.gate)); push_nv(wsh, gate_rows(L.up)); down_regions(wd, L.down);
        push_mx(wqb, mla[1].qb); push_mx(wo, mla[1].o);
        compare((u64)M * H, [&](bf* c) { qkv(s0, L.q, c); }, wq);
        compare((u64)2 * M * HALF, [&](bf* c) { pair(s0, L, c); }, wsh);
        compare((u64)M * H, [&](bf* c) { down(s0, L, c); }, wd);
        compare((u64)M * 8192, [&](bf* c) { mxgemv(s0, mla[1].qb, c); }, wqb);
        compare((u64)M * H, [&](bf* c) { mxgemv(s0, mla[1].o, c); }, wo);
    }
    printf("bitwise: %zu of %zu outputs differ\n", diff, checked);

    // ── timing harness: lower decile of per-sample GPU time, queued behind a gate ──
    std::vector<cudaEvent_t> ev(2 * 64);
    for (auto& e : ev) CK(cudaEventCreate(&e));
    unsigned int unit = 200;  // spin iterations per us (4 KiB table), calibrated below
    auto sp = [&](cudaStream_t s, bool pdl, unsigned int us, unsigned int mask = 1023u) {
        if (us) launch(s, pdl, spin, dim3(64), 128u, dTab, mask, unit * us);
    };
    auto ar_wait = [&](unsigned int us) { if (us) launch(s0, false, hold, dim3(64), 128u, dTab, 1023u, unit * us); };
    // `body(i, a, b)` records events a and b around what it times, n samples a rep.
    auto time_us = [&](int n, const std::function<void(int, cudaEvent_t, cudaEvent_t)>& body) {
        std::vector<float> us;
        for (int r = 0; r < reps; r++) {
            sp(s0, false, 2000);
            for (int i = 0; i < n; i++) body(i, ev[2 * i], ev[2 * i + 1]);
            CK(cudaStreamSynchronize(s0));
            for (int i = 0; i < n; i++) { float ms; CK(cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); us.push_back(ms * 1e3f); }
        }
        std::sort(us.begin(), us.end());
        return us[us.size() / 10];
    };
    for (int pass = 0; pass < 3; pass++) {
        const float t = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
            CK(cudaEventRecord(a, s0)); sp(s0, false, 50); CK(cudaEventRecord(b, s0)); });
        unit = std::max(1u, (unsigned int)(unit * 50.f / t));
    }
    auto side = [&](int mode, const std::vector<Region>& w) {  // the side stream asks for `w` from here
        CK(cudaEventRecord(fork_ev, s0)); CK(cudaStreamWaitEvent(s1, fork_ev, 0));
        l2a(s1, mode, ctas, budgeted(w, budget));
    };
    printf("spin: %u loads/us; ctas %u, budget %llu MiB, wait %u us\n", unit, ctas, budget >> 20, wait_us);

    // ── 2. Mechanism ──
    printf("\nmechanism: flush, prefetch W, wait gap, read W (res = resident fraction)\n");
    for (u64 mb : {4ull, 8ull, 12ull, 16ull}) {
        const u64 w = mb << 20;
        const float cold = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
            read(s0, false, flushbuf, 128ull << 20); CK(cudaEventRecord(a, s0)); read(s0, false, probe, w); CK(cudaEventRecord(b, s0)); });
        const float hot = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
            read(s0, false, probe, w); CK(cudaEventRecord(a, s0)); read(s0, false, probe, w); CK(cudaEventRecord(b, s0)); });
        printf("  %2llu MB: cold %6.1f us (%5.1f GB/s), hot %6.1f us\n", mb, cold, w / cold / 1e3, hot);
        for (int mode = 0; mode < M_MODES; mode++)
            for (unsigned int c : {4u, 16u, 48u}) {
                persist_begin(mode);
                const float pf = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
                    read(s0, false, flushbuf, 128ull << 20); CK(cudaEventRecord(a, s0));
                    l2a(s0, mode, c, {whole(probe, w)}); CK(cudaEventRecord(b, s0)); });
                printf("    %-8s %2u ctas: pf %6.1f us  res", MODE[mode], c, pf);
                for (unsigned int gap : {0u, 20u, 60u}) {
                    const float t = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
                        read(s0, false, flushbuf, 128ull << 20); l2a(s0, mode, c, {whole(probe, w)}); sp(s0, false, gap);
                        CK(cudaEventRecord(a, s0)); read(s0, false, probe, w); CK(cudaEventRecord(b, s0)); });
                    printf("  %2u us %5.2f", gap, (cold - t) / std::max(cold - hot, 1e-3f));
                }
                printf("\n");
                persist_end(mode);
            }
    }

    // ── 3. Survival ──
    printf("\nsurvival: prefetch 8 MB, stream S MB, read the 8 MB (resident fraction)\n");
    {
        const u64 w = 8ull << 20;
        const float cold = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
            read(s0, false, flushbuf, 128ull << 20); CK(cudaEventRecord(a, s0)); read(s0, false, probe, w); CK(cudaEventRecord(b, s0)); });
        const float hot = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
            read(s0, false, probe, w); CK(cudaEventRecord(a, s0)); read(s0, false, probe, w); CK(cudaEventRecord(b, s0)); });
        const char* how[3] = {"cs", "plain", "cs+l2pf"};
        for (int mode = 0; mode < M_MODES; mode++) {
            persist_begin(mode);
            for (int h = 0; h < 3; h++) {
                printf("  %-8s through %-7s", MODE[mode], how[h]);
                for (u64 smb : {0ull, 32ull, 100ull, 170ull}) {
                    const float t = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
                        read(s0, false, flushbuf, 128ull << 20);
                        l2a(s0, mode, ctas, {whole(probe, w)}); sp(s0, false, 60);
                        if (smb) read(s0, false, moe, smb << 20, h != 1, h == 2 ? 128ull << 10 : 0);
                        CK(cudaEventRecord(a, s0)); read(s0, false, probe, w); CK(cudaEventRecord(b, s0)); });
                    printf("  %3llu MB %5.2f", smb, (cold - t) / std::max(cold - hot, 1e-3f));
                }
                printf("\n");
            }
            persist_end(mode);
        }
    }

    // ── 4. Interference ──
    printf("\ninterference: ten L2-latency PDL kernels (1 MiB table), with a 12 MB prefetch beside them\n");
    {
        auto chain = [&] { for (int i = 0; i < 10; i++) sp(s0, true, 3, (1u << 18) - 1u); };
        const float alone = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
            read(s0, false, flushbuf, 128ull << 20); CK(cudaEventRecord(a, s0)); chain(); CK(cudaEventRecord(b, s0)); });
        printf("  alone %6.1f us\n", alone);
        for (int mode = 0; mode < M_MODES; mode++) {
            persist_begin(mode);
            const float with = time_us(4, [&](int, cudaEvent_t a, cudaEvent_t b) {
                read(s0, false, flushbuf, 128ull << 20); CK(cudaEventRecord(a, s0));
                side(mode, {whole(probe, 12ull << 20)}); chain(); CK(cudaEventRecord(b, s0)); });
            printf("  %-8s %6.1f us (%+5.1f)\n", MODE[mode], with, with - alone);
            persist_end(mode);
        }
    }

    // ── 5. Sites ──
    printf("\nsites (per segment, lower decile; saved = base - arm)\n");
    struct Site { const char* name; int per_step; int layers; std::function<void(int, int, cudaEvent_t, cudaEvent_t)> run; };
    const int BASE = -1;
    auto moe_pass = [&] { read(s0, false, moe, 100ull << 20, 1); };
    std::vector<Site> sites = {
        {"f  wait+chain -> q k v", 34, KDA_LAYERS, [&](int arm, int l, cudaEvent_t a, cudaEvent_t b) {
             const Kda& L = kda[l]; moe_pass(); CK(cudaEventRecord(a, s0));
             if (arm != BASE) { std::vector<Region> w; push_nv(w, L.q); push_nv(w, L.k); side(arm, w); }
             ar_wait(wait_us); sp(s0, true, 35);
             qkv(s0, L.q, dC[0]); qkv(s0, L.k, dC[0]); qkv(s0, L.v, dC[0]); CK(cudaEventRecord(b, s0)); }},
        {"a  wait+chain -> shared, router", 42, KDA_LAYERS, [&](int arm, int l, cudaEvent_t a, cudaEvent_t b) {
             const Kda& L = kda[l]; moe_pass(); qkv(s0, L.v, dC[1]); CK(cudaEventRecord(a, s0));  // o-proj stand-in
             if (arm != BASE) {
                 std::vector<Region> w; push_nv(w, gate_rows(L.gate)); push_nv(w, gate_rows(L.up));
                 down_regions(w, L.down); w.push_back(whole(L.router, ROUTER)); side(arm, w);
             }
             ar_wait(wait_us); sp(s0, true, 35); pair(s0, L, dC[0]); sp(s0, true, 2); down(s0, L, dC[0]);
             read(s0, true, L.router, ROUTER); CK(cudaEventRecord(b, s0)); }},
        {"q  indexer chain -> q_b", 11, MLA_LAYERS, [&](int arm, int l, cudaEvent_t a, cudaEvent_t b) {
             const Mla& L = mla[l]; moe_pass(); CK(cudaEventRecord(a, s0));
             if (arm != BASE) { std::vector<Region> w; push_mx(w, L.qb); side(arm, w); }
             sp(s0, true, 90); mxgemv(s0, L.qb, dC[0]); CK(cudaEventRecord(b, s0)); }},
        {"o  attention -> W_uv, o", 11, MLA_LAYERS, [&](int arm, int l, cudaEvent_t a, cudaEvent_t b) {
             const Mla& L = mla[l]; moe_pass(); mxgemv(s0, L.qb, dC[1]); read(s0, true, L.wuk, WUV);
             CK(cudaEventRecord(a, s0));
             if (arm != BASE) { std::vector<Region> w; w.push_back(whole(L.wuv, WUV)); push_mx(w, L.o); side(arm, w); }
             sp(s0, true, 40); read(s0, true, L.wuv, WUV); mxgemv(s0, L.o, dC[0]); CK(cudaEventRecord(b, s0)); }},
    };
    for (const Site& site : sites) {
        const float base = time_us(site.layers, [&](int l, cudaEvent_t a, cudaEvent_t b) { site.run(BASE, l, a, b); });
        printf("  %-32s base %7.1f us\n", site.name, base);
        for (int mode = 0; mode < M_PERSIST; mode++) {  // a window covers one region only
            persist_begin(mode);
            const float t = time_us(site.layers, [&](int l, cudaEvent_t a, cudaEvent_t b) { site.run(mode, l, a, b); });
            persist_end(mode);
            printf("    %-8s %7.1f us  saved %+6.1f us  /step %+6.2f ms\n", MODE[mode], t, base - t,
                   (base - t) * site.per_step / 1e3);
        }
    }
    CK(cudaDeviceSynchronize());
    printf("%s: bitwise %zu of %zu outputs differ\n", diff == 0 ? "PASS" : "FAIL", diff, checked);
    return diff == 0 ? 0 : 1;
}
