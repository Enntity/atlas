// SPDX-License-Identifier: AGPL-3.0-only
// Per-row bit parity and cost of the qwen4_exp wide exact GEMV tiers
// (ATLAS_QWEN4EXP_BATCH_FAST, crates/spark-model/src/layers/ops/qwen4exp_wide_rows.rs)
// for a 9..32-row batched verify step, against the single-row kernels C1
// decode runs and the 8-row (4-row) chunks the lane ran before.
//
//   BF16 GEMV      qwen4exp_bf16_rows16/32 M = 1..16/32 vs dense_gemv_bf16 per row
//                  (GDN qkvz / out_proj TP2 shards, a ragged LM-head shard)
//   Q+gate         qwen4exp_qg_rows16/32 M = 1..16/32 vs w4a16_gemv_qg per row
//                  (24 heads TP1, 12 heads TP2; head_dim 256)
//   W4A16          w4a16_gemv_batch16/32 M = 1..16/32 vs w4a16_gemv_sw per row
//                  (router 512 x 2560, K/V 512 x 2560, o_proj 2560 x 3072)
//
// Inputs: Gaussian BF16 with +-0, BF16 subnormals and +-1e30 planted, so a
// row's sums overflow to inf; NVFP4 codes over all 16 nibbles and E4M3
// scales including zero and subnormal ones. Each row is checked against a
// launch of the single-row kernel on that row alone, so a pass means the
// row's bits depend neither on M nor on the other rows of the launch.
//
// Then GPU time per launch at M = 1, 4, 8, 16, 24, 32, the old chunks (8-row
// `dense_gemv_bf16_batchm`, 4-row `w4a16_gemv_qg_batch4`, 8-row
// `w4a16_gemv_batch8`) against one wide launch, weights cycled over a ring
// of copies (>= 6x the 24 MiB L2) so each pass streams from DRAM; the
// chunks of one call share a copy, as the runtime's do.
//
// Build/run (repo root, GB10): scripts/dev/qwen4exp_wide_rows_bench.sh check|time|sweep
//
// Measured 2026-10-06 on GB10 (ennspark03): PASS at every M above. us per
// call [GB/s of weight bytes], old chunks -> one wide launch:
//                         M=8          16           24           32
//   GDN qkvz TP2     182 -> 173   398 -> 178   575 -> 258   781 -> 310 [135]
//   GDN out_proj TP2  68 ->  70   115 ->  72   168 -> 100   213 -> 121 [130]
//   LM head half    2802 -> 2625 5555 -> 2553 8379 -> 3863 11094 -> 4487
//   Q+gate 12 heads   83 ->   -   161 -> 130   238 -> 202   324 -> 242
//   router            14 ->   -    27 ->  22    39 ->  35    51 ->  43 (batch16)
//   K/V               12 ->   -    25 ->  21    37 ->  34    49 ->  39 (batch16)
//   o_proj            41 ->   -    80 ->  82   119 -> 129   158 -> 161 (batch16:
//                                                           no gain, not wired)
// w4a16_gemv_batch32 loses to two batch16 passes at every shape here.
#include "qwen4exp_ptx_harness.h"

#include <random>
#include <tuple>

typedef unsigned short bf;
typedef unsigned char u8;

static std::string g_dir = ".";
static int g_fail = 0;
static std::mt19937 g_rng(getenv("SEED") ? atoi(getenv("SEED")) : 31337);

static PtxModule& mod(const char* stem) {
    static std::vector<std::pair<std::string, PtxModule*>> mods;
    for (auto& m : mods) if (m.first == stem) return *m.second;
    PtxModule* m = new PtxModule;
    m->load(g_dir + "/" + stem + ".ptx");
    mods.push_back({stem, m});
    return *m;
}

static std::vector<bf> rbf(size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<bf> v(n);
    for (auto& x : v) {
        const unsigned k = g_rng() % 211;
        const bf sign = (bf)(g_rng() & 0x8000);
        if (k == 0) x = sign;                                  // +-0
        else if (k == 1) x = (bf)(1 + g_rng() % 0x7F) | sign;  // subnormal
        else if (k == 2) x = f2bf(1e30f) | sign;               // overflow bait
        else x = f2bf(d(g_rng));
    }
    return v;
}
static std::vector<u8> rcodes(size_t n) {
    std::vector<u8> v(n);
    for (auto& x : v) x = (u8)g_rng();
    return v;
}
// E4M3 scales: finite, positive, including 0 and the subnormal range.
static std::vector<u8> rscales(size_t n) {
    std::vector<u8> v(n);
    for (auto& x : v) {
        const unsigned k = g_rng() % 64;
        x = k == 0 ? 0 : k == 1 ? (u8)(g_rng() & 7) : (u8)(0x20 + g_rng() % 0x30);
    }
    return v;
}

static void same(const char* what, const std::vector<bf>& ref, const std::vector<bf>& got,
                 size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) d += ref[i] != got[i];
    if (d) {
        printf("  MISMATCH %-56s %zu of %zu\n", what, d, n);
        g_fail++;
    }
}

// ---- BF16 ------------------------------------------------------------------
struct Bf16Tier { const char* fn; unsigned rows, npb; };
static const Bf16Tier BF16_TIERS[] = {{"qwen4exp_bf16_rows16", 16, 4}, {"qwen4exp_bf16_rows32", 32, 8}};

static void bf16_launch(CUfunction f, unsigned npb, const bf* A, const bf* W, bf* C, unsigned m,
                        unsigned N, unsigned K, unsigned stride) {
    Args a;
    a.add(A).add(W).add(C).add(m).add(N).add(K).add(stride);
    launch(f, dim3((N + npb - 1) / npb), dim3(64 * npb), 0, a);
}

static void bf16_check() {
    CUfunction g1 = mod("dense_gemv_bf16").fn("dense_gemv_bf16");
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2", 8192, 2560}, {"GDN out_proj TP2", 2560, 3072}, {"LM head shard", 4101, 2560}};
    for (auto& s : shapes) {
        Buf<bf> W, A, R, G;
        W.alloc((size_t)s.N * s.K); A.alloc((size_t)32 * s.K);
        R.alloc((size_t)32 * s.N); G.alloc((size_t)32 * s.N);
        W.put(rbf(W.n, 0.05f)); A.put(rbf(A.n, 1.0f));
        for (unsigned r = 0; r < 32; r++) {
            Args a;
            a.add(A.p + (size_t)r * s.K).add(W.p).add(R.p + (size_t)r * s.N).add(s.N).add(s.K);
            launch(g1, dim3((s.N + 3) / 4), dim3(256), 0, a);
        }
        const auto ref = R.get();
        for (auto& t : BF16_TIERS) {
            CUfunction f = mod("qwen4exp_wide_rows").fn(t.fn);
            for (unsigned m = 1; m <= t.rows; m++) {
                // Rows [32 - m, 32): a window that is not the first rows.
                const unsigned r0 = 32 - m;
                G.fill(0x55);
                bf16_launch(f, t.npb, A.p + (size_t)r0 * s.K, W.p, G.p, m, s.N, s.K, s.N);
                CK(cudaDeviceSynchronize());
                const auto got = G.get();
                char what[128];
                snprintf(what, sizeof what, "%s M=%u %s", t.fn, m, s.n);
                same(what, std::vector<bf>(ref.begin() + (size_t)r0 * s.N, ref.end()), got,
                     (size_t)m * s.N);
            }
            printf("  %s  %-22s M=1..%u %s vs dense_gemv_bf16 per row\n", g_fail ? "?? " : "ok ",
                   t.fn, t.rows, s.n);
        }
        W.free_(); A.free_(); R.free_(); G.free_();
    }
}

// ---- NVFP4 -----------------------------------------------------------------
struct Fp4 { Buf<u8> packed, scale; float s2; };
static Fp4 make_fp4(unsigned N, unsigned K) {
    Fp4 w;
    w.packed.alloc((size_t)N * K / 2); w.scale.alloc((size_t)N * K / 16);
    w.packed.put(rcodes(w.packed.n)); w.scale.put(rscales(w.scale.n));
    w.s2 = 1.0f / 448.f;
    return w;
}

struct QgTier { const char* fn; unsigned rows, npb; };
static const QgTier QG_TIERS[] = {{"qwen4exp_qg_rows16", 16, 4}, {"qwen4exp_qg_rows32", 32, 8}};

static void qg_launch(CUfunction f, unsigned npb, const bf* A, const Fp4& w, bf* C, unsigned m,
                      unsigned N, unsigned K, unsigned stride, unsigned heads, unsigned hd) {
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C).add(m).add(N).add(K).add(stride)
        .add(heads).add(hd);
    launch(f, dim3((N + npb - 1) / npb), dim3(64 * npb), 0, a);
}
static void qg1_launch(const bf* A, const Fp4& w, bf* C, unsigned N, unsigned K, unsigned heads,
                       unsigned hd) {
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C).add(N).add(K).add(heads).add(hd);
    launch(mod("w4a16_gemv").fn("w4a16_gemv_qg"), dim3((N + 3) / 4), dim3(256), 0, a);
}

static void qg_check() {
    const unsigned K = 2560, hd = 256;
    for (unsigned heads : {12u, 24u}) {
        const unsigned N = heads * hd * 2;
        Fp4 w = make_fp4(N, K);
        Buf<bf> A, R, G;
        A.alloc((size_t)32 * K); R.alloc((size_t)32 * N); G.alloc((size_t)32 * N);
        A.put(rbf(A.n, 1.0f));
        for (unsigned r = 0; r < 32; r++) qg1_launch(A.p + (size_t)r * K, w, R.p + (size_t)r * N, N, K, heads, hd);
        const auto ref = R.get();
        for (auto& t : QG_TIERS) {
            CUfunction f = mod("qwen4exp_wide_rows").fn(t.fn);
            for (unsigned m = 1; m <= t.rows; m++) {
                const unsigned r0 = 32 - m;
                G.fill(0x55);
                qg_launch(f, t.npb, A.p + (size_t)r0 * K, w, G.p, m, N, K, N, heads, hd);
                CK(cudaDeviceSynchronize());
                char what[128];
                snprintf(what, sizeof what, "%s M=%u heads=%u", t.fn, m, heads);
                same(what, std::vector<bf>(ref.begin() + (size_t)r0 * N, ref.end()), G.get(), (size_t)m * N);
            }
            printf("  %s  %-22s M=1..%u %u heads vs w4a16_gemv_qg per row\n", g_fail ? "?? " : "ok ",
                   t.fn, t.rows, heads);
        }
        A.free_(); R.free_(); G.free_();
    }
}

static void w4_launch(const char* fn, const bf* A, const Fp4& w, bf* C, unsigned m, unsigned N,
                      unsigned K) {
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C).add(m).add(N).add(K);
    launch(mod("w4a16_gemv").fn(fn), dim3((N + 3) / 4), dim3(256), 0, a);
}

static void w4_check() {
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"router 512x2560", 512, 2560}, {"o_proj TP2 2560x3072", 2560, 3072}, {"K/V 256x2560", 256, 2560}};
    for (auto& s : shapes) {
        Fp4 w = make_fp4(s.N, s.K);
        Buf<bf> A, R, G;
        A.alloc((size_t)32 * s.K); R.alloc((size_t)32 * s.N); G.alloc((size_t)32 * s.N);
        A.put(rbf(A.n, 1.0f));
        for (unsigned r = 0; r < 32; r++) {
            Args a;
            a.add(A.p + (size_t)r * s.K).add(w.packed.p).add(w.scale.p).add(w.s2)
                .add(R.p + (size_t)r * s.N).add(s.N).add(s.K);
            launch(mod("w4a16_gemv").fn("w4a16_gemv_sw"), dim3((s.N + 7) / 8), dim3(256), 0, a);
        }
        const auto ref = R.get();
        for (auto& tier : {std::make_tuple("w4a16_gemv_batch16", 16u), std::make_tuple("w4a16_gemv_batch32", 32u)}) {
            const char* fn = std::get<0>(tier);
            const unsigned rows = std::get<1>(tier);
            for (unsigned m = 1; m <= rows; m++) {
                const unsigned r0 = 32 - m;
                G.fill(0x55);
                w4_launch(fn, A.p + (size_t)r0 * s.K, w, G.p, m, s.N, s.K);
                CK(cudaDeviceSynchronize());
                char what[128];
                snprintf(what, sizeof what, "%s M=%u %s", fn, m, s.n);
                same(what, std::vector<bf>(ref.begin() + (size_t)r0 * s.N, ref.end()), G.get(), (size_t)m * s.N);
            }
            printf("  %s  %-22s M=1..%u %s vs w4a16_gemv_sw per row\n", g_fail ? "?? " : "ok ", fn,
                   rows, s.n);
        }
        A.free_(); R.free_(); G.free_();
    }
}

// ---- timing ----------------------------------------------------------------
static const unsigned TIME_M[] = {1, 4, 8, 16, 24, 32};

// One row of the table: `body(m, copy)` timed at each M.
template <typename F>
static void time_row(const char* name, double bytes, int copies, int iters, F body) {
    printf("  %-26s", name);
    int it = 0;
    for (unsigned m : TIME_M) {
        const float ms = time_ms([&] { body(m, it++ % copies); }, iters, 5);
        printf(" %7.1f [%3.0f]", ms * 1e3, bytes / (ms * 1e-3) / 1e9);
    }
    printf("\n");
}
static void time_header(const char* what, double bytes) {
    printf("\n%s (%.1f MB): us per call [GB/s of weight bytes] at M =", what, bytes / 1e6);
    for (unsigned m : TIME_M) printf(" %8u    ", m);
    printf("\n");
}
// Chunks of at most `cap` rows: `one(first, rows)` per chunk.
template <typename F>
static void chunks(unsigned m, unsigned cap, F one) {
    for (unsigned f = 0; f < m; f += cap) one(f, std::min(cap, m - f));
}

static void bf16_time(bool sweep) {
    CUfunction bm = mod("dense_gemv_bf16_batchm").fn("dense_gemv_bf16_batchm");
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2 8192x2560", 8192, 2560}, {"GDN out_proj TP2 2560x3072", 2560, 3072},
        {"LM head half 124160x2560", 124160, 2560}};
    std::vector<std::tuple<const char*, unsigned, unsigned>> tiers = {
        {"qwen4exp_bf16_rows16", 16, 4}, {"qwen4exp_bf16_rows32", 32, 8}};
    if (sweep)
        tiers.insert(tiers.end(), {{"qw_sweep_bf16_m16_n8_s2", 16, 8}, {"qw_sweep_bf16_m32_n16_s2", 32, 16},
                                   {"qw_sweep_bf16_m32_n4_s1", 32, 4}});
    for (auto& s : shapes) {
        const double wb = (double)s.N * s.K * 2;
        const int copies = std::max(2, (int)(160e6 / wb) + 1);
        std::vector<Buf<bf>> W(copies);
        for (auto& w : W) w.alloc((size_t)s.N * s.K);
        Buf<bf> A, C;
        A.alloc((size_t)32 * s.K); C.alloc((size_t)32 * s.N);
        A.put(rbf(A.n, 1.0f));
        const int iters = s.N > 100000 ? 4 : 20;
        time_header(s.n, wb);
        time_row("8-row batchm chunks", wb, copies, iters, [&](unsigned m, int c) {
            chunks(m, 8, [&](unsigned f, unsigned r) {
                Args a;
                a.add(A.p + (size_t)f * s.K).add(W[c].p).add(C.p + (size_t)f * s.N).add(r).add(s.N).add(s.K).add(s.N);
                launch(bm, dim3((s.N + 3) / 4), dim3(256), 0, a);
            });
        });
        for (auto& tier : tiers) {
            const char* fn = std::get<0>(tier);
            const unsigned rows = std::get<1>(tier), npb = std::get<2>(tier);
            CUfunction f = mod("qwen4exp_wide_rows").fn(fn);
            time_row(fn, wb, copies, iters, [&](unsigned m, int c) {
                chunks(m, rows, [&](unsigned fr, unsigned r) {
                    bf16_launch(f, npb, A.p + (size_t)fr * s.K, W[c].p, C.p + (size_t)fr * s.N, r, s.N, s.K, s.N);
                });
            });
        }
        for (auto& w : W) w.free_();
        A.free_(); C.free_();
    }
}

static void fp4_time(bool sweep) {
    const unsigned K = 2560, hd = 256;
    // Q+gate.
    for (unsigned heads : {12u, 24u}) {
        const unsigned N = heads * hd * 2;
        const double wb = (double)N * K / 2 + (double)N * K / 16;
        const int copies = std::max(2, (int)(160e6 / wb) + 1);
        std::vector<Fp4> W;
        for (int c = 0; c < copies; c++) W.push_back(make_fp4(N, K));
        Buf<bf> A, C;
        A.alloc((size_t)32 * K); C.alloc((size_t)32 * N);
        A.put(rbf(A.n, 1.0f));
        char title[64];
        snprintf(title, sizeof title, "Q+gate %u heads %ux%u", heads, N, K);
        time_header(title, wb);
        CUfunction q4 = mod("w4a16_gemv").fn("w4a16_gemv_qg_batch4");
        time_row("4-row qg_batch4 chunks", wb, copies, 20, [&](unsigned m, int c) {
            chunks(m, 4, [&](unsigned f, unsigned r) {
                if (r == 4) {
                    Args a;
                    a.add(A.p + (size_t)f * K).add(W[c].packed.p).add(W[c].scale.p).add(W[c].s2)
                        .add(C.p + (size_t)f * N).add(N).add(K).add(heads).add(hd);
                    launch(q4, dim3((N + 3) / 4), dim3(256), 0, a);
                } else {
                    for (unsigned i = 0; i < r; i++)
                        qg1_launch(A.p + (size_t)(f + i) * K, W[c], C.p + (size_t)(f + i) * N, N, K, heads, hd);
                }
            });
        });
        std::vector<std::tuple<const char*, unsigned, unsigned>> tiers = {
            {"qwen4exp_qg_rows16", 16, 4}, {"qwen4exp_qg_rows32", 32, 8}};
        if (sweep)
            tiers.insert(tiers.end(), {{"qw_sweep_qg_m16_n8_s2", 16, 8}, {"qw_sweep_qg_m32_n16_s2", 32, 16},
                                       {"qw_sweep_qg_m8_n4_s2", 8, 4}});
        for (auto& tier : tiers) {
            const char* fn = std::get<0>(tier);
            const unsigned rows = std::get<1>(tier), npb = std::get<2>(tier);
            CUfunction f = mod("qwen4exp_wide_rows").fn(fn);
            time_row(fn, wb, copies, 20, [&](unsigned m, int c) {
                chunks(m, rows, [&](unsigned fr, unsigned r) {
                    qg_launch(f, npb, A.p + (size_t)fr * K, W[c], C.p + (size_t)fr * N, r, N, K, N, heads, hd);
                });
            });
        }
        for (auto& w : W) { w.packed.free_(); w.scale.free_(); }
        A.free_(); C.free_();
    }
    // The scalar template tiers.
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"router 512x2560", 512, 2560}, {"o_proj TP2 2560x3072", 2560, 3072}, {"K/V 256x2560", 256, 2560}};
    for (auto& s : shapes) {
        const double wb = (double)s.N * s.K / 2 + (double)s.N * s.K / 16;
        const int copies = std::max(2, (int)(160e6 / wb) + 1);
        std::vector<Fp4> W;
        for (int c = 0; c < copies; c++) W.push_back(make_fp4(s.N, s.K));
        Buf<bf> A, C;
        A.alloc((size_t)32 * s.K); C.alloc((size_t)32 * s.N);
        A.put(rbf(A.n, 1.0f));
        time_header(s.n, wb);
        for (auto& tier : {std::make_tuple("8-row batch8 chunks", 8u, "w4a16_gemv_batch8"),
                           std::make_tuple("w4a16_gemv_batch16", 16u, "w4a16_gemv_batch16"),
                           std::make_tuple("w4a16_gemv_batch32", 32u, "w4a16_gemv_batch32")}) {
            const char* name = std::get<0>(tier);
            const unsigned cap = std::get<1>(tier);
            const char* fn = std::get<2>(tier);
            time_row(name, wb, copies, 50, [&](unsigned m, int c) {
                chunks(m, cap, [&](unsigned f, unsigned r) {
                    w4_launch(fn, A.p + (size_t)f * s.K, W[c], C.p + (size_t)f * s.N, r, s.N, s.K);
                });
            });
        }
        for (auto& w : W) { w.packed.free_(); w.scale.free_(); }
        A.free_(); C.free_();
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    init_driver();
    if (mode == "check") {
        printf("per-row parity:\n");
        bf16_check();
        qg_check();
        w4_check();
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    const bool sweep = mode == "sweep";
    bf16_time(sweep);
    fp4_time(sweep);
    return 0;
}
