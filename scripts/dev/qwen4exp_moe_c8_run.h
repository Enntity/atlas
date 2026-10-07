// SPDX-License-Identifier: AGPL-3.0-only
// The `time` and `check` drivers of scripts/dev/qwen4exp_moe_c8_bench.cu.
#pragma once
#include "qwen4exp_moe_c8_bench.h"

static int g_fail = 0;
static bool same(const std::string& what, const std::vector<unsigned char>& a,
                 const std::vector<unsigned char>& b) {
    size_t diff = 0, first = 0;
    for (size_t i = 0; i + 1 < a.size(); i += 2)
        if (a[i] != b[i] || a[i + 1] != b[i + 1]) { if (!diff) first = i / 2; diff++; }
    if (diff) {
        printf("  MISMATCH %-56s %zu of %zu values (first %zu)\n", what.c_str(), diff, a.size() / 2, first);
        g_fail++;
    }
    return diff == 0;
}

// Variants whose gate/up intermediates are not materialized say so by name
// ("fused"): only the down outputs are compared.
static int run_check(Pool& pool) {
    Variant ref = production();
    std::vector<Variant> vs = variants();
    Bufs rb = alloc_bufs(), gb = alloc_bufs();
    std::vector<int> bad(vs.size(), 0);
    std::mt19937 rng(5);
    const size_t rk = (size_t)MAXR * TOPK;
    for (unsigned rows : {1u, 2u, 3u, 4u, 5u, 8u, 12u, 16u, 24u, 32u, 33u, 40u, 48u, 64u}) {
        for (double reuse : {0.0, 0.5, 0.9, 1.0}) {
            for (float scale : {0.02f, 1.0f, 30.0f}) {
                auto a = rand_bf16((size_t)rows * H, scale);
                auto ids = c8_routes(rows, reuse, rng);
                if (reuse == 1.0)  // every row row 0's picks, shuffled
                    for (unsigned r = 1; r < rows; r++) {
                        std::copy(ids.begin(), ids.begin() + TOPK, ids.begin() + r * TOPK);
                        std::shuffle(ids.begin() + r * TOPK, ids.begin() + (r + 1) * TOPK, rng);
                    }
                for (Bufs* b : {&rb, &gb}) {
                    CK(cudaMemcpy(b->A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(b->ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                }
                ref.plan(pool, rb, rows); ref.gate_up(pool, rb, rows); ref.silu_down(pool, rb, rows);
                const size_t n = (size_t)rows * TOPK;
                auto r_gate = dget(rb.gate, n * I * 2), r_up = dget(rb.up, n * I * 2);
                auto r_shg = dget(rb.shg, rows * I * 2), r_shu = dget(rb.shu, rows * I * 2);
                auto r_down = dget(rb.down, n * H * 2), r_shd = dget(rb.shd, rows * H * 2);
                for (size_t vi = 0; vi < vs.size(); vi++) {
                    Variant& v = vs[vi];
                    for (void* p : {gb.gate, gb.up}) CK(cudaMemset(p, 0x55, rk * I * 2));
                    for (void* p : {gb.shg, gb.shu}) CK(cudaMemset(p, 0x55, (size_t)MAXR * I * 2));
                    CK(cudaMemset(gb.down, 0x55, rk * H * 2));
                    CK(cudaMemset(gb.shd, 0x55, (size_t)MAXR * H * 2));
                    v.plan(pool, gb, rows); v.gate_up(pool, gb, rows); v.silu_down(pool, gb, rows);
                    char tag[128];
                    snprintf(tag, sizeof tag, "%s R=%u reuse=%.1f x%g", v.name.c_str(), rows, reuse, scale);
                    const std::string t(tag);
                    bool ok = true;
                    if (v.name.find("fused") == std::string::npos) {
                        ok &= same(t + " gate", r_gate, dget(gb.gate, n * I * 2));
                        ok &= same(t + " up", r_up, dget(gb.up, n * I * 2));
                        ok &= same(t + " sh gate", r_shg, dget(gb.shg, rows * I * 2));
                        ok &= same(t + " sh up", r_shu, dget(gb.shu, rows * I * 2));
                    }
                    ok &= same(t + " down", r_down, dget(gb.down, n * H * 2));
                    ok &= same(t + " sh down", r_shd, dget(gb.shd, rows * H * 2));
                    bad[vi] += !ok;
                }
            }
        }
    }
    for (size_t vi = 0; vi < vs.size(); vi++)
        printf("  %s  %-36s R = 1..64; reuse 0 / 0.5 / 0.9 / identical; 3 scales; every output byte "
               "vs the production rows pair\n", bad[vi] ? "BAD" : "ok ", vs[vi].name.c_str());
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

// us per launch over `iters` route sets (each its own plan), GB/s of the
// unique local expert bytes + the shared expert.
static void run_time(Pool& pool, const std::string& only) {
    std::vector<Variant> vs;
    vs.push_back(production());
    for (auto& v : variants())
        if (only.empty() || v.name.find(only) != std::string::npos) vs.push_back(v);
    Bufs base = alloc_bufs();
    Timer tm;
    const int iters = 40;
    const double gu_e = 2.0 * (I * H / 2 + I * H / 16), sd_e = H * I / 2 + H * I / 16;
    struct Cfg { const char* name; unsigned rows; double uniq; };
    const char* trace = getenv("ROUTES");
    std::vector<Cfg> cfgs = {{"C1 decode", 1, 0}, {"C1 verify", 4, 0}, {"C4 verify", 16, 40},
                             {"C8 u50", 32, 50}, {"C8 u70", 32, 70}, {"C8 u90", 32, 90},
                             {"C8 indep", 32, 0}};
    if (trace) cfgs = {{"trace", (unsigned)atoi(getenv("ROUTE_ROWS") ? getenv("ROUTE_ROWS") : "32"), 0}};
    printf("\n%-10s %-24s %8s %16s %16s %8s\n", "routing", "variant", "plan", "gate_up (GB/s)",
           "silu_down (GB/s)", "layer");
    const char* only_cfg = getenv("CFG");
    for (const Cfg& c : cfgs) {
        if (only_cfg && std::string(c.name) != only_cfg) continue;
        std::vector<std::vector<unsigned>> routes;
        if (trace) {
            routes = trace_routes(trace, c.rows);
            if (routes.empty()) { printf("no %u-row launches in %s\n", c.rows, trace); return; }
        } else {
            const double reuse = c.uniq > 0 ? solve_reuse(c.rows, c.uniq) : 0.0;
            std::mt19937 rng(11);
            for (int i = 0; i < iters; i++) routes.push_back(c8_routes(c.rows, reuse, rng));
        }
        const int n = (int)routes.size();
        double uniq = 0;
        std::vector<Bufs> bs;
        for (auto& r : routes) {
            uniq += local_unique(r);
            Bufs b = base;
            b.ids = dput(r);
            b.order = dzero(MAXR * TOPK * 4 + 4096);
            b.ws = dzero(1 << 20);
            bs.push_back(b);
        }
        uniq /= n;
        const double gu_b = (uniq + 1) * gu_e, sd_b = (uniq + 1) * sd_e;
        printf("%-10s rows %u, %.1f unique local experts (+ shared), %.1f + %.1f MB\n", c.name, c.rows,
               uniq, gu_b / 1e6, sd_b / 1e6);
        for (Variant& v : vs) {
            for (int i = 0; i < n; i++) v.plan(pool, bs[i], c.rows);
            const double pl = tm.run(n, [&](int i) { v.plan(pool, bs[i], c.rows); });
            const double gu = tm.run(n, [&](int i) { v.gate_up(pool, bs[i], c.rows); });
            const double sd = tm.run(n, [&](int i) { v.silu_down(pool, bs[i], c.rows); });
            const double all = tm.run(n, [&](int i) {
                v.plan(pool, bs[i], c.rows); v.gate_up(pool, bs[i], c.rows); v.silu_down(pool, bs[i], c.rows);
            });
            printf("%-10s %-24s %8.1f %8.1f (%5.0f) %8.1f (%5.0f) %8.1f\n", "", v.name.c_str(), pl, gu,
                   gu_b / gu / 1e3, sd, sd_b / sd / 1e3, all);
        }
        for (auto& b : bs) { CK(cudaFree(b.ids)); CK(cudaFree(b.order)); CK(cudaFree(b.ws)); }
    }
}
