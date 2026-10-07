// SPDX-License-Identifier: AGPL-3.0-only
// C8 verify-row MoE (qwen4_exp, EP2 rank: 256 local experts of 512, top-10,
// hidden 2560, intermediate 640, shared expert 640) -- where the routed MoE's
// time goes and what the exact alternatives cost.
//
//   probe   pure DRAM read of the unique expert bytes a C8 step touches, in
//           the access shapes the kernels use (contiguous, per-tile chunks)
//   time    qwen4exp_moe_rows_{gate_up,silu_down} (production) vs the
//           qwen4exp_moe_c8 variants, us/launch and GB/s of unique bytes,
//           at C1 (1 row) .. C8 (32 rows) routing
//   check   every variant's output bytes vs the production rows kernels
//           (which equal serial decode's single-row kernels:
//           qwen4exp_batch_exact_bench.cu) over 1..64 rows, varied overlap
//
// Routing: R rows = S sequences x 4 verify rows. Each pick reuses, with
// probability `reuse`, an expert an earlier row of the launch picked (the
// same sequence's first), else a uniform draw of 512; `reuse` is solved so
// the local unique count hits the target. Or a trace (`ROUTES=file`: lines of
// R*10 expert ids, one launch per line; ATLAS_QWEN4EXP_MOE_ROUTE_DUMP).
//
// POOL=3 (time): every local id on 3 physical experts (~8 MB, L2-resident):
// the kernels' compute floor at the same routing. CFG="C8 u70" picks one
// routing row.
//
// Build/run: scripts/dev/qwen4exp_moe_c8_bench.sh [probe|time|check] (repo root, GB10).
//
// Preliminary (2026-10-06, ennspark03, single runs, synthetic routing):
//   probe: the unique bytes read in 10-64 KB chunks a CTA run at 232-249 GB/s
//     (vLLM's routed MoE in the C8 prose profile is ~242-248 GB/s of ~70
//     unique experts' bytes: at this roofline).
//   time, 32 rows, ~70 unique: rows pair 750 + 463 us (172 / 138 GB/s),
//     POOL=3 floor 544 + 385 us -- compute-bound, not DRAM-bound.
#include "qwen4exp_moe_c8_run.h"
#include "qwen4exp_moe_c8_variants.h"

// ── probes: read `bytes` at `src` chunks, CTA-per-chunk, 16-byte loads ──
__global__ void probe_chunks(const uint4* const* chunks, unsigned chunk_bytes, unsigned n_chunks,
                             unsigned* sink) {
    const unsigned c = blockIdx.x;
    if (c >= n_chunks) return;
    const uint4* p = chunks[c];
    unsigned acc = 0;
    for (unsigned i = threadIdx.x; i < chunk_bytes / 16; i += blockDim.x) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

// Persistent: each CTA walks chunk c = blockIdx.x, + gridDim.x, ...
__global__ void probe_persist(const uint4* const* chunks, unsigned chunk_bytes, unsigned n_chunks,
                              unsigned* sink) {
    unsigned acc = 0;
    for (unsigned c = blockIdx.x; c < n_chunks; c += gridDim.x) {
        const uint4* p = chunks[c];
#pragma unroll 4
        for (unsigned i = threadIdx.x; i < chunk_bytes / 16; i += blockDim.x) {
            const uint4 v = __ldcs(p + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

static void probe(Pool& pool) {
    printf("\nprobe: DRAM read of the unique expert bytes (us, GB/s); 256-expert pool %.0f MB\n",
           pool.bytes() / 1e6);
    unsigned* sink = (unsigned*)dzero(64);
    Timer tm;
    for (unsigned uniq : {5u, 35u, 70u, 100u}) {
        // `uniq` distinct local experts per launch, a fresh set each iteration.
        const int iters = 50;
        for (unsigned which = 0; which < 3; which++) {
            const char* names[3] = {"gate+up", "down", "all"};
            for (unsigned chunk : {10240u, 23040u, 65536u, 0u}) {
                std::vector<const uint4**> lists;
                std::vector<unsigned> counts;
                unsigned cb = 0;
                for (int it = 0; it < iters; it++) {
                    std::vector<const void*> regions;
                    std::vector<size_t> sizes;
                    for (unsigned e : pool.draw_distinct(uniq)) {
                        if (which != 1) {
                            regions.push_back(pool.gate[e].packed); sizes.push_back(I * H / 2);
                            regions.push_back(pool.up[e].packed); sizes.push_back(I * H / 2);
                        }
                        if (which != 0) { regions.push_back(pool.down[e].packed); sizes.push_back(H * I / 2); }
                    }
                    cb = chunk ? chunk : 4096;
                    std::vector<const uint4*> ch;
                    for (size_t r = 0; r < regions.size(); r++)
                        for (size_t o = 0; o + cb <= sizes[r]; o += cb)
                            ch.push_back((const uint4*)((const char*)regions[r] + o));
                    lists.push_back(dput(ch));
                    counts.push_back((unsigned)ch.size());
                }
                const double bytes = (double)counts[0] * cb;
                double us;
                if (chunk) {
                    us = tm.run(iters, [&](int i) {
                        probe_chunks<<<counts[i], 256>>>(lists[i], cb, counts[i], sink);
                    });
                } else {
                    us = tm.run(iters, [&](int i) {
                        probe_persist<<<48 * 4, 512>>>(lists[i], cb, counts[i], sink);
                    });
                }
                printf("  uniq %3u %-8s %-16s %8.1f us %6.1f GB/s  (%.1f MB)\n", uniq, names[which],
                       chunk ? (std::to_string(chunk) + " B/CTA").c_str() : "persistent 4 KB",
                       us, bytes / us / 1e3, bytes / 1e6);
                for (auto* l : lists) CK(cudaFree(l));
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "time";
    CU(cuInit(0));
    CK(cudaFree(0));
    // POOL=n: n physical experts behind the 256 local ids (POOL=3: ~8 MB, L2
// resident -- the kernels' compute floor at the same routing).
    Pool pool(mode == "check" ? 24 : getenv("POOL") ? atoi(getenv("POOL")) : 256);
    if (mode == "probe") { probe(pool); return 0; }
    if (mode == "check") return run_check(pool);
    run_time(pool, argc > 3 ? argv[3] : "");
    return 0;
}
