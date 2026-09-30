// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise A/B of the GLM semantic-index scorers:
// glm_index_logits_bf16_wmma_row8_pool32 (production) versus
// glm_index_logits_bf16_mma_v2. Realistic data: 32 heads x 128 BF16 queries,
// BF16 head weights, pooled BF16 keys (kpool 4) in a shuffled 16-token block
// table (plus one 64-token table case). Every case compares every FP32 logit
// bit for bit, over several v2 grid widths, with distinct output sentinels so
// an unwritten logit also counts as a difference. Then times 4096-row pieces.
//
//   nvcc -arch=sm_121f -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_index_logits_v2_bench.cu -o v2_bench
//   ./v2_bench [iters=7] [mode=all|check|time|prof] [max_extent=135168]
// Device memory: ~1.2 GB at the defaults. Exit: 0 bit-identical, 1 otherwise.
//
// A GPU time-sliced with other processes stretches long launches, so timing
// reports the fastest of `iters` launches of the whole piece and, as an
// uncontended estimate, the sum of per-tile minimums over 256-row tiles.
#include "glm_indexer_wmma.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define V2_ARGS const __nv_bfloat16* q, const __nv_bfloat16* w, const __nv_bfloat16* c, \
    float* o, const unsigned* bt, unsigned rows, unsigned start, unsigned stride, unsigned heads, \
    unsigned dim, unsigned pool, unsigned bs, unsigned long long bstride
#define V2_PASS q, w, c, o, bt, rows, start, stride, heads, dim, pool, bs, bstride

// Sweep variants beside the exported 8-warp, 32-pool configuration.
template <unsigned W, unsigned P, unsigned MinBlocks>
__global__ void __launch_bounds__(W * 32, MinBlocks) v2_variant(V2_ARGS) {
    glm_index_logits_mma_v2_impl<W, P>(V2_PASS);
}

typedef void (*Kern)(V2_ARGS);
struct Variant { const char* name; Kern k; unsigned warps, pools, smem; };
static const Variant kVariants[] = {
    {"v2_w8_p32 (export)", glm_index_logits_bf16_mma_v2, 8, 32, index_v2_smem_bytes<8, 32>()},
    {"v2_w4_p32", v2_variant<4, 32, 3>, 4, 32, index_v2_smem_bytes<4, 32>()},
    {"v2_w8_p64", v2_variant<8, 64, 1>, 8, 64, index_v2_smem_bytes<8, 64>()},
};

static unsigned short f2bf(float f) {
    const __nv_bfloat16 b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}

struct Cache { unsigned bs; unsigned long long stride; void *dcache, *dtable; };

// Pooled keys for `tokens` positions laid out as the runtime does: block b of
// the logical sequence lives at physical block table[b], bs/4 pools of 256 B.
static Cache make_cache(unsigned bs, unsigned tokens, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const unsigned blocks = (tokens + bs - 1) / bs;
    std::vector<unsigned> table(blocks);
    for (unsigned i = 0; i < blocks; i++) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    const unsigned long long stride = (unsigned long long)(bs / 4) * 256;
    std::vector<unsigned short> cache((size_t)blocks * stride / 2);
    for (auto& x : cache) x = f2bf(nd(rng));
    Cache c{bs, stride, nullptr, nullptr};
    CK(cudaMalloc(&c.dcache, cache.size() * 2));
    CK(cudaMalloc(&c.dtable, table.size() * 4));
    CK(cudaMemcpy(c.dcache, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.dtable, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    return c;
}

struct Bufs { __nv_bfloat16 *dq, *dw; float *dout0, *dout1; };

// One scorer launch over rows [r0, r0 + rows) of a piece starting at `start`,
// exactly as the runtime's row-tile loop offsets query, weights and position.
// v == nullptr selects the production WMMA kernel.
static void launch(const Variant* v, unsigned gx, const Bufs& b, const Cache& c, unsigned r0,
                   unsigned rows, unsigned start, unsigned stride) {
    const __nv_bfloat16* q = b.dq + (size_t)r0 * 32 * 128;
    const __nv_bfloat16* w = b.dw + (size_t)r0 * 32;
    const auto* cache = (const __nv_bfloat16*)c.dcache;
    const auto* table = (const unsigned*)c.dtable;
    if (!v) {
        glm_index_logits_bf16_wmma_row8_pool32<<<dim3((stride + 31) / 32, (rows + 7) / 8), 256>>>(
            q, w, cache, b.dout0 + (size_t)r0 * stride, table, rows, start + r0, stride, 32, 128,
            4, c.bs, c.stride);
    } else {
        v->k<<<dim3(gx, (rows + v->warps - 1) / v->warps), v->warps * 32, v->smem>>>(
            q, w, cache, b.dout1 + (size_t)r0 * stride, table, rows, start + r0, stride, 32, 128,
            4, c.bs, c.stride);
    }
}

// Mirrors index_logits_launch's v2 grid width (glm_indexer.rs).
static unsigned default_gx(const Variant& v, unsigned rows, unsigned stride) {
    const unsigned chunks = (stride + v.pools - 1) / v.pools;
    const unsigned row_blocks = (rows + v.warps - 1) / v.warps;
    const unsigned want = std::max(1u, (48u * 2 * 8 + row_blocks - 1) / row_blocks);
    return std::max(1u, std::min(want, chunks / 16));
}

static size_t compare(const Bufs& b, unsigned rows, unsigned stride, std::vector<unsigned>& h0,
                      std::vector<unsigned>& h1) {
    const size_t n = (size_t)rows * stride;
    h0.resize(n);
    h1.resize(n);
    CK(cudaMemcpy(h0.data(), b.dout0, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h1.data(), b.dout1, n * 4, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) diff += h0[i] != h1[i];
    return diff;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 7;
    const std::string mode = argc > 2 ? argv[2] : "all";
    const unsigned max_extent = argc > 3 ? atoi(argv[3]) : 135168;
    const unsigned max_rows = 4096;
    std::mt19937 rng(20260929);
    std::normal_distribution<float> nd(0.f, 1.f);

    for (const auto& v : kVariants) {
        CK(cudaFuncSetAttribute(v.k, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem));
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, v.k));
        int occ = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, v.k, v.warps * 32, v.smem));
        printf("%-20s regs=%d local=%zu smem=%u ctas/sm=%d\n", v.name, fa.numRegs,
               fa.localSizeBytes, v.smem, occ);
    }

    // Query magnitudes vary per head so ReLU clips a realistic mix of dots.
    std::vector<unsigned short> q((size_t)max_rows * 32 * 128), w((size_t)max_rows * 32);
    for (size_t i = 0; i < q.size(); i++) q[i] = f2bf(nd(rng) * (0.25f + 0.05f * ((i / 128) % 32)));
    for (auto& x : w) x = f2bf(nd(rng) * 0.5f);
    Bufs b;
    const size_t out_bytes = (size_t)max_rows * ((max_extent + 3) / 4) * 4;
    CK(cudaMalloc(&b.dq, q.size() * 2));
    CK(cudaMalloc(&b.dw, w.size() * 2));
    CK(cudaMalloc(&b.dout0, out_bytes));
    CK(cudaMalloc(&b.dout1, out_bytes));
    CK(cudaMemcpy(b.dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    const Cache c16 = make_cache(16, max_extent, rng);
    const Cache c64 = make_cache(64, 40000, rng);

    if (mode == "prof") {  // one launch of each kernel on the 57K reference piece, for ncu
        const unsigned rows = 4096, start = 57344, stride = (start + rows + 3) / 4;
        launch(nullptr, 0, b, c16, 0, rows, start, stride);
        launch(&kVariants[0], default_gx(kVariants[0], rows, stride), b, c16, 0, rows, start, stride);
        CK(cudaDeviceSynchronize());
        return 0;
    }
    size_t total_diff = 0;
    if (mode != "time") {
        // (rows, seq_len_start, cache): partial row tiles, unaligned starts,
        // the all-future rows of a start-0 piece, first-chunk and long pieces.
        struct Case { unsigned rows, start; const Cache* c; };
        const Case cases[] = {
            {1, 2051, &c16},    {8, 2044, &c16},     {9, 2047, &c16},     {13, 4093, &c16},
            {4096, 0, &c16},    {2052, 2048, &c16},  {4096, 2052, &c16},  {777, 30001, &c16},
            {4096, 8196, &c16}, {100, 131000, &c16}, {4096, 57348, &c16},
            {4096, max_extent - 4096, &c16}, {1000, 30000, &c64},
        };
        std::vector<unsigned> h0, h1;
        size_t compared = 0;
        for (const auto& cs : cases) {
            const unsigned stride = (cs.start + cs.rows + 3) / 4;
            CK(cudaMemset(b.dout0, 0xFF, (size_t)cs.rows * stride * 4));
            launch(nullptr, 0, b, *cs.c, 0, cs.rows, cs.start, stride);
            CK(cudaGetLastError());
            for (const auto& v : kVariants) {
                const unsigned chunks = (stride + v.pools - 1) / v.pools;
                const unsigned gxs[] = {1, 2, 3, 7, chunks, default_gx(v, cs.rows, stride)};
                for (unsigned gx : gxs) {
                    CK(cudaMemset(b.dout1, 0xEE, (size_t)cs.rows * stride * 4));
                    launch(&v, gx, b, *cs.c, 0, cs.rows, cs.start, stride);
                    CK(cudaGetLastError());
                    CK(cudaDeviceSynchronize());
                    const size_t diff = compare(b, cs.rows, stride, h0, h1);
                    total_diff += diff;
                    compared += (size_t)cs.rows * stride;
                    if (diff) {
                        printf("DIFF %s rows=%u start=%u stride=%u bs=%u gx=%u: %zu differ\n",
                               v.name, cs.rows, cs.start, stride, cs.c->bs, gx, diff);
                    }
                }
            }
            printf("case rows=%u start=%u stride=%u bs=%u checked\n", cs.rows, cs.start, stride,
                   cs.c->bs);
        }
        printf("bitwise: %zu differing of %zu compared logits\n", total_diff, compared);
    }
    if (mode == "check") return total_diff == 0 ? 0 : 1;

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto best_ms = [&](auto&& fn) {
        fn();
        float best = 1e30f;
        for (int i = 0; i < iters; i++) {
            CK(cudaEventRecord(e0));
            fn();
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        return best;
    };
    // Whole-piece minimum, and the sum of 256-row tile minimums.
    auto measure = [&](const Variant* v, unsigned gx, unsigned rows, unsigned start,
                       unsigned stride, float& whole, float& tiled) {
        whole = best_ms([&] { launch(v, gx, b, c16, 0, rows, start, stride); });
        tiled = 0;
        for (unsigned r0 = 0; r0 < rows; r0 += 256) {
            const unsigned n = std::min(256u, rows - r0);
            const unsigned tgx = v ? default_gx(*v, n, stride) : 0;
            tiled += best_ms([&] { launch(v, tgx, b, c16, r0, n, start, stride); });
        }
    };
    // Extents 2K..131K (61,440 + 4096 is the scout's 57K reference call).
    const unsigned extents[] = {2048, 8192, 16384, 32768, 57344, 65536, 131072};
    for (unsigned extent : extents) {
        const unsigned rows = 4096, end = std::min(extent + rows, max_extent);
        const unsigned start = end - rows, stride = (end + 3) / 4;
        const double gflop = (double)rows * stride * 32 * 128 * 2 / 1e9;
        float old_whole, old_tiled;
        measure(nullptr, 0, rows, start, stride, old_whole, old_tiled);
        printf("rows=%u start=%u stride=%u  wmma whole %.3f ms %.1f TF/s | tiled %.3f ms %.1f TF/s\n",
               rows, start, stride, old_whole, gflop / old_whole, old_tiled, gflop / old_tiled);
        for (const auto& v : kVariants) {
            const unsigned dx = default_gx(v, rows, stride);
            for (unsigned gx : {dx, 2 * dx}) {
                float whole, tiled;
                measure(&v, gx, rows, start, stride, whole, tiled);
                printf("    %-20s gx=%-3u whole %.3f ms %.1f TF/s (%.2fx) | tiled %.3f ms %.1f TF/s (%.2fx)\n",
                       v.name, gx, whole, gflop / whole, old_whole / whole, tiled, gflop / tiled,
                       old_tiled / tiled);
            }
        }
    }
    // Small row counts at a 64K extent (decode-like rows and short appends).
    for (unsigned rows : {8u, 64u, 512u}) {
        const unsigned start = 65536, stride = (start + rows + 3) / 4;
        const double gflop = (double)rows * stride * 32 * 128 * 2 / 1e9;
        const float old_ms = best_ms([&] { launch(nullptr, 0, b, c16, 0, rows, start, stride); });
        const Variant& v = kVariants[0];
        const unsigned gx = default_gx(v, rows, stride);
        const float ms = best_ms([&] { launch(&v, gx, b, c16, 0, rows, start, stride); });
        printf("rows=%u start=%u  wmma %.3f ms %.1f TF/s | %s gx=%u %.3f ms %.1f TF/s (%.2fx)\n",
               rows, start, old_ms, gflop / old_ms, v.name, gx, ms, gflop / ms, old_ms / ms);
    }
    return total_diff == 0 ? 0 : 1;
}
