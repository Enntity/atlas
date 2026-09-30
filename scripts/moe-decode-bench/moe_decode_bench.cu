// SPDX-License-Identifier: AGPL-3.0-only
// Standalone GLM-5.3-Flash routed-MoE verify-decode microbenchmark (GB10,
// one expert-TP rank: all 288 experts local at the rank's I/2 = 1024 slice).
//
// One DFlash verify step's routed FFN for T rows (top-8 of 288 experts, U
// distinct experts), launched as serving launches it:
//   production  moe_build_tile_worklist -> compact k64 gate/up ->
//               silu_mul_quant_nvfp4 + D2D -> dense K128 down -> unpermute
//   k128w       the prefill K128W kernels over the same rows
//   m16         ATLAS_GLM_MOE_DECODE_M16: worklist -> M16 gate/up+SiLU -> M16 down
//   m16+zskip   ... with ATLAS_GLM_MOE_DOWN_ZSKIP's down
//   read-only   the same weight bytes read once and nothing else (roofline)
// over a ring of routings with disjoint expert sets, so every repetition reads
// cold weights (as the 42 MoE layers of a real step do). Every variant is
// gated byte for byte against production on every routing, and the down
// kernels once more on inputs rewritten to signed E2M1 zeros.
//
// Build + run on a GB10 host, from this directory (-arch=sm_121a alone does
// not enable the mxf4nvf4 MMA in ptxas; --fmad=false as KERNEL.toml):
//   nvcc -O3 -std=c++17 --fmad=false -gencode arch=compute_121a,code=sm_121a \
//     -o /tmp/moe_decode_bench moe_decode_bench.cu
//   /tmp/moe_decode_bench [T:U ...]        (default 1:8 2:14 4:22 5:27 7:36 8:41)
//   MOE_DECODE_REPS=n (default 42 per variant)
// Exit: 0 ok, 1 a variant's bytes differ, 2 setup error.

#include "../../kernels/gb10/glm-5.3-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../kernels/gb10/glm-5.3-flash/nvfp4/moe_permute.cu"
#include "../../kernels/gb10/glm-5.3-flash/nvfp4/moe_silu_mul.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static const unsigned E = 288, TOPK = 8, H = 4096, I = 1024;
static const unsigned NT_GU = I / 128, NT_DN = H / 128;   // N128 tiles of gate/up and down
// Packed E2M1 + E4M3 block scales of one [1024 x 4096] / [4096 x 1024] slice.
static const double PROJ_BYTES = (double)I * H / 2 + (double)I * H / 16;

struct Lcg {
    unsigned long long s;
    unsigned next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(s >> 33); }
    float unit() { return (next() & 0xFFFFFF) / 16777216.0f; }
};

__global__ void stream_read(const uint4* __restrict__ src, size_t n, unsigned* out) {
    unsigned x = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x12345678u) out[0] = x;
}

// Roofline: the active experts' packed + scale tables of up to two
// projections, read once with coalesced 16-byte loads and nothing else.
// Grid (slices, experts), 256 threads.
__global__ void roof_read(const unsigned long long* __restrict__ pp, const unsigned long long* __restrict__ sp,
                          const unsigned long long* __restrict__ pp2, const unsigned long long* __restrict__ sp2,
                          const int* __restrict__ active, size_t packed16, size_t scale16, unsigned* out) {
    const int e = active[blockIdx.y];
    const uint4* tab[4] = {(const uint4*)pp[e], (const uint4*)sp[e],
                           pp2 ? (const uint4*)pp2[e] : nullptr, sp2 ? (const uint4*)sp2[e] : nullptr};
    unsigned x = 0;
    for (int j = 0; j < 4; ++j) {
        if (!tab[j]) continue;
        const size_t n = (j & 1) ? scale16 : packed16, per = (n + gridDim.x - 1) / gridDim.x;
        const size_t lo = blockIdx.x * per, hi = lo + per < n ? lo + per : n;
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const uint4 v = tab[j][i];
            x ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (x == 0x12345678u) out[0] = x;
}

template <class T> static T* dev(size_t n) { T* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> static T* up(const std::vector<T>& h) { T* p = dev<T>(h.size()); CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> down(const T* p, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static void fill_bytes(std::vector<unsigned char>& v, Lcg& r, bool scale) {
    for (auto& b : v) b = scale ? (unsigned char)(0x28 + r.next() % 24) : (unsigned char)r.next();
}

struct ExpertTable {  // [K/2, N] packed + [K/16, N] scales + scale2 per expert
    unsigned long long *pp, *sp;
    float* s2;
};

static ExpertTable make_table(Lcg& r, unsigned n, unsigned k, float s2base) {
    std::vector<unsigned long long> pp(E), sp(E);
    std::vector<float> s2(E);
    std::vector<unsigned char> w((size_t)k / 2 * n), s((size_t)k / 16 * n);
    for (unsigned e = 0; e < E; ++e) {
        fill_bytes(w, r, false);
        fill_bytes(s, r, true);
        pp[e] = (unsigned long long)up(w);
        sp[e] = (unsigned long long)up(s);
        s2[e] = s2base * (1.0f + (e % 7) * 0.125f);
    }
    return {up(pp), up(sp), up(s2)};
}

// One routing of the ring: T rows x top-8 over `active` (every active expert
// used at least once), sorted as moe_sort_by_expert leaves it.
struct Route {
    int *off, *sorted, *t2p, *ids, *active;
    float* w;
    unsigned* work;   // [total, pad, pad, pad, items...] as the serving scratch
    unsigned distinct, max_rows;
};

static Route make_route(Lcg& r, unsigned T, const std::vector<unsigned>& active) {
    const unsigned TE = T * TOPK, U = active.size();
    std::vector<int> ids(TE);
    std::vector<float> wts(TE);
    unsigned fresh = 0;  // active experts not yet routed
    for (unsigned t = 0; t < T; ++t) {
        for (unsigned k = 0; k < TOPK; ++k) {
            int e;
            bool dup;
            do {
                // Unused experts first while the remaining slots need them.
                const bool need = U - fresh >= TE - (t * TOPK + k);
                e = (fresh < U && (need || r.next() % 3 == 0)) ? active[fresh] : active[r.next() % (fresh ? fresh : 1)];
                dup = false;
                for (unsigned j = 0; j < k; ++j) dup |= ids[t * TOPK + j] == e;
            } while (dup);
            if (fresh < U && e == (int)active[fresh]) ++fresh;
            ids[t * TOPK + k] = e;
            wts[t * TOPK + k] = 0.05f + r.unit() * 0.3f;
        }
    }
    std::vector<int> offs(E + 1, 0), sorted(TE), t2p(TE);
    for (int e : ids) offs[e + 1]++;
    for (unsigned e = 0; e < E; ++e) offs[e + 1] += offs[e];
    std::vector<int> fill(offs.begin(), offs.end() - 1);
    for (unsigned i = 0; i < TE; ++i) { const int p = fill[ids[i]]++; sorted[p] = i / TOPK; t2p[i] = p; }
    Route q{};
    for (unsigned e = 0; e < E; ++e) {
        const unsigned m = offs[e + 1] - offs[e];
        q.distinct += m > 0;
        q.max_rows = std::max(q.max_rows, m);
    }
    q.off = up(offs); q.sorted = up(sorted); q.t2p = up(t2p); q.ids = up(ids); q.w = up(wts);
    q.active = up(std::vector<int>(active.begin(), active.end()));
    q.work = dev<unsigned>(4 + (size_t)TE * NT_GU * 2);
    return q;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    float once(const std::function<void()>& f) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float x; CK(cudaEventElapsedTime(&x, a, b));
        return x;
    }
};

static float pct(std::vector<float> v, float p) { std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; }
static float median(const std::vector<float>& v) { return pct(v, 0.5f); }

int main(int argc, char** argv) {
    std::vector<std::pair<unsigned, unsigned>> loads;
    for (int i = 1; i < argc; ++i) {
        unsigned t, u;
        if (sscanf(argv[i], "%u:%u", &t, &u) != 2 || u < TOPK || u > E || t * TOPK < u) {
            fprintf(stderr, "workload %s: want T:U with 8 <= U <= min(8T, %u)\n", argv[i], E);
            return 2;
        }
        loads.push_back({t, u});
    }
    if (loads.empty()) loads = {{1, 8}, {2, 14}, {4, 22}, {5, 27}, {7, 36}, {8, 41}};
    const int reps = getenv("MOE_DECODE_REPS") ? std::max(atoi(getenv("MOE_DECODE_REPS")), 4) : 42;
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    Timer tm;

    {
        // Streaming-read rate: the best grid of a few, fast decile of 9 each
        // (a shared host perturbs it; the read-only variant below is the
        // roofline the kernels are held to).
        const size_t bytes = (size_t)1 << 29;
        uint4* a = dev<uint4>(bytes / 16);
        CK(cudaMemset(a, 1, bytes));
        unsigned* u = dev<unsigned>(1);
        double dram = 0;
        for (int mult : {2, 4, 8, 16}) {
            auto rd = [&] { stream_read<<<sms * mult, 512>>>(a, bytes / 16, u); };
            rd(); CK(cudaDeviceSynchronize());
            std::vector<float> t;
            for (int i = 0; i < 9; ++i) t.push_back(tm.once(rd));
            dram = std::max(dram, bytes / pct(t, 0.1f) / 1e6);
        }
        printf("probe DRAM stream read %.1f GB/s (%d SMs)\n", dram, sms);
        CK(cudaFree(a)); CK(cudaFree(u));
    }

    Lcg r{0x5eed};
    ExpertTable gate = make_table(r, I, H, 1.0f / 256), upt = make_table(r, I, H, 1.0f / 128),
                dn = make_table(r, H, I, 1.0f / 64);
    int bad_total = 0;
    for (const auto& load : loads) {
        const unsigned T = load.first, U = load.second;
        const unsigned TE = T * TOPK;
        // Ring of disjoint active sets (a 256 MB flush when only one fits).
        std::vector<unsigned> perm(E);
        for (unsigned e = 0; e < E; ++e) perm[e] = e;
        for (unsigned e = E - 1; e > 0; --e) std::swap(perm[e], perm[r.next() % (e + 1)]);
        const unsigned ring = std::max(E / U, 1u);
        std::vector<Route> routes;
        for (unsigned i = 0; i < ring; ++i) {
            std::vector<unsigned> active(perm.begin() + i * U, perm.begin() + (i + 1) * U);
            std::sort(active.begin(), active.end());
            // Route order is random within the set, as a router's would be.
            for (unsigned e = U - 1; e > 0; --e) std::swap(active[e], active[r.next() % (e + 1)]);
            routes.push_back(make_route(r, T, active));
            if (routes.back().distinct != U) { fprintf(stderr, "route covers %u of %u\n", routes.back().distinct, U); return 2; }
        }
        unsigned char* flush = ring < 2 ? dev<unsigned char>((size_t)256 << 20) : nullptr;
        unsigned max_rows = 0;
        for (auto& q : routes) max_rows = std::max(max_rows, q.max_rows);
        const unsigned max_m_tiles = (max_rows + 63) / 64;

        std::vector<unsigned char> a((size_t)T * H / 2), as((size_t)T * H / 16);
        fill_bytes(a, r, false); fill_bytes(as, r, true);
        unsigned char *d_a = up(a), *d_as = up(as);
        __nv_bfloat16 *c_gate = dev<__nv_bfloat16>((size_t)TE * I), *c_up = dev<__nv_bfloat16>((size_t)TE * I);
        // As serving: SiLU stages NVFP4 in the down-output arena, then copies
        // it over the dead up rows, which the down projection reads.
        const size_t qb = (size_t)TE * I / 2, sb = (size_t)TE * I / 16;
        __nv_bfloat16* c_down = dev<__nv_bfloat16>((size_t)TE * H);
        unsigned char* staged = (unsigned char*)c_down;
        unsigned char* q = (unsigned char*)c_up;
        __nv_bfloat16* out = dev<__nv_bfloat16>((size_t)T * H);
        const unsigned max_tiles = TE * NT_GU;

        auto worklist = [&](const Route& rt) {
            moe_build_tile_worklist<<<1, 256>>>(rt.off, gate.pp, rt.work + 4, (int*)rt.work, E, NT_GU, 64);
        };
        auto gate_up = [&](const Route& rt) {
            moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up<<<dim3(max_tiles, 2), 128>>>(
                d_a, d_as, gate.pp, gate.sp, gate.s2, c_gate, upt.pp, upt.sp, upt.s2, c_up,
                rt.off, rt.sorted, E, I, H, rt.work + 4, (int*)rt.work, max_tiles);
        };
        auto silu = [&](const Route&) {
            silu_mul_quant_nvfp4<<<TE, 128>>>(c_gate, c_up, staged, staged + qb, nullptr, TE, I);
            CK(cudaMemcpyAsync(q, staged, qb + sb, cudaMemcpyDeviceToDevice));
        };
        auto down_proj = [&](const Route& rt) {
            moe_w4a4_grouped_gemm_prequant_t_k128<<<dim3(NT_DN, max_m_tiles, E), 256>>>(
                q, q + qb, dn.pp, dn.sp, dn.s2, c_down, rt.off, nullptr, E, H, I);
        };
        auto unperm = [&](const Route& rt) {
            moe_unpermute_reduce_indexed_ep<<<T, 256>>>(c_down, out, rt.t2p, rt.ids, rt.w, H, T, TOPK, 0, E);
        };
        struct Stage { const char* name; std::function<void(const Route&)> f; double bytes; };
        struct Variant { const char* name; std::vector<Stage> stages; };
        const double gu_bytes = U * 2 * PROJ_BYTES, dn_bytes = U * PROJ_BYTES;
        std::vector<Variant> variants;
        variants.push_back({"production", {
            {"worklist", worklist, 0}, {"gate_up k64 compact", gate_up, gu_bytes},
            {"silu_quant + d2d", silu, 0}, {"down k128 dense", down_proj, dn_bytes}, {"unpermute", unperm, 0}}});

        // K128W (the prefill kernels) over the same rows.
        int* d_prefix = dev<int>(E + 1);
        const unsigned bound = (TE + 63) / 64 + E;
        auto prefix = [&](const Route& rt) { moe_mtile_prefix<<<1, 1024>>>(rt.off, gate.pp, d_prefix, E); };
        auto gate_up_w = [&](const Route& rt) {
            moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w<<<dim3(I / 128, bound), 256>>>(
                d_a, d_as, gate.pp, gate.sp, gate.s2, nullptr, rt.off, rt.sorted, E, I, H, d_prefix,
                upt.pp, upt.sp, upt.s2, q, q + qb);
        };
        auto down_w = [&](const Route& rt) {
            moe_w4a4_grouped_gemm_prequant_t_k128w_compact<<<dim3(H / 256, bound), 256>>>(
                q, q + qb, dn.pp, dn.sp, dn.s2, c_down, rt.off, nullptr, E, H, I, d_prefix);
        };
        variants.push_back({"k128w", {
            {"mtile_prefix", prefix, 0}, {"gate_up_silu k128w", gate_up_w, gu_bytes},
            {"down k128w", down_w, dn_bytes}, {"unpermute", unperm, 0}}});
        // M16 decode twins (ATLAS_GLM_MOE_DECODE_M16): at most 16 rows per expert.
        const unsigned bound16 = std::min(TE, E);
        unsigned* d_work16 = dev<unsigned>(4 + (size_t)TE * 2);
        auto prefix16 = [&](const Route& rt) {
            moe_build_tile_worklist<<<1, 256>>>(rt.off, gate.pp, d_work16 + 4, (int*)d_work16, E, 1, 64);
        };
        auto gate_up_m16 = [&](const Route& rt) {
            glm_moe_decode_m16_gate_up_silu_k128w<<<dim3(I / 128, bound16), 256>>>(
                d_a, d_as, gate.pp, gate.sp, gate.s2, nullptr, rt.off, rt.sorted, E, I, H, d_work16,
                upt.pp, upt.sp, upt.s2, q, q + qb);
        };
        auto down_m16 = [&](const Route& rt) {
            glm_moe_decode_m16_k128w<<<dim3(H / 256, bound16), 256>>>(
                q, q + qb, dn.pp, dn.sp, dn.s2, c_down, rt.off, nullptr, E, H, I, d_work16);
        };
        auto down_m16z = [&](const Route& rt) {
            glm_moe_decode_m16_k128w_zskip<<<dim3(H / 256, bound16), 256>>>(
                q, q + qb, dn.pp, dn.sp, dn.s2, c_down, rt.off, nullptr, E, H, I, d_work16);
        };
        if (max_rows <= 16) {
            variants.push_back({"m16", {
                {"worklist", prefix16, 0}, {"gate_up_silu m16", gate_up_m16, gu_bytes},
                {"down m16", down_m16, dn_bytes}, {"unpermute", unperm, 0}}});
            variants.push_back({"m16+zskip", {
                {"worklist", prefix16, 0}, {"gate_up_silu m16", gate_up_m16, gu_bytes},
                {"down m16 zskip", down_m16z, dn_bytes}, {"unpermute", unperm, 0}}});
        }
        // Read-only roofline over the same weight bytes (timing only).
        unsigned* d_roof = dev<unsigned>(1);
        auto roof_gu = [&](const Route& rt) {
            roof_read<<<dim3(16, U), 256>>>(gate.pp, gate.sp, upt.pp, upt.sp, rt.active,
                (size_t)I * H / 2 / 16, (size_t)I * H / 16 / 16, d_roof);
        };
        auto roof_dn = [&](const Route& rt) {
            roof_read<<<dim3(16, U), 256>>>(dn.pp, dn.sp, nullptr, nullptr, rt.active,
                (size_t)I * H / 2 / 16, (size_t)I * H / 16 / 16, d_roof);
        };
        const size_t roofline = variants.size();
        variants.push_back({"read-only", {{"gate+up tables", roof_gu, gu_bytes}, {"down tables", roof_dn, dn_bytes}}});

        // Production outputs per route, then every other variant against them.
        struct Ref { std::vector<__nv_bfloat16> c, o; std::vector<unsigned char> q; };
        auto run = [&](const Variant& v, const Route& rt) {
            CK(cudaMemset(c_gate, 0x5a, (size_t)TE * I * 2)); CK(cudaMemset(c_up, 0x5a, (size_t)TE * I * 2));
            CK(cudaMemset(c_down, 0x5a, (size_t)TE * H * 2)); CK(cudaMemset(out, 0x5a, (size_t)T * H * 2));
            Ref ref;
            for (auto& st : v.stages) {
                // The NVFP4 down input is final once the down projection is next.
                if (strncmp(st.name, "down", 4) == 0) { CK(cudaDeviceSynchronize()); ref.q = down(q, qb + sb); }
                st.f(rt);
            }
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            ref.c = down(c_down, (size_t)TE * H); ref.o = down(out, (size_t)T * H);
            return ref;
        };
        printf("\nT=%u rows, U=%u distinct experts, %u routed rows, ring %u%s, max rows/expert %u, %d reps\n",
               T, U, TE, ring, flush ? " + flush" : "", max_rows, reps);
        std::vector<Ref> refs;
        for (auto& rt : routes) refs.push_back(run(variants[0], rt));
        {
            // The down input's E2M1 zeros: single codes, and the weight rows
            // (two codes) that are zero in every row of their expert.
            size_t zero = 0, skip = 0, rows_kp = 0;
            for (unsigned ri = 0; ri < ring; ++ri) {
                const auto offs = down(routes[ri].off, E + 1);
                const auto& qq = refs[ri].q;
                for (size_t i = 0; i < qb; ++i) zero += ((qq[i] & 0x07) == 0) + ((qq[i] & 0x70) == 0);
                for (unsigned e = 0; e < E; ++e) {
                    if (offs[e + 1] == offs[e]) continue;
                    for (unsigned kp = 0; kp < I / 2; ++kp) {
                        unsigned any = 0;
                        for (int row = offs[e]; row < offs[e + 1]; ++row) any |= qq[(size_t)row * (I / 2) + kp] & 0x77;
                        skip += any == 0;
                    }
                    rows_kp += I / 2;
                }
            }
            printf("  down input: %.1f%% E2M1 zeros, %.1f%% of the routed experts' down weight rows all-zero (skippable)\n",
                   100.0 * zero / (2.0 * qb * ring), 100.0 * skip / rows_kp);
        }
        for (size_t vi = 1; vi < roofline; ++vi) {
            size_t bad_q = 0, bad_c = 0, bad_o = 0;
            float diff = 0.f;
            for (unsigned ri = 0; ri < ring; ++ri) {
                const Ref got = run(variants[vi], routes[ri]);
                const Ref& ref = refs[ri];
                for (size_t i = 0; i < qb + sb; ++i) bad_q += got.q[i] != ref.q[i];
                for (size_t i = 0; i < (size_t)TE * H; ++i) bad_c += memcmp(&got.c[i], &ref.c[i], 2) != 0;
                for (size_t i = 0; i < (size_t)T * H; ++i) {
                    bad_o += memcmp(&got.o[i], &ref.o[i], 2) != 0;
                    diff = std::max(diff, std::fabs(__bfloat162float(got.o[i]) - __bfloat162float(ref.o[i])));
                }
            }
            printf("  %-12s vs production over %u routes: NVFP4 down-input bytes differing %zu, down bf16 differing %zu, "
                   "layer out bf16 differing %zu (max abs diff %g)\n", variants[vi].name, ring, bad_q, bad_c, bad_o, diff);
            bad_total += (bad_q || bad_c || bad_o);
        }

        if (max_rows <= 16) {
            // Signed-zero stress of the down kernels: every third sorted row
            // all -0 codes, every third a mix of +0/-0 codes, the rest as
            // routed (so tiles skip none, some and all of their weight rows).
            size_t bad = 0, cells = 0;
            for (unsigned ri = 0; ri < ring; ++ri) {
                const Route& rt = routes[ri];
                std::vector<unsigned char> z = refs[ri].q;
                static const unsigned char mix[4] = {0x00, 0x80, 0x08, 0x88};
                for (unsigned row = 0; row < TE; ++row)
                    for (unsigned kp = 0; kp < I / 2 && row % 3 != 2; ++kp)
                        z[(size_t)row * (I / 2) + kp] = row % 3 == 0 ? 0x88 : mix[(kp + row) & 3];
                std::vector<__nv_bfloat16> want;
                const std::function<void(const Route&)> downs[3] = {down_proj, down_m16, down_m16z};
                for (auto& f : downs) {
                    CK(cudaMemcpy(q, z.data(), qb + sb, cudaMemcpyHostToDevice));
                    CK(cudaMemset(c_down, 0x5a, (size_t)TE * H * 2));
                    prefix16(rt); f(rt);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    const auto got = down(c_down, (size_t)TE * H);
                    if (want.empty()) { want = got; continue; }
                    for (size_t i = 0; i < got.size(); ++i) bad += memcmp(&got[i], &want[i], 2) != 0;
                    cells += got.size();
                }
            }
            printf("  signed-zero stress, M16 downs vs dense K128: bf16 differing %zu of %zu\n", bad, cells);
            bad_total += bad != 0;
        }

        // Timing. Variants are interleaved within a repetition and each takes
        // the ring's next routing, so every timed launch reads cold weights.
        auto cold = [&](int i) {
            if (flush) CK(cudaMemsetAsync(flush, i, (size_t)256 << 20));
            return &routes[i % ring];
        };
        std::vector<std::vector<float>> whole(variants.size());
        for (int i = 0; i < reps; ++i)
            for (size_t vi = 0; vi < variants.size(); ++vi) {
                const Route* rt = cold(i * variants.size() + vi);
                CK(cudaDeviceSynchronize());
                whole[vi].push_back(tm.once([&] { for (auto& st : variants[vi].stages) st.f(*rt); }));
            }
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            auto& v = variants[vi];
            for (auto& st : v.stages) {
                // Each stage timed alone; the stages before it run untimed.
                std::vector<float> t;
                for (int i = 0; i < reps; ++i) {
                    const Route* rt = cold(i);
                    for (auto& pre : v.stages) {
                        if (&pre == &st) break;
                        pre.f(*rt);
                    }
                    CK(cudaDeviceSynchronize());
                    t.push_back(tm.once([&] { st.f(*rt); }));
                }
                const float ms = median(t), lo = pct(t, 0.1f);
                if (st.bytes > 0)
                    printf("  %-12s %-22s %8.1f us (p10 %7.1f)  %6.1f MB  %6.1f GB/s\n", v.name, st.name, ms * 1e3,
                           lo * 1e3, st.bytes / 1e6, st.bytes / ms / 1e6);
                else
                    printf("  %-12s %-22s %8.1f us (p10 %7.1f)\n", v.name, st.name, ms * 1e3, lo * 1e3);
            }
            if (vi == roofline) continue;   // reads only: no layer to compare
            // Paired against production: the median of per-repetition ratios.
            std::vector<float> ratio;
            for (int i = 0; i < reps; ++i) ratio.push_back(whole[vi][i] / whole[0][i]);
            printf("  %-12s %-22s %8.1f us (p10 %7.1f)  paired vs production %+.1f%%\n", v.name, "LAYER", median(whole[vi]) * 1e3,
                   pct(whole[vi], 0.1f) * 1e3, 100.0 * (median(ratio) - 1.0));
        }
        CK(cudaFree(d_prefix)); CK(cudaFree(d_roof)); CK(cudaFree(d_work16));
        for (auto& rt : routes) { CK(cudaFree(rt.off)); CK(cudaFree(rt.sorted)); CK(cudaFree(rt.t2p)); CK(cudaFree(rt.ids)); CK(cudaFree(rt.w)); CK(cudaFree(rt.work)); CK(cudaFree(rt.active)); }
        CK(cudaFree(d_a)); CK(cudaFree(d_as)); CK(cudaFree(c_gate)); CK(cudaFree(c_up)); CK(cudaFree(c_down)); CK(cudaFree(out));
        if (flush) CK(cudaFree(flush));
    }
    return bad_total ? 1 : 0;
}
