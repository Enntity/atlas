// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of ATLAS_GLM_LAYER_FORK (crates/spark-model/src/layers/glm_layer_fork.rs)
// on emulated GLM verify layers of M = 3 rows, one GB10:
//
//   MoE layer:  [hc/norm chain] shared gate/up (pair) -> SiLU -> shared down -> router GEMV
//               -> top-k / sort / worklist / quantize -> routed GEMMs -> unpermute+blend
//   fork:       [hc/norm chain] router GEMV | fork | side: shared gate/up -> SiLU -> down
//               main: top-k .. quantize -> routed GEMMs | join | unpermute+blend
//   MLA layer:  latent cache write -> index wk -> index norm -> index gate -> tail write
//               -> pool finalize -> latent dequant -> q_b -> W_uk absorb
//   fork:       latent cache write | fork | side: index chain   main: dequant -> q_b | join | absorb
//
// Real kernels: the shared expert (w4a16_gemv_tc8_pair_touch, w4a16_gemv_tc8_touch on this rank's
// half: N = 1024 gate/up, the K-slice of down) and q_b (mxfp8_gemv_tc8, 8192 x 1536). Stand-ins:
// `spin` for the latency-bound small kernels (no DRAM traffic), `router_gemv` (288 x 4096 BF16,
// 72 CTAs like dense_gemv_bf16_batchm_ahead), `narrow_gemv` (128 x 4096 BF16 on 8 one-warp CTAs,
// the shape of the cuBLASLt `Kernel2` the index projections run), `stream_read` (the routed
// GEMMs: `routed_mb` MB of distinct bytes per layer), `blend` / `absorb` (the join's consumers,
// reading both chains). Launch modes follow production: the MoE chain and q_b with PDL, the
// index chain, dequant and absorb plain.
//
// 1. Bitwise: every layer's blend / absorb output, in-line vs forked, and forked with the side
//    stream held 300 us after each fork. Every buffer one chain reads from the other (the normed
//    rows, the shared output, the index result) starts as garbage, so a fork or join that did
//    not order the chains shows up as differing bytes. Must print "bitwise: 0 of N outputs differ".
// 2. Time per emulated layer (lower decile over layers x reps, GPU queued behind a spin as in a
//    verify step): in-line, in-line plus the fork's event record/wait pairs on one stream (their
//    cost alone: they cut the PDL edges at the fork and join points), forked. "step" scales the
//    saving to 42 MoE / 11 MLA layers.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_layer_fork_bench.cu -o layer_fork_bench
//   ./layer_fork_bench [layers=8] [reps=15] [routed_mb=48] [pre_us=20] [tiny_us=5]
// Device memory: about 0.6 GB at the defaults. Exit 0 iff the bitwise check passes.
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

// A latency-bound small kernel: `iters` dependent loads from a 4 KiB table.
extern "C" __global__ void spin(unsigned int* tab, unsigned int iters) {
    atlas_pdl_enter();
    unsigned int x = threadIdx.x;
    for (unsigned int i = 0; i < iters; i++) x = tab[(x + i) & 1023u];
    if (x == 0xFFFFFFFFu) tab[0] = 0u;  // never: the table holds indices below 1024
}

extern "C" __global__ void nop(unsigned int* tab) {
    atlas_pdl_enter();
    if (tab == nullptr) return;
}

extern "C" __global__ void fill(u8* p, unsigned long long n, unsigned int kind, unsigned int seed) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int h = (unsigned int)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        u8 b = (u8)h;
        if (kind == 1) b = 0x20 + (b & 0x1F);              // E4M3 scales, moderate positive
        if (kind == 2 && (b & 0x7F) == 0x7F) b ^= 1;       // E4M3 values without NaN
        if (kind == 3) b = 120 + (b & 7);                  // E8M0
        if (kind == 4) b = (i & 1) ? (b & 0x3B) : b;       // BF16 of magnitude below 2^-4 or so
        p[i] = b;
    }
}

// c[m, n] = a[m, :] . w[n, :], BF16 weights [n, k]; one warp per output row.
extern "C" __global__ void router_gemv(const bf* a, const bf* w, bf* c, unsigned int m, unsigned int n,
                                       unsigned int k) {
    atlas_pdl_enter();
    const unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= n) return;
    for (unsigned int r = 0; r < m; r++) {
        float acc = 0.f;
        for (unsigned int i = lane; i < k; i += 32)
            acc += __bfloat162float(a[(size_t)r * k + i]) * __bfloat162float(w[(size_t)row * k + i]);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        if (lane == 0) c[(size_t)r * n + row] = __float2bfloat16(acc);
    }
}

// The same product on `gridDim.x` one-warp CTAs, each walking its rows in turn.
extern "C" __global__ void narrow_gemv(const bf* a, const bf* w, bf* c, unsigned int m, unsigned int n,
                                       unsigned int k) {
    atlas_pdl_enter();
    const unsigned int lane = threadIdx.x;
    for (unsigned int row = blockIdx.x; row < n; row += gridDim.x)
        for (unsigned int r = 0; r < m; r++) {
            float acc = 0.f;
            for (unsigned int i = lane; i < k; i += 32)
                acc += __bfloat162float(a[(size_t)r * k + i]) * __bfloat162float(w[(size_t)row * k + i]);
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
            if (lane == 0) c[(size_t)r * n + row] = __float2bfloat16(acc);
        }
}

extern "C" __global__ void silu_mul(const bf* g, const bf* u, bf* o, unsigned int n) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __bfloat162float(g[i]);
    o[i] = __float2bfloat16(x / (1.f + __expf(-x)) * __bfloat162float(u[i]));
}

// Reads `n16` 16-byte words; each CTA leaves the XOR of its words in sums[blockIdx.x].
extern "C" __global__ void stream_read(const uint4* p, unsigned long long n16, unsigned int* sums) {
    atlas_pdl_enter();
    unsigned int x = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xFFFFFFFFu, x, o);
    __shared__ unsigned int warp[32];
    if (threadIdx.x % 32 == 0) warp[threadIdx.x / 32] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int i = 1; i < blockDim.x / 32; i++) x ^= warp[i];  // x is warp 0's
        sums[blockIdx.x] = x;
    }
}

// out[i] = a[i] + b[i % nb] + (the routed checksum's low bits): reads both chains.
extern "C" __global__ void blend(const bf* a, const bf* b, unsigned int nb, const unsigned int* sums,
                                 bf* out, unsigned int n) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = sums ? (float)(sums[i % 64] & 7u) : 0.f;
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i % nb]) + s);
}

template <typename... Args>
static void launch(bool pdl, void (*kern)(Args...), dim3 grid, unsigned int block, cudaStream_t s,
                   Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cfg.stream = s;
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at;
    cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

static u8* dev(unsigned long long bytes, unsigned int kind, unsigned int seed) {
    u8* p;
    CK(cudaMalloc(&p, bytes));
    fill<<<1024, 256>>>(p, bytes, kind, seed);
    return p;
}

static const unsigned int H = 4096, M = 3, SHARED = 1024, EXPERTS = 288, QLORA = 1536, QB = 8192,
                          IDX = 128;
static const float SCALE2 = 0.37f;
static const unsigned int TOUCH_CTAS = 32;
static const unsigned long long TOUCH_BYTES = 12ull << 20;

// One MoE layer's weights and buffers (NVFP4 shared expert of this rank).
struct Moe {
    u8 *gw, *gs, *uw, *us, *dw, *ds;  // gate, up [1024, 4096]; down [4096, 2048] (cols 1024..2047)
    bf *in, *router, *gate, *up, *act, *down, *logits, *out;  // `in`: the normed rows
    u8* routed;
    unsigned int* sums;
};
// One MLA layer's.
struct Mla {
    u8 *qw, *qs;  // q_b MXFP8 [8192, 1536]
    bf *in, *wk, *wg, *keys, *gates, *idx, *q, *out;  // `in`: the normed rows
};

int main(int argc, char** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 8;
    const int reps = argc > 2 ? atoi(argv[2]) : 15;
    const unsigned long long routed_mb = argc > 3 ? atoi(argv[3]) : 48;
    const unsigned int pre_us = argc > 4 ? atoi(argv[4]) : 20, tiny_us = argc > 5 ? atoi(argv[5]) : 5;
    cudaStream_t main_s, side;
    CK(cudaStreamCreateWithFlags(&main_s, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    cudaEvent_t fork_ev, join_ev;
    CK(cudaEventCreateWithFlags(&fork_ev, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&join_ev, cudaEventDisableTiming));

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<unsigned short> hA((size_t)M * H);
    for (auto& x : hA) { bf b = __float2bfloat16(nd(rng)); memcpy(&x, &b, 2); }
    bf *x, *qlat;
    CK(cudaMalloc(&x, hA.size() * 2));
    CK(cudaMemcpy(x, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&qlat, (size_t)M * QLORA * 2));
    CK(cudaMemcpy(qlat, hA.data(), (size_t)M * QLORA * 2, cudaMemcpyHostToDevice));
    std::vector<unsigned int> hTab(1024);
    for (auto& t : hTab) t = rng() & 1023u;
    unsigned int* tab;
    CK(cudaMalloc(&tab, 1024 * 4));
    CK(cudaMemcpy(tab, hTab.data(), 1024 * 4, cudaMemcpyHostToDevice));

    const unsigned long long routed_bytes = routed_mb << 20;
    auto bfbuf = [](size_t elems) { bf* p; CK(cudaMalloc(&p, elems * 2)); return p; };
    std::vector<Moe> moe(layers);
    std::vector<Mla> mla(layers);
    for (int l = 0; l < layers; l++) {
        const unsigned int s = 1000u * l;
        Moe& e = moe[l];
        e.gw = dev((size_t)SHARED * H / 2, 0, s + 1); e.gs = dev((size_t)SHARED * H / 16, 1, s + 2);
        e.uw = dev((size_t)SHARED * H / 2, 0, s + 3); e.us = dev((size_t)SHARED * H / 16, 1, s + 4);
        e.dw = dev((size_t)H * 2 * SHARED / 2, 0, s + 5); e.ds = dev((size_t)H * 2 * SHARED / 16, 1, s + 6);
        e.router = (bf*)dev((size_t)EXPERTS * H * 2, 4, s + 7);
        e.routed = dev(routed_bytes, 0, s + 8);
        e.in = bfbuf((size_t)M * H);
        e.gate = bfbuf((size_t)M * SHARED); e.up = bfbuf((size_t)M * SHARED); e.act = bfbuf((size_t)M * SHARED);
        e.down = bfbuf((size_t)M * H); e.logits = bfbuf((size_t)M * EXPERTS); e.out = bfbuf((size_t)M * H);
        CK(cudaMalloc(&e.sums, 192 * 4));
        Mla& a = mla[l];
        a.qw = dev((size_t)QB * QLORA, 2, s + 11); a.qs = dev((size_t)QB * QLORA / 32, 3, s + 12);
        a.wk = (bf*)dev((size_t)IDX * H * 2, 4, s + 13); a.wg = (bf*)dev((size_t)IDX * H * 2, 4, s + 14);
        a.in = bfbuf((size_t)M * H);
        a.keys = bfbuf((size_t)M * IDX); a.gates = bfbuf((size_t)M * IDX); a.idx = bfbuf((size_t)M * IDX);
        a.q = bfbuf((size_t)M * QB); a.out = bfbuf((size_t)M * QB);
    }
    CK(cudaDeviceSynchronize());

    // Spin iterations per microsecond, from a lone spin.
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float per_us = 0.f;
    {
        const unsigned int iters = 20000;
        float best = 1e9f;
        for (int r = 0; r < 5; r++) {
            CK(cudaEventRecord(t0, main_s));
            launch(false, spin, dim3(64), 128u, main_s, tab, iters);
            CK(cudaEventRecord(t1, main_s));
            CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            best = std::min(best, ms);
        }
        per_us = iters / (best * 1000.f);
    }
    auto sp = [&](bool pdl, unsigned int us, cudaStream_t s) {
        launch(pdl, spin, dim3(64), 128u, s, tab, (unsigned int)(per_us * us));
    };
    auto fence = [&](cudaEvent_t ev, cudaStream_t from, cudaStream_t to) {
        CK(cudaEventRecord(ev, from));
        CK(cudaStreamWaitEvent(to, ev, 0));
    };
    const unsigned int sh_grid = SHARED / 16, dn_grid = H / 16;
    auto shared_expert = [&](const Moe& e, cudaStream_t s) {
        launch(true, w4a16_gemv_tc8_pair_touch, dim3(sh_grid, 1, 2), 256u, s, (const bf*)e.in, (const u8*)e.gw,
               (const u8*)e.gs, SCALE2, e.gate, (const u8*)e.uw, (const u8*)e.us, SCALE2, e.up, M, SHARED, H,
               (unsigned int)std::min<unsigned long long>(SHARED, TOUCH_BYTES / (H / 2 + H / 16)),
               std::min(sh_grid, TOUCH_CTAS));
        launch(true, silu_mul, dim3((M * SHARED + 255) / 256), 256u, s, (const bf*)e.gate, (const bf*)e.up,
               e.act, M * SHARED);
        // This rank's K-slice [1024, 2048) of a 2048-wide down.
        launch(true, w4a16_gemv_tc8_touch, dim3(dn_grid), 256u, s, (const bf*)e.act,
               (const u8*)(e.dw + SHARED / 2), (const u8*)(e.ds + SHARED / 16), SCALE2, e.down, M, H, SHARED,
               SHARED, SHARED / 8, (unsigned int)std::min<unsigned long long>(H, TOUCH_BYTES / (SHARED / 2 + SHARED / 16)),
               std::min(dn_grid, TOUCH_CTAS));
    };
    // The bitwise pass delays the side stream right after each fork by this
    // much, so a join that did not hold would let the consumer read garbage.
    unsigned int side_delay_us = 0;
    // The normed rows of a layer, written on the compute stream (x + x).
    auto norm = [&](bf* in, bool pdl) {
        launch(pdl, blend, dim3((M * H + 255) / 256), 256u, main_s, (const bf*)x, (const bf*)x, M * H,
               (const unsigned int*)nullptr, in, M * H);
    };
    auto fork_to_side = [&] {
        fence(fork_ev, main_s, side);
        if (side_delay_us) sp(false, side_delay_us, side);
    };
    // mode 0: in line; 1: in line plus the fork's fences on one stream; 2: forked.
    auto moe_layer = [&](int l, int mode) {
        const Moe& e = moe[l];
        launch(false, nop, dim3(8), 256u, main_s, tab);  // the attention all-reduce restarts the chain
        sp(true, pre_us, main_s);                        // HC post, HC partial, finalize
        norm(e.in, true);                                // norm
        if (mode != 2) shared_expert(e, main_s);
        launch(true, router_gemv, dim3(EXPERTS / 4), 128u, main_s, (const bf*)e.in, (const bf*)e.router,
               e.logits, M, EXPERTS, H);
        if (mode == 1) fence(fork_ev, main_s, main_s);
        if (mode == 2) { fork_to_side(); shared_expert(e, side); }
        for (int i = 0; i < 4; i++) sp(true, tiny_us, main_s);  // top-k, sort, worklist, quantize
        launch(true, stream_read, dim3(192), 256u, main_s, (const uint4*)e.routed, routed_bytes / 16, e.sums);
        if (mode == 1) fence(join_ev, main_s, main_s);
        if (mode == 2) fence(join_ev, side, main_s);
        launch(true, blend, dim3((M * H + 255) / 256), 256u, main_s, (const bf*)e.down, (const bf*)e.in, M * H,
               (const unsigned int*)e.sums, e.out, M * H);
    };
    auto index_chain = [&](const Mla& a, cudaStream_t s) {
        launch(false, narrow_gemv, dim3(8), 32u, s, (const bf*)a.in, (const bf*)a.wk, a.keys, M, IDX, H);
        sp(false, 2, s);  // index layernorm
        launch(false, narrow_gemv, dim3(8), 32u, s, (const bf*)a.in, (const bf*)a.wg, a.gates, M, IDX, H);
        sp(false, 2, s);  // tail write
        launch(false, blend, dim3((M * IDX + 255) / 256), 256u, s, (const bf*)a.keys, (const bf*)a.gates, M * IDX,
               (const unsigned int*)nullptr, a.idx, M * IDX);  // pool finalize
    };
    auto mla_layer = [&](int l, int mode) {
        const Mla& a = mla[l];
        launch(false, nop, dim3(8), 256u, main_s, tab);
        norm(a.in, false);     // the layer's normed input (made before q_a / kv_a)
        sp(false, 5, main_s);  // mla_cache_assemble, latent cache write
        if (mode == 0) index_chain(a, main_s);
        if (mode == 1) { fence(fork_ev, main_s, main_s); index_chain(a, main_s); }
        if (mode == 2) { fork_to_side(); index_chain(a, side); }
        sp(false, 4, main_s);  // latent dequant
        launch(true, mxfp8_gemv_tc8, dim3(QB / 16), 256u, main_s, (const bf*)qlat, (const u8*)a.qw,
               (const u8*)a.qs, a.q, M, QB, QLORA, QB);
        if (mode == 1) fence(join_ev, main_s, main_s);
        if (mode == 2) fence(join_ev, side, main_s);
        launch(false, blend, dim3((M * QB + 255) / 256), 256u, main_s, (const bf*)a.q, (const bf*)a.idx, M * IDX,
               (const unsigned int*)nullptr, a.out, M * QB);  // W_uk absorb
    };

    // ── 1. Bitwise ──
    size_t diff = 0, checked = 0;
    {
        std::vector<std::vector<unsigned short>> ref(2 * layers);
        // In line, then forked, then forked with the side stream 300 us late.
        for (int arm = 0; arm < 3; arm++) {
            const int mode = arm ? 2 : 0;
            side_delay_us = arm == 2 ? 300 : 0;
            for (int l = 0; l < layers; l++) {  // garbage in every buffer a chain reads from the other
                for (bf* p : {moe[l].in, mla[l].in}) CK(cudaMemsetAsync(p, 0x5A, (size_t)M * H * 2, main_s));
                CK(cudaMemsetAsync(moe[l].down, 0x5A, (size_t)M * H * 2, main_s));
                CK(cudaMemsetAsync(moe[l].out, 0x11, (size_t)M * H * 2, main_s));
                CK(cudaMemsetAsync(mla[l].idx, 0x5A, (size_t)M * IDX * 2, main_s));
                CK(cudaMemsetAsync(mla[l].out, 0x11, (size_t)M * QB * 2, main_s));
            }
            for (int l = 0; l < layers; l++) { moe_layer(l, mode); mla_layer(l, mode); }
            CK(cudaStreamSynchronize(main_s));
            CK(cudaStreamSynchronize(side));
            for (int l = 0; l < layers; l++) {
                std::vector<unsigned short> o((size_t)M * H), q((size_t)M * QB);
                CK(cudaMemcpy(o.data(), moe[l].out, o.size() * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(q.data(), mla[l].out, q.size() * 2, cudaMemcpyDeviceToHost));
                if (arm == 0) { ref[2 * l] = o; ref[2 * l + 1] = q; continue; }
                for (size_t i = 0; i < o.size(); i++, checked++) diff += o[i] != ref[2 * l][i];
                for (size_t i = 0; i < q.size(); i++, checked++) diff += q[i] != ref[2 * l + 1][i];
            }
        }
    }
    side_delay_us = 0;
    printf("bitwise: %zu of %zu outputs differ\n", diff, checked);

    // ── 2. Time ──
    std::vector<cudaEvent_t> ev(layers + 1);
    for (auto& event : ev) CK(cudaEventCreate(&event));
    auto time_us = [&](const std::function<void(int)>& layer) {
        std::vector<float> us;
        for (int r = 0; r < reps; r++) {
            sp(false, 2000, main_s);  // queue the pass behind a 2 ms spin
            CK(cudaEventRecord(ev[0], main_s));
            for (int l = 0; l < layers; l++) { layer(l); CK(cudaEventRecord(ev[l + 1], main_s)); }
            CK(cudaEventSynchronize(ev[layers]));
            for (int l = 0; l < layers; l++) {
                float ms; CK(cudaEventElapsedTime(&ms, ev[l], ev[l + 1]));
                us.push_back(ms * 1000.f);
            }
        }
        CK(cudaStreamSynchronize(side));
        std::sort(us.begin(), us.end());
        return us[us.size() / 10];
    };
    printf("spin: %.1f iterations/us; routed stand-in %llu MB, pre-chain %u us, 4 small kernels of %u us\n",
           per_us, routed_mb, pre_us, tiny_us);
    const char* names[3] = {"in line", "in line + fences", "forked"};
    float moe_t[3], mla_t[3];
    for (int mode = 0; mode < 3; mode++) {
        moe_t[mode] = time_us([&](int l) { moe_layer(l, mode); });
        mla_t[mode] = time_us([&](int l) { mla_layer(l, mode); });
    }
    for (int mode = 0; mode < 3; mode++)
        printf("  %-18s MoE layer %7.1f us (step x42 %+7.3f ms)   MLA layer %7.1f us (step x11 %+7.3f ms)\n",
               names[mode], moe_t[mode], (moe_t[0] - moe_t[mode]) * 42e-3f, mla_t[mode],
               (mla_t[0] - mla_t[mode]) * 11e-3f);
    printf("(step columns: in-line time minus this arm's, positive = saved)\n");
    return diff == 0 ? 0 : 1;
}
