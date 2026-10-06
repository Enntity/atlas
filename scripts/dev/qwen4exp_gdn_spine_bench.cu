// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of the GDN prefill state spine at the qwen4_exp
// TP2 rank shape (24 v-heads, 8 k-heads, k = v = 128, 64-token chunks):
// `gated_delta_rule_chunk_delta_h_pipe` (the shipped spine, one CTA a head)
// against `gated_delta_rule_chunk_delta_h_pipe_dv64` (two column blocks a
// head). S_out, uc_out and the final state must match byte for byte.
//
// Build (repo root, GB10):
//   D=$(mktemp -d)
//   nvcc --ptx -arch=sm_121f -O3 --fmad=false -o $D/gdn_fla.ptx kernels/gb10/common/gated_delta_rule_fla.cu
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_gdn_spine_bench.cu -lcuda
//   $D/bench $D [tokens=16000] [v_heads=24] [k_heads=8]
#include "qwen4exp_ptx_harness.h"
#include <random>

static const unsigned KD = 128, VD = 128, C = 64;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [tokens] [nv] [nk]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    const unsigned T = argc > 2 ? atoi(argv[2]) : 16000;
    const unsigned NV = argc > 3 ? atoi(argv[3]) : 24;
    const unsigned NK = argc > 4 ? atoi(argv[4]) : 8;
    const unsigned NC = (T + C - 1) / C;
    const unsigned QK = 2 * NK * KD + NV * VD;   // packed q | k | v row
    init_driver();
    PtxModule m;
    m.load(dir + "/gdn_fla.ptx");
    const unsigned smem_pipe = 2 * (C * (2 * KD + VD) * 2) + 2 * C * 4 + 2 * (C + 1) * 4;
    const unsigned smem_dv = 2 * C * KD * 2 * 2 + 2 * C * 64 * 2 + 2 * C * 4 + 2 * (C + 1) * 4;
    CUfunction k_pipe = m.fn("gated_delta_rule_chunk_delta_h_pipe", smem_pipe);
    CUfunction k_dv = m.fn("gated_delta_rule_chunk_delta_h_pipe_dv64", smem_dv);

    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t blocks = (size_t)NC * NV;
    std::vector<unsigned short> w(blocks * C * KD), u(blocks * C * VD), qkv((size_t)T * QK);
    for (auto& x : w) x = f2bf(0.05f * nd(rng));
    for (auto& x : u) x = f2bf(0.5f * nd(rng));
    for (auto& x : qkv) x = f2bf(0.1f * nd(rng));
    std::vector<float> gc(blocks * C), h0((size_t)NV * KD * VD);
    for (size_t b = 0; b < blocks; ++b) {
        float acc = 0.0f;
        for (unsigned i = 0; i < C; ++i) {
            acc -= 0.002f + 0.01f * fabsf(nd(rng));
            gc[b * C + i] = acc;
        }
    }
    for (auto& x : h0) x = 0.01f * nd(rng);

    Buf<unsigned short> d_w, d_u, d_qkv, d_s1, d_s2, d_uc1, d_uc2;
    Buf<float> d_gc, d_h1, d_h2, d_gate;
    d_w.alloc(w.size()); d_w.put(w);
    d_u.alloc(u.size()); d_u.put(u);
    d_qkv.alloc(qkv.size()); d_qkv.put(qkv);
    d_gc.alloc(gc.size()); d_gc.put(gc);
    d_gate.alloc((size_t)T * 2 * NV);
    d_h1.alloc(h0.size()); d_h2.alloc(h0.size());
    d_s1.alloc(blocks * KD * VD); d_s2.alloc(blocks * KD * VD);
    d_uc1.alloc(blocks * C * VD); d_uc2.alloc(blocks * C * VD);
    const unsigned short* key = d_qkv.p + NK * KD;   // the k region of each packed row

    auto run = [&](CUfunction k, dim3 grid, unsigned threads, unsigned smem, float* h,
                   unsigned short* s_out, unsigned short* uc_out) {
        Args a;
        a.add(h).add(d_w.p).add(d_u.p).add(key).add(d_gate.p).add(d_gc.p).add(s_out).add(uc_out)
         .add(1u).add(T).add(NC).add(NK).add(NV).add(KD).add(VD).add(QK).add(2 * NV).add(0u)
         .add((int*)nullptr).add((int*)nullptr).add(0u);
        launch(k, grid, dim3(threads), smem, a);
    };
    auto pipe = [&] { run(k_pipe, dim3(NV, 1), 256, smem_pipe, d_h1.p, d_s1.p, d_uc1.p); };
    auto dv = [&] { run(k_dv, dim3(NV, 2), 128, smem_dv, d_h2.p, d_s2.p, d_uc2.p); };

    d_h1.put(h0); d_h2.put(h0);
    // Same fill on both sides: rows past a partial chunk's end are written by neither.
    d_s1.fill(0x11); d_s2.fill(0x11); d_uc1.fill(0x33); d_uc2.fill(0x33);
    pipe();
    dv();
    CK(cudaDeviceSynchronize());
    size_t ds = diff_bytes(d_s1.get(), d_s2.get());
    size_t du = diff_bytes(d_uc1.get(), d_uc2.get());
    size_t dh = diff_bytes(d_h1.get(), d_h2.get());
    printf("bitwise pipe vs pipe_dv64 (T=%u, %u v-heads, %u k-heads, %u chunks): S_out %zu, uc_out %zu, state %zu differing bytes\n",
           T, NV, NK, NC, ds, du, dh);
    const bool ok = ds == 0 && du == 0 && dh == 0;
    float t1 = time_ms([&] { d_h1.put(h0); pipe(); }, 3, 3);
    float t2 = time_ms([&] { d_h2.put(h0); dv(); }, 3, 3);
    printf("spine: pipe %.3f ms -> pipe_dv64 %.3f ms (%.2fx)  [each includes a %zu KB state upload]\n",
           t1, t2, t1 / t2, h0.size() * 4 / 1024);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
