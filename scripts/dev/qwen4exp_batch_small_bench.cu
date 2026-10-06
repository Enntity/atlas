// SPDX-License-Identifier: AGPL-3.0-only
// Per-row bit parity and cost of the qwen4_exp exact small-kernel batching
// (ATLAS_QWEN4EXP_BATCH_SMALL, crates/spark-model/src/layers/ops/qwen4exp_batch_small.rs):
// one launch over every row of a batched decode or verify step where the
// batched step ran the single-row kernel once per row.
//
//   MoE top-k     moe_topk_softmax_rows (R blocks)        vs R x moe_topk_softmax
//                 (512 BF16 logits, top-10, normalized; ties planted inside a
//                 warp, across warps and across a thread's two experts;
//                 moe_topk_softmax_batched is shown breaking them differently)
//   MoE blend     moe_weighted_sum_blend_rows (x, R)      vs R x moe_weighted_sum_blend
//                 (per-row shared rows, the EP zero row at stride 0, a null
//                 shared row; with and without the shared-expert gate)
//   GDN step      qwen4exp_gdn_decode_fused_rows (vh, R)  vs per sequence the
//                 four-kernel chain dense_gemv_ba_gates -> causal_conv1d_update_
//                 l2norm_f32 -> gated_delta_rule_decode_f32 -> gated_rms_norm_
//                 f32_input_sigmoid, and vs qwen4exp_gdn_decode_fused; each
//                 sequence on its own recurrence and conv state, `steps`
//                 recurrent steps, every byte left behind (normed rows, gates,
//                 betas, conv windows, recurrence states). TP2 (8 key / 24
//                 value heads) and TP1 (16 / 48).
//
// R = 1..8, 16, 32 (16 and 32 as 8-row launches for the GDN step, as the
// runtime chunks them). Then GPU time per layer at R rows, the old launches
// against the new one, inside a CUDA graph (as serving replays decode) over
// 36 layer copies of weights and per-sequence states (DRAM-streamed).
//
// Build/run: scripts/dev/qwen4exp_batch_exact_bench.sh small-check|small-time
// (repo root, GB10, inside atlas-release-builder:1.93.1 for the runtime's
// CUDA 13.0). PTX built with the production flags, JIT-loaded as the server
// loads it (qwen4exp_ptx_harness.h).
#include "qwen4exp_ptx_harness.h"

#include <random>

typedef unsigned int u32;
typedef unsigned short bf;

static std::string g_dir = ".";
static int g_fail = 0;
static std::mt19937 g_rng(getenv("SEED") ? atoi(getenv("SEED")) : 2718);
static cudaStream_t g_s = nullptr;

static PtxModule& mod(const char* stem) {
    static std::vector<std::pair<std::string, PtxModule*>> mods;
    for (auto& m : mods) if (m.first == stem) return *m.second;
    PtxModule* m = new PtxModule;
    m->load(g_dir + "/" + stem + ".ptx");
    mods.push_back({stem, m});
    return *m;
}

// Random values with zeros, -0.0 and a few large entries mixed in.
static float rnd(float scale) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const unsigned k = g_rng() % 64;
    if (k == 0) return 0.0f;
    if (k == 1) return -0.0f;
    if (k == 2) return 30.0f * scale * u(g_rng);
    return scale * u(g_rng);
}
static std::vector<bf> rbf(size_t n, float scale) {
    std::vector<bf> v(n);
    for (auto& x : v) x = f2bf(rnd(scale));
    return v;
}
static std::vector<float> rf(size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = rnd(scale);
    return v;
}

template <typename T>
static bool same(const std::string& what, const Buf<T>& a, const Buf<T>& b, size_t count) {
    std::vector<T> ha = a.get(), hb = b.get();
    ha.resize(count);
    hb.resize(count);
    const size_t bad = diff_bytes(ha, hb);
    if (bad) {
        printf("  MISMATCH %-58s %zu bytes differ\n", what.c_str(), bad);
        g_fail++;
    }
    return bad == 0;
}

static const u32 ROWS[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32};

// ══ MoE router top-k ═════════════════════════════════════════════════════
static const u32 NE = 512, TOPK = 10, H = 2560;

// Logits with planted exact ties among the leaders: equal values at experts
// i and i+1 (same warp, neighbouring lanes), i and i+32*k (other warps), and
// i and i+256 (one thread's two experts).
static std::vector<bf> tied_logits(u32 rows) {
    std::vector<bf> v = rbf((size_t)rows * NE, 2.0f);
    std::uniform_int_distribution<u32> e(0, NE - 1);
    for (u32 r = 0; r < rows; r++) {
        bf* row = v.data() + (size_t)r * NE;
        for (int t = 0; t < 6; t++) {
            const u32 a = e(g_rng);
            static const u32 OFF[] = {1, 32, 96, 256, 255, 33};
            const u32 b = (a + OFF[t]) % NE;
            const bf top = f2bf(8.0f + 0.25f * (g_rng() % 4));
            row[a] = top;
            row[b] = top;
        }
    }
    return v;
}

static void topk_check() {
    CUfunction one = mod("moe_topk").fn("moe_topk_softmax");
    CUfunction rows_k = mod("moe_topk").fn("moe_topk_softmax_rows");
    CUfunction batched = mod("moe_topk").fn("moe_topk_softmax_batched");
    int batched_diff = 0, bad = 0;
    for (u32 rows : ROWS) {
        for (int rep = 0; rep < 8; rep++) {
            for (u32 norm : {1u, 0u}) {
                Buf<bf> logits;
                logits.alloc((size_t)rows * NE);
                logits.put(tied_logits(rows));
                Buf<u32> ri, gi, bi;
                Buf<float> rw, gw, bw;
                for (auto* b : {&ri, &gi, &bi}) { b->alloc(rows * TOPK); b->fill(0x55); }
                for (auto* b : {&rw, &gw, &bw}) { b->alloc(rows * TOPK); b->fill(0x55); }
                u32 ne = NE, k = TOPK, stride = NE;
                for (u32 r = 0; r < rows; r++) {
                    Args a;
                    a.add(logits.p + (size_t)r * NE).add(ri.p + r * TOPK).add(rw.p + r * TOPK)
                        .add(ne).add(k).add(norm);
                    launch(one, dim3(1), dim3(256), 0, a);
                }
                Args a;
                a.add(logits.p).add(gi.p).add(gw.p).add(ne).add(k).add(norm).add(stride);
                launch(rows_k, dim3(rows), dim3(256), 0, a);
                Args b;
                b.add(logits.p).add(bi.p).add(bw.p).add(ne).add(k).add(norm);
                launch(batched, dim3(rows), dim3(256), 0, b);
                CK(cudaDeviceSynchronize());
                char what[96];
                snprintf(what, sizeof what, "moe_topk_softmax_rows R=%u norm=%u indices", rows, norm);
                bad += !same(what, ri, gi, rows * TOPK);
                snprintf(what, sizeof what, "moe_topk_softmax_rows R=%u norm=%u weights", rows, norm);
                bad += !same(what, rw, gw, rows * TOPK);
                batched_diff += diff_bytes(ri.get(), bi.get()) != 0;
                for (auto* b : {&ri, &gi, &bi}) b->free_();
                for (auto* b : {&rw, &gw, &bw}) b->free_();
                logits.free_();
            }
        }
    }
    printf("  %s moe_topk_softmax_rows: R = 1..8, 16, 32 x 8 tie-planted draws x norm 0/1, "
           "indices + weights byte-equal to R x moe_topk_softmax\n", bad ? "BAD" : "ok ");
    printf("       (for contrast: moe_topk_softmax_batched picked other experts in %d of %zu "
           "of those draws)\n", batched_diff, sizeof(ROWS) / sizeof(ROWS[0]) * 16);
}

// ══ MoE weighted sum + shared blend ══════════════════════════════════════
static void blend_check() {
    CUfunction one = mod("moe_expert_gemv").fn("moe_weighted_sum_blend");
    CUfunction rows_k = mod("moe_expert_gemv").fn("moe_weighted_sum_blend_rows");
    int bad = 0;
    // shared: 0 per-row rows, 1 one zero row at stride 0 (EP), 2 null.
    for (u32 rows : ROWS) {
        for (int shared = 0; shared < 3; shared++) {
            for (bool gated : {true, false}) {
                Buf<bf> eo, sh, x, gw, ref, got;
                Buf<float> w;
                eo.alloc((size_t)rows * TOPK * H); eo.put(rbf(eo.n, 1.0f));
                sh.alloc((size_t)rows * H);
                sh.put(shared == 0 ? rbf(sh.n, 1.0f) : std::vector<bf>(sh.n, 0));
                x.alloc((size_t)rows * H); x.put(rbf(x.n, 1.0f));
                gw.alloc(H); gw.put(rbf(H, 0.05f));
                w.alloc(rows * TOPK); w.put(rf(w.n, 0.3f));
                ref.alloc((size_t)rows * H); ref.fill(0x55);
                got.alloc((size_t)rows * H); got.fill(0x55);
                bf* shp = shared == 2 ? nullptr : sh.p;
                bf* g = gated ? gw.p : nullptr;
                u32 h = H, k = TOPK, kk = H, stride = shared == 0 ? H : 0;
                for (u32 r = 0; r < rows; r++) {
                    Args a;
                    a.add(ref.p + (size_t)r * H).add(eo.p + (size_t)r * TOPK * H).add(w.p + r * TOPK)
                        .add(shp ? shp + (size_t)r * stride : shp).add(x.p + (size_t)r * H).add(g)
                        .add(h).add(k).add(kk);
                    launch(one, dim3((H + 255) / 256), dim3(256), 0, a);
                }
                Args a;
                a.add(got.p).add(eo.p).add(w.p).add(shp).add(x.p).add(g).add(h).add(k).add(kk).add(stride);
                launch(rows_k, dim3((H + 255) / 256, rows), dim3(256), 0, a);
                CK(cudaDeviceSynchronize());
                char what[96];
                snprintf(what, sizeof what, "moe_weighted_sum_blend_rows R=%u shared=%d gated=%d", rows,
                         shared, gated);
                bad += !same(what, ref, got, (size_t)rows * H);
                for (auto* b : {&eo, &sh, &x, &gw, &ref, &got}) b->free_();
                w.free_();
            }
        }
    }
    printf("  %s moe_weighted_sum_blend_rows: R = 1..8, 16, 32 x shared rows / EP zero row / null x "
           "gate on/off, every output byte equal to R x moe_weighted_sum_blend\n",
           bad ? "BAD" : "ok ");
}

// ══ GDN decode step ══════════════════════════════════════════════════════
static const u32 D = 128, DCONV = 4, ROWS_MAX = 8;
struct GdnShape { u32 nk, nv; const char* name; };
static u32 conv_dim(const GdnShape& s) { return 2 * s.nk * D + s.nv * D; }
static u32 qkvz_dim(const GdnShape& s) { return conv_dim(s) + s.nv * D; }

struct GdnWeights { Buf<bf> conv_w, ba_w, norm_w; Buf<float> a_log, dt_bias; };
struct GdnSeq { Buf<float> h, conv; };

static void gdn_weights(const GdnShape& s, GdnWeights& w) {
    w.conv_w.alloc((size_t)conv_dim(s) * DCONV); w.conv_w.put(rbf(w.conv_w.n, 0.5f));
    w.ba_w.alloc((size_t)2 * s.nv * H); w.ba_w.put(rbf(w.ba_w.n, 0.05f));
    w.norm_w.alloc(D); w.norm_w.put(rbf(D, 1.0f));
    w.a_log.alloc(s.nv); w.a_log.put(rf(s.nv, 1.0f));
    w.dt_bias.alloc(s.nv); w.dt_bias.put(rf(s.nv, 1.0f));
}
static void gdn_seq(const GdnShape& s, GdnSeq& q, bool init) {
    q.h.alloc((size_t)s.nv * D * D);
    q.conv.alloc((size_t)conv_dim(s) * DCONV);
    if (init) { q.h.put(rf(q.h.n, 0.05f)); q.conv.put(rf(q.conv.n, 1.0f)); }
}

// Step activations: rows of the mixer input, the qkvz projection, and the
// outputs (gates [rows, 2nv], normed [rows, nv*128]); conv_out/gdn_out are
// the chain's FP32 scratch.
struct GdnIo { Buf<bf> ba_in, qkvz, out; Buf<float> gates, conv_out; };
static void gdn_io(const GdnShape& s, GdnIo& io, u32 rows) {
    io.ba_in.alloc((size_t)rows * H);
    io.qkvz.alloc((size_t)rows * qkvz_dim(s));
    io.out.alloc((size_t)rows * s.nv * D);
    io.gates.alloc((size_t)rows * 2 * s.nv);
    io.conv_out.alloc(qkvz_dim(s));
}

struct GdnKernels { CUfunction ba, conv, rec, norm, fused, rows; };
static GdnKernels gdn_kernels() {
    return {mod("ssm_preprocess").fn("dense_gemv_ba_gates"),
            mod("causal_conv1d").fn("causal_conv1d_update_l2norm_f32"),
            mod("gated_delta_rule").fn("gated_delta_rule_decode_f32"),
            mod("rms_norm").fn("gated_rms_norm_f32_input_sigmoid"),
            mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_decode_fused"),
            mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_decode_fused_rows")};
}

// Sequence r's four-kernel chain (ssm_batched_recurrent.rs's per-sequence loop).
static void gdn_chain(const GdnKernels& k, const GdnShape& s, const GdnWeights& w, GdnSeq& q,
                      GdnIo& io, u32 r) {
    const u32 cd = conv_dim(s), key_dim = s.nk * D, n_ba = 2 * s.nv, hid = H, vpg = s.nv / s.nk;
    const u32 one = 1, dconv = DCONV, qk = 2 * key_dim, d = D;
    const float l2 = 1e-6f, eps = 1e-6f;
    bf* qkv = io.qkvz.p + (size_t)r * qkvz_dim(s);
    float* gates = io.gates.p + (size_t)r * 2 * s.nv;
    float* beta = gates + s.nv;
    float* conv_out = io.conv_out.p;
    float* gdn_out = conv_out + cd;
    { Args a; a.add(io.ba_in.p + (size_t)r * H).add(w.ba_w.p).add(w.a_log.p).add(w.dt_bias.p)
          .add(gates).add(beta).add(n_ba).add(hid).add(vpg);
      launch(k.ba, dim3((n_ba + 3) / 4), dim3(256), 0, a, g_s); }
    { Args a; a.add(q.conv.p).add(qkv).add(w.conv_w.p).add((const float*)nullptr).add(conv_out)
          .add(one).add(cd).add(dconv).add(qk).add(d).add(l2);
      launch(k.conv, dim3((cd + 255) / 256), dim3(256), 0, a, g_s); }
    { Args a; a.add(q.h.p).add(conv_out).add(conv_out + key_dim).add(conv_out + 2 * key_dim)
          .add(gates).add(beta).add(gdn_out).add(one).add(s.nk).add(s.nv).add(d).add(d);
      launch(k.rec, dim3(s.nv), dim3(128), 0, a, g_s); }
    { Args a; a.add(gdn_out).add(qkv + cd).add(w.norm_w.p).add(io.out.p + (size_t)r * s.nv * D)
          .add(d).add(eps).add(d).add(d);
      launch(k.norm, dim3(s.nv), dim3(128), 0, a, g_s); }
}

// Sequence r through the single-sequence fused kernel.
static void gdn_fused(const GdnKernels& k, const GdnShape& s, const GdnWeights& w, GdnSeq& q,
                      GdnIo& io, u32 r) {
    const float l2 = 1e-6f, eps = 1e-6f;
    const u32 hid = H, d = D;
    bf* qkv = io.qkvz.p + (size_t)r * qkvz_dim(s);
    float* gates = io.gates.p + (size_t)r * 2 * s.nv;
    Args a;
    a.add(q.h.p).add(q.conv.p).add(qkv).add(w.conv_w.p).add(io.ba_in.p + (size_t)r * H).add(w.ba_w.p)
        .add(w.a_log.p).add(w.dt_bias.p).add(gates).add(gates + s.nv).add(qkv + conv_dim(s))
        .add(w.norm_w.p).add(io.out.p + (size_t)r * s.nv * D).add(s.nk).add(s.nv).add(hid).add(d)
        .add(l2).add(eps);
    launch(k.fused, dim3(s.nv), dim3(128), 0, a, g_s);
}

struct RowStates { float* h[ROWS_MAX]; float* conv[ROWS_MAX]; };

// Rows [first, first + count) in one launch of the rows kernel.
static void gdn_rows(const GdnKernels& k, const GdnShape& s, const GdnWeights& w,
                     std::vector<GdnSeq>& seqs, GdnIo& io, u32 first, u32 count) {
    RowStates st = {};
    for (u32 i = 0; i < count; i++) { st.h[i] = seqs[first + i].h.p; st.conv[i] = seqs[first + i].conv.p; }
    const float l2 = 1e-6f, eps = 1e-6f;
    const u32 hid = H, d = D, stride = qkvz_dim(s);
    float* gates = io.gates.p + (size_t)first * 2 * s.nv;
    Args a;
    a.add(st).add(io.qkvz.p + (size_t)first * stride).add(w.conv_w.p).add(io.ba_in.p + (size_t)first * H)
        .add(w.ba_w.p).add(w.a_log.p).add(w.dt_bias.p).add(gates).add(w.norm_w.p)
        .add(io.out.p + (size_t)first * s.nv * D).add(s.nk).add(s.nv).add(hid).add(d).add(stride)
        .add(l2).add(eps);
    launch(k.rows, dim3(s.nv, count), dim3(128), 0, a, g_s);
}
static void gdn_rows_all(const GdnKernels& k, const GdnShape& s, const GdnWeights& w,
                         std::vector<GdnSeq>& seqs, GdnIo& io, u32 rows) {
    for (u32 first = 0; first < rows; first += ROWS_MAX)
        gdn_rows(k, s, w, seqs, io, first, std::min(ROWS_MAX, rows - first));
}

static bool gdn_check(const GdnShape& s, int steps) {
    const GdnKernels k = gdn_kernels();
    GdnWeights w;
    gdn_weights(s, w);
    int bad = 0;
    for (u32 rows : ROWS) {
        // Three arms on identical state: chain, per-sequence fused, rows.
        std::vector<GdnSeq> sa(rows), sb(rows), sc(rows);
        for (u32 r = 0; r < rows; r++) {
            gdn_seq(s, sa[r], true);
            gdn_seq(s, sb[r], false);
            gdn_seq(s, sc[r], false);
            for (GdnSeq* o : {&sb[r], &sc[r]}) {
                CK(cudaMemcpy(o->h.p, sa[r].h.p, sa[r].h.n * 4, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(o->conv.p, sa[r].conv.p, sa[r].conv.n * 4, cudaMemcpyDeviceToDevice));
            }
        }
        GdnIo ia, ib, ic;
        gdn_io(s, ia, rows); gdn_io(s, ib, rows); gdn_io(s, ic, rows);
        for (int st = 0; st < steps; st++) {
            auto q = rbf(ia.qkvz.n, st == 0 ? 3.0f : 1.0f);
            auto x = rbf(ia.ba_in.n, 1.0f);
            for (GdnIo* io : {&ia, &ib, &ic}) {
                io->qkvz.put(q); io->ba_in.put(x); io->out.fill(0x7F); io->gates.fill(0x7F);
            }
            for (u32 r = 0; r < rows; r++) gdn_chain(k, s, w, sa[r], ia, r);
            for (u32 r = 0; r < rows; r++) gdn_fused(k, s, w, sb[r], ib, r);
            gdn_rows_all(k, s, w, sc, ic, rows);
            CK(cudaDeviceSynchronize());
            char what[128];
            for (int arm = 0; arm < 2; arm++) {
                GdnIo& io = arm ? ib : ia;
                std::vector<GdnSeq>& ss = arm ? sb : sa;
                const char* vs = arm ? "fused" : "chain";
                snprintf(what, sizeof what, "%s R=%u step %d vs %s: normed rows", s.name, rows, st, vs);
                bad += !same(what, io.out, ic.out, io.out.n);
                snprintf(what, sizeof what, "%s R=%u step %d vs %s: gates+betas", s.name, rows, st, vs);
                bad += !same(what, io.gates, ic.gates, io.gates.n);
                for (u32 r = 0; r < rows; r++) {
                    snprintf(what, sizeof what, "%s R=%u step %d vs %s: seq %u conv", s.name, rows, st, vs, r);
                    bad += !same(what, ss[r].conv, sc[r].conv, sc[r].conv.n);
                    snprintf(what, sizeof what, "%s R=%u step %d vs %s: seq %u recurrence", s.name, rows, st, vs, r);
                    bad += !same(what, ss[r].h, sc[r].h, sc[r].h.n);
                }
            }
        }
        for (auto* v : {&sa, &sb, &sc}) for (auto& q : *v) { q.h.free_(); q.conv.free_(); }
        for (GdnIo* io : {&ia, &ib, &ic}) {
            io->ba_in.free_(); io->qkvz.free_(); io->out.free_(); io->gates.free_(); io->conv_out.free_();
        }
    }
    printf("  %s qwen4exp_gdn_decode_fused_rows %s: R = 1..8, 16, 32 x %d steps, every normed / "
           "gate / beta / conv / recurrence byte equal to the per-sequence four-kernel chain and "
           "to qwen4exp_gdn_decode_fused\n", bad ? "BAD" : "ok ", s.name, steps);
    return bad == 0;
}

// ══ timing ═══════════════════════════════════════════════════════════════
// Median us per layer of `body(layer)` over `layers` distinct layer sets,
// captured into one CUDA graph (serving replays decode as graphs).
template <typename F>
static double graph_us(F&& body, int layers, int reps = 7) {
    cudaGraph_t g;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    for (int l = 0; l < layers; l++) body(l);
    CK(cudaStreamEndCapture(g_s, &g));
    CK(cudaGraphInstantiate(&exec, g, 0));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> v;
    for (int r = 0; r < reps + 1; r++) {
        CK(cudaEventRecord(a, g_s));
        CK(cudaGraphLaunch(exec, g_s));
        CK(cudaEventRecord(b, g_s));
        CK(cudaStreamSynchronize(g_s));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (r) v.push_back(1000.0 * ms / layers);
    }
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(g));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static const int LAYERS = 36;

static void moe_small_time() {
    CUfunction t1 = mod("moe_topk").fn("moe_topk_softmax"), tr = mod("moe_topk").fn("moe_topk_softmax_rows");
    CUfunction b1 = mod("moe_expert_gemv").fn("moe_weighted_sum_blend");
    CUfunction br = mod("moe_expert_gemv").fn("moe_weighted_sum_blend_rows");
    const u32 R = 32;
    Buf<bf> logits, eo, x, gw, out;
    Buf<u32> idx;
    Buf<float> w;
    logits.alloc((size_t)R * NE); logits.put(rbf(logits.n, 2.0f));
    eo.alloc((size_t)R * TOPK * H); eo.put(rbf(eo.n, 1.0f));
    x.alloc((size_t)R * H); x.put(rbf(x.n, 1.0f));
    gw.alloc(H); gw.put(rbf(H, 0.05f));
    out.alloc((size_t)R * H);
    idx.alloc(R * TOPK);
    w.alloc(R * TOPK); w.put(rf(w.n, 0.3f));
    u32 ne = NE, k = TOPK, norm = 1, h = H, zero = 0;
    printf("\n  MoE glue, us per layer in a graph: R single-row launches -> one rows launch\n");
    printf("  %5s  %-24s  %-24s\n", "rows", "top-k", "weighted sum + blend (EP)");
    for (u32 rows : ROWS) {
        const double ta = graph_us([&](int) {
            for (u32 r = 0; r < rows; r++) {
                Args a; a.add(logits.p + (size_t)r * NE).add(idx.p + r * TOPK).add(w.p + r * TOPK)
                    .add(ne).add(k).add(norm);
                launch(t1, dim3(1), dim3(256), 0, a, g_s);
            }
        }, LAYERS);
        const double tb = graph_us([&](int) {
            Args a; a.add(logits.p).add(idx.p).add(w.p).add(ne).add(k).add(norm).add(ne);
            launch(tr, dim3(rows), dim3(256), 0, a, g_s);
        }, LAYERS);
        const double ba = graph_us([&](int) {
            for (u32 r = 0; r < rows; r++) {
                Args a; a.add(out.p + (size_t)r * H).add(eo.p + (size_t)r * TOPK * H).add(w.p + r * TOPK)
                    .add((bf*)nullptr).add(x.p + (size_t)r * H).add(gw.p).add(h).add(k).add(h);
                launch(b1, dim3((H + 255) / 256), dim3(256), 0, a, g_s);
            }
        }, LAYERS);
        const double bb = graph_us([&](int) {
            Args a; a.add(out.p).add(eo.p).add(w.p).add((bf*)nullptr).add(x.p).add(gw.p).add(h).add(k)
                .add(h).add(zero);
            launch(br, dim3((H + 255) / 256, rows), dim3(256), 0, a, g_s);
        }, LAYERS);
        printf("  %5u  %7.1f -> %6.1f (%4.1fx)  %7.1f -> %6.1f (%4.1fx)\n", rows, ta, tb, ta / tb, ba, bb,
               ba / bb);
    }
}

static void gdn_time(const GdnShape& s) {
    const GdnKernels k = gdn_kernels();
    const u32 RMAX = 8;
    std::vector<GdnWeights> ws(LAYERS);
    std::vector<std::vector<GdnSeq>> seqs(LAYERS, std::vector<GdnSeq>(RMAX));
    for (int l = 0; l < LAYERS; l++) {
        gdn_weights(s, ws[l]);
        for (auto& q : seqs[l]) gdn_seq(s, q, true);
    }
    GdnIo io;
    gdn_io(s, io, RMAX);
    io.qkvz.put(rbf(io.qkvz.n, 1.0f));
    io.ba_in.put(rbf(io.ba_in.n, 1.0f));
    printf("\n  GDN step %s, us per layer in a graph (%d layers, per-sequence states):\n", s.name, LAYERS);
    printf("  %5s  %10s  %10s  %10s  %s\n", "rows", "4-chain", "fused/seq", "rows", "chain->rows");
    for (u32 rows = 1; rows <= RMAX; rows++) {
        const double a = graph_us([&](int l) { for (u32 r = 0; r < rows; r++) gdn_chain(k, s, ws[l], seqs[l][r], io, r); }, LAYERS);
        const double b = graph_us([&](int l) { for (u32 r = 0; r < rows; r++) gdn_fused(k, s, ws[l], seqs[l][r], io, r); }, LAYERS);
        const double c = graph_us([&](int l) { gdn_rows_all(k, s, ws[l], seqs[l], io, rows); }, LAYERS);
        printf("  %5u  %10.1f  %10.1f  %10.1f  %.2fx  (%.0f GB/s of recurrence state)\n", rows, a, b, c,
               a / c, rows * 2.0 * s.nv * D * D * 4 / c / 1e3);
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    const int steps = argc > 3 ? atoi(argv[3]) : 4;
    init_driver();
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    const GdnShape tp2 = {8, 24, "TP2 8k/24v"}, tp1 = {16, 48, "TP1 16k/48v"};
    if (mode == "check") {
        printf("per-row parity (batch-small):\n");
        topk_check();
        blend_check();
        gdn_check(tp2, steps);
        gdn_check(tp1, steps);
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    moe_small_time();
    gdn_time(tp2);
    gdn_time(tp1);
    return 0;
}
