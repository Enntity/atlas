// SPDX-License-Identifier: AGPL-3.0-only
// The device paths and the driver of scripts/dev/qwen4exp_moe_fidelity.cu.
#pragma once

struct Tables { u64 *gp, *gs, *upk, *us, *dp, *ds; float *g2, *u2, *d2; };

// The used experts of a layer on the device: the served [N, K/2] layout and
// the prefill [K/2, N] copy (byte transposes), as 512-entry tables.
struct LayerDev {
    Tables o, t;
    std::vector<void*> bufs;
    static unsigned char* up(std::vector<void*>& b, const std::vector<unsigned char>& v) {
        unsigned char* p = dput(v); b.push_back(p); return p;
    }
    static std::vector<unsigned char> tr(const std::vector<unsigned char>& v, unsigned rows, unsigned cols) {
        std::vector<unsigned char> o(v.size());
        for (unsigned r = 0; r < rows; r++)
            for (unsigned c = 0; c < cols; c++) o[(size_t)c * rows + r] = v[(size_t)r * cols + c];
        return o;
    }
    LayerDev(const std::map<unsigned, Expert>& ex) {
        std::vector<u64> P[2][6];
        std::vector<float> S[3];
        for (auto& v : P) for (auto& w : v) w.assign(NE, 0);
        for (auto& v : S) v.assign(NE, 0.f);
        for (auto& [e, x] : ex) {
            const HostProj* pr[3] = {&x.g, &x.u, &x.d};
            for (int j = 0; j < 3; j++) {
                const HostProj& h = *pr[j];
                P[0][2 * j][e] = (u64)up(bufs, h.p);
                P[0][2 * j + 1][e] = (u64)up(bufs, h.s);
                P[1][2 * j][e] = (u64)up(bufs, tr(h.p, h.n, h.k / 2));
                P[1][2 * j + 1][e] = (u64)up(bufs, tr(h.s, h.n, h.k / 16));
                S[j][e] = h.s2;
            }
        }
        Tables* T[2] = {&o, &t};
        for (int l = 0; l < 2; l++) {
            T[l]->gp = dput(P[l][0]); T[l]->gs = dput(P[l][1]); T[l]->upk = dput(P[l][2]);
            T[l]->us = dput(P[l][3]); T[l]->dp = dput(P[l][4]); T[l]->ds = dput(P[l][5]);
            T[l]->g2 = dput(S[0]); T[l]->u2 = dput(S[1]); T[l]->d2 = dput(S[2]);
        }
    }
    ~LayerDev() { for (void* p : bufs) cudaFree(p); }
};

// Each path's [rows * 10, 2560] BF16 expert outputs, (row, slot) order.
static std::vector<unsigned short> run_path(char which, const LayerDev& L, const Rec& r) {
    const unsigned R = r.rows, S = R * TOPK;
    void* A = dput(r.x); void* ids = dput(r.ids);
    void *order = dzero(8192), *ws = dzero(1 << 20), *act = dzero((size_t)(S + R) * I * 4);
    void *go = dzero((size_t)S * I * 2), *uo = dzero((size_t)S * I * 2), *down = dzero((size_t)S * H * 2);
    void *shg = dzero((size_t)R * I * 2), *shu = dzero((size_t)R * I * 2), *shd = dzero((size_t)R * H * 2);
    void* null = nullptr;
    float z = 0.f;
    unsigned topk = TOPK, rows = R, n_i = I, k_h = H, n_h = H, k_i = I;
    const Tables& o = L.o;
    std::vector<unsigned short> y((size_t)S * H);
    if (which == 'b') {
        const char* M = "qwen4exp_moe_rows";
        unsigned slots = S;
        launch(load(M, "qwen4exp_moe_rows_plan"), dim3(1), dim3(256), {&ids, &order, &slots});
        launch(load(M, "qwen4exp_moe_rows_gate_up"), dim3(I / 8, S + R, 2), dim3(128),
               {&A, (void*)&o.gp, (void*)&o.gs, (void*)&o.g2, &go, (void*)&o.upk, (void*)&o.us, (void*)&o.u2, &uo,
                &ids, &order, &null, &null, &z, &shg, &null, &null, &z, &shu, &n_i, &k_h, &topk, &rows});
        launch(load(M, "qwen4exp_moe_rows_silu_down"), dim3(H / 64, (R + 7) / 8 + S), dim3(256),
               {&go, &uo, (void*)&o.dp, (void*)&o.ds, (void*)&o.d2, &down, &ids, &order, &shg, &shu, &null, &null,
                &z, &shd, &n_h, &k_i, &topk, &rows}, 64 * (I / 2 + I / 16) + 2 * I * 4);
        auto v = dget(down, y.size() * 2);
        memcpy(y.data(), v.data(), v.size());
    } else if (which == 'c') {
        const char* M = "qwen4exp_moe_c8_tc";
        const unsigned units = S + (R + 15) / 16;
        launch(load(M, "qwen4exp_moe_c8_tc_plan"), dim3(1), dim3(1024), {&ids, &ws, &topk, &rows});
        launch(load(M, "qwen4exp_moe_c8_tc_gate_up"), dim3(I / 8, units), dim3(256),
               {&A, (void*)&o.gp, (void*)&o.gs, (void*)&o.g2, (void*)&o.upk, (void*)&o.us, (void*)&o.u2, &null,
                &null, &z, &null, &null, &z, &ws, &null, &null, &null, &null, &act, &topk, &rows});
        launch(load(M, "qwen4exp_moe_c8_tc_down"), dim3(H / 64, units), dim3(256),
               {&act, (void*)&o.dp, (void*)&o.ds, (void*)&o.d2, &null, &null, &z, &ws, &down, &shd, &topk, &rows});
        auto v = dget(down, y.size() * 2);
        memcpy(y.data(), v.data(), v.size());
    } else {  // 'd': the prefill chain over all 512 experts, rows sorted by expert
        const Tables& t = L.t;
        std::vector<std::pair<unsigned, unsigned>> srt;  // (expert, slot q)
        for (unsigned q = 0; q < S; q++) srt.push_back({r.ids[q], q});
        std::sort(srt.begin(), srt.end());
        std::vector<int> off(NE + 1, 0), tok;
        for (auto& p : srt) { off[p.first + 1]++; tok.push_back((int)(p.second / TOPK)); }
        for (unsigned e = 0; e < NE; e++) off[e + 1] += off[e];
        void *doff = dput(off), *dtok = dput(tok), *a8 = dzero((size_t)R * H), *act8 = dzero((size_t)S * I);
        unsigned n = R * H, ne = NE;
        const char* M = "moe_prefill_q38";
        launch(load(M, "moe_q38_a_to_e4m3"), dim3((n / 4 + 255) / 256), dim3(256), {&A, &a8, &n});
        launch(load(M, "moe_q38w_gate_up_silu"), dim3(I / 64, 1, NE), dim3(256),
               {&a8, (void*)&t.gp, (void*)&t.gs, (void*)&t.g2, (void*)&t.upk, (void*)&t.us, (void*)&t.u2, &act8,
                &doff, &dtok, &ne, &n_i, &k_h});
        launch(load(M, "moe_q38w_down"), dim3(H / 128, 1, NE), dim3(256),
               {&act8, (void*)&t.dp, (void*)&t.ds, (void*)&t.d2, &down, &doff, &ne, &n_h, &k_i});
        auto v = dget(down, y.size() * 2);
        for (unsigned i = 0; i < S; i++) memcpy(&y[(size_t)srt[i].second * H], &v[(size_t)i * H * 2], H * 2);
        for (void* p : {doff, dtok, a8, act8}) cudaFree(p);
    }
    for (void* p : {A, ids, order, ws, act, go, uo, down, shg, shu, shd}) cudaFree(p);
    return y;
}

// The float64 reference of every (row, slot): with the decode kernels' routed
// clamp (refc) and without (refn). Counts the clamp's bites.
static void reference(const std::map<unsigned, Expert>& ex, const Rec& r, std::vector<double>& refn,
                      std::vector<double>& refc, long& bites) {
    const unsigned S = r.rows * TOPK;
    refn.assign((size_t)S * H, 0); refc.assign((size_t)S * H, 0);
    long b = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : b)
    for (unsigned q = 0; q < S; q++) {
        const Expert& e = ex.at(r.ids[q]);
        std::vector<double> x(H), g(I), u(I), an(I), ac(I);
        for (unsigned k = 0; k < H; k++) x[k] = bf(r.x[(size_t)(q / TOPK) * H + k]);
        gemv64(e.g, x.data(), g.data());
        gemv64(e.u, x.data(), u.data());
        for (unsigned i = 0; i < I; i++) {
            an[i] = g[i] / (1 + exp(-g[i])) * u[i];
            const double gc = std::min(g[i], 10.0), uc = std::min(std::max(u[i], -10.0), 10.0);
            b += (gc != g[i]) + (uc != u[i]);
            ac[i] = gc / (1 + exp(-gc)) * uc;
        }
        gemv64(e.d, an.data(), &refn[(size_t)q * H]);
        gemv64(e.d, ac.data(), &refc[(size_t)q * H]);
    }
    bites += b;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: fidelity <ptx dir> <model dir>\n"); return 2; }
    g_dir = argv[1];
    CU(cuInit(0));
    CK(cudaFree(0));
    Ckpt ck(argv[2]);
    std::vector<Rec> recs;
    const char* dump = getenv("DUMP");
    std::vector<unsigned> layers;
    { std::istringstream ss(getenv("LAYERS") ? getenv("LAYERS") : "1 24 47"); unsigned l; while (ss >> l) layers.push_back(l); }
    const unsigned nrec = getenv("RECS") ? atoi(getenv("RECS")) : 4;
    if (dump) recs = read_dump(dump);
    printf("qwen4_exp routed-MoE fidelity vs float64 (%s inputs); rel L2 and max |err| over every (row, slot) output\n",
           dump ? "dumped decode" : "SYNTHETIC N(0, SIGMA)");
    for (unsigned L : layers) {
        std::vector<Rec> rs;
        for (auto& r : recs) if (r.layer == L && rs.size() < nrec) rs.push_back(r);
        if (!dump) {
            std::mt19937 rng(L);
            const float sigma = getenv("SIGMA") ? atof(getenv("SIGMA")) : 1.0f;
            Rec r{L, 32, c8_routes(32, 0.0, rng), std::vector<float>(32 * TOPK, 0.1f), rand_bf16((size_t)32 * H, sigma)};
            rs.push_back(r);
        }
        if (rs.empty()) { printf("layer %u: no records\n", L); continue; }
        std::set<unsigned> used;
        for (auto& r : rs) used.insert(r.ids.begin(), r.ids.end());
        std::map<unsigned, Expert> ex;
        const std::string base = "model.language_model.layers." + std::to_string(L) + ".mlp.experts.";
        for (unsigned e : used)
            ex[e] = {load_proj(ck, base + std::to_string(e) + ".gate_proj", I, H),
                     load_proj(ck, base + std::to_string(e) + ".up_proj", I, H),
                     load_proj(ck, base + std::to_string(e) + ".down_proj", H, I)};
        LayerDev dev(ex);
        // [path][ref n/c][slot outputs | routed sums]; floor = BF16(ref).
        Err e[5][2][2];
        long bites = 0, rows = 0;
        for (auto& r : rs) {
            std::vector<double> rn, rc;
            reference(ex, r, rn, rc, bites);
            auto yb = run_path('b', dev, r), yc = run_path('c', dev, r), yd = run_path('d', dev, r);
            rows += r.rows;
            for (unsigned row = 0; row < r.rows; row++)
                for (unsigned n = 0; n < H; n++) {
                    double z[5] = {0, 0, 0, 0, 0}, zr[2] = {0, 0};
                    for (unsigned s = 0; s < TOPK; s++) {
                        const size_t i = (size_t)(row * TOPK + s) * H + n;
                        const double w = r.w[row * TOPK + s];
                        const double v[5] = {bf_round(rn[i]), bf(yb[i]), bf(yc[i]), bf(yd[i]), bf_round(rc[i])};
                        for (int p = 0; p < 4; p++) {
                            e[p][0][0].add(v[p], p == 0 ? rn[i] : rn[i]);
                            e[p][1][0].add(p == 0 ? v[4] : v[p], rc[i]);
                            z[p] += w * (p == 0 ? v[0] : v[p]);
                        }
                        z[4] += w * v[4];
                        e[4][0][0].add(bf(yb[i]), bf(yd[i]));  // (b) against (d)
                        e[4][1][0].add(bf(yb[i]), bf(yc[i]));  // (b) against (c)
                        zr[0] += w * rn[i]; zr[1] += w * rc[i];
                    }
                    for (int p = 0; p < 4; p++) {
                        e[p][0][1].add(z[p], zr[0]);
                        e[p][1][1].add(p == 0 ? z[4] : z[p], zr[1]);
                    }
                }
        }
        printf("\nlayer %u: %ld rows, %zu experts, routed clamp bites (g > 10 or |u| > 10): %ld\n", L, rows, used.size(), bites);
        printf("  %-22s | vs float64, no clamp (model)  | vs float64, decode clamp      |\n", "");
        printf("  %-22s | expert out    | routed sum    | expert out    | routed sum    |\n", "path");
        printf("  %-22s | relL2   maxabs| relL2   maxabs| relL2   maxabs| relL2   maxabs|\n", "");
        const char* names[4] = {"BF16 floor (ref->bf16)", "(b) decode rows (FP32)", "(c) TC (BF16 MMA)", "(d) prefill FP8 (q38)"};
        for (int p = 0; p < 4; p++) {
            printf("  %-22s |", names[p]);
            e[p][0][0].print(); e[p][0][1].print(); e[p][1][0].print(); e[p][1][1].print();
            printf("\n");
        }
        printf("  (b) vs (d) expert out:"); e[4][0][0].print();
        printf("   (b) vs (c) expert out:"); e[4][1][0].print();
        printf("\n");
    }
    return 0;
}
