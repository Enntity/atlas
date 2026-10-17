// SPDX-License-Identifier: AGPL-3.0-only

// End-to-end byte test of the composed reference-policy KDA pipeline. Unlike
// the per-stage tests, every intermediate comes from our own kernels: only raw
// layer inputs, parameters and (for sequences) our own carried conv/recurrent
// state feed each step. Checks:
//   * captured fixtures: M5 prefill (000324) -> decode (000325) and M128
//     (000016) against the CAPTURED reference outputs (raw/), and
//   * the carried-state sequence seq0..seq7 (4 prefills incl. a 3-chunk+tail
//     invocation, then 4 decodes) against the oracle outputs, final states and
//     conv states.
// The launch order here is the production order documented in kda_fla.cu.
#include "harness.cuh"
#include "kda_fla.cu"
#include <cmath>

using namespace kda_harness;

namespace {

constexpr int DIM = 3 * HL * D;

struct Params {
    __nv_bfloat16* conv_w;
    float* a_log;
    float* dt_bias;
};

Params load_params(int r) {
    std::string p = oracle_root() + "/params/rank" + std::to_string(r) + "-";
    return {to_device<__nv_bfloat16>(read_file(p + "conv_w_bf16.bin")),
            to_device<float>(read_file(p + "a_log_f32.bin")), to_device<float>(read_file(p + "dt_bias_f32.bin"))};
}

struct Scratch {
    __nv_bfloat16 *conv, *q, *k, *Ai, *w, *u, *kg, *h, *vnew, *out;
    float *beta, *g, *A, *Aqk;
};

Scratch alloc_scratch(int T) {
    size_t n = (size_t)T * HL * D, nt = (T + BT - 1) / BT;
    Scratch s;
    s.conv = device_zeros<__nv_bfloat16>((size_t)T * DIM * 2);
    s.q = device_zeros<__nv_bfloat16>(n * 2);
    s.k = device_zeros<__nv_bfloat16>(n * 2);
    s.Ai = device_zeros<__nv_bfloat16>((size_t)T * HL * BT * 2);
    s.w = device_zeros<__nv_bfloat16>(n * 2);
    s.u = device_zeros<__nv_bfloat16>(n * 2);
    s.kg = device_zeros<__nv_bfloat16>(n * 2);
    s.h = device_zeros<__nv_bfloat16>(nt * HL * D * D * 2);
    s.vnew = device_zeros<__nv_bfloat16>(n * 2);
    s.out = device_zeros<__nv_bfloat16>(n * 2);
    s.beta = device_zeros<float>((size_t)T * HL * 4);
    s.g = device_zeros<float>(n * 4);
    s.A = device_zeros<float>((size_t)T * HL * BT * 4);
    s.Aqk = device_zeros<float>((size_t)T * HL * BT * 4);
    return s;
}

// Prefill of T rows; conv_state/state carried in place. Returns output bytes.
std::vector<uint8_t> prefill(const Params& p, float* conv_state, float* state, const std::vector<uint8_t>& x,
                             const std::vector<uint8_t>& g1, const std::vector<uint8_t>& braw, int T) {
    Scratch s = alloc_scratch(T);
    auto dx = to_device<__nv_bfloat16>(x);
    auto dg1 = to_device<__nv_bfloat16>(g1);
    auto db = to_device<__nv_bfloat16>(braw);
    const unsigned nt = (T + BT - 1) / BT;
    const float scale = 0.08838834764831845f;
    kda_fla_conv_prefill<<<(DIM + 255) / 256, 256>>>(conv_state, dx, p.conv_w, s.conv, DIM, T, 1);
    kda_fla_l2norm<<<(2 * T * HL + 7) / 8, 128>>>(s.conv, s.q, s.k, T, HL, 1e-6f);
    kda_fla_beta<<<(T * HL + 255) / 256, 256>>>(db, s.beta, T * HL);
    kda_fla_gate_cumsum<<<(HL * D + 127) / 128, 128>>>(dg1, p.dt_bias, p.a_log, s.g, T, HL, -5.0f,
                                                       (float)1.4426950408889634);
    KDA_CHECK(cudaMemset(s.A, 0, (size_t)T * HL * BT * 4));
    KDA_CHECK(cudaMemset(s.Aqk, 0, (size_t)T * HL * BT * 4));
    kda_fla_kkt_inter<<<dim3(nt, 16, HL), 32>>>(s.q, s.k, s.g, s.beta, s.A, s.Aqk, scale, T, HL);
    kda_fla_kkt_intra<<<dim3(nt, 4, HL), 256>>>(s.q, s.k, s.g, s.beta, s.A, s.Aqk, scale, T, HL);
    kda_fla_solve_tril<<<dim3(nt, HL), 64>>>(s.A, s.Ai, T, HL);
    kda_fla_recompute_w_u<<<dim3(nt, HL), 128>>>(s.k, s.conv + 2 * HL * D, DIM, s.beta, s.Ai, s.g, s.w, s.u, s.kg,
                                                 T, HL);
    kda_fla_delta_h<<<dim3(D / kda_fla_state::DELTA_BV, HL), 128>>>(s.kg, s.w, s.u, s.g, state, s.h, s.vnew, T, HL);
    kda_fla_output<<<dim3(D / kda_fla_state::OUT_BV, nt, HL), 128>>>(s.q, s.vnew, s.g, s.Aqk, s.h, s.out, scale, T,
                                                                      HL);
    KDA_CHECK(cudaGetLastError());
    return to_host(s.out, (size_t)T * HL * D * 2);
}

std::vector<uint8_t> decode(const Params& p, float* conv_state, float* state, const std::vector<uint8_t>& x,
                            const std::vector<uint8_t>& g1, const std::vector<uint8_t>& braw) {
    auto dx = to_device<__nv_bfloat16>(x);
    auto dg1 = to_device<__nv_bfloat16>(g1);
    auto db = to_device<__nv_bfloat16>(braw);
    auto conv = device_zeros<__nv_bfloat16>(DIM * 2);
    auto out = device_zeros<__nv_bfloat16>(HL * D * 2);
    kda_fla_conv_decode<<<(DIM + 255) / 256, 256>>>(conv_state, dx, p.conv_w, conv, DIM);
    kda_fla_recurrent_decode<<<dim3(D / 32, HL), 128>>>(conv, dg1, db, p.a_log, p.dt_bias, state, out, HL);
    KDA_CHECK(cudaGetLastError());
    return to_host(out, HL * D * 2);
}

// Reference conv state [dim, 3] bf16 -> Atlas [dim, 4] f32 slots 1..3.
std::vector<uint8_t> conv_state_slots(const std::vector<uint8_t>& atlas) {
    std::vector<uint8_t> r((size_t)DIM * 3 * 2);
    const float* a = (const float*)atlas.data();
    uint16_t* o = (uint16_t*)r.data();
    for (int c = 0; c < DIM; ++c)
        for (int j = 0; j < 3; ++j) {
            uint32_t bits;
            memcpy(&bits, &a[c * 4 + 1 + j], 4);
            o[c * 3 + j] = (uint16_t)(bits >> 16);  // values are raw BF16 inputs: exact
        }
    return r;
}

std::string raw(int r, int inv, const char* stage) {
    char b[160];
    snprintf(b, sizeof b, "raw/ffn-L0-rank%d-%06d-kda-core-%s.bin", r, inv, stage);
    return oracle_root() + "/" + b;
}

size_t run_rank(int r) {
    Params p = load_params(r);
    std::string R = "rank" + std::to_string(r);
    size_t bad = 0;
    // Captured: 000324 (M5) then 000325 (decode), fresh state.
    {
        auto cs = device_zeros<float>((size_t)DIM * 4 * 4);
        auto st = device_zeros<float>((size_t)HL * D * D * 4);
        auto o = prefill(p, cs, st, read_file(raw(r, 324, "input-qkv_proj_states")), read_file(raw(r, 324, "input-g1")),
                         read_file(raw(r, 324, "input-beta")), 5);
        bad += compare((R + " 000324 out").c_str(), o, read_file(raw(r, 324, "output")), 2);
        auto od = decode(p, cs, st, read_file(raw(r, 325, "input-qkv_proj_states")), read_file(raw(r, 325, "input-g1")),
                         read_file(raw(r, 325, "input-beta")));
        bad += compare((R + " 000325 out").c_str(), od, read_file(raw(r, 325, "output")), 2);
    }
    // Captured: 000016 (M128), fresh state.
    {
        auto cs = device_zeros<float>((size_t)DIM * 4 * 4);
        auto st = device_zeros<float>((size_t)HL * D * D * 4);
        auto o = prefill(p, cs, st, read_file(raw(r, 16, "input-qkv_proj_states")), read_file(raw(r, 16, "input-g1")),
                         read_file(raw(r, 16, "input-beta")), 128);
        bad += compare((R + " 000016 out").c_str(), o, read_file(raw(r, 16, "output")), 2);
    }
    // Carried-state sequence.
    auto cs = device_zeros<float>((size_t)DIM * 4 * 4);
    auto st = device_zeros<float>((size_t)HL * D * D * 4);
    const char* steps[] = {"seq0-prefill5", "seq1-prefill64", "seq2-prefill59", "seq3-prefill200",
                           "seq4-decode",   "seq5-decode",    "seq6-decode",    "seq7-decode"};
    for (const char* s : steps) {
        std::string c = R + "-" + s;
        int T = rows_of(c);
        bool dec = std::string(s).find("decode") != std::string::npos;
        auto o = dec ? decode(p, cs, st, stage(c, "in_x"), stage(c, "in_g1"), stage(c, "in_beta_raw"))
                     : prefill(p, cs, st, stage(c, "in_x"), stage(c, "in_g1"), stage(c, "in_beta_raw"), T);
        bad += compare((c + " out").c_str(), o, stage(c, "output"), 2);
        bad += compare((c + " state").c_str(), transpose_state(to_host(st, (size_t)HL * D * D * 4), HL),
                       stage(c, "final_state"), 4);
        bad += compare((c + " conv_state").c_str(), conv_state_slots(to_host(cs, (size_t)DIM * 4 * 4)),
                       stage(c, "conv_state_after"), 2);
    }
    return bad;
}

}  // namespace

int main() {
    size_t bad = run_rank(0) + run_rank(1);
    printf("%s: %zu total mismatches\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
