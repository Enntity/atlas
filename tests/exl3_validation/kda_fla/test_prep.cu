// SPDX-License-Identifier: AGPL-3.0-only

// Byte-exact test of the reference-policy KDA prep kernels (conv, L2 norm,
// beta, gate cumsum, decode conv) against the GPU stage oracle, both ranks.
//   KDA_ORACLE=/home/colum/kda-oracle/inductor ./test_prep
#include "harness.cuh"
#include "kda_fla_prep.cuh"

using namespace kda_harness;

static constexpr unsigned int DIM = 3 * HL * D;

// Reference conv state bf16 [dim, 3] -> Atlas f32 [dim, 4] (slot 0 unused).
static std::vector<uint8_t> atlas_conv_state(const std::vector<uint8_t>& ref) {
    std::vector<uint8_t> out((size_t)DIM * 4 * 4, 0);
    const uint16_t* r = (const uint16_t*)ref.data();
    float* o = (float*)out.data();
    for (size_t c = 0; c < DIM; ++c)
        for (int j = 0; j < 3; ++j) {
            uint32_t bits = (uint32_t)r[c * 3 + j] << 16;
            memcpy(&o[c * 4 + 1 + j], &bits, 4);
        }
    return out;
}

// Atlas f32 [dim, 4] -> reference bf16 [dim, 3] (values are raw BF16 inputs).
static std::vector<uint8_t> ref_conv_state(const std::vector<uint8_t>& atlas) {
    std::vector<uint8_t> out((size_t)DIM * 3 * 2);
    const float* a = (const float*)atlas.data();
    uint16_t* o = (uint16_t*)out.data();
    for (size_t c = 0; c < DIM; ++c)
        for (int j = 0; j < 3; ++j) {
            uint32_t bits;
            memcpy(&bits, &a[c * 4 + 1 + j], 4);
            o[c * 3 + j] = (uint16_t)(bits >> 16);
        }
    return out;
}

static std::string raw_input(const std::string& c, const char* what) {
    // c = rankR-000NNN
    std::string rank = c.substr(4, 1), inv = c.substr(6);
    return "raw/ffn-L0-rank" + rank + "-" + inv + "-kda-core-input-" + what;
}

static bool is_seq(const std::string& c) { return c.find("seq") != std::string::npos; }

static size_t run_prefill(const std::string& c) {
    const int T = rows_of(c);
    const std::string rank = c.substr(0, 5);
    auto x = is_seq(c) ? stage(c, "in_x") : stage("", raw_input(c, "qkv_proj_states"));
    auto g1 = is_seq(c) ? stage(c, "in_g1") : stage("", raw_input(c, "g1"));
    auto braw = is_seq(c) ? stage(c, "in_beta_raw") : stage("", raw_input(c, "beta"));
    const bool has_init = is_seq(c) && c.find("seq0") == std::string::npos;
    std::vector<uint8_t> state0((size_t)DIM * 16, 0);
    if (has_init) state0 = atlas_conv_state(stage(c, "conv_state_before"));
    auto cw = stage("params", rank + "-conv_w_bf16");
    auto alog = stage("params", rank + "-a_log_f32");
    auto dtb = stage("params", rank + "-dt_bias_f32");

    auto* d_x = to_device<__nv_bfloat16>(x);
    auto* d_state = to_device<float>(state0);
    auto* d_w = to_device<__nv_bfloat16>(cw);
    auto* d_conv = device_zeros<__nv_bfloat16>((size_t)T * DIM * 2);
    auto* d_q = device_zeros<__nv_bfloat16>((size_t)T * HL * D * 2);
    auto* d_k = device_zeros<__nv_bfloat16>((size_t)T * HL * D * 2);
    auto* d_braw = to_device<__nv_bfloat16>(braw);
    auto* d_beta = device_zeros<float>((size_t)T * HL * 4);
    auto* d_g1 = to_device<__nv_bfloat16>(g1);
    auto* d_alog = to_device<float>(alog);
    auto* d_dtb = to_device<float>(dtb);
    auto* d_g = device_zeros<float>((size_t)T * HL * D * 4);

    kda_fla_conv_prefill<<<(DIM + 255) / 256, 256>>>(d_state, d_x, d_w, d_conv, DIM, T, has_init ? 1 : 0);
    kda_fla_l2norm<<<(2 * T * HL + 7) / 8, 128>>>(d_conv, d_q, d_k, T, HL, 1e-6f);
    kda_fla_beta<<<(T * HL + 255) / 256, 256>>>(d_braw, d_beta, T * HL);
    kda_fla_gate_cumsum<<<(HL * D + 127) / 128, 128>>>(d_g1, d_dtb, d_alog, d_g, T, HL, -5.0f,
                                                      (float)1.4426950408889634);
    KDA_CHECK(cudaGetLastError());

    printf("== %s (T=%d, has_init=%d)\n", c.c_str(), T, has_init);
    size_t bad = 0;
    bad += compare("conv_out", to_host(d_conv, (size_t)T * DIM * 2), stage(c, "conv_out"), 2, DIM);
    bad += compare("conv_state_after", ref_conv_state(to_host(d_state, (size_t)DIM * 16)),
                   stage(c, "conv_state_after"), 2, 3);
    bad += compare("q_l2", to_host(d_q, (size_t)T * HL * D * 2), stage(c, "q_l2"), 2, HL * D);
    bad += compare("k_l2", to_host(d_k, (size_t)T * HL * D * 2), stage(c, "k_l2"), 2, HL * D);
    bad += compare("beta", to_host(d_beta, (size_t)T * HL * 4), stage(c, "beta"), 4, HL);
    bad += compare("g_cumsum", to_host(d_g, (size_t)T * HL * D * 4), stage(c, "g_cumsum"), 4, HL * D);
    for (void* p : {(void*)d_x, (void*)d_state, (void*)d_w, (void*)d_conv, (void*)d_q, (void*)d_k, (void*)d_braw,
                    (void*)d_beta, (void*)d_g1, (void*)d_alog, (void*)d_dtb, (void*)d_g})
        cudaFree(p);
    return bad;
}

static size_t run_decode(const std::string& c) {
    const std::string rank = c.substr(0, 5);
    const bool captured = !is_seq(c);
    auto x = captured ? stage("", raw_input(c, "qkv_proj_states")) : stage(c, "in_x");
    auto before = captured ? stage(rank + "-000324", "conv_state_after") : stage(c, "conv_state_before");
    auto* d_x = to_device<__nv_bfloat16>(x);
    auto* d_state = to_device<float>(atlas_conv_state(before));
    auto* d_w = to_device<__nv_bfloat16>(stage("params", rank + "-conv_w_bf16"));
    auto* d_conv = device_zeros<__nv_bfloat16>((size_t)DIM * 2);
    kda_fla_conv_decode<<<(DIM + 255) / 256, 256>>>(d_state, d_x, d_w, d_conv, DIM);
    KDA_CHECK(cudaGetLastError());
    printf("== %s (decode)\n", c.c_str());
    size_t bad = compare("conv_out", to_host(d_conv, (size_t)DIM * 2), stage(c, "conv_out"), 2, DIM);
    if (!captured)
        bad += compare("conv_state_after", ref_conv_state(to_host(d_state, (size_t)DIM * 16)),
                       stage(c, "conv_state_after"), 2, 3);
    for (void* p : {(void*)d_x, (void*)d_state, (void*)d_w, (void*)d_conv}) cudaFree(p);
    return bad;
}

int main() {
    size_t bad = 0;
    for (auto& c : prefill_cases()) bad += run_prefill(c);
    for (auto& c : decode_cases()) bad += run_decode(c);
    printf(bad ? "FAIL: %zu mismatches\n" : "PASS\n", bad);
    return bad ? 1 : 0;
}
