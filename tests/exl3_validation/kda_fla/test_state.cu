// SPDX-License-Identifier: AGPL-3.0-only

// Byte-exact test of kda_fla_delta_h and kda_fla_output against the reference
// stage oracle, for every prefill case on both ranks. delta_h consumes oracle
// kg/w/u/g_cumsum/initial_state; output consumes oracle q_l2/v_new/g_cumsum/Aqk/h.
#include "harness.cuh"
#include "kda_fla_state.cuh"
#include <cmath>

using namespace kda_harness;

static size_t run_case(const std::string& c) {
    const int T = rows_of(c);
    const int NT = (T + BT - 1) / BT;
    const size_t plane = (size_t)T * HL * D;
    const size_t state_bytes = (size_t)HL * D * D * 4;
    printf("== %s (T=%d)\n", c.c_str(), T);

    auto kg = to_device<__nv_bfloat16>(stage(c, "kg"));
    auto w = to_device<__nv_bfloat16>(stage(c, "w"));
    auto u = to_device<__nv_bfloat16>(stage(c, "u"));
    auto g = to_device<float>(stage(c, "g_cumsum"));
    const bool has_init = c.find("seq") != std::string::npos;
    std::vector<uint8_t> init = has_init ? transpose_state(stage(c, "initial_state"), HL)
                                         : std::vector<uint8_t>(state_bytes, 0);
    auto state = to_device<float>(init);
    auto h = device_zeros<__nv_bfloat16>((size_t)NT * HL * D * D * 2);
    auto vnew = device_zeros<__nv_bfloat16>(plane * 2);

    kda_fla_delta_h<<<dim3(D / kda_fla_state::DELTA_BV, HL), 128>>>(kg, w, u, g, state, h, vnew, T, HL);
    KDA_CHECK(cudaGetLastError());
    size_t bad = 0;
    bad += compare("h", to_host(h, (size_t)NT * HL * D * D * 2), stage(c, "h"), 2);
    bad += compare("v_new", to_host(vnew, plane * 2), stage(c, "v_new"), 2, HL * D);
    bad += compare("final_state", to_host(state, state_bytes), transpose_state(stage(c, "final_state"), HL), 4);

    auto q = to_device<__nv_bfloat16>(stage(c, "q_l2"));
    auto vref = to_device<__nv_bfloat16>(stage(c, "v_new"));
    auto aqk = to_device<float>(stage(c, "Aqk"));
    auto href = to_device<__nv_bfloat16>(stage(c, "h"));
    auto o = device_zeros<__nv_bfloat16>(plane * 2);
    const float scale = (float)(1.0 / std::sqrt(128.0));
    kda_fla_output<<<dim3(D / kda_fla_state::OUT_BV, NT, HL), 128>>>(q, vref, g, aqk, href, o, scale, T, HL);
    KDA_CHECK(cudaGetLastError());
    bad += compare("output", to_host(o, plane * 2), stage(c, "output"), 2, HL * D);

    for (void* p : {(void*)kg, (void*)w, (void*)u, (void*)g, (void*)state, (void*)h, (void*)vnew, (void*)q,
                    (void*)vref, (void*)aqk, (void*)href, (void*)o})
        cudaFree(p);
    return bad;
}

int main() {
    size_t bad = 0;
    for (const auto& c : prefill_cases()) bad += run_case(c);
    printf(bad ? "FAIL: %zu mismatches\n" : "PASS\n", bad);
    return bad ? 1 : 0;
}
