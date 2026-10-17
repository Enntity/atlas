// SPDX-License-Identifier: AGPL-3.0-only

// Byte-exact test of kda_fla_recurrent_decode against the reference decode
// stage oracle (captured decode 000325 and carried-state decodes seq4..seq7).
#include "harness.cuh"
#include "kda_fla_decode.cuh"

using namespace kda_harness;

int main() {
    size_t total_bad = 0;
    for (const std::string& c : decode_cases()) {
        const std::string rank = c.substr(0, 5);
        std::vector<uint8_t> g1, beta;
        if (c.find("000325") != std::string::npos) {
            const std::string raw = oracle_root() + "/raw/ffn-L0-" + rank + "-000325-kda-core-input-";
            g1 = read_file(raw + "g1.bin");
            beta = read_file(raw + "beta.bin");
        } else {
            g1 = stage(c, "in_g1");
            beta = stage(c, "in_beta_raw");
        }
        const std::string p = oracle_root() + "/params/" + rank;
        auto* d_qkv = to_device<__nv_bfloat16>(stage(c, "conv_out"));
        auto* d_g = to_device<__nv_bfloat16>(g1);
        auto* d_b = to_device<__nv_bfloat16>(beta);
        auto* d_alog = to_device<float>(read_file(p + "-a_log_f32.bin"));
        auto* d_bias = to_device<float>(read_file(p + "-dt_bias_f32.bin"));
        auto* d_state = to_device<float>(transpose_state(stage(c, "initial_state"), HL));
        auto* d_out = device_zeros<__nv_bfloat16>((size_t)HL * D * 2);

        kda_fla_recurrent_decode<<<dim3(D / 32, HL), 128>>>(d_qkv, d_g, d_b, d_alog, d_bias, d_state, d_out, HL);
        KDA_CHECK(cudaGetLastError());

        printf("== %s\n", c.c_str());
        size_t bad = compare("output", to_host(d_out, (size_t)HL * D * 2), stage(c, "output"), 2, D);
        bad += compare("final_state", transpose_state(to_host(d_state, (size_t)HL * D * D * 4), HL),
                       stage(c, "final_state"), 4, D * D);
        total_bad += bad;
        for (void* ptr : {(void*)d_qkv, (void*)d_g, (void*)d_b, (void*)d_alog, (void*)d_bias, (void*)d_state,
                          (void*)d_out})
            KDA_CHECK(cudaFree(ptr));
    }
    printf("TOTAL mismatches: %zu\n", total_bad);
    return total_bad == 0 ? 0 : 1;
}
