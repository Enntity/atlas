// SPDX-License-Identifier: AGPL-3.0-only

// Byte-exact oracle test for kda_fla_solve_tril and kda_fla_recompute_w_u.
// Each kernel runs from oracle inputs (A -> Ai; Ai/k/v/beta/g -> w, u, kg)
// for every prefill case of both ranks. Exit 0 only if every byte matches.
#include "harness.cuh"
#include "kda_fla_wy.cuh"

using namespace kda_harness;

int main() {
    size_t total_bad = 0;
    for (const std::string& c : prefill_cases()) {
        const int T = rows_of(c);
        const unsigned nt = (T + BT - 1) / BT;
        printf("== %s (T=%d)\n", c.c_str(), T);

        auto A = stage(c, "A");
        auto Ai_want = stage(c, "Ai");
        float* dA = to_device<float>(A);
        __nv_bfloat16* dAi = device_zeros<__nv_bfloat16>(Ai_want.size());
        kda_fla_solve_tril<<<dim3(nt, HL), 64>>>(dA, dAi, T, HL);
        KDA_CHECK(cudaGetLastError());
        total_bad += compare("Ai", to_host(dAi, Ai_want.size()), Ai_want, 2, HL * BT);

        auto k = stage(c, "k_l2");
        auto conv = stage(c, "conv_out");
        auto beta = stage(c, "beta");
        auto g = stage(c, "g_cumsum");
        auto w_want = stage(c, "w");
        auto u_want = stage(c, "u");
        auto kg_want = stage(c, "kg");
        __nv_bfloat16* dk = to_device<__nv_bfloat16>(k);
        __nv_bfloat16* dconv = to_device<__nv_bfloat16>(conv);
        float* dbeta = to_device<float>(beta);
        float* dg = to_device<float>(g);
        __nv_bfloat16* dAiw = to_device<__nv_bfloat16>(Ai_want);
        __nv_bfloat16* dw = device_zeros<__nv_bfloat16>(w_want.size());
        __nv_bfloat16* du = device_zeros<__nv_bfloat16>(u_want.size());
        __nv_bfloat16* dkg = device_zeros<__nv_bfloat16>(kg_want.size());
        const unsigned row = 3 * HL * D;
        kda_fla_recompute_w_u<<<dim3(nt, HL), 128>>>(dk, dconv + 2 * HL * D, row, dbeta, dAiw, dg, dw, du, dkg,
                                                      T, HL);
        KDA_CHECK(cudaGetLastError());
        total_bad += compare("u", to_host(du, u_want.size()), u_want, 2, HL * D);
        total_bad += compare("w", to_host(dw, w_want.size()), w_want, 2, HL * D);
        total_bad += compare("kg", to_host(dkg, kg_want.size()), kg_want, 2, HL * D);

        for (void* p : {(void*)dA, (void*)dAi, (void*)dk, (void*)dconv, (void*)dbeta, (void*)dg, (void*)dAiw,
                        (void*)dw, (void*)du, (void*)dkg})
            KDA_CHECK(cudaFree(p));
    }
    printf("TOTAL mismatches: %zu\n", total_bad);
    return total_bad == 0 ? 0 : 1;
}
