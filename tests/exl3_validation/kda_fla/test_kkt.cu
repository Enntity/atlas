// SPDX-License-Identifier: AGPL-3.0-only

// Byte-exact test of kda_fla_kkt_inter + kda_fla_kkt_intra against the
// reference KKT stage oracle (A, Aqk) for every prefill case on both ranks.
#include "harness.cuh"
#include "kda_fla_kkt.cuh"

using namespace kda_harness;

int main() {
    size_t total_bad = 0;
    const float scale = 0.08838834764831845f;  // 128 ** -0.5, as the reference passes it
    for (const auto& c : prefill_cases()) {
        const int T = rows_of(c);
        auto q = to_device<__nv_bfloat16>(stage(c, "q_l2"));
        auto k = to_device<__nv_bfloat16>(stage(c, "k_l2"));
        auto g = to_device<float>(stage(c, "g_cumsum"));
        auto b = to_device<float>(stage(c, "beta"));
        const size_t abytes = (size_t)T * HL * BT * sizeof(float);
        auto A = device_zeros<float>(abytes);
        auto Aqk = device_zeros<float>(abytes);
        const unsigned nt = (T + BT - 1) / BT;
        kda_fla_kkt_inter<<<dim3(nt, 16, HL), 32>>>(q, k, g, b, A, Aqk, scale, T, HL);
        KDA_CHECK(cudaGetLastError());
        kda_fla_kkt_intra<<<dim3(nt, 4, HL), 256>>>(q, k, g, b, A, Aqk, scale, T, HL);
        KDA_CHECK(cudaGetLastError());
        printf("== %s (T=%d)\n", c.c_str(), T);
        size_t bad = compare("A", to_host(A, abytes), stage(c, "A"), 4, HL * BT);
        bad += compare("Aqk", to_host(Aqk, abytes), stage(c, "Aqk"), 4, HL * BT);
        total_bad += bad;
        cudaFree(q); cudaFree(k); cudaFree(g); cudaFree(b); cudaFree(A); cudaFree(Aqk);
    }
    printf("TOTAL mismatches: %zu\n", total_bad);
    return total_bad == 0 ? 0 : 1;
}
