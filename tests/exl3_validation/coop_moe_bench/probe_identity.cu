// SPDX-License-Identifier: AGPL-3.0-only
// Identity probe for the routed-MoE arithmetic: which of the existing paths
// already agrees with which, before attributing any difference to the grouped
// multi-row work. For one expert with `count` = slots selected by every slot,
// compares every output element of:
//   native  = exl3_moe_mul1_matmul_precision_mma
//   split   = exl3_moe_mul1_matmul_precision_mma_split
//   coopv1  = exl3_coop_moe_thin_b4_k32_n256 (per-slot coop thin)
//   multirow= exl3_coop_moe8_multirow_b4_k32_n256 (grouped thin)
//   fatmr   = exl3_coop_moe_fat_multirow_b4_k16_n128 (grouped fat)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "exl3_expert.cuh"
#include "exl3_coop_gemm.cu"
#include "exl3_moe_precision.cu"
#include "exl3_moe_mma.cu"

#define CK(x)                                                                 \
    do {                                                                      \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static inline unsigned long long next() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static const int K = 4096, N = 2048, EXPERTS = 8, SLOTS_MAX = 128;
static unsigned THIN_CAP = 32;
static const int THIN_SMEM = 6 * (1024 + 4096), FAT_SMEM = 6 * (512 + 1024);
static const int V1_SMEM = 90 * 1024;

int main() {
    const size_t matrix_bytes = (size_t)(K / 16) * (N / 16) * 128;
    std::vector<unsigned long long> trellis(matrix_bytes / 8);
    std::vector<Exl3Expert> table(EXPERTS);
    for (int e = 0; e < EXPERTS; ++e) {
        for (auto& x : trellis) x = next();
        void* d = nullptr;
        CK(cudaMalloc(&d, matrix_bytes));
        CK(cudaMemcpy(d, trellis.data(), matrix_bytes, cudaMemcpyHostToDevice));
        std::vector<__half> hs(K);
        for (auto& x : hs) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
        void* suh = nullptr;
        CK(cudaMalloc(&suh, K * 2));
        CK(cudaMemcpy(suh, hs.data(), K * 2, cudaMemcpyHostToDevice));
        table[e] = Exl3Expert{(unsigned long long)d, (unsigned long long)suh, 0, 4u, 0u};
    }
    Exl3Expert* dt = nullptr;
    CK(cudaMalloc(&dt, sizeof(Exl3Expert) * EXPERTS));
    CK(cudaMemcpy(dt, table.data(), sizeof(Exl3Expert) * EXPERTS, cudaMemcpyHostToDevice));

    float* xh_f32 = nullptr;
    __half* xh_f16 = nullptr;
    CK(cudaMalloc(&xh_f32, (size_t)SLOTS_MAX * K * 4));
    CK(cudaMalloc(&xh_f16, (size_t)SLOTS_MAX * K * 2));
    float* ya = nullptr;
    float* yb = nullptr;
    CK(cudaMalloc(&ya, (size_t)SLOTS_MAX * N * 4));
    CK(cudaMalloc(&yb, (size_t)SLOTS_MAX * N * 4));
    int* ids = nullptr;
    unsigned* counts = nullptr;
    CK(cudaMalloc(&ids, (size_t)SLOTS_MAX * 4));
    CK(cudaMalloc(&counts, sizeof(unsigned) * EXPERTS));
    int* groups = nullptr;
    CK(cudaMalloc(&groups, (size_t)2 * (1 + 4 * SLOTS_MAX) * 4));
    const size_t lock_ints = (size_t)2 * SLOTS_MAX * (N / 128);
    int* locks = nullptr;
    CK(cudaMalloc(&locks, lock_ints * 4));

    CK(cudaFuncSetAttribute((void*)exl3_coop_moe_thin_b4_k32_n256,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, V1_SMEM));
    int v1_occ = 0, sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &v1_occ, (void*)exl3_coop_moe_thin_b4_k32_n256, 512, V1_SMEM));
    const int v1_grid = v1_occ * sms;

    std::vector<__nv_bfloat16> hx(K);
    for (auto& x : hx) x = __float2bfloat16(((next() >> 40) / float(1 << 24)) * 4.0f - 2.0f);
    __nv_bfloat16* dx = nullptr;
    CK(cudaMalloc(&dx, K * 2));
    CK(cudaMemcpy(dx, hx.data(), K * 2, cudaMemcpyHostToDevice));

    auto diff = [](const char* name, const std::vector<float>& a, const std::vector<float>& b) {
        size_t bad = 0;
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (memcmp(&a[i], &b[i], 4) != 0) {
                ++bad;
                const double d = std::fabs((double)a[i] - (double)b[i]) /
                                 std::max(1e-30, std::fabs((double)a[i]));
                worst = std::max(worst, d);
            }
        }
        printf("  native vs %-20s: %5zu / %5zu differ, worst rel %.3e\n", name, bad, a.size(),
               worst);
    };

    struct Case {
        const char* name;
        int slots;
        int expert;
    };
    const Case cases[] = {
        {"thin, 1 unit (8 rows)", 8, 3},
        {"thin, 2 units (20 rows)", 20, 3},
        {"fat, 5 units (65 rows)", 65, 5},
    };
    for (const Case& c : cases) {
        const int slots = c.slots, rows = slots;
        std::vector<int> hids((size_t)slots, c.expert);
        std::vector<unsigned> hcounts(EXPERTS, 0);
        hcounts[c.expert] = (unsigned)slots;
        CK(cudaMemcpy(ids, hids.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(counts, hcounts.data(), sizeof(unsigned) * EXPERTS, cudaMemcpyHostToDevice));
        const bool fat = hcounts[c.expert] > THIN_CAP;

        unsigned u_rows = (unsigned)rows, u_k = K, u_n = N, u_topk = 1, u_experts = EXPERTS,
                 exp0 = 0;
        {
            void* args[] = {&dx, &ids, &dt, &xh_f32, &u_rows, &u_k, &u_topk, &exp0, &u_experts};
            CK(cudaLaunchKernel((void*)exl3_moe_input_hadamard_f16, dim3(K / 128, slots, 1),
                                dim3(128, 1, 1), args, 0, 0));
        }
        {
            void* args[] = {&dx, &ids, &dt, &xh_f16, &u_rows, &u_k, &u_topk, &u_experts};
            CK(cudaLaunchKernel((void*)exl3_moe_input_hadamard_f16h, dim3(K / 128, slots, 1),
                                dim3(128, 1, 1), args, 0, 0));
        }
        auto native = [&](bool split, float* y) {
            unsigned tk = 32, tn = 256, sl = 8, hp = 1, ftk = 16, ftn = 128, fsl = 1, fhp = 0;
            void* args[] = {&xh_f32, &ids, &dt, &y, &counts, &u_rows, &u_n, &u_k, &u_topk,
                            &u_experts, &THIN_CAP, &tk, &tn, &sl, &hp, &ftk, &ftn, &fsl, &fhp};
            void* fn = (void*)(split ? exl3_moe_mul1_matmul_precision_mma_split
                                     : exl3_moe_mul1_matmul_precision_mma);
            CK(cudaLaunchKernel(fn, dim3(N / (split ? 16 : 128), slots, 1), dim3(128, 1, 1), args,
                                0, 0));
        };
        std::vector<float> ref((size_t)slots * N), got((size_t)slots * N);
        CK(cudaMemset(locks, 0, lock_ints * 4));
        native(false, ya);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ref.data(), ya, ref.size() * 4, cudaMemcpyDeviceToHost));
        printf("\n[%s] expert %d, count %d\n", c.name, c.expert, slots);

        auto capture = [&](auto fn) {
            CK(cudaMemset(locks, 0, lock_ints * 4));
            CK(cudaMemset(yb, 0xee, (size_t)slots * N * 4));
            fn(yb);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got.data(), yb, got.size() * 4, cudaMemcpyDeviceToHost));
        };
        capture([&](float* y) { native(true, y); });
        diff("native_split", ref, got);
        capture([&](float* y) {
            int u_slots = slots, u_k = K, u_n = N, u_experts = EXPERTS, u_slices = 8;
            unsigned sc = THIN_CAP;
            int items = u_slots * u_slices, grid = std::min(items, v1_grid);
            if (grid < 1) grid = 1;
            void* args[] = {&xh_f32, &ids, &dt, &y, &counts, &locks, &u_slots, &u_k, &u_n,
                            &u_experts, &sc, &u_slices};
            CK(cudaLaunchCooperativeKernel((void*)exl3_coop_moe_thin_b4_k32_n256, dim3(grid, 1, 1),
                                           dim3(512, 1, 1), args, V1_SMEM, 0));
        });
        diff("coop_v1_per_slot", ref, got);
        auto group = [&](float* y) {
            int u_slots = slots, u_experts = EXPERTS, m = 16;
            unsigned sc = THIN_CAP;
            (void)y;
            void* args[] = {&ids, &groups, &u_slots, &u_experts, &sc, &m};
            CK(cudaLaunchKernel((void*)exl3_moe_group_units, dim3(1, 1, 1), dim3(256, 1, 1), args,
                                0, 0));
        };
        if (!fat) {
            capture([&](float* y) {
                group(y);
                int u_slots = slots, u_k = K, u_n = N, u_experts = EXPERTS, u_slices = 8, proj = 1;
                unsigned sc = THIN_CAP;
                void* args[] = {&xh_f16, &xh_f16, &groups, &dt, &dt, &y, &y, &counts, &locks,
                                &proj, &u_slots, &u_k, &u_n, &u_experts, &sc, &u_slices};
                CK(cudaLaunchCooperativeKernel((void*)exl3_coop_moe8_multirow_b4_k32_n256,
                                               dim3(96, 1, 1), dim3(256, 1, 1), args, THIN_SMEM,
                                               0));
            });
            diff("coop_multirow_thin", ref, got);
        } else {
            capture([&](float* y) {
                group(y);
                int u_slots = slots, u_k = K, u_n = N, u_experts = EXPERTS, u_slices = 1, proj = 1;
                int* fat_region = groups + (1 + 4 * slots);
                void* args[] = {&xh_f16, &xh_f16, &fat_region, &dt, &dt, &y, &y, &locks, &proj,
                                &u_slots, &u_k, &u_n, &u_experts, &u_slices};
                CK(cudaLaunchCooperativeKernel((void*)exl3_coop_moe_fat_multirow_b4_k16_n128,
                                               dim3(96, 1, 1), dim3(256, 1, 1), args, FAT_SMEM, 0));
            });
            diff("coop_fat_multirow", ref, got);
        }
    }
    return 0;
}
