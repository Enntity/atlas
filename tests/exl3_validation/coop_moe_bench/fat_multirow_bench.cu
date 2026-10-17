// SPDX-License-Identifier: AGPL-3.0-only
// Grouped multi-row routed-MoE microbenchmark (ATLAS_EXL3_COOP_MOE_FAT_MULTIROW).
//
// Compares the per-slot reference exl3_moe_mul1_matmul_precision_mma (the
// kernel that is ~94 % of Atlas prefill GPU time) against the grouped path:
// exl3_moe_group_units + exl3_coop_moe8_multirow_b4_k32_n256 (thin units) +
// exl3_coop_moe_fat_multirow_b4_k16_n128 (fat units), both fed the same F16
// input transform the production path uses.
//
//  - BIT-IDENTITY: every slot's F32 row must equal the reference kernel's,
//    byte for byte, over routing distributions with expert counts below and
//    above thin_cap and with groups above the 16-row M tile.
//  - TIMING: reference (input transform + per-slot MMA) vs grouped path
//    (F16 transform + unit split + thin + fat) at a realistic prefill shape
//    (rows 128, topk 8 -> 1024 slots, 288 experts).
//
// Build in the spark1 builder container (no GPU there) and run on the Spark
// host, under the shared bench lock. Flags match the production EXL3 modules
// (kernels/gb10/common/KERNEL.toml): --fmad=false -DTQ_PLUS_SIGNS
// --expt-relaxed-constexpr, arch from kernels/gb10/HARDWARE.toml (sm_121f):
//   nvcc -O3 -arch=sm_121f --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr \
//        -I<repo>/kernels/gb10/glm-5.3-flash/exl3 -o mb_fat fat_multirow_bench.cu
// Env: ITERS (timed iterations, default 50), ROWS (default 128), TOPK (8),
//      EXPERTS (288), THIN_CAP (32).
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Production profile (spark-model moe/policy.rs GLM53_TP2_REFERENCE).
static const unsigned TILE_K = 32, TILE_N = 256, SLICES = 8, HALF_PARTIAL = 1;
static const unsigned FAT_TILE_K = 16, FAT_TILE_N = 128, FAT_SLICES = 1, FAT_HALF_PARTIAL = 0;

// The two dynamic shared-memory sizes the Rust side passes (moe/coop_fat.rs).
static const int THIN_SMEM = 6 * (1024 + 4096);
static const int FAT_SMEM = 6 * (512 + 1024);

struct Proj {
    int k, n;
    Exl3Expert* table;
    size_t matrix_bytes;
};

// One expert descriptor set: random 4-bit mul1 trellis, random suh/svh.
static Proj make_proj(int k, int n, int experts) {
    Proj p{k, n, nullptr, (size_t)(k / 16) * (n / 16) * 128};
    std::vector<unsigned long long> trellis(p.matrix_bytes / 8);
    std::vector<Exl3Expert> table(experts);
    for (int e = 0; e < experts; ++e) {
        for (auto& x : trellis) x = next();
        void* d = nullptr;
        CK(cudaMalloc(&d, p.matrix_bytes));
        CK(cudaMemcpy(d, trellis.data(), p.matrix_bytes, cudaMemcpyHostToDevice));
        void* suh = nullptr;
        CK(cudaMalloc(&suh, (size_t)k * 2));
        std::vector<__half> hs(k);
        for (auto& x : hs) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
        CK(cudaMemcpy(suh, hs.data(), (size_t)k * 2, cudaMemcpyHostToDevice));
        void* svh = nullptr;
        CK(cudaMalloc(&svh, (size_t)n * 2));
        std::vector<__half> hv(n);
        for (auto& x : hv) x = __float2half_rn(0.5f + ((next() >> 40) / float(1 << 24)));
        CK(cudaMemcpy(svh, hv.data(), (size_t)n * 2, cudaMemcpyHostToDevice));
        table[e] = Exl3Expert{(unsigned long long)d, (unsigned long long)suh,
                              (unsigned long long)svh, 4u, 0u};
    }
    CK(cudaMalloc(&p.table, sizeof(Exl3Expert) * experts));
    CK(cudaMemcpy(p.table, table.data(), sizeof(Exl3Expert) * experts, cudaMemcpyHostToDevice));
    return p;
}

struct Case {
    const char* name;
    int pool;  // experts the rows draw from; smaller = higher counts per expert
};

int main() {
    int experts = getenv("EXPERTS") ? atoi(getenv("EXPERTS")) : 288;
    int rows = getenv("ROWS") ? atoi(getenv("ROWS")) : 128;
    int topk = getenv("TOPK") ? atoi(getenv("TOPK")) : 8;
    unsigned thin_cap = getenv("THIN_CAP") ? atoi(getenv("THIN_CAP")) : 32;
    int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 50;
    int slots = rows * topk;
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    // Task shapes: gate|up (k=4096, n=2048) and down (k=2048, n=4096).
    Proj gate = make_proj(4096, 2048, experts);
    Proj down = make_proj(2048, 4096, experts);

    // Activations: one BF16 row per token, F16 scratch per slot.
    std::vector<__nv_bfloat16> hx((size_t)rows * 4096);
    for (auto& x : hx) x = __float2bfloat16(((next() >> 40) / float(1 << 24)) * 4.0f - 2.0f);
    __nv_bfloat16* dx = nullptr;
    CK(cudaMalloc(&dx, hx.size() * 2));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));

    float *xh_f32 = nullptr, *y_ref = nullptr, *y_new = nullptr, *y_mma = nullptr;
    __half* xh_f16 = nullptr;
    CK(cudaMalloc(&xh_f32, (size_t)slots * 4096 * 4));
    CK(cudaMalloc(&xh_f16, (size_t)slots * 4096 * 2));
    CK(cudaMalloc(&y_ref, (size_t)slots * 4096 * 4));
    CK(cudaMalloc(&y_new, (size_t)slots * 4096 * 4));
    CK(cudaMalloc(&y_mma, (size_t)slots * 4096 * 4));
    int* ids = nullptr;
    unsigned* counts = nullptr;
    CK(cudaMalloc(&ids, (size_t)slots * 4));
    CK(cudaMalloc(&counts, (size_t)experts * 4));
    // Two `1 + 4 * slots` unit regions (thin, then fat), as in moe/coop_fat.rs.
    int* groups = nullptr;
    CK(cudaMalloc(&groups, (size_t)2 * (1 + 4 * slots) * 4));
    // At most 2 * units lock rows of n/128 ints.
    const size_t lock_ints = (size_t)2 * slots * (4096 / 128);
    int* locks = nullptr;
    CK(cudaMalloc(&locks, lock_ints * 4));

    void* thin_fn = (void*)exl3_coop_moe8_multirow_b4_k32_n256;
    void* fat_fn = (void*)exl3_coop_moe_fat_multirow_b4_k16_n128;
    CK(cudaFuncSetAttribute(thin_fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CK(cudaFuncSetAttribute(fat_fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int thin_occ = 0, fat_occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&thin_occ, thin_fn, 256, THIN_SMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fat_occ, fat_fn, 256, FAT_SMEM));
    const int thin_grid = thin_occ * sms, fat_grid = fat_occ * sms;
    printf("SM %d, thin occupancy %d/SM (grid %d), fat occupancy %d/SM (grid %d)\n", sms,
           thin_occ, thin_grid, fat_occ, fat_grid);
    printf("rows %d, topk %d, slots %d, experts %d, thin_cap %u, iters %d\n", rows, topk, slots,
           experts, thin_cap, iters);

    unsigned u_rows = (unsigned)rows, u_topk = (unsigned)topk, u_experts = (unsigned)experts;
    // ---- Per-slot reference: F32 input transform + exl3_moe_mul1_matmul_precision_mma.
    auto reference = [&](const Proj& p, float* y) {
        unsigned u_k = (unsigned)p.k, u_n = (unsigned)p.n, expanded = 0;
        Exl3Expert* table = p.table;
        {
            void* args[] = {&dx, &ids, &table, &xh_f32, &u_rows, &u_k, &u_topk, &expanded,
                            &u_experts};
            CK(cudaLaunchKernel((void*)exl3_moe_input_hadamard_f16, dim3(p.k / 128, slots, 1),
                                dim3(128, 1, 1), args, 0, 0));
        }
        unsigned tk = TILE_K, tn = TILE_N, sl = SLICES, hp = HALF_PARTIAL;
        unsigned ftk = FAT_TILE_K, ftn = FAT_TILE_N, fsl = FAT_SLICES, fhp = FAT_HALF_PARTIAL;
        void* args[] = {&xh_f32, &ids, &table, &y, &counts, &u_rows, &u_n, &u_k, &u_topk,
                        &u_experts, &thin_cap, &tk, &tn, &sl, &hp, &ftk, &ftn, &fsl, &fhp};
        CK(cudaLaunchKernel((void*)exl3_moe_mul1_matmul_precision_mma,
                            dim3(p.n / 128, slots, 1), dim3(128, 1, 1), args, 0, 0));
    };

    // ---- Grouped path: F16 transform + unit split + thin units + fat units.
    auto grouped = [&](const Proj& p, float* y) {
        unsigned u_k = (unsigned)p.k, u_n = (unsigned)p.n;
        Exl3Expert* table = p.table;
        {
            void* args[] = {&dx, &ids, &table, &xh_f16, &u_rows, &u_k, &u_topk, &u_experts};
            CK(cudaLaunchKernel((void*)exl3_moe_input_hadamard_f16h, dim3(p.k / 128, slots, 1),
                                dim3(128, 1, 1), args, 0, 0));
        }
        {
            int m_tile = 16;
            int u_slots = slots;
            void* args[] = {&ids, &groups, &u_slots, &experts, &thin_cap, &m_tile};
            CK(cudaLaunchKernel((void*)exl3_moe_group_units, dim3(1, 1, 1), dim3(256, 1, 1), args,
                                0, 0));
        }
        {
            int proj = 1, sl = SLICES, u_slots = slots;
            unsigned sc = thin_cap;
            void* args[] = {&xh_f16, &xh_f16, &groups, &table, &table, &y, &y, &counts, &locks,
                            &proj, &u_slots, &u_k, &u_n, &u_experts, &sc, &sl};
            CK(cudaLaunchCooperativeKernel(thin_fn, dim3(thin_grid, 1, 1), dim3(256, 1, 1), args,
                                           THIN_SMEM, 0));
        }
        {
            int proj = 1, fsl = FAT_SLICES, u_slots = slots;
            int* fat_region = groups + (1 + 4 * slots);
            void* args[] = {&xh_f16, &xh_f16, &fat_region, &table, &table, &y, &y, &locks, &proj,
                            &u_slots, &u_k, &u_n, &u_experts, &fsl};
            CK(cudaLaunchCooperativeKernel(fat_fn, dim3(fat_grid, 1, 1), dim3(256, 1, 1), args,
                                           FAT_SMEM, 0));
        }
    };

    // Read the two unit counts back (thin region at 0, fat region at region).
    auto unit_counts = [&](int& g_thin, int& g_fat) {
        int head[2] = {0, 0};
        CK(cudaMemcpy(&head[0], groups, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&head[1], groups + (1 + 4 * slots), 4, cudaMemcpyDeviceToHost));
        g_thin = head[0];
        g_fat = head[1];
    };
    auto locks_zero = [&]() {
        std::vector<int> hlocks(lock_ints);
        CK(cudaMemcpy(hlocks.data(), locks, lock_ints * 4, cudaMemcpyDeviceToHost));
        return std::all_of(hlocks.begin(), hlocks.end(), [](int v) { return v == 0; });
    };

    const Case cases[] = {
        {"pool 288 (realistic prefill, mostly thin)", 288},
        {"pool 48 (counts 17..32, thin groups > 16 rows)", 48},
        {"pool 8 (counts > 128, fat groups over many M tiles)", 8},
    };
    int failures = 0;
    for (const Case& c : cases) {
        // Each row draws topk distinct experts from the first `pool` experts.
        std::vector<int> hids((size_t)slots);
        std::vector<unsigned> hcounts(experts, 0);
        for (int row = 0; row < rows; ++row) {
            std::vector<int> pick;
            while ((int)pick.size() < topk) {
                int e = (int)(next() % (unsigned)c.pool);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (int j = 0; j < topk; ++j) {
                hids[(size_t)row * topk + j] = pick[j];
                hcounts[pick[j]]++;
            }
        }
        CK(cudaMemcpy(ids, hids.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(counts, hcounts.data(), (size_t)experts * 4, cudaMemcpyHostToDevice));

        int max_count = 0, n_fat = 0, n_over16 = 0;
        for (int e = 0; e < experts; ++e) {
            max_count = std::max(max_count, (int)hcounts[e]);
            if (hcounts[e] > (int)thin_cap) n_fat++;
            if (hcounts[e] > 16) n_over16++;
        }
        printf("\n[%s]\n", c.name);
        printf("  counts: max %d, experts >16 rows %d, >thin_cap %d\n", max_count, n_over16, n_fat);

        int g_thin = 0, g_fat = 0;
        bool all_same = true, locks_ok = true;
        for (const Proj* p : {&gate, &down}) {
            const int n = p->n;
            std::vector<float> ref((size_t)slots * n), got((size_t)slots * n);
            CK(cudaMemset(y_ref, 0xff, (size_t)slots * n * 4));
            CK(cudaMemset(y_new, 0xff, (size_t)slots * n * 4));
            CK(cudaMemset(locks, 0, lock_ints * 4));
            reference(*p, y_ref);
            grouped(*p, y_new);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ref.data(), y_ref, ref.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(got.data(), y_new, got.size() * 4, cudaMemcpyDeviceToHost));
            const bool same = !memcmp(ref.data(), got.data(), ref.size() * 4);
            unit_counts(g_thin, g_fat);
            const bool lz = locks_zero();
            locks_ok = locks_ok && lz;
            all_same = all_same && same;
            printf("  %-5s k=%-4d n=%-5d units thin %-4d fat %-4d  %d values per-slot byte "
                   "compare: %-9s locks %s\n",
                   p == &gate ? "gate|up" : "down", p->k, p->n, g_thin, g_fat, (int)ref.size(),
                   same ? "IDENTICAL" : "MISMATCH", lz ? "reset" : "LEAKED");
            if (!same) {
                ++failures;
                for (size_t i = 0; i < ref.size(); ++i)
                    if (memcmp(&ref[i], &got[i], 4) != 0) {
                        printf("    first mismatch slot %zu column %zu: ref %08x got %08x\n",
                               i / n, i % n, *(unsigned*)&ref[i], *(unsigned*)&got[i]);
                        break;
                    }
            }
            if (!lz) ++failures;
        }

        // ---- Timing for this distribution.
        for (const Proj* p : {&gate, &down}) {
            unit_counts(g_thin, g_fat);
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a));
            CK(cudaEventCreate(&b));
            CK(cudaMemset(locks, 0, lock_ints * 4));
            for (int i = 0; i < 3; ++i) reference(*p, y_mma);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for (int i = 0; i < iters; ++i) reference(*p, y_mma);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ref_ms = 0;
            CK(cudaEventElapsedTime(&ref_ms, a, b));
            const double ref_us = ref_ms * 1e3 / iters;

            CK(cudaMemset(locks, 0, lock_ints * 4));
            for (int i = 0; i < 3; ++i) grouped(*p, y_new);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for (int i = 0; i < iters; ++i) grouped(*p, y_new);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float new_ms = 0;
            CK(cudaEventElapsedTime(&new_ms, a, b));
            const double new_us = new_ms * 1e3 / iters;
            CK(cudaEventDestroy(a));
            CK(cudaEventDestroy(b));
            const double ref_bytes = (double)slots * p->matrix_bytes;
            const double new_bytes = (double)(g_thin + g_fat) * p->matrix_bytes;
            printf("  TIMING %-5s k=%-4d n=%-5d  before %9.1f us (weights %6.2f GB)  after %9.1f "
                   "us (weights %6.2f GB)  speedup %5.2fx  locks %s\n",
                   p == &gate ? "gate|up" : "down", p->k, p->n, ref_us, ref_bytes / 1e9, new_us,
                   new_bytes / 1e9, ref_us / new_us, locks_zero() ? "reset" : "LEAKED");
        }
    }

    printf("\n%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
