// SPDX-License-Identifier: AGPL-3.0-only

// Bounded GB10 correctness/performance gate. Build from the repository root:
// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_sparse_mla_kv_reuse.cu -o /tmp/bench-glm-kv-reuse
// No model weights, collectives, persistent kernels, or large allocations.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_tc.cu"
#include "glm_sparse_mla_head32_tc_kv_reuse.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

template <typename T> struct Buffer {
    T* ptr;
    explicit Buffer(size_t count) {
        // Width-zero attention has an empty index tensor but still exercises
        // output writes; keep its unused device pointer valid.
        CUDA_CHECK(cudaMalloc(&ptr, std::max(count, size_t(1)) * sizeof(T)));
    }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void copy(const std::vector<T>& v) {
        if (v.empty()) return;
        CUDA_CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

static void run(unsigned rows, unsigned heads, unsigned width, bool alias_kv,
                bool masked, unsigned repeats, float scale = 0.0625f,
                bool mask_final_three = false) {
    if (!alias_kv) { std::fprintf(stderr, "K=V candidate requires aliased input\n"); std::exit(3); }
    constexpr unsigned dim = 512, block_size = 16, history = 16384;
    const size_t nq = size_t(rows) * heads * dim;
    const size_t nc = size_t(history) * dim;
    std::mt19937 rng(7391 + rows + heads + width);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> q(nq), k(nc), v(nc);
    for (auto& x : q) x = __float2bfloat16(dist(rng));
    for (auto& x : k) x = __float2bfloat16(dist(rng));
    for (auto& x : v) x = __float2bfloat16(dist(rng));
    std::vector<unsigned> table(history / block_size);
    for (unsigned i = 0; i < table.size(); ++i) table[i] = i;
    std::reverse(table.begin(), table.end());
    std::vector<int> indices(size_t(rows) * width);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned i = 0; i < width; ++i) {
            // Includes internal holes, an all-masked row, repeated indices,
            // noncontiguous physical pages, and an incomplete final tile.
            const bool tail_mask = mask_final_three && width >= 3 && i >= width - 3;
            indices[size_t(r) * width + i] = tail_mask || (masked && (r == 0 || i % 13 == 0))
                ? -1 : int(rng() % history);
        }
    }
    Buffer<__nv_bfloat16> dq(nq), dk(nc), dv(nc), baseline(nq), candidate(nq);
    Buffer<int> di(indices.size());
    Buffer<unsigned> dt(table.size());
    dq.copy(q); dk.copy(k); dv.copy(v); di.copy(indices); dt.copy(table);
    // A skipped write must fail even for empty/all-masked attention, whose
    // correct result is zero. BF16 0xffff is NaN, checked below.
    CUDA_CHECK(cudaMemset(baseline.ptr, 0xff, nq * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(candidate.ptr, 0xff, nq * sizeof(__nv_bfloat16)));
    const auto launch = [&](bool optimized) {
        const auto vp = alias_kv ? dk.ptr : dv.ptr;
            if (optimized) {
            glm_sparse_mla_prefill_bf16_head32_tc_kv_reuse<<<dim3((heads + 31) / 32, rows), 256, 68352>>>(
                dq.ptr, dk.ptr, vp, di.ptr, candidate.ptr, dt.ptr,
                rows, heads, dim, width, block_size, scale);
        } else {
            glm_sparse_mla_prefill_bf16_head32_tc<<<dim3((heads + 31) / 32, rows), 256, 101120>>>(
                dq.ptr, dk.ptr, vp, di.ptr, baseline.ptr, dt.ptr,
                rows, heads, dim, width, block_size, scale);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    CUDA_CHECK(cudaFuncSetAttribute(glm_sparse_mla_prefill_bf16_head32_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, 101120));
    CUDA_CHECK(cudaFuncSetAttribute(glm_sparse_mla_prefill_bf16_head32_tc_kv_reuse, cudaFuncAttributeMaxDynamicSharedMemorySize, 68352));
    launch(false); launch(true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> a(nq), b(nq);
    CUDA_CHECK(cudaMemcpy(a.data(), baseline.ptr, nq * sizeof(a[0]), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), candidate.ptr, nq * sizeof(b[0]), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    float max_diff = 0.0f;
    double baseline_norm = 0.0, delta_norm = 0.0;
    for (size_t i = 0; i < nq; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(a[i]))) ++mismatches;
        const float av = __bfloat162float(a[i]), bv = __bfloat162float(b[i]);
        if (!std::isfinite(av) || !std::isfinite(bv)) std::exit(2);
        max_diff = std::max(max_diff, std::abs(av - bv));
        baseline_norm += double(av) * av;
        delta_norm += double(av - bv) * (av - bv);
        if (std::abs(av - bv) > 0.002f + 0.01f * std::abs(av)) {
            std::fprintf(stderr, "baseline element tolerance failed: index=%zu baseline=%g candidate=%g\n", i, av, bv);
            std::exit(2);
        }
    }
    if (mismatches) { std::fprintf(stderr, "K=V reuse changed TC arithmetic: %zu differences\n", mismatches); std::exit(2); }
    // Independent double-precision oracle on small shapes, including masking.
    float max_reference_error = 0.0f;
    if (rows <= 7) {
        const auto& values = alias_kv ? k : v;
        std::vector<double> scores(width);
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned h = 0; h < heads; ++h) {
                double max_score = -INFINITY;
                for (unsigned i = 0; i < width; ++i) {
                    const int token = indices[size_t(r) * width + i];
                    double score = -INFINITY;
                    if (token >= 0) {
                        const size_t pos = size_t(table[token / block_size]) * block_size
                            + token % block_size;
                        score = 0.0;
                        for (unsigned d = 0; d < dim; ++d)
                            score += double(__bfloat162float(q[(size_t(r) * heads + h) * dim + d]))
                                * __bfloat162float(k[pos * dim + d]);
                        score *= double(scale);
                    }
                    scores[i] = score;
                    max_score = std::max(max_score, score);
                }
                double denom = 0.0;
                for (auto& s : scores) {
                    s = std::isfinite(s) ? std::exp(s - max_score) : 0.0;
                    denom += s;
                }
                for (unsigned d = 0; d < dim; ++d) {
                    double expected = 0.0;
                    for (unsigned i = 0; i < width; ++i) {
                        const int token = indices[size_t(r) * width + i];
                        if (token < 0) continue;
                        const size_t pos = size_t(table[token / block_size]) * block_size
                            + token % block_size;
                        expected += scores[i] * __bfloat162float(values[pos * dim + d]);
                    }
                    expected = denom ? expected / denom : 0.0;
                    const float actual = __bfloat162float(b[(size_t(r) * heads + h) * dim + d]);
                    max_reference_error = std::max(max_reference_error, float(std::abs(actual - expected)));
                    if (std::abs(actual - expected) > 0.002 + 0.01 * std::abs(expected)) {
                        std::fprintf(stderr, "independent attention oracle failed\n");
                        std::exit(2);
                    }
                }
            }
        }
    }
    const double relative_l2 = std::sqrt(delta_norm / std::max(baseline_norm, 1e-30));
    if (relative_l2 > 0.008) {
        std::fprintf(stderr, "baseline relative L2 tolerance failed: %g\n", relative_l2);
        std::exit(2);
    }
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> times[2];
    // Alternate order to reduce clock/thermal bias; correctness warmups above.
    for (unsigned rep = 0; rep < repeats; ++rep) {
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned mode = (rep + j) % 2;
            CUDA_CHECK(cudaEventRecord(start));
            launch(mode != 0);
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
            times[mode].push_back(ms);
        }
    }
    for (auto& t : times) std::sort(t.begin(), t.end());
    const float before = times[0][repeats / 2], after = times[1][repeats / 2];
    std::printf("{\"rows\":%u,\"heads\":%u,\"width\":%u,\"alias_kv\":%s,"
        "\"masked\":%s,\"scale\":%g,\"mask_final_three\":%s,"
        "\"elements\":%zu,\"mismatches\":%zu,\"reference_max_error\":%g,"
        "\"relative_l2\":%g,\"max_diff\":%g,\"baseline_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.3f}\n",
        rows, heads, width, alias_kv ? "true" : "false", masked ? "true" : "false",
        scale, mask_final_three ? "true" : "false",
        nq, mismatches, max_reference_error, relative_l2, max_diff, before, after, before / after);
    std::fflush(stdout);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(end));
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 1) {
        std::fprintf(stderr, "This bounded gate is intended for GB10/sm121.\n");
        return 1;
    }
    // Independent oracle: same exact K/V latent, masked rows/holes, tail,
    // reverse pages, duplicate IDs and partial heads. Bit identity to TC also required.
    run(1, 1, 1, true, false, 5);
    run(7, 13, 19, true, true, 5);
    run(3, 32, 257, true, true, 5);
    run(3, 32, 2051, true, false, 5);
    run(3, 32, 2051, true, true, 5, 0.0625f, true);
    run(3, 32, 2051, true, true, 5, 0.25f, true);
    run(3, 13, 0, true, false, 5);
    run(128, 32, 2048, true, false, 5);
    run(512, 32, 2051, true, false, 5);
    run(1024, 32, 2051, true, false, 5, 0.0625f, true);
    run(2048, 32, 2051, true, false, 5, 0.0625f, true);
    return 0;
}
