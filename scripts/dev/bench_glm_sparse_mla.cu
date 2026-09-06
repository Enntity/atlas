// SPDX-License-Identifier: AGPL-3.0-only

// Bounded GB10 correctness/performance gate. Build from the repository root:
// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_sparse_mla.cu -o /tmp/bench-glm-sparse
// No model weights, collectives, persistent kernels, or large allocations.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../kernels/gb10/common/glm_indexer.cu"
#include "glm_sparse_mla_warp.cuh"

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
                bool masked, unsigned repeats, float scale = 0.125f,
                bool mask_final_three = false) {
    constexpr unsigned dim = 512, block_size = 16, history = 4096;
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
    std::shuffle(table.begin(), table.end(), rng);
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
        const dim3 grid((heads + 7) / 8, rows);
        if (optimized) {
            glm_sparse_mla_prefill_bf16_head8_warp<<<grid, 256>>>(
                dq.ptr, dk.ptr, vp, di.ptr, candidate.ptr, dt.ptr,
                rows, heads, dim, width, block_size, scale);
        } else {
            glm_sparse_mla_prefill_bf16_head8<<<grid, 256, 152 * sizeof(float)>>>(
                dq.ptr, dk.ptr, vp, di.ptr, baseline.ptr, dt.ptr,
                rows, heads, dim, width, block_size, scale);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    launch(false); launch(true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> a(nq), b(nq);
    CUDA_CHECK(cudaMemcpy(a.data(), baseline.ptr, nq * sizeof(a[0]), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), candidate.ptr, nq * sizeof(b[0]), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < nq; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(a[i]))) ++mismatches;
        const float av = __bfloat162float(a[i]), bv = __bfloat162float(b[i]);
        if (!std::isfinite(av) || !std::isfinite(bv)) std::exit(2);
        max_diff = std::max(max_diff, std::abs(av - bv));
    }
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
                    if (std::abs(actual - expected) > 0.002 + 0.004 * std::abs(expected)) {
                        std::fprintf(stderr, "independent attention oracle failed\n");
                        std::exit(2);
                    }
                }
            }
        }
    }
    if (mismatches) {
        std::fprintf(stderr, "bit identity failed: %zu / %zu, max diff %g\n", mismatches, nq, max_diff);
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
        "\"baseline_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.3f}\n",
        rows, heads, width, alias_kv ? "true" : "false", masked ? "true" : "false",
        scale, mask_final_three ? "true" : "false",
        nq, mismatches, max_reference_error, before, after, before / after);
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
    run(1, 1, 1, false, false, 5);
    run(7, 13, 19, false, true, 5);
    run(3, 8, 257, true, true, 5);
    run(32, 32, 2048, false, false, 7);
    run(128, 32, 2048, true, false, 7);
    run(512, 32, 2048, true, false, 7);
    // GLM's loader sets head_dim = qk_nope_head_dim + qk_rope_head_dim
    // = 256 + 0. paged_glm.rs passes effective_attn_scale(hd), which is
    // 1/sqrt(256) = 0.0625, independent of the absorbed 512-wide KV latent.
    constexpr float glm_scale = 0.0625f;
    // The selector allocates index_topk + index_kpool - 1 = 2051 entries.
    // Exercise both a fully valid final partial tile and its masked-tail form.
    run(3, 32, 2051, true, false, 5, glm_scale);
    run(3, 32, 2051, false, true, 5, glm_scale, true);
    run(3, 13, 0, false, false, 5, glm_scale);
    return 0;
}
