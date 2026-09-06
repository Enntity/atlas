// SPDX-License-Identifier: AGPL-3.0-only

// Checkpoint-free SM121 gate; all allocations together stay below 150 MiB.
// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_indexer.cu -o /tmp/bench-glm-indexer
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "../../kernels/gb10/common/glm_indexer.cu"
#include "../../kernels/gb10/common/glm_indexer_wmma.cu"

#define CUDA_CHECK(call) do { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

static void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(2);
    }
}

template <typename T> struct Buffer {
    T* ptr;
    explicit Buffer(size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void copy(const std::vector<T>& source) {
        CUDA_CHECK(cudaMemcpy(ptr, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

static float bf(__nv_bfloat16 value) { return __bfloat162float(value); }

static std::vector<unsigned> selected_pools(
    const std::vector<int>& selected, const std::vector<float>& scores,
    unsigned sequence_length, bool zero_keys) {
    constexpr unsigned pool = 4, topk = 2048;
    const unsigned count = sequence_length / pool;
    const unsigned budget = std::min(count, topk / pool);
    std::vector<unsigned> result;
    for (unsigned i = 0; i < topk; i += pool) {
        if (selected[i] < 0) {
            for (unsigned j = 0; j < pool; ++j)
                require(selected[i + j] == -1, "partly masked selected pool");
            continue;
        }
        require(selected[i] % int(pool) == 0, "selected pool is not aligned");
        const unsigned p = unsigned(selected[i]) / pool;
        require(p < count, "selector exposed a causal/future pool");
        for (unsigned j = 0; j < pool; ++j)
            require(selected[i + j] == int(p * pool + j), "pool expansion mismatch");
        result.push_back(p);
    }
    require(result.size() == budget, "selector returned wrong pool count");
    std::sort(result.begin(), result.end());
    require(std::adjacent_find(result.begin(), result.end()) == result.end(),
            "selector returned duplicate pools");
    for (unsigned i = 0; i < 3; ++i) {
        const int expected = i < sequence_length % pool ? int(count * pool + i) : -1;
        require(selected[topk + i] == expected, "selector tail or tail mask mismatch");
    }
    // Tied scores may legitimately choose different equal-valued pool IDs.
    // Independently verify that every chosen score is at least the kth score.
    if (budget) {
        std::vector<float> ranked(scores.begin(), scores.begin() + count);
        std::nth_element(ranked.begin(), ranked.begin() + budget - 1, ranked.end(),
                         std::greater<float>());
        const float threshold = ranked[budget - 1];
        for (unsigned p : result)
            require(scores[p] >= threshold, "selector omitted a strictly better pool");
    }
    if (zero_keys)
        for (unsigned p : result) require(scores[p] == 0.0f, "zero-key score is nonzero");
    return result;
}

static void run(unsigned rows, unsigned history, bool signed_weights, bool zero_keys,
                bool pool32) {
    constexpr unsigned heads = 32, dim = 128, pool = 4, block = 16;
    constexpr unsigned topk = 2048, selected_width = topk + pool - 1;
    require(history >= rows && rows > 0, "invalid benchmark shape");
    const unsigned start = history - rows;
    const unsigned stride = (history + pool - 1) / pool;
    const unsigned pages = (history + block - 1) / block;
    // Padding between physical cache blocks catches flattened-cache assumptions.
    constexpr unsigned page_elements = (block / pool) * dim + 64;
    constexpr unsigned long long page_bytes = page_elements * sizeof(__nv_bfloat16);
    const size_t nq = size_t(rows) * heads * dim;
    const size_t nw = size_t(rows) * heads;
    const size_t nk = size_t(pages) * page_elements;
    const size_t ns = size_t(rows) * stride;
    const size_t ni = size_t(rows) * selected_width;
    const size_t device_bytes = (nq + nw + nk) * 2 + pages * sizeof(unsigned)
        + 2 * ns * sizeof(float) + 2 * ni * sizeof(int);
    // Score/output validation copies one row at a time, bounding host memory.
    const size_t host_bytes_estimate = (nq + nw + nk) * 2 + pages * sizeof(unsigned)
        + 4 * stride * sizeof(float) + 4 * selected_width * sizeof(int);
    require(device_bytes + host_bytes_estimate < 150ULL * 1024 * 1024,
            "benchmark exceeds its combined host/device memory bound");
    std::mt19937 rng(5173 + rows + history + signed_weights);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> query(nq), weights(nw), keys(nk);
    constexpr float scales[] = {0.03125f, 0.25f, 1.0f, 4.0f};
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned h = 0; h < heads; ++h) {
            for (unsigned d = 0; d < dim; ++d)
                query[(size_t(r) * heads + h) * dim + d] =
                    __float2bfloat16(random(rng) * scales[(r + h) % 4]);
            const float weight = random(rng);
            weights[size_t(r) * heads + h] =
                __float2bfloat16(signed_weights ? weight : std::abs(weight));
        }
    }
    for (auto& value : keys) value = __float2bfloat16(zero_keys ? 0.0f : random(rng));
    std::vector<unsigned> table(pages);
    for (unsigned p = 0; p < pages; ++p) table[p] = p;
    std::shuffle(table.begin(), table.end(), rng);
    Buffer<__nv_bfloat16> dq(nq), dw(nw), dk(nk);
    Buffer<unsigned> dt(pages);
    Buffer<float> da(ns), db(ns);
    Buffer<int> ia(ni), ib(ni);
    dq.copy(query); dw.copy(weights); dk.copy(keys); dt.copy(table);
    CUDA_CHECK(cudaMemset(da.ptr, 0xff, ns * sizeof(float)));
    CUDA_CHECK(cudaMemset(db.ptr, 0xff, ns * sizeof(float)));
    const auto launch = [&](bool candidate) {
        if (candidate && pool32) {
            glm_index_logits_bf16_wmma_row8_pool32<<<dim3((stride + 31) / 32, (rows + 7) / 8), 256>>>(
                dq.ptr, dw.ptr, dk.ptr, db.ptr, dt.ptr, rows, start, stride,
                heads, dim, pool, block, page_bytes);
        } else if (candidate) {
            glm_index_logits_bf16_wmma_row8<<<dim3((stride + 15) / 16, (rows + 7) / 8), 256>>>(
                dq.ptr, dw.ptr, dk.ptr, db.ptr, dt.ptr, rows, start, stride,
                heads, dim, pool, block, page_bytes);
        } else {
            glm_index_logits_bf16_row8<<<dim3((stride + 7) / 8, (rows + 7) / 8), 256,
                                        8 * dim * sizeof(__nv_bfloat16)>>>(
                dq.ptr, dw.ptr, dk.ptr, da.ptr, dt.ptr, rows, start, stride,
                heads, dim, pool, block, page_bytes);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    launch(false); launch(true);
    glm_index_topk_expand<<<rows, 256, 4 * sizeof(unsigned)>>>(
        da.ptr, ia.ptr, rows, start, stride, topk, pool, selected_width);
    CUDA_CHECK(cudaGetLastError());
    glm_index_topk_expand<<<rows, 256, 4 * sizeof(unsigned)>>>(
        db.ptr, ib.ptr, rows, start, stride, topk, pool, selected_width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> a(stride), b(stride);
    std::vector<int> sa(selected_width), sb(selected_width);
    double square_error = 0.0, square_reference = 0.0, max_abs = 0.0;
    double max_relative = 0.0, oracle_max_abs = 0.0;
    size_t finite_scores = 0, pool_total = 0, pool_overlap = 0, changed_rows = 0;
    for (unsigned r = 0; r < rows; ++r) {
        CUDA_CHECK(cudaMemcpy(a.data(), da.ptr + size_t(r) * stride,
                              stride * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), db.ptr + size_t(r) * stride,
                              stride * sizeof(float), cudaMemcpyDeviceToHost));
        const unsigned visible = (start + r + 1) / pool;
        for (unsigned p = 0; p < stride; ++p) {
            if (p >= visible) {
                require(a[p] == -INFINITY && b[p] == -INFINITY, "causal mask mismatch");
                continue;
            }
            require(std::isfinite(a[p]) && std::isfinite(b[p]), "nonfinite visible score");
            const double error = std::abs(double(a[p]) - b[p]);
            max_abs = std::max(max_abs, error);
            max_relative = std::max(max_relative, error / std::max(1.0e-6, std::abs(double(a[p]))));
            square_error += error * error;
            square_reference += double(a[p]) * a[p];
            ++finite_scores;
            require(error <= 1.0e-5 + 1.0e-4 * std::abs(double(a[p])), "substantive score error");
            if (rows <= 7) {
                const unsigned token = p * pool;
                const size_t key_offset = size_t(table[token / block]) * page_elements
                    + ((token % block) / pool) * dim;
                double expected = 0.0;
                for (unsigned h = 0; h < heads; ++h) {
                    double dot = 0.0;
                    for (unsigned d = 0; d < dim; ++d)
                        dot += double(bf(query[(size_t(r) * heads + h) * dim + d]))
                            * bf(keys[key_offset + d]);
                    expected += bf(weights[size_t(r) * heads + h]) * std::max(dot, 0.0);
                }
                expected /= std::sqrt(double(dim * heads));
                const double oracle_error = std::abs(double(b[p]) - expected);
                oracle_max_abs = std::max(oracle_max_abs, oracle_error);
                require(oracle_error <= 1.0e-5 + 1.0e-4 * std::abs(expected),
                        "independent double score oracle failed");
            }
        }
        CUDA_CHECK(cudaMemcpy(sa.data(), ia.ptr + size_t(r) * selected_width,
                              selected_width * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sb.data(), ib.ptr + size_t(r) * selected_width,
                              selected_width * sizeof(int), cudaMemcpyDeviceToHost));
        const auto pa = selected_pools(sa, a, start + r + 1, zero_keys);
        const auto pb = selected_pools(sb, b, start + r + 1, zero_keys);
        size_t overlap = 0;
        for (unsigned p : pa) overlap += std::binary_search(pb.begin(), pb.end(), p);
        pool_overlap += overlap;
        pool_total += pa.size();
        changed_rows += overlap != pa.size();
    }
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> times[2];
    for (unsigned rep = 0; rep < 5; ++rep) {
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned mode = (rep + j) % 2;
            CUDA_CHECK(cudaEventRecord(begin));
            launch(mode != 0);
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            times[mode].push_back(ms);
        }
    }
    for (auto& samples : times) std::sort(samples.begin(), samples.end());
    std::printf("{\"rows\":%u,\"history\":%u,\"pool_tile\":%u,\"signed_weights\":%s,\"zero_keys\":%s,"
        "\"device_mib\":%.3f,\"max_abs\":%.9g,\"max_relative\":%.9g,\"rms\":%.9g,"
        "\"relative_rms\":%.9g,\"oracle_max_abs\":%.9g,\"top512_changed_rows\":%zu,"
        "\"top512_missing_pools\":%zu,\"top512_overlap\":%.9g,"
        "\"baseline_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.3f}\n",
        rows, history, pool32 ? 32u : 16u, signed_weights ? "true" : "false", zero_keys ? "true" : "false",
        double(device_bytes) / (1024 * 1024), max_abs, max_relative,
        std::sqrt(square_error / std::max(size_t(1), finite_scores)),
        std::sqrt(square_error / std::max(1.0e-30, square_reference)), oracle_max_abs,
        changed_rows, pool_total - pool_overlap, pool_total ? double(pool_overlap) / pool_total : 1.0,
        times[0][2], times[1][2], times[0][2] / times[1][2]);
    std::fflush(stdout);
    CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    require(prop.major == 12 && prop.minor == 1, "This bounded gate requires GB10/sm121");
    for (bool pool32 : {false, true}) {
        run(7, 7, false, false, pool32);
        run(7, 83, true, false, pool32);
        run(7, 4099, true, false, pool32);
        run(7, 4099, false, true, pool32);
        run(128, 4096, false, false, pool32);
        run(128, 32768, true, false, pool32);
        run(1024, 4096, true, false, pool32);
        run(1024, 32768, false, false, pool32);
    }
    return 0;
}
