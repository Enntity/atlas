// SPDX-License-Identifier: AGPL-3.0-only

// Small-allocation graph replay correctness gate, no checkpoint required.
// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_indexer_dynamic.cu -o /tmp/bench-glm-dynamic
// GPU allocation plus peak host validation storage stays below 100 MiB.
// Alternatives: -DGLM_BENCH_CAPACITY=16384 -DGLM_BENCH_HEADS=2
// All-zero index keys exercise tied top-K sets: -DGLM_BENCH_TIED_KEYS=1
#ifndef GLM_BENCH_CAPACITY
#define GLM_BENCH_CAPACITY 16400
#endif
#ifndef GLM_BENCH_HEADS
#define GLM_BENCH_HEADS 32
#endif
#ifndef GLM_BENCH_TIED_KEYS
#define GLM_BENCH_TIED_KEYS 0
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include "../../kernels/gb10/common/glm_indexer.cu"
#define HDIM 512
#include "../../kernels/gb10/common/paged_decode_attn.cu"

#define CHECK(call) do { const cudaError_t err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* text) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", text); std::exit(2); }
}
static size_t allocated_bytes = 0;
template <typename T> struct Buffer {
    static constexpr size_t guard = 128 / sizeof(T);
    T* allocation;
    T* ptr;
    size_t count;
    explicit Buffer(size_t count_) : count(count_) {
        const size_t bytes = (count + 2 * guard) * sizeof(T);
        allocated_bytes += bytes;
        require(allocated_bytes < 80ULL * 1024 * 1024, "device allocation budget exceeded");
        CHECK(cudaMalloc(&allocation, bytes));
        CHECK(cudaMemset(allocation, 0xa5, bytes));
        ptr = allocation + guard;
    }
    ~Buffer() { cudaFree(allocation); }
    void guards() const {
        unsigned char before[128], after[128];
        CHECK(cudaMemcpy(before, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(after, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i)
            require(before[i] == 0xa5 && after[i] == 0xa5, "poisoned allocation guard changed");
    }
    std::vector<T> read(size_t offset, size_t size) const {
        require(offset + size <= count, "host validation read exceeds allocation");
        std::vector<T> result(size);
        CHECK(cudaMemcpy(result.data(), ptr + offset, size * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
};

__global__ void initialize_bf16(__nv_bfloat16* dst, size_t count, unsigned seed) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned value = unsigned(i) ^ seed;
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu; value ^= value >> 16;
    dst[i] = __float2bfloat16((float(value & 65535) / 65535.0f - 0.5f) * 0.25f);
}
static void initialize(Buffer<__nv_bfloat16>& buffer, unsigned seed, cudaStream_t stream) {
    initialize_bf16<<<(buffer.count + 255) / 256, 256, 0, stream>>>(buffer.ptr, buffer.count, seed);
    CHECK(cudaGetLastError());
}
static void equal_bytes(const std::vector<__nv_bfloat16>& a,
                        const std::vector<__nv_bfloat16>& b, const char* message) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0, message);
}
static bool sentinel(const std::vector<__nv_bfloat16>& values) {
    for (const auto value : values) {
        unsigned short bits;
        std::memcpy(&bits, &value, 2);
        if (bits != 0x5a5a) return false;
    }
    return true;
}
static std::vector<unsigned> selected_set(const std::vector<int>& ids,
                                        const std::vector<float>& scores, unsigned length) {
    std::vector<unsigned> pools;
    for (unsigned i = 0; i < 2048; i += 4) {
        require(ids[i] >= 0 && ids[i] % 4 == 0, "invalid expanded pool");
        const unsigned p = unsigned(ids[i]) / 4;
        require(p < length / 4, "selected future pool");
        for (unsigned j = 0; j < 4; ++j)
            require(ids[i + j] == int(p * 4 + j), "noncontiguous pool expansion");
        pools.push_back(p);
    }
    std::sort(pools.begin(), pools.end());
    require(std::adjacent_find(pools.begin(), pools.end()) == pools.end(), "duplicate selected pool");
    for (unsigned i = 0; i < 3; ++i)
        require(ids[2048 + i] == (i < length % 4 ? int(length / 4 * 4 + i) : -1),
                "stale or incorrect unfinished tail");
    std::vector<float> ranked(scores.begin(), scores.begin() + length / 4);
    std::nth_element(ranked.begin(), ranked.begin() + 511, ranked.end(), std::greater<float>());
    for (unsigned p : pools) require(scores[p] >= ranked[511], "selected below top-512 threshold");
    for (unsigned p = 0; p < length / 4; ++p)
        if (scores[p] > ranked[511])
            require(std::binary_search(pools.begin(), pools.end(), p), "omitted strictly better pool");
    return pools;
}

int main() {
    constexpr unsigned rows = 3, capacity = GLM_BENCH_CAPACITY, block = 16, pages = capacity / block;
    constexpr unsigned stride = capacity / 4, heads = GLM_BENCH_HEADS, dim = 512, width = 2051;
    static_assert(capacity >= 4096 && capacity <= 16400 && capacity % block == 0,
                  "capacity must contain complete pages and preserve the small-allocation gate");
    static_assert(heads > 0 && heads <= 64, "attention heads must be in [1,64]");
    static_assert(pages % 37 != 0, "page shuffle requires a multiplier coprime to page count");
    constexpr unsigned pool_page = (block / 4) * 128, tail_page = block * 128 * 2;
    constexpr unsigned long long pool_bytes = pool_page * 2, tail_bytes = tail_page * 2;
    constexpr float scale = 1.0f / 16.0f;
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffer<__nv_bfloat16> iq(rows * 32 * 128), iw(rows * 32), aq(rows * heads * dim);
    Buffer<__nv_bfloat16> kc(size_t(pages) * block * dim), vc(size_t(pages) * block * dim);
    Buffer<__nv_bfloat16> old_pool(pages * pool_page), new_pool(pages * pool_page);
    Buffer<__nv_bfloat16> old_tail(pages * tail_page), new_tail(pages * tail_page);
    Buffer<__nv_bfloat16> raw_keys(rows * 128), raw_gates(rows * 128), ape(4 * 128);
    Buffer<unsigned> lengths(rows), table(rows * pages), dense_len(rows);
    Buffer<long long> slots(rows);
    Buffer<float> old_scores(rows * stride), new_scores(rows * stride);
    Buffer<int> old_ids(rows * width), new_ids(rows * width);
    Buffer<__nv_bfloat16> old_output(rows * heads * dim), new_output(rows * heads * dim);
    Buffer<__nv_bfloat16> dense_probe(rows * heads * dim), sparse_probe(rows * heads * dim);
    Buffer<__nv_bfloat16> same_ids_output(rows * heads * dim);
    initialize(iq, 7, stream); initialize(iw, 17, stream); initialize(aq, 27, stream);
    initialize(kc, 37, stream); initialize(vc, 47, stream);
    initialize(old_pool, 57, stream); initialize(new_pool, 57, stream);
    initialize(old_tail, 67, stream); initialize(new_tail, 67, stream);
    initialize(raw_keys, 77, stream); initialize(raw_gates, 87, stream); initialize(ape, 97, stream);
    if (GLM_BENCH_TIED_KEYS) {
        // Zero raw and historical keys as well, so subsequent captured pool
        // finalization retains exactly tied zero scores at every replay.
        for (auto* buffer : {&old_pool, &new_pool, &old_tail, &new_tail, &raw_keys})
            CHECK(cudaMemsetAsync(buffer->ptr, 0, buffer->count * 2, stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    const auto maintain = [&](unsigned row, bool candidate) {
        auto* tail = candidate ? new_tail.ptr : old_tail.ptr;
        auto* pool = candidate ? new_pool.ptr : old_pool.ptr;
        glm_index_tail_write_bf16<<<1, 128, 0, stream>>>(raw_keys.ptr + row * 128,
            raw_gates.ptr + row * 128, tail, slots.ptr + row, 1, block, 4, 128, tail_bytes);
        glm_index_kpool_finalize_bf16<<<1, 128, 0, stream>>>(tail, ape.ptr, pool,
            slots.ptr + row, 1, block, 4, 128, tail_bytes, pool_bytes);
    };
    const auto dense = [&](unsigned row, const unsigned* len, __nv_bfloat16* out) {
        paged_decode_attn<<<dim3(heads, 1), 256, 0, stream>>>(aq.ptr + row * heads * dim,
            kc.ptr, vc.ptr, out, reinterpret_cast<const int*>(table.ptr + row * pages),
            reinterpret_cast<const int*>(len), pages, heads, 1, dim, block, scale, heads * dim, 0);
    };
    const auto sparse = [&](unsigned row, const int* ids, __nv_bfloat16* out, bool guarded) {
        if (guarded) {
            glm_sparse_mla_prefill_bf16_dynamic<<<dim3(heads, 1), 256, 19 * sizeof(float), stream>>>(
                aq.ptr + row * heads * dim, kc.ptr, vc.ptr, ids, out, table.ptr + row * pages,
                1, heads, dim, width, block, scale, lengths.ptr + row, 2048);
        } else {
            glm_sparse_mla_prefill_bf16<<<dim3(heads, 1), 256, 19 * sizeof(float), stream>>>(
                aq.ptr + row * heads * dim, kc.ptr, vc.ptr, ids, out, table.ptr + row * pages,
                1, heads, dim, width, block, scale);
        }
    };
    const auto candidate = [&] {
        for (unsigned r = 0; r < rows; ++r) {
            maintain(r, true);
            glm_index_logits_bf16_dynamic<<<dim3((stride + 7) / 8, 1), 256, 0, stream>>>(
                iq.ptr + r * 32 * 128, iw.ptr + r * 32, new_pool.ptr, new_scores.ptr + r * stride,
                table.ptr + r * pages, 1, lengths.ptr + r, stride, 32, 128, 4, block, pool_bytes, 2048);
            glm_index_topk_expand_dynamic<<<1, 256, 4 * sizeof(unsigned), stream>>>(
                new_scores.ptr + r * stride, new_ids.ptr + r * width, 1, lengths.ptr + r,
                stride, 2048, 4, width, dense_len.ptr + r);
            dense(r, dense_len.ptr + r, new_output.ptr + r * heads * dim);
            sparse(r, new_ids.ptr + r * width, new_output.ptr + r * heads * dim, true);
            // Dedicated sentinels show that inactive nodes do not write.
            dense(r, dense_len.ptr + r, dense_probe.ptr + r * heads * dim);
            sparse(r, new_ids.ptr + r * width, sparse_probe.ptr + r * heads * dim, true);
        }
    };
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    candidate();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    const unsigned cases[] = {1, 3, 4, 15, 16, 2047, 2048, 2049, 2050, 2051, 2052,
                              4096, 16384, capacity - 1, capacity, 3, 2051, 0,
                              capacity + 1, 0xffffffffu};
    constexpr unsigned case_count = sizeof(cases) / sizeof(cases[0]);
    unsigned tie_variants = 0;
    for (unsigned epoch = 0; epoch < case_count; ++epoch) {
        const std::vector<unsigned> lens = {
            cases[epoch], cases[(epoch + 6) % case_count], cases[(epoch + 10) % case_count]};
        std::vector<unsigned> tables(rows * pages, 0xffffffffu);
        std::vector<long long> slot_host(rows, -1);
        for (unsigned r = 0; r < rows; ++r) {
            if (lens[r] == 0 || lens[r] > capacity) continue;
            for (unsigned p = 0; p < (lens[r] + block - 1) / block; ++p)
                tables[r * pages + p] = (p * 37 + r * 131 + epoch * 53) % pages;
            const unsigned pos = lens[r] - 1;
            slot_host[r] = static_cast<long long>(tables[r * pages + pos / block]) * block + pos % block;
        }
        CHECK(cudaMemcpyAsync(lengths.ptr, lens.data(), rows * 4, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(table.ptr, tables.data(), tables.size() * 4, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(slots.ptr, slot_host.data(), rows * 8, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemsetAsync(new_scores.ptr, 0xff, new_scores.count * 4, stream));
        CHECK(cudaMemsetAsync(new_ids.ptr, 0x7f, new_ids.count * 4, stream));
        for (auto* out : {&new_output, &dense_probe, &sparse_probe})
            CHECK(cudaMemsetAsync(out->ptr, 0x5a, out->count * 2, stream));
        CHECK(cudaMemsetAsync(dense_len.ptr, 0xff, rows * 4, stream));
        CHECK(cudaGraphLaunch(executable, stream));
        for (unsigned r = 0; r < rows; ++r) {
            const unsigned length = lens[r];
            if (length == 0 || length > capacity) continue;
            maintain(r, false);
            if (length <= 2048) {
                dense(r, lengths.ptr + r, old_output.ptr + r * heads * dim);
            } else {
                glm_index_logits_bf16<<<dim3((stride + 7) / 8, 1), 256, 0, stream>>>(
                    iq.ptr + r * 32 * 128, iw.ptr + r * 32, old_pool.ptr, old_scores.ptr + r * stride,
                    table.ptr + r * pages, 1, length - 1, stride, 32, 128, 4, block, pool_bytes);
                glm_index_topk_expand<<<1, 256, 4 * sizeof(unsigned), stream>>>(
                    old_scores.ptr + r * stride, old_ids.ptr + r * width, 1, length - 1,
                    stride, 2048, 4, width);
                sparse(r, old_ids.ptr + r * width, old_output.ptr + r * heads * dim, false);
                sparse(r, new_ids.ptr + r * width, same_ids_output.ptr + r * heads * dim, false);
            }
        }
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));
        const auto dense_host = dense_len.read(0, rows);
        for (unsigned r = 0; r < rows; ++r) {
            const unsigned length = lens[r];
            const bool valid = length > 0 && length <= capacity;
            const bool short_row = valid && length <= 2048;
            require(dense_host[r] == (short_row ? length : 0), "stale dense length after replay");
            const auto scores = new_scores.read(r * stride, stride);
            const auto ids = new_ids.read(r * width, width);
            const auto result = new_output.read(r * heads * dim, heads * dim);
            const auto dp = dense_probe.read(r * heads * dim, heads * dim);
            const auto sp = sparse_probe.read(r * heads * dim, heads * dim);
            if (!valid || short_row) {
                for (float score : scores) require(score == -INFINITY, "inactive score was not masked");
                for (int id : ids) require(id == -1, "inactive selector retained stale IDs");
            }
            if (!short_row) require(sentinel(dp), "inactive dense kernel wrote output");
            if (length <= 2048) require(sentinel(sp), "inactive sparse kernel wrote output");
            if (!valid) {
                if (length == 0) require(sentinel(result), "zero-length chain wrote output");
                else for (auto x : result) require(__bfloat162float(x) == 0.0f, "invalid long row read stale IDs");
                continue;
            }
            const auto oracle = old_output.read(r * heads * dim, heads * dim);
            if (short_row) {
                equal_bytes(result, oracle, "dense arithmetic changed under capture");
                equal_bytes(result, dp, "dense branch did not own final output");
            } else {
                const auto eager_scores = old_scores.read(r * stride, stride);
                require(std::memcmp(scores.data(), eager_scores.data(), stride * 4) == 0,
                        "captured scalar score arithmetic differs from eager");
                const auto selected = selected_set(ids, scores, length);
                const auto eager_selected = selected_set(old_ids.read(r * width, width), scores, length);
                if (selected != eager_selected) ++tie_variants;
                equal_bytes(result, same_ids_output.read(r * heads * dim, heads * dim),
                            "sparse arithmetic differs for identical IDs");
                equal_bytes(result, sp, "sparse branch did not own final output");
                // Atomic top-K order is intentionally unspecified. Compare the
                // full eager chain numerically only when its selected set agrees.
                if (selected == eager_selected) for (unsigned i = 0; i < result.size(); ++i) {
                    const float a = __bfloat162float(result[i]), b = __bfloat162float(oracle[i]);
                    require(std::isfinite(a) && std::isfinite(b) &&
                            std::abs(a - b) <= 0.0002f + 0.02f * std::abs(b),
                            "eager/graph attention differs beyond ordering tolerance");
                }
            }
            const unsigned page = unsigned(slot_host[r] / block);
            equal_bytes(new_tail.read(page * tail_page, tail_page), old_tail.read(page * tail_page, tail_page),
                        "graph tail maintenance differs from eager");
            equal_bytes(new_pool.read(page * pool_page, pool_page), old_pool.read(page * pool_page, pool_page),
                        "graph pool finalization differs from eager");
        }
        new_scores.guards(); old_scores.guards(); new_ids.guards(); old_ids.guards();
        new_output.guards(); old_output.guards(); dense_probe.guards(); sparse_probe.guards();
        new_pool.guards(); old_pool.guards(); new_tail.guards(); old_tail.guards();
        std::printf("PASS replay=%u lengths=[%u,%u,%u]\n", epoch, lens[0], lens[1], lens[2]);
    }
    std::printf("PASS fixed graph, changed device lengths/tables/slots; capacity=%u heads=%u tied_keys=%d "
                "device_bytes=%zu tie_set_variants=%u\n",
                capacity, heads, GLM_BENCH_TIED_KEYS, allocated_bytes, tie_variants);
    CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
}
