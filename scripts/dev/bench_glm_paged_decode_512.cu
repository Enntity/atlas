// SPDX-License-Identifier: AGPL-3.0-only

// Bounded GLM absorbed-MLA HDIM512 regression; no checkpoint/server required.
// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_paged_decode_512.cu -o /tmp/bench-glm-paged512
// Tests first: CPU double softmax oracle, scalar/batched exact equality,
// eager/captured exact equality, metadata refresh, row permutations and guards.
// Only the correct HDIM512 kernel is launched; never run HDIM576 on this shape.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
// Exact production module selected by the repaired GLM kernel binding.
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu"
#ifndef GLM_PAGED_QUERY_PADDING
#define GLM_PAGED_QUERY_PADDING 0
#endif

#define CHECK(call) do { const cudaError_t err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
static size_t allocated = 0;
template<class T> struct Buffer {
    static constexpr size_t guard = 128 / sizeof(T);
    T* allocation;
    T* ptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        allocated += (count + 2 * guard) * sizeof(T);
        require(allocated < 32ULL * 1024 * 1024, "device allocation budget exceeded");
        CHECK(cudaMalloc(&allocation, (count + 2 * guard) * sizeof(T)));
        CHECK(cudaMemset(allocation, 0xa5, (count + 2 * guard) * sizeof(T)));
        ptr = allocation + guard;
    }
    ~Buffer() { cudaFree(allocation); }
    void upload(const std::vector<T>& host, cudaStream_t stream) {
        require(host.size() == count, "upload size mismatch");
        CHECK(cudaMemcpyAsync(ptr, host.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> read() const {
        std::vector<T> host(count);
        CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    void guards() const {
        unsigned char before[128], after[128];
        CHECK(cudaMemcpy(before, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(after, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i)
            require(before[i] == 0xa5 && after[i] == 0xa5, "allocation guard overwritten");
    }
};
static __nv_bfloat16 bf16_bits(unsigned short bits) {
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, 2);
    return value;
}
static __nv_bfloat16 sample(size_t i, unsigned seed) {
    unsigned x = unsigned(i) ^ seed;
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return __float2bfloat16((float(x & 65535) / 65535.0f - 0.5f));
}
static void equal(const std::vector<__nv_bfloat16>& a,
                  const std::vector<__nv_bfloat16>& b, const char* message) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0, message);
}

int main() {
    constexpr unsigned max_rows = 4, heads = 32, dim = 512, block = 16;
    constexpr unsigned max_length = 2048, table_stride = max_length / block;
    constexpr unsigned physical_pages = 521; // prime; shuffle multiplier37 is coprime.
    constexpr unsigned row_elements = heads * dim;
    constexpr unsigned query_stride = row_elements + GLM_PAGED_QUERY_PADDING;
    static_assert(GLM_PAGED_QUERY_PADDING >= 0 && GLM_PAGED_QUERY_PADDING % 2 == 0,
                  "optional query padding must preserve packed BF16 alignment");
    constexpr float scale = 1.0f / 16.0f; // GLM scales by original QK dimension256.
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffer<__nv_bfloat16> q(max_rows * query_stride);
    Buffer<__nv_bfloat16> k(size_t(physical_pages) * block * dim), v(k.count);
    Buffer<__nv_bfloat16> eager(max_rows * row_elements), replay(eager.count), scalar(eager.count);
    Buffer<int> lengths(max_rows), tables(max_rows * table_stride);
    std::vector<__nv_bfloat16> kh(k.count), vh(v.count);
    for (size_t i = 0; i < k.count; ++i) {
        kh[i] = sample(i, 123); vh[i] = sample(i, 456);
    }
    k.upload(kh, stream); v.upload(vh, stream);
    CHECK(cudaStreamSynchronize(stream));
    std::printf("device_bytes=%zu host_plus_device_estimate<40MiB HDIM=%u heads=%u\n",
                allocated, dim, heads);

    const auto launch = [&](unsigned rows, __nv_bfloat16* out) {
        paged_decode_attn<<<dim3(heads, rows), 256, 0, stream>>>(q.ptr, k.ptr, v.ptr, out,
            tables.ptr, lengths.ptr, table_stride, heads, 1, dim, block, scale, query_stride, 0);
        CHECK(cudaGetLastError());
    };
    const int cases[4][4] = {{1, 15, 16, 17}, {2048, 2047, 511, 33},
                            {3, 4, 1023, 1024}, {0, 17, 0, 1}};
    const unsigned orders[4][4] = {{0, 1, 2, 3}, {3, 1, 0, 2},
                                  {2, 0, 3, 1}, {0, 1, 2, 3}};
    unsigned passed = 0;
    for (unsigned rows = 1; rows <= max_rows; ++rows) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(rows, replay.ptr);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (unsigned epoch = 0; epoch < 4; ++epoch) {
            // Fixed device pointers; contents and row-to-sequence mapping change on replay.
            std::vector<int> lens(max_rows, 0), bt(tables.count, -1);
            std::vector<__nv_bfloat16> qh(q.count, bf16_bits(0x7fc1)); // poison row padding
            for (unsigned row = 0; row < rows; ++row) {
                const unsigned identity = orders[epoch][row];
                lens[row] = cases[epoch][identity];
                for (unsigned p = 0; p < unsigned(lens[row] + block - 1) / block; ++p)
                    bt[row * table_stride + p] =
                        int((p * 37 + identity * 131 + epoch * 53) % physical_pages);
                if (lens[row] > 0)
                    for (unsigned d = 0; d < row_elements; ++d)
                        qh[row * query_stride + d] = sample(d + identity * row_elements, 789 + epoch);
            }
            q.upload(qh, stream); lengths.upload(lens, stream); tables.upload(bt, stream);
            for (auto* output : {&eager, &replay, &scalar})
                CHECK(cudaMemsetAsync(output->ptr, 0x5a, output->count * 2, stream));
            launch(rows, eager.ptr);
            CHECK(cudaGraphLaunch(executable, stream));
            for (unsigned row = 0; row < rows; ++row) {
                paged_decode_attn<<<dim3(heads, 1), 256, 0, stream>>>(
                    q.ptr + row * query_stride, k.ptr, v.ptr, scalar.ptr + row * row_elements,
                    tables.ptr + row * table_stride, lengths.ptr + row, table_stride, heads,
                    1, dim, block, scale, query_stride, 0);
                CHECK(cudaGetLastError());
            }
            CHECK(cudaStreamSynchronize(stream));
            const auto actual = eager.read();
            equal(actual, replay.read(), "captured/eager output mismatch");
            equal(actual, scalar.read(), "batched/row-wise output mismatch");
            for (size_t i = size_t(rows) * row_elements; i < actual.size(); ++i) {
                unsigned short bits; std::memcpy(&bits, &actual[i], 2);
                require(bits == 0x5a5a, "inactive output row overwritten");
            }
            double worst = 0.0;
            for (unsigned row = 0; row < rows; ++row) {
                const unsigned len = unsigned(lens[row]);
                if (len == 0) {
                    for (unsigned d = 0; d < row_elements; ++d) {
                        unsigned short bits;
                        std::memcpy(&bits, &actual[row * row_elements + d], 2);
                        require(bits == 0x5a5a, "zero-length row did not preserve output");
                    }
                    continue;
                }
                std::vector<size_t> offsets(len);
                for (unsigned t = 0; t < len; ++t)
                    offsets[t] = (size_t(bt[row * table_stride + t / block]) * block + t % block) * dim;
                for (unsigned head = 0; head < heads; ++head) {
                    const auto* query = qh.data() + row * query_stride + head * dim;
                    std::vector<double> scores(len);
                    double maximum = -1e100;
                    for (unsigned t = 0; t < len; ++t) {
                        double dot = 0.0;
                        for (unsigned d = 0; d < dim; ++d)
                            dot += double(__bfloat162float(query[d])) * __bfloat162float(kh[offsets[t] + d]);
                        scores[t] = dot * scale;
                        maximum = std::max(maximum, scores[t]);
                    }
                    double denominator = 0.0;
                    for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
                    std::vector<double> expected(dim, 0.0);
                    for (unsigned t = 0; t < len; ++t)
                        for (unsigned d = 0; d < dim; ++d)
                            expected[d] += scores[t] * __bfloat162float(vh[offsets[t] + d]);
                    for (unsigned d = 0; d < dim; ++d) {
                        const double reference = expected[d] / denominator;
                        const double got = __bfloat162float(actual[row * row_elements + head * dim + d]);
                        const double error = std::abs(got - reference);
                        // Covers BF16 final rounding plus small FP32 softmax/reduction error.
                        const double tolerance = 2e-5 + 0.004 * std::abs(reference);
                        if (!std::isfinite(got) || error > tolerance) {
                            std::fprintf(stderr, "rows=%u epoch=%u row=%u head=%u d=%u len=%u got=%.9g ref=%.9g err=%.9g tol=%.9g\n",
                                rows, epoch, row, head, d, len, got, reference, error, tolerance);
                            require(false, "CPU double attention oracle mismatch");
                        }
                        worst = std::max(worst, error);
                    }
                }
            }
            equal(qh, q.read(), "read-only query including poisoned padding modified");
            require(bt == tables.read() && lens == lengths.read(), "read-only metadata modified");
            q.guards(); k.guards(); v.guards(); eager.guards(); replay.guards(); scalar.guards();
            lengths.guards(); tables.guards();
            std::printf("PASS rows=%u epoch=%u lengths=[%d,%d,%d,%d] maxabs=%.9g eager=graph=scalar BITEXACT\n",
                        rows, epoch, lens[0], lens[1], lens[2], lens[3], worst);
            ++passed;
        }
        CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
    }
    equal(kh, k.read(), "read-only K cache modified");
    equal(vh, v.read(), "read-only V cache modified");
    CHECK(cudaStreamDestroy(stream));
    std::printf("PASS all %u HDIM512 cases; allocation guards intact\n", passed);
    return 0;
}
