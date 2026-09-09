// SPDX-License-Identifier: AGPL-3.0-only
// CUDA 13: nvcc -std=c++17 -O3 --fmad=false -Xcompiler=-ffp-contract=off -arch=sm_121a
//   scripts/dev/bench_glm_pair_draft_head.cu -o bench-glm-pair-draft-head
// Correctness/memcheck: compute-sanitizer --tool memcheck --error-exitcode 99
//   ./bench-glm-pair-draft-head --repetitions 0
// Warm timing (both orders): ./bench-glm-pair-draft-head --repetitions 20
// Projection-only synthetic full TP2 shard: no Model, E1, sampling, or NCCL claim.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "../../kernels/gb10/common/dense_gemv_bf16.cu"
#include "../../kernels/gb10/common/dense_gemv_bf16_batch2.cu"

static_assert(CUDART_VERSION >= 13000 && CUDART_VERSION < 14000, "CUDA 13 required");
static constexpr unsigned rows = 2, n = 77428, k = 4096, stride = 154856;
static constexpr size_t cap = 768ull * 1024 * 1024;
static constexpr uint16_t poison = 0xa5a5;
static size_t live_bytes = 0, peak_bytes = 0;
#define CHECK(call) do { const auto err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(2); }
}
static size_t mul(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b, "size overflow");
    return a * b;
}
struct Buffer {
    static constexpr size_t guard = 64; // 128 bytes, preserves vector alignment.
    uint16_t *allocation, *ptr;
    size_t count, bytes;
    explicit Buffer(size_t count_) : count(count_) {
        require(count <= std::numeric_limits<size_t>::max() - 2 * guard, "guard overflow");
        bytes = mul(count + 2 * guard, sizeof(uint16_t));
        require(live_bytes <= cap && bytes <= cap - live_bytes, "768MiB allocation cap");
        CHECK(cudaMalloc(&allocation, bytes));
        live_bytes += bytes;
        peak_bytes = std::max(peak_bytes, live_bytes);
        ptr = allocation + guard;
        CHECK(cudaMemset(allocation, 0xa5, bytes));
    }
    ~Buffer() { CHECK(cudaFree(allocation)); live_bytes -= bytes; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    __nv_bfloat16* bf16() const { return reinterpret_cast<__nv_bfloat16*>(ptr); }
    void upload(const std::vector<uint16_t>& values) const {
        require(values.size() == count, "upload extent");
        CHECK(cudaMemcpy(ptr, values.data(), mul(count, 2), cudaMemcpyHostToDevice));
    }
    void clear() const { CHECK(cudaMemset(ptr, 0xa5, mul(count, 2))); }
    std::vector<uint16_t> read() const {
        std::vector<uint16_t> result(count);
        CHECK(cudaMemcpy(result.data(), ptr, mul(count, 2), cudaMemcpyDeviceToHost));
        return result;
    }
    void guards() const {
        std::array<uint16_t, guard> before{}, after{};
        CHECK(cudaMemcpy(before.data(), allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(after.data(), ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < guard; ++i)
            require(before[i] == poison && after[i] == poison, "allocation canary changed");
    }
};
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
static uint16_t bits(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float value(uint16_t x) {
    const uint32_t u = uint32_t(x) << 16;
    float result; std::memcpy(&result, &u, sizeof(result)); return result;
}
static uint16_t weight(unsigned output, unsigned column) {
    const uint32_t h = mix(output * 0x9e3779b9u ^ column * 0x85ebca6bu ^ 0x537129adu);
    return bits(std::ldexp(float(int(h & 2047) - 1023), -12 - int((h >> 11) % 5)));
}
static std::vector<uint16_t> inputs(unsigned profile) {
    std::vector<uint16_t> result(rows * k);
    for (unsigned owner = 0; owner < rows; ++owner) {
        for (unsigned column = 0; column < k; ++column) {
            const uint32_t h = mix(column ^ (0x58d7913bu * (owner + 1)));
            float x = std::ldexp(float(int(h & 2047) - 1023), -8 - int((h >> 11) % 7));
            if (profile == 1) // Distinct, cancellation-heavy rows with mixed magnitudes.
                x = (column & 1 ? -1.0f : 1.0f) *
                    std::ldexp(float(129 + int(h % 127)), -7 - int((column + owner) % 6));
            if (profile == 2) // Vector/warp boundary impulses, different for each owner.
                x = column == (owner ? 4095u : 7u) ? (owner ? -2.0f : 1.0f) : 0.0f;
            result[owner * k + column] = bits(x);
        }
    }
    require(!std::equal(result.begin(), result.begin() + k, result.begin() + k),
            "input owners must differ");
    return result;
}
// Independent host emulation of the scalar float instruction/reduction order.
// Volatile intermediates prohibit host FMA and force each multiply/add to FP32.
static float add(float a, float b) { volatile float result = a + b; return result; }
static float product(float a, float b) { volatile float result = a * b; return result; }
static uint16_t oracle(const uint16_t* input, unsigned output) {
    std::array<float, 64> sums{};
    for (unsigned lane = 0; lane < 64; ++lane)
        for (unsigned vector = lane; vector < k / 8; vector += 64)
            for (unsigned j = 0; j < 8; ++j) {
                const unsigned column = vector * 8 + j;
                sums[lane] = add(sums[lane], product(value(input[column]), value(weight(output, column))));
            }
    for (unsigned offset = 16; offset; offset >>= 1) {
        const auto previous = sums;
        for (unsigned lane = 0; lane < 64; ++lane) {
            const unsigned source = lane % 32 + offset < 32 ? lane + offset : lane;
            sums[lane] = add(previous[lane], previous[source]);
        }
    }
    return bits(add(sums[0], sums[32]));
}
static void launch(bool paired, const Buffer& input, const Buffer& weights,
                   const Buffer& output, unsigned rank_offset) {
    require(rank_offset == 0 || rank_offset == n, "rank offset");
    if (paired) {
        dense_gemv_bf16_batch2<<<n / 4, 256>>>(input.bf16(), weights.bf16(),
                                             output.bf16() + rank_offset, n, k, stride);
        CHECK(cudaGetLastError());
    } else {
        for (unsigned owner = 0; owner < rows; ++owner) {
            dense_gemv_bf16<<<n / 4, 256>>>(input.bf16() + owner * k, weights.bf16(),
                                          output.bf16() + owner * stride + rank_offset, n, k);
            CHECK(cudaGetLastError());
        }
    }
}
static std::vector<uint16_t> valid(const Buffer& output, unsigned rank_offset) {
    const auto all = output.read();
    std::vector<uint16_t> result(rows * n);
    for (unsigned owner = 0; owner < rows; ++owner)
        for (unsigned column = 0; column < stride; ++column) {
            const auto v = all[owner * stride + column];
            if (column >= rank_offset && column < rank_offset + n) {
                require((v & 0x7f80) != 0x7f80, "nonfinite logit");
                result[owner * n + column - rank_offset] = v;
            } else require(v == poison, "unused rank half/stride overwritten");
        }
    output.guards();
    return result;
}
static void check_oracle(const std::vector<uint16_t>& result, const std::vector<uint16_t>& input) {
    for (unsigned owner = 0; owner < rows; ++owner)
        for (unsigned sample = 0; sample < 18; ++sample) {
            const unsigned column = sample < 4 ? sample : sample < 8 ? n - (sample - 3)
                                                   : mix(0x3471adu + sample) % n;
            const auto expected = oracle(input.data() + owner * k, column);
            if (result[owner * n + column] != expected) {
                std::fprintf(stderr, "oracle owner=%u column=%u actual=%04x expected=%04x\n",
                             owner, column, unsigned(result[owner * n + column]), unsigned(expected));
                require(false, "sampled host FP32-order oracle");
            }
        }
}
static float timed(bool paired, unsigned repetitions, const Buffer& input,
                   const Buffer& weights, const Buffer& output) {
    for (unsigned warm = 0; warm < 3; ++warm) launch(paired, input, weights, output, 0);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start));
    for (unsigned i = 0; i < repetitions; ++i) launch(paired, input, weights, output, 0);
    CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, start, end));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
    require(std::isfinite(ms) && ms > 0, "invalid timing");
    return ms / float(repetitions);
}
static unsigned repetitions(int argc, char** argv) {
    require(argc == 3 && std::strcmp(argv[1], "--repetitions") == 0,
            "usage: --repetitions INTEGER[0,100]; 0 runs all correctness cases without timing");
    require(argv[2][0] != '\0', "empty repetitions");
    unsigned result = 0;
    for (const char* p = argv[2]; *p; ++p) {
        require(*p >= '0' && *p <= '9' && result <= 10, "invalid/overflow repetitions");
        result = result * 10 + unsigned(*p - '0');
        require(result <= 100, "repetitions exceed 100");
    }
    require(std::strlen(argv[2]) <= 3, "noncanonical repetitions length");
    return result;
}
int main(int argc, char** argv) {
    const unsigned repeat = repetitions(argc, argv);
    static_assert(n % 4 == 0 && k % 8 == 0 && stride == 2 * n, "fixed launch geometry");
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, 0));
    require(prop.major == 12 && prop.minor == 1, "requires SM121 GB10");
    CHECK(cudaSetDevice(0));
    Buffer weights(mul(n, k)), input(mul(rows, k)), scalar(mul(rows, stride)), paired(mul(rows, stride));
    std::printf("N=%u K=%u rows=2 stride=%u rank_offsets=0,%u weight_bytes=%zu "
                "guarded_cudaMalloc_bytes=%zu cap=%zu host_weight_chunk_bytes=%u\n",
                n, k, stride, n, mul(mul(n, k), 2), peak_bytes, cap, 64 * k * 2);
    // Entire independent matrix is resident, not aliased/recycled expert rows.
    // Bounded host upload staging; CUDA context/event overhead is outside cudaMalloc accounting.
    std::vector<uint16_t> chunk(64 * k);
    for (unsigned first = 0; first < n; first += 64) {
        const unsigned count = std::min(64u, n - first);
        for (unsigned row = 0; row < count; ++row)
            for (unsigned column = 0; column < k; ++column)
                chunk[row * k + column] = weight(first + row, column);
        CHECK(cudaMemcpy(weights.ptr + size_t(first) * k, chunk.data(),
                         mul(mul(count, k), 2), cudaMemcpyHostToDevice));
    }
    for (unsigned profile = 0; profile < 3; ++profile) {
        const auto original = inputs(profile);
        std::vector<uint16_t> canonical;
        for (unsigned reversed = 0; reversed < 2; ++reversed) {
            auto data = original;
            if (reversed) std::swap_ranges(data.begin(), data.begin() + k, data.begin() + k);
            input.upload(data);
            for (unsigned rank = 0; rank < 2; ++rank) {
                scalar.clear(); paired.clear();
                launch(false, input, weights, scalar, rank * n);
                launch(true, input, weights, paired, rank * n);
                CHECK(cudaDeviceSynchronize());
                const auto expected = valid(scalar, rank * n), actual = valid(paired, rank * n);
                require(actual == expected, "scalar vs batch2 BF16 logits not bit-identical");
                check_oracle(actual, data);
                if (!reversed && !rank) canonical = expected;
                for (unsigned owner = 0; owner < rows; ++owner)
                    for (unsigned column = 0; column < n; ++column)
                        require(actual[owner * n + column] == canonical[(owner ^ reversed) * n + column],
                                "owner reversal/rank placement changed logits");
                require(input.read() == data, "input modified");
                input.guards(); weights.guards();
                std::printf("PASS profile=%u reversed=%u rank=%u exact_logits=%u oracle_samples=36\n",
                            profile, reversed, rank, rows * n);
            }
        }
    }
    if (repeat) {
        const auto data = inputs(0); input.upload(data);
        for (unsigned order = 0; order < 2; ++order) {
            scalar.clear(); paired.clear();
            float times[2]{};
            for (unsigned step = 0; step < 2; ++step) {
                const unsigned mode = step ^ order;
                times[mode] = timed(mode != 0, repeat, input, weights, mode ? paired : scalar);
            }
            const auto reference = valid(scalar, 0), result = valid(paired, 0);
            require(reference == result, "timed output changed"); check_oracle(result, data);
            require(input.read() == data, "timing modified input");
            weights.guards(); input.guards();
            std::printf("TIMING order=%s repetitions=%u scalar_two_ms=%.6f batch2_ms=%.6f ratio=%.6f\n",
                        order ? "batch2-first" : "scalar-first", repeat, times[0], times[1], times[0] / times[1]);
        }
    }
    std::printf("PASS all_logits_exact guards_intact peak_cudaMalloc_bytes=%zu; projection only\n", peak_bytes);
}
