// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -std=c++17 -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_w4a16_m10.cu -o bench-w4a16-m10
// Fixture-local specialization only. No model weights, collectives, or serving selection.
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
#include "../../kernels/gb10/common/w4a16_gemv.cu"

// Unactivated specialization of the unchanged production arithmetic template.
// Deliberately no launch_bounds or alternative reduction policy.
extern "C" __global__ void fixture_w4a16_gemv_exact10(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, float scale2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemv_batchm_impl<10>(A, B_packed, B_scale, scale2, C, M, N, K);
}

#define CHECK(call) do { const auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    std::exit(1); } } while (0)

static constexpr size_t limit = 16 * 1024 * 1024;
static size_t live_bytes = 0, peak_bytes = 0;
static void require(bool ok, const char* why, int code = 2) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(code); }
}
static size_t mul(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b,
            "size multiplication overflow", 4);
    return a * b;
}
template<class T> struct Buffer {
    static constexpr size_t guard = 128 / sizeof(T);
    T *allocation, *ptr;
    size_t count, bytes;
    explicit Buffer(size_t n) : count(n) {
        require(n <= std::numeric_limits<size_t>::max() - 2 * guard,
                "guard arithmetic overflow", 4);
        bytes = mul(n + 2 * guard, sizeof(T));
        require(live_bytes <= limit && bytes <= limit - live_bytes,
                "16MiB live allocation budget", 4);
        CHECK(cudaMalloc(&allocation, bytes));
        live_bytes += bytes; peak_bytes = std::max(peak_bytes, live_bytes);
        CHECK(cudaMemset(allocation, 0xa5, bytes));
        ptr = allocation + guard;
    }
    ~Buffer() { CHECK(cudaFree(allocation)); live_bytes -= bytes; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& src) const {
        require(src.size() == count, "upload extent");
        CHECK(cudaMemcpy(ptr, src.data(), mul(count, sizeof(T)), cudaMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> result(count);
        CHECK(cudaMemcpy(result.data(), ptr, mul(count, sizeof(T)), cudaMemcpyDeviceToHost));
        return result;
    }
    void guards() const {
        unsigned char before[128], after[128];
        CHECK(cudaMemcpy(before, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(after, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i)
            require(before[i] == 0xa5 && after[i] == 0xa5, "allocation canary", 3);
    }
};

struct Lcg {
    uint64_t state;
    uint32_t next() { state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return uint32_t(state >> 32); }
    float value() { return (float(next() & 65535) / 65535.0f - 0.5f) * 3.0f; }
};
static unsigned short bits(__nv_bfloat16 value) {
    unsigned short result; std::memcpy(&result, &value, 2); return result;
}
static __nv_bfloat16 from_bits(unsigned short value) {
    __nv_bfloat16 result; std::memcpy(&result, &value, 2); return result;
}
enum class Fault { None, Output, Guard, Unused, Budget };
struct Options { unsigned repetitions = 0; Fault fault = Fault::None; };
static Options options(int argc, char** argv) {
    Options out; bool seen_reps = false, seen_fault = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--repetitions")) {
            require(!seen_reps && ++i < argc, "invalid/repeated --repetitions", 64);
            seen_reps = true; const char* s = argv[i];
            require(*s != '\0', "empty repetitions", 64);
            for (; *s; ++s) {
                require(*s >= '0' && *s <= '9', "repetitions must be decimal0..100", 64);
                out.repetitions = out.repetitions * 10 + unsigned(*s - '0');
                require(out.repetitions <= 100, "repetitions must be <=100", 64);
            }
        } else if (!std::strcmp(argv[i], "--fault")) {
            require(!seen_fault && ++i < argc, "invalid/repeated --fault", 64);
            seen_fault = true;
            if (!std::strcmp(argv[i], "output")) out.fault = Fault::Output;
            else if (!std::strcmp(argv[i], "guard")) out.fault = Fault::Guard;
            else if (!std::strcmp(argv[i], "unused")) out.fault = Fault::Unused;
            else if (!std::strcmp(argv[i], "budget")) out.fault = Fault::Budget;
            else require(false, "unknown fault mode", 64);
        } else require(false, "expected --repetitions N or --fault output|guard|unused|budget", 64);
    }
    require(out.fault == Fault::None || out.repetitions == 0,
            "fault modes require repetitions0", 64);
    return out;
}
static void flip_byte(void* address) {
    unsigned char value;
    CHECK(cudaMemcpy(&value, address, 1, cudaMemcpyDeviceToHost));
    value ^= 1;
    CHECK(cudaMemcpy(address, &value, 1, cudaMemcpyHostToDevice));
}

static void run(unsigned n, unsigned k, unsigned profile, const Options& opt) {
    require(k % 16 == 0 && n > 0, "unsupported fixture geometry", 64);
    const unsigned seeds[] = {1, 99, 12345};
    Lcg rng{uint64_t(seeds[profile < 6 ? profile / 2 : 0]) * 0x9e3779b97f4a7c15ULL};
    const bool impulse = profile == 8;
    const float scale2 = profile == 6 ? 0.0f :
        (profile < 6 && profile % 2 == 0 ? 0.0123f : 1.0f);
    std::vector<__nv_bfloat16> input(mul(16, k), from_bits(0x7fc1));
    std::vector<unsigned char> weight(mul(n, k) / 2), scales(mul(n, k) / 16);
    for (unsigned row = 0; row < 10; ++row)
        for (unsigned d = 0; d < k; ++d)
            input[size_t(row) * k + d] = __float2bfloat16(
                impulse ? (d == row * 7 % k ? 1.0f : 0.0f) : rng.value());
    for (auto& w : weight) w = static_cast<unsigned char>(rng.next());
    if (impulse) {
        for (size_t i = 0; i < weight.size(); ++i) {
            const unsigned lo = 1 + unsigned(i % 7) + (i % 2 ? 8 : 0);
            const unsigned hi = 1 + unsigned((i + 3) % 7) + (i % 2 ? 0 : 8);
            weight[i] = static_cast<unsigned char>(lo | hi << 4);
        }
    }
    for (auto& s : scales) s = impulse ? 0x38 :
        (profile == 7 ? 1 + rng.next() % 7 : 0x30 + rng.next() % 24);
    Buffer<unsigned char> w(weight.size()), ws(scales.size());
    Buffer<__nv_bfloat16> a(input.size()), candidate(mul(16, n)), wide(mul(16, n)),
        pair(mul(16, n)), scalar(mul(16, n));
    a.upload(input); w.upload(weight); ws.upload(scales);
    const dim3 grid((n + 3) / 4);
    const auto launch = [&](unsigned arm) {
        if (arm == 0) {
            fixture_w4a16_gemv_exact10<<<grid, 256>>>(a.ptr, w.ptr, ws.ptr, scale2, candidate.ptr, 10, n, k);
            CHECK(cudaGetLastError());
        } else if (arm == 1) {
            w4a16_gemv_batch16<<<grid, 256>>>(a.ptr, w.ptr, ws.ptr, scale2, wide.ptr, 10, n, k);
            CHECK(cudaGetLastError());
        } else if (arm == 2) {
            for (unsigned segment = 0; segment < 2; ++segment) {
                w4a16_gemv_batch5<<<grid, 256>>>(a.ptr + size_t(segment * 5) * k,
                    w.ptr, ws.ptr, scale2, pair.ptr + size_t(segment * 5) * n, 5, n, k);
                CHECK(cudaGetLastError());
            }
        } else {
            for (unsigned row = 0; row < 10; ++row) {
                w4a16_gemv<<<grid, 256>>>(a.ptr + size_t(row) * k,
                    w.ptr, ws.ptr, scale2, scalar.ptr + size_t(row) * n, n, k);
                CHECK(cudaGetLastError());
            }
        }
    };
    const auto reset = [&] {
        for (auto* buf : {&candidate, &wide, &pair, &scalar})
            CHECK(cudaMemset(buf->ptr, 0xff, mul(buf->count, 2)));
    };
    const auto validate = [&] {
        const auto c = candidate.read(), b = wide.read(), p = pair.read(), s = scalar.read();
        require(!std::memcmp(c.data(), b.data(), mul(c.size(), 2)), "candidate versus batch16 bytes");
        require(!std::memcmp(c.data(), p.data(), mul(c.size(), 2)), "candidate versus two M5 bytes");
        require(!std::memcmp(c.data(), s.data(), mul(c.size(), 2)), "candidate versus scalar bytes");
        for (size_t i = 0; i < c.size(); ++i) {
            if (i < size_t(10) * n)
                require(std::isfinite(__bfloat162float(c[i])), "nonfinite live output");
            else require(bits(c[i]) == 0xffff, "unused output row modified");
        }
        a.guards(); w.guards(); ws.guards(); candidate.guards(); wide.guards(); pair.guards(); scalar.guards();
        return c;
    };
    reset(); launch(0); launch(1); launch(2); launch(3); CHECK(cudaDeviceSynchronize());
    if (opt.fault == Fault::Output) flip_byte(candidate.ptr);
    if (opt.fault == Fault::Unused) flip_byte(candidate.ptr + size_t(10) * n);
    if (opt.fault == Fault::Guard) flip_byte(candidate.allocation);
    const auto canonical = validate();
    if (impulse) {
        const float lut[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
        for (unsigned row = 0; row < 10; ++row) {
            const unsigned d = row * 7 % k;
            for (unsigned col = 0; col < n; ++col) {
                const unsigned char packed = weight[size_t(col) * (k / 2) + d / 2];
                const unsigned nibble = d % 2 ? packed >> 4 : packed & 15;
                require(bits(canonical[size_t(row) * n + col]) == bits(__float2bfloat16(lut[nibble])),
                        "independent signed impulse oracle");
            }
        }
    }
    const std::array<std::array<unsigned, 10>, 2> orders{{
        {{5, 6, 7, 8, 9, 0, 1, 2, 3, 4}}, {{4, 1, 3, 0, 2, 7, 9, 5, 8, 6}}
    }};
    for (const auto& order : orders) {
        auto permuted = input;
        for (unsigned row = 0; row < 10; ++row)
            std::copy_n(input.begin() + size_t(order[row]) * k, k,
                        permuted.begin() + size_t(row) * k);
        a.upload(permuted); reset(); launch(0); launch(1); launch(2); launch(3);
        CHECK(cudaDeviceSynchronize()); const auto actual = validate();
        for (unsigned row = 0; row < 10; ++row)
            require(!std::memcmp(actual.data() + size_t(row) * n,
                    canonical.data() + size_t(order[row]) * n, mul(n, 2)), "row permutation bytes");
    }
    a.upload(input);
    std::printf("M=10 N=%u K=%u profile=%u scale2=%.9g exact_batch16=1 exact_two_M5=1 exact_scalar=1 "
                "finite=1 unused_rows=6 permutations=2 live_bytes=%zu", n, k, profile, scale2, live_bytes);
    const bool timed = opt.repetitions && profile == 0 && n != 7;
    // First three cyclic orders balance all positions; the remaining two
    // distinct permutations leave each arm in each position once or twice.
    const std::array<std::array<unsigned, 3>, 5> timing_orders{{
        {{0, 1, 2}}, {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}, {{1, 0, 2}}
    }};
    std::array<std::array<float, 3>, 5> round_us{};
    if (timed) {
        for (unsigned i = 0; i < 10; ++i) { launch(0); launch(1); launch(2); }
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
        const auto measure = [&](unsigned arm) {
            CHECK(cudaEventRecord(start));
            for (unsigned i = 0; i < opt.repetitions; ++i) launch(arm);
            CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, end));
            require(std::isfinite(ms) && ms > 0, "invalid CUDA event duration");
            return ms * 1000 / opt.repetitions;
        };
        std::array<std::vector<float>, 3> samples;
        for (unsigned round = 0; round < 5; ++round) {
            for (unsigned arm : timing_orders[round]) {
                round_us[round][arm] = measure(arm);
                samples[arm].push_back(round_us[round][arm]);
            }
        }
        for (auto& values : samples) std::sort(values.begin(), values.end());
        const float exact = samples[0][2], batch16 = samples[1][2], two_m5 = samples[2][2];
        std::printf(" exact10_us=%.3f batch16_M10_us=%.3f two_M5_us=%.3f "
                    "speedup_vs_batch16=%.3f speedup_vs_two_M5=%.3f",
                    exact, batch16, two_m5, batch16 / exact, two_m5 / exact);
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
        // Recheck the live output and inactive-row payload after repeated launches.
        launch(3); CHECK(cudaDeviceSynchronize()); validate();
    }
    a.guards(); w.guards(); ws.guards(); candidate.guards(); wide.guards(); pair.guards(); scalar.guards();
    std::printf(" PASS\n");
    if (timed) {
        std::printf("timing_policy N=%u K=%u rounds=5 repetitions=%u warmup_per_arm=10 "
                    "arms=0:exact10,1:batch16,2:two_M5 residual_order_imbalance=1_or_2_per_position\n",
                    n, k, opt.repetitions);
        for (unsigned round = 0; round < 5; ++round)
            std::printf("timing_round N=%u K=%u round=%u order=%u,%u,%u "
                        "exact10_us=%.6f batch16_M10_us=%.6f two_M5_us=%.6f\n",
                        n, k, round, timing_orders[round][0], timing_orders[round][1],
                        timing_orders[round][2], round_us[round][0], round_us[round][1], round_us[round][2]);
    }
}

int main(int argc, char** argv) {
    const Options opt = options(argc, argv); // No CUDA calls before complete CLI validation.
    if (opt.fault == Fault::Budget) { Buffer<unsigned char> oversized(limit + 1); return 5; }
    unsigned cases = 0;
    for (const auto& shape : {std::array<unsigned, 2>{7, 80}, {4096, 4096}, {512, 4096}, {8192, 2048}}) {
        for (unsigned profile = 0; profile < 9; ++profile) {
            run(shape[0], shape[1], profile, opt); ++cases;
            require(opt.fault == Fault::None, "requested injected fault was not detected", 5);
        }
    }
    require(live_bytes == 0, "unreleased device allocations", 4);
    std::printf("cases=%u peak_device_allocated_bytes=%zu remaining_bytes=%zu PASS\n",
                cases, peak_bytes, live_bytes);
    return 0;
}
