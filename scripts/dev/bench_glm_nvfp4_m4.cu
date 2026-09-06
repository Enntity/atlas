// SPDX-License-Identifier: AGPL-3.0-only

// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_nvfp4_m4.cu -o /tmp/bench-glm-nvfp4-m4
// Run only while the model is stopped, under an external bounded timeout.
// Synthetic exact-M4 vs four scalar projections; no model, network or NCCL.
// Argument: timing iterations per round, 0..200 (default20, 0 correctness only).
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../kernels/gb10/common/w4a16_gemv.cu"

#define CUDA_CHECK(call) do { const cudaError_t err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); } } while (0)
static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
static constexpr size_t memory_limit = 128ULL * 1024 * 1024;
static size_t device_live = 0, device_peak = 0;

template <typename T> struct Buffer {
    static constexpr size_t guard_bytes = 128;
    static constexpr size_t guard = guard_bytes / sizeof(T);
    T *allocation, *ptr;
    size_t count, bytes;
    explicit Buffer(size_t n) : count(n), bytes((n + 2 * guard) * sizeof(T)) {
        require(bytes < memory_limit && device_live < memory_limit - bytes,
                "aggregate device allocation must stay below128MiB");
        CUDA_CHECK(cudaMalloc(&allocation, bytes));
        device_live += bytes;
        device_peak = std::max(device_peak, device_live);
        CUDA_CHECK(cudaMemset(allocation, 0xa5, bytes));
        ptr = allocation + guard; // Retain16-byte vector-load alignment.
    }
    ~Buffer() { cudaFree(allocation); device_live -= bytes; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& values) const {
        require(values.size() == count, "upload size mismatch");
        CUDA_CHECK(cudaMemcpy(ptr, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(count);
        CUDA_CHECK(cudaMemcpy(values.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
    void check_guards() const {
        unsigned char before[guard_bytes], after[guard_bytes];
        CUDA_CHECK(cudaMemcpy(before, allocation, guard_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after, ptr + count, guard_bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard_bytes; ++i)
            require(before[i] == 0xa5 && after[i] == 0xa5, "allocation guard changed");
    }
};

// Independent host decode of finite E4M3, not the kernel's CUDA conversion.
static double scale_value(unsigned char value) {
    const unsigned exponent = (value >> 3) & 15, mantissa = value & 7;
    require(exponent != 15 || mantissa != 7, "NaN synthetic scale");
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -9)
        : std::ldexp(1.0 + double(mantissa) / 8.0, int(exponent) - 7);
    return value & 128 ? -magnitude : magnitude;
}

static void run_case(const char* name, unsigned n, unsigned k, bool zero_row,
                     unsigned iterations) {
    constexpr unsigned rows = 4;
    // The existing batch template has CTA barriers after its output-tail
    // return. Production GLM outputs satisfy N%4==0; do not adversarially
    // launch N%4 tails in this safety-focused harness. K packing requires16.
    require(n > 0 && n % 4 == 0 && k > 0 && k % 16 == 0,
            "fixture requires positive N%4==0 and K%16==0");
    const size_t packed_count = size_t(n) * k / 2;
    const size_t scale_count = size_t(n) * k / 16;
    const size_t host_bytes = packed_count + scale_count
        + (3 * size_t(rows) * k + 3 * size_t(rows) * n) * sizeof(__nv_bfloat16);
    require(host_bytes + device_live + packed_count + scale_count
            + (size_t(rows) * k + 2 * size_t(rows) * n) * 2 + 5 * 256 < memory_limit,
            "estimated combined host/device fixture storage must stay below128MiB");
    std::mt19937 random(6193 + n + k + unsigned(zero_row));
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    const float row_scales[rows] = {0.03125f, 0.25f, 1.0f, 4.0f};
    const unsigned char scale_codes[] = {0x00, 0x01, 0x08, 0x18, 0x28, 0x34, 0x38, 0x42};
    constexpr float tensor_scale = 0.375f;
    std::vector<__nv_bfloat16> input(size_t(rows) * k);
    std::vector<unsigned char> packed(packed_count), scales(scale_count);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned d = 0; d < k; ++d)
            input[size_t(row) * k + d] = __float2bfloat16(
                zero_row && row == 0 ? 0.0f : uniform(random) * row_scales[row]);
    for (auto& value : packed) value = static_cast<unsigned char>(random());
    for (auto& value : scales) value = scale_codes[random() % 8];
    Buffer<__nv_bfloat16> a(input.size()), scalar(size_t(rows) * n), batch(size_t(rows) * n);
    Buffer<unsigned char> w(packed.size()), s(scales.size());
    a.upload(input); w.upload(packed); s.upload(scales);
    const auto launch = [&](bool candidate) {
        if (candidate) {
            w4a16_gemv_batch4<<<n / 4, 256>>>(a.ptr, w.ptr, s.ptr, tensor_scale,
                batch.ptr, rows, n, k);
        } else {
            for (unsigned row = 0; row < rows; ++row)
                w4a16_gemv<<<n / 4, 256>>>(a.ptr + size_t(row) * k, w.ptr, s.ptr,
                    tensor_scale, scalar.ptr + size_t(row) * n, n, k);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    const double fp4[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
    double maximum_oracle_error = 0;
    for (unsigned permutation = 0; permutation < 2; ++permutation) {
        const unsigned order[rows] = {3, 1, 0, 2};
        std::vector<__nv_bfloat16> ordered(input.size());
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned source = permutation ? order[row] : row;
            std::copy_n(input.data() + size_t(source) * k, k, ordered.data() + size_t(row) * k);
        }
        a.upload(ordered);
        CUDA_CHECK(cudaMemset(scalar.ptr, 0xff, scalar.count * 2));
        CUDA_CHECK(cudaMemset(batch.ptr, 0xff, batch.count * 2));
        launch(false); launch(true);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto control = scalar.download(), candidate = batch.download();
        size_t mismatches = 0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            require(std::isfinite(__bfloat162float(control[i]))
                    && std::isfinite(__bfloat162float(candidate[i])), "nonfinite/unwritten output");
            mismatches += std::memcmp(&control[i], &candidate[i], 2) != 0;
        }
        if (mismatches) std::fprintf(stderr, "%s permutation%u: %zu bit mismatches\n",
                                    name, permutation, mismatches);
        require(mismatches == 0, "M4 differs from four scalar projections");
        // All columns on tiny fixtures, otherwise five distinct output columns.
        std::vector<unsigned> columns = {0, 1, n / 3, n / 2, n - 1};
        if (n <= 16) { columns.clear(); for (unsigned col = 0; col < n; ++col) columns.push_back(col); }
        for (unsigned row = 0; row < rows; ++row) for (unsigned col : columns) {
            double expected = 0, absolute_sum = 0;
            for (unsigned d = 0; d < k; ++d) {
                const unsigned byte = packed[size_t(col) * (k / 2) + d / 2];
                const unsigned nibble = d % 2 ? byte >> 4 : byte & 15;
                const double term = double(__bfloat162float(ordered[size_t(row) * k + d]))
                    * fp4[nibble] * scale_value(scales[size_t(col) * (k / 16) + d / 16])
                    * tensor_scale;
                expected += term; absolute_sum += std::abs(term);
            }
            const double actual = __bfloat162float(candidate[size_t(row) * n + col]);
            const double error = std::abs(actual - expected);
            maximum_oracle_error = std::max(maximum_oracle_error, error);
            // BF16 final rounding plus an FP32 reduction/cancellation budget.
            require(error <= 0.004 * std::abs(expected) + 1e-6 * absolute_sum + 1e-7,
                    "independent CPU double dot oracle failed");
        }
        a.check_guards(); w.check_guards(); s.check_guards();
        scalar.check_guards(); batch.check_guards();
    }
    std::printf("PASS %s M4 N=%u K=%u zero_row=%u cpu_maxabs=%g fixture_device_bytes=%zu",
                name, n, k, unsigned(zero_row), maximum_oracle_error, device_live);
    if (iterations) {
        for (unsigned i = 0; i < 2; ++i) { launch(false); launch(true); }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
        const auto measure = [&](bool candidate) {
            CUDA_CHECK(cudaEventRecord(start));
            for (unsigned i = 0; i < iterations; ++i) launch(candidate);
            CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
            return ms * 1000.0f / iterations;
        };
        std::vector<float> control_times, batch_times;
        for (unsigned round = 0; round < 5; ++round) {
            if (round % 2) {
                batch_times.push_back(measure(true)); control_times.push_back(measure(false));
            } else {
                control_times.push_back(measure(false)); batch_times.push_back(measure(true));
            }
        }
        std::sort(control_times.begin(), control_times.end());
        std::sort(batch_times.begin(), batch_times.end());
        std::printf(" four_scalar_us=%.3f batch4_us=%.3f speedup=%.3f",
                    control_times[2], batch_times[2], control_times[2] / batch_times[2]);
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(end));
    }
    std::printf("\n");
}

int main(int argc, char** argv) {
    unsigned iterations = 20;
    if (argc > 2) { std::fprintf(stderr, "usage: %s [iterations0..200]\n", argv[0]); return 1; }
    if (argc == 2) {
        errno = 0;
        char* end = nullptr;
        const unsigned long value = std::strtoul(argv[1], &end, 10);
        if (errno || end == argv[1] || *end || argv[1][0] == '-' || value > 200) {
            std::fprintf(stderr, "iterations must be an integer0..200\n"); return 1;
        }
        iterations = static_cast<unsigned>(value);
    }
    run_case("tiny_k_tail", 12, 48, false, 0);
    run_case("tiny_zero_row", 12, 48, true, 0);
    run_case("k_tail", 128, 2064, false, 0);
    run_case("kda_hot", 4096, 4096, false, iterations);
    run_case("shared_gate_up", 2048, 4096, false, iterations);
    run_case("shared_down", 4096, 2048, false, iterations);
    require(device_live == 0, "fixture allocations leaked");
    std::printf("PASS all shapes and row permutations; peak_device_bytes=%zu (<128MiB)\n", device_peak);
    return 0;
}
