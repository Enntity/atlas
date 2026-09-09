// SPDX-License-Identifier: AGPL-3.0-only
// Standalone actual-kernel numerical gate; no Model/NCCL/serving qualification.
// Root pins the included source separately for common RED and GLM-shadow GREEN:
// nvcc -std=c++17 -O3 -arch=sm_121a -lineinfo \
//   -DATLAS_SSM_NORM_SOURCE='"/ABS/ssm_state_norm.cu"' \
//   bench-ssm-norm-release.cu -o bench-ssm-norm-release
// compute-sanitizer --tool memcheck --error-exitcode 99 ./bench-ssm-norm-release
// compute-sanitizer --tool racecheck --error-exitcode 99 ./bench-ssm-norm-release
// An undefined write race can happen to select lane0: numerical PASS alone
// cannot disprove it. Never alter data/tolerances between source comparisons.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
#ifndef ATLAS_SSM_NORM_SOURCE
#error "Define ATLAS_SSM_NORM_SOURCE as the exact pinned kernel file"
#endif
#include ATLAS_SSM_NORM_SOURCE

constexpr unsigned heads = 32, layers = 3, kd = 128, vd = 128;
constexpr unsigned guard = 64, repeats = 16, cases = 8;
constexpr size_t matrix = size_t(kd) * vd;
constexpr size_t payload = size_t(heads) * matrix;
constexpr size_t stride = payload + 2 * guard;
constexpr size_t max_device_bytes = layers * stride * sizeof(float)
                                  + (layers + 2) * sizeof(void*);
static_assert(max_device_bytes < 7 * 1024 * 1024, "bounded device allocations");
constexpr double limit = 200.0;
#define CUDA(call) do { const cudaError_t error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "CUDA line%d: %s\n", __LINE__, cudaGetErrorString(error)); \
    std::exit(1); } } while (0)

template<class T> static T narrow(float x) {
    if constexpr (std::is_same_v<T, float>) return x;
    else return __float2half_rn(x);
}
template<class T> static float wide(T x) {
    if constexpr (std::is_same_v<T, float>) return x;
    else return __half2float(x);
}
template<class T> static bool same(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template<class T> static bool close(T actual, T expected) {
    if (!std::isfinite(wide(actual))) return false;
    if constexpr (std::is_same_v<T, float>) {
        return std::abs(double(actual) - expected) <= 1e-6 + 2e-5 * std::abs(expected);
    } else {
        // At most one half ULP from the CPU double-norm/round-to-nearest oracle.
        uint16_t a, b;
        std::memcpy(&a, &actual, sizeof(a));
        std::memcpy(&b, &expected, sizeof(b));
        return (a >> 15) == (b >> 15) && std::abs(int(a) - int(b)) <= 1;
    }
}
static float input_value(unsigned test, unsigned row, unsigned col, unsigned layer) {
    float value;
    if (test == 0) value = 0.0f;
    else if (test == 1) {
        constexpr float below[4] = {1.0f, .5f, .25f, .125f};
        value = below[col / 32]; // Frobenius norm approximately73.76, unchanged.
    } else if (test == 2) value = 1.5625f; // Exactly200, must remain bitwise unchanged.
    else if (test == 3) value = 1.5703125f; // Exactly201, just above clamp boundary.
    else {
        constexpr float unequal[4] = {8.0f, 2.0f, .5f, .125f};
        value = unequal[(col / 32 + test - 4) % 4];
        // Rotate the dominant warp through all four second-stage lanes.
    }
    return ((row + col + layer) & 1) ? -value : value;
}

template<class T> static size_t run() {
    const char* dtype = std::is_same_v<T, float> ? "f32" : "f16";
    const T marker = narrow<T>(123.5f);
    std::vector<T> input(layers * stride, marker), gold(input.size()), actual(input.size());
    std::array<double, layers * heads> norms{};
    for (unsigned l = 0; l < layers; ++l) {
        for (unsigned h = 0; h < heads; ++h) {
            const unsigned test = (h + l) % cases;
            const size_t base = size_t(l) * stride + guard + size_t(h) * matrix;
            double sum = 0;
            for (unsigned j = 0; j < kd; ++j) {
                for (unsigned c = 0; c < vd; ++c) {
                    const size_t i = base + size_t(j) * vd + c;
                    input[i] = narrow<T>(input_value(test, j, c, l));
                    const double x = wide(input[i]);
                    sum += x * x;
                }
            }
            norms[l * heads + h] = std::sqrt(sum);
        }
    }
    gold = input;
    for (unsigned l = 0; l < layers; ++l) {
        for (unsigned h = 0; h < heads; ++h) {
            const double norm = norms[l * heads + h];
            if (norm <= limit) continue;
            const size_t base = size_t(l) * stride + guard + size_t(h) * matrix;
            for (size_t i = 0; i < matrix; ++i)
                gold[base + i] = narrow<T>(float(double(wide(input[base + i])) * limit / norm));
        }
    }
    T* device = nullptr;
    T** table = nullptr;
    CUDA(cudaMalloc(reinterpret_cast<void**>(&device), input.size() * sizeof(T)));
    CUDA(cudaMalloc(reinterpret_cast<void**>(&table), (layers + 2) * sizeof(T*)));
    // Guard both ends of the device pointer table too; kernel receives table+1.
    std::array<T*, layers + 2> pointers{};
    pointers.front() = device;
    pointers.back() = device + input.size() - 1;
    constexpr unsigned order[layers] = {2, 0, 1};
    for (unsigned l = 0; l < layers; ++l)
        pointers[l + 1] = device + size_t(order[l]) * stride + guard;
    CUDA(cudaMemcpy(table, pointers.data(), sizeof(pointers), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t bad = 0, reported = 0;
    auto failure = [&](const char* why, unsigned iteration, unsigned l, unsigned h,
                       double got, double expected) {
        ++bad;
        if (reported++ < 12)
            std::fprintf(stderr, "FAIL %s %s iteration%u layer%u head%u got=%.10g expected=%.10g\n",
                         dtype, why, iteration, l, h, got, expected);
    };
    for (unsigned iteration = 0; iteration < repeats; ++iteration) {
        CUDA(cudaMemcpyAsync(device, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
        // Actual production dimensions and kernel ABI; extra x-CTA exercises
        // the existing head>=num_heads guard without creating another state.
        if constexpr (std::is_same_v<T, float>)
            ssm_state_clamp_norm_fused<<<dim3(heads + 1, layers), vd, 0, stream>>>(table + 1, heads, kd, vd);
        else
            ssm_state_clamp_norm_fused_f16<<<dim3(heads + 1, layers), vd, 0, stream>>>(table + 1, heads, kd, vd);
        CUDA(cudaGetLastError());
        CUDA(cudaMemcpyAsync(actual.data(), device, actual.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA(cudaStreamSynchronize(stream));
        for (unsigned l = 0; l < layers; ++l) {
            for (unsigned g = 0; g < guard; ++g) {
                if (!same(actual[size_t(l) * stride + g], marker)
                    || !same(actual[size_t(l) * stride + guard + payload + g], marker))
                    failure("canary", iteration, l, 0, g, 0);
            }
            for (unsigned h = 0; h < heads; ++h) {
                const size_t base = size_t(l) * stride + guard + size_t(h) * matrix;
                const double before = norms[l * heads + h];
                double after_sq = 0;
                bool mismatch = false;
                for (size_t i = 0; i < matrix; ++i) {
                    const T a = actual[base + i], e = gold[base + i];
                    mismatch |= before <= limit ? !same(a, e) : !close(a, e);
                    const double x = wide(a);
                    after_sq += x * x;
                }
                const double after = std::sqrt(after_sq), expected_norm = std::min(before, limit);
                const double norm_tolerance = std::is_same_v<T, float> ? .002 : .15;
                if (mismatch || !std::isfinite(after) || std::abs(after - expected_norm) > norm_tolerance)
                    failure("matrix/norm", iteration, l, h, after, expected_norm);
                if (iteration == 0 && l == 0 && h < cases)
                    std::printf("%s case%u input_norm=%.9f actual_norm=%.9f expected_norm=%.9f\n",
                                dtype, h, before, after, expected_norm);
            }
        }
    }
    std::array<T*, layers + 2> returned{};
    CUDA(cudaMemcpy(returned.data(), table, sizeof(returned), cudaMemcpyDeviceToHost));
    if (returned != pointers) failure("pointer table mutated", repeats, 0, 0, 1, 0);
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(table));
    CUDA(cudaFree(device));
    std::printf("%s %s failures=%zu launches=%u guarded_layers=%u heads=%u k=%u v=%u\n",
                dtype, bad ? "FAIL" : "PASS", bad, repeats, layers, heads, kd, vd);
    return bad;
}

int main(int argc, char**) {
    if (argc != 1) { std::fprintf(stderr, "No runtime knobs; source comparison uses fixed data.\n"); return 2; }
    CUDA(cudaSetDevice(0));
    std::printf("source=%s max_device_bytes=%zu threshold=%.0f\n", ATLAS_SSM_NORM_SOURCE, max_device_bytes, limit);
    const size_t f32_bad = run<float>();
    const size_t f16_bad = run<__half>();
    return f32_bad + f16_bad ? 2 : 0;
}
