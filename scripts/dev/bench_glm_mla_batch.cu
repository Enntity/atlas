// SPDX-License-Identifier: AGPL-3.0-only

// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_mla_batch.cu -o /tmp/bench-glm-mla-batch
// Bounded exactness and event-timing gate; no model weights or collectives.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu"

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

template <typename T> struct Buffer {
    T* ptr;
    explicit Buffer(size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void copy(const std::vector<T>& src) {
        CUDA_CHECK(cudaMemcpy(ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

static void run(unsigned rows, unsigned heads, unsigned n, unsigned k,
                bool padded, unsigned repetitions) {
    const unsigned input_head_stride = k + (padded ? 8 : 0);
    const unsigned output_head_stride = n + (padded ? 8 : 0);
    const unsigned input_row_stride = heads * input_head_stride + (padded ? 32 : 0);
    const unsigned output_row_stride = heads * output_head_stride + (padded ? 32 : 0);
    const size_t output_elements = size_t(rows) * output_row_stride;
    std::mt19937 rng(9187 + rows * 7 + n + k + padded);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> input(size_t(rows) * input_row_stride);
    std::vector<__nv_bfloat16> weight(size_t(heads) * n * k);
    for (auto& value : input) value = __float2bfloat16(dist(rng));
    for (auto& value : weight) value = __float2bfloat16(dist(rng));
    Buffer<__nv_bfloat16> di(input.size()), dw(weight.size());
    Buffer<__nv_bfloat16> baseline(output_elements), candidate(output_elements);
    di.copy(input);
    dw.copy(weight);
    CUDA_CHECK(cudaMemset(baseline.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(candidate.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
    const dim3 grid((n + 7) / 8, heads);
    const auto launch = [&](bool optimized) {
        if (!optimized) {
            for (unsigned row = 0; row < rows; ++row)
                mla_batched_gemv<<<grid, 256>>>(
                    di.ptr + size_t(row) * input_row_stride, dw.ptr,
                    baseline.ptr + size_t(row) * output_row_stride,
                    n, k, input_head_stride, output_head_stride);
        } else if (rows == 2) {
            mla_batched_gemv_batch2<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else if (rows == 3) {
            mla_batched_gemv_batch3<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else {
            mla_batched_gemv_batch5<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    launch(false);
    launch(true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> a(output_elements), b(output_elements);
    CUDA_CHECK(cudaMemcpy(a.data(), baseline.ptr, output_elements * sizeof(a[0]), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), candidate.ptr, output_elements * sizeof(b[0]), cudaMemcpyDeviceToHost));
    size_t mismatches = 0, guard_errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_elements; ++i) {
        const unsigned in_row = i % output_row_stride;
        const bool live = in_row < heads * output_head_stride && in_row % output_head_stride < n;
        if (!live) {
            unsigned short bits;
            std::memcpy(&bits, &b[i], sizeof(bits));
            if (bits != 0xffff) ++guard_errors;
            continue;
        }
        if (std::memcmp(&a[i], &b[i], sizeof(a[i]))) ++mismatches;
        const float av = __bfloat162float(a[i]), bv = __bfloat162float(b[i]);
        if (!std::isfinite(av) || !std::isfinite(bv)) ++guard_errors;
        max_diff = std::max(max_diff, std::abs(av - bv));
    }
    // Independent host dot checks ensure the two paths do not share a silent
    // head/row addressing error. BF16 rounding dominates this loose FP32 gate.
    float max_reference_error = 0.0f;
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned head = 0; head < heads; ++head) {
            for (unsigned col : {0u, n / 2, n - 1}) {
                double expected = 0.0;
                for (unsigned d = 0; d < k; ++d)
                    expected += double(__bfloat162float(input[size_t(row) * input_row_stride
                        + head * input_head_stride + d]))
                        * __bfloat162float(weight[(size_t(head) * n + col) * k + d]);
                const float actual = __bfloat162float(b[size_t(row) * output_row_stride
                    + head * output_head_stride + col]);
                const float error = std::abs(actual - float(expected));
                max_reference_error = std::max(max_reference_error, error);
                if (error > 0.005f * std::abs(float(expected)) + 0.0002f) ++guard_errors;
            }
        }
    }
    std::printf("rows=%u heads=%u N=%u K=%u padded=%d mismatches=%zu guard_errors=%zu "
                "max_diff=%g reference_error=%g", rows, heads, n, k, padded,
                mismatches, guard_errors, max_diff, max_reference_error);
    if (mismatches || guard_errors) {
        std::printf(" FAIL\n");
        std::exit(2);
    }
    if (repetitions) {
        for (unsigned i = 0; i < 10; ++i) { launch(false); launch(true); }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        const auto measure = [&](bool optimized) {
            CUDA_CHECK(cudaEventRecord(start));
            for (unsigned i = 0; i < repetitions; ++i) launch(optimized);
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
            return ms * 1000.0f / repetitions;
        };
        // Interleave arms at round granularity and report medians to reduce
        // clock/order effects while leaving allocation outside every timing.
        std::vector<float> control, optimized;
        for (unsigned round = 0; round < 5; ++round) {
            if (round % 2) {
                optimized.push_back(measure(true));
                control.push_back(measure(false));
            } else {
                control.push_back(measure(false));
                optimized.push_back(measure(true));
            }
        }
        std::sort(control.begin(), control.end());
        std::sort(optimized.begin(), optimized.end());
        std::printf(" baseline_us=%.3f batched_us=%.3f speedup=%.3f",
                    control[2], optimized[2], control[2] / optimized[2]);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    std::printf(" PASS\n");
}

int main(int argc, char** argv) {
    const unsigned repetitions = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100;
    if (repetitions > 1000) {
        std::fprintf(stderr, "repetitions must be <=1000\n");
        return 1;
    }
    for (unsigned rows : {2u, 3u, 5u}) {
        run(rows, 3, 8, 8, true, 0);
        run(rows, 32, 512, 256, true, 0);
        run(rows, 32, 256, 512, true, 0);
        run(rows, 32, 512, 256, false, repetitions);
        run(rows, 32, 256, 512, false, repetitions);
    }
    return 0;
}
