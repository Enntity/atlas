// SPDX-License-Identifier: AGPL-3.0-only

// nvcc -O3 -arch=sm_121a scripts/dev/bench_glm_mla_batch.cu -o /tmp/bench-glm-mla-batch
// Bounded exactness and event-timing gate; no model weights or collectives.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static size_t live_bytes = 0, peak_bytes = 0;
static constexpr size_t allocation_limit = 16 * 1024 * 1024;

template <typename T> struct Buffer {
    static constexpr size_t guard = 128 / sizeof(T);
    T* allocation;
    T* ptr;
    size_t count;
    explicit Buffer(size_t count_) : count(count_) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T) - 2 * guard) {
            std::fprintf(stderr, "FAIL: allocation arithmetic overflow\n");
            std::exit(3);
        }
        const size_t bytes = (count + 2 * guard) * sizeof(T);
        if (bytes > allocation_limit - live_bytes) {
            std::fprintf(stderr, "FAIL: fixture exceeds16MiB live device allocations\n");
            std::exit(3);
        }
        CUDA_CHECK(cudaMalloc(&allocation, bytes));
        live_bytes += bytes;
        peak_bytes = std::max(peak_bytes, live_bytes);
        CUDA_CHECK(cudaMemset(allocation, 0xa5, bytes));
        ptr = allocation + guard;
    }
    ~Buffer() {
        CUDA_CHECK(cudaFree(allocation));
        live_bytes -= (count + 2 * guard) * sizeof(T);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void copy(const std::vector<T>& src) {
        if (src.size() != count) std::exit(3);
        CUDA_CHECK(cudaMemcpy(ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void guards() const {
        unsigned char before[128], after[128];
        CUDA_CHECK(cudaMemcpy(before, allocation, 128, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i) {
            if (before[i] != 0xa5 || after[i] != 0xa5) {
                std::fprintf(stderr, "FAIL: allocation guard overwritten\n");
                std::exit(3);
            }
        }
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
    const unsigned short poison = 0x7fc1;
    for (auto& value : input) std::memcpy(&value, &poison, 2);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned head = 0; head < heads; ++head)
            for (unsigned d = 0; d < k; ++d)
                input[size_t(row) * input_row_stride + head * input_head_stride + d] =
                    __float2bfloat16(dist(rng));
    for (auto& value : weight) value = __float2bfloat16(dist(rng));
    Buffer<__nv_bfloat16> di(input.size()), dw(weight.size());
    Buffer<__nv_bfloat16> baseline(output_elements), candidate(output_elements), scalar(output_elements);
    di.copy(input);
    dw.copy(weight);
    CUDA_CHECK(cudaMemset(baseline.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(candidate.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
    const dim3 grid((n + 7) / 8, heads);
    const auto launch = [&](bool optimized) {
        if (!optimized) {
            if (rows == 10) {
                for (unsigned segment = 0; segment < 2; ++segment)
                    mla_batched_gemv_batch5<<<grid, 256>>>(
                        di.ptr + size_t(segment * 5) * input_row_stride, dw.ptr,
                        baseline.ptr + size_t(segment * 5) * output_row_stride,
                        n, k, input_head_stride, output_head_stride,
                        input_row_stride, output_row_stride);
            } else {
                for (unsigned row = 0; row < rows; ++row)
                    mla_batched_gemv<<<grid, 256>>>(
                        di.ptr + size_t(row) * input_row_stride, dw.ptr,
                        baseline.ptr + size_t(row) * output_row_stride,
                        n, k, input_head_stride, output_head_stride);
            }
        } else if (rows == 2) {
            mla_batched_gemv_batch2<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else if (rows == 3) {
            mla_batched_gemv_batch3<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else if (rows == 4) {
            mla_batched_gemv_batch4<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else if (rows == 5) {
            mla_batched_gemv_batch5<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else if (rows == 10) {
            mla_batched_gemv_batch10<<<grid, 256>>>(di.ptr, dw.ptr, candidate.ptr,
                n, k, input_head_stride, output_head_stride, input_row_stride, output_row_stride);
        } else {
            std::fprintf(stderr, "FAIL: unsupported fixture row count\n");
            std::exit(3);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    launch(false);
    launch(true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> a(output_elements), b(output_elements);
    CUDA_CHECK(cudaMemcpy(a.data(), baseline.ptr, output_elements * sizeof(a[0]), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), candidate.ptr, output_elements * sizeof(b[0]), cudaMemcpyDeviceToHost));
    if (rows == 10) {
        // Independent legacy scalar control in addition to the timed two-M5
        // control. Compare padding too; no extra calls enter event timings.
        CUDA_CHECK(cudaMemset(scalar.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
        for (unsigned row = 0; row < rows; ++row)
            mla_batched_gemv<<<grid, 256>>>(
                di.ptr + size_t(row) * input_row_stride, dw.ptr,
                scalar.ptr + size_t(row) * output_row_stride,
                n, k, input_head_stride, output_head_stride);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> reference(output_elements);
        CUDA_CHECK(cudaMemcpy(reference.data(), scalar.ptr, output_elements * 2, cudaMemcpyDeviceToHost));
        if (std::memcmp(reference.data(), b.data(), output_elements * 2)) {
            std::fprintf(stderr, "FAIL: M10 differs from independent scalar control\n");
            std::exit(2);
        }
    }
    size_t mismatches = 0, guard_errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_elements; ++i) {
        const unsigned in_row = i % output_row_stride;
        const bool live = in_row < heads * output_head_stride && in_row % output_head_stride < n;
        if (!live) {
            unsigned short bits_a, bits_b;
            std::memcpy(&bits_a, &a[i], sizeof(bits_a));
            std::memcpy(&bits_b, &b[i], sizeof(bits_b));
            if (bits_a != 0xffff || bits_b != 0xffff) ++guard_errors;
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
            // Exhaustive CPU dots for M4/M10; sampled columns for M2/M3/M5.
            for (unsigned col = 0; col < n; ++col) {
                if (rows != 4 && rows != 10 && col != 0 && col != n / 2 && col != n - 1) continue;
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
    if (rows == 4 || rows == 10) {
        // Same allocations and weights, two nontrivial row permutations.
        // Compare full rows (including padding) with the canonical result;
        // this also verifies that pointer/row identity is not cached in CUDA.
        const std::vector<std::vector<unsigned>> orders = rows == 4
            ? std::vector<std::vector<unsigned>>{{3, 1, 0, 2}, {2, 0, 3, 1}}
            : std::vector<std::vector<unsigned>>{{5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
                                                 {4, 1, 3, 0, 2, 7, 9, 5, 8, 6}};
        for (const auto& order : orders) {
            auto permuted = input;
            for (unsigned row = 0; row < rows; ++row)
                std::copy_n(input.begin() + size_t(order[row]) * input_row_stride,
                            input_row_stride, permuted.begin() + size_t(row) * input_row_stride);
            di.copy(permuted);
            CUDA_CHECK(cudaMemset(baseline.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMemset(candidate.ptr, 0xff, output_elements * sizeof(__nv_bfloat16)));
            launch(false); launch(true);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> pa(output_elements), pb(output_elements);
            CUDA_CHECK(cudaMemcpy(pa.data(), baseline.ptr, pa.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pb.data(), candidate.ptr, pb.size() * 2, cudaMemcpyDeviceToHost));
            for (unsigned row = 0; row < rows; ++row) {
                const auto* expected = b.data() + size_t(order[row]) * output_row_stride;
                const size_t offset = size_t(row) * output_row_stride;
                if (std::memcmp(expected, pa.data() + offset, output_row_stride * 2) ||
                    std::memcmp(expected, pb.data() + offset, output_row_stride * 2)) {
                    std::fprintf(stderr, " FAIL: M%u row permutation mismatch at row%u\n", rows, row);
                    std::exit(2);
                }
            }
            di.guards(); dw.guards(); baseline.guards(); candidate.guards();
        }
        di.copy(input);
        std::printf(" permutations=2_BITEXACT");
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
        std::printf(" control=%s baseline_us=%.3f batched_us=%.3f speedup=%.3f",
                    rows == 10 ? "two_M5" : "scalar_rows",
                    control[2], optimized[2], control[2] / optimized[2]);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    di.guards(); dw.guards(); baseline.guards(); candidate.guards(); scalar.guards();
    std::printf(" PASS\n");
}

int main(int argc, char** argv) {
    const unsigned repetitions = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100;
    if (repetitions > 100) {
        std::fprintf(stderr, "repetitions must be <=100\n");
        return 1;
    }
    for (unsigned rows : {2u, 3u, 4u, 5u, 10u}) {
        run(rows, 3, 8, 8, true, 0);
        if (rows == 10) run(rows, 3, 9, 12, true, 0);
        run(rows, 32, 512, 256, true, 0);
        run(rows, 32, 256, 512, true, 0);
        run(rows, 32, 512, 256, false, repetitions);
        run(rows, 32, 256, 512, false, repetitions);
    }
    std::printf("peak_device_allocated_bytes=%zu remaining_bytes=%zu\n", peak_bytes, live_bytes);
    return 0;
}
