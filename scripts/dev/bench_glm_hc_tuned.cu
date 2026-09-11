// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false -Xptxas=-v scripts/dev/bench_glm_hc_tuned.cu -o /tmp/bench-glm-hc-tuned
// No weights or model state: exact shipped finalizer versus half-warp Sinkhorn.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu"
#include "glm_hc_tuned.cuh"
#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/glm_hc_prefill_vec.cu"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)
template <typename T> struct Buffer {
    T* p;
    explicit Buffer(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void copy(const std::vector<T>& x) {
        CUDA_CHECK(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};
template <typename T> static std::vector<T> read(const Buffer<T>& x, size_t n) {
    std::vector<T> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), x.p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void run(unsigned rows, unsigned iters, float magnitude, unsigned variant, unsigned repeats = 9) {
    constexpr unsigned H = 4096, HC = 4, D = H * HC;
    constexpr float norm_eps = 1e-6f, hc_eps = 1e-6f;
    std::mt19937 rng(9121 + rows + iters);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> x(size_t(rows) * D), raw(size_t(rows) * 24), base(24);
    const std::vector<float> scale{0.17f, -0.23f, 0.31f};
    for (auto& v : x) v = random(rng) * 1.7f;
    for (auto& v : raw) v = random(rng) * magnitude;
    for (auto& v : base) v = random(rng) * 0.7f;
    // A zero highway/raw row exercises RMS epsilon and zero collapsed output.
    if (rows > 1) {
        std::fill(x.begin(), x.begin() + D, 0.f);
        std::fill(raw.begin(), raw.begin() + 24, 0.f);
    }
    Buffer<float> dx(x.size()), dr(raw.size()), db(24), ds(3);
    Buffer<__nv_bfloat16> y0(size_t(rows) * H), y1(size_t(rows) * H);
    Buffer<float> p0(size_t(rows) * 4), p1(size_t(rows) * 4);
    Buffer<float> c0(size_t(rows) * 16), c1(size_t(rows) * 16);
    dx.copy(x); dr.copy(raw); db.copy(base); ds.copy(scale);
    CUDA_CHECK(cudaMemset(y0.p, 0xff, size_t(rows) * H * 2));
    CUDA_CHECK(cudaMemset(y1.p, 0xff, size_t(rows) * H * 2));
    CUDA_CHECK(cudaMemset(p0.p, 0xff, size_t(rows) * 4 * 4));
    CUDA_CHECK(cudaMemset(p1.p, 0xff, size_t(rows) * 4 * 4));
    CUDA_CHECK(cudaMemset(c0.p, 0xff, size_t(rows) * 16 * 4));
    CUDA_CHECK(cudaMemset(c1.p, 0xff, size_t(rows) * 16 * 4));
    const auto launch = [&](bool optimized) {
        if (optimized && variant == 0)
            glm_hc_pre_from_raw_mix_vec<<<rows, 256>>>(dx.p, dr.p, ds.p, db.p,
                y1.p, p1.p, c1.p, H, HC, iters, norm_eps, hc_eps);
        else if (optimized)
            glm_hc_pre_from_raw_mix_tuned<1><<<rows, 256>>>(dx.p, dr.p, ds.p, db.p,
                y1.p, p1.p, c1.p, H, HC, iters, norm_eps, hc_eps);
        else
            hc_pre_from_raw_mix<<<rows, 256>>>(dx.p, dr.p, ds.p, db.p,
                y0.p, p0.p, c0.p, H, HC, iters, norm_eps, hc_eps);
        CUDA_CHECK(cudaGetLastError());
    };
    launch(false); launch(true); CUDA_CHECK(cudaDeviceSynchronize());
    auto ya = read(y0, size_t(rows) * H), yb = read(y1, size_t(rows) * H);
    auto pa = read(p0, size_t(rows) * 4), pb = read(p1, size_t(rows) * 4);
    auto ca = read(c0, size_t(rows) * 16), cb = read(c1, size_t(rows) * 16);
    size_t mismatch = 0;
    float max_diff = 0.f, oracle_error = 0.f;
    for (size_t i = 0; i < ya.size(); ++i) {
        if (std::memcmp(&ya[i], &yb[i], 2)) ++mismatch;
        const float a = __bfloat162float(ya[i]), b = __bfloat162float(yb[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) std::exit(2);
        max_diff = std::max(max_diff, std::abs(a - b));
    }
    for (const auto pair : {std::make_pair(&pa, &pb), std::make_pair(&ca, &cb)}) {
        for (size_t i = 0; i < pair.first->size(); ++i) {
            const float a = (*pair.first)[i], b = (*pair.second)[i];
            if (!std::isfinite(a) || !std::isfinite(b)) std::exit(2);
            if (std::memcmp(&a, &b, 4)) ++mismatch;
            max_diff = std::max(max_diff, std::abs(a - b));
        }
    }
    if (mismatch) {
        std::fprintf(stderr, "HC arithmetic changed: %zu differing elements, max=%g\n", mismatch, max_diff);
        std::exit(2);
    }
    // Independent double oracle checks all three outputs at small shapes.
    if (rows <= 7) for (unsigned t = 0; t < rows; ++t) {
        double sum = 0.;
        for (unsigned d = 0; d < D; ++d) sum += double(x[size_t(t) * D + d]) * x[size_t(t) * D + d];
        const double norm = 1. / std::sqrt(sum / D + norm_eps);
        double pre[4], post[4], comb[16];
        for (unsigned i = 0; i < 4; ++i) {
            pre[i] = 1. / (1. + std::exp(-(double(raw[size_t(t) * 24 + i]) * norm * scale[0] + base[i]))) + hc_eps;
            post[i] = 2. / (1. + std::exp(-(double(raw[size_t(t) * 24 + 4 + i]) * norm * scale[1] + base[4 + i])));
        }
        for (unsigned i = 0; i < 16; ++i)
            comb[i] = double(raw[size_t(t) * 24 + 8 + i]) * norm * scale[2] + base[8 + i];
        for (unsigned i = 0; i < 4; ++i) {
            double mx = -INFINITY, total = 0.;
            for (unsigned j = 0; j < 4; ++j) mx = std::max(mx, comb[i * 4 + j]);
            for (unsigned j = 0; j < 4; ++j) { comb[i * 4 + j] = std::exp(comb[i * 4 + j] - mx); total += comb[i * 4 + j]; }
            for (unsigned j = 0; j < 4; ++j) comb[i * 4 + j] = comb[i * 4 + j] / total + hc_eps;
        }
        const auto columns = [&](double eps) {
            for (unsigned j = 0; j < 4; ++j) {
                double total = eps;
                for (unsigned i = 0; i < 4; ++i) total += comb[i * 4 + j];
                for (unsigned i = 0; i < 4; ++i) comb[i * 4 + j] /= total;
            }
        };
        columns(hc_eps);
        for (unsigned iter = 0; iter + 1 < iters; ++iter) {
            for (unsigned i = 0; i < 4; ++i) {
                double total = hc_eps;
                for (unsigned j = 0; j < 4; ++j) total += comb[i * 4 + j];
                for (unsigned j = 0; j < 4; ++j) comb[i * 4 + j] /= total;
            }
            columns(hc_eps);
        }
        columns(0.);
        for (unsigned d = 0; d < H; ++d) {
            double expected = 0.;
            for (unsigned i = 0; i < 4; ++i) expected += pre[i] * x[size_t(t) * D + i * H + d];
            const double delta = std::abs(__bfloat162float(yb[size_t(t) * H + d]) - expected);
            oracle_error = std::max(oracle_error, float(delta));
            if (delta > 0.002 + 0.004 * std::abs(expected)) { std::fprintf(stderr, "HC y oracle failed\n"); std::exit(2); }
        }
        for (unsigned i = 0; i < 4; ++i)
            if (std::abs(pb[size_t(t) * 4 + i] - post[i]) > 2e-5) { std::fprintf(stderr, "HC post oracle failed\n"); std::exit(2); }
        for (unsigned i = 0; i < 16; ++i)
            if (std::abs(cb[size_t(t) * 16 + i] - comb[i]) > 2e-5) { std::fprintf(stderr, "HC comb oracle failed\n"); std::exit(2); }
    }
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> times[2];
    for (unsigned r = 0; r < repeats; ++r) for (unsigned j = 0; j < 2; ++j) {
        const unsigned mode = (r + j) % 2;
        CUDA_CHECK(cudaEventRecord(begin)); launch(mode != 0); CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end)); times[mode].push_back(ms);
    }
    for (auto& values : times) std::sort(values.begin(), values.end());
    const float before = times[0][repeats / 2], after = times[1][repeats / 2];
    std::printf("{\"stage\":\"pre\",\"variant\":%u,\"rows\":%u,\"iters\":%u,\"magnitude\":%g,\"mismatches\":%zu,\"max_diff\":%g,\"oracle_error\":%g,\"baseline_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.3f}\n",
        variant, rows, iters, magnitude, mismatch, max_diff, oracle_error, before, after, before / after);
    std::fflush(stdout);
    CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
}

static void post_run(unsigned rows, unsigned threads, bool in_place, unsigned repeats = 9) {
    constexpr unsigned H = 4096;
    const size_t n = size_t(rows) * 4 * H;
    std::mt19937 rng(9943 + rows);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> res(n), post(size_t(rows) * 4), comb(size_t(rows) * 16);
    std::vector<__nv_bfloat16> block(size_t(rows) * H);
    for (auto& v : res) v = random(rng) * 3.f;
    for (auto& v : block) v = __float2bfloat16(random(rng) * 2.f);
    for (auto& v : post) v = random(rng) + 1.f;
    for (unsigned t = 0; t < rows; ++t) for (unsigned j = 0; j < 4; ++j) {
        float total = 0.f;
        for (unsigned i = 0; i < 4; ++i) {
            comb[size_t(t) * 16 + i * 4 + j] = random(rng) + 1.01f;
            total += comb[size_t(t) * 16 + i * 4 + j];
        }
        for (unsigned i = 0; i < 4; ++i) comb[size_t(t) * 16 + i * 4 + j] /= total;
    }
    Buffer<float> r0(n), r1(n), o0(n), o1(n), dp(post.size()), dc(comb.size());
    Buffer<__nv_bfloat16> dx(block.size());
    r0.copy(res); r1.copy(res); dp.copy(post); dc.copy(comb); dx.copy(block);
    CUDA_CHECK(cudaMemset(o0.p, 0xff, n * 4)); CUDA_CHECK(cudaMemset(o1.p, 0xff, n * 4));
    const auto launch = [&](bool optimized) {
        if (optimized)
            glm_hc_post_vec4<<<rows, threads>>>(dx.p, r1.p, dp.p, dc.p,
                in_place ? r1.p : o1.p, H, 4);
        else
            hc_post<<<rows, 256>>>(dx.p, r0.p, dp.p, dc.p,
                in_place ? r0.p : o0.p, H, 4);
        CUDA_CHECK(cudaGetLastError());
    };
    launch(false); launch(true); CUDA_CHECK(cudaDeviceSynchronize());
    const auto a = read(in_place ? r0 : o0, n), b = read(in_place ? r1 : o1, n);
    size_t mismatch = 0;
    float max_diff = 0.f, oracle_error = 0.f;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) std::exit(2);
        if (std::memcmp(&a[i], &b[i], 4)) ++mismatch;
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    if (mismatch) {
        std::fprintf(stderr, "HC post arithmetic changed: %zu differences, max=%g\n", mismatch, max_diff);
        std::exit(2);
    }
    if (rows <= 7) for (unsigned t = 0; t < rows; ++t)
        for (unsigned j = 0; j < 4; ++j) for (unsigned d = 0; d < H; ++d) {
            double expected = double(post[size_t(t) * 4 + j]) * __bfloat162float(block[size_t(t) * H + d]);
            for (unsigned i = 0; i < 4; ++i)
                expected += double(comb[size_t(t) * 16 + i * 4 + j]) * res[(size_t(t) * 4 + i) * H + d];
            const double error = std::abs(b[(size_t(t) * 4 + j) * H + d] - expected);
            oracle_error = std::max(oracle_error, float(error));
            if (error > 2e-6 + 1e-6 * std::abs(expected)) {
                std::fprintf(stderr, "HC post double oracle failed\n"); std::exit(2);
            }
        }
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> times[2];
    for (unsigned rep = 0; rep < repeats; ++rep) for (unsigned j = 0; j < 2; ++j) {
        const unsigned mode = (rep + j) % 2;
        CUDA_CHECK(cudaEventRecord(begin)); launch(mode != 0); CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end)); times[mode].push_back(ms);
    }
    for (auto& values : times) std::sort(values.begin(), values.end());
    const float before = times[0][repeats / 2], after = times[1][repeats / 2];
    std::printf("{\"stage\":\"post\",\"rows\":%u,\"threads\":%u,\"in_place\":%s,\"mismatches\":%zu,\"max_diff\":%g,\"oracle_error\":%g,\"baseline_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.3f}\n",
        rows, threads, in_place ? "true" : "false", mismatch, max_diff, oracle_error, before, after, before / after);
    std::fflush(stdout);
    CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
}

int main(int argc, char** argv) {
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 1) return 1;
    const bool pre = argc == 1 || (argc == 2 && !std::strcmp(argv[1], "pre"));
    const bool post = argc == 1 || (argc == 2 && !std::strcmp(argv[1], "post"));
    if (!pre && !post) return 1;
    if (pre) for (unsigned variant : {0u, 1u}) {
        run(1, 1, 4.f, variant); run(3, 20, 4.f, variant); run(7, 20, 80.f, variant);
        run(1024, 20, 8.f, variant); run(2048, 20, 8.f, variant);
    }
    if (post) for (unsigned threads : {128u, 256u}) {
        post_run(1, threads, false); post_run(3, threads, true);
        post_run(7, threads, false); post_run(1024, threads, true);
        post_run(2048, threads, true);
    }
}
