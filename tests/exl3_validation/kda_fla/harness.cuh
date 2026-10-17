// SPDX-License-Identifier: AGPL-3.0-only

// Host harness for byte-exact tests of the EXL3 reference-policy KDA kernels
// against the GPU stage oracle (arithmetic-audit/kda-reference-stage-oracle-*).
// Each test loads oracle stage files as kernel inputs, runs one kernel and
// compares its output bytes with the oracle's next stage. A test that cannot
// find its fixtures fails (exit 2); it never reports a skip as a pass.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define KDA_CHECK(x)                                                                         \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                             \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(3);                                                                         \
        }                                                                                    \
    } while (0)

namespace kda_harness {

constexpr int HL = 32;
constexpr int D = 128;
constexpr int BT = 64;

// Oracle root, e.g. /home/colum/kda-oracle/inductor, from KDA_ORACLE.
inline std::string oracle_root() {
    const char* r = getenv("KDA_ORACLE");
    if (!r) {
        fprintf(stderr, "KDA_ORACLE is not set\n");
        exit(2);
    }
    return r;
}

inline std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "missing fixture %s\n", path.c_str());
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if (fread(b.data(), 1, n, f) != (size_t)n) {
        fprintf(stderr, "short read %s\n", path.c_str());
        exit(2);
    }
    fclose(f);
    return b;
}

// Stage file of one case, e.g. stage("rank0-000016", "q_l2").
inline std::vector<uint8_t> stage(const std::string& c, const std::string& name) {
    return read_file(oracle_root() + "/" + c + "/" + name + ".bin");
}

template <typename T>
T* to_device(const std::vector<uint8_t>& h) {
    void* d = nullptr;
    KDA_CHECK(cudaMalloc(&d, h.size() ? h.size() : 1));
    if (!h.empty()) KDA_CHECK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    return (T*)d;
}

template <typename T>
T* device_zeros(size_t bytes) {
    void* d = nullptr;
    KDA_CHECK(cudaMalloc(&d, bytes ? bytes : 1));
    KDA_CHECK(cudaMemset(d, 0, bytes ? bytes : 1));
    return (T*)d;
}

inline std::vector<uint8_t> to_host(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    KDA_CHECK(cudaDeviceSynchronize());
    KDA_CHECK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    return h;
}

// Element-wise byte comparison; elem = 2 (bf16) or 4 (f32). Prints the first
// few mismatches and returns the mismatch count.
inline size_t compare(const char* what, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want,
                      size_t elem, size_t row_elems = 0) {
    if (got.size() != want.size()) {
        printf("%-28s SIZE MISMATCH got %zu want %zu\n", what, got.size(), want.size());
        return (size_t)-1;
    }
    size_t n = got.size() / elem, bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&got[i * elem], &want[i * elem], elem) != 0) {
            if (bad < 5) {
                float g = 0, w = 0;
                if (elem == 4) {
                    memcpy(&g, &got[i * elem], 4);
                    memcpy(&w, &want[i * elem], 4);
                } else {
                    uint32_t gb = (uint32_t)(got[i * 2] | (got[i * 2 + 1] << 8)) << 16;
                    uint32_t wb = (uint32_t)(want[i * 2] | (want[i * 2 + 1] << 8)) << 16;
                    memcpy(&g, &gb, 4);
                    memcpy(&w, &wb, 4);
                }
                if (row_elems)
                    printf("  %s[%zu] (row %zu, col %zu): got %.9g want %.9g\n", what, i, i / row_elems,
                           i % row_elems, g, w);
                else
                    printf("  %s[%zu]: got %.9g want %.9g\n", what, i, g, w);
            }
            ++bad;
        }
    }
    printf("%-28s %zu / %zu mismatches\n", what, bad, n);
    return bad;
}

// Transpose per-head [V,K] (reference) <-> [K,V] (Atlas) f32 state planes.
inline std::vector<uint8_t> transpose_state(const std::vector<uint8_t>& s, int heads) {
    std::vector<uint8_t> o(s.size());
    const float* in = (const float*)s.data();
    float* out = (float*)o.data();
    for (int h = 0; h < heads; ++h)
        for (int a = 0; a < D; ++a)
            for (int b = 0; b < D; ++b) out[(size_t)h * D * D + b * D + a] = in[(size_t)h * D * D + a * D + b];
    return o;
}

inline int rows_of(const std::string& c) {
    // Case names: rankR-000324 (5), rankR-000016 (128), rankR-seqN-prefillT, rankR-000325 / seqN-decode (1).
    if (c.find("000324") != std::string::npos) return 5;
    if (c.find("000016") != std::string::npos) return 128;
    size_t p = c.find("prefill");
    if (p != std::string::npos) return atoi(c.c_str() + p + 7);
    return 1;
}

// All prefill cases (both ranks) present in the oracle.
inline std::vector<std::string> prefill_cases() {
    std::vector<std::string> v;
    for (int r = 0; r < 2; ++r) {
        std::string R = "rank" + std::to_string(r);
        v.push_back(R + "-000324");
        v.push_back(R + "-000016");
        v.push_back(R + "-seq0-prefill5");
        v.push_back(R + "-seq1-prefill64");
        v.push_back(R + "-seq2-prefill59");
        v.push_back(R + "-seq3-prefill200");
    }
    return v;
}

inline std::vector<std::string> decode_cases() {
    std::vector<std::string> v;
    for (int r = 0; r < 2; ++r) {
        std::string R = "rank" + std::to_string(r);
        v.push_back(R + "-000325");
        for (int s = 4; s < 8; ++s) v.push_back(R + "-seq" + std::to_string(s) + "-decode");
    }
    return v;
}

}  // namespace kda_harness
