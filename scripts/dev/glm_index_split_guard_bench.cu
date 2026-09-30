// SPDX-License-Identifier: AGPL-3.0-only
// Standalone check of glm_index_clamp_ids, the range guard ATLAS_GLM_INDEX_SPLIT
// runs over the token-id rows a rank receives from its peer. Two received
// swaps of `rows` x `width` ids each, as one owner's exchange lands them:
//   (a) in-range ids ([-1, limit), both bounds present) keep every byte and
//       count nothing;
//   (b) planted ids >= limit and < -1 become -1, every other id keeps its
//       bytes, and the device counter holds exactly the planted count (a
//       second pass over the clamped rows counts nothing more);
//   (c) microseconds per launch, per CTA count (`*`: the engine's grid,
//       2048 ids a CTA).
//
//   nvcc -O3 -std=c++17 -arch=sm_121a -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_index_split_guard_bench.cu -o guard_bench
//   ./guard_bench [rows=2048] [width=2048] [limit=524288] [iters=200]
// Device memory: 2 x rows x width x 4 bytes (34 MB at the defaults).
#include "glm_indexer.cu"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char** argv) {
    const unsigned rows = argc > 1 ? atoi(argv[1]) : 2048;
    const unsigned width = argc > 2 ? atoi(argv[2]) : 2048;
    const int limit = argc > 3 ? atoi(argv[3]) : 524288;
    const int iters = argc > 4 ? atoi(argv[4]) : 200;
    const unsigned count = rows * width;  // ids per swap
    std::mt19937 rng(1234);

    // In-range rows as the selector writes them: positions below `limit`, -1
    // where nothing is selected, and both bounds.
    std::vector<int> valid((size_t)2 * count);
    std::uniform_int_distribution<int> position(0, limit - 1);
    for (auto& id : valid) id = rng() % 8 ? position(rng) : -1;
    valid[0] = -1;
    valid[1] = 0;
    valid[2] = limit - 1;
    valid.back() = limit - 1;

    // The same rows with out-of-range ids planted at distinct places.
    const int bad_values[] = {limit, limit + 1, INT_MAX, 0x40000000, -2, -3, INT_MIN, -0x40000000};
    std::vector<int> corrupt = valid;
    std::vector<size_t> at(valid.size());
    for (size_t i = 0; i < at.size(); i++) at[i] = i;
    std::shuffle(at.begin(), at.end(), rng);
    const size_t planted = std::min<size_t>(at.size() / 2, 100003);
    for (size_t k = 0; k < planted; k++) corrupt[at[k]] = bad_values[k % 8];
    std::vector<int> expect = corrupt;
    for (size_t k = 0; k < planted; k++) expect[at[k]] = -1;

    int* d_ids;
    unsigned* d_count;
    CK(cudaMalloc(&d_ids, valid.size() * 4));
    CK(cudaMalloc(&d_count, 4));
    // One launch per received swap, as IndexSplit::exchange issues them.
    auto guard = [&](unsigned ctas) {
        for (int swap = 0; swap < 2; swap++)
            glm_index_clamp_ids<<<ctas, 256>>>(d_ids + (size_t)swap * count, count, limit, d_count);
        CK(cudaGetLastError());
    };
    // Rows and counter after guarding `host` from a zeroed counter, `passes` times.
    auto run = [&](const std::vector<int>& host, unsigned ctas, int passes, std::vector<int>& out) {
        CK(cudaMemcpy(d_ids, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_count, 0, 4));
        for (int p = 0; p < passes; p++) guard(ctas);
        out.resize(host.size());
        unsigned counted;
        CK(cudaMemcpy(out.data(), d_ids, host.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&counted, d_count, 4, cudaMemcpyDeviceToHost));
        return counted;
    };

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("rows=%u width=%u ids/swap=%u limit=%d planted=%zu\n", rows, width, count, limit, planted);
    printf("%7s  %12s %10s %12s %10s %12s\n", "ctas", "valid bytes", "counted", "clamped rows", "counted", "us/launch");
    bool ok = true;
    const unsigned full = (count + 255) / 256;  // one id per thread
    const unsigned engine = (count + 2047) / 2048;
    for (unsigned ctas : {48u, 96u, 192u, 384u, 768u, engine, full}) {
        ctas = std::min(ctas, full);
        std::vector<int> out;
        const unsigned clean = run(valid, ctas, 1, out);
        const bool identical = !memcmp(out.data(), valid.data(), valid.size() * 4);
        const unsigned counted = run(corrupt, ctas, 2, out);
        const bool clamped = !memcmp(out.data(), expect.data(), expect.size() * 4);
        // Timing over valid rows: the production case, nothing written.
        CK(cudaMemcpy(d_ids, valid.data(), valid.size() * 4, cudaMemcpyHostToDevice));
        guard(ctas);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; i++) guard(ctas);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("%7u%c %12s %10u %12s %10u %12.1f\n", ctas, ctas == engine ? '*' : ' ',
               identical ? "IDENTICAL" : "DIFFER", clean, clamped ? "EXACT" : "WRONG", counted,
               ms * 1000.f / (2 * iters));
        ok = ok && identical && clean == 0 && clamped && counted == planted;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}
