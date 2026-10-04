// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check of the strict structured-output TP2 verify head
// (ATLAS_GLM_STRICT_SPEC): each rank's argmax_bf16_value_ban_allow over its
// half of a GLM-5.3 row (154,880 tokens, two 77,440-token shards) under its
// slice of the row's grammar bitmask, then argmax_pair_merge_ban, must pick
// the best ALLOWED token of the whole row, value first then the lower index,
// exactly as a single-GPU masked argmax would; with the min_tokens ban on
// banned rows. Both ranks are emulated on one GPU and both merge views (rank
// 0 and rank 1) must agree. Also: under an all-ones mask the masked kernel
// returns argmax_bf16_value_ban's pairs bit for bit.
//
// Values are multiples of 1/8 in [-12.5, 12.5), so ties are common and the
// tie rule is exercised. Row shapes: random density, all ones, a single
// allowed token (in either shard), allowed tokens in one shard only, nothing
// allowed (both sentinels: token 0), and ban rows whose best allowed token is
// banned.
//
//   nvcc -arch=sm_121a -O3 -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_strict_split_argmax_bitcheck.cu -o glm_strict_split_argmax_bitcheck
//   ./glm_strict_split_argmax_bitcheck [cases=64]
//
// Prints one PASS line, or the first mismatches and exits 1.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include "argmax_bf16.cu"

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(2);                                                             \
        }                                                                        \
    } while (0)

static const unsigned V = 154880, SHARD = V / 2, W = V / 32, ROWS = 9;
static const unsigned BAN[4] = {154820u, 154827u, 0xFFFFFFFFu, 0xFFFFFFFFu};

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static unsigned next_u32() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (unsigned)(rng >> 32);
}

static bool bit(const std::vector<unsigned>& m, unsigned r, unsigned t) {
    return (m[(size_t)r * W + t / 32] >> (t % 32)) & 1u;
}

// Single-GPU reference: best allowed (and unbanned on ban rows) token.
static unsigned reference(const std::vector<float>& v, const std::vector<unsigned>& m,
                          unsigned r, bool banned_row) {
    float best = -1e30f;
    unsigned idx = 0;
    for (unsigned t = 0; t < V; ++t) {
        if (!bit(m, r, t)) continue;
        if (banned_row && (t == BAN[0] || t == BAN[1])) continue;
        const float x = v[(size_t)r * V + t];
        if (x > best || (x == best && t < idx)) {
            best = x;
            idx = t;
        }
    }
    return idx;
}

// Shard-local ban index, as glm_vocab_split's `local_id` computes it.
static unsigned local_ban(unsigned id, unsigned start) {
    return (id >= start && id - start < SHARD) ? id - start : 0xFFFFFFFFu;
}

int main(int argc, char** argv) {
    const int cases = argc > 1 ? atoi(argv[1]) : 64;
    std::vector<float> vals((size_t)ROWS * V);
    std::vector<__nv_bfloat16> logits((size_t)ROWS * V);
    std::vector<unsigned> masks((size_t)ROWS * W), ones((size_t)ROWS * W, 0xFFFFFFFFu);
    __nv_bfloat16* d_logits;
    unsigned *d_masks, *d_ones, *d_pairs, *d_ref_pairs, *d_out;
    CK(cudaMalloc(&d_logits, logits.size() * 2));
    CK(cudaMalloc(&d_masks, masks.size() * 4));
    CK(cudaMalloc(&d_ones, ones.size() * 4));
    // Masked pairs of rank 0 and 1, then their all-ones-mask pairs.
    CK(cudaMalloc(&d_pairs, 4 * ROWS * 4 * 4));
    CK(cudaMalloc(&d_ref_pairs, 2 * ROWS * 4 * 4));
    CK(cudaMalloc(&d_out, 2 * ROWS * 4));
    CK(cudaMemcpy(d_ones, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
    long checked = 0, bad = 0;
    for (int c = 0; c < cases; ++c) {
        for (size_t i = 0; i < vals.size(); ++i) {
            vals[i] = (float)((int)(next_u32() % 200) - 100) / 8.0f;
            logits[i] = __float2bfloat16(vals[i]);
            vals[i] = __bfloat162float(logits[i]);
        }
        for (unsigned r = 0; r < ROWS; ++r) {
            unsigned* row = &masks[(size_t)r * W];
            const unsigned shape = (r + c) % 7;
            for (unsigned w = 0; w < W; ++w) {
                unsigned x = next_u32() & next_u32();
                if (shape == 1) x = 0xFFFFFFFFu;
                if (shape == 2 || shape == 3 || shape == 6) x = 0;
                if (shape == 4 && w >= W / 2) x = 0;
                if (shape == 5 && w < W / 2) x = 0;
                row[w] = x;
            }
            if (shape == 2 || shape == 3) {
                const unsigned t = (shape == 2 ? 0 : SHARD) + next_u32() % SHARD;
                row[t / 32] |= 1u << (t % 32);
            }
        }
        const bool ban_rows = c % 2 == 1;
        if (ban_rows) {
            // Make a banned token the best allowed one of rows 0..3.
            for (unsigned r = 0; r < 4; ++r) {
                const unsigned t = BAN[r % 2];
                masks[(size_t)r * W + t / 32] |= 1u << (t % 32);
                logits[(size_t)r * V + t] = __float2bfloat16(20.0f);
                vals[(size_t)r * V + t] = 20.0f;
            }
        }
        const unsigned ban_lo = ban_rows ? 0x0Fu : 0u;
        CK(cudaMemcpy(d_logits, logits.data(), logits.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_masks, masks.data(), masks.size() * 4, cudaMemcpyHostToDevice));
        for (unsigned rank = 0; rank < 2; ++rank) {
            const unsigned start = rank * SHARD;
            const unsigned b0 = local_ban(BAN[0], start), b1 = local_ban(BAN[1], start);
            argmax_bf16_value_ban_allow<<<ROWS, 1024>>>(
                d_logits + start, d_pairs + rank * ROWS * 4, SHARD, V, d_masks + start / 32, W,
                b0, b1, 0xFFFFFFFFu, 0xFFFFFFFFu);
            // All-ones mask vs the unmasked kernel: identical pairs.
            argmax_bf16_value_ban<<<ROWS, 1024>>>(
                d_logits + start, d_ref_pairs + rank * ROWS * 4, SHARD, V,
                b0, b1, 0xFFFFFFFFu, 0xFFFFFFFFu);
            argmax_bf16_value_ban_allow<<<ROWS, 1024>>>(
                d_logits + start, d_pairs + (2 + rank) * ROWS * 4, SHARD, V, d_ones + start / 32, W,
                b0, b1, 0xFFFFFFFFu, 0xFFFFFFFFu);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned> refp(2 * ROWS * 4);
        CK(cudaMemcpy(refp.data(), d_ref_pairs, refp.size() * 4, cudaMemcpyDeviceToHost));
        for (int rank = 0; rank < 2; ++rank) {
            argmax_pair_merge_ban<<<1, 32>>>(
                d_pairs + rank * ROWS * 4, d_pairs + (1 - rank) * ROWS * 4, d_out + rank * ROWS,
                ROWS, SHARD, rank, ban_lo, 0);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned> out(2 * ROWS);
        CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
        for (unsigned r = 0; r < ROWS; ++r) {
            const unsigned want = reference(vals, masks, r, (ban_lo >> r) & 1u);
            for (int rank = 0; rank < 2; ++rank) {
                ++checked;
                if (out[rank * ROWS + r] != want && bad++ < 10)
                    fprintf(stderr, "case %d row %u rank %d: got %u want %u\n", c, r, rank,
                            out[rank * ROWS + r], want);
            }
        }
        std::vector<unsigned> onesp(2 * ROWS * 4);
        CK(cudaMemcpy(onesp.data(), d_pairs + 2 * ROWS * 4, onesp.size() * 4,
                      cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < onesp.size(); ++i) {
            ++checked;
            if (onesp[i] != refp[i] && bad++ < 10)
                fprintf(stderr, "case %d ones-mask pair word %zu: %08x vs unmasked %08x\n", c,
                        i, onesp[i], refp[i]);
        }
    }
    if (bad) {
        printf("FAIL glm_strict_split_argmax_bitcheck: %ld of %ld checks\n", bad, checked);
        return 1;
    }
    printf("PASS glm_strict_split_argmax_bitcheck: %d cases x %u rows, %ld checks "
           "(masked TP2 argmax == single-GPU masked argmax; all-ones mask == unmasked)\n",
           cases, ROWS, checked);
    return 0;
}
