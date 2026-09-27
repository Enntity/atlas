// SPDX-License-Identifier: AGPL-3.0-only

// DFlash2 drafter kernels: grouped dynamic in-block conv, per-row top-K over
// BF16 logits, and the greedy candidate-selector walk.
//
// Row convention (all three kernels): the γ-block of one sequence is `gamma`
// consecutive rows; row t sits at in-block position p = t % block_size.
// Row 0 of a block is the anchor (bonus token); rows 1..gamma-1 are the mask
// rows whose candidates the selector walk chooses between.
//
// Arithmetic is FP32 with a single BF16 rounding on store. The drafter only
// affects acceptance, so bit-parity with the PyTorch reference (which rounds
// after every BF16 op) is not required.

#include <cuda_bf16.h>
#include <math.h>

#define DFLASH2_MAX_TAPS 4
#define DFLASH2_TOPK 16
#define DFLASH2_TOPK_THREADS 256
#define DFLASH2_MAX_RANK 1024

// In-place grouped dynamic convolution along the block (row) axis.
//
//   c[t, tap, col] = base[tap, col] + delta[t, delta_offset + tap * groups + col / group_size]
//   x'[t, col]     = sum_{tap < taps, tap <= p(t)} c[t, tap, col] * x[t - tap, col]
//
// x:      [rows, hidden] BF16, overwritten with the conv output.
// delta:  [rows, delta_stride] BF16 kernel_projection output; one side of the
//         `[2 sides, taps, groups]` row layout starts at `delta_offset`.
// base:   [taps, hidden] BF16 (base_kernel already offset to the side).
// Taps never reach across a block boundary (p(t) >= tap), so rows of
// different sequences packed back-to-back stay independent.
//
// Each thread owns one column and walks the rows in order, holding the
// ORIGINAL previous rows in registers — that is what makes in-place safe.
//
// Grid: (ceil(hidden / 256), 1, 1)  Block: (256, 1, 1)
// Requires taps <= DFLASH2_MAX_TAPS, hidden % group_size == 0.
extern "C" __global__ void dflash2_grouped_conv_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ delta,
    const __nv_bfloat16* __restrict__ base,
    unsigned int rows,
    unsigned int hidden,
    unsigned int group_size,
    unsigned int taps,
    unsigned int block_size,
    unsigned int delta_stride,
    unsigned int delta_offset
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    const unsigned int groups = hidden / group_size;
    const unsigned int group = col / group_size;

    float base_c[DFLASH2_MAX_TAPS];
    float history[DFLASH2_MAX_TAPS];  // history[k] = x[t - 1 - k, col]
#pragma unroll
    for (unsigned int k = 0; k < DFLASH2_MAX_TAPS; ++k) {
        base_c[k] = k < taps ? __bfloat162float(base[(size_t)k * hidden + col]) : 0.0f;
        history[k] = 0.0f;
    }

    for (unsigned int t = 0; t < rows; ++t) {
        const unsigned int position = t % block_size;
        const size_t index = (size_t)t * hidden + col;
        const float current = __bfloat162float(x[index]);
        const __nv_bfloat16* d = delta + (size_t)t * delta_stride + delta_offset + group;
        float acc = (base_c[0] + __bfloat162float(d[0])) * current;
#pragma unroll
        for (unsigned int k = 1; k < DFLASH2_MAX_TAPS; ++k) {
            if (k < taps && position >= k) {
                acc += (base_c[k] + __bfloat162float(d[(size_t)k * groups])) * history[k - 1];
            }
        }
#pragma unroll
        for (unsigned int k = DFLASH2_MAX_TAPS - 1; k > 0; --k) {
            history[k] = history[k - 1];
        }
        history[0] = current;
        x[index] = __float2bfloat16(acc);
    }
}

// Candidate ordering: larger value first, ties to the lower vocabulary id.
__device__ __forceinline__ bool dflash2_better(
    float value,
    unsigned int id,
    float other_value,
    unsigned int other_id
) {
    return value > other_value || (value == other_value && id < other_id);
}

// Per-row top-K (K = DFLASH2_TOPK) of BF16 logits, sorted descending, ties to
// the lower id. NaN and -inf entries are never selected; a row with fewer
// than K finite logits pads with (id 0, value -inf).
//
// logits:   [rows, vocab] BF16
// out_ids:  [rows, DFLASH2_TOPK] u32
// out_vals: [rows, DFLASH2_TOPK] f32 (the unary logit of each candidate)
//
// Phase 1: every thread keeps a sorted register top-K over its strided slice
// (visited in increasing id order, so strict `>` keeps the lower id on ties).
// Phase 2: K rounds of block-wide argmax over the per-thread list heads.
//
// Grid: (rows, 1, 1)  Block: (DFLASH2_TOPK_THREADS = 256, 1, 1)
extern "C" __global__ void dflash2_topk_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out_ids,
    float* __restrict__ out_vals,
    unsigned int vocab
) {
    __shared__ float s_val[DFLASH2_TOPK * DFLASH2_TOPK_THREADS];
    __shared__ unsigned int s_id[DFLASH2_TOPK * DFLASH2_TOPK_THREADS];
    __shared__ float w_val[DFLASH2_TOPK_THREADS / 32];
    __shared__ unsigned int w_id[DFLASH2_TOPK_THREADS / 32];
    __shared__ unsigned int w_tid[DFLASH2_TOPK_THREADS / 32];
    __shared__ unsigned int winner_tid;

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const __nv_bfloat16* src = logits + (size_t)row * vocab;

    float vals[DFLASH2_TOPK];
    unsigned int ids[DFLASH2_TOPK];
#pragma unroll
    for (unsigned int k = 0; k < DFLASH2_TOPK; ++k) {
        vals[k] = -INFINITY;
        ids[k] = 0xFFFFFFFFu;
    }
    for (unsigned int i = tid; i < vocab; i += DFLASH2_TOPK_THREADS) {
        const float v = __bfloat162float(src[i]);
        if (v > vals[DFLASH2_TOPK - 1]) {
            vals[DFLASH2_TOPK - 1] = v;
            ids[DFLASH2_TOPK - 1] = i;
#pragma unroll
            for (unsigned int k = DFLASH2_TOPK - 1; k > 0; --k) {
                if (vals[k] > vals[k - 1]) {
                    const float tv = vals[k];
                    vals[k] = vals[k - 1];
                    vals[k - 1] = tv;
                    const unsigned int ti = ids[k];
                    ids[k] = ids[k - 1];
                    ids[k - 1] = ti;
                }
            }
        }
    }
#pragma unroll
    for (unsigned int k = 0; k < DFLASH2_TOPK; ++k) {
        s_val[k * DFLASH2_TOPK_THREADS + tid] = vals[k];
        s_id[k * DFLASH2_TOPK_THREADS + tid] = ids[k];
    }
    unsigned int head = 0;
    __syncthreads();

    for (unsigned int out = 0; out < DFLASH2_TOPK; ++out) {
        float v = head < DFLASH2_TOPK ? s_val[head * DFLASH2_TOPK_THREADS + tid] : -INFINITY;
        unsigned int id = head < DFLASH2_TOPK ? s_id[head * DFLASH2_TOPK_THREADS + tid] : 0xFFFFFFFFu;
        unsigned int owner = tid;
#pragma unroll
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, v, offset);
            const unsigned int oi = __shfl_down_sync(0xFFFFFFFFu, id, offset);
            const unsigned int ot = __shfl_down_sync(0xFFFFFFFFu, owner, offset);
            if (dflash2_better(ov, oi, v, id)) {
                v = ov;
                id = oi;
                owner = ot;
            }
        }
        if (lane == 0) {
            w_val[warp] = v;
            w_id[warp] = id;
            w_tid[warp] = owner;
        }
        __syncthreads();
        if (warp == 0) {
            const bool live = lane < DFLASH2_TOPK_THREADS / 32;
            v = live ? w_val[lane] : -INFINITY;
            id = live ? w_id[lane] : 0xFFFFFFFFu;
            owner = live ? w_tid[lane] : 0xFFFFFFFFu;
#pragma unroll
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, v, offset);
                const unsigned int oi = __shfl_down_sync(0xFFFFFFFFu, id, offset);
                const unsigned int ot = __shfl_down_sync(0xFFFFFFFFu, owner, offset);
                if (dflash2_better(ov, oi, v, id)) {
                    v = ov;
                    id = oi;
                    owner = ot;
                }
            }
            if (lane == 0) {
                const size_t dst = (size_t)row * DFLASH2_TOPK + out;
                out_ids[dst] = id < vocab ? id : 0u;
                out_vals[dst] = id < vocab ? v : -INFINITY;
                winner_tid = owner;
            }
        }
        __syncthreads();
        if (tid == winner_tid) {
            ++head;
        }
        // No trailing barrier needed: winner_tid / w_* are rewritten only
        // after the next round's first __syncthreads(), which every thread
        // reaches after reading winner_tid here.
    }
}

// Greedy (temperature 0) candidate-selector walk, one block per sequence.
//
// For mask depth l in 0..gamma-1 (block row 1 + l), candidate c < K:
//   score[l][c] = unary[l][c]
//               + sum_r pred[prev_token][r] * hidden[l][r] * succ[cand[l][c]][r]
// where prev_token is the anchor for l = 0 and the token chosen at depth
// l - 1 afterwards; the argmax candidate (ties to the lower c) is the draft.
// Only the edges on the greedy path are scored.
//
// cand_ids:  [batch * gamma, K] u32   (dflash2_topk_bf16 output, all rows)
// cand_vals: [batch * gamma, K] f32
// hidden:    [batch * gamma, rank] BF16 hidden_projection of final-normed rows
// pred/succ: [vocab, rank] BF16 predecessor / successor codebooks
// anchors:   [batch] u32 block anchor (bonus) token
// ban_depth: [batch] u32: depths 1..ban_depth[b] never draft end0..end3 (the
//            target may not end the turn there under min_tokens); may be null
// tokens:    [batch * gamma] u32 out: row 0 = top-1 of the anchor row (plain
//            argmax, matching the non-selector path), rows 1.. = walk.
//
// Grid: (batch, 1, 1)  Block: (32 * DFLASH2_TOPK = 512, 1, 1) — warp c scores
// candidate c. Requires rank <= DFLASH2_MAX_RANK.
extern "C" __global__ void dflash2_selector_walk(
    const unsigned int* __restrict__ cand_ids,
    const float* __restrict__ cand_vals,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pred,
    const __nv_bfloat16* __restrict__ succ,
    const unsigned int* __restrict__ anchors,
    const unsigned int* __restrict__ ban_depth,
    unsigned int* __restrict__ tokens,
    unsigned int gamma,
    unsigned int rank,
    unsigned int vocab,
    unsigned int end0,
    unsigned int end1,
    unsigned int end2,
    unsigned int end3
) {
    __shared__ float s_query[DFLASH2_MAX_RANK];
    __shared__ float s_score[DFLASH2_TOPK];
    __shared__ unsigned int s_prev;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const size_t first_row = (size_t)blockIdx.x * gamma;
    const unsigned int banned_to = ban_depth ? ban_depth[blockIdx.x] : 0u;

    if (tid == 0) {
        tokens[first_row] = cand_ids[first_row * DFLASH2_TOPK];
        const unsigned int anchor = anchors[blockIdx.x];
        s_prev = anchor < vocab ? anchor : 0u;
    }
    __syncthreads();

    for (unsigned int depth = 1; depth < gamma; ++depth) {
        const size_t row = first_row + depth;
        const __nv_bfloat16* p_row = pred + (size_t)s_prev * rank;
        const __nv_bfloat16* h_row = hidden + row * rank;
        for (unsigned int r = tid; r < rank; r += blockDim.x) {
            s_query[r] = __bfloat162float(p_row[r]) * __bfloat162float(h_row[r]);
        }
        __syncthreads();
        if (warp < DFLASH2_TOPK) {
            const unsigned int cand = cand_ids[row * DFLASH2_TOPK + warp];
            const __nv_bfloat16* s_row = succ + (size_t)(cand < vocab ? cand : 0u) * rank;
            float acc = 0.0f;
            for (unsigned int r = lane; r < rank; r += 32) {
                acc += s_query[r] * __bfloat162float(s_row[r]);
            }
#pragma unroll
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_xor_sync(0xFFFFFFFFu, acc, offset);
            }
            if (lane == 0) {
                const bool end = cand == end0 || cand == end1 || cand == end2 || cand == end3;
                s_score[warp] = depth <= banned_to && end
                    ? -INFINITY
                    : cand_vals[row * DFLASH2_TOPK + warp] + acc;
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int best = 0;
            for (unsigned int c = 1; c < DFLASH2_TOPK; ++c) {
                if (s_score[c] > s_score[best]) best = c;
            }
            const unsigned int token = cand_ids[row * DFLASH2_TOPK + best];
            tokens[row] = token;
            s_prev = token < vocab ? token : 0u;
        }
        __syncthreads();
    }
}
