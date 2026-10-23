// SPDX-License-Identifier: AGPL-3.0-only
//
// Core sampling pipeline (`sample_with_params_history` + seeded
// implementation). Split out of `sampler.rs` to keep the parent file
// under the 500-line cap. The parent re-exports these via `pub use`.

use super::{SamplingParams, apply_penalties_and_bias, argmax_first_wins_f32, record_entropy};

pub fn sample_with_params_history(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
) -> u32 {
    sample_with_params_seeded(data, params, token_history, params.seed)
}

/// The tokens a row leaves out: a vocabulary-split verify head gathers only
/// each rank's best logits, so the rest of a row is absent (NaN in the row).
/// Every absent token comes after `(bound, bound_id)` in the sampler's order
/// (logit descending, then the lower id), and `log_mass` is
/// `ln Σ_absent exp(logit / temperature)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Absent {
    pub bound: f32,
    pub bound_id: u32,
    pub log_mass: f32,
}

impl Absent {
    /// Whether token `id` at `logit` precedes every absent token. On a scaled
    /// row (logit / temperature, BF16 logits keep their order) `bound` is
    /// scaled the same way.
    fn beaten_by(&self, bound: f32, logit: f32, id: u32) -> bool {
        logit > bound || (logit == bound && id <= self.bound_id)
    }
}

/// Core sampling pipeline with explicit seed control.
/// `seed` overrides the RNG for deterministic sampling. None = thread_rng.
pub fn sample_with_params_seeded(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> u32 {
    // A whole row always decides the pick.
    sample_core(data, params, token_history, seed).unwrap_or(0)
}

/// [`sample_with_params_seeded`] over a row of `vocab` tokens of which only
/// `present` (id, logit after the pipeline) are known: the pick is the one
/// the full row would give for the same draw, or `None` when the present
/// tokens do not decide it: a top-k, min-p or top-p cut that could reach an
/// absent token, or a draw over the whole vocabulary. The answer depends on
/// the row alone and comes before the draw, so a caller that then samples the
/// full row samples the same distribution. Penalties and bias must be
/// neutral (the caller's pipeline applied them).
pub fn sample_with_absent(
    present: &[(u32, f32)],
    vocab: usize,
    params: &SamplingParams,
    seed: Option<u64>,
    absent: Absent,
) -> Option<u32> {
    let first = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if params.temperature <= 0.0 {
        let &(id, v) = present.iter().min_by(|a, b| first(a, b))?;
        return absent.beaten_by(absent.bound, v, id).then_some(id);
    }
    // The full row's top-n-sigma filter ran unless a mask made its sum -inf
    // (absent tokens are finite there); by Cantelli it keeps at least the
    // best `vocab·k²/(1+k²)` logits (k = top_n_sigma), so `draw` checks that
    // every token the pick depends on is among those.
    let sigma_kept = (params.top_n_sigma > 0.0
        && !present.iter().any(|&(_, v)| v == f32::NEG_INFINITY))
    .then(|| {
        let k2 = (params.top_n_sigma as f64).powi(2);
        (vocab as f64 * k2 / (1.0 + k2)).floor() as usize
    });
    let logits: Vec<(u32, f32)> = present
        .iter()
        .filter(|(_, v)| v.is_finite())
        .map(|&(i, v)| (i, v / params.temperature))
        .collect();
    if logits.is_empty() {
        return None;
    }
    draw(logits, params, seed, Some((absent, sigma_kept)))
}

fn sample_core(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> Option<u32> {
    let n = data.len() / 4;
    let top_n_sigma = params.top_n_sigma;

    // Read raw logits into a mutable vec for in-place modifications.
    // Penalties (repetition / presence / frequency / LZ / DRY) and
    // logit_bias are applied to `raw_logits` BEFORE the greedy bypass
    // below, so they take effect even at `temperature == 0.0`. Atlas
    // previously short-circuited to `argmax(raw_logits)` for greedy,
    // silently dropping caller-configured penalties — the 2026-05-01
    // sweep showed this caused Gemma-4-31B's haiku to enter a
    // repetition loop ("la... la... laaaL!") even with the model's
    // configured `repetition_penalty=1.1` because the harness uses
    // `temperature=0`. HF Transformers, vLLM, and llama.cpp all run
    // LogitsProcessor (penalties + bias) before greedy argmax — Atlas
    // is the outlier here.
    // Chunked conversion instead of per-element indexed `read_f32`: the
    // indexed form does four bounds-checked byte reads per element through a
    // shared counter, which the compiler lowers poorly; `chunks_exact` is the
    // canonical vectorisable shape. Same bytes, same values.
    let mut raw_logits: Vec<f32> = data
        .chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // ── 0. Penalties (repetition / presence / frequency / LZ / DRY) +
    //       logit bias, applied in place via the shared SSOT helper.
    // Identical behavior to the previous inline block; the same helper is
    // now also invoked on the MTP verify + bootstrap paths so all three
    // emit/verify sites apply the same penalties+bias+history.
    apply_penalties_and_bias(&mut raw_logits, params, token_history);

    // ── Greedy bypass (post-penalty argmax) ──
    // At `temperature == 0.0` we return argmax of the penalty/bias-modified
    // logits. We bypass top_n_sigma, temperature scaling, top_k, top_p,
    // min_p — all of those either filter (set values to -inf) or apply
    // monotonic transforms, neither of which can re-order the maximum. Only
    // penalties + logit_bias actually re-order logits, so as long as those
    // ran first, this argmax is correct AND respects caller config.
    //
    // Exact ties resolve to the LOWEST index, as on every other greedy pick
    // in the engine (the GPU `argmax_bf16` kernels and the verify path):
    // quantized logits tie often, so one rule is what lets a speculative
    // verify commit exactly the tokens this pick would.
    if params.temperature <= 0.0 {
        return Some(argmax_first_wins_f32(&raw_logits));
    }
    let temperature = params.temperature;

    // ── 1. Top-n-sigma: filter noise in logit space (temperature-invariant) ──
    // Keep tokens with logit >= mean - n*sigma. Filters NVFP4 quantization noise.
    if top_n_sigma > 0.0 {
        let sum: f32 = raw_logits.iter().sum();
        let mean = sum / n as f32;
        let var: f32 = raw_logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let sigma = var.sqrt();
        if sigma > 0.0 {
            let threshold = mean - top_n_sigma * sigma;
            for logit in raw_logits.iter_mut() {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    // ── 2. Temperature scaling ──
    let logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite()) // Skip -inf tokens from top-n-sigma
        .map(|(i, v)| (i as u32, v / temperature))
        .collect();

    if logits.is_empty() {
        // Fallback: if top-n-sigma filtered everything, use argmax of original
        return Some(argmax_first_wins_f32(&raw_logits));
    }
    draw(logits, params, seed, None)
}

/// Steps 3 on of the sampler over the scaled, finite `logits`. With
/// `absent` (and top-n-sigma's guaranteed keep, when its filter ran on the
/// full row), `None` when the present tokens do not decide the pick.
fn draw(
    mut logits: Vec<(u32, f32)>,
    params: &SamplingParams,
    seed: Option<u64>,
    absent: Option<(Absent, Option<usize>)>,
) -> Option<u32> {
    let (top_k, top_p, min_p) = (params.top_k as usize, params.top_p, params.min_p);
    let sigma_kept = absent.and_then(|(_, kept)| kept);
    let absent = absent.map(|(a, _)| a);
    // Every absent token's scaled logit is at most `bound_scaled` (division
    // by a positive temperature is monotone; it keeps distinct BF16 logits
    // distinct, so the order is the raw one).
    let bound_scaled = absent.map(|a| a.bound / params.temperature);
    let beats_absent = |logit: f32, id: u32| {
        absent.is_none_or(|a| a.beaten_by(bound_scaled.unwrap_or(a.bound), logit, id))
    };

    // ── 3. Rank-dependent filtering (top-k / min-p / top-p) ──
    // These need descending order — but a full O(n·log n) sort of the whole
    // ~248K vocab every token is wasteful when top_k caps survivors at a small
    // k (the model default here is top_k=20; a full sort was ~2.3ms/tok). Use
    // an O(n) quickselect to isolate the top-k, then sort only those k. Pure
    // temperature sampling (no top_k/top_p/min_p) needs no ordering at all —
    // the multinomial draw below is order-independent — so skip sorting.
    //
    // The order is total: value descending, then the lower index, so exact
    // ties (frequent in BF16 logits) fall the same way on every run and on a
    // split verify head's candidates as on the full row.
    let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    // Whether absent tokens are still candidates (no top-k / min-p cut yet).
    let mut open = absent.is_some();
    let sorted = if top_k > 0 && top_k < logits.len() {
        // Quickselect the top-k into [0, top_k) (O(n)), drop the rest, then
        // sort just the k survivors (O(k·log k)). Identical result to
        // full-sort-then-truncate; min-p/top-p below run over these k.
        logits.select_nth_unstable_by(top_k, cmp_desc);
        logits.truncate(top_k);
        logits.sort_unstable_by(cmp_desc);
        // The k present survivors are the row's top k when the last beats
        // every absent token.
        let (id, logit) = logits[top_k - 1];
        if !beats_absent(logit, id) {
            return None;
        }
        open = false;
        true
    } else if open {
        // A top-k past the present tokens, or none with no min-p/top-p:
        // absent tokens stay in the draw.
        if top_k > 0 || (min_p <= 0.0 && top_p >= 1.0) {
            return None;
        }
        logits.sort_unstable_by(cmp_desc);
        true
    } else if min_p > 0.0 || top_p < 1.0 {
        // No top-k cap, but min-p/top-p still need full descending order.
        logits.sort_unstable_by(cmp_desc);
        true
    } else {
        false
    };

    // ── 4. Softmax ──
    // `sorted` ⇒ logits[0] is the max; otherwise reduce for it. min_p/top_p
    // below are only reachable when `sorted` is true, so their reliance on
    // descending order still holds.
    let max_val = if sorted {
        logits[0].1
    } else {
        logits
            .iter()
            .map(|&(_, v)| v)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_val).exp()))
        .collect();
    // An absent token's probability is at most this.
    let absent_prob = bound_scaled.map(|b| (b - max_val).exp());

    // ── 4b. Entropy: H = -Σ p·ln(p) over the post-softmax distribution ──
    // (a row with absent tokens has no whole distribution to report)
    if absent.is_none() {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let h: f32 = probs
                .iter()
                .map(|&(_, w)| {
                    let p = w * inv;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            record_entropy(h);
        }
    }

    // ── 5. Min-p: keep tokens with prob >= min_p * max_prob ──
    if min_p > 0.0 {
        let max_prob = probs[0].1; // Already sorted descending
        let threshold = min_p * max_prob;
        if open {
            // Every absent token must fall below the floor.
            if absent_prob.is_some_and(|p| p >= threshold) {
                return None;
            }
            open = false;
        }
        probs.retain(|p| p.1 >= threshold);
    }
    // The tokens the top-p cut and the draw see must all survive the full
    // row's top-n-sigma filter.
    if sigma_kept.is_some_and(|kept| open || probs.len() > kept) {
        return None;
    }

    // ── 6. Top-p (nucleus) ──
    if top_p < 1.0 {
        let tail = match absent {
            Some(a) if open => (a.log_mass - max_val).exp(),
            _ => 0.0,
        };
        let sum: f32 = probs.iter().map(|p| p.1).sum::<f32>() + tail;
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        let mut cut = false;
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob / sum;
            if cumsum >= top_p {
                cutoff = i + 1;
                cut = true;
                break;
            }
        }
        // With absent tokens still in, the nucleus must close on a present
        // token that beats every absent one (`probs` is a prefix of `logits`).
        if open && !(cut && beats_absent(logits[cutoff - 1].1, logits[cutoff - 1].0)) {
            return None;
        }
        probs.truncate(cutoff);
    }

    // Multinomial sample from the filtered distribution.
    let sum: f32 = probs.iter().map(|p| p.1).sum();
    let random_val: f32 = if let Some(s) = seed {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        rng.r#gen::<f32>()
    } else {
        rand::random::<f32>()
    };
    let threshold = random_val * sum;
    let mut cumsum = 0.0f32;
    for &(idx, prob) in &probs {
        cumsum += prob;
        if cumsum >= threshold {
            return Some(idx);
        }
    }
    Some(probs.last().map_or(0, |p| p.0))
}
