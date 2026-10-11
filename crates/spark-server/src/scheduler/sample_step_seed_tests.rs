// SPDX-License-Identifier: AGPL-3.0-only
//! Q3: the first-token sample routes must carry the request seed into
//! the sampler. These go through the param builders the routes use
//! ([`neutral_sampling_params`] for `sample_token` / `sample_first_token`,
//! [`grammar_sample_params`] for the grammar / MTP-bootstrap tail), so a
//! builder that drops the seed — as the grammar tail's hardcoded
//! `seed: None` did — fails here.
use super::{grammar_sample_params, neutral_sampling_params};
use crate::scheduler::logit_processors::SamplingLevers;
use spark_runtime::sampler::sample_with_params;

/// Near-uniform logits so a T=1 draw is genuinely stochastic.
fn flat_logits() -> Vec<u8> {
    (0..1000)
        .map(|i| (i as f32 * 0.01).sin())
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn levers(mtp_minp: bool) -> SamplingLevers {
    SamplingLevers {
        mtp_minp,
        ..Default::default()
    }
}

/// The grammar first-token path: `sample_first_token` builds its
/// penalties with [`neutral_sampling_params`], and `sample_token_with_grammar`
/// samples its stochastic tail with [`grammar_sample_params`].
fn grammar_first_token_params(seed: Option<u64>) -> spark_runtime::sampler::SamplingParams {
    let penalties = neutral_sampling_params(1.0, 0, 1.0, 0.0, seed);
    grammar_sample_params(1.0, 0, 1.0, &penalties, &levers(true))
}

#[test]
fn neutral_params_carry_seed() {
    assert_eq!(
        neutral_sampling_params(1.0, 0, 1.0, 0.0, Some(42)).seed,
        Some(42)
    );
    assert_eq!(neutral_sampling_params(1.0, 0, 1.0, 0.0, None).seed, None);
}

/// The HIGH finding: the grammar tail must take the seed from
/// `penalties` (old code: hardcoded `None`), and a `None` seed — the MTP
/// bootstrap's `penalty_params_for(Verify, .., None)` — stays `None`.
#[test]
fn grammar_params_take_seed_and_min_p_from_penalties() {
    let mut penalties = neutral_sampling_params(0.7, 20, 0.9, 0.05, Some(7));
    let p = grammar_sample_params(0.7, 20, 0.9, &penalties, &levers(true));
    assert_eq!(p.seed, Some(7));
    assert_eq!(
        (p.temperature, p.top_k, p.top_p, p.min_p),
        (0.7, 20, 0.9, 0.05)
    );
    // ATLAS_NO_MTP_MINP kill-switch still zeroes min_p, seed untouched.
    let p = grammar_sample_params(0.7, 20, 0.9, &penalties, &levers(false));
    assert_eq!((p.min_p, p.seed), (0.0, Some(7)));
    penalties.seed = None;
    assert_eq!(
        grammar_sample_params(0.7, 20, 0.9, &penalties, &levers(true)).seed,
        None
    );
}

/// End to end through the grammar builders and the real sampler: the
/// same seed reproduces the same first token. Compared over 32 seeds so
/// an unseeded (thread_rng) build fails with overwhelming probability
/// rather than passing on a lucky single-token collision.
#[test]
fn same_seed_reproduces_grammar_first_token() {
    let logits = flat_logits();
    let draw = |s: u64| sample_with_params(&logits, &grammar_first_token_params(Some(s)));
    let a: Vec<u32> = (0..32).map(draw).collect();
    let b: Vec<u32> = (0..32).map(draw).collect();
    assert_eq!(a, b, "same seed must reproduce the same first token");
}

/// The seed actually reaches the RNG: across many seeds the draws are
/// not all one token. (Any single pair of seeds may legitimately
/// collide, so no pairwise inequality is asserted.)
#[test]
fn seeds_vary_the_grammar_first_token() {
    let logits = flat_logits();
    let distinct: std::collections::HashSet<u32> = (0..64)
        .map(|s| sample_with_params(&logits, &grammar_first_token_params(Some(s))))
        .collect();
    assert!(distinct.len() > 1, "64 seeds all drew {distinct:?}");
}

/// Draw seeds for a sequence: the first token (raw request seed, as
/// `sample_first_token` gets it) followed by `n` decode draws exactly as
/// `process_seq_logits` derives them, with each draw recorded in
/// `output_tokens` the way `emit_token` does.
fn draw_seeds(seed: u64, first_token_in_output: bool, n: usize) -> Vec<u64> {
    use super::{decode_rng_base, rng_step_seed};
    let base = decode_rng_base(Some(seed), first_token_in_output);
    // Decode draw j sees `output_tokens.len() == recorded + j`.
    let recorded = usize::from(first_token_in_output);
    std::iter::once(seed)
        .chain((recorded..recorded + n).map(|len| rng_step_seed(base, len).unwrap()))
        .collect()
}

/// Normal admission (first token in `output_tokens`): draw k uses
/// `seed + k`, byte-identical to the pre-Q3 decode seeds.
#[test]
fn draw_index_counts_recorded_first_token() {
    assert_eq!(draw_seeds(100, true, 4), vec![100, 101, 102, 103, 104]);
}

/// Spontaneous `<think>` / `max_tokens == 0` admission: the first token
/// is sampled but not recorded. Old code seeded the second draw with
/// `seed + output_tokens.len() == seed + 0`, repeating the first draw.
#[test]
fn draw_index_counts_unrecorded_first_token() {
    assert_eq!(draw_seeds(100, false, 4), vec![100, 101, 102, 103, 104]);
}

#[test]
fn rng_helpers_preserve_unseeded_and_wrap() {
    use super::{decode_rng_base, rng_step_seed};
    assert_eq!(decode_rng_base(None, false), None);
    assert_eq!(rng_step_seed(None, 7), None);
    assert_eq!(decode_rng_base(Some(u64::MAX), false), Some(0));
    assert_eq!(rng_step_seed(Some(u64::MAX), 2), Some(1));
}
