// SPDX-License-Identifier: AGPL-3.0-only

//! Per-choice RNG seeds for `n > 1` requests.
//!
//! vLLM gives every child of a parallel-sampling request its own generator
//! seeded with `seed + index` (`vllm/v1/engine/parallel_sampling.py:80`), and
//! each generator then advances independently. Atlas has no per-request
//! generator object: it reseeds every draw with `base + draw_index`
//! (`scheduler::sample_step::rng_step_seed`). With `base = seed + i` choice
//! `i`'s draw `k` reused choice 0's draw `k + i` — the choices were the same
//! random stream shifted by `i` tokens. Choice `i > 0` therefore gets a
//! splitmix64-mixed base, so the per-choice draw windows `base..base + N` are
//! disjoint except with probability ~`n² N / 2^64`.
//!
//! Deliberately free of `crate::` imports (pure, unit-testable standalone).

/// splitmix64 increment (2^64 / φ).
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The splitmix64 output function (Steele, Lea & Flood 2014): a bijective
/// avalanche mix of one 64-bit word.
fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// RNG base seed for choice `index` of a request seeded with `seed`.
///
/// Choice 0 keeps the request seed, so `n = 1` (and every streaming request,
/// which is always `n = 1`) samples exactly as before and choice 0 of an
/// `n > 1` request reproduces the single-choice output for the same seed,
/// as in vLLM. Choice `i > 0` is the `i`-th splitmix64 output of a stream
/// seeded with `seed`: distinct for distinct `i` (the mix is a bijection) and
/// far from `seed` and from each other.
pub(crate) fn choice_seed(seed: u64, index: u64) -> u64 {
    if index == 0 {
        return seed;
    }
    splitmix64_mix(seed.wrapping_add(index.wrapping_mul(GOLDEN_GAMMA)))
}

#[cfg(test)]
mod tests {
    use super::choice_seed;
    use std::collections::HashSet;

    /// Draw `k` of a sequence with RNG base `base` — the scheduler's
    /// `rng_step_seed(Some(base), k)`.
    fn draw(base: u64, k: u64) -> u64 {
        base.wrapping_add(k)
    }

    /// The defect: with `seed + i`, choice 1's first draw is choice 0's
    /// second draw.
    #[test]
    fn offset_seeds_collide_but_choice_seeds_do_not() {
        let seed = 42;
        assert_eq!(draw(seed + 1, 0), draw(seed, 1), "old seed + i scheme");
        assert_ne!(draw(choice_seed(seed, 1), 0), draw(choice_seed(seed, 0), 1));
    }

    /// No draw seed repeats across the first N draws of n choices, for
    /// several request seeds including the wrap-around edges.
    #[test]
    fn no_collisions_among_the_first_draws_of_every_choice() {
        const CHOICES: u64 = 16;
        const DRAWS: u64 = 4096;
        for seed in [
            0,
            1,
            42,
            1234,
            u64::MAX,
            u64::MAX - 7,
            0x9E37_79B9_7F4A_7C15,
        ] {
            let mut seen = HashSet::new();
            for i in 0..CHOICES {
                let base = choice_seed(seed, i);
                for k in 0..DRAWS {
                    assert!(
                        seen.insert(draw(base, k)),
                        "seed {seed}: choice {i} draw {k} collides"
                    );
                }
            }
        }
    }

    /// Choice 0 is the request seed (n = 1 and streaming are unchanged), and
    /// the derivation is deterministic.
    #[test]
    fn choice_zero_keeps_the_request_seed() {
        for seed in [0, 7, u64::MAX] {
            assert_eq!(choice_seed(seed, 0), seed);
            assert_eq!(choice_seed(seed, 3), choice_seed(seed, 3));
            assert_ne!(choice_seed(seed, 1), choice_seed(seed, 2));
        }
    }
}
