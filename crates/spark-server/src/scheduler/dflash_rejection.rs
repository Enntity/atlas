// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness-first DFlash2 standard rejection sampler.
//!
//! Mia's GLM recipe keeps the selector's realized sparse top-16 logits and
//! verifies probabilistic drafts with the Leviathan probability-ratio test.
//! Atlas historically discarded q and compared drafts with target argmax,
//! which is only valid at temperature zero. This host implementation is the
//! executable reference for the native GPU kernel: it favors transparent
//! math over speed and is only used for grammarless, penalty-neutral requests.

use anyhow::{Result, ensure};
use spark_model::speculative::SparseDraftDistribution;
use spark_model::traits::Model;

use super::{ActiveSeq, bf16_to_f32};

pub(super) struct StandardRejectionVerdict {
    pub accepted: usize,
    pub bonus: u32,
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline]
fn uniform(seed: u64, position: usize, domain: u64) -> f64 {
    let bits = splitmix64(seed ^ (position as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ domain);
    (((bits >> 11) as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
}

fn target_distribution(
    bytes: &[u8],
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
) -> Vec<f64> {
    let mut logits = bytes
        .chunks_exact(2)
        .take(vocab)
        .map(|chunk| bf16_to_f32(chunk[0], chunk[1]))
        .collect::<Vec<_>>();

    if top_n_sigma > 0.0 {
        let mean = logits.iter().map(|&v| v as f64).sum::<f64>() / vocab as f64;
        let variance = logits
            .iter()
            .map(|&v| {
                let delta = v as f64 - mean;
                delta * delta
            })
            .sum::<f64>()
            / vocab as f64;
        let floor = mean - top_n_sigma as f64 * variance.sqrt();
        for value in &mut logits {
            if (*value as f64) < floor {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    let inverse_temperature = 1.0 / temperature.max(f32::MIN_POSITIVE) as f64;
    let mut ranked = logits
        .iter()
        .enumerate()
        .filter_map(|(token, &logit)| {
            logit
                .is_finite()
                .then_some((token as u32, logit as f64 * inverse_temperature))
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if top_k > 0 {
        ranked.truncate((top_k as usize).min(ranked.len()));
    }
    let max_logit = ranked.first().map_or(f64::NEG_INFINITY, |entry| entry.1);
    let mut weighted = ranked
        .into_iter()
        .map(|(token, logit)| (token, (logit - max_logit).exp()))
        .collect::<Vec<_>>();
    if min_p > 0.0
        && let Some(max_weight) = weighted.first().map(|entry| entry.1)
    {
        let threshold = min_p as f64 * max_weight;
        weighted.retain(|entry| entry.1 >= threshold);
    }
    if top_p < 1.0 {
        let total = weighted.iter().map(|entry| entry.1).sum::<f64>();
        let mut cumulative = 0.0;
        let mut keep = weighted.len();
        for (index, entry) in weighted.iter().enumerate() {
            cumulative += entry.1 / total;
            if cumulative >= top_p as f64 {
                keep = index + 1;
                break;
            }
        }
        weighted.truncate(keep);
    }
    let total = weighted.iter().map(|entry| entry.1).sum::<f64>();
    let mut probabilities = vec![0.0f64; vocab];
    if total > 0.0 {
        for (token, weight) in weighted {
            probabilities[token as usize] = weight / total;
        }
    }
    probabilities
}

fn sparse_q(
    distribution: &SparseDraftDistribution,
    row: usize,
    temperature: f32,
) -> Vec<(u32, f64)> {
    let base = row * distribution.top_k;
    let ids = &distribution.candidate_ids[base..base + distribution.top_k];
    let scores = &distribution.scores[base..base + distribution.top_k];
    let inverse_temperature = 1.0 / temperature.max(f32::MIN_POSITIVE) as f64;
    let max_score = scores
        .iter()
        .map(|&score| score as f64 * inverse_temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weighted = ids
        .iter()
        .copied()
        .zip(
            scores
                .iter()
                .map(|&score| (score as f64 * inverse_temperature - max_score).exp()),
        )
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|entry| entry.1).sum::<f64>();
    for entry in &mut weighted {
        entry.1 /= total;
    }
    weighted
}

fn sample_weights(weights: &[f64], random: f64) -> u32 {
    let total = weights.iter().sum::<f64>();
    let threshold = random * total;
    let mut cumulative = 0.0;
    for (token, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= threshold {
            return token as u32;
        }
    }
    weights
        .iter()
        .rposition(|weight| *weight > 0.0)
        .unwrap_or(0) as u32
}

/// Return `None` when stateful logits processing would be required. The raw
/// equality path remains the conservative fallback until the GPU sampler owns
/// grammar/penalty processing too.
pub(super) fn standard_rejection(
    model: &dyn Model,
    active: &ActiveSeq,
    drafts: &[u32],
    distribution: &SparseDraftDistribution,
) -> Result<Option<StandardRejectionVerdict>> {
    if active.temperature <= 0.0
        || active.grammar_state.is_some()
        || active.inside_thinking
        || active.repetition_penalty != 1.0
        || active.presence_penalty != 0.0
        || active.frequency_penalty != 0.0
        || active.lz_penalty != 0.0
        || active.dry_multiplier != 0.0
        || !active.logit_bias.is_empty()
    {
        return Ok(None);
    }
    ensure!(
        distribution.top_k > 0
            && distribution.candidate_ids.len() >= drafts.len() * distribution.top_k
            && distribution.scores.len() >= drafts.len() * distribution.top_k,
        "DFlash2 sparse proposal geometry does not cover the draft block"
    );
    let vocab = model.vocab_size();
    let row_bytes = vocab * 2;
    let rows = drafts.len() + 1;
    let mut logits = vec![0u8; rows * row_bytes];
    model.copy_logits_to_host(model.logits_buffer_ptr(), &mut logits)?;
    let seed = active.seed.unwrap_or(0xD5A5_53F1_A5E2_0001);
    let position_base = active.output_tokens.len();

    for (row, &draft) in drafts.iter().enumerate() {
        let target = target_distribution(
            &logits[row * row_bytes..(row + 1) * row_bytes],
            vocab,
            active.temperature,
            active.top_k,
            active.top_p,
            active.top_n_sigma,
            active.min_p,
        );
        let q = sparse_q(distribution, row, active.temperature);
        let q_draft = q
            .iter()
            .find_map(|&(token, probability)| (token == draft).then_some(probability))
            .unwrap_or(0.0);
        let p_draft = target.get(draft as usize).copied().unwrap_or(0.0);
        let accept_u = uniform(seed, position_base + row, 0xA11C_E001);
        if p_draft > accept_u * q_draft {
            continue;
        }

        let mut residual = target;
        for (token, probability) in q {
            if let Some(value) = residual.get_mut(token as usize) {
                *value = (*value - probability).max(0.0);
            }
        }
        let bonus = sample_weights(&residual, uniform(seed, position_base + row, 0xB0A5_0002));
        return Ok(Some(StandardRejectionVerdict {
            accepted: row,
            bonus,
        }));
    }

    let bonus_target = target_distribution(
        &logits[drafts.len() * row_bytes..rows * row_bytes],
        vocab,
        active.temperature,
        active.top_k,
        active.top_p,
        active.top_n_sigma,
        active.min_p,
    );
    Ok(Some(StandardRejectionVerdict {
        accepted: drafts.len(),
        bonus: sample_weights(
            &bonus_target,
            uniform(seed, position_base + drafts.len(), 0xB0A5_0002),
        ),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_q_normalizes_realized_row() {
        let distribution = SparseDraftDistribution {
            top_k: 2,
            candidate_ids: vec![7, 9],
            scores: vec![0.0, 0.0],
        };
        let q = sparse_q(&distribution, 0, 1.0);
        assert_eq!(q[0].0, 7);
        assert!((q.iter().map(|entry| entry.1).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn residual_sampler_never_selects_zero_mass_tail() {
        assert_eq!(sample_weights(&[0.0, 1.0, 0.0], 0.75), 1);
    }
}
