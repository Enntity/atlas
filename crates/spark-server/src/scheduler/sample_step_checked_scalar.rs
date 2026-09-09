// SPDX-License-Identifier: AGPL-3.0-only
//! Shared scalar sampler with explicit legacy/checked copy-error policy.
use super::*;
use crate::scheduler::fast_greedy::CopyFailurePolicy;

#[cfg(test)]
#[path = "sample_step_checked_scalar_tests.rs"]
mod tests;

/// Sample one token from device logits with optional grammar constraint.
///
/// Like `sample_token` but also applies grammar bitmask when `grammar_state`
/// is provided. Always uses host-side sampling when grammar is active (can't
/// use GPU argmax since grammar bitmask is CPU-side).
///
/// `penalties` + `history` carry the sequence's configured repetition /
/// presence / frequency / LZ / DRY penalties (built via [`penalty_params_for`])
/// and the output-token history. These are applied via the shared
/// [`apply_penalties_and_bias`] helper AFTER the grammar bitmask + EOS
/// suppression and BEFORE the temperature decision — the same order the
/// non-MTP `process_seq_logits` path uses — so MTP-bootstrap-emitted tokens
/// see the same penalties as the non-MTP path. Backward-compatible: a
/// no-op when the penalties are neutral (rep==1.0, dry==0.0, etc.).
pub fn sample_token_with_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
    penalties: &SamplingParams,
    history: &[u32],
    levers: &crate::scheduler::logit_processors::SamplingLevers,
) -> Result<u32> {
    sample_with_policy(
        model,
        logits,
        temperature,
        top_k,
        top_p,
        suppress_ids,
        grammar_state,
        penalties,
        history,
        levers,
        CopyFailurePolicy::LegacyFallback,
    )
}

/// Grammarless scalar sampling with strict copy-error propagation.
/// No serving caller yet; preserves the inherited BF16 sampling mathematics.
pub fn sample_token_with_grammar_checked(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
    penalties: &SamplingParams,
    history: &[u32],
    levers: &crate::scheduler::logit_processors::SamplingLevers,
) -> Result<u32> {
    anyhow::ensure!(
        grammar_state.is_none(),
        "checked scalar sampling does not support grammar"
    );
    sample_with_policy(
        model,
        logits,
        temperature,
        top_k,
        top_p,
        suppress_ids,
        grammar_state,
        penalties,
        history,
        levers,
        CopyFailurePolicy::Propagate,
    )
}

// One authoritative body: copy-error policy never changes sampling mathematics.
fn sample_with_policy(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
    mut grammar_state: Option<&mut GrammarState>,
    penalties: &SamplingParams,
    history: &[u32],
    levers: &crate::scheduler::logit_processors::SamplingLevers,
    policy: CopyFailurePolicy,
) -> Result<u32> {
    // ── FAST PATH (#3, 2026-06-02): on-GPU greedy pick under grammar ──
    // The MTP bootstrap sample (~1 token/step) otherwise D2Hs + dequants the
    // full 248k vocab + applies the bitmask on host. When greedy (temp=0 or
    // ATLAS_FORCE_TEMP_ZERO), penalties neutral, and no suppress list, the
    // masked-greedy pick == the GPU argmax whenever that argmax is grammar-
    // allowed (global max ∩ allowed-set = the max). Emit it directly; fall back
    // to the host path below only when the argmax is grammar-disallowed.
    // Mirrors the verify-path fast path. Kill-switch ATLAS_DISABLE_FAST_GREEDY=1.
    //
    // #237 (fix 4a): penalty-neutrality relaxed to the SSOT `fast_greedy`
    // gate shared with the verify helper — reduce-only penalties cannot flip
    // an argmax that is not in the scoped `history` and has a positive raw
    // logit (proof in `fast_greedy` module docs). `history` here is already
    // the scoped span the slow path feeds to `apply_penalties_and_bias`.
    if levers.fast_greedy_grammar
        && suppress_ids.is_empty()
        && (temperature == 0.0 || levers.force_temp_zero)
    {
        let gate = crate::scheduler::fast_greedy::classify_penalties(penalties);
        if gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
            let top1 = model.argmax_on_device(logits, 0)?;
            let immune = gate == crate::scheduler::fast_greedy::PenaltyGate::Neutral
                || policy.immune(top1, history, || {
                    crate::scheduler::fast_greedy::logit_is_positive_checked(
                        model,
                        logits,
                        0,
                        model.vocab_size(),
                        top1,
                    )
                })?;
            if immune {
                let allowed = match grammar_state.as_mut() {
                    Some(gs) => {
                        if gs.is_terminated() {
                            true
                        } else {
                            gs.fill_bitmask();
                            gs.is_token_allowed(top1)
                        }
                    }
                    None => true,
                };
                if allowed {
                    return Ok(top1);
                }
            }
        }
    }

    let vocab_size = model.vocab_size();
    let mut bf16_buf = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits, &mut bf16_buf)?;
    let mut f32_logits: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let lo = bf16_buf[i * 2];
            let hi = bf16_buf[i * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    // Apply grammar bitmask (when a grammar is active).
    if let Some(gs) = grammar_state {
        gs.fill_bitmask();
        gs.apply_bitmask_to_logits(&mut f32_logits);
    }
    // SSOT penalties + bias on the post-mask logits, using the seq's
    // output-token history — identical stage to the non-MTP path.
    apply_penalties_and_bias(&mut f32_logits, penalties, history);
    if temperature == 0.0 {
        let best = f32_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    // Penalties already applied in place above; pass neutral penalty params
    // to `sample_with_params` (which re-runs the helper with empty history,
    // a guaranteed no-op) so the stochastic top-k/top-p/min-p pipeline runs.
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            // P1-4 (2026-07-09): thread the sequence's resolved min_p from
            // the SSOT `penalties` struct (built by `penalty_params_for`,
            // which copies `a.min_p` — request value + MODEL.toml
            // `min_p_floor` applied in `sampling_setup`). This was a
            // hardcoded 0.0, so the MTP BOOTSTRAP token — one of only two
            // stochastic sample points under MTP — bypassed the FP8
            // argmax-flip safety net the floor documents. Kill-switch:
            // ATLAS_NO_MTP_MINP=1 restores the 0.0 literal.
            min_p: effective_min_p(penalties.min_p, levers),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}
