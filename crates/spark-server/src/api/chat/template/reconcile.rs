// SPDX-License-Identifier: AGPL-3.0-only
//! `<think>` span-vs-flag reconciliation (extracted; identical content).

use super::*;

pub(super) fn reconcile_prompt_thinking(
    prompt_tokens: &[u32],
    think_start_token_id: Option<u32>,
    think_end_token_id: Option<u32>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    max_thinking_budget: u32,
) -> (bool, Option<u32>) {
    if let Some(think_start) = think_start_token_id {
        // An explicit empty reasoning suffix has already closed thinking before
        // generation (GLM tools). Do not mistake a historical pair followed by
        // another assistant header or prompt text for this terminal boundary.
        if think_end_token_id.is_some_and(|end| prompt_tokens.ends_with(&[think_start, end])) {
            return (false, None);
        }
        let has_unclosed_think = open_think_in_tail(prompt_tokens, think_start, think_end_token_id);
        if has_unclosed_think && !enable_thinking {
            tracing::info!(
                "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
                 overriding enable_thinking=true with budget={}",
                max_thinking_budget,
            );
            (true, atlas_kernels::thinking_ceiling(max_thinking_budget))
        } else {
            (enable_thinking, thinking_budget)
        }
    } else {
        (enable_thinking, thinking_budget)
    }
}

/// Whether the prompt's last tokens leave a reasoning block open.
fn open_think_in_tail(prompt_tokens: &[u32], think_start: u32, think_end: Option<u32>) -> bool {
    let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
    match (tail.iter().rposition(|t| *t == think_start), think_end) {
        (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
        (Some(_), None) => true,
        (None, _) => false,
    }
}

/// `--honor-thinking-off`: a request that resolved thinking off, rendered by
/// a template that opens `<think>` whatever it was told (GLM-5.3), gets the
/// block closed, as that template itself writes a turn with no reasoning.
/// Returns whether it closed one; [`reconcile_prompt_thinking`] then reads
/// the closed pair as thinking off instead of forcing reasoning on.
pub(super) fn close_open_think_when_off(
    prompt_tokens: &mut Vec<u32>,
    think_start_token_id: Option<u32>,
    think_end_token_id: Option<u32>,
    enable_thinking: bool,
) -> bool {
    let (Some(start), Some(end)) = (think_start_token_id, think_end_token_id) else {
        return false;
    };
    if enable_thinking || !open_think_in_tail(prompt_tokens, start, Some(end)) {
        return false;
    }
    prompt_tokens.push(end);
    tracing::info!("--honor-thinking-off: closed the template's open \\<think\\>");
    true
}
