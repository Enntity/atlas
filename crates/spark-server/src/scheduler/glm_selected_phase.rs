// SPDX-License-Identifier: AGPL-3.0-only
//! Selected GLM verification never commits past a reasoning phase boundary.
use crate::scheduler::ActiveSeq;

/// Picks share the pre-emission thinking state. Make the first boundary the
/// bonus token, so the next transaction samples under the new phase instead
/// of accepting later rows selected under stale masks. This runs BEFORE every
/// accepted-count publication, target trim and producer detachment. A native
/// GLM EOS also ends this prefix: terminal emission must not ingest later rows,
/// even if another emission guard subsequently suppresses the EOS.
pub(in crate::scheduler) fn accepted_before_boundary(
    a: &ActiveSeq,
    selected: &[u32],
    accepted: usize,
    native: Option<u32>,
) -> usize {
    if !a.inside_thinking {
        return accepted;
    }
    selected
        .iter()
        .take(accepted + 1)
        .position(|&token| {
            a.think_end_token == Some(token)
                || crate::glm_tool_boundary::native_eos_while_thinking(
                    native,
                    a.inside_thinking,
                    token,
                    &a.eos_tokens,
                )
                || (a.tools_present
                    && native == Some(token)
                    && a.tool_call_start_token == Some(token))
        })
        .unwrap_or(accepted)
}
