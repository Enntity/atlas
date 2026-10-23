// SPDX-License-Identifier: AGPL-3.0-only

//! Admission and logprob reads on the vocabulary-split verify head.
use super::*;

/// Greedy requests without logprobs retain the raw argmax fast path. A
/// sampled request or a logprob request needs the split exchange protocol;
/// refuse it before starting a verify rather than silently sampling greedily.
pub(super) fn ensure_available(model: &dyn Model, a: &ActiveSeq) -> anyhow::Result<()> {
    anyhow::ensure!(
        !model.verify_logits_argmax_only()
            || (a.temperature <= 0.0 && a.top_logprobs.is_none())
            || model.verify_split_sampling(),
        "split-head sampling/logprobs require the complete verify exchange"
    );
    Ok(())
}

pub(super) use crate::scheduler::verify_pipeline_helper::scored::abort_exchange;

#[cfg(test)]
#[path = "split_head_tests.rs"]
mod tests;
