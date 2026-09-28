// SPDX-License-Identifier: AGPL-3.0-only

//! Pure sequential planning. No execution, wire codec, or live-state authority.
//!
//! This validates caller-declared positions, generations and capabilities, not
//! their agreement with live requests or another rank. Binding, reservation,
//! replay prevention and the legacy full-prompt worker exchange remain separate
//! obligations. No scheduler or Model method consumes this plan yet.

use anyhow::{Result, ensure};
use std::collections::HashSet;

/// Logical request lifetime; neither a physical SSM index nor a KV block ID.
/// Generations are preserved only: legacy v1/v2 do not transmit this field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestKey {
    pub wire_slot: u32,
    pub generation: u64,
}

/// Caller-assigned identity, not a session registry or replay-prevention token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepId {
    pub session: u64,
    pub sequence: u64,
}

/// A range in the intent's token payload, in token elements rather than bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSpan {
    pub offset: usize,
    pub count: usize,
}

impl TokenSpan {
    fn end(self) -> Result<usize> {
        self.offset
            .checked_add(self.count)
            .ok_or_else(|| anyhow::anyhow!("execution-plan token span overflow"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkItem {
    /// Exactly one independent token. Speculative temporal rows are unsupported.
    DecodeOne { token: TokenSpan },
    /// Full prompt payload, matching PrefillSlice/Model::prefill_chunk.
    /// Only chunk_len tokens count against the scheduled/serial-arena budgets.
    PrefillChunk {
        prompt: TokenSpan,
        prompt_revision: u64,
        chunk_start: usize,
        chunk_len: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestWork {
    pub key: RequestKey,
    /// The declared already-computed length; live-state comparison is deferred.
    pub expected_position: usize,
    pub kind: WorkItem,
}

#[derive(Clone, Debug)]
pub struct ScheduledIntent {
    pub step: StepId,
    pub work: Vec<RequestWork>,
    /// Canonical concatenation in work order: one token for DecodeOne, the
    /// complete prompt for PrefillChunk. No overlaps, gaps or trailing tokens.
    pub tokens: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    LegacySerial,
    /// Explicit rejection targets; neither has a lowering in this milestone.
    Speculative,
    PackedMixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionFeature {
    Multimodal,
    Adapters,
    PromptLogprobs,
    PrefixReuse,
    Swapping,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillStatePolicy {
    Preserve,
    /// Existing EP worker order: prefill_chunk, normalize_ssm_states, then
    /// return to the next command. Required when binding that worker path.
    NormalizeAfterChunk,
}

/// Explicit caller-resolved profile. This declaration does not authorize a
/// hardware route; the later rank-bound executor must validate its capability.
pub struct PlanProfile {
    pub execution: ExecutionMode,
    pub features: Vec<ExecutionFeature>,
    pub prefill_state: PrefillStatePolicy,
}

/// Already-resolved limits, with no environment reads or implicit defaults.
/// Request-type limits may be zero to forbid that work type. Arena capacity is
/// per serial operation, not the sum of requests' scheduled tokens.
#[derive(Clone, Copy, Debug)]
pub struct PlanLimits {
    pub max_requests: usize,
    pub max_decode_requests: usize,
    pub max_prefill_requests: usize,
    pub max_context_tokens: usize,
    pub max_scheduled_tokens: usize,
    pub max_payload_tokens: usize,
    pub max_arena_tokens: usize,
    pub max_prefill_chunk_tokens: usize,
    pub vocab_size: u32,
    pub wire_slot_capacity: u32,
}

impl PlanLimits {
    fn validate(self) -> Result<()> {
        ensure!(
            self.max_requests > 0
                && self.max_context_tokens > 0
                && self.max_scheduled_tokens > 0
                && self.max_payload_tokens > 0
                && self.max_arena_tokens > 0
                && self.vocab_size > 0
                && self.wire_slot_capacity > 0,
            "execution-plan limits must be explicit and nonzero"
        );
        ensure!(
            self.max_prefill_requests == 0 || self.max_prefill_chunk_tokens > 0,
            "execution-plan prefill requires a positive chunk budget"
        );
        self.max_payload_tokens
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("execution-plan payload byte limit overflow"))?;
        Ok(())
    }
}

/// Semantic obligations, not GPU commands. ConsumeLogits means finish consuming
/// or independently owning the result before continuing; retaining an arena
/// pointer does not satisfy it. No speculative rollback/sampling is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialSubstep {
    DecodeOne {
        work_index: usize,
        position_after: usize,
    },
    PrefillChunk {
        work_index: usize,
        position_after: usize,
        is_last: bool,
    },
    NormalizeSsm {
        work_index: usize,
    },
    ConsumeLogits {
        work_index: usize,
    },
}

/// Immutable validated intent and serial lowering. No mutable accessors, device
/// pointers, live state, communication or GPU allocations are retained here.
pub struct ValidatedStepPlan {
    intent: ScheduledIntent,
    substeps: Vec<SequentialSubstep>,
    scheduled_tokens: usize,
}

impl ValidatedStepPlan {
    pub fn new(intent: ScheduledIntent, profile: PlanProfile, limits: PlanLimits) -> Result<Self> {
        limits.validate()?;
        ensure!(
            profile.execution == ExecutionMode::LegacySerial && profile.features.is_empty(),
            "execution-plan profile has no supported sequential lowering"
        );
        ensure!(
            !intent.work.is_empty() && intent.work.len() <= limits.max_requests,
            "execution-plan request count is empty or exceeds budget"
        );
        ensure!(
            intent.tokens.len() <= limits.max_payload_tokens,
            "execution-plan full token payload exceeds budget"
        );
        ensure!(
            intent.tokens.iter().all(|&token| token < limits.vocab_size),
            "execution-plan token outside vocabulary"
        );
        let capacity = intent
            .work
            .len()
            .checked_mul(3)
            .ok_or_else(|| anyhow::anyhow!("execution-plan substep capacity overflow"))?;
        let mut substeps = Vec::new();
        substeps.try_reserve_exact(capacity)?;
        let mut seen = HashSet::new();
        let (mut cursor, mut scheduled_tokens, mut decodes, mut prefills) =
            (0usize, 0usize, 0usize, 0usize);
        for (index, work) in intent.work.iter().enumerate() {
            ensure!(
                work.key.wire_slot < limits.wire_slot_capacity && seen.insert(work.key.wire_slot),
                "execution-plan request slot is duplicate, reused or out of range"
            );
            let (span, count) = match work.kind {
                WorkItem::DecodeOne { token } => {
                    ensure!(token.count == 1, "DecodeOne requires exactly one token");
                    decodes += 1;
                    ensure!(
                        decodes <= limits.max_decode_requests,
                        "execution-plan decode request budget exceeded"
                    );
                    (token, 1)
                }
                WorkItem::PrefillChunk {
                    prompt,
                    chunk_start,
                    chunk_len,
                    ..
                } => {
                    prefills += 1;
                    ensure!(
                        prefills <= limits.max_prefill_requests,
                        "execution-plan prefill request budget exceeded"
                    );
                    ensure!(
                        prompt.count > 0 && prompt.count <= limits.max_context_tokens,
                        "execution-plan prompt exceeds context or is empty"
                    );
                    ensure!(
                        chunk_len > 0 && chunk_len <= limits.max_prefill_chunk_tokens,
                        "execution-plan prefill chunk is empty or exceeds budget"
                    );
                    let end = chunk_start
                        .checked_add(chunk_len)
                        .ok_or_else(|| anyhow::anyhow!("execution-plan chunk end overflow"))?;
                    ensure!(
                        chunk_start == work.expected_position && end <= prompt.count,
                        "execution-plan prefill bounds disagree with declared position/prompt"
                    );
                    (prompt, chunk_len)
                }
            };
            let end = span.end()?;
            ensure!(
                span.offset == cursor && end <= intent.tokens.len(),
                "execution-plan spans must be canonical and within payload"
            );
            cursor = end;
            ensure!(
                count <= limits.max_arena_tokens,
                "execution-plan serial operation exceeds arena"
            );
            scheduled_tokens = scheduled_tokens
                .checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("execution-plan scheduled-token sum overflow"))?;
            ensure!(
                scheduled_tokens <= limits.max_scheduled_tokens,
                "execution-plan scheduled-token budget exceeded"
            );
            let position_after = work
                .expected_position
                .checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("execution-plan position overflow"))?;
            ensure!(
                position_after <= limits.max_context_tokens,
                "execution-plan position exceeds context"
            );
            match work.kind {
                WorkItem::DecodeOne { .. } => {
                    substeps.push(SequentialSubstep::DecodeOne {
                        work_index: index,
                        position_after,
                    });
                    substeps.push(SequentialSubstep::ConsumeLogits { work_index: index });
                }
                WorkItem::PrefillChunk { prompt, .. } => {
                    let is_last = position_after == prompt.count;
                    substeps.push(SequentialSubstep::PrefillChunk {
                        work_index: index,
                        position_after,
                        is_last,
                    });
                    if profile.prefill_state == PrefillStatePolicy::NormalizeAfterChunk {
                        substeps.push(SequentialSubstep::NormalizeSsm { work_index: index });
                    }
                    if is_last {
                        substeps.push(SequentialSubstep::ConsumeLogits { work_index: index });
                    }
                }
            }
        }
        ensure!(
            cursor == intent.tokens.len(),
            "execution-plan payload has trailing tokens"
        );
        Ok(Self {
            intent,
            substeps,
            scheduled_tokens,
        })
    }

    pub fn step(&self) -> StepId {
        self.intent.step
    }
    pub fn work(&self) -> &[RequestWork] {
        &self.intent.work
    }
    pub fn tokens(&self) -> &[u32] {
        &self.intent.tokens
    }
    pub fn substeps(&self) -> &[SequentialSubstep] {
        &self.substeps
    }
    pub fn scheduled_tokens(&self) -> usize {
        self.scheduled_tokens
    }
}

#[cfg(test)]
#[path = "execution_plan_tests.rs"]
mod tests;
