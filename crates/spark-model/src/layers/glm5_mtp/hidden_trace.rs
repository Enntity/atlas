// SPDX-License-Identifier: AGPL-3.0-only
//! Request-owned diagnostic; equal hidden hashes do not prove equal private KV.
use super::*;
use crate::traits::SequenceState;
use anyhow::{Context, ensure};
use sha2::{Digest, Sha256};

pub(super) const ROW_BYTES: usize = 8192;

#[path = "hidden_trace_kv.rs"]
mod kv;
#[path = "hidden_trace_profile.rs"]
mod profile;
#[path = "hidden_trace_prompt.rs"]
mod prompt;
use profile::ColdProfile;
pub(crate) use prompt::{arm_prompt, prompt_selected, spend_prompt};

#[cfg(test)]
pub(crate) fn fixture_set_enabled(state: &mut dyn ProposerState, enabled: bool) {
    state
        .as_any_mut()
        .downcast_mut::<Glm5MtpProposerState>()
        .unwrap()
        .hidden_trace
        .enabled = enabled;
}
#[cfg(test)]
pub(crate) fn fixture_prompt_hashes(
    state: &Glm5MtpProposerState,
    generation: u64,
    p: usize,
) -> Result<[[u8; 32]; 7]> {
    let e = state.hidden_trace.prompt.evidence(0, generation, p)?;
    Ok([
        e.primer_source,
        e.bootstrap_source,
        e.primer_tokens,
        e.bootstrap_token,
        e.primer_kv,
        e.bootstrap_kv,
        e.written_prefix,
    ])
}

/// Live model owners plus sticky installation history, never config inference.
#[derive(Clone, Copy)]
pub(crate) struct AdapterOwnership {
    pub pool: bool,
    pub overlays: bool,
    pub rotatable: bool,
    pub install_attempted: bool,
}
impl AdapterOwnership {
    pub(crate) fn ensure_absent(self) -> Result<()> {
        ensure!(
            !self.pool && !self.overlays && !self.rotatable && !self.install_attempted,
            "GLM hidden trace requires no live or previously installed adapters"
        );
        Ok(())
    }
}

pub(super) fn parse(value: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(value, None | Some("0") | Some("1")),
        "ATLAS_GLM_MTP_HIDDEN_TRACE must be 0 or 1"
    );
    Ok(value == Some("1"))
}

pub(super) fn configured() -> Result<bool> {
    let enabled = parse_environment(std::env::var("ATLAS_GLM_MTP_HIDDEN_TRACE"))?;
    if enabled {
        ensure!(
            crate::speculative::glm_repair_policy::enabled(),
            "GLM hidden trace requires accepted-pair repair"
        );
        crate::speculative::glm_repair_policy::validate_environment()?;
    }
    Ok(enabled)
}

fn parse_environment(value: std::result::Result<String, std::env::VarError>) -> Result<bool> {
    match value {
        Ok(value) => parse(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("ATLAS_GLM_MTP_HIDDEN_TRACE must be Unicode 0 or 1")
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    slot: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Request {
    identity: Identity,
    pub position: usize,
    pub seed: u32,
    pub hidden_row: usize,
    pub saved: DevicePtr,
    pub rank: usize,
}

#[derive(Clone, Copy, Debug)]
struct Attempt {
    request: Request,
    ordinal: u8,
    next_step: usize,
}

#[derive(Default)]
pub(super) struct HiddenTrace {
    pub enabled: bool,
    identity: Option<Identity>,
    spent: u8,
    active: Option<Attempt>,
    pub(super) prompt: prompt::Prompt,
}
impl HiddenTrace {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new(self.enabled);
    }

    fn begin(&mut self, request: Request) -> Result<bool> {
        if !self.enabled {
            return Ok(false);
        }
        self.active = None;
        if let Some(old) = self.identity {
            ensure!(
                request.identity.generation >= old.generation,
                "GLM hidden trace stale request generation"
            );
            if request.identity.generation == old.generation {
                ensure!(
                    request.identity == old,
                    "GLM hidden trace request slot changed within generation"
                );
            } else {
                self.spent = 0;
            }
        }
        ensure!(
            request.identity.generation != 0,
            "GLM hidden trace requires capture generation"
        );
        self.identity = Some(request.identity);
        if self.spent >= 8 {
            return Ok(false);
        }
        self.spent += 1;
        self.active = Some(Attempt {
            request,
            ordinal: self.spent,
            next_step: 0,
        });
        Ok(true)
    }

    pub fn input(
        &mut self,
        token: u32,
        position: usize,
        step: usize,
        ptr: DevicePtr,
        cache_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<StepTrace>> {
        let Some(active) = &mut self.active else {
            return Ok(None);
        };
        let expected = if step == 0 {
            active.request.saved
        } else {
            ctx.buffers.norm_output()
        };
        ensure!(
            step < 4
                && step == active.next_step
                && active.request.position.checked_add(step) == Some(position)
                && ptr == expected
                && ctx.config.hidden_size == 4096
                && (token as usize) < ctx.config.vocab_size
                && (step != 0 || token == active.request.seed),
            "GLM hidden trace step/source metadata"
        );
        active.next_step += 1;
        let mut record = StepTrace {
            request: active.request,
            ordinal: active.ordinal,
            step,
            token,
            cache_before: cache_rows,
            cache_after: 0,
            input: [0; 32],
            post_eh: None,
            final_hidden: [0; 32],
            kv: None,
            kv_before_spent: false,
            prompt: if active.ordinal == 1 && step == 0 && (2..=256).contains(&cache_rows) {
                Some(self.prompt.evidence(
                    active.request.identity.slot,
                    active.request.identity.generation,
                    cache_rows,
                )?)
            } else {
                None
            },
        };
        record.input = snapshot(ctx.gpu, ptr, ctx.graph_capture, stream)?;
        Ok(Some(record))
    }
}

pub(super) struct StepTrace {
    request: Request,
    ordinal: u8,
    step: usize,
    token: u32,
    cache_before: usize,
    cache_after: usize,
    input: [u8; 32],
    post_eh: Option<[u8; 32]>,
    final_hidden: [u8; 32],
    kv: Option<kv::Probe>,
    kv_before_spent: bool,
    prompt: Option<prompt::Evidence>,
}
impl StepTrace {
    pub fn kv_before(
        &mut self,
        cache: &PagedKvCache,
        state: &Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.ordinal != 1 || self.step != 0 {
            return Ok(());
        }
        ensure!(
            !self.kv_before_spent,
            "GLM first KV prefix already attempted"
        );
        self.kv_before_spent = true;
        let repair_state::RepairPhase::Proposed(plan) = state.repair else {
            anyhow::bail!("GLM first KV requires prepared repair");
        };
        ensure!(
            self.post_eh.is_some()
                && self.kv.is_none()
                && self.request.hidden_row == 0
                && plan.generation() == self.request.identity.generation
                && plan.position() == self.request.position
                && state.seq_len == self.cache_before
                && state.seq_len.checked_add(1) == Some(self.request.position)
                && state.seq_len.checked_add(4) == Some(plan.speculative_cache_end()),
            "GLM first KV request/repair/cursor/post-EH mismatch"
        );
        if self.prompt.is_some() {
            state
                .hidden_trace
                .prompt
                .validate_body(cache, &state.block_table, ctx, stream)?;
        }
        self.kv = Some(kv::Probe::before(
            cache,
            &state.block_table,
            state.seq_len,
            ctx,
            stream,
        )?);
        Ok(())
    }
    pub fn kv_after(
        &mut self,
        cache: &PagedKvCache,
        state: &Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.ordinal != 1 || self.step != 0 {
            return Ok(());
        }
        ensure!(
            state.seq_len == self.cache_before,
            "GLM first KV body changed cursor"
        );
        self.kv
            .as_mut()
            .context("GLM first KV missing prefix")?
            .after(cache, &state.block_table, state.seq_len, ctx, stream)
    }
    pub fn post_eh(&mut self, ptr: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        if self.step != 0 {
            return Ok(());
        }
        ensure!(
            self.post_eh.is_none()
                && ptr == ctx.buffers.hidden_states()
                && ctx.buffers.sizes().hidden_states >= ROW_BYTES
                && ctx.config.hidden_size == 4096,
            "GLM hidden trace post-EH owner/capacity/duplicate"
        );
        self.post_eh = Some(snapshot(ctx.gpu, ptr, ctx.graph_capture, stream)?);
        Ok(())
    }
    pub fn final_hidden(
        &mut self,
        ptr: DevicePtr,
        cache_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            ptr == ctx.buffers.norm_output()
                && ctx.buffers.sizes().norm_output >= ROW_BYTES
                && ctx.config.hidden_size == 4096
                && (self.step != 0 || self.post_eh.is_some())
                && (self.ordinal != 1
                    || self.step != 0
                    || self.kv.as_ref().is_some_and(|p| p.appended.is_some()))
                && self.cache_before.checked_add(1) == Some(cache_rows),
            "GLM hidden trace final owner/capacity/cursor/post-EH"
        );
        self.final_hidden = snapshot(ctx.gpu, ptr, ctx.graph_capture, stream)?;
        self.cache_after = cache_rows;
        Ok(())
    }
    pub fn emit(self, draft: u32, pairs: Option<[u8; 16]>, eh_nvfp4: bool, head_nvfp4: bool) {
        tracing::info!(rank=self.request.rank, slot=self.request.identity.slot, generation=self.request.identity.generation,
            attempt=self.ordinal, position=self.request.position, seed=self.request.seed, hidden_row=self.request.hidden_row,
            step=self.step, input_token=self.token, self.cache_before, self.cache_after, draft, eh_nvfp4, head_nvfp4,
            input_sha256=%Hex(&self.input), final_sha256=%Hex(&self.final_hidden), argmax_pair_bytes=?pairs,
            trace_version=4u8, post_eh_sha256=%OptionalHex(self.post_eh.as_ref()),
            kv_prefix_sha256=%OptionalHex(self.kv.as_ref().map(|p| &p.prefix)),
            kv_appended_sha256=%OptionalHex(self.kv.as_ref().and_then(|p| p.appended.as_ref())),
            kv_block_map_sha256=%OptionalHex(self.kv.as_ref().map(|p| &p.block_map)),
            prompt_primer_source_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.primer_source)),
            prompt_bootstrap_source_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.bootstrap_source)),
            prompt_primer_tokens_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.primer_tokens)),
            prompt_bootstrap_token_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.bootstrap_token)),
            prompt_primer_kv_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.primer_kv)),
            prompt_bootstrap_kv_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.bootstrap_kv)),
            prompt_written_prefix_sha256=%OptionalHex(self.prompt.as_ref().map(|p|&p.written_prefix)),
            "GLM MTP HIDDEN_TRACE");
    }
}
struct OptionalHex<'a>(Option<&'a [u8; 32]>);
impl std::fmt::Display for OptionalHex<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(hash) => Hex(hash).fmt(f),
            None => f.write_str("None"),
        }
    }
}
struct Hex<'a>(&'a [u8; 32]);
impl std::fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn snapshot(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    capturing: bool,
    stream: u64,
) -> Result<[u8; 32]> {
    ensure!(
        !capturing && !gpu.stream_is_capturing(stream),
        "GLM hidden trace requires eager stream before I/O"
    );
    validate_row(ptr)?;
    let mut row = [0u8; ROW_BYTES];
    gpu.copy_d2h_on_stream(ptr, &mut row, stream)?;
    Ok(Sha256::digest(row).into())
}

/// Called by the real model wrapper after repair, on both head and worker.
#[allow(clippy::too_many_arguments)]
pub(crate) fn arm_prepared(
    seq: &mut SequenceState,
    token: u32,
    position: usize,
    drafts: usize,
    saved: DevicePtr,
    hidden_row: usize,
    grammar: bool,
    ctx: &ForwardContext,
    stream: u64,
    adapter_ownership: impl FnOnce() -> AdapterOwnership,
) -> Result<()> {
    let profile = ColdProfile::from(seq);
    let Some(state) = seq
        .proposer_state
        .as_mut()
        .and_then(|p| p.as_any_mut().downcast_mut::<Glm5MtpProposerState>())
    else {
        return Ok(());
    };
    if !state.hidden_trace.enabled {
        return Ok(());
    }
    let request = Request {
        identity: Identity {
            slot: seq.slot_idx,
            generation: seq.mtp_capture_gen,
        },
        position,
        seed: token,
        hidden_row,
        saved,
        rank: ctx.comm.map_or(usize::MAX, |c| c.rank()),
    };
    if !state.hidden_trace.begin(request)? {
        return Ok(());
    }
    let result = (|| {
        profile.validate(ctx, drafts, grammar, adapter_ownership)?;
        if state.hidden_trace.spent == 1 && (2..=256).contains(&state.seq_len) {
            state
                .hidden_trace
                .prompt
                .evidence(seq.slot_idx, seq.mtp_capture_gen, state.seq_len)?;
        }
        let repair_state::RepairPhase::Proposed(plan) = state.repair else {
            anyhow::bail!("GLM hidden trace must be armed after repair");
        };
        ensure!(
            plan.generation() == seq.mtp_capture_gen
                && plan.position() == position
                && state.seq_len.checked_add(4) == Some(plan.speculative_cache_end())
                && seq.seq_len == position
                && seq.tokens.len() == position
                && (token as usize) < ctx.config.vocab_size
                && hidden_row < 5,
            "GLM hidden trace prepared proposal metadata"
        );
        ensure!(
            !ctx.graph_capture && !ctx.gpu.stream_is_capturing(stream),
            "GLM hidden trace requires eager stream before I/O"
        );
        validate_row(saved)?;
        Ok(())
    })();
    if result.is_err() {
        state.hidden_trace.active = None;
    }
    result
}

fn validate_row(ptr: DevicePtr) -> Result<()> {
    ensure!(
        !ptr.is_null() && ptr.0.is_multiple_of(2),
        "GLM hidden trace null/unaligned BF16 row"
    );
    ptr.0
        .checked_add(ROW_BYTES as u64)
        .context("GLM hidden trace row address overflow")?;
    Ok(())
}

#[cfg(test)]
#[path = "hidden_trace_tests.rs"]
mod tests;
