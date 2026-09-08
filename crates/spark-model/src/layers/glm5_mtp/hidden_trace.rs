// SPDX-License-Identifier: AGPL-3.0-only
//! Request-owned diagnostic; equal hidden hashes do not prove equal private KV.
use super::*;
use crate::traits::SequenceState;
use anyhow::{Context, ensure};
use sha2::{Digest, Sha256};

pub(super) const ROW_BYTES: usize = 8192;

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
}
impl StepTrace {
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
            trace_version=2u8, post_eh_sha256=%OptionalHex(self.post_eh.as_ref()),
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
        adapter_ownership().ensure_absent()?;
        ensure!(
            ctx.config.model_type == "glm5_next"
                && ctx.config.hidden_size == 4096
                && ctx.config.tp_world_size == 2
                && ctx.config.ep_world_size == 2
                && ctx.levers.max_decode_seqs == 1
                && ctx.levers.drafter.prefill
                && !ctx.levers.drafter.carry
                && ctx
                    .comm
                    .is_some_and(|c| c.world_size() == 2 && c.rank() < 2)
                && drafts == 4
                && !grammar
                && seq.adapter_id == 0
                && seq.adapter_slot < 0
                && seq.cached_prefix_tokens == 0
                && seq.cached_prefix_blocks == 0
                && seq.marconi_skip_to == 0
                && seq.disk_block_ids.is_empty()
                && ctx.config.adapter_max_rank == 0
                && ctx.routed_lora_layers.is_none()
                && !matches!(ctx.moe_lora_route, crate::layer::MoeLoraRoute::Refuse),
            "GLM hidden trace requires exact cold C1 TP2/EP2 MTP4 repair profile without adapters"
        );
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
