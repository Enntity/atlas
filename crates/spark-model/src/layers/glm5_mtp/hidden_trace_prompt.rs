// SPDX-License-Identifier: AGPL-3.0-only
//! Consumption-time prompt sources and immediate writer observations.
use super::super::kv_rows_plan::DeviceSpan;
use super::*;
use crate::model::glm_mtp_prompt_trace::Capture;

pub(crate) fn prompt_selected(seq: &SequenceState) -> bool {
    (2..=256).contains(&seq.prompt_len)
        && seq
            .proposer_state
            .as_ref()
            .and_then(|s| s.as_any().downcast_ref::<Glm5MtpProposerState>())
            .is_some_and(|s| s.hidden_trace.enabled && s.seq_len == 0)
}
pub(crate) fn spend_prompt(seq: &mut SequenceState) -> Result<()> {
    let s = seq
        .proposer_state
        .as_mut()
        .and_then(|s| s.as_any_mut().downcast_mut::<Glm5MtpProposerState>())
        .context("GLM prompt state")?;
    s.hidden_trace
        .prompt
        .begin((seq.slot_idx, seq.mtp_capture_gen))
}
pub(crate) fn arm_prompt(
    seq: &mut SequenceState,
    capture: Capture,
    ctx: &ForwardContext,
    stream: u64,
    owners: impl FnOnce() -> AdapterOwnership,
) -> Result<()> {
    let profile = ColdProfile::from(seq);
    let s = seq
        .proposer_state
        .as_mut()
        .and_then(|s| s.as_any_mut().downcast_mut::<Glm5MtpProposerState>())
        .context("GLM prompt state")?;
    s.hidden_trace.prompt.arm_begun(capture)?;
    let validation = (|| {
        profile.validate_request(ctx, owners)?;
        ensure!(
            s.hidden_trace.enabled
                && s.seq_len == 0
                && matches!(s.repair, repair_state::RepairPhase::Capture)
                && capture.identity() == (seq.slot_idx, seq.mtp_capture_gen)
                && capture.rows() == seq.prompt_len
                && seq.tokens.len() == seq.prompt_len
                && seq.seq_len == seq.prompt_len
                && !ctx.graph_capture
                && !ctx.gpu.stream_is_capturing(stream),
            "GLM prompt phase/profile mismatch"
        );
        Ok(())
    })();
    if validation.is_err() {
        s.hidden_trace.prompt.fail();
    }
    validation
}

#[derive(Clone, Copy, Default)]
pub(super) struct Evidence {
    pub primer_source: [u8; 32],
    pub bootstrap_source: [u8; 32],
    pub primer_tokens: [u8; 32],
    pub bootstrap_token: [u8; 32],
    pub primer_kv: [u8; 32],
    pub bootstrap_kv: [u8; 32],
    pub written_prefix: [u8; 32],
}
#[derive(Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Empty,
    Armed,
    PrimerRead,
    PrimerWritten,
    BootstrapRead,
    Complete,
    Failed,
}
#[derive(Default)]
pub(in crate::layers::glm5_mtp) struct Prompt {
    spent: Option<(usize, u64)>,
    capture: Option<Capture>,
    phase: Phase,
    evidence: Evidence,
    continuation: Option<Sha256>,
    scratch: Option<Box<[u8]>>,
    stream: u64,
    pool: Option<kv::Binding>,
}

#[cfg(test)]
#[path = "hidden_trace_prompt_tests.rs"]
mod tests;
impl Prompt {
    #[cfg(test)]
    pub fn arm(&mut self, capture: Capture) -> Result<()> {
        self.begin(capture.identity())?;
        self.arm_begun(capture)
    }
    fn begin(&mut self, identity: (usize, u64)) -> Result<()> {
        ensure!(
            self.spent.is_none_or(|old| identity.1 > old.1),
            "GLM prompt observation already spent/stale"
        );
        *self = Self {
            spent: Some(identity),
            phase: Phase::Failed,
            ..Self::default()
        };
        Ok(())
    }
    fn arm_begun(&mut self, capture: Capture) -> Result<()> {
        ensure!(
            self.spent == Some(capture.identity())
                && self.capture.is_none()
                && self.phase == Phase::Failed,
            "GLM prompt owner not freshly spent"
        );
        self.capture = Some(capture);
        self.phase = Phase::Armed;
        Ok(())
    }
    pub fn active(&self) -> bool {
        self.capture.is_some()
    }
    pub fn fail(&mut self) {
        if self.active() {
            self.phase = Phase::Failed;
            self.scratch = None;
            self.continuation = None;
        }
    }
    pub(super) fn evidence(&self, slot: usize, generation: u64, rows: usize) -> Result<Evidence> {
        ensure!(
            self.phase == Phase::Complete
                && self
                    .capture
                    .is_some_and(|c| c.identity() == (slot, generation) && c.rows() == rows),
            "GLM prompt evidence incomplete/stale"
        );
        Ok(self.evidence)
    }
    pub fn primer_before(
        &mut self,
        tokens: &[u32],
        source: DeviceSpan,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let prior = std::mem::replace(&mut self.phase, Phase::Failed);
        let c = self.capture.context("GLM prompt missing owner")?;
        ensure!(
            prior == Phase::Armed
                && tokens.len() == c.rows() - 1
                && c.matches(source.ptr, source.bytes, 0, tokens.len()),
            "GLM prompt primer span/phase"
        );
        let mut scratch = vec![0; 32768].into_boxed_slice();
        let hash = source_hash(
            b"atlas/glm53/mtp-source/primer/v1\0",
            c.rows(),
            source,
            ctx,
            stream,
            &mut scratch,
        )?;
        self.evidence.primer_source = hash;
        self.evidence.primer_tokens = token_hash(
            b"atlas/glm53/mtp-source/primer-tokens/v1\0",
            tokens.len(),
            tokens,
        );
        self.stream = stream;
        self.scratch = Some(scratch);
        self.phase = Phase::PrimerRead;
        Ok(())
    }
    pub fn primer_after(
        &mut self,
        cache: &PagedKvCache,
        blocks: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let prior = std::mem::replace(&mut self.phase, Phase::Failed);
        let mut scratch = self.scratch.take().context("GLM prompt missing scratch")?;
        let p = self.capture.context("GLM prompt missing owner")?.rows();
        ensure!(
            prior == Phase::PrimerRead && stream == self.stream,
            "GLM prompt primer completion order/stream"
        );
        let mut hash = kv::Probe::hash(b"atlas/glm53/mtp-kv/prefix/v1\0", p - 1);
        let mut composed = kv::Probe::hash(b"atlas/glm53/mtp-kv/prefix/v1\0", p);
        let binding = kv::read_interval(cache, blocks, 0, p - 1, ctx, stream, &mut scratch, |b| {
            hash.update(b);
            composed.update(b);
        })?;
        self.evidence.primer_kv = hash.finalize().into();
        self.continuation = Some(composed);
        self.phase = Phase::PrimerWritten;
        self.pool = Some(binding);
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn bootstrap_before(
        &mut self,
        input: &crate::speculative::glm_repair::RepairInput<'_>,
        tokens: &[u32],
        source: DeviceSpan,
        row: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let prior = std::mem::replace(&mut self.phase, Phase::Failed);
        let c = self.capture.context("GLM prompt missing owner")?;
        let p = c.rows();
        ensure!(
            prior == Phase::PrimerWritten
                && c.same_owner(
                    input.capture.ptr,
                    input.capture.bytes,
                    input.generation,
                    input.prompt_len
                )
                && input.capture_generation == input.generation
                && input.captured_rows == p
                && input.position == p + 1
                && input.hidden_row == 0
                && row == p - 1
                && tokens.len() == 1
                && input.tokens.get(p..p + 1) == Some(tokens)
                && c.matches(source.ptr, source.bytes, p - 1, 1),
            "GLM prompt bootstrap owner/span/phase"
        );
        let mut scratch = vec![0; 32768].into_boxed_slice();
        self.evidence.bootstrap_source = source_hash(
            b"atlas/glm53/mtp-source/bootstrap/v1\0",
            p - 1,
            source,
            ctx,
            stream,
            &mut scratch,
        )?;
        self.evidence.bootstrap_token =
            token_hash(b"atlas/glm53/mtp-source/bootstrap-token/v1\0", p, tokens);
        self.stream = stream;
        self.scratch = Some(scratch);
        self.phase = Phase::BootstrapRead;
        Ok(())
    }
    pub fn bootstrap_after(
        &mut self,
        cache: &PagedKvCache,
        blocks: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let prior = std::mem::replace(&mut self.phase, Phase::Failed);
        let mut scratch = self.scratch.take().context("GLM prompt missing scratch")?;
        let p = self.capture.context("GLM prompt missing owner")?.rows();
        ensure!(
            prior == Phase::BootstrapRead && stream == self.stream,
            "GLM prompt bootstrap completion order/stream"
        );
        self.pool
            .context("GLM prompt missing KV owner")?
            .validate(cache, blocks, ctx, stream)?;
        let mut hash = kv::Probe::hash(b"atlas/glm53/mtp-kv/appended/v1\0", p - 1);
        let mut composed = self
            .continuation
            .take()
            .context("GLM prompt missing continuation")?;
        let binding = kv::read_interval(cache, blocks, p - 1, 1, ctx, stream, &mut scratch, |b| {
            hash.update(b);
            composed.update(b);
        })?;
        self.evidence.bootstrap_kv = hash.finalize().into();
        self.evidence.written_prefix = composed.finalize().into();
        self.phase = Phase::Complete;
        self.pool = Some(binding);
        Ok(())
    }
    pub fn validate_body(
        &self,
        cache: &PagedKvCache,
        blocks: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.phase == Phase::Complete,
            "GLM prompt writer evidence incomplete"
        );
        self.pool
            .context("GLM prompt missing completed KV owner")?
            .validate(cache, blocks, ctx, stream)
    }
}

fn token_hash(domain: &[u8], index: usize, tokens: &[u32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((index as u64).to_le_bytes());
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hash.finalize().into()
}
fn source_hash(
    domain: &[u8],
    index: usize,
    source: DeviceSpan,
    ctx: &ForwardContext,
    stream: u64,
    scratch: &mut [u8],
) -> Result<[u8; 32]> {
    ensure!(
        !ctx.graph_capture
            && !ctx.gpu.stream_is_capturing(stream)
            && scratch.len() == 32768
            && source.bytes > 0
            && source.bytes <= 255 * 8192
            && source.bytes % 8192 == 0,
        "GLM prompt source capture/size"
    );
    source.end()?;
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((index as u64).to_le_bytes());
    hash.update(4096u32.to_le_bytes());
    hash.update(2u32.to_le_bytes());
    for offset in (0..source.bytes).step_by(scratch.len()) {
        let n = (source.bytes - offset).min(scratch.len());
        ctx.gpu
            .copy_d2h_on_stream(source.ptr.offset(offset), &mut scratch[..n], stream)?;
        hash.update(&scratch[..n]);
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
pub(super) fn fixture_observe(
    head: &Glm5MtpHead,
    state: &mut Glm5MtpProposerState,
    generation: u64,
    p: usize,
    ctx: &ForwardContext,
) -> Result<()> {
    // Existing proposal fixtures skip eager prefill. Seed their real pool/source
    // bytes and run the production observation phases, never fabricate hashes.
    let source = ctx.gpu.alloc(p * 8192)?;
    ctx.gpu.copy_h2d(&vec![0x51; p * 8192], source)?;
    let capture = crate::model::glm_mtp_prompt_trace::fixture_capture(source, p, p, generation, 0)?;
    let tokens: Vec<_> = (0..=p).map(|i| (i % 8) as u32).collect();
    let mut cache = head.kv_cache.lock();
    while state.block_table.len() < p.div_ceil(16) {
        state.block_table.push(cache.alloc_block()?);
    }
    let mut phase = Prompt::default();
    phase.arm(capture)?;
    phase.primer_before(
        &tokens[1..p],
        DeviceSpan {
            ptr: source,
            bytes: (p - 1) * 8192,
        },
        ctx,
        7,
    )?;
    phase.primer_after(&cache, &state.block_table, ctx, 7)?;
    let span = crate::speculative::glm_repair::RepairSpan {
        ptr: source,
        bytes: p * 8192,
    };
    let input = crate::speculative::glm_repair::RepairInput {
        token: 3,
        tokens: &tokens,
        prompt_len: p,
        position: p + 1,
        drafts: 4,
        generation,
        capture_generation: generation,
        captured_rows: p,
        context_tokens: 2044,
        capture: span,
        normalized: span,
        bonus: span,
        hidden_row: 0,
    };
    phase.bootstrap_before(
        &input,
        &tokens[p..],
        DeviceSpan {
            ptr: source.offset((p - 1) * 8192),
            bytes: 8192,
        },
        p - 1,
        ctx,
        7,
    )?;
    phase.bootstrap_after(&cache, &state.block_table, ctx, 7)?;
    state.hidden_trace.prompt = phase;
    Ok(())
}
