// SPDX-License-Identifier: AGPL-3.0-only

use super::kv_rows_plan::DeviceSpan;
use super::repair_state::RepairPhase;
use super::*;
use crate::speculative::glm_pair_plan::{
    BootstrapInput, FinishPlan, Limits, Profile, ProposalPlan,
};
use crate::speculative::glm_repair::{GlmPairRepair, RepairInput, RepairSpan};
use anyhow::{Context, ensure};

struct Prepared {
    finish: FinishPlan,
    next: ProposalPlan,
    tokens: Vec<u32>,
    source: DeviceSpan,
    stage_from: Option<DevicePtr>,
    needed_blocks: usize,
}

fn span(raw: RepairSpan) -> Result<DeviceSpan> {
    ensure!(
        raw.ptr.0 != 0 && raw.ptr.0.is_multiple_of(2),
        "GLM repair missing/unaligned storage"
    );
    let result = DeviceSpan {
        ptr: raw.ptr,
        bytes: raw.bytes,
    };
    result.end()?;
    Ok(result)
}

impl Glm5MtpHead {
    fn plan_repair(
        &self,
        input: &RepairInput<'_>,
        state: &Glm5MtpProposerState,
        ctx: &ForwardContext,
    ) -> Result<Prepared> {
        ensure!(
            input.generation != 0 && input.generation == input.capture_generation,
            "GLM repair does not own capture generation"
        );
        ensure!(
            input.tokens.len() == input.position && input.drafts == 4,
            "GLM repair requires complete live target tokens and four drafts"
        );
        ensure!(
            (input.token as usize) < ctx.config.vocab_size,
            "GLM repair pending token outside vocabulary"
        );
        ensure!(
            ctx.config.model_type == "glm5_next" && !ctx.graph_capture,
            "GLM repair is GLM eager proposer only"
        );
        let row_bytes = ctx
            .config
            .hidden_size
            .checked_mul(2)
            .context("GLM repair row overflow")?;
        ensure!(row_bytes > 0, "GLM repair empty hidden width");
        let capture = span(input.capture)?;
        let normalized = span(input.normalized)?;
        let bonus = span(input.bonus)?;
        ensure!(
            capture.bytes / row_bytes >= 4
                && normalized.bytes / row_bytes >= 5
                && bonus.bytes >= row_bytes,
            "GLM repair storage capacity is insufficient"
        );
        ensure!(
            !capture.overlaps(normalized)?
                && !capture.overlaps(bonus)?
                && !normalized.overlaps(bonus)?,
            "GLM repair capture/target/bonus alias"
        );
        let cache = self.kv_cache.lock();
        let capacity = cache
            .num_blocks()
            .checked_mul(cache.block_size())
            .context("GLM repair cache overflow")?;
        let limits = Limits::new(
            Profile {
                sequences: 1,
                drafts: 4,
                continuous: true,
                grammar: false,
                adaptive_depth: false,
                catchup: false,
                carry: false,
                prefix_reuse: false,
            },
            input.context_tokens,
            capacity,
            capture.bytes / row_bytes,
        )?;
        let (finish, tokens, source, stage_from) = match state.repair {
            RepairPhase::Capture => {
                ensure!(
                    input.hidden_row == 0,
                    "GLM bootstrap requires decoded hidden row zero"
                );
                ensure!(
                    input.captured_rows <= capture.bytes / row_bytes,
                    "GLM capture metadata exceeds owned allocation"
                );
                let finish = limits.bootstrap(BootstrapInput {
                    generation: input.generation,
                    capture_generation: input.capture_generation,
                    prompt_tokens: input.prompt_len,
                    target_position: input.position,
                    token_rows: input.tokens.len(),
                    normalized_hidden_rows: input.captured_rows,
                    cached_rows: state.seq_len,
                })?;
                let write = finish.write().context("GLM bootstrap missing pair span")?;
                let offset = write
                    .hidden_start()
                    .checked_mul(row_bytes)
                    .context("GLM repair source overflow")?;
                let bytes = write
                    .rows()
                    .checked_mul(row_bytes)
                    .context("GLM repair span overflow")?;
                ensure!(
                    offset
                        .checked_add(bytes)
                        .is_some_and(|end| end <= capture.bytes),
                    "GLM bootstrap source exceeds capture allocation"
                );
                (
                    finish,
                    input.tokens[write.token_start()..write.token_start() + write.rows()].to_vec(),
                    DeviceSpan {
                        ptr: capture.ptr.offset(offset),
                        bytes,
                    },
                    None,
                )
            }
            RepairPhase::Pending(_) => {
                let pending =
                    state
                        .repair
                        .pending(input.generation, input.position, input.hidden_row)?;
                ensure!(
                    state.seq_len == pending.cached_rows && state.last_num_drafted == 4,
                    "GLM pending private cache changed after verified record"
                );
                let finish = pending.plan;
                let rows = finish.write().map_or(0, |write| write.rows());
                let bytes = rows
                    .checked_mul(row_bytes)
                    .context("GLM staged span overflow")?;
                ensure!(
                    rows <= 4 && bytes <= capture.bytes && bytes <= normalized.bytes,
                    "GLM accepted source exceeds live storage"
                );
                let base = input
                    .position
                    .checked_sub(rows + 1)
                    .context("GLM accepted target underflow")?;
                ensure!(
                    input.tokens.get(base..input.position) == Some(&pending.tokens[..rows + 1]),
                    "GLM accepted tokens no longer match verified inputs"
                );
                (
                    finish,
                    pending.tokens[1..1 + rows].to_vec(),
                    DeviceSpan {
                        ptr: capture.ptr,
                        bytes,
                    },
                    (rows > 0).then_some(normalized.ptr),
                )
            }
            _ => anyhow::bail!("GLM repair cannot resume an outstanding/failed proposal"),
        };
        let next = limits.propose(
            finish.state(),
            input.generation,
            input.position,
            finish.state().cache_rows(),
            input.drafts,
        )?;
        let needed_blocks = next
            .speculative_cache_end()
            .checked_add(1)
            .context("GLM reserve overflow")?
            .div_ceil(cache.block_size());
        self.validate_kv_blocks(&cache, &state.block_table)?;
        ensure!(
            needed_blocks.saturating_sub(state.block_table.len()) <= cache.num_free_blocks(),
            "GLM repair private cache exhausted"
        );
        if !tokens.is_empty() {
            self.validate_kv_inputs(&tokens, source, ctx, &cache)?;
        }
        Ok(Prepared {
            finish,
            next,
            tokens,
            source,
            stage_from,
            needed_blocks,
        })
    }
}

impl GlmPairRepair for Glm5MtpHead {
    fn validate_prepare(
        &self,
        input: &RepairInput<'_>,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("GLM repair requires GLM-owned state")?;
        self.plan_repair(input, state, ctx).map(|_| ())
    }

    fn prepare(
        &self,
        input: &RepairInput<'_>,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("GLM repair requires GLM-owned state")?;
        let result = (|| {
            let plan = self.plan_repair(input, state, ctx)?;
            let bootstrap = matches!(state.repair, RepairPhase::Capture);
            // From the first mutable operation onward failure cannot fall back.
            state.repair = RepairPhase::Failed;
            {
                let mut cache = self.kv_cache.lock();
                while state.block_table.len() < plan.needed_blocks {
                    state.block_table.push(cache.alloc_block()?);
                }
            }
            if let Some(source) = plan.stage_from {
                ctx.gpu
                    .copy_d2d_async(source, plan.source.ptr, plan.source.bytes, stream)?;
            }
            if let Some(write) = plan.finish.write() {
                if bootstrap && state.hidden_trace.prompt.active() {
                    state.hidden_trace.prompt.bootstrap_before(
                        input,
                        &plan.tokens,
                        plan.source,
                        write.cache_start(),
                        ctx,
                        stream,
                    )?;
                }
                if let Err(error) = self.write_kv_rows(
                    &plan.tokens,
                    plan.source,
                    write.cache_start(),
                    &state.block_table,
                    ctx,
                    stream,
                ) {
                    state.hidden_trace.prompt.fail();
                    return Err(error);
                }
                if bootstrap && state.hidden_trace.prompt.active() {
                    let cache = self.kv_cache.lock();
                    state.hidden_trace.prompt.bootstrap_after(
                        &cache,
                        &state.block_table,
                        ctx,
                        stream,
                    )?;
                }
            }
            state.seq_len = plan.finish.state().cache_rows();
            state.last_num_drafted = 0;
            state.repair = RepairPhase::Proposed(plan.next);
            Ok(())
        })();
        if result.is_err() {
            state.hidden_trace.prompt.fail();
        }
        result
    }
}
