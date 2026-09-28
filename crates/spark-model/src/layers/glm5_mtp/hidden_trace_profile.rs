// SPDX-License-Identifier: AGPL-3.0-only
//! Exact shared cold-request guard for proposal and prompt observations.
use super::*;
pub(super) struct ColdProfile {
    adapter_id: u64,
    adapter_slot: i32,
    prefix_tokens: usize,
    prefix_blocks: usize,
    marconi: usize,
    disk_empty: bool,
}
impl ColdProfile {
    pub fn from(seq: &SequenceState) -> Self {
        Self {
            adapter_id: seq.adapter_id as u64,
            adapter_slot: seq.adapter_slot,
            prefix_tokens: seq.cached_prefix_tokens,
            prefix_blocks: seq.cached_prefix_blocks,
            marconi: seq.marconi_skip_to,
            disk_empty: seq.disk_block_ids.is_empty(),
        }
    }
    pub fn validate(
        &self,
        ctx: &ForwardContext,
        drafts: usize,
        grammar: bool,
        owners: impl FnOnce() -> AdapterOwnership,
    ) -> Result<()> {
        self.validate_request(ctx, owners)?;
        ensure!(
            drafts == 4 && !grammar,
            "GLM hidden trace requires requested MTP4 without grammar"
        );
        Ok(())
    }
    pub fn validate_request(
        &self,
        ctx: &ForwardContext,
        owners: impl FnOnce() -> AdapterOwnership,
    ) -> Result<()> {
        owners().ensure_absent()?;
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
                && self.adapter_id == 0
                && self.adapter_slot < 0
                && self.prefix_tokens == 0
                && self.prefix_blocks == 0
                && self.marconi == 0
                && self.disk_empty
                && ctx.config.adapter_max_rank == 0
                && ctx.routed_lora_layers.is_none()
                && !matches!(ctx.moe_lora_route, crate::layer::MoeLoraRoute::Refuse),
            "GLM hidden trace requires exact cold C1 TP2/EP2 MTP4 repair profile without adapters"
        );
        Ok(())
    }
}
