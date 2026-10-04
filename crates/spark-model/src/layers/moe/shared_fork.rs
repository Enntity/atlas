// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_LAYER_FORK=moe|1` (`layers/glm_layer_fork.rs`): the TP-split
//! shared expert of a grouped routed FFN on the layer's auxiliary stream.
//!
//! `forward_prefill_mode` forks after the router GEMV (and its LoRA fold) and
//! joins before the unpermute, the first reader of the shared output. Inside
//! the fork the shared expert reads the normed `input` and writes
//! `ssm_deinterleaved`, `ssm_qkvz` and `attn_output`; the compute stream reads
//! `input` and writes `scratch`, `gate_logits`, `moe_router_in_f32` and the
//! `expert_*` arenas. The B-tile reader and the pre-expert norm also write
//! the shared expert's buffers, and a LoRA fold has its own scratch, so those
//! layers keep the in-line order.

use super::*;
use crate::layers::glm_layer_fork::{self, ForkLane};

impl MoeLayer {
    /// The lane the split shared expert forks onto, when `split` and the
    /// switch, the model and the stream allow it.
    pub(super) fn shared_fork(
        &self,
        split: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Option<ForkLane> {
        let ready = split
            && glm_layer_fork::mode().is_ok_and(|m| m.moe)
            && ctx.config.model_type == "glm5_next"
            && glm_layer_fork::eager(ctx, stream)
            && self.pre_expert_norm.is_none()
            && self.lora.is_none()
            && !self.btile_storage.is_published();
        ready.then_some(ForkLane {
            side: self.prefill_stream,
            fork: self.event_a,
            join: self.event_b,
        })
    }
}
