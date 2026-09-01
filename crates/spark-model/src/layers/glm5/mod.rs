// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM-5.3-Flash decoder layer.

mod core;
mod decode;
mod decode_kda;
mod ffn;
mod init;
mod kda_projection;
mod prefill;
mod prefill_batched;
mod prefill_layer;
mod types;
mod verify;
mod verify_dsa;
mod verify_dsa_multi;
mod verify_kda;
mod verify_multi;
mod verify_prefill;
mod verify_prefill_kda;
mod verify_state;

pub(crate) use decode::dsa_verify_pool_bucket;
pub use init::validate_kernel_contract;
pub use types::{
    DsaWeights, Glm5Layer, GlmAttentionWeights, GlmDenseFfnWeights, GlmExl3MoeWeights, GlmFfn,
    GlmHcWeights, GlmSharedExpertSchedule, KdaWeights,
};
