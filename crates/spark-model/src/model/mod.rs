// SPDX-License-Identifier: AGPL-3.0-only

//! Generic transformer model.
//!
//! The model loop (embed -> layers -> norm -> lm_head) is architecture-
//! agnostic. Layer-specific logic lives in `TransformerLayer` implementations.
//!
//! Wave 4b1 split: this module was originally a single 8,690 LoC `model.rs`.
//! Sub-modules now hold:
//!   - `types`             struct TransformerModel + PinnedMetaStaging
//!   - `ssm_pool`          SsmStatePool
//!   - `ssm_snapshot`      SsmSnapshotPool
//!   - `block_mgmt`        free-fn helpers (apply_evicted_blocks etc.)
//!   - `impl_a1/2/3`       first inherent `impl TransformerModel` block
//!   - `impl_b1/2/3`       second inherent `impl TransformerModel` block
//!   - `trait_impl`        single `impl Model for TransformerModel` block
//!     **FLAGGED ≤500 LoC cap** — Rust does NOT allow
//!     splitting one trait impl across files (E0119),
//!     and breaking it into inherent-helper delegation
//!     is a semantic refactor outside this wave's scope.
//!   - `drop`              `impl Drop for TransformerModel`
//!   - `tests`             extracted unit tests

#![allow(unused_imports, dead_code)]

pub(crate) mod block_mgmt;
pub(crate) mod drafter_context;
pub(crate) mod drop;
mod glm_c2_handoff;
pub mod glm_c2_pair_policy;
mod glm_c2_pair_transport;
mod glm_c2_pair_verify;
mod glm_c2_sequence_allocation;
mod glm_c2_sequence_ownership;
#[cfg(feature = "glm-c2-test-utils")]
pub mod glm_c2_test_support;
#[cfg(all(test, not(feature = "glm-c2-test-utils")))]
pub(crate) mod glm_c2_test_support;
pub mod glm_c4;
pub(crate) mod glm_cache_plan;
pub mod glm_independent;
pub(crate) mod glm_k3_head;
mod glm_fused_chunk;
mod glm_long_verify;
mod glm_prefill_sp;
mod glm_vocab_split;
pub(crate) use glm_vocab_split::prepare_shard_mxfp8 as prepare_glm_head_mxfp8;
pub(crate) mod glm_mtp_prompt_trace;
mod glm_mtp_repair;
pub(crate) mod glm_owner8_wire;
mod glm_owner_compute;
mod glm_owner_metadata;
mod glm_owner_policy;
mod glm_owner_preflight;
mod glm_owner_transport;
pub(crate) mod glm_owner_wire;
pub use glm_c2_handoff::GlmPairedInput;
pub(crate) mod dspark_generation;
#[cfg(test)]
mod dspark_generation_tests;
pub(crate) mod dspark_pool;
#[cfg(test)]
mod dspark_pool_tests;
mod final_norm;
#[cfg(test)]
mod glm_c2_handoff_tests;
pub(crate) mod impl_a1;
pub(crate) mod impl_a1_init;
pub(crate) mod impl_a2;
pub(crate) mod impl_a3;
mod impl_a3_embed;
pub(crate) mod impl_b1;
pub(crate) mod impl_b2;
pub(crate) mod impl_b3;
pub(crate) mod impl_b3_accessors;
pub(crate) mod impl_b3_dflash;
pub(crate) mod impl_lora;
pub(crate) mod impl_lora_swap;
mod impl_ngram;
pub(crate) mod mtp_carry;
pub(crate) mod pinned_pack;
pub(crate) mod ssm_batched_copy;
pub(crate) mod ssm_indexed_decode;
pub(crate) mod ssm_pool;
pub(crate) mod ssm_pools;
pub(crate) mod ssm_snapshot;
pub(crate) mod ssm_snapshot_faultin;
pub(crate) mod ssm_snapshot_spill;
mod ssm_snapshot_teardown;
pub(crate) mod ssm_spill_gate;
pub(crate) mod ssm_spill_staging;
pub(crate) mod ssm_tier;
pub(crate) mod token_overlay;
pub(crate) mod trait_impl;
pub(crate) mod types;
pub(crate) mod vision_transport;

// Served NLLB-200 / M2M-100 encoder-decoder model (CUDA/GB10 serving path).
#[cfg(feature = "cuda")]
pub mod nllb;

pub use types::TransformerModel;
