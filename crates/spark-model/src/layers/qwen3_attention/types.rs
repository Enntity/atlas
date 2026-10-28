// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 attention type exports.

pub use super::types_layer::Qwen3AttentionLayer;
pub use super::types_weights::{HcWeights, MlaWeights};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeadGateActivation {
    Sigmoid,
    Softplus,
}
