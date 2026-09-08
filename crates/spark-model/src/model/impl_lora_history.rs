// SPDX-License-Identifier: AGPL-3.0-only
//! Read the actual live model owners and sticky install history for diagnostics.
use super::TransformerModel;
use crate::layers::glm5_mtp::hidden_trace::AdapterOwnership;

impl TransformerModel {
    pub(crate) fn hidden_trace_adapter_ownership(&self) -> AdapterOwnership {
        AdapterOwnership {
            pool: self.lora.is_some(),
            overlays: self.overlays.is_some(),
            rotatable: self.lora_rotatable,
            install_attempted: self.lora_install_attempted,
        }
    }
}

#[cfg(test)]
#[path = "impl_lora_history_tests.rs"]
mod tests;
