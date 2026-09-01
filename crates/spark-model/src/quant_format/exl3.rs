// SPDX-License-Identifier: AGPL-3.0-only

use super::QuantFormat;
use crate::weight_map::Nvfp4Variant;

/// EXL3 is consumed by GLM's architecture-specific mixed-dtype loader rather
/// than the generic NVFP4/FP8 weight-map paths.
#[derive(Debug, Default)]
pub struct Exl3Format;

impl QuantFormat for Exl3Format {
    fn name(&self) -> &'static str {
        "exl3"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        // Generic callers only see GLM's native dense tensors; routed experts
        // are dispatched by Glm5NextWeightLoader before this legacy variant is
        // relevant. Bf16Raw is therefore the honest compatibility sentinel.
        Nvfp4Variant::Bf16Raw
    }

    fn is_ignored(&self, _module_path: &str) -> bool {
        false
    }
}
