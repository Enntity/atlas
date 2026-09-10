// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM reasoning-to-tool boundary, resolved from actual model metadata.
//! This is not text salvage: callers must match the tokenizer's native opener.
pub(crate) fn native_opener(
    model_type: &str,
    tools_active: bool,
    native: Option<u32>,
) -> Option<u32> {
    if model_type == "glm5_next" && tools_active {
        native
    } else {
        None
    }
}
