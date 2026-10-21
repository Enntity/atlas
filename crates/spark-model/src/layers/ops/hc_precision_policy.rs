// SPDX-License-Identifier: AGPL-3.0-only

//! Select the GLM highway format without coupling native MTP to DFlash.

pub(super) fn bf16_highway(
    model_type: &str,
    requested: bool,
    dflash: bool,
    strict_native_mtp: bool,
) -> bool {
    model_type == "glm5_next" && requested && (dflash || strict_native_mtp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_mtp_keeps_the_requested_bf16_target_without_dflash() {
        assert!(bf16_highway("glm5_next", true, false, true));
        assert!(bf16_highway("glm5_next", true, true, false));
        assert!(!bf16_highway("glm5_next", true, false, false));
        assert!(!bf16_highway("glm5_next", false, false, true));
        assert!(!bf16_highway("qwen3_next", true, true, true));
    }
}
