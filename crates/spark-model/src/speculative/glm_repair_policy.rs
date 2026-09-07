// SPDX-License-Identifier: AGPL-3.0-only

//! Rollout gating only. Actual positions, state ownership and storage are
//! validated separately immediately before each existing E1 command.

use anyhow::{Result, ensure};

pub fn enabled() -> bool {
    std::env::var("ATLAS_GLM_MTP_REPAIR").as_deref() == Ok("1")
}

pub fn parse(raw: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(raw, None | Some("0") | Some("1")),
        "ATLAS_GLM_MTP_REPAIR must be 0 or 1"
    );
    Ok(raw == Some("1"))
}

#[derive(Clone, Copy, Debug)]
pub struct RepairPolicy<'a> {
    pub model_type: &'a str,
    pub world: usize,
    pub tp: usize,
    pub ep: usize,
    pub active: usize,
    pub admitted: usize,
    pub context: usize,
    pub drafts: usize,
    pub native_only: bool,
    pub bf16: bool,
    pub prefix_reuse: bool,
    pub force: bool,
}

impl RepairPolicy<'_> {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.model_type == "glm5_next"
                && self.world == 2
                && self.tp == 2
                && self.ep == 2
                && self.active == 1
                && self.admitted == 1
                && self.drafts == 4
                && self.native_only
                && self.bf16
                && !self.prefix_reuse,
            "GLM repair requires native MTP4, GLM TP2/EP2, active/admitted C1, BF16, and no prefix reuse"
        );
        ensure!(
            self.context >= 4 && self.context.checked_add(4).is_some_and(|n| n <= 2048),
            "GLM repair context + four drafts must be <=2048 with four staging rows"
        );
        ensure!(self.force, "GLM repair requires resolved MTP gate force");
        Ok(())
    }
}

/// Explicit resolved drafter policy matters: old CARRY=0 variables are ignored.
pub fn validate_environment() -> Result<()> {
    for name in [
        "ATLAS_GLM_MTP_DISTRIBUTED",
        "ATLAS_GLM_MTP_BATCHED_PREFILL",
        "ATLAS_MTP_SPEC_THINK",
        "ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE",
    ] {
        ensure!(
            std::env::var(name).as_deref() == Ok("1"),
            "GLM repair requires {name}=1"
        );
    }
    for name in [
        "ATLAS_GLM_MTP_SERIAL_PREFILL",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_SINGLE_DEPTH_ADAPT",
        "ATLAS_DFLASH_ADAPTIVE",
        "ATLAS_DFLASH_RESUME_GUARD",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_MTP_REFEED_ACCEPTED",
    ] {
        ensure!(
            matches!(std::env::var(name).ok().as_deref(), None | Some("0")),
            "GLM repair requires {name} disabled"
        );
    }
    ensure!(
        crate::speculative::draft_conf_tau() == 0.0,
        "GLM repair cannot discard unverified drafts by confidence"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy() -> RepairPolicy<'static> {
        RepairPolicy {
            model_type: "glm5_next",
            world: 2,
            tp: 2,
            ep: 2,
            active: 1,
            admitted: 1,
            context: 2044,
            drafts: 4,
            native_only: true,
            bf16: true,
            prefix_reuse: false,
            force: true,
        }
    }
    #[test]
    fn exact_lane_and_overflow_checked() {
        assert!(policy().validate().is_ok());
        for context in [0, 3, 2045, usize::MAX] {
            assert!(
                RepairPolicy {
                    context,
                    ..policy()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            RepairPolicy {
                admitted: 2,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepairPolicy {
                prefix_reuse: true,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepairPolicy {
                force: false,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepairPolicy {
                drafts: 3,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(parse(Some("true")).is_err());
        assert!(!parse(None).unwrap());
    }
}
