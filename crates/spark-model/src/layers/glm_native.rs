// SPDX-License-Identifier: AGPL-3.0-only

//! nvidia GLM-5.3-Flash-FP4-Local served in its checkpoint formats (default
//! off): each layer group behind its own switch, and one umbrella.
//!
//! | switch                              | group (checkpoint format)                 |
//! |-------------------------------------|-------------------------------------------|
//! | `ATLAS_GLM_FP4L_MLA_FP8`            | MLA q_a / q_b / kv_a / o (FP8_PB_WO)      |
//! | `ATLAS_GLM_FP4L_SHARED_FP8`         | shared experts 3..44 (FP8_PB_WO)          |
//! | `ATLAS_GLM_FP4L_DENSE_MXFP8`        | dense MLP 0..2 (MXFP8)                    |
//! | `ATLAS_GLM_FP4L_KDA_MXFP8_DECODE`   | KDA q/k/v/o decode + verify (MXFP8)       |
//! | `ATLAS_GLM_FP4L_KDA_W8A8_PREFILL`   | KDA q/k/v/o prefill (MXFP8 W8A8)          |
//! | `ATLAS_GLM_FP4L_NATIVE`             | all of the above                          |
//!
//! `ATLAS_GLM_FP4L_FP8_PREFILL` picks how the FP8 block groups run prefill
//! (`ops::Fp8bPrefill`: `w8a8` default, `w8a16`, `bf16`); the MXFP8 dense MLP
//! takes W8A8 under `w8a8` and the exact BF16 expansion otherwise.
//!
//! Each switch is inert on a checkpoint that does not ship its group in that
//! format (logged once by the loader). Values: `1` / `true` on, anything
//! else off, like the loader's other flags.

/// Which layer groups load their checkpoint format natively.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativePolicy {
    pub mla_fp8: bool,
    pub shared_fp8: bool,
    pub dense_mxfp8: bool,
    pub kda_decode: bool,
    pub kda_w8a8: bool,
}

impl NativePolicy {
    pub const UMBRELLA: &'static str = "ATLAS_GLM_FP4L_NATIVE";
    pub const MLA: &'static str = "ATLAS_GLM_FP4L_MLA_FP8";
    pub const SHARED: &'static str = "ATLAS_GLM_FP4L_SHARED_FP8";
    pub const DENSE: &'static str = "ATLAS_GLM_FP4L_DENSE_MXFP8";
    pub const KDA_DECODE: &'static str = "ATLAS_GLM_FP4L_KDA_MXFP8_DECODE";
    pub const KDA_W8A8: &'static str = "ATLAS_GLM_FP4L_KDA_W8A8_PREFILL";

    /// The policy from `get` (an environment lookup).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let on = |name: &str| get(name).is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let all = on(Self::UMBRELLA);
        Self {
            mla_fp8: all || on(Self::MLA),
            shared_fp8: all || on(Self::SHARED),
            dense_mxfp8: all || on(Self::DENSE),
            kda_decode: all || on(Self::KDA_DECODE),
            kda_w8a8: all || on(Self::KDA_W8A8),
        }
    }

    /// The policy as this process's environment sets it.
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// The groups as one word, for the ranks' startup agreement.
    pub fn bits(self) -> u64 {
        [
            self.mla_fp8,
            self.shared_fp8,
            self.dense_mxfp8,
            self.kda_decode,
            self.kda_w8a8,
        ]
        .iter()
        .enumerate()
        .map(|(i, &on)| (on as u64) << i)
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(vars: &[(&str, &str)]) -> NativePolicy {
        NativePolicy::from_lookup(|name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        })
    }

    #[test]
    fn everything_is_off_by_default() {
        assert_eq!(policy(&[]), NativePolicy::default());
        assert_eq!(policy(&[(NativePolicy::UMBRELLA, "0")]).bits(), 0);
    }

    #[test]
    fn the_umbrella_turns_every_group_on() {
        let all = policy(&[(NativePolicy::UMBRELLA, "1")]);
        assert_eq!(all.bits(), 0b11111);
        assert_eq!(policy(&[(NativePolicy::UMBRELLA, "TRUE")]), all);
    }

    #[test]
    fn each_switch_turns_its_group_on_alone() {
        for (i, name) in [
            NativePolicy::MLA,
            NativePolicy::SHARED,
            NativePolicy::DENSE,
            NativePolicy::KDA_DECODE,
            NativePolicy::KDA_W8A8,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(policy(&[(name, "1")]).bits(), 1 << i, "{name}");
            assert_eq!(policy(&[(name, "2")]).bits(), 0, "{name}");
        }
    }
}
