// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-registered semantic "must-fail" perturbation knobs for the native
//! EXL3 GLM-5.3 model path (statistical-gate controls). Parsed and validated
//! once per model load from `ATLAS_EXL3_GATE_PERTURBATION`; never per token.
//! Any unknown or invalid value is a startup error.

/// Environment variable selecting at most one gate control.
pub const ENV: &str = "ATLAS_EXL3_GATE_PERTURBATION";

/// The named `topk7_layer` control reduces exactly the reference top-8.
const TOPK7_BASE: usize = 8;
/// The named `router_scale_2p4` control replaces exactly the reference 2.5.
const ROUTER_SCALE_BASE: f64 = 2.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3GatePerturbation {
    /// mHC post multiplier 2.0 -> 1.9 (every native mHC site).
    MhcPost19,
    /// Routed scaling factor 2.5 -> 2.4 (every routed MoE layer).
    RouterScale24,
    /// Router top-k 8 -> 7 on one routed MoE layer.
    Topk7Layer { layer: u32 },
    /// Skip the shared expert on one routed MoE layer.
    DropSharedLayer { layer: u32 },
    /// Round the running routed-expert sum to BF16 after every addition.
    Bf16RoutedAccum,
}

impl Exl3GatePerturbation {
    /// Pure parse of the variable's value (`None` = unset). Empty means unset.
    /// Layer indices are only checked syntactically here; model geometry is
    /// checked by [`Self::validate_for_model`].
    pub fn parse(value: Option<&str>) -> Result<Option<Self>, String> {
        let Some(value) = value else {
            return Ok(None);
        };
        let s = value.trim();
        if s.is_empty() {
            return Ok(None);
        }
        let layer = |name: &str, rest: &str| {
            rest.parse::<u32>()
                .map_err(|_| format!("{ENV}: bad layer index in {name}={rest:?}"))
        };
        match s {
            "mhc_post_1p9" => Ok(Some(Self::MhcPost19)),
            "router_scale_2p4" => Ok(Some(Self::RouterScale24)),
            "bf16_routed_accum" => Ok(Some(Self::Bf16RoutedAccum)),
            _ => {
                if let Some(rest) = s.strip_prefix("topk7_layer=") {
                    Ok(Some(Self::Topk7Layer {
                        layer: layer("topk7_layer", rest)?,
                    }))
                } else if let Some(rest) = s.strip_prefix("drop_shared_layer=") {
                    Ok(Some(Self::DropSharedLayer {
                        layer: layer("drop_shared_layer", rest)?,
                    }))
                } else {
                    Err(format!(
                        "unknown {ENV} value {s:?}; expected one of mhc_post_1p9, \
                         router_scale_2p4, topk7_layer=<L>, drop_shared_layer=<L>, \
                         bf16_routed_accum"
                    ))
                }
            }
        }
    }

    /// Thin wrapper over [`Self::parse`] reading the process environment.
    pub fn parse_env() -> Result<Option<Self>, String> {
        match std::env::var(ENV) {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(format!("{ENV} is not valid UTF-8")),
        }
    }

    /// Model-load entry point: parse the environment once, validate it
    /// against `config`, and announce an active control. Errors abort load.
    pub fn load_for_model(
        config: &atlas_core::config::ModelConfig,
    ) -> anyhow::Result<Option<Self>> {
        let parsed = Self::parse_env().map_err(anyhow::Error::msg)?;
        if let Some(p) = parsed {
            p.validate_for_model(
                config.num_hidden_layers,
                &config.mlp_only_layers,
                config.num_experts_per_tok,
                config.routed_scaling_factor,
            )
            .map_err(anyhow::Error::msg)?;
            tracing::warn!(
                "{ENV} ACTIVE: {} on the native EXL3 GLM-5.3 path. This is a \
                 statistical-gate must-fail control; outputs are deliberately perturbed.",
                p.banner_name()
            );
        }
        Ok(parsed)
    }

    /// Validate against the loaded model: layer-conditional controls must
    /// name a routed MoE layer (`layer < num_layers` and not in
    /// `mlp_only_layers`); `topk7_layer` requires the reference top-8 and
    /// `router_scale_2p4` the reference routed scale 2.5.
    pub fn validate_for_model(
        self,
        num_layers: usize,
        mlp_only_layers: &[usize],
        experts_per_tok: usize,
        routed_scaling_factor: f64,
    ) -> Result<(), String> {
        let routed = |name: &str, layer: u32| {
            let index = layer as usize;
            if index >= num_layers || mlp_only_layers.contains(&index) {
                let first = (0..num_layers).find(|l| !mlp_only_layers.contains(l));
                return Err(format!(
                    "{ENV}: {name}={layer} is not a routed MoE layer (model has {num_layers} \
                     layers, first routed layer {first:?}, dense layers {mlp_only_layers:?})"
                ));
            }
            Ok(())
        };
        match self {
            Self::Topk7Layer { layer } => {
                routed("topk7_layer", layer)?;
                if experts_per_tok != TOPK7_BASE {
                    return Err(format!(
                        "{ENV}: topk7_layer requires num_experts_per_tok={TOPK7_BASE}, \
                         model has {experts_per_tok}"
                    ));
                }
                Ok(())
            }
            Self::DropSharedLayer { layer } => routed("drop_shared_layer", layer),
            Self::RouterScale24 if routed_scaling_factor != ROUTER_SCALE_BASE => Err(format!(
                "{ENV}: router_scale_2p4 requires routed_scaling_factor={ROUTER_SCALE_BASE}, \
                 model has {routed_scaling_factor}"
            )),
            Self::MhcPost19 | Self::RouterScale24 | Self::Bf16RoutedAccum => Ok(()),
        }
    }

    /// Name for the startup banner.
    pub fn banner_name(self) -> String {
        match self {
            Self::MhcPost19 => "mhc_post_1p9 (mHC post multiplier 2.0 -> 1.9)".into(),
            Self::RouterScale24 => "router_scale_2p4 (routed scaling 2.5 -> 2.4)".into(),
            Self::Topk7Layer { layer } => format!("topk7_layer={layer} (router top-k 8 -> 7)"),
            Self::DropSharedLayer { layer } => {
                format!("drop_shared_layer={layer} (shared expert skipped)")
            }
            Self::Bf16RoutedAccum => {
                "bf16_routed_accum (running routed sum rounded to BF16 per expert)".into()
            }
        }
    }
}

/// Routed scaling factor consumed by the top-k kernel.
pub fn routed_scale(base: f32, p: Option<Exl3GatePerturbation>) -> f32 {
    match p {
        Some(Exl3GatePerturbation::RouterScale24) => 2.4,
        _ => base,
    }
}

/// Router top-k for `layer`.
pub fn layer_topk(base: usize, layer: usize, p: Option<Exl3GatePerturbation>) -> usize {
    match p {
        Some(Exl3GatePerturbation::Topk7Layer { layer: n }) if n as usize == layer => 7,
        _ => base,
    }
}

/// Whether `layer` skips its shared expert.
pub fn drop_shared(layer: usize, p: Option<Exl3GatePerturbation>) -> bool {
    matches!(p, Some(Exl3GatePerturbation::DropSharedLayer { layer: n }) if n as usize == layer)
}

/// Whether the routed combine rounds its running sum to BF16 per expert.
pub fn bf16_routed_accum(p: Option<Exl3GatePerturbation>) -> bool {
    matches!(p, Some(Exl3GatePerturbation::Bf16RoutedAccum))
}

#[cfg(test)]
mod tests {
    use super::Exl3GatePerturbation as P;
    use super::*;

    #[test]
    fn unset_and_empty_parse_to_none() {
        assert_eq!(P::parse(None), Ok(None));
        assert_eq!(P::parse(Some("")), Ok(None));
        assert_eq!(P::parse(Some("  ")), Ok(None));
    }

    #[test]
    fn every_named_value_parses() {
        for (text, expected) in [
            ("mhc_post_1p9", P::MhcPost19),
            ("router_scale_2p4", P::RouterScale24),
            ("bf16_routed_accum", P::Bf16RoutedAccum),
            (" topk7_layer=3 ", P::Topk7Layer { layer: 3 }),
            ("topk7_layer=44", P::Topk7Layer { layer: 44 }),
            ("drop_shared_layer=17", P::DropSharedLayer { layer: 17 }),
        ] {
            assert_eq!(P::parse(Some(text)), Ok(Some(expected)), "{text}");
        }
    }

    #[test]
    fn invalid_values_are_errors() {
        for text in [
            "bad_value",
            "MHC_POST_1P9",
            "mhc_post_1p9,router_scale_2p4",
            "topk7_layer=",
            "topk7_layer=foo",
            "topk7_layer=-1",
            "drop_shared_layer=3.0",
            "topk7_layer",
        ] {
            assert!(P::parse(Some(text)).is_err(), "{text}");
        }
    }

    #[test]
    fn layer_controls_must_name_a_routed_layer_of_the_model() {
        let dense = [0, 1, 2];
        for layer in [3, 44] {
            assert_eq!(
                P::Topk7Layer { layer }.validate_for_model(45, &dense, 8, 2.5),
                Ok(())
            );
            assert_eq!(
                P::DropSharedLayer { layer }.validate_for_model(45, &dense, 8, 2.5),
                Ok(())
            );
        }
        for layer in [0, 2, 45, 1000] {
            assert!(
                P::Topk7Layer { layer }
                    .validate_for_model(45, &dense, 8, 2.5)
                    .is_err()
            );
            assert!(
                P::DropSharedLayer { layer }
                    .validate_for_model(45, &dense, 8, 2.5)
                    .is_err()
            );
        }
        // Geometry comes from the model, not a hardcoded 45/3.
        assert_eq!(
            P::DropSharedLayer { layer: 1 }.validate_for_model(4, &[0], 8, 2.5),
            Ok(())
        );
        assert!(
            P::DropSharedLayer { layer: 4 }
                .validate_for_model(4, &[0], 8, 2.5)
                .is_err()
        );
    }

    #[test]
    fn topk7_requires_reference_top8() {
        assert!(
            P::Topk7Layer { layer: 3 }
                .validate_for_model(45, &[0, 1, 2], 6, 2.5)
                .is_err()
        );
        assert!(
            P::Topk7Layer { layer: 3 }
                .validate_for_model(45, &[0, 1, 2], 7, 2.5)
                .is_err()
        );
        assert_eq!(
            P::MhcPost19.validate_for_model(45, &[0, 1, 2], 6, 2.5),
            Ok(())
        );
    }

    #[test]
    fn router_scale_requires_reference_scale() {
        assert_eq!(
            P::RouterScale24.validate_for_model(45, &[0], 8, 2.5),
            Ok(())
        );
        assert!(
            P::RouterScale24
                .validate_for_model(45, &[0], 8, 1.0)
                .is_err()
        );
        assert_eq!(P::MhcPost19.validate_for_model(45, &[0], 8, 1.0), Ok(()));
    }

    #[test]
    fn routed_scale_only_changes_under_router_scale() {
        assert_eq!(routed_scale(2.5, None), 2.5);
        assert_eq!(routed_scale(2.5, Some(P::RouterScale24)), 2.4);
        assert_eq!(routed_scale(2.5, Some(P::MhcPost19)), 2.5);
        assert_eq!(routed_scale(2.5, Some(P::Topk7Layer { layer: 3 })), 2.5);
    }

    #[test]
    fn topk_changes_only_on_the_named_layer() {
        let p = Some(P::Topk7Layer { layer: 5 });
        assert_eq!(layer_topk(8, 5, p), 7);
        assert_eq!(layer_topk(8, 4, p), 8);
        assert_eq!(layer_topk(8, 6, p), 8);
        assert_eq!(layer_topk(8, 5, None), 8);
        assert_eq!(layer_topk(8, 5, Some(P::DropSharedLayer { layer: 5 })), 8);
    }

    #[test]
    fn shared_drop_only_on_the_named_layer() {
        let p = Some(P::DropSharedLayer { layer: 9 });
        assert!(drop_shared(9, p));
        assert!(!drop_shared(10, p));
        assert!(!drop_shared(9, None));
        assert!(!drop_shared(9, Some(P::Topk7Layer { layer: 9 })));
    }

    #[test]
    fn bf16_accum_flag_only_under_its_control() {
        assert!(bf16_routed_accum(Some(P::Bf16RoutedAccum)));
        assert!(!bf16_routed_accum(None));
        assert!(!bf16_routed_accum(Some(P::RouterScale24)));
    }
}
