// SPDX-License-Identifier: AGPL-3.0-only

//! Probe + format detection against the REAL FP4-Local namespace.
//!
//! `test_data/qwen4_exp_fp4local_headers.json` is every tensor header of
//! nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6 except routed experts other
//! than 0 and 256 (`scripts/qwen4_exp_fp4local_header_extract.py`). The store
//! is rebuilt the way the loaders leave it: CSF pairs become one decoded
//! `weight_scale`, n-gram table shards are deferred. The "old release" store
//! is the same namespace with the MXFP8 backbone turned back into BF16 and
//! the norms into BF16, which is what the NVFP4 release ships.

use super::*;
use crate::weight_loader::qwen4_exp::audit_namespace;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::nvfp4_csf::{CsfPart, csf_part};
use spark_runtime::weights::{DeferredTensor, WeightTensor, is_ngram_table};
use std::collections::HashMap;

fn config() -> ModelConfig {
    let mut c = atlas_core::config::parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_fp4local_config.json"
    )))
    .unwrap();
    c.weight_prefix = "model.language_model".into();
    c
}

/// `(name, dtype, shape)` as the store holds them after upload.
fn uploaded() -> Vec<(String, WeightDtype, Vec<usize>)> {
    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_data/qwen4_exp_fp4local_headers.json"
        )))
        .unwrap();
    let mut out = Vec::new();
    for (name, v) in raw {
        let shape: Vec<usize> = v[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        match csf_part(&name) {
            // [rows/16, 16*(1+cols/2)] -> [rows, cols] E4M3.
            Some((base, CsfPart::Fixed)) => out.push((
                base.to_string(),
                WeightDtype::FP8E4M3,
                vec![shape[0] * 16, (shape[1] / 16 - 1) * 2],
            )),
            Some((_, CsfPart::Exceptions)) => {}
            // All other CSF components are assembled into the decoded scale
            // by the loader and must not appear as uploaded tensors.
            Some((_, _)) => {}
            None => out.push((
                name,
                WeightDtype::from_safetensors_str(v[0].as_str().unwrap()).unwrap(),
                shape,
            )),
        }
    }
    out
}

fn build(tensors: Vec<(String, WeightDtype, Vec<usize>)>) -> WeightStore {
    let mut map = HashMap::new();
    let mut deferred = Vec::new();
    for (name, dtype, shape) in tensors {
        if is_ngram_table(&name) {
            deferred.push((name, dtype, shape));
        } else {
            map.insert(
                name,
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape,
                    dtype,
                },
            );
        }
    }
    let mut s = WeightStore::from_map(map);
    for (name, dtype, shape) in deferred {
        s.defer(
            name,
            DeferredTensor {
                path: "x.safetensors".into(),
                offset: 0,
                shape,
                dtype,
            },
        );
    }
    s
}

/// The NVFP4 release's backbone: every MXFP8 projection BF16, its scale gone,
/// every norm BF16.
fn as_old_release(
    t: Vec<(String, WeightDtype, Vec<usize>)>,
) -> Vec<(String, WeightDtype, Vec<usize>)> {
    let fp8: std::collections::HashSet<String> = t
        .iter()
        .filter(|(n, d, _)| *d == WeightDtype::FP8E4M3 && n.ends_with(".weight"))
        .map(|(n, _, _)| n.trim_end_matches(".weight").to_string())
        .collect();
    t.into_iter()
        .filter(|(n, d, _)| {
            !(n.ends_with(".weight_scale")
                && *d == WeightDtype::UInt8
                && fp8.contains(n.trim_end_matches(".weight_scale")))
        })
        .map(|(n, d, s)| {
            let d = if (d == WeightDtype::FP8E4M3 && fp8.contains(n.trim_end_matches(".weight")))
                || d == WeightDtype::FP32 && n.contains("norm")
            {
                WeightDtype::BF16
            } else {
                d
            };
            (n, d, s)
        })
        .collect()
}

#[test]
fn audit_sees_the_whole_fp4_local_namespace() {
    let c = config();
    let r = audit_namespace(&build(uploaded()), &c);
    assert_eq!((r.gdn_layers, r.attn_layers), (36, 12));
    assert_eq!(r.expert_tensors, 48);
    assert_eq!(r.hc_tensors, 96);
    assert_eq!(r.indexer_tensors, 12);
    assert_eq!(r.ple_shards, 128, "the 128 deferred shards at layers.1");
    assert_eq!(r.vision_tensors, 27);
    assert!(r.has_embed && r.has_lm_head);
    assert_eq!(r.stray_layer_norms, 0);
    r.ensure_loadable().unwrap();
}

#[test]
fn detects_fp4_local() {
    let s = build(uploaded());
    let f = Qwen4ExpFormat::detect(&s, &config()).unwrap();
    assert!(f.mxfp8_backbone);
    let line = f.summary(&s);
    assert!(
        line.starts_with("FP4-Local; backbone MXFP8")
            && line.ends_with("ViT MLP W4A16 NVFP4 -> BF16"),
        "{line}"
    );
}

#[test]
fn the_old_release_is_not_fp4_local() {
    let mut c = config();
    for spec in c
        .quantization_config
        .as_mut()
        .unwrap()
        .quantized_layers
        .values_mut()
    {
        spec.weight_scale_encoding.clear();
    }
    let s = build(as_old_release(uploaded()));
    let f = Qwen4ExpFormat::detect(&s, &c).unwrap();
    assert!(!f.mxfp8_backbone);
    // The ViT MLP stays W4A16 in this synthetic store (only the backbone and
    // the norms were turned back into BF16).
    assert!(
        f.summary(&s)
            .starts_with("FP8/NVFP4 release; backbone BF16;")
    );
    audit_namespace(&s, &c).ensure_loadable().unwrap();
}

#[test]
fn a_half_mxfp8_backbone_is_refused() {
    let c = config();
    let l0 = c.layer_prefix(0);
    let t: Vec<_> = uploaded()
        .into_iter()
        .filter(|(n, _, _)| *n != format!("{l0}.linear_attn.in_proj_qkv.weight_scale"))
        .map(|(n, d, s)| {
            if n == format!("{l0}.linear_attn.in_proj_qkv.weight") {
                (n, WeightDtype::BF16, s)
            } else {
                (n, d, s)
            }
        })
        .collect();
    let e = format!("{:#}", Qwen4ExpFormat::detect(&build(t), &c).unwrap_err());
    assert!(e.contains("backbone probes are MXFP8"), "{e}");
}

/// Every MXFP8 tensor in the checkpoint passes the probe: 438 projections in
/// the five components the release policy knows, and nothing else.
#[test]
fn every_mxfp8_tensor_is_a_known_component_and_probes_clean() {
    let c = config();
    let store = build(uploaded());
    let stems: Vec<String> = store
        .names()
        .filter(|n| n.ends_with(".weight") && store.get(n).unwrap().dtype == WeightDtype::FP8E4M3)
        .map(|n| n.trim_end_matches(".weight").to_string())
        .collect();
    assert_eq!(stems.len(), 438);
    let mut known = std::collections::HashSet::new();
    for i in 0..c.num_hidden_layers {
        for comp in layer_components(c.layer_types[i]) {
            known.extend(comp.stems(&c.layer_prefix(i)));
        }
    }
    for b in 0..27 {
        known.extend(Mxfp8Component::Vision.stems(&format!("model.visual.blocks.{b}")));
    }
    for s in &stems {
        assert!(known.contains(s), "{s} is MXFP8 but in no component");
        assert!(mxfp8_geometry(&store, s).unwrap().is_some(), "{s}");
    }
}

#[test]
fn native_modes_audit_late_layers_and_local_experts() {
    let mut config = config();
    // The captured fixture contains expert 0; expert 1 is remote on rank 0.
    config.num_experts = 2;
    config.ep_world_size = 2;
    config.ep_rank = 0;
    for index in [0, config.num_hidden_layers - 1] {
        let lp = config.layer_prefix(index);
        let check = |tensors| {
            super::super::native_format::validate_layer(&build(tensors), &lp, &config, index)
        };
        check(two_experts()).unwrap();
        let stem = layer_components(config.layer_types[index])[0].stems(&lp)[0].clone();
        for name in [
            format!("{stem}.weight"),
            format!("{lp}.mlp.experts.0.gate_proj.weight"),
        ] {
            let mut tensors = two_experts();
            let tensor = tensors.iter_mut().find(|t| t.0 == name).unwrap();
            tensor.1 = WeightDtype::BF16;
            assert!(check(tensors).is_err(), "accepted fallback for {name}");
        }
        let mut tensors = two_experts();
        tensors.retain(|t| t.0 != format!("{lp}.mlp.experts.0.down_proj.weight_scale_2"));
        assert!(check(tensors).is_err());
    }
}

#[test]
fn native_modes_validate_every_layer_on_both_ep_ranks() {
    let mut config = config();
    // Serving caps the usable vocabulary to the tokenizer; tensors stay padded.
    assert_eq!(config.vocab_size, 248_320);
    config.vocab_size = 248_077;
    config.num_experts = 2;
    config.ep_world_size = 2;
    for rank in 0..2 {
        config.ep_rank = rank;
        let tensors = || {
            two_experts()
                .into_iter()
                .filter_map(|(name, dtype, shape)| {
                    if name.contains(".mlp.experts.256.") {
                        (rank == 1).then(|| {
                            (
                                name.replace(".mlp.experts.256.", ".mlp.experts.1."),
                                dtype,
                                shape,
                            )
                        })
                    } else if name.contains(".mlp.experts.0.") {
                        (rank == 0).then_some((name, dtype, shape))
                    } else {
                        Some((name, dtype, shape))
                    }
                })
                .collect()
        };
        super::super::native_format::validate_target(&build(tensors()), &config).unwrap();
        let missing = format!(
            "{}.mlp.experts.{rank}.up_proj.weight",
            config.layer_prefix(23)
        );
        let mut broken: Vec<_> = tensors();
        broken.retain(|t| t.0 != missing);
        assert!(super::super::native_format::validate_target(&build(broken), &config).is_err());
    }
}

fn two_experts() -> Vec<(String, WeightDtype, Vec<usize>)> {
    uploaded()
        .into_iter()
        .map(|(name, dtype, mut shape)| {
            if name.ends_with(".mlp.gate.weight") {
                shape[0] = 2;
            }
            (name, dtype, shape)
        })
        .collect()
}

#[test]
fn native_modes_reject_malformed_expert_geometry() {
    let mut c = config();
    c.num_experts = 2;
    c.ep_world_size = 2;
    let lp = c.layer_prefix(0);
    for suffix in [
        "mlp.gate.weight",
        "mlp.shared_expert_gate.weight",
        "mlp.experts.0.gate_proj.weight",
        "mlp.experts.0.down_proj.weight_scale",
        "mlp.experts.0.up_proj.weight_scale_2",
    ] {
        let name = format!("{lp}.{suffix}");
        for bad_dtype in [false, true] {
            let mut tensors = two_experts();
            let t = tensors.iter_mut().find(|t| t.0 == name).unwrap();
            if bad_dtype {
                t.1 = WeightDtype::Int32;
            } else {
                t.2 = vec![1, 1, 1];
            }
            assert!(
                super::super::native_format::validate_layer(&build(tensors), &lp, &c, 0).is_err(),
                "{name}"
            );
        }
    }
}

#[test]
fn native_modes_validate_mtp_before_conversion() {
    let mut c = config();
    c.num_experts = 2;
    // The production MTP loader receives the TP-local target config and then
    // restores full attention width through `mtp_drafter_config`.
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.num_attention_heads /= 2;
    c.num_key_value_heads /= 2;
    let drafter_config = super::super::helpers::mtp_drafter_config(&c);
    let tensors = || {
        let mut data = two_experts();
        let expert: Vec<_> = data
            .iter()
            .filter(|t| t.0.starts_with("mtp.layers.0.mlp.experts.0."))
            .map(|(n, d, s)| (n.replace(".experts.0.", ".experts.1."), *d, s.clone()))
            .collect();
        data.extend(expert);
        data
    };
    super::super::native_format::validate_mtp(&build(tensors()), &drafter_config).unwrap();
    for name in [
        "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.experts.1.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    ] {
        let mut data = tensors();
        data.iter_mut().find(|t| t.0 == name).unwrap().2 = vec![1];
        assert!(
            super::super::native_format::validate_mtp(&build(data), &drafter_config).is_err(),
            "{name}"
        );
    }
}

#[test]
fn native_modes_reject_truncated_or_reformatted_target_head() {
    let mut c = config();
    c.num_experts = 2;
    c.ep_world_size = 2;
    c.vocab_size = 248_077;
    for (dtype, shape) in [
        (WeightDtype::BF16, vec![248_076, c.hidden_size]),
        (WeightDtype::BF16, vec![248_320, c.hidden_size - 1]),
        (WeightDtype::BF16, vec![248_320]),
        (WeightDtype::FP32, vec![248_320, c.hidden_size]),
    ] {
        let mut tensors = two_experts();
        let head = tensors
            .iter_mut()
            .find(|t| t.0 == "lm_head.weight")
            .unwrap();
        head.1 = dtype;
        head.2 = shape;
        assert!(super::super::native_format::validate_target(&build(tensors), &c).is_err());
    }
}
