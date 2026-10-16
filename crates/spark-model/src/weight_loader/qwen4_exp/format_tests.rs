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
    let f = Qwen4ExpFormat::detect(&build(uploaded()), &config()).unwrap();
    assert!(f.is_fp4_local());
    assert!(f.mxfp8_backbone && f.csf_expert_scales);
    assert!(f.vision_mlp_nvfp4 && f.mtp_experts_nvfp4);
    assert_eq!(f.ple_table, Some(WeightDtype::UInt8));
    // 48 x 2 hc sites + the model mixer, plus norm_conv/key/query.
    assert_eq!(f.f32_norms, 97 + 3);
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
    assert!(!f.is_fp4_local());
    assert!(!f.mxfp8_backbone && !f.csf_expert_scales);
    assert_eq!(f.f32_norms, 0);
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
