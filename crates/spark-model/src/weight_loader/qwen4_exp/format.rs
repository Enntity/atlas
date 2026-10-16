// SPDX-License-Identifier: AGPL-3.0-only

//! Which on-disk release of Qwen3.8-Flash-Next this is, detected once.
//!
//! Three formats share one architecture and one namespace:
//!
//! | release            | backbone (GDN/attn/indexer/shared/ViT attn) | routed-expert scales | ViT MLP + merger | MTP experts      |
//! |--------------------|---------------------------------------------|----------------------|------------------|------------------|
//! | FP8 / NVFP4 (old)  | BF16 (or block FP8)                         | plain E4M3           | BF16             | BF16 or FP8 128² |
//! | **FP4-Local**      | **MXFP8** (E4M3 + UE8M0 per 32)             | **CSF** pairs        | **W4A16 NVFP4**  | **NVFP4**        |
//!
//! None of these needs its own arm. The CSF scales are decoded at upload
//! (`spark_runtime::weights::nvfp4_csf`), every MXFP8 tensor goes through
//! the exact MXFP8 -> BF16 expansion in `dense_auto`, the ViT NVFP4 linears
//! are dequantized by `vision_dense_auto`, and the MTP experts load natively
//! (`mtp.rs`). What lives here is the REPORT (one line saying which format
//! loaded) and the release of each MXFP8 tensor's source bytes once every
//! reader has read them.

use anyhow::{Context, Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{mxfp8_geometry, release_raw_mxfp8};

/// A part of the backbone that ships MXFP8 on FP4-Local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp8Component {
    /// `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`.
    Gdn,
    /// `self_attn.{q,k,v,o}_proj`.
    Attn,
    /// `self_attn.indexer.index_qk_proj`.
    Indexer,
    /// `mlp.shared_expert.{gate,up,down}_proj`.
    SharedExpert,
    /// `model.visual.blocks.B.attn.{qkv,proj}`.
    Vision,
}

impl Mxfp8Component {
    /// The tensor stems of this component in decoder layer `lp` (or, for
    /// `Vision`, ViT block `lp`).
    pub fn stems(self, lp: &str) -> Vec<String> {
        let names: &[&str] = match self {
            Self::Gdn => &[
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_a",
                "linear_attn.in_proj_b",
                "linear_attn.out_proj",
            ],
            Self::Attn => &[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
            Self::Indexer => &["self_attn.indexer.index_qk_proj"],
            Self::SharedExpert => &[
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
            ],
            Self::Vision => &["attn.qkv", "attn.proj"],
        };
        names.iter().map(|n| format!("{lp}.{n}")).collect()
    }
}

/// Release the MXFP8 sources of `components` under `lp`. Returns the bytes
/// released.
///
/// The ONE release point for a decoder layer's MXFP8 bytes, called after the
/// layer is fully built: by then every consumer has read them (the decode
/// twins, the exact BF16 copies `dense_auto` makes, and the QSA indexer's
/// `index_qk_proj` in `attach_qsa`), and nothing in the loader reads them
/// again. A stem that is not MXFP8 is skipped.
pub(super) fn release_mxfp8_sources(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    lp: &str,
    components: &[Mxfp8Component],
) -> Result<usize> {
    let mut bytes = 0;
    for c in components {
        for stem in c.stems(lp) {
            if mxfp8_geometry(store, &stem)?.is_none() {
                continue;
            }
            for part in ["weight", "weight_scale"] {
                let name = format!("{stem}.{part}");
                if !store.was_reclaimed(&name) {
                    bytes += store.get(&name)?.byte_size();
                }
            }
            release_raw_mxfp8(store, gpu, &stem)
                .with_context(|| format!("releasing MXFP8 source {stem}"))?;
        }
    }
    Ok(bytes)
}

/// The MXFP8 components a decoder layer of this type carries.
pub(super) fn layer_components(t: LayerType) -> &'static [Mxfp8Component] {
    match t {
        LayerType::LinearAttention => &[Mxfp8Component::Gdn, Mxfp8Component::SharedExpert],
        LayerType::FullAttention => &[
            Mxfp8Component::Attn,
            Mxfp8Component::Indexer,
            Mxfp8Component::SharedExpert,
        ],
        _ => &[],
    }
}

/// What the store says this checkpoint is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4ExpFormat {
    /// MXFP8 backbone projections found (layer 0's GDN, the first attention
    /// layer's `q_proj`, layer 0's shared expert all probed).
    pub mxfp8_backbone: bool,
    /// Routed-expert scales declared CSF in `quantized_layers`.
    pub csf_expert_scales: bool,
    /// The ViT MLP ships packed NVFP4 (W4A16) rather than BF16.
    pub vision_mlp_nvfp4: bool,
    /// The MTP block's routed experts ship per-expert NVFP4.
    pub mtp_experts_nvfp4: bool,
    /// Element type of the PLE n-gram table shards on disk.
    pub ple_table: Option<WeightDtype>,
    /// F32 norm tensors (`hc_norm`, PLE `norm_*`) that load rounded to BF16.
    pub f32_norms: usize,
}

impl Qwen4ExpFormat {
    pub fn detect(store: &WeightStore, config: &ModelConfig) -> Result<Self> {
        let l0 = config.layer_prefix(0);
        let first_attn = config
            .layer_types
            .iter()
            .position(|t| *t == LayerType::FullAttention)
            .map(|i| config.layer_prefix(i));
        let mut probes = vec![
            format!("{l0}.linear_attn.in_proj_qkv"),
            format!("{l0}.mlp.shared_expert.gate_proj"),
        ];
        probes.extend(first_attn.map(|p| format!("{p}.self_attn.q_proj")));
        let mut mxfp8 = 0;
        for p in &probes {
            mxfp8 += mxfp8_geometry(store, p)?.is_some() as usize;
        }
        // All or none: a checkpoint with an MXFP8 GDN and a BF16 attention
        // is not a release anyone has published, and guessing which half is
        // right would be guessing.
        if mxfp8 != 0 && mxfp8 != probes.len() {
            bail!(
                "qwen4_exp: {mxfp8} of {} backbone probes are MXFP8 ({probes:?})",
                probes.len()
            );
        }
        let csf_expert_scales = config.quantization_config.as_ref().is_some_and(|q| {
            q.quantized_layers
                .iter()
                .any(|(p, s)| p.contains(".mlp.experts") && s.weight_scale_encoding == "csf")
        });
        let dtype = |n: &str| store.get(n).ok().map(|t| t.dtype);
        // `ple_layer_ids` is one-indexed: `[2]` lives at `layers.1.ple`.
        let ple_table = config.ple_layer_ids.first().and_then(|&l| {
            let lp = config.layer_prefix(l.saturating_sub(1));
            let base = format!("{lp}.ple.ple_embedding.ngram_embedding");
            store
                .deferred(&format!("{base}.shard_0.weight"))
                .map(|d| d.dtype)
                .or_else(|| dtype(&format!("{base}.shard_0.weight")))
        });
        let f32_norms = store
            .names()
            .filter(|n| {
                (n.ends_with(".hc_norm.weight") || n.contains(".ple.norm_"))
                    && dtype(n) == Some(WeightDtype::FP32)
            })
            .count();
        Ok(Self {
            mxfp8_backbone: mxfp8 != 0,
            csf_expert_scales,
            vision_mlp_nvfp4: dtype("model.visual.blocks.0.mlp.linear_fc1.weight")
                == Some(WeightDtype::UInt8),
            mtp_experts_nvfp4: dtype("mtp.layers.0.mlp.experts.0.gate_proj.weight")
                == Some(WeightDtype::UInt8),
            ple_table,
            f32_norms,
        })
    }

    /// nvidia's FP4-Local release: MXFP8 backbone plus CSF expert scales.
    pub fn is_fp4_local(&self) -> bool {
        self.mxfp8_backbone && self.csf_expert_scales
    }

    pub fn log(&self) {
        tracing::info!(
            "qwen4_exp checkpoint format: {} — backbone {}, routed-expert scales {}, \
             ViT MLP {}, MTP experts {}, PLE table {:?}, {} F32 norm(s) rounded to BF16 (RNE)",
            if self.is_fp4_local() {
                "FP4-Local"
            } else {
                "FP8/NVFP4 release"
            },
            if self.mxfp8_backbone {
                "MXFP8 -> exact BF16 copies"
            } else {
                "BF16"
            },
            if self.csf_expert_scales {
                "CSF (decoded at upload)"
            } else {
                "plain E4M3"
            },
            if self.vision_mlp_nvfp4 {
                "W4A16 NVFP4 -> BF16"
            } else {
                "BF16"
            },
            if self.mtp_experts_nvfp4 {
                "native NVFP4"
            } else {
                "BF16/FP8 -> NVFP4"
            },
            self.ple_table,
            self.f32_norms,
        );
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
