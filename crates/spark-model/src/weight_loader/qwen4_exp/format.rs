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
//! How each FP4-Local difference loads, and where:
//!
//! * CSF expert scales: decoded at upload, losslessly
//!   (`spark_runtime::weights::nvfp4_csf`); the MoE never sees CSF.
//! * MXFP8 GDN and QSA attention projections: the checkpoint's own bytes as
//!   decode twins (`gdn_mxfp8`, `attn_mxfp8`; `ATLAS_QWEN4EXP_MXFP8_DECODE=0`
//!   decodes BF16 instead) and their exact BF16 dequant for prefill, which
//!   `ATLAS_QWEN4EXP_PREFILL_W8A8` (`w8a8`) replaces with W8A8 GEMMs on the
//!   twins, freeing it.
//! * MXFP8 QSA indexer `index_qk_proj` and ViT attention: the exact BF16
//!   dequant (`dense_auto`).
//! * MXFP8 shared expert: its exact BF16 dequant, REQUANTIZED to NVFP4 by
//!   the shared MoE loader, as the BF16 releases' shared expert is. Not
//!   exact, and kept as policy: the fused MoE kernels read NVFP4.
//! * W4A16 NVFP4 ViT MLP and merger: expanded to BF16 (`vision_dense_auto`,
//!   one rounding; see there).
//! * NVFP4 MTP experts: loaded natively (`mtp.rs`).
//! * F32 norms: rounded to BF16, RNE (`helpers::bf16_norm`).
//!
//! What lives here is the detection the loader branches on
//! (`mxfp8_backbone`), one summary line, and the release of each MXFP8
//! tensor's source bytes once every reader has read them.

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
}

impl Qwen4ExpFormat {
    /// Detect the format and log one summary line.
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
        let format = Self {
            mxfp8_backbone: mxfp8 != 0,
        };
        tracing::info!("qwen4_exp checkpoint format: {}", format.summary(store));
        Ok(format)
    }

    /// The release and what its backbone and ViT MLP load as. The CSF
    /// decode, the MTP expert layout and the PLE table each log their own
    /// line where they load.
    fn summary(&self, store: &WeightStore) -> String {
        let vit_nvfp4 = store
            .get("model.visual.blocks.0.mlp.linear_fc1.weight")
            .is_ok_and(|t| t.dtype == WeightDtype::UInt8);
        format!(
            "{}; backbone {}; ViT MLP {}",
            if self.mxfp8_backbone {
                "FP4-Local"
            } else {
                "FP8/NVFP4 release"
            },
            if self.mxfp8_backbone {
                "MXFP8 (GDN/QSA decode twins, prefill W8A8 on them or on their exact BF16; \
                 indexer and ViT attention exact BF16; shared expert BF16 requantized to NVFP4)"
            } else {
                "BF16"
            },
            if vit_nvfp4 {
                "W4A16 NVFP4 -> BF16"
            } else {
                "BF16"
            },
        )
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
