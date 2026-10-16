// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's MXFP8 GDN projections: BF16 for prefill, the checkpoint's own
//! bytes for decode (`ATLAS_QWEN4EXP_MXFP8_DECODE`).
//!
//! `nvidia/Qwen3.8-Flash-Next-FP4-Local` ships `in_proj_qkv`, `in_proj_z`
//! and `out_proj` on all 36 GDN layers as MXFP8 (E4M3 + UE8M0 per 32 K
//! values), where the NVFP4 release shipped BF16. The BF16 build
//! (`dense_bf16_layer`) dequantizes them exactly and keeps that copy for
//! prefill and as the parity reference; this adds the checkpoint's bytes,
//! sliced to the rank, as the decode copy (`Qwen3SsmLayer::
//! install_mxfp8_decode`). No requantization anywhere, so decode stays the
//! checkpoint's math: the MXFP8 GEMV and the BF16 one multiply identical
//! operands and differ only in FP32 summation order
//! (`ops/mxfp8_rows_test_gpu.rs`).
//!
//! DEFAULT ON when the checkpoint is MXFP8: decode reads 1.03 bytes a weight
//! instead of 2, ~1.0 GB less per decode step on a TP2 rank. `=0` decodes the
//! BF16 copy instead, for A/B; it is also the only way to combine this
//! checkpoint with `ATLAS_QWEN4EXP_FP8_GDN` / `ATLAS_QWEN4EXP_BF16_GDN=0`,
//! which requantize the BF16 copy (lossy) and would otherwise be a second
//! decode format beside this one.
//!
//! In_proj_a / in_proj_b (48 x 2560 each, also MXFP8) stay BF16-only: the
//! fused BA + gates GEMV reads them, and at 0.5 MB a layer they are 0.2% of
//! the GDN bytes.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::tp_shard::TpGdnDims;
use crate::weight_loader::qwen35::load_layers::linear_attn_arms::dense_bf16_layer;
use crate::weight_map::{DenseWeight, Nvfp4Variant, mxfp8_cols_twin, mxfp8_rows_twin, raw_mxfp8};

/// `ATLAS_QWEN4EXP_MXFP8_DECODE` (read once): `0` decodes an MXFP8
/// checkpoint's BF16 dequant, anything else (unset included) its own bytes.
///
/// Lenient like the other default-on switches (`!= "0"`): the startup-parity
/// table reads it for EVERY model, and a refusal there would fail any TP
/// boot, GLM included, over a variable only this checkpoint uses.
pub(crate) fn mxfp8_decode_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse(std::env::var("ATLAS_QWEN4EXP_MXFP8_DECODE").ok().as_deref()))
}

fn parse(value: Option<&str>) -> bool {
    value != Some("0")
}

/// The GDN projection names whose MXFP8 bytes decode reads.
const QKV: &str = "in_proj_qkv";
const Z: &str = "in_proj_z";
const OUT: &str = "out_proj";

/// The BF16 GDN build plus the MXFP8 decode copies; see the module docs.
#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let la = format!("{lp}.linear_attn");
    let raw = |name: &str| {
        let prefix = format!("{la}.{name}");
        raw_mxfp8(store, &prefix)?
            .with_context(|| format!("{prefix}: not MXFP8 in a checkpoint whose GDN is"))
    };
    let (qkv, z, out) = (raw(QKV)?, raw(Z)?, raw(OUT)?);
    // The twins and the BF16 build below read the same raw bytes; the
    // loader's per-layer sweep (`format::release_mxfp8_sources`) frees them.
    let dims = TpGdnDims::from_config(config);
    let (rank, size, stream) = (dims.tp_rank, dims.tp_size, gpu.default_stream());
    let value = [dims.full_value_dim()];
    let (qkvz_w, qn, qk) = mxfp8_rows_twin(
        gpu,
        &[(qkv, &dims.qkv_segments()), (z, &value)],
        rank,
        size,
        stream,
    )?;
    let (out_w, on, ok) = mxfp8_cols_twin(gpu, out, rank, size, stream)?;
    let mut layer = dense_bf16_layer(
        layer_idx,
        store,
        lp,
        gpu,
        variant,
        config,
        config.hidden_size,
        input_norm,
        post_attn_norm,
        ffn,
    )?;
    layer.install_mxfp8_decode(gpu, config, (qkvz_w, [qn, qk]), (out_w, [on, ok]))?;
    Ok(Box::new(layer))
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn only_zero_is_off_and_garbage_is_the_default() {
        assert!(!parse(Some("0")));
        for v in [
            None,
            Some("1"),
            Some(""),
            Some("true"),
            Some("off"),
            Some(" 0"),
        ] {
            assert!(parse(v), "{v:?}");
        }
    }
}
