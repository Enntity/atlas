// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's MXFP8 QSA attention projections: BF16 for prefill, the
//! checkpoint's own bytes for decode (`ATLAS_QWEN4EXP_MXFP8_DECODE`, the
//! switch the GDN copies use).
//!
//! The NVFP4 release ships q/k/v/o_proj BF16 and the shared arm requantizes
//! them to NVFP4 unless `--attn-proj-dtype bf16`. FP4-Local ships them MXFP8;
//! requantizing their dequant to NVFP4 is a lossy round trip the checkpoint
//! never made, so an MXFP8 layer always takes the arm's BF16 build (exact
//! dequant, prefill), and with the switch on (the default) decode reads the
//! checkpoint's bytes through `Qwen3AttentionLayer::install_dense_mxfp8`.
//! `=0` decodes the BF16 copy.
//!
//! Bytes per TP2 rank and layer: q 6144x2560, k/v 256x2560, o 2560x3072 =
//! 24.8M weights; BF16 49.5 MB, MXFP8 25.5 MB, the NVFP4 requantization the
//! NVFP4 release decodes ~14 MB. Exactness, not bandwidth, is the reason.
//!
//! The QSA indexer's `index_qk_proj` (also MXFP8) is NOT covered: the indexer
//! reads the store's BF16 tensor directly (`attach::attach_qsa`), so it needs
//! a BF16 dequant from the checkpoint loader, and decode keeps reading BF16.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::gdn_mxfp8::mxfp8_decode_requested;
use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::tp_shard::TpAttentionDims;
use crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4;
use crate::weight_map::{
    DenseWeight, Mxfp8Weight, Nvfp4Variant, RawMxfp8, mxfp8_cols_twin, mxfp8_rows_twin, raw_mxfp8,
    release_raw_mxfp8,
};

const PROJ: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// The shared full-attention arm, plus the MXFP8 handling above when this
/// layer's projections are MXFP8. Same arguments as
/// `build_full_attention_nvfp4`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_full_attention(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    kv_dtype: KvCacheDtype,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let p = format!("{lp}.self_attn");
    let raws = PROJ
        .iter()
        .map(|name| raw_mxfp8(store, &format!("{p}.{name}")))
        .collect::<Result<Vec<_>>>()?;
    let build = |config: &ModelConfig| {
        build_full_attention_nvfp4(
            layer_idx,
            store,
            lp,
            gpu,
            variant,
            config,
            h,
            absmax_k,
            quantize_k,
            stream,
            kv_dtype,
            attn_idx,
            input_norm,
            post_attn_norm,
            ffn,
        )
    };
    if raws.iter().all(Option::is_none) {
        return build(config);
    }
    let raws: Vec<RawMxfp8> = raws
        .into_iter()
        .zip(PROJ)
        .map(|(r, name)| r.with_context(|| format!("{p}: {name} is not MXFP8 beside the rest")))
        .collect::<Result<_>>()?;
    // Twins first: the BF16 build dequantizes the same tensors and may
    // release their raw bytes (`mxfp8_load`'s contract).
    let twins = if mxfp8_decode_requested()? {
        Some(twins(gpu, config, &raws, stream)?)
    } else {
        None
    };
    let bf16 = ModelConfig {
        attn_proj_bf16: true,
        ..config.clone()
    };
    let mut layer = build(&bf16)?;
    if let Some(twins) = twins {
        layer
            .as_any_mut()
            .and_then(|a| a.downcast_mut::<crate::layers::Qwen3AttentionLayer>())
            .with_context(|| format!("{p}: not a Qwen3AttentionLayer"))?
            .install_dense_mxfp8(twins)?;
    }
    for name in PROJ {
        release_raw_mxfp8(store, gpu, &format!("{p}.{name}"))?;
    }
    Ok(layer)
}

/// This rank's Q/K/V rows and O columns of the checkpoint tensors, with the
/// shapes `TpAttentionDims` (the BF16 build's slicer) expects.
fn twins(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    raws: &[RawMxfp8],
    stream: u64,
) -> Result<[(Mxfp8Weight, [usize; 2]); 4]> {
    let d = TpAttentionDims::from_config(config);
    let want = [
        [d.full_q_n, d.h],
        [d.full_kv_n, d.h],
        [d.full_kv_n, d.h],
        [d.h, d.full_o_in],
    ];
    for ((raw, want), name) in raws.iter().zip(want).zip(PROJ) {
        ensure!(
            [raw.n, raw.k] == want,
            "{name}: MXFP8 [{}, {}], the config's full shape is {want:?}",
            raw.n,
            raw.k,
        );
    }
    let (rank, size) = (d.tp_rank, d.tp_size);
    let rows = |raw: RawMxfp8| -> Result<(Mxfp8Weight, [usize; 2])> {
        let (w, n, k) = mxfp8_rows_twin(gpu, &[(raw, &[raw.n])], rank, size, stream)?;
        Ok((w, [n, k]))
    };
    let (o, on, ok) = mxfp8_cols_twin(gpu, raws[3], rank, size, stream)?;
    Ok([
        rows(raws[0])?,
        rows(raws[1])?,
        rows(raws[2])?,
        (o, [on, ok]),
    ])
}
