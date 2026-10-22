// SPDX-License-Identifier: AGPL-3.0-only

//! The loader's small private helpers.
//!
//! Split out of `qwen4_exp.rs` for the file-size cap. The seam is a real one:
//! none of these touches the loader's state or its trait impl — they are the
//! free functions the impl and its submodules call.

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::DenseWeight;

/// A norm weight as the F32 the mHC and PLE kernels read.
///
/// The FP8 and NVFP4 releases ship these BF16; they are widened exactly
/// (`bits << 16`), so those releases compute bit-identically to before. The
/// FP4-Local release ships the 97 `hc_norm`s and the three PLE norms as F32
/// ("original bytes preserved"); they are passed through untouched. Every
/// consumer kernel already forms `1 + w` in FP32, so only the argument type
/// changed (`const float*`), not the arithmetic.
///
/// Anything else is refused: a norm read through the wrong element type
/// would run the model on garbage.
pub(super) fn f32_norm(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
    expected: usize,
) -> Result<DenseWeight> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.shape == [expected],
        "{name}: norm shape {:?}, expected [{expected}]",
        tensor.shape
    );
    match tensor.dtype {
        WeightDtype::BF16 | WeightDtype::FP32 => {
            crate::weight_map::dense_keep_f32(store, name, gpu)
        }
        other => bail!("{name}: norm weight is {other:?}, expected BF16 or F32"),
    }
}

/// A ones-filled `[n]` BF16 norm scale.
///
/// BF16 1.0 is `0x3F80`, so the buffer cannot be produced with `memset`.
pub(super) fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// `model.language_model` for the multimodal layout, `model` otherwise.
pub(super) fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// The model-level hyper-connection mixer that collapses the residual streams.
pub(super) fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}

/// A drafter-only precision switch (`ATLAS_QWEN4EXP_MTP_BF16_*`) is `1`.
fn mtp_env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// The config the MTP drafter is built with: the TP=1 view (`mtp.rs`), and
/// with `ATLAS_QWEN4EXP_MTP_BF16_ATTN=1` its Q/K/V/O kept at the checkpoint's
/// BF16 (the `--attn-proj-dtype bf16` arms); the target keeps its own. The
/// shared expert cannot follow: `ExpertWeight` is NVFP4.
pub(super) fn mtp_drafter_config(config: &ModelConfig) -> ModelConfig {
    let mut config = config.tp1_view();
    config.attn_proj_bf16 |= mtp_env_on("ATLAS_QWEN4EXP_MTP_BF16_ATTN");
    config
}

#[cfg(test)]
#[path = "norm_tests.rs"]
mod tests;
