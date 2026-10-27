// SPDX-License-Identifier: AGPL-3.0-only

//! Metal-build stub of the cuda-only `cuda_backend` module.
//!
//! spark-model names `pdl_enabled` unconditionally
//! (`layers/ops/gemv_touch.rs`), so the metal build needs the symbol
//! even though PDL itself is a CUDA programmatic-dependent-launch
//! feature. Metal has no PDL: always off.

/// CUDA Programmatic Dependent Launch; never enabled on metal.
pub fn pdl_enabled() -> bool {
    false
}
