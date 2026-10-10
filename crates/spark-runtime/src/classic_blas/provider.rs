// SPDX-License-Identifier: AGPL-3.0-only
use crate::gpu::Bf16GemmNumerics;
use anyhow::{ensure, Result};
use std::path::Path;
/// Validate actual dladdr providers before exposing a classic numerical capability.
pub(super) fn validate_provider_pair(
    numerics: Bf16GemmNumerics,
    classic_version: i32,
    lt_version: i32,
    classic: &Path,
    lt: &Path,
) -> Result<()> {
    let required = numerics
        .provider_version()
        .ok_or_else(|| anyhow::anyhow!("classic provider requires explicit classic profile"))?;
    ensure!(classic_version == required && lt_version == required,
        "classic/Lt profile requires coherent version{required}; got {classic_version}/{lt_version}");
    ensure!(
        classic.is_absolute()
            && lt.is_absolute()
            && classic != lt
            && classic.parent() == lt.parent(),
        "classic/Lt providers must be one explicit directory"
    );
    ensure!(
        classic
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("libcublas.so."))
            && lt
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("libcublasLt.so.")),
        "classic/Lt providers have unexpected SONAME paths"
    );
    Ok(())
}
