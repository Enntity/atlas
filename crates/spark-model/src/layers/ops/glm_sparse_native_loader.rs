// SPDX-License-Identifier: AGPL-3.0-only
//! Startup-only loading; retain the native mapping for copied ABI pointers.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::path::Path;
use std::sync::OnceLock;

use super::plan::NativeArgs;

pub(super) const FLAG: &str = "ATLAS_GLM_SPARSE_NATIVE";
const LIBRARY: &str = "ATLAS_GLM_SPARSE_NATIVE_LIBRARY";
pub(super) type Run = unsafe extern "C" fn(*const NativeArgs) -> i32;
type Init = unsafe extern "C" fn() -> i32;

struct Native {
    _library: Library,
    path: String,
    init: Init,
    run: Run,
}

static LOADED: OnceLock<Result<Native, String>> = OnceLock::new();

pub(super) fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => bail!("{FLAG} must be 0 or 1"),
    }
}

fn env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn enabled() -> Result<bool> {
    parse(env(FLAG)?.as_deref())
}

pub(super) fn validate_path(path: Option<&str>) -> Result<&str> {
    let path = path.context("ATLAS_GLM_SPARSE_NATIVE_LIBRARY is required when enabled")?;
    ensure!(
        Path::new(path).is_absolute(),
        "native sparse library path must be absolute"
    );
    Ok(path)
}

fn load(path: &str) -> Result<Native> {
    // SAFETY: the operator explicitly selects the trusted pinned library. Retain
    // its mapping and verify its ABI before any stateful native function call.
    let library = unsafe { Library::new(path) }.context("load native GLM sparse library")?;
    let version = unsafe {
        library.get::<unsafe extern "C" fn() -> i32>(b"atlas_glm_sparse_native_version\0")?
    };
    ensure!(
        unsafe { version() } == 1,
        "native GLM sparse ABI mismatch (requires1)"
    );
    let init = unsafe { *library.get::<Init>(b"atlas_glm_sparse_native_init\0")? };
    let run = unsafe { *library.get::<Run>(b"atlas_glm_sparse_native_run\0")? };
    Ok(Native {
        _library: library,
        path: path.to_owned(),
        init,
        run,
    })
}

pub(super) fn initialize() -> Result<()> {
    let path = env(LIBRARY)?;
    let path = validate_path(path.as_deref())?;
    let native = LOADED
        .get_or_init(|| load(path).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        native.path == path,
        "native sparse library cannot change within a serving process"
    );
    // Cache only the mapping/symbols, never CUDA initialization success. Every
    // factory build initializes on its current serving context before KV sizing.
    // ABI1 init loads/configures kernels without device buffers or attention.
    let status = unsafe { (native.init)() };
    ensure!(
        status == 0,
        "native GLM sparse initialization failed ({status})"
    );
    tracing::info!(
        library = path,
        abi = 1,
        "GLM native sparse prefill initialized before KV sizing; rows2048..4100, continuation only"
    );
    Ok(())
}

pub(super) fn run() -> Result<Run> {
    let native = LOADED
        .get()
        .context("native sparse was not initialized before serving")?
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(native.run)
}
