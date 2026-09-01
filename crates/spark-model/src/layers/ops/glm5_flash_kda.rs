// SPDX-License-Identifier: AGPL-3.0-only

//! Dynamic bridge to the pinned FlashKDA chunked-prefill kernel.

use anyhow::{Result, anyhow, bail, ensure};
use spark_runtime::gpu::DevicePtr;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
const DEFAULT_LIBRARY: &str = "libatlas_glm53_flash_kda.so";
const WORKSPACE_SYMBOL: &[u8] = b"atlas_flash_kda_workspace_size\0";
const PREFILL_SYMBOL: &[u8] = b"atlas_flash_kda_prefill_fp32_state\0";

type WorkspaceSizeFn = unsafe extern "C" fn(c_int, c_int, c_int) -> i64;

// Keep this argument-for-argument identical to
// `3rdparty_patches/flash_kda/atlas_flash_kda_bridge.cu`. Const qualifiers do
// not affect the C ABI, so every device address is represented as `void*`.
type PrefillFn = unsafe extern "C" fn(
    *mut c_void, // query
    *mut c_void, // key
    *mut c_void, // value
    *mut c_void, // forget projection
    *mut c_void, // beta [heads,total_tokens]
    *mut c_void, // recurrent state pool
    *mut c_void, // output
    *mut c_void, // workspace
    *mut c_void, // A_log
    *mut c_void, // dt_bias
    *mut c_void, // cu_seqlens i64
    *mut c_void, // physical state-slot ids i32
    c_int,
    c_int,
    c_int,
    c_int,
    c_float,
    c_float,
    *mut c_void, // CUDA stream
) -> c_int;

struct Lib {
    workspace_size: WorkspaceSizeFn,
    prefill: PrefillFn,
}

// SAFETY: both function pointers are immutable and their dlopen handle is
// deliberately kept mapped for the lifetime of the process.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Option<Lib>> = OnceLock::new();

fn lib() -> Option<&'static Lib> {
    LIB.get_or_init(|| unsafe {
        let path = std::env::var("ATLAS_GLM53_FLASH_KDA_LIB")
            .unwrap_or_else(|_| DEFAULT_LIBRARY.to_string());
        let cpath = std::ffi::CString::new(path.clone()).ok()?;
        let handle = dlopen(cpath.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            tracing::warn!("GLM FlashKDA bridge: dlopen('{path}') failed");
            return None;
        }
        let workspace_size = dlsym(handle, WORKSPACE_SYMBOL.as_ptr().cast());
        let prefill = dlsym(handle, PREFILL_SYMBOL.as_ptr().cast());
        if workspace_size.is_null() || prefill.is_null() {
            tracing::warn!("GLM FlashKDA bridge: required symbols are missing from '{path}'");
            return None;
        }
        tracing::info!("GLM FlashKDA bridge loaded from '{path}'");
        Some(Lib {
            workspace_size: std::mem::transmute::<*mut c_void, WorkspaceSizeFn>(workspace_size),
            prefill: std::mem::transmute::<*mut c_void, PrefillFn>(prefill),
        })
    })
    .as_ref()
}

pub fn available() -> bool {
    lib().is_some()
}

pub fn glm53_flash_kda_workspace_size(
    total_tokens: u32,
    heads: u32,
    sequences: u32,
) -> Result<usize> {
    ensure!(total_tokens > 0, "GLM FlashKDA requires at least one token");
    ensure!(heads > 0, "GLM FlashKDA requires at least one head");
    ensure!(sequences > 0, "GLM FlashKDA requires at least one sequence");
    let bridge = lib().ok_or_else(|| anyhow!("GLM FlashKDA bridge unavailable"))?;
    let bytes = unsafe {
        (bridge.workspace_size)(total_tokens as c_int, heads as c_int, sequences as c_int)
    };
    ensure!(
        bytes > 0,
        "GLM FlashKDA returned invalid workspace size {bytes}"
    );
    usize::try_from(bytes).map_err(Into::into)
}

/// Packed varlen chunk input for GLM's KDA recurrence.
///
/// Q/K/V/forget/output are `[total_tokens,heads,128]`; `beta_ht` is the TMA
/// layout `[heads,total_tokens]`; recurrent state is one FP32
/// `[state_capacity,heads,value=128,key=128]` pool. `state_slot_ids` maps each
/// logical sequence in `cu_seqlens` to its physical persistent slot.
pub struct Glm53FlashKdaPrefillArgs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub forget: DevicePtr,
    pub beta_ht: DevicePtr,
    pub recurrent_state: DevicePtr,
    pub output: DevicePtr,
    pub workspace: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub state_slot_ids: DevicePtr,
    pub total_tokens: u32,
    pub heads: u32,
    pub sequences: u32,
    pub state_capacity: u32,
    pub query_scale: f32,
    pub lower_bound: f32,
}

impl Glm53FlashKdaPrefillArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.total_tokens > 0, "GLM FlashKDA requires tokens");
        ensure!(self.heads > 0, "GLM FlashKDA requires heads");
        ensure!(self.sequences > 0, "GLM FlashKDA requires sequences");
        ensure!(
            self.state_capacity >= self.sequences,
            "GLM FlashKDA state capacity is smaller than the active batch"
        );
        ensure!(
            self.query_scale.is_finite() && self.query_scale > 0.0,
            "GLM FlashKDA query scale must be finite and positive"
        );
        ensure!(
            self.lower_bound.is_finite() && self.lower_bound <= 0.0,
            "GLM FlashKDA lower bound must be finite and non-positive"
        );
        for (name, pointer) in [
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("forget", self.forget),
            ("beta_ht", self.beta_ht),
            ("recurrent_state", self.recurrent_state),
            ("output", self.output),
            ("workspace", self.workspace),
            ("a_log", self.a_log),
            ("dt_bias", self.dt_bias),
            ("cu_seqlens", self.cu_seqlens),
            ("state_slot_ids", self.state_slot_ids),
        ] {
            ensure!(!pointer.is_null(), "GLM FlashKDA {name} pointer is null");
        }
        Ok(())
    }
}

pub fn glm53_flash_kda_prefill(args: &Glm53FlashKdaPrefillArgs, stream: u64) -> Result<()> {
    args.validate()?;
    let bridge = lib().ok_or_else(|| anyhow!("GLM FlashKDA bridge unavailable"))?;
    let pointer = |value: DevicePtr| value.0 as *mut c_void;
    let status = unsafe {
        (bridge.prefill)(
            pointer(args.query),
            pointer(args.key),
            pointer(args.value),
            pointer(args.forget),
            pointer(args.beta_ht),
            pointer(args.recurrent_state),
            pointer(args.output),
            pointer(args.workspace),
            pointer(args.a_log),
            pointer(args.dt_bias),
            pointer(args.cu_seqlens),
            pointer(args.state_slot_ids),
            args.total_tokens as c_int,
            args.heads as c_int,
            args.sequences as c_int,
            args.state_capacity as c_int,
            args.query_scale as c_float,
            args.lower_bound as c_float,
            stream as *mut c_void,
        )
    };
    if status != 0 {
        bail!("GLM FlashKDA bridge returned CUDA status {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_args() -> Glm53FlashKdaPrefillArgs {
        let mut pointer = 1;
        let mut next = || {
            let result = DevicePtr(pointer);
            pointer += 1;
            result
        };
        Glm53FlashKdaPrefillArgs {
            query: next(),
            key: next(),
            value: next(),
            forget: next(),
            beta_ht: next(),
            recurrent_state: next(),
            output: next(),
            workspace: next(),
            a_log: next(),
            dt_bias: next(),
            cu_seqlens: next(),
            state_slot_ids: next(),
            total_tokens: 50,
            heads: 4,
            sequences: 2,
            state_capacity: 4,
            query_scale: 1.0 / 128.0_f32.sqrt(),
            lower_bound: -5.0,
        }
    }

    #[test]
    fn production_fragmented_state_contract_validates() {
        valid_args().validate().unwrap();
        assert_eq!(WORKSPACE_SYMBOL, b"atlas_flash_kda_workspace_size\0");
        assert_eq!(PREFILL_SYMBOL, b"atlas_flash_kda_prefill_fp32_state\0");
    }

    #[test]
    fn invalid_capacity_scale_bound_and_pointer_fail_closed() {
        let mut args = valid_args();
        args.state_capacity = 1;
        assert!(
            args.validate()
                .unwrap_err()
                .to_string()
                .contains("capacity")
        );
        args = valid_args();
        args.query_scale = f32::NAN;
        assert!(args.validate().unwrap_err().to_string().contains("scale"));
        args = valid_args();
        args.lower_bound = 1.0;
        assert!(args.validate().unwrap_err().to_string().contains("bound"));
        args = valid_args();
        args.state_slot_ids = DevicePtr::NULL;
        assert!(args.validate().unwrap_err().to_string().contains("slot"));
    }
}
