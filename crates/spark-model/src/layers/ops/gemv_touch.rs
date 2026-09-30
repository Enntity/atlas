// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DECODE_GEMV_BATCH=1`: weight touch before the PDL wait for the
//! verify-step GEMVs.
//!
//! A verify step launches every kernel with programmatic dependent launch, so
//! a projection's CTAs are resident while the small kernels in front of it
//! (HC pre, norm, conv, recurrent) still run, waiting with the memory bus idle
//! (2026-09-30 nsys: 49 us before KDA q, 28 before o, 31 before the shared
//! gate, 50 before MLA q_a). The `*_touch` twins of the GEMV kernels spend that
//! wait pulling their weight rows into L2 (`atlas_pdl_enter_touch`,
//! kernels/gb10/common/atlas_pdl.cuh); the body then reads L2 at about twice
//! the DRAM rate. A twin runs the body of the kernel it replaces, so every
//! output bit is unchanged (scripts/dev/glm_decode_gemv_touch_bench.cu).
//!
//! The twins of the scalar `w4a16_gemv_batch2/3` and `_batch5_qkv` tiers are
//! also what puts those kernels on the PDL list: their untouched originals
//! are launched without it and start only after their predecessor completes.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// `(module, function)` of every touch twin.
const TWINS: [(&str, &str); 4] = [
    ("w4a16_gemv", "w4a16_gemv_tc8_touch"),
    ("w4a16_gemv", "w4a16_gemv_batch3_touch"),
    ("w4a16_gemv", "w4a16_gemv_batch5_qkv_touch"),
    ("mxfp8_gemv", "mxfp8_gemv_tc8_touch"),
];

/// Read once: the flag, and the touch geometry `(bytes, ctas)`.
///
/// `bytes` bounds what one kernel pulls in (half of GB10's 24 MiB L2, so a
/// weight wider than that does not evict its own first rows); `ctas` is how
/// many CTAs issue the loads (they must all be resident during the wait, and
/// more of them only delays a kernel that has nothing to wait for).
/// `ATLAS_GLM_DECODE_GEMV_TOUCH_MB` / `_TOUCH_CTAS` override them for tuning.
fn settings() -> Option<(u64, u32)> {
    static SETTINGS: std::sync::OnceLock<Option<(u64, u32)>> = std::sync::OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let var = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| (1..=1024).contains(v))
                .unwrap_or(default)
        };
        (std::env::var("ATLAS_GLM_DECODE_GEMV_BATCH").as_deref() == Ok("1")).then(|| {
            (
                var("ATLAS_GLM_DECODE_GEMV_TOUCH_MB", 12) << 20,
                var("ATLAS_GLM_DECODE_GEMV_TOUCH_CTAS", 32) as u32,
            )
        })
    })
}

/// Look every twin up once, where the kernels are resolved (layer
/// constructors), so the boot audit sees them before it seals. A no-op with
/// the flag off: no lookup, and every launch stays on the original kernels.
pub fn gemv_touch_resolve(gpu: &dyn GpuBackend) {
    if settings().is_some() {
        for (module, func) in TWINS {
            let _ = gpu.op_cache().kernel(gpu, module, func);
        }
    }
}

/// A resolved touch twin and the geometry it is launched with.
#[derive(Clone, Copy, Debug)]
pub struct GemvTouch {
    pub kernel: KernelHandle,
    /// Most bytes (values plus scales) one launch pulls into L2.
    pub bytes: u64,
    /// Most CTAs that issue the loads.
    pub ctas: u32,
}

/// The touch twin `func` when the flag is on and this target carries it.
pub fn gemv_touch(gpu: &dyn GpuBackend, func: &'static str) -> Option<GemvTouch> {
    let (bytes, ctas) = settings()?;
    let (module, func) = TWINS.into_iter().find(|&(_, f)| f == func)?;
    let kernel = gpu.op_cache().kernel(gpu, module, func).ok()?;
    (kernel.0 != 0).then_some(GemvTouch {
        kernel,
        bytes,
        ctas,
    })
}

impl GemvTouch {
    /// The kernels' trailing `(touch_rows, touch_ctas)` for a weight of `n`
    /// rows of `row_bytes` (values plus scales) on a grid `grid_x` CTAs wide.
    fn geometry(&self, n: u32, row_bytes: u32, grid_x: u32) -> (u32, u32) {
        let rows = (self.bytes / u64::from(row_bytes.max(1))).min(u64::from(n)) as u32;
        (rows, self.ctas.min(grid_x))
    }

    /// `w4a16_gemv_tc8` / `w4a16_gemv_tc8_ld` through their touch twin
    /// (`w4a16_gemv_tc8_touch`): `weight` rows are `ld_half` packed bytes and
    /// `ld_groups` scale bytes apart (`k / 2` and `k / 16` for a whole weight).
    #[allow(clippy::too_many_arguments)]
    pub fn w4a16_tc8(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ld_half: u32,
        ld_groups: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=8).contains(&m) && k.is_multiple_of(16) && ld_half >= k / 2 && ld_groups >= k / 16,
            "w4a16 tensor-core touch GEMV: m={m} k={k} ld={ld_half}/{ld_groups}"
        );
        let grid = div_ceil(n, 16);
        let (rows, ctas) = self.geometry(n, k / 2 + k / 16, grid);
        KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(ld_half)
            .arg_u32(ld_groups)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// `w4a16_gemv_batch2` / `_batch3` (`m` = 2 or 3) through their PDL touch
    /// twin (`w4a16_gemv_batch3_touch`).
    #[allow(clippy::too_many_arguments)]
    pub fn w4a16_batch3(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (2..=3).contains(&m),
            "w4a16 batch3 touch GEMV takes 2 or 3 rows, got {m}"
        );
        let grid = div_ceil(n, 4);
        let (rows, ctas) = self.geometry(n, k / 2 + k / 16, grid);
        KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// `mxfp8_gemv_tc8` through its touch twin (`mxfp8_gemv_tc8_touch`).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp8_tc8(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        data: DevicePtr,
        scales: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        out_stride: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=8).contains(&m) && k.is_multiple_of(32),
            "mxfp8 touch GEMV: m={m} k={k} unsupported"
        );
        let grid = div_ceil(n, 16);
        let (rows, ctas) = self.geometry(n, k + k / 32, grid);
        KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(data)
            .arg_ptr(scales)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(out_stride)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// The two trailing arguments of `w4a16_gemv_batch5_qkv_touch`: the Q
    /// plane's touch geometry (`n` outputs of `k` inputs, four per CTA).
    pub fn batch5_qkv_args(&self, n: u32, k: u32) -> (u32, u32) {
        self.geometry(n, k / 2 + k / 16, div_ceil(n, 4))
    }
}

/// The touch twin of the NVFP4 verify tier for `m` (2..=8) rows, launched in
/// its place: `w4a16_gemv_batch2/3` below four rows, else `tier` when it is
/// `w4a16_gemv_tc8`. None when there is no twin to take (flag off, another
/// tier, a target without it): the caller launches the tier itself.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_verify_touch(
    gpu: &dyn GpuBackend,
    tier: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Option<Result<()>> {
    if (2..=3).contains(&m) {
        let touch = gemv_touch(gpu, "w4a16_gemv_batch3_touch")?;
        return Some(touch.w4a16_batch3(gpu, input, weight, output, m, n, k, stream));
    }
    if crate::layers::w4a16_gemv_tiers::tc_rows(tier) != Some(8) {
        return None;
    }
    let touch = gemv_touch(gpu, "w4a16_gemv_tc8_touch")?;
    Some(touch.w4a16_tc8(gpu, input, weight, output, m, n, k, k / 2, k / 16, stream))
}

#[cfg(test)]
#[path = "gemv_touch_tests.rs"]
mod tests;
