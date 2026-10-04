// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_L2_AHEAD`: prefetch the weights the verify step reads next into
//! L2 from a side stream, while its single stream waits on an all-reduce or
//! runs a latency-bound chain (default off).
//!
//! A verify step's target forward runs on one stream. Between the weight
//! GEMVs it waits on the RDMA all-reduce (the one-shot kernel is not on the
//! PDL list, so nothing behind it starts) and runs chains of small kernels
//! (HC partial / finalize / norm, the sparse-MLA indexer, the attention core)
//! with DRAM idle. The touch twins (`gemv_touch`) only use the PDL wait in
//! front of their own GEMV. At the sites below the stream records an event, a
//! side stream waits on it and launches `glm_l2_ahead`
//! (kernels/gb10/glm-5.3-flash/nvfp4/glm_l2_ahead.cuh) over the leading bytes
//! of the next weights, so their GEMVs find them in L2. The kernel only loads
//! or prefetches immutable weights and the main stream never waits on the side
//! stream: every launch, its order and every output bit of the forward are
//! unchanged (scripts/dev/glm_l2_ahead_bench.cu checks the GEMVs bitwise).
//!
//! Sites (`ATLAS_GLM_L2_AHEAD_SITES`, letters, default `afqo`):
//! - `a`: at a layer's attention all-reduce, its FFN's leading weights (the
//!   shared expert's gate/up and down, the router).
//! - `f`: at a layer's FFN all-reduce, the next layer's leading attention
//!   weights (KDA q, k; sparse MLA q_a, kv_a, q_b).
//! - `q`: sparse MLA, before the indexer chain of a per-owner verify: q_b.
//! - `o`: sparse MLA, after the W_uk absorb: W_uv, then o.
//!
//! `a` and `f` fire from `model::l2_ahead_comm`, which stands in for the
//! communicator of an eager K-gamma verify; `q` and `o` from the paged GLM
//! attention at 32 rows or fewer. A stream being captured never forks.
//!
//! `ATLAS_GLM_L2_AHEAD` picks the mechanism: `1` or `touch` (a discarded byte
//! load per 32-byte sector, as the touch twins), `lines` / `sectors`
//! (`prefetch.global.L2` per 128- / 32-byte unit), `last` (the evict_last
//! hint per sector) or `bulk` (`cp.async.bulk.prefetch.L2`). `_MB` (1..=24,
//! default 12) caps the bytes one site asks for, `_CTAS` (1..=96, default 32)
//! sizes the side kernel. No allocation: one stream and one event, created at
//! boot only when the flag is on.
//!
//! Prior art (docs/glm-prior-art.md): jayleaton's L2 prefetch
//! (<https://github.com/jayleaton/glm53-tensorfold-spark> patches/0460,
//! Apache-2.0): a side-stream prefetch kernel forked at a layer's attention and
//! FFN all-gathers and after its attention projections, with a byte budget a
//! site; the FFN's shared expert and router at the attention collective (our
//! `a`). MiaAI-Lab's TensorFold recipe patch `0046-glm-l2-prefetch`
//! (Apache-2.0) adds the next layer's weights at the FFN collective (our `f`)
//! and the output site after the query absorb, kv_b's value half then o (our
//! `o`). Ours: firing from a communicator wrapper around the TP reduces,
//! strided TP-split regions, the per-sector modes and site `q`; no code
//! copied.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::weight_map::{Mxfp8Weight, QuantizedWeight};

/// Where a prefetch is asked for: its letter in `ATLAS_GLM_L2_AHEAD_SITES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L2Site {
    /// Attention all-reduce: this layer's FFN weights.
    Attn,
    /// FFN all-reduce: the next layer's attention weights.
    Ffn,
    /// Sparse-MLA indexer chain: q_b.
    Index,
    /// Sparse-MLA attention core: W_uv and o.
    Output,
}

const SITE_LETTERS: [char; 4] = ['a', 'f', 'q', 'o'];

/// Regions one launch takes (`L2A_REGIONS` in the kernel).
pub const L2_REGIONS: usize = 8;

/// `rows` rows of `row_bytes` bytes, `ld` bytes apart, of an immutable weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L2Region {
    pub ptr: DevicePtr,
    pub ld: u64,
    pub row_bytes: u32,
    pub rows: u32,
}

impl L2Region {
    const NONE: Self = Self {
        ptr: DevicePtr::NULL,
        ld: 0,
        row_bytes: 0,
        rows: 0,
    };

    /// `bytes` contiguous bytes (one row).
    pub fn whole(ptr: DevicePtr, bytes: usize) -> Self {
        let bytes = bytes.min(u32::MAX as usize) as u32;
        Self {
            ptr,
            ld: u64::from(bytes),
            row_bytes: bytes,
            rows: 1,
        }
    }

    pub fn bytes(self) -> u64 {
        u64::from(self.row_bytes) * u64::from(self.rows)
    }

    /// What a GEMV reads of an NVFP4 weight of `n` rows by `k`, rows
    /// `ld_half` packed and `ld_groups` scale bytes apart: scales, then values.
    pub fn nvfp4(w: &QuantizedWeight, n: u32, k: u32, ld_half: u32, ld_groups: u32) -> [Self; 2] {
        let region = |ptr, row_bytes: u32, ld: u32| {
            if ld == row_bytes {
                Self::whole(ptr, row_bytes as usize * n as usize)
            } else {
                Self {
                    ptr,
                    ld: u64::from(ld),
                    row_bytes,
                    rows: n,
                }
            }
        };
        [
            region(w.weight_scale, k / 16, ld_groups),
            region(w.weight, k / 2, ld_half),
        ]
    }

    /// A whole MXFP8 weight of `n` rows by `k`: scales, then values.
    pub fn mxfp8(w: &Mxfp8Weight, n: usize, k: usize) -> [Self; 2] {
        [
            Self::whole(w.scales, n * k / 32),
            Self::whole(w.data, n * k),
        ]
    }
}

/// The leading `budget` bytes of `regions` in order, at most [`L2_REGIONS`]:
/// whole regions, then whole rows of the next (a one-row region: its first
/// bytes). Empty regions and null pointers (a released weight) are skipped.
pub fn l2_budgeted(regions: &[L2Region], budget: u64) -> Vec<L2Region> {
    let mut left = budget;
    let mut out = Vec::with_capacity(L2_REGIONS);
    for &r in regions.iter().filter(|r| r.bytes() > 0 && !r.ptr.is_null()) {
        if left == 0 || out.len() == L2_REGIONS {
            break;
        }
        let r = if r.bytes() <= left {
            r
        } else if r.rows > 1 {
            L2Region {
                rows: (left / u64::from(r.row_bytes)) as u32,
                ..r
            }
        } else {
            L2Region::whole(r.ptr, left as usize)
        };
        if r.rows == 0 {
            break;
        }
        left -= r.bytes();
        out.push(r);
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Settings {
    /// `L2A_*` mode of the kernel.
    mode: u32,
    budget: u64,
    ctas: u32,
    sites: [bool; 4],
}

/// The flag's settings from the four variables; `None` when off.
fn parse(
    mode: Option<&str>,
    mb: Option<&str>,
    ctas: Option<&str>,
    sites: Option<&str>,
) -> Result<Option<Settings>> {
    let mode = match mode.unwrap_or("0") {
        "0" => return Ok(None),
        "lines" => 0,
        "sectors" => 1,
        "last" => 2,
        "1" | "touch" => 3,
        "bulk" => 4,
        other => {
            bail!("ATLAS_GLM_L2_AHEAD: 0, 1, touch, lines, sectors, last or bulk, not {other:?}")
        }
    };
    let number = |name: &str, value: Option<&str>, max: u64, default: u64| match value {
        None => Ok(default),
        Some(v) => match v.parse::<u64>() {
            Ok(n) if (1..=max).contains(&n) => Ok(n),
            _ => bail!("ATLAS_GLM_L2_AHEAD_{name}: 1..={max}, not {v:?}"),
        },
    };
    let letters = sites.unwrap_or("afqo");
    if letters.is_empty() || letters.chars().any(|c| !SITE_LETTERS.contains(&c)) {
        bail!("ATLAS_GLM_L2_AHEAD_SITES: letters of \"afqo\", not {letters:?}");
    }
    Ok(Some(Settings {
        mode,
        budget: number("MB", mb, 24, 12)? << 20,
        ctas: number("CTAS", ctas, 96, 32)? as u32,
        sites: SITE_LETTERS.map(|c| letters.contains(c)),
    }))
}

/// Read once. A malformed value turns the flag off with an error in the log
/// (the forward is the same either way).
fn settings() -> Option<Settings> {
    static SETTINGS: OnceLock<Option<Settings>> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok();
        parse(
            var("ATLAS_GLM_L2_AHEAD").as_deref(),
            var("ATLAS_GLM_L2_AHEAD_MB").as_deref(),
            var("ATLAS_GLM_L2_AHEAD_CTAS").as_deref(),
            var("ATLAS_GLM_L2_AHEAD_SITES").as_deref(),
        )
        .unwrap_or_else(|e| {
            tracing::error!("{e:#}; L2 prefetch off");
            None
        })
    })
}

/// The kernel, its side stream and the fork event.
struct Lane {
    kernel: KernelHandle,
    stream: u64,
    event: u64,
}

static LANE: OnceLock<Option<Lane>> = OnceLock::new();

/// Set by the first failed fork: the lane stays off from then on.
static LANE_FAILED: AtomicBool = AtomicBool::new(false);

/// Resolve the kernel and create the side stream and event once, from the GLM
/// KDA layer constructor (the kernel lives only in the GLM-5.3-Flash target).
/// Off: no lookup and nothing created.
pub fn l2_ahead_resolve(gpu: &dyn GpuBackend) {
    LANE.get_or_init(|| {
        let s = settings()?;
        let kernel = crate::layers::try_kernel(gpu, "w4a16_gemv", "glm_l2_ahead");
        let lane = (kernel.0 != 0)
            .then(|| Ok::<_, anyhow::Error>((gpu.create_stream()?, gpu.create_event()?)))
            .transpose();
        match lane {
            Ok(Some((stream, event))) => {
                tracing::info!(
                    "ATLAS_GLM_L2_AHEAD: side-stream L2 prefetch, mode {} / {} MiB a site / {} CTAs / sites {}",
                    s.mode,
                    s.budget >> 20,
                    s.ctas,
                    SITE_LETTERS
                        .iter()
                        .zip(s.sites)
                        .filter_map(|(c, on)| on.then_some(*c))
                        .collect::<String>(),
                );
                Some(Lane {
                    kernel,
                    stream,
                    event,
                })
            }
            Ok(None) => {
                tracing::warn!("ATLAS_GLM_L2_AHEAD ignored: this target has no glm_l2_ahead");
                None
            }
            Err(e) => {
                tracing::warn!("ATLAS_GLM_L2_AHEAD ignored: {e:#}");
                None
            }
        }
    });
}

/// Whether `site` prefetches (flag on, site listed, kernel resolved, no fork
/// failed).
pub fn l2_ahead_enabled(site: L2Site) -> bool {
    settings().is_some_and(|s| s.sites[site as usize])
        && LANE.get().is_some_and(Option::is_some)
        && !LANE_FAILED.load(Ordering::Relaxed)
}

/// Fork the side stream from `stream` here and ask for the leading budget of
/// `regions` (in the order the main stream reads them). A no-op when `site` is
/// off or `stream` is being captured. Best effort: the callers sit just before
/// a collective, so a failed fork logs once and turns the lane off rather than
/// failing this rank's forward while its peer waits in the collective.
pub fn l2_ahead_prefetch(gpu: &dyn GpuBackend, stream: u64, site: L2Site, regions: &[L2Region]) {
    if let (Some(s), Some(Some(lane))) = (settings(), LANE.get()) {
        best_effort(&LANE_FAILED, || fork(gpu, lane, s, stream, site, regions));
    }
}

/// Runs `fork` unless `failed`; its first error is logged and sets `failed`.
fn best_effort(failed: &AtomicBool, fork: impl FnOnce() -> Result<()>) {
    if failed.load(Ordering::Relaxed) {
        return;
    }
    if let Err(e) = fork()
        && !failed.swap(true, Ordering::Relaxed)
    {
        tracing::error!("ATLAS_GLM_L2_AHEAD: {e:#}; L2 prefetch off");
    }
}

fn fork(
    gpu: &dyn GpuBackend,
    lane: &Lane,
    s: Settings,
    stream: u64,
    site: L2Site,
    regions: &[L2Region],
) -> Result<()> {
    if !s.sites[site as usize] || gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    let regions = l2_budgeted(regions, s.budget);
    if regions.is_empty() {
        return Ok(());
    }
    gpu.record_event(lane.event, stream)?;
    gpu.stream_wait_event(lane.stream, lane.event)?;
    let mut launch = KernelLaunch::new(gpu, lane.kernel)
        .grid([s.ctas, 1, 1])
        .block([256, 1, 1]);
    for i in 0..L2_REGIONS {
        let r = regions.get(i).copied().unwrap_or(L2Region::NONE);
        launch = launch
            .arg_ptr(r.ptr)
            .arg_u64(r.ld)
            .arg_u32(r.row_bytes)
            .arg_u32(r.rows);
    }
    launch.arg_u32(s.mode).launch(lane.stream)
}

#[cfg(test)]
#[path = "l2_ahead_tests.rs"]
mod tests;
