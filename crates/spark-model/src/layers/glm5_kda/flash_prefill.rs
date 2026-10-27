// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in, allocation-free bridge to the separately pinned FlashKDA library
//! (MoonshotAI FlashKDA, <https://github.com/MoonshotAI/FlashKDA> @ 1ce47ea3,
//! MIT; loaded at run time, not part of this tree).
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use spark_runtime::gpu::DevicePtr;
use std::ffi::c_void;

use crate::layer::ForwardContext;

type Run = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u64,
    u64,
    u64,
    i32,
    i32,
    f32,
    f32,
    *mut c_void,
) -> i32;

/// `atlas_mango_flash_prefill_planes` (bridge ABI 2): q, k and v in their
/// own planes, then gate, raw beta, a, bias, state, output, workspace, aux,
/// their capacities, rows, heads, scale, lower bound and the stream.
type RunPlanes = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u64,
    u64,
    i32,
    i32,
    f32,
    f32,
    *mut c_void,
) -> i32;

pub(super) struct FlashPrefill {
    // Retain the mapping as long as any copied function pointer can be called.
    _library: Library,
    run: Run,
    /// The planar entry, when the bridge has it and
    /// `ATLAS_GLM_KDA_FLASH_PLANES=1` asks for it.
    run_planes: Option<RunPlanes>,
    workspace_size: unsafe extern "C" fn(i32, i32, i32) -> i64,
    /// Local KDA heads: 32 at TP2, 16 at TP4 (`QUALIFIED_HEADS`).
    heads: usize,
}

/// Local head counts the bridge takes (TP2's 32, TP4's 16): FlashKDA's
/// kernels are generic over heads, and the bridge's packing and state
/// transposes run one block per head.
const QUALIFIED_HEADS: [usize; 2] = [16, 32];

fn enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => bail!("ATLAS_KDA_FLASH_PREFILL must be 0 or 1, got {value:?}"),
    }
}

/// Rows one library call accepts (compiled into the pinned bridge).
const PIECE_ROWS: std::ops::RangeInclusive<usize> = 2048..=4100;

pub(super) fn eligible(tokens: usize, decode: bool, graph_capture: bool) -> bool {
    !decode && !graph_capture && tokens >= *PIECE_ROWS.start()
}

/// Near-equal consecutive pieces within `PIECE_ROWS` covering `tokens`
/// rows; the recurrent state carries across calls in place.
fn pieces(tokens: usize) -> impl Iterator<Item = (usize, usize)> {
    let n = tokens.div_ceil(*PIECE_ROWS.end());
    let piece = tokens.div_ceil(n);
    (0..n).map(move |i| (i * piece, piece.min(tokens - i * piece)))
}

fn checked_ranges(ranges: &[(DevicePtr, usize)]) -> Result<()> {
    for (i, &(ptr, bytes)) in ranges.iter().enumerate() {
        ensure!(
            !ptr.is_null() && bytes > 0,
            "FlashKDA null or empty operand"
        );
        let end = ptr
            .0
            .checked_add(bytes as u64)
            .context("FlashKDA address overflow")?;
        for &(other, n) in &ranges[..i] {
            let other_end = other
                .0
                .checked_add(n as u64)
                .context("FlashKDA address overflow")?;
            ensure!(
                end <= other.0 || other_end <= ptr.0,
                "FlashKDA overlapping operands"
            );
        }
    }
    Ok(())
}

impl FlashPrefill {
    pub(super) fn load(heads: usize, dim: usize, lower: f32) -> Result<Option<Self>> {
        if !enabled(std::env::var("ATLAS_KDA_FLASH_PREFILL").ok().as_deref())? {
            return Ok(None);
        }
        let path = std::env::var("ATLAS_KDA_FLASH_LIBRARY")
            .context("ATLAS_KDA_FLASH_LIBRARY is required when FlashKDA is enabled")?;
        Self::open(&path, heads, dim, lower).map(Some)
    }

    /// The bridge at `path` for `heads` local heads.
    fn open(path: &str, heads: usize, dim: usize, lower: f32) -> Result<Self> {
        ensure!(
            QUALIFIED_HEADS.contains(&heads) && dim == 128 && lower == -5.0,
            "FlashKDA screen is qualified only for H16/H32, D128, lower_bound=-5 (got H{heads}/D{dim})"
        );
        ensure!(
            std::path::Path::new(path).is_absolute(),
            "FlashKDA library path must be absolute"
        );
        // SAFETY: the operator explicitly selects this trusted pinned native library.
        // ABI version is checked before resolving/calling its stateful entry point.
        let library =
            unsafe { Library::new(path) }.with_context(|| format!("Load FlashKDA {path}"))?;
        let version = unsafe {
            library.get::<unsafe extern "C" fn() -> i32>(b"atlas_mango_flash_abi_version\0")?
        };
        let version = unsafe { version() };
        ensure!(matches!(version, 1 | 2), "FlashKDA bridge ABI mismatch");
        let run = unsafe { *library.get::<Run>(b"atlas_mango_flash_prefill\0")? };
        // ATLAS_GLM_KDA_FLASH_PLANES: the convolution writes the bridge's
        // q / k / v planes itself, so the bridge copies nothing.
        let planes_requested = std::env::var("ATLAS_GLM_KDA_FLASH_PLANES").as_deref() == Ok("1");
        ensure!(
            !planes_requested || version >= 2,
            "ATLAS_GLM_KDA_FLASH_PLANES=1 needs FlashKDA bridge ABI 2 (this one is {version})"
        );
        let run_planes = planes_requested
            .then(|| unsafe { library.get::<RunPlanes>(b"atlas_mango_flash_prefill_planes\0") })
            .transpose()?
            .map(|f| *f);
        let workspace_size = unsafe {
            *library.get::<unsafe extern "C" fn(i32, i32, i32) -> i64>(
                b"atlas_flash_kda_workspace_size\0",
            )?
        };
        tracing::info!(
            library = path,
            "FlashKDA prefill enabled for >=2048 rows (2048..=4100 per call); decode and verification unchanged"
        );
        Ok(Self {
            _library: library,
            run,
            run_planes,
            workspace_size,
            heads,
        })
    }

    /// Whether [`Self::forward_planes`] serves: the convolution then writes
    /// q / k / v planes for it.
    pub(super) fn takes_planes(&self) -> bool {
        self.run_planes.is_some()
    }

    /// [`Self::forward`] over q / k / v planes (`3 x [tokens, heads * 128]`
    /// at `planes`): the same library call on the same values, without the
    /// bridge's packed copy.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_planes(
        &self,
        planes: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        bias: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            eligible(tokens, false, ctx.graph_capture),
            "Unqualified FlashKDA invocation"
        );
        let plane_row = self.heads * 128 * 2;
        let plane = tokens * plane_row;
        let (b, sizes) = (ctx.buffers, ctx.buffers.sizes());
        for (row, rows) in pieces(tokens) {
            let [q, k, v] = [0, 1, 2].map(|i| planes.offset(i * plane + row * plane_row));
            self.run_piece_planes(
                [
                    q,
                    k,
                    v,
                    gate.offset(row * plane_row),
                    beta.offset(row * self.heads * 2),
                    a,
                    bias,
                    state,
                    output.offset(row * plane_row),
                ],
                rows,
                [
                    (b.expert_gate_out(), sizes.expert_gate_out),
                    (b.expert_up_out(), sizes.expert_up_out),
                ],
                stream,
            )?;
        }
        Ok(())
    }

    /// One planar library call over `tokens` rows of `[q, k, v, gate, beta,
    /// a, bias, state, output]`, with the workspace and aux scratch given as
    /// `(ptr, capacity)`.
    fn run_piece_planes(
        &self,
        [q, k, v, gate, beta, a, bias, state, output]: [DevicePtr; 9],
        tokens: usize,
        [(workspace, workspace_cap), (aux, aux_cap)]: [(DevicePtr, usize); 2],
        stream: u64,
    ) -> Result<()> {
        let run = self
            .run_planes
            .context("FlashKDA planar entry not loaded")?;
        ensure!(PIECE_ROWS.contains(&tokens), "Unqualified FlashKDA piece");
        let heads = self.heads;
        let plane = tokens * heads * 128 * 2;
        let state_bytes = heads * 128 * 128 * 4;
        let beta_bytes = tokens * heads * 2;
        let aux_required = (state_bytes + beta_bytes).next_multiple_of(128) + 20;
        let requested = unsafe { (self.workspace_size)(tokens as i32, heads as i32, 1) };
        ensure!(requested > 0, "FlashKDA invalid workspace size");
        let workspace_required = usize::try_from(requested)?;
        ensure!(
            workspace_cap >= workspace_required && aux_cap >= aux_required,
            "FlashKDA scratch capacity exceeded"
        );
        checked_ranges(&[
            (q, plane),
            (k, plane),
            (v, plane),
            (gate, plane),
            (beta, beta_bytes),
            (a, heads * 4),
            (bias, heads * 128 * 4),
            (state, state_bytes),
            (output, plane),
            (workspace, workspace_required),
            (aux, aux_required),
        ])?;
        let p = |x: DevicePtr| x.0 as *mut c_void;
        // SAFETY: as `run_piece`: qualified geometry, capacities and disjoint
        // live byte ranges checked above, all enqueued on this stream.
        let status = unsafe {
            run(
                p(q),
                p(k),
                p(v),
                p(gate),
                p(beta),
                p(a),
                p(bias),
                p(state),
                p(output),
                p(workspace),
                p(aux),
                workspace_cap as u64,
                aux_cap as u64,
                tokens as i32,
                heads as i32,
                1.0 / (128.0f32).sqrt(),
                -5.0,
                stream as *mut c_void,
            )
        };
        ensure!(
            status == 0,
            "FlashKDA launch failed ({status}); state may be partially updated"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        qkv: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        bias: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            eligible(tokens, false, ctx.graph_capture),
            "Unqualified FlashKDA invocation"
        );
        let plane_row = self.heads * 128 * 2;
        for (row, rows) in pieces(tokens) {
            self.forward_piece(
                qkv.offset(row * 3 * plane_row),
                gate.offset(row * plane_row),
                beta.offset(row * self.heads * 2),
                a,
                bias,
                state,
                output.offset(row * plane_row),
                rows,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_piece(
        &self,
        qkv: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        bias: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let sizes = ctx.buffers.sizes();
        let b = ctx.buffers;
        self.run_piece(
            [qkv, gate, beta, a, bias, state, output],
            tokens,
            [
                (b.ssm_qkvz(), sizes.ssm_qkvz),
                (b.expert_gate_out(), sizes.expert_gate_out),
                (b.expert_up_out(), sizes.expert_up_out),
            ],
            stream,
        )
    }

    /// One library call over `tokens` rows of `[qkv, gate, beta, a, bias,
    /// state, output]`, with the packed, workspace and aux scratch given as
    /// `(ptr, capacity)`.
    fn run_piece(
        &self,
        [qkv, gate, beta, a, bias, state, output]: [DevicePtr; 7],
        tokens: usize,
        [
            (packed, packed_cap),
            (workspace, workspace_cap),
            (aux, aux_cap),
        ]: [(DevicePtr, usize); 3],
        stream: u64,
    ) -> Result<()> {
        ensure!(PIECE_ROWS.contains(&tokens), "Unqualified FlashKDA piece");
        let heads = self.heads;
        let plane = tokens * heads * 128 * 2;
        let state_bytes = heads * 128 * 128 * 4;
        let beta_bytes = tokens * heads * 2;
        let aux_required = (state_bytes + beta_bytes).next_multiple_of(128) + 20;
        // Ask the exact loaded kernel library, keeping its sizing formula authoritative.
        let requested = unsafe { (self.workspace_size)(tokens as i32, heads as i32, 1) };
        ensure!(requested > 0, "FlashKDA invalid workspace size");
        let workspace_required = usize::try_from(requested)?;
        ensure!(
            packed_cap >= 3 * plane
                && workspace_cap >= workspace_required
                && aux_cap >= aux_required,
            "FlashKDA scratch capacity exceeded"
        );
        checked_ranges(&[
            (qkv, 3 * plane),
            (gate, plane),
            (beta, beta_bytes),
            (a, heads * 4),
            (bias, heads * 128 * 4),
            (state, state_bytes),
            (output, plane),
            (packed, 3 * plane),
            (workspace, workspace_required),
            (aux, aux_required),
        ])?;
        let p = |x: DevicePtr| x.0 as *mut c_void;
        // SAFETY: qualified geometry, capacities and disjoint live byte ranges were
        // checked above. The existing forward CUDA context/stream owns all pointers.
        // The shim enqueues all work on this stream and restores canonical state.
        let status = unsafe {
            (self.run)(
                p(qkv),
                p(gate),
                p(beta),
                p(a),
                p(bias),
                p(state),
                p(output),
                p(packed),
                p(workspace),
                p(aux),
                packed_cap as u64,
                workspace_cap as u64,
                aux_cap as u64,
                tokens as i32,
                heads as i32,
                1.0 / (128.0f32).sqrt(),
                -5.0,
                stream as *mut c_void,
            )
        };
        ensure!(
            status == 0,
            "FlashKDA launch failed ({status}); state may be partially updated"
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "flash_prefill_test_gpu.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configuration_is_explicit_and_strict() {
        assert!(!enabled(None).unwrap());
        assert!(!enabled(Some("0")).unwrap());
        assert!(enabled(Some("1")).unwrap());
        assert!(enabled(Some("true")).is_err());
    }
    #[test]
    fn qualification_excludes_decode_capture_and_short_tails() {
        for n in [2048, 4100, 4101, 8196, 16388] {
            assert!(eligible(n, false, false));
        }
        for n in [0, 1, 3, 2047] {
            assert!(!eligible(n, false, false));
        }
        assert!(!eligible(4096, true, false));
        assert!(!eligible(4096, false, true));
    }
    #[test]
    fn pieces_tile_rows_within_the_bridge_range() {
        for tokens in [
            2048, 4096, 4100, 4101, 4104, 6000, 8192, 8196, 12292, 16388, 65520,
        ] {
            let mut next = 0;
            for (row, rows) in pieces(tokens) {
                assert_eq!(row, next);
                assert!(PIECE_ROWS.contains(&rows), "{tokens}: piece {rows}");
                next += rows;
            }
            assert_eq!(next, tokens);
        }
    }
    #[test]
    fn overlaps_and_address_wrap_are_rejected() {
        assert!(checked_ranges(&[(DevicePtr(128), 128), (DevicePtr(256), 16)]).is_ok());
        assert!(checked_ranges(&[(DevicePtr(128), 129), (DevicePtr(256), 16)]).is_err());
        assert!(checked_ranges(&[(DevicePtr(u64::MAX), 2)]).is_err());
        assert!(checked_ranges(&[(DevicePtr::NULL, 4)]).is_err());
    }
}
