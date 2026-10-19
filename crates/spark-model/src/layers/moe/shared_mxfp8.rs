// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's MXFP8 shared expert, on the checkpoint's own bytes.
//!
//! `nvidia/Qwen3.8-Flash-Next-FP4-Local` ships every layer's
//! `mlp.shared_expert.{gate,up,down}_proj` MXFP8 (E4M3 + UE8M0 per 32 K
//! values), where the NVFP4/FP8 releases ship BF16. The fused routed + shared
//! MoE kernels read an NVFP4 shared expert, so this one runs beside them, as
//! the reference runs it (vLLM b12x MXFP8 linears around `silu_and_mul`):
//! the routed kernels get null shared weights (the TC v3 units then plan no
//! shared unit; the v1/v2 units and the originals write zeros), and this
//! writes the shared expert's down output where they would have, before the
//! blend or the EP reduce reads it.
//!
//! * Decode, verify and batch rows ([`Mxfp8SharedExpert::decode`]): `[gate;
//!   up]` on the `_swiglu` MXFP8 GEMV tiers (BF16 activations, SiLU * up in
//!   the epilogue), down on the plain tiers. Both are row-invariant, so a row
//!   computes the same bytes alone, in a verify or in a batch (spec ==
//!   serial).
//! * Prefill ([`Mxfp8SharedExpert::prefill`]): W8A8 ([`ops::MxW8a8`], MXFP8
//!   activations on cuBLASLt's block-scaled GEMM) over the same twins, also
//!   row-invariant, so an SP half equals that half of the whole chunk
//!   (`ATLAS_QWEN4EXP_PREFILL_SP_SHARED`). Without
//!   `ATLAS_QWEN4EXP_PREFILL_W8A8` (`=0`, or under `_PREFILL_BF16_PROJ`,
//!   prefill at decode precision) it runs the decode tiers. W8A8, not the
//!   reference's other mode (A16: BF16 activations on the exact BF16
//!   dequant), because it is the faster one: GB10, per layer, 2048 / 8192 /
//!   16384 rows 0.25 / 0.99 / 2.14 ms against 0.38 / 1.89 / 3.81 ms
//!   (`shared_mxfp8_prefill_bench`), with no resident BF16 copy. Its output
//!   is ~4.5e-2 (rel-L2) from A16's on layers 0, 3, 47; the NVFP4
//!   requantization this replaced was ~14% (decode) and ~17% (prefill, q38).
//!
//! The SiLU * up rounds as vLLM's `silu_and_mul` on BF16 GEMM outputs: gate
//! and up to BF16, SiLU to BF16, the product to BF16 (`mx_swiglu`). The
//! sigmoid `shared_expert_gate` stays the checkpoint's BF16 in the blend.
//!
//! Replicated on every EP rank, as the NVFP4 shared expert was: each rank
//! computes all of it, and the EP reduce adds it once after the all-reduce.
//! There (TP=EP=2 decode, verify and batch), it runs on a side stream
//! ([`SideLane`]) forked at the MoE's entry and joined after the all-reduce
//! is issued: it reads only the MoE input, so it overlaps the router, the
//! routed experts and the all-reduce's wait on the peer instead of adding
//! its weight bytes to the step. The same kernels on the same operands:
//! the bytes do not change. GB10, one EP2 rank's MoE layer (the TC v3 units,
//! `scripts/dev/qwen4exp_moe_c8_bench.sh time tc3`, min of 7, us; NVFP4
//! shared unit / MXFP8 in line / MXFP8 on the side lane): 1 row 77.0 / 89.4
//! / 79.6, 4 rows 265.3 / 266.5 / 261.9, 16 rows 520.3 / 521.8 / 521.5, 32
//! rows (C8, 70 unique experts) 852.4 / 856.7 / 845.1.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::*;
use crate::layers::ops::{
    MX_QUANT_KERNEL, MXFP8_ACT_MODULE, MXFP8_BLOCK, MxAct, MxW8a8, qwen4exp_w8a8,
};
use crate::weight_map::{Mxfp8Weight, RawMxfp8, mxfp8_rows_twin, raw_mxfp8};

/// The kernel module of the shared expert's GEMV tiers and prefill SiLU.
const MODULE: &str = "mxfp8_gemv";

#[derive(Debug, Clone, Copy)]
struct Kernels {
    /// `mxfp8_gemv_tc{8,16,32}`: down.
    rows: [KernelHandle; 3],
    /// `mxfp8_gemv_tc{8,16,32}_swiglu`: gate/up + SiLU * up.
    swiglu: [KernelHandle; 3],
    /// Prefill: the input's MXFP8 quant, and the SiLU's.
    quant: KernelHandle,
    swiglu_quant: KernelHandle,
}

/// Rows the side lane's own buffers hold: any decode, verify or batch step
/// (a wider one runs in line).
const SIDE_ROWS: usize = 256;

/// The EP decode's side stream for the shared expert, its fork / join
/// events and its own SiLU * up and output rows (no buffer the routed path
/// writes). One per process: the layers fork and join it in turn.
#[derive(Debug, Clone, Copy)]
struct SideLane {
    stream: u64,
    fork: u64,
    join: u64,
    act: DevicePtr,
    out: DevicePtr,
}

impl SideLane {
    fn get(gpu: &dyn GpuBackend, [hidden, inter]: [usize; 2]) -> Result<Self> {
        static LANE: std::sync::Mutex<Option<SideLane>> = std::sync::Mutex::new(None);
        let mut lane = LANE.lock().map_err(|_| anyhow::anyhow!("side lane lock"))?;
        if let Some(l) = *lane {
            return Ok(l);
        }
        let l = Self {
            stream: gpu.create_stream()?,
            fork: gpu.create_event()?,
            join: gpu.create_event()?,
            act: gpu.alloc(SIDE_ROWS * inter * 2)?,
            out: gpu.alloc(SIDE_ROWS * hidden * 2)?,
        };
        *lane = Some(l);
        Ok(l)
    }
}

/// One layer's MXFP8 shared expert; see the module docs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mxfp8SharedExpert {
    /// `[gate; up]`, `[2 * inter, hidden]`.
    gate_up: Mxfp8Weight,
    /// `[hidden, inter]`.
    down: Mxfp8Weight,
    /// The W8A8 prefill operands over the twins (gate/up, down).
    w8a8: Option<(MxW8a8, MxW8a8)>,
    /// Under EP (see the module docs).
    side: Option<SideLane>,
    k: Kernels,
    hidden: usize,
    inter: usize,
}

impl Mxfp8SharedExpert {
    /// The twins of layer `lp`'s MXFP8 shared expert, copied from the raw
    /// checkpoint tensors (the loader's per-layer sweep releases those), and
    /// their W8A8 operands when `ATLAS_QWEN4EXP_PREFILL_W8A8` is on.
    /// Synchronizes the default stream.
    pub(crate) fn load(
        store: &WeightStore,
        lp: &str,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let (hidden, inter) = (config.hidden_size, config.shared_expert_intermediate_size);
        let raw = |name: &str| {
            let prefix = format!("{lp}.mlp.shared_expert.{name}");
            raw_mxfp8(store, &prefix)?
                .with_context(|| format!("{prefix}: not MXFP8 in a checkpoint whose backbone is"))
        };
        let parts = [raw("gate_proj")?, raw("up_proj")?, raw("down_proj")?];
        let w8a8 = qwen4exp_w8a8::enabled().then_some(config.max_batch_tokens);
        let mut mx =
            Self::from_raw(gpu, parts, [hidden, inter], w8a8).with_context(|| lp.to_string())?;
        if config.ep_world_size > 1 {
            mx.side = Some(SideLane::get(gpu, [hidden, inter])?);
        }
        Ok(mx)
    }

    /// [`Self::load`] from raw `[gate, up, down]` MXFP8 tensors of a
    /// `[hidden, inter]` expert, with W8A8 operands tuned up to `w8a8` rows
    /// when it is `Some`.
    pub(crate) fn from_raw(
        gpu: &dyn GpuBackend,
        [gate, up, down]: [RawMxfp8; 3],
        [hidden, inter]: [usize; 2],
        w8a8: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            [gate, up].iter().all(|w| (w.n, w.k) == (inter, hidden))
                && (down.n, down.k) == (hidden, inter)
                && inter.is_multiple_of(MXFP8_BLOCK),
            "MXFP8 shared expert gate {:?} / up {:?} / down {:?}, not [{inter}, {hidden}] / \
             [{hidden}, {inter}]",
            (gate.n, gate.k),
            (up.n, up.k),
            (down.n, down.k),
        );
        let stream = gpu.default_stream();
        let (gate_up, ..) =
            mxfp8_rows_twin(gpu, &[(gate, &[inter]), (up, &[inter])], 0, 1, stream)?;
        let (down, ..) = mxfp8_rows_twin(gpu, &[(down, &[hidden])], 0, 1, stream)?;
        let w8a8 = match w8a8 {
            Some(m_max) => Some((
                MxW8a8::from_twin(gpu, &gate_up, [2 * inter, hidden], m_max, stream)?,
                MxW8a8::from_twin(gpu, &down, [hidden, inter], m_max, stream)?,
            )),
            None => None,
        };
        let k = |name: &str| gpu.kernel(MODULE, name);
        let k = Kernels {
            rows: [
                k("mxfp8_gemv_tc8")?,
                k("mxfp8_gemv_tc16")?,
                k("mxfp8_gemv_tc32")?,
            ],
            swiglu: [
                k("mxfp8_gemv_tc8_swiglu")?,
                k("mxfp8_gemv_tc16_swiglu")?,
                k("mxfp8_gemv_tc32_swiglu")?,
            ],
            quant: gpu.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL)?,
            swiglu_quant: k("mxfp8_swiglu_quant_act")?,
        };
        Ok(Self {
            gate_up,
            down,
            w8a8,
            side: None,
            k,
            hidden,
            inter,
        })
    }

    /// `rows` rows of `input` (`[rows, hidden]`) into `out` (`[rows, hidden]`)
    /// on the decode tiers; `act` (`[rows, inter]` BF16) takes SiLU * up.
    pub(crate) fn decode(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
        act: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (h, i) = (self.hidden, self.inter);
        ops::mxfp8_swiglu_rows(
            gpu,
            &self.k.swiglu,
            &self.gate_up,
            input,
            act,
            rows,
            i,
            h,
            stream,
        )?;
        ops::mxfp8_gemv_rows(
            gpu,
            &self.k.rows,
            &self.down,
            act,
            out,
            rows,
            h,
            i,
            h,
            stream,
        )
    }

    /// A prefill chunk's `rows` rows: W8A8 when installed (`mx` holds the
    /// input's MXFP8 image, then SiLU * up's, so it must fit `[rows,
    /// hidden]`; `gu` takes the `[rows, 2 * inter]` BF16 gate/up), else
    /// [`Self::decode`] (into `act`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
        mx: MxAct,
        [act, gu]: [DevicePtr; 2],
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some((gate_up, down)) = self.w8a8 else {
            return self.decode(gpu, input, rows, act, out, stream);
        };
        ops::mx_quant_rows(gpu, self.k.quant, input, mx, 0..rows, self.hidden, stream)?;
        gate_up.gemm(mx, gu, rows, stream)?;
        // The gate/up GEMM has read the input's image (stream order).
        let (m, i) = (rows as u32, self.inter as u32);
        ops::mxfp8_swiglu_quant_act(gpu, self.k.swiglu_quant, gu, mx.q, mx.sf, m, i, stream)?;
        down.gemm(mx, out, rows, stream)
    }
}

impl MoeLayer {
    /// An EP decode / verify / batch step of `rows` rows: fork the MXFP8
    /// shared expert of `input` onto the side lane (module docs) and return
    /// where its output lands, for [`Self::forward_ep_reduce`] to blend
    /// after [`Self::join_mxfp8_shared`]. `None`: nothing forked, the step
    /// runs it in line (`run_unfused_shared_expert`).
    pub(super) fn fork_mxfp8_shared(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let Some((mx, lane)) = self
            .mxfp8_shared_expert
            .as_ref()
            .and_then(|mx| mx.side.map(|l| (mx, l)))
        else {
            return Ok(None);
        };
        if ctx.comm.is_none() || ctx.config.ep_world_size < 2 || rows > SIDE_ROWS {
            return Ok(None);
        }
        let gpu = ctx.gpu;
        gpu.record_event(lane.fork, stream)?;
        gpu.stream_wait_event(lane.stream, lane.fork)?;
        mx.decode(gpu, input, rows, lane.act, lane.out, lane.stream)?;
        gpu.record_event(lane.join, lane.stream)?;
        Ok(Some(lane.out))
    }

    /// `stream` waits for the forked shared expert ([`Self::fork_mxfp8_shared`]).
    pub(super) fn join_mxfp8_shared(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let lane = self
            .mxfp8_shared_expert
            .and_then(|mx| mx.side)
            .ok_or_else(|| anyhow::anyhow!("no MXFP8 shared expert side lane to join"))?;
        ctx.gpu.stream_wait_event(stream, lane.join)
    }

    /// Install this layer's MXFP8 shared expert. The NVFP4 shared expert
    /// must be absent (the loader skips it): every fused kernel then runs
    /// without one, and this is the only shared expert there is.
    pub(crate) fn set_mxfp8_shared_expert(&mut self, shared: Mxfp8SharedExpert) -> Result<()> {
        self.btile_storage.require_legacy()?;
        let sh = &self.weights.shared_expert;
        ensure!(
            sh.gate_proj.is_null()
                && sh.up_proj.is_null()
                && sh.down_proj.is_null()
                && self.bf16_shared_expert.is_none()
                && self.fp8_shared_expert.is_none(),
            "an MXFP8 shared expert replaces the layer's shared expert; it never joins one"
        );
        self.mxfp8_shared_expert = Some(shared);
        Ok(())
    }

    /// The prefill arm of an MXFP8 shared expert ([`Mxfp8SharedExpert::
    /// prefill`]) over `n` rows of `input` into `down_out`, with the prefill
    /// shared expert's scratch (`act`: `ssm_deinterleaved`, `gu`:
    /// `ssm_qkvz`). `Ok(false)`: no MXFP8 shared expert, nothing launched.
    pub(super) fn try_mxfp8_shared_prefill(
        &self,
        input: DevicePtr,
        n: u32,
        [act, gu, down_out]: [DevicePtr; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(mx) = &self.mxfp8_shared_expert else {
            return Ok(false);
        };
        let (n, sizes) = (n as usize, ctx.buffers.sizes());
        ensure!(
            sizes.ssm_deinterleaved >= n * mx.inter * 2
                && sizes.ssm_qkvz >= n * 2 * mx.inter * 2
                && sizes.attn_output >= n * mx.hidden * 2,
            "MXFP8 shared expert: {n} prefill rows overrun its scratch"
        );
        let scratch = ops::mx_act_scratch(ctx.buffers, n, mx.hidden)?;
        mx.prefill(ctx.gpu, input, n, scratch, [act, gu], down_out, stream)?;
        Ok(true)
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "shared_mxfp8_test_gpu.rs"]
mod gpu_tests;
