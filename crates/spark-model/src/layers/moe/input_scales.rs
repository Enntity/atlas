// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_FP4L_INPUT_SCALES=1`: the routed experts' NVFP4 activations
//! quantized with the checkpoint's calibrated input scales, as the engine
//! NVIDIA's GLM-5.3-Flash-FP4-Local card names (vLLM's b12x A4 prefill:
//! `vllm/model_executor/layers/fused_moe/b12x.py`, flashinfer b12x
//! `prefill_a4.py`), instead of with global scale 1.
//!
//! - gate/up input: one global scale for every expert, `g1 = 1 /
//!   max_e(input_scale)` over the gate and up projections (b12x takes the
//!   `amin` of `1 / w13_input_scale`; vLLM's FlashInfer lanes the same max),
//!   so a token is still quantized once: block scale `e4m3(amax * g1 / 6)`,
//!   codes `e2m1(x * g1 / sf)` (`nvfp4_quantize_group` with `scale2 = 1 /
//!   g1`).
//! - down input (the BF16-rounded SiLU·mul): each expert's own `g2[e] = 1 /
//!   w2_input_scale[e]` (`silu_nvfp4_group_scale`'s `g`).
//! - each GEMM's per-expert alpha takes the activation scale back: the gate
//!   and up tables' `weight_scale_2 * max input_scale`, the down table's
//!   `weight_scale_2 * input_scale[e]` (separate tables; the W4A16 kernels
//!   keep the plain ones).
//!
//! FP4-Local ships `input_scale` per expert in `model-inputscales.safetensors`
//! (the NVFP4 release one per layer). Without the switch, or a checkpoint
//! without them, nothing changes.

use super::*;
use anyhow::{bail, ensure};

/// `ATLAS_GLM_FP4L_INPUT_SCALES` is `1`. Read once.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_FP4L_INPUT_SCALES").as_deref() == Ok("1"))
}

/// `ATLAS_GLM_FP4L_DECODE_A16` is `1`: every routed FFN row that is not a
/// prompt row (decode, DFlash verify and owner-batched verify rows, inside a
/// [`DecodeRows`] scope) runs W4A16, BF16 activations against the NVFP4
/// weights, as the reference does (vLLM b12x with `B12X_W4A16_A4_PREFILL=1`:
/// the scheduler's prompt ranges A4, every other row A16). Read once.
pub(crate) fn decode_a16_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_FP4L_DECODE_A16").as_deref() == Ok("1"))
}

/// The routed lane of the batch in progress on this thread
/// ([`A16Scope`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Activations {
    /// NVFP4 activations (prompt rows, or the switch off).
    A4,
    /// W4A16 on the stream decode twins' `_a16` kernels: the prequant path's
    /// routing, tiles and schedule with BF16 activations
    /// (glm_moe_decode_a16.cuh).
    A16Stream,
    /// W4A16 on the grouped BF16-activation kernels (`prequant_moe` false):
    /// non-prompt batches the stream twins do not take.
    A16Grouped,
}

thread_local! {
    static DECODE_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LANE: std::cell::Cell<Activations> = const { std::cell::Cell::new(Activations::A4) };
}

/// While held, this thread's layers serve decode or verify rows: the GLM
/// layers' decode entry points (`decode`, `decode_batched`,
/// `decode_multi_seq`, `decode_glm_long_owners`) hold one, prompt prefill
/// does not. Every rank runs the same entry for a step, so every rank picks
/// the same lane.
pub(crate) struct DecodeRows(bool);

impl DecodeRows {
    pub(crate) fn enter() -> Self {
        Self(DECODE_ROWS.with(|c| c.replace(true)))
    }
}

impl Drop for DecodeRows {
    fn drop(&mut self) {
        DECODE_ROWS.with(|c| c.set(self.0));
    }
}

thread_local! {
    static PROMPT_ROWS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// While held, this thread's layers serve a prompt chunk of `prompt` rows
/// followed by verify rows (`prefill_with_glm_passengers`): under
/// [`decode_a16_requested`] the routed FFN runs the chunk's rows A4 and the
/// rest A16 ([`MoeLayer::forward_prefill_mixed`]), as the reference splits a
/// forward at the scheduler's prompt ranges.
pub(crate) struct PromptRows(Option<usize>);

impl PromptRows {
    pub(crate) fn enter(prompt: usize) -> Self {
        Self(PROMPT_ROWS.with(|c| c.replace(Some(prompt))))
    }

    /// The prompt rows of the mixed batch in progress, taken: the routed FFN
    /// it splits runs both halves without it.
    pub(super) fn take() -> Option<usize> {
        PROMPT_ROWS.with(|c| c.take())
    }
}

impl Drop for PromptRows {
    fn drop(&mut self) {
        PROMPT_ROWS.with(|c| c.set(self.0));
    }
}

/// The lane of one routed FFN, held while it runs.
pub(super) struct A16Scope(Activations);

impl A16Scope {
    /// A4 unless [`decode_a16_requested`] and a [`DecodeRows`] scope holds;
    /// then the stream twins' W4A16 kernels when `stream` (they take the
    /// batch), else the grouped W4A16 path.
    pub(super) fn enter(stream: bool) -> Self {
        let lane = if decode_a16_requested() && DECODE_ROWS.with(|c| c.get()) {
            if stream {
                Activations::A16Stream
            } else {
                Activations::A16Grouped
            }
        } else {
            Activations::A4
        };
        Self(LANE.with(|c| c.replace(lane)))
    }

    /// The lane of the routed FFN in progress.
    pub(super) fn lane() -> Activations {
        LANE.with(|c| c.get())
    }
}

impl Drop for A16Scope {
    fn drop(&mut self) {
        LANE.with(|c| c.set(self.0));
    }
}

/// The calibrated activation scales of one routed FFN and the GEMM tables
/// that undo them.
pub(crate) struct InputScales {
    /// gate / up / down tables of `gate_ptrs_t` / `up_ptrs_t` / `down_ptrs_t`
    /// with the activation scale folded into `scale2_vals`.
    pub(super) gate: ExpertPtrTable,
    pub(super) up: ExpertPtrTable,
    pub(super) down: ExpertPtrTable,
    /// The gate/up input's `scale2` (`max_e input_scale`), `1 / g1`.
    pub(super) a1: f32,
    /// `[E]` f32 `g2[e] = 1 / down input_scale[e]`.
    pub(super) a2_gscale: DevicePtr,
    /// `silu_mul_quant_nvfp4_by_expert`, for the unfused SiLU quantization.
    pub(super) silu_by_expert: KernelHandle,
}

fn read_f32s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn upload_f32s(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let p = gpu.alloc(v.len() * 4)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// The positive finite input scale of `w`, if it ships one.
fn input_scale(gpu: &dyn GpuBackend, w: &QuantizedWeight) -> Result<Option<f32>> {
    if w.input_scale.is_null() {
        return Ok(None);
    }
    let s = read_f32s(gpu, w.input_scale, 1)?[0];
    ensure!(
        s.is_finite() && s > 0.0,
        "routed expert input_scale {s} is not positive"
    );
    Ok(Some(s))
}

impl MoeLayer {
    /// Under the switch, the layer's [`InputScales`] from its experts' input
    /// scales and transposed tables. Every local expert must ship gate, up
    /// and down scales; a layer whose experts ship none keeps global scale 1.
    pub(super) fn build_input_scales(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !requested() {
            return Ok(());
        }
        let (Some(gate_t), Some(up_t), Some(down_t)) =
            (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t)
        else {
            bail!("ATLAS_GLM_FP4L_INPUT_SCALES: the routed tables are not built");
        };
        let n = self.weights.experts.len();
        let (mut a1, mut a2) = (0.0f32, Vec::with_capacity(n));
        let mut present = 0usize;
        let require_calibration = crate::layers::glm_native::mode::selected()?.is_some();
        for e in &self.weights.experts {
            crate::layers::glm_native::mode::validate_calibration(
                require_calibration,
                [
                    !e.gate_proj.is_null(),
                    !e.up_proj.is_null(),
                    !e.down_proj.is_null(),
                ],
                [
                    !e.gate_proj.input_scale.is_null(),
                    !e.up_proj.input_scale.is_null(),
                    !e.down_proj.input_scale.is_null(),
                ],
            )?;
            let (g, u, d) = (
                input_scale(gpu, &e.gate_proj)?,
                input_scale(gpu, &e.up_proj)?,
                input_scale(gpu, &e.down_proj)?,
            );
            match (g, u, d) {
                (Some(g), Some(u), Some(d)) => {
                    a1 = a1.max(g).max(u);
                    a2.push(d);
                    present += 1;
                }
                // A non-local expert (expert parallel): no rows reach it.
                (None, None, None) => a2.push(1.0),
                _ => bail!("ATLAS_GLM_FP4L_INPUT_SCALES: an expert ships only some input scales"),
            }
        }
        if present == 0 {
            ensure!(
                !require_calibration,
                "GLM serving mode found no calibrated local experts"
            );
            if gpu.op_cache().once("moe:input_scales_none") {
                tracing::warn!(
                    "ATLAS_GLM_FP4L_INPUT_SCALES=1 ignored: the routed experts ship no input_scale"
                );
            }
            return Ok(());
        }
        let fold = |t: &ExpertPtrTable, by: &dyn Fn(usize) -> f32| -> Result<ExpertPtrTable> {
            let s2 = read_f32s(gpu, t.scale2_vals, n)?;
            let folded: Vec<f32> = s2.iter().enumerate().map(|(e, &v)| v * by(e)).collect();
            Ok(t.with_scale2(upload_f32s(gpu, &folded)?))
        };
        let gate = fold(gate_t, &|_| a1)?;
        let up = fold(up_t, &|_| a1)?;
        let down = fold(down_t, &|e| a2[e])?;
        let g2: Vec<f32> = a2.iter().map(|&s| 1.0 / s).collect();
        let a2_gscale = upload_f32s(gpu, &g2)?;
        let silu_by_expert =
            super::super::try_kernel(gpu, "moe_silu_mul", "silu_mul_quant_nvfp4_by_expert");
        ensure!(
            silu_by_expert.0 != 0,
            "ATLAS_GLM_FP4L_INPUT_SCALES: the target lacks silu_mul_quant_nvfp4_by_expert"
        );
        if gpu.op_cache().once("moe:input_scales") {
            tracing::info!(
                "ATLAS_GLM_FP4L_INPUT_SCALES: routed NVFP4 activations take the checkpoint's input scales (gate/up max {a1:.3e}, {present} experts)"
            );
        }
        self.input_scales = Some(InputScales {
            gate,
            up,
            down,
            a1,
            a2_gscale,
            silu_by_expert,
        });
        Ok(())
    }

    /// The tables a W4A4 launch over `table` reads, and whether they are the
    /// calibrated ones: `table` itself without the switch.
    pub(super) fn a4_table<'a>(&'a self, table: &'a ExpertPtrTable) -> Result<&'a ExpertPtrTable> {
        let Some(s) = &self.input_scales else {
            return Ok(table);
        };
        for t in [&s.gate, &s.up, &s.down] {
            if t.packed_ptrs == table.packed_ptrs {
                return Ok(t);
            }
        }
        bail!("ATLAS_GLM_FP4L_INPUT_SCALES: a W4A4 launch over an unscaled expert table")
    }

    /// A mixed batch's routed FFN ([`PromptRows`]): its verify rows under a
    /// [`DecodeRows`] scope first, their output set aside, then the prompt
    /// rows, then the verify rows' output back after the prompt rows' in
    /// `moe_output`. The routed, shared and router kernels are row-invariant,
    /// so each row's output is the one its half alone gives. False (nothing
    /// run) unless [`decode_a16_requested`] and the batch has verify rows.
    pub(super) fn forward_prefill_mixed(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
        defer_shared_hc: bool,
    ) -> Result<bool> {
        let Some(prompt) = PromptRows::take().filter(|&p| p < num_tokens) else {
            return Ok(false);
        };
        if !decode_a16_requested() {
            return Ok(false);
        }
        ensure!(
            !defer_shared_hc,
            "ATLAS_GLM_FP4L_DECODE_A16: a mixed batch cannot defer its shared expert blend"
        );
        let row = ctx.config.hidden_size * 2;
        let rest = num_tokens - prompt;
        let aside = mixed_aside(ctx.gpu, rest * row, row)?;
        {
            let _rows = DecodeRows::enter();
            self.forward_prefill_mode(input.offset(prompt * row), rest, ctx, stream, false)?;
        }
        let out = ctx.buffers.moe_output();
        ctx.gpu.copy_d2d_async(out, aside, rest * row, stream)?;
        self.forward_prefill_mode(input, prompt, ctx, stream, false)?;
        ctx.gpu
            .copy_d2d_async(aside, out.offset(prompt * row), rest * row, stream)?;
        Ok(true)
    }

    /// The prequant routed path serves this batch: `ATLAS_NVFP4_PREQUANT_MOE`,
    /// unless its [`A16Scope`] took the grouped W4A16 path.
    pub(super) fn prequant_moe(&self) -> bool {
        self.nvfp4_prequant_moe && A16Scope::lane() != Activations::A16Grouped
    }

    /// Whether this batch runs the stream twins' W4A16 kernels.
    pub(super) fn a16_stream(&self) -> bool {
        A16Scope::lane() == Activations::A16Stream
    }

    /// The gate/up input quantizer's `scale2` (1 without the switch).
    pub(super) fn a1_scale2(&self) -> f32 {
        self.input_scales.as_ref().map_or(1.0, |s| s.a1)
    }

    /// The down input quantizer's per-expert global scales (null without).
    pub(super) fn a2_gscale(&self) -> DevicePtr {
        self.input_scales
            .as_ref()
            .map_or(DevicePtr::NULL, |s| s.a2_gscale)
    }
}

/// The verify rows' routed output of a mixed batch while its prompt rows
/// run: one buffer for [`crate::layer::glm_long_owner::MAX_ROWS`] rows,
/// allocated on the first mixed batch.
fn mixed_aside(gpu: &dyn GpuBackend, bytes: usize, row: usize) -> Result<DevicePtr> {
    static ASIDE: std::sync::Mutex<Option<(DevicePtr, usize)>> = std::sync::Mutex::new(None);
    let mut aside = ASIDE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, cap)) = *aside
        && cap >= bytes
    {
        return Ok(p);
    }
    let cap = bytes.max(crate::layer::glm_long_owner::MAX_ROWS * row);
    let p = gpu.alloc(cap)?;
    *aside = Some((p, cap));
    Ok(p)
}
