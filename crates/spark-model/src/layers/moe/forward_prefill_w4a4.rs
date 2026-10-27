// SPDX-License-Identifier: AGPL-3.0-only

//! qwen4_exp's routed-MoE prefill (`ATLAS_QWEN4EXP_PREFILL_MOE=1`) at the
//! reference's numerics: prompt rows on NVFP4 activations x the
//! checkpoint's NVFP4 weights, both as stored, on native FP4 tensor cores
//! (`kernels/gb10/qwen3.8-flash-next/nvfp4/moe_prefill_w4a4.cu`), the
//! activations quantized under the checkpoint's input scales
//! (`w4a4_scales.rs`) exactly as NVIDIA's b12x A4 prefill does:
//!
//! 1. `moe_w4a4_quant_input`: the token-major input to NVFP4 once (g1);
//! 2. `moe_w4a4_gate_up_silu`: gate + up, `bf16(bf16(silu(g)) * u)`, the
//!    activation to NVFP4 under the expert's g2;
//! 3. `moe_w4a4_down`: into `expert_down_out`.
//!
//! Same buffers and row order as the chain it replaced (expert-sorted rows;
//! scratch in `expert_up_out` / `expert_gate_out`, the down rows in
//! `expert_down_out`), so the router, the local unpermute and the SP pipes
//! around it are unchanged. Remote experts (EP) have null tables and are
//! skipped. Row invariant: per-row quantization, one in-order accumulator
//! per output (`scripts/dev/qwen4exp_moe_w4a4_bench.sh check`).
//!
//! A layer without usable input scales (the MTP layer) runs the BF16 TC
//! kernels instead (`forward_prefill_tcp`: W4A16, the reference's arm for
//! non-calibrated layers), as does every layer under
//! `ATLAS_QWEN4EXP_PREFILL_MOE_BF16` / `_PREFILL_BF16_PROJ` (decode
//! numerics). Both read the checkpoint's N-major planes, so the startup
//! transpose pass builds no routed prefill copy ([`skips_routed_transpose`]):
//! 31.6 GiB a rank at TP=EP=2.
//!
//! GB10 (clock-capped as served), 256 local experts, 16000-token chunk
//! (81K local rows), one layer (`scripts/dev/qwen4exp_moe_w4a4_bench.sh
//! time`): W4A4 8.36 ms (quant 0.48, gate_up+SiLU 4.09, down 3.79) against
//! the E4M3 chain it replaced (q38 W2 on the N-major planes) 14.71 and the
//! BF16 TC chain 20.25.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

/// Whether the startup transpose pass leaves `config`'s routed experts in
/// their checkpoint layout: the q38 prefill arm (W4A4 or BF16 TC), which
/// reads them so, serves this model.
pub(crate) fn skips_routed_transpose(config: &atlas_core::config::ModelConfig) -> bool {
    super::q38_requested() && config.model_type == "qwen4_exp"
}

/// Rows of a CTA tile and the gate/up and down column tiles of the kernels.
const W4_BM: u32 = 128;
const W4_GU_COLS: u32 = 64;
const W4_DN_COLS: u32 = 128;

/// Byte offset of the scale plane after a `bytes`-byte packed plane.
fn scales_at(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

impl MoeLayer {
    /// The startup check behind the N-major routed arms: every local
    /// expert's packed planes 16-byte and scale planes 8-byte aligned (the
    /// kernels' cp.async sizes; the row strides keep them), NVFP4 scales.
    pub(super) fn check_routed_planes(&self) -> Result<()> {
        anyhow::ensure!(
            self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4,
            "qwen4_exp prefill MoE: routed experts are not NVFP4"
        );
        for (e, expert) in self.weights.experts.iter().enumerate() {
            if expert.gate_proj.is_null() {
                continue;
            }
            for w in [&expert.gate_proj, &expert.up_proj, &expert.down_proj] {
                anyhow::ensure!(
                    w.weight.0 % 16 == 0 && w.weight_scale.0 % 8 == 0,
                    "qwen4_exp prefill MoE: expert {e} planes misaligned \
                     (packed {:#x}, scales {:#x})",
                    w.weight.0,
                    w.weight_scale.0
                );
            }
        }
        Ok(())
    }

    /// The routed gate/up, SiLU and down of a prefill chunk on the W4A4
    /// kernels; `Ok(false)`, nothing launched, when the layer has no input
    /// scales. `[n, rows]` = the chunk's tokens and routed rows; `dims` =
    /// (hidden, inter, num_experts, rows a grid row covers).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_w4a4_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        [n, rows]: [u32; 2],
        (h, inter, num_experts, rows_per_expert_grid): (u32, u32, u32, u32),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(s) = &self.w4a4 else {
            return Ok(false);
        };
        let gpu = ctx.gpu;
        let kernel = |name| crate::layers::try_kernel(gpu, "moe_prefill_w4a4", name);
        let (k_q, k_gu, k_dn) = (
            kernel("moe_w4a4_quant_input"),
            kernel("moe_w4a4_gate_up_silu"),
            kernel("moe_w4a4_down"),
        );
        anyhow::ensure!(
            k_q.0 != 0 && k_gu.0 != 0 && k_dn.0 != 0,
            "qwen4_exp W4A4 prefill: the target lacks moe_prefill_w4a4"
        );
        anyhow::ensure!(
            h.is_multiple_of(128) && inter.is_multiple_of(128),
            "qwen4_exp W4A4 prefill: hidden {h} / inter {inter} not multiples of 128"
        );
        // Scratch: the NVFP4 input (+ scales) in `expert_up_out`, the NVFP4
        // activation (+ scales) in `expert_gate_out` -- each well inside its
        // [rows, inter] BF16.
        let a_bytes = n as usize * h as usize / 2;
        let act_bytes = rows as usize * inter as usize / 2;
        let sizes = ctx.buffers.sizes();
        anyhow::ensure!(
            sizes.expert_up_out >= scales_at(a_bytes) + a_bytes / 8
                && sizes.expert_gate_out >= scales_at(act_bytes) + act_bytes / 8,
            "qwen4_exp W4A4 prefill: scratch too small for {n} tokens / {rows} rows"
        );
        let (a4, act) = (ctx.buffers.expert_up_out(), ctx.buffers.expert_gate_out());
        let (a4s, acts) = (
            a4.offset(scales_at(a_bytes)),
            act.offset(scales_at(act_bytes)),
        );
        let groups = n * h / 16;
        KernelLaunch::new(gpu, k_q)
            .grid([groups.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(a4)
            .arg_ptr(a4s)
            .arg_f32(s.g1)
            .arg_u32(groups)
            .launch(stream)?;
        // The kernels stride over row tiles, so any grid computes every row.
        let grid_m = rows_per_expert_grid.div_ceil(W4_BM).max(1);
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / W4_GU_COLS, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(a4)
            .arg_ptr(a4s)
            .arg_ptr(s.gate.packed_ptrs)
            .arg_ptr(s.gate.scale_ptrs)
            .arg_ptr(s.gate.scale2_vals)
            .arg_ptr(s.up.packed_ptrs)
            .arg_ptr(s.up.scale_ptrs)
            .arg_ptr(s.up.scale2_vals)
            .arg_ptr(s.a2_gscale)
            .arg_ptr(act)
            .arg_ptr(acts)
            .arg_ptr(expert_offsets)
            .arg_ptr(sorted_token_ids)
            .arg_u32(num_experts)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / W4_DN_COLS, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(acts)
            .arg_ptr(s.down.packed_ptrs)
            .arg_ptr(s.down.scale_ptrs)
            .arg_ptr(s.down.scale2_vals)
            .arg_ptr(ctx.buffers.expert_down_out())
            .arg_ptr(expert_offsets)
            .arg_u32(num_experts)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::scales_at;

    #[test]
    fn scale_planes_start_256_byte_aligned_after_the_codes() {
        assert_eq!(scales_at(0), 0);
        assert_eq!(scales_at(1), 256);
        assert_eq!(scales_at(16000 * 1280), 16000 * 1280);
    }
}
