// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MOE=1`: the routed-expert prefill GEMMs of
//! Qwen3.8-Flash-Next on `moe_prefill_q38.cu` -- bit-identical to the default
//! `moe_w4a16_fused_gate_up_t_k64` -> `moe_silu_mul` ->
//! `moe_w4a16_grouped_gemm_ptrtable_t_k64` chain this target runs, in three
//! launches:
//!
//! 1. `moe_q38_a_to_e4m3`: the token-major input to E4M3 once (the default
//!    converts it inside the K loop of each of its ten column tiles);
//! 2. `moe_q38_gate_up_silu`: gate + up + SiLU*mul, writing the activation
//!    as E4M3 (the default's down converts its BF16 to exactly that);
//! 3. `moe_q38_down`: the down projection into `expert_down_out`.
//!
//! GB10, 256 local experts, 16000-token chunk (~80K routed rows): 24.5 ->
//! 16.5 ms a layer (`scripts/dev/qwen4exp_moe_prefill_bench.cu`, which checks
//! every byte of the activation's E4M3 image and of the down output).
//!
//! It takes only the shape that chain serves: NVFP4 transposed tables, SiLU,
//! no LoRA on the experts, none of the alternative gate/up or down arms
//! selected. Remote experts (EP) have null tables and return early, as in the
//! default kernels, so `prepare_ep_prefill_outputs`' zeroing still stands.
//!
//! `ATLAS_QWEN4EXP_PREFILL_MOE_CHECK=<n>`: for the first `n` layer calls, run
//! the q38 chain, keep its `expert_down_out` on the host, run the default
//! chain over the same buffers, and fail on any differing byte
//! (synchronizing, ~0.8 GB of host memory at a 16K chunk; diagnostic only).

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_QWEN4EXP_PREFILL_MOE=1`.
pub(crate) fn q38_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::forward_prefill_routed::env_flag("ATLAS_QWEN4EXP_PREFILL_MOE"))
}

/// Layer calls left to cross-check (`1` or `true` means 16).
fn check_left() -> &'static std::sync::atomic::AtomicI64 {
    static LEFT: std::sync::OnceLock<std::sync::atomic::AtomicI64> = std::sync::OnceLock::new();
    LEFT.get_or_init(|| {
        let n = match std::env::var("ATLAS_QWEN4EXP_PREFILL_MOE_CHECK").as_deref() {
            Ok("1") | Ok("true") => 16,
            Ok(v) => v.parse().unwrap_or(0),
            Err(_) => 0,
        };
        std::sync::atomic::AtomicI64::new(n)
    })
}

/// The q38 chain's `expert_down_out`, held for [`finish_q38_check`].
static PENDING: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

/// After the default chain ran a checked call: require its `expert_down_out`
/// to equal the q38 chain's, byte for byte.
pub(super) fn finish_q38_check(ctx: &ForwardContext, rows: u32, h: u32, stream: u64) -> Result<()> {
    let Some(fast) = PENDING
        .lock()
        .map_err(|_| anyhow::anyhow!("q38 check lock"))?
        .take()
    else {
        return Ok(());
    };
    ctx.gpu.synchronize(stream)?;
    let mut reference = vec![0u8; rows as usize * h as usize * 2];
    ctx.gpu
        .copy_d2h(ctx.buffers.expert_down_out(), &mut reference)?;
    let diff = reference.iter().zip(&fast).filter(|(a, b)| a != b).count();
    anyhow::ensure!(
        diff == 0,
        "ATLAS_QWEN4EXP_PREFILL_MOE_CHECK: the q38 MoE prefill differs from the default \
         in {diff}/{} expert_down_out bytes ({rows} routed rows)",
        reference.len()
    );
    tracing::info!("ATLAS_QWEN4EXP_PREFILL_MOE_CHECK: {rows} routed rows byte-identical");
    Ok(())
}

impl MoeLayer {
    /// Run the routed gate/up, SiLU and down of a prefill chunk on the q38
    /// kernels when they serve it; `Ok(false)` launched nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        rows_per_expert_grid: u32,
        total_expanded: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (Some(gp), Some(up), Some(dp)) =
            (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t)
        else {
            return Ok(false);
        };
        let fp8_down = std::env::var("ATLAS_MOE_PREFILL_FP8_DOWN").ok().as_deref() == Some("1");
        let serves = q38_requested()
            && ctx.config.model_type == "qwen4_exp"
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.btile_storage.is_published()
            && !self.nvfp4_mmq_layout
            && !self.nvfp4_prequant_moe
            && !self.gateup_fp4
            && !self.down_fp4
            && !self.nvfp4_gate_up_m128
            && !self.nvfp4_down_m32
            && !self.gelu_activation
            && self.lora.is_none()
            && !fp8_down
            && !super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && h.is_multiple_of(128)
            && inter.is_multiple_of(64)
            && h.is_multiple_of(32)
            && inter.is_multiple_of(32);
        if !serves {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        let k_a8 = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_a_to_e4m3");
        let k_gu = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_gate_up_silu");
        let k_dn = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_down");
        if k_a8.0 == 0 || k_gu.0 == 0 || k_dn.0 == 0 {
            return Ok(false);
        }
        // Scratch: the E4M3 input in `expert_up_out` ([n, h] bytes, inside its
        // [n * top_k, inter] BF16) and the E4M3 activation in
        // `expert_gate_out` ([rows, inter] bytes, inside the same).
        let a8 = ctx.buffers.expert_up_out();
        let act8 = ctx.buffers.expert_gate_out();
        let cells = n * h;
        KernelLaunch::new(gpu, k_a8)
            .grid([(cells / 4).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(a8)
            .arg_u32(cells)
            .launch(stream)?;
        // The kernels stride over 128-row tiles, so any grid computes every
        // row; size it like the default's persistent grid (rows, not tiles).
        let grid_m = rows_per_expert_grid.div_ceil(128).max(1);
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / 64, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(a8)
            .arg_ptr(gp.packed_ptrs)
            .arg_ptr(gp.scale_ptrs)
            .arg_ptr(gp.scale2_vals)
            .arg_ptr(up.packed_ptrs)
            .arg_ptr(up.scale_ptrs)
            .arg_ptr(up.scale2_vals)
            .arg_ptr(act8)
            .arg_ptr(expert_offsets)
            .arg_ptr(sorted_token_ids)
            .arg_u32(num_experts)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / 128, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(act8)
            .arg_ptr(dp.packed_ptrs)
            .arg_ptr(dp.scale_ptrs)
            .arg_ptr(dp.scale2_vals)
            .arg_ptr(ctx.buffers.expert_down_out())
            .arg_ptr(expert_offsets)
            .arg_u32(num_experts)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
        if check_left().fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0 {
            // Keep this result and let the default chain run over the same
            // buffers; `finish_q38_check` compares once it has.
            gpu.synchronize(stream)?;
            let mut fast = vec![0u8; total_expanded as usize * h as usize * 2];
            gpu.copy_d2h(ctx.buffers.expert_down_out(), &mut fast)?;
            *PENDING
                .lock()
                .map_err(|_| anyhow::anyhow!("q38 check lock"))? = Some(fast);
            return Ok(false);
        }
        Ok(true)
    }
}
