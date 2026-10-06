// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_FAST=1`: Qwen3.8-Flash-Next's expert-sorted row grid
//! for the originals-layout MoE decode kernels
//! (`kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_rows.cu`).
//!
//! One plan launch sorts the rows' `(row, slot)` entries by expert, then ONE
//! gate/up and ONE silu/down launch run, per entry, the CTA body of
//! `moe_expert_{gate_up,silu_down}_shared` — so every output byte is what
//! one launch per row of those kernels writes — in an order that puts every
//! row's pick of one expert, and every row's shared expert, back to back:
//! the union of the rows' experts streams from DRAM once.
//! `scripts/dev/qwen4exp_batch_exact_bench.cu` checks every output byte.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// Entries one plan launch sorts (its block is 128 threads; any count works,
/// this only bounds the per-thread rank loop).
pub const QWEN4EXP_MOE_ROWS_MAX_SLOTS: usize = 1024;

/// The three kernel handles, all 0 unless the exact batching lane is on for a
/// qwen4_exp model and the target ships the module.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpMoeRows {
    pub plan: KernelHandle,
    pub gate_up: KernelHandle,
    pub silu_down: KernelHandle,
}

impl Qwen4ExpMoeRows {
    pub const OFF: Self = Self {
        plan: KernelHandle(0),
        gate_up: KernelHandle(0),
        silu_down: KernelHandle(0),
    };

    pub fn resolve(gpu: &dyn GpuBackend, config: &ModelConfig) -> Self {
        // Only looked up when wanted, so no other model or default boot asks.
        if !(crate::model::qwen4exp_batch_fast::requested()
            && config.model_type == "qwen4_exp"
            && !config.expert_tp)
        {
            return Self::OFF;
        }
        let m = "qwen4exp_moe_rows";
        Self {
            plan: try_kernel(gpu, m, "qwen4exp_moe_rows_plan"),
            gate_up: try_kernel(gpu, m, "qwen4exp_moe_rows_gate_up"),
            silu_down: try_kernel(gpu, m, "qwen4exp_moe_rows_silu_down"),
        }
    }

    pub fn ready(&self) -> bool {
        self.plan.0 != 0 && self.gate_up.0 != 0 && self.silu_down.0 != 0
    }

    /// `order[rank] = q` for the `slots` entries of `expert_indices`, ranked
    /// by `(expert id, q)`.
    pub fn plan(
        &self,
        gpu: &dyn GpuBackend,
        expert_indices: DevicePtr,
        order: DevicePtr,
        slots: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=QWEN4EXP_MOE_ROWS_MAX_SLOTS).contains(&slots),
            "qwen4exp MoE rows plan: {slots} entries"
        );
        KernelLaunch::new(gpu, self.plan)
            .grid([1, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(expert_indices)
            .arg_ptr(order)
            .arg_u32(slots as u32)
            .launch(stream)
    }

    /// `moe_expert_gate_up_shared` for `rows` rows of `input` (`[rows, k]`)
    /// in one launch: `gate_out`/`up_out` `[rows * top_k, n]` (row `r`'s
    /// `[top_k, n]` block at `r * top_k`), `sh_*_out` `[rows, n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        gate_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        gate_out: DevicePtr,
        up_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        up_out: DevicePtr,
        expert_indices: DevicePtr,
        order: DevicePtr,
        sh_gate: &QuantizedWeight,
        sh_gate_out: DevicePtr,
        sh_up: &QuantizedWeight,
        sh_up_out: DevicePtr,
        (n, k, top_k, rows): (u32, u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.gate_up)
            .grid([div_ceil(n, 8), rows * top_k + rows, 2])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(gate_ptrs.0)
            .arg_ptr(gate_ptrs.1)
            .arg_ptr(gate_ptrs.2)
            .arg_ptr(gate_out)
            .arg_ptr(up_ptrs.0)
            .arg_ptr(up_ptrs.1)
            .arg_ptr(up_ptrs.2)
            .arg_ptr(up_out)
            .arg_ptr(expert_indices)
            .arg_ptr(order)
            .arg_ptr(sh_gate.weight)
            .arg_ptr(sh_gate.weight_scale)
            .arg_f32(sh_gate.weight_scale_2)
            .arg_ptr(sh_gate_out)
            .arg_ptr(sh_up.weight)
            .arg_ptr(sh_up.weight_scale)
            .arg_f32(sh_up.weight_scale_2)
            .arg_ptr(sh_up_out)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }

    /// `moe_expert_silu_down_shared` for `rows` rows in one launch: `output`
    /// `[rows * top_k, n]`, `sh_down_out` `[rows, n]`, reading the gate/up
    /// rows [`Self::gate_up`] wrote.
    #[allow(clippy::too_many_arguments)]
    pub fn silu_down(
        &self,
        gpu: &dyn GpuBackend,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_ptrs: (DevicePtr, DevicePtr, DevicePtr),
        output: DevicePtr,
        expert_indices: DevicePtr,
        order: DevicePtr,
        sh_gate_in: DevicePtr,
        sh_up_in: DevicePtr,
        sh_down: &QuantizedWeight,
        sh_down_out: DevicePtr,
        (n, k, top_k, rows): (u32, u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.silu_down)
            .grid([div_ceil(n, 8), rows * top_k + rows, 1])
            .block([128, 1, 1])
            .shared_mem(k * 4)
            .arg_ptr(gate_out)
            .arg_ptr(up_out)
            .arg_ptr(down_ptrs.0)
            .arg_ptr(down_ptrs.1)
            .arg_ptr(down_ptrs.2)
            .arg_ptr(output)
            .arg_ptr(expert_indices)
            .arg_ptr(order)
            .arg_ptr(sh_gate_in)
            .arg_ptr(sh_up_in)
            .arg_ptr(sh_down.weight)
            .arg_ptr(sh_down.weight_scale)
            .arg_f32(sh_down.weight_scale_2)
            .arg_ptr(sh_down_out)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }
}
