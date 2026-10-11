// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::exl3_gate_perturbation::Exl3GatePerturbation;
impl NativeMoeFfn {
    pub(in crate::layers::glm5_exl3_ffn) fn new(
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        mut shared: NativeDenseFfn,
        mut tables: Vec<ExpertTable>,
        mut companions: Vec<Exl3DenseDeviceTensor>,
        capacity: usize,
        stream: u64,
        activation: Exl3FfnActivation,
        policy: RoutedPrecisionPolicy,
        scratch: Option<MoeScratchBorrow>,
    ) -> Result<Self> {
        let mut owned_scratch = None;
        let mut buffers = [DevicePtr::NULL; 11];
        let setup = (|| -> Result<_> {
            ensure!(
                tables.len() == 3 && companions.len() == 2,
                "EXL3 MoE requires three expert tables/router+bias"
            );
            ensure!(
                config.tp_world_size > 0
                    && config
                        .moe_intermediate_size
                        .is_multiple_of(config.tp_world_size),
                "EXL3 MoE intermediate size must divide TP world"
            );
            ensure!(
                config.scoring_func == "sigmoid"
                    && config.moe_expert_groups == Some(1)
                    && config.moe_topk_groups == Some(1),
                "EXL3 MoE requires explicit sigmoid/no-group routing"
            );
            ensure!(
                config.num_experts > 0
                    && config.num_experts <= 512
                    && config.num_experts_per_tok > 0
                    && config.num_experts_per_tok <= 32
                    && config.num_experts_per_tok <= config.num_experts,
                "EXL3 MoE expert/top-k counts invalid"
            );
            ensure!(
                (config.routed_scaling_factor as f32).is_finite()
                    && config.routed_scaling_factor > 0.0,
                "EXL3 MoE routed scale must be positive finite F32"
            );
            u32::try_from(config.hidden_size).context("EXL3 MoE hidden exceeds kernel ABI")?;
            u32::try_from(config.moe_intermediate_size)
                .context("EXL3 MoE intermediate exceeds kernel ABI")?;
            let intermediate = config.moe_intermediate_size / config.tp_world_size;
            let h = config.hidden_size;
            let experts = config.num_experts;
            let topk = config.num_experts_per_tok;
            ensure!(
                capacity > 0 && topk > 0 && capacity.checked_mul(topk).is_some_and(|n| n <= 65535),
                "EXL3 MoE capacity*topk must fit CUDA grid.y"
            );
            ensure!(
                h > 0
                    && intermediate > 0
                    && h.is_multiple_of(128)
                    && intermediate.is_multiple_of(128),
                "EXL3 MoE widths must be positive H128 multiples"
            );
            ensure!(
                shared.is_reconstructed(),
                "native MoE shared experts require global BF16 reconstruction"
            );
            shared.validate(capacity, stream, config.tp_rank, config.tp_world_size)?;
            ensure!(
                shared.hidden == h,
                "EXL3 shared/routed hidden size mismatch"
            );
            for (index, table) in tables.iter().enumerate() {
                let (k, n, axis) = if index == 2 {
                    (intermediate, h, ShardAxis::Input)
                } else {
                    (h, intermediate, ShardAxis::Output)
                };
                table.validate(experts, config.tp_rank, config.tp_world_size, k, n, axis)?;
            }
            ensure!(
                companions[0].dtype() == Exl3DenseDtype::Bf16
                    && companions[0].shape() == [experts, h]
                    && companions[1].dtype() == Exl3DenseDtype::F32
                    && companions[1].shape() == [experts],
                "EXL3 router/bias must have exact BF16/F32 device shapes"
            );
            for c in &companions {
                c.ptr()?;
            }
            activation.validate()?;
            policy.validate()?;
            gpu.require_bf16_gemm_numerics(spark_runtime::gpu::Bf16GemmNumerics::CublasLt)?;
            let kernels = [
                gpu.kernel("residual_add", "bf16_to_f32")?,
                gpu.kernel("exl3_moe", "exl3_topk_sigmoid")?,
                gpu.kernel("exl3_moe_precision", "exl3_moe_input_hadamard_f16")?,
                gpu.kernel("exl3_moe_mma", "exl3_moe_mul1_matmul_precision_mma")?,
                gpu.kernel("exl3_moe_precision", "exl3_moe_output_hadamard_precision")?,
                gpu.kernel("exl3_moe_precision", "exl3_moe_combine_f32")?,
                gpu.kernel("exl3_moe_precision", "exl3_moe_activate_down_hadamard")?,
                gpu.kernel("exl3_moe_precision", "exl3_moe_route_counts")?,
                gpu.kernel("exl3_moe_precision", "exl3_moe_down_weighted_output")?,
            ];
            let geometry = MoeScratchGeometry::new(config, capacity, stream)?;
            let borrowed = match &scratch {
                Some(borrowed) => borrowed.clone(),
                None => {
                    owned_scratch = Some(MoeScratch::allocate(gpu, geometry)?);
                    owned_scratch.as_ref().expect("scratch allocated").borrow()
                }
            };
            borrowed.validate(geometry, capacity, stream)?;
            buffers = borrowed.pointers()?;
            Ok((kernels, intermediate))
        })();
        match setup {
            Ok((kernels, intermediate)) => Ok(Self {
                layer: None,
                capture: None,
                shared,
                tables,
                companions,
                buffers,
                scratch: scratch
                    .unwrap_or_else(|| owned_scratch.as_ref().expect("scratch owner").borrow()),
                owned_scratch,
                kernels,
                activation,
                precision: policy,
                hidden: config.hidden_size,
                intermediate,
                experts: config.num_experts,
                topk: config.num_experts_per_tok,
                normalize: config.norm_topk_prob,
                scale: config.routed_scaling_factor as f32,
                rank: config.tp_rank,
                world: config.tp_world_size,
                capacity,
                stream,
                active: true,
                drop_shared: false,
                round_routed_accum: false,
            }),
            Err(error) => {
                let mut cleanup = None;
                retain_error(&mut cleanup, shared.release(gpu));
                for t in &mut tables {
                    retain_error(&mut cleanup, t.release(gpu));
                }
                for c in &mut companions {
                    retain_error(&mut cleanup, c.release(gpu));
                }
                if let Some(owner) = &mut owned_scratch {
                    retain_error(&mut cleanup, owner.release(gpu));
                }
                Err(match cleanup {
                    None => error,
                    Some(e) => error.context(format!("EXL3 MoE cleanup failed: {e:#}")),
                })
            }
        }
    }
}
impl NativeMoeFfn {
    /// Apply a validated statistical-gate control to this routed layer.
    /// Layer-conditional controls only touch the named layer.
    pub(in crate::layers::glm5_exl3_ffn) fn apply_perturbation(
        &mut self,
        layer: usize,
        perturbation: Option<Exl3GatePerturbation>,
    ) -> Result<()> {
        use crate::exl3_gate_perturbation as gate;
        let topk = gate::layer_topk(self.topk, layer, perturbation);
        // Every routed kernel takes top-k as a runtime argument and indexes
        // compact [rows, topk] slots; scratch is sized for the configured
        // top-k, so only a reduction is admissible.
        ensure!(
            topk > 0 && topk <= self.topk,
            "EXL3 gate perturbation may only reduce routed top-k"
        );
        self.topk = topk;
        self.scale = gate::routed_scale(self.scale, perturbation);
        self.drop_shared = gate::drop_shared(layer, perturbation);
        self.round_routed_accum = gate::bf16_routed_accum(perturbation);
        Ok(())
    }
    /// `exl3_moe_combine_f32` (has_shared, round_accum). A dropped shared
    /// expert leaves buffers[6] stale, so it must never be added.
    pub(super) fn combine_flags(&self) -> [u32; 2] {
        [
            u32::from(!self.drop_shared),
            u32::from(self.round_routed_accum),
        ]
    }
}
