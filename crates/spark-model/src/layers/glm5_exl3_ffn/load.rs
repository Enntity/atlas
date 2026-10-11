// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::FfnComponent;
use crate::weight_loader::Exl3ProjectionSpec;
use spark_runtime::weights::exl3::dense::{
    Exl3CheckpointDtype, Exl3DenseDeviceTensor, Exl3DenseDtype,
};

pub fn load_glm5_exl3_ffn(
    checkpoint: &Exl3CheckpointRouter,
    config: &ModelConfig,
    layer: usize,
    gpu: &dyn GpuBackend,
    capacity: usize,
    stream: u64,
) -> Result<FfnComponent> {
    load_glm5_exl3_ffn_with_numerics(
        checkpoint,
        config,
        layer,
        gpu,
        capacity,
        stream,
        crate::weight_loader::glm53_tp2_packed_numerics,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn load_glm5_exl3_ffn_with_numerics(
    checkpoint: &Exl3CheckpointRouter,
    config: &ModelConfig,
    layer: usize,
    gpu: &dyn GpuBackend,
    capacity: usize,
    stream: u64,
    select: crate::weight_loader::glm5::exl3_numerics::Exl3NumericsSelector,
) -> Result<FfnComponent> {
    load_glm5_exl3_ffn_with_scratch(
        checkpoint, config, layer, gpu, capacity, stream, select, None, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn load_glm5_exl3_ffn_with_scratch(
    checkpoint: &Exl3CheckpointRouter,
    config: &ModelConfig,
    layer: usize,
    gpu: &dyn GpuBackend,
    capacity: usize,
    stream: u64,
    select: crate::weight_loader::glm5::exl3_numerics::Exl3NumericsSelector,
    scratch: Option<MoeScratchBorrow>,
    perturbation: Option<crate::exl3_gate_perturbation::Exl3GatePerturbation>,
) -> Result<FfnComponent> {
    ensure!(
        capacity > 0 && u32::try_from(capacity).is_ok(),
        "EXL3 FFN capacity must fit positive ABI"
    );
    ensure!(
        config.ep_world_size == 1 && config.ep_rank == 0 && !config.expert_tp,
        "native EXL3 FFN requires pure TP before GPU loading"
    );
    if let Some(borrowed) = &scratch {
        borrowed.validate(
            MoeScratchGeometry::new(config, capacity, stream)?,
            capacity,
            stream,
        )?;
    }
    let capture = diagnostics::Exl3FfnRouter::from_env(
        layer,
        config.num_hidden_layers,
        config.tp_rank,
        config.tp_world_size,
    )?;
    let p = format!("{}.layers.{layer}.mlp", config.weight_prefix);
    let activation = Exl3FfnActivation::from_glm_config(config)?;
    let dense_activation = NativeDenseActivation::from_text_policy(activation)?;
    if config.mlp_only_layers.contains(&layer) {
        let mut dense = load_dense(
            checkpoint,
            config,
            &p,
            gpu,
            capacity,
            stream,
            dense_activation,
            NativeDenseFfnRole::DensePacked,
            select,
        )?;
        dense.layer = Some(layer);
        dense.capture = capture;
        return Ok(FfnComponent::Exl3(Box::new(Glm5Exl3FfnLayer::Dense(
            Box::new(dense),
        ))));
    }
    ensure!(
        config.scoring_func == "sigmoid"
            && config.moe_expert_groups == Some(1)
            && config.moe_topk_groups == Some(1),
        "GLM EXL3 MoE requires explicit sigmoid/no-group router policy"
    );
    ensure!(
        config.num_experts > 0
            && config.num_experts <= 512
            && config.num_experts_per_tok > 0
            && config.num_experts_per_tok <= config.num_experts
            && config.num_experts_per_tok <= 32,
        "GLM EXL3 MoE expert/top-k count invalid"
    );
    ensure!(
        config.routed_scaling_factor.is_finite() && config.routed_scaling_factor > 0.0,
        "GLM EXL3 MoE routed scaling factor invalid"
    );
    ensure!(
        capacity
            .checked_mul(config.num_experts_per_tok)
            .is_some_and(|n| n <= 65535),
        "native EXL3 routed capacity must fit CUDA slot grid"
    );
    let shared = load_dense(
        checkpoint,
        config,
        &format!("{p}.shared_experts"),
        gpu,
        capacity,
        stream,
        dense_activation,
        NativeDenseFfnRole::SharedExperts,
        select,
    )?;
    let mut tables = Vec::new();
    let mut companions = Vec::new();
    let result = (|| -> Result<()> {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let mut shards = Vec::new();
            for expert in 0..config.num_experts {
                let name = format!("{p}.experts.{expert}.{projection}");
                let load = Exl3ProjectionSpec::from_rank_local_config(config, &name)
                    .and_then(|spec| spec.load(checkpoint, gpu));
                match load {
                    Ok(shard) => shards.push(shard),
                    Err(error) => {
                        let mut cleanup = None;
                        for shard in &mut shards {
                            retain_error(&mut cleanup, shard.release(gpu));
                        }
                        return Err(match cleanup {
                            None => error,
                            Some(e) => error.context(format!("EXL3 expert rollback failed: {e:#}")),
                        });
                    }
                }
            }
            tables.push(tables::ExpertTable::upload(gpu, shards)?);
        }
        for (name, dtype, shape) in [
            (
                "gate.weight",
                Exl3CheckpointDtype::F16,
                vec![config.num_experts, config.hidden_size],
            ),
            (
                "gate.e_score_correction_bias",
                Exl3CheckpointDtype::F32,
                vec![config.num_experts],
            ),
        ] {
            let view = checkpoint.dense_view(&format!("{p}.{name}"), dtype, &shape)?;
            companions.push(Exl3DenseDeviceTensor::upload(
                gpu,
                &view,
                if name == "gate.weight" {
                    Exl3DenseDtype::Bf16
                } else {
                    Exl3DenseDtype::F32
                },
                std::slice::from_ref(&(0..shape[0])),
            )?);
        }
        Ok(())
    })();
    if let Err(error) = result {
        let mut shared = shared;
        let mut cleanup = None;
        retain_error(&mut cleanup, shared.release(gpu));
        for table in &mut tables {
            retain_error(&mut cleanup, table.release(gpu));
        }
        for tensor in &mut companions {
            retain_error(&mut cleanup, tensor.release(gpu));
        }
        return Err(match cleanup {
            None => error,
            Some(e) => error.context(format!("EXL3 MoE rollback failed: {e:#}")),
        });
    }
    let mut moe = NativeMoeFfn::new(
        gpu,
        config,
        shared,
        tables,
        companions,
        capacity,
        stream,
        activation,
        super::moe::RoutedPrecisionPolicy::GLM53_TP2_REFERENCE,
        scratch,
    )?;
    moe.layer = Some(layer);
    moe.shared.layer = Some(layer);
    moe.capture = capture;
    if let Err(error) = moe.apply_perturbation(layer, perturbation) {
        let cleanup = moe.release(gpu);
        return Err(match cleanup {
            Ok(()) => error,
            Err(e) => error.context(format!("EXL3 MoE rollback failed: {e:#}")),
        });
    }
    Ok(FfnComponent::Exl3(Box::new(Glm5Exl3FfnLayer::Moe(
        Box::new(moe),
    ))))
}
#[allow(clippy::too_many_arguments)]
fn load_dense(
    checkpoint: &Exl3CheckpointRouter,
    config: &ModelConfig,
    prefix: &str,
    gpu: &dyn GpuBackend,
    capacity: usize,
    stream: u64,
    activation: NativeDenseActivation,
    role: NativeDenseFfnRole,
    select: crate::weight_loader::glm5::exl3_numerics::Exl3NumericsSelector,
) -> Result<NativeDenseFfn> {
    if role == NativeDenseFfnRole::SharedExperts {
        let weights =
            super::shared::SharedBf16Weights::load(checkpoint, prefix, config, gpu, stream)?;
        return NativeDenseFfn::from_shared(gpu, weights, capacity, stream, activation);
    }
    let mut shards = Vec::new();
    let mut policies = Vec::new();
    for name in ["gate_proj", "up_proj", "down_proj"] {
        let result =
            Exl3ProjectionSpec::from_rank_local_config(config, &format!("{prefix}.{name}"))
                .and_then(|spec| {
                    let view = checkpoint.linear_view(
                        &format!("{prefix}.{name}"),
                        spec.in_features(),
                        spec.out_features(),
                    )?;
                    let plan = spec.shard_plan(&view)?;
                    let policy = select(
                        plan.world(),
                        plan.in_features(),
                        plan.out_features(),
                        plan.bits(),
                        plan.out_features(),
                        0,
                    )?;
                    spark_runtime::weights::exl3::native::Exl3LinearWorkspace::preflight_numerics(
                        gpu,
                        plan.in_features(),
                        plan.out_features(),
                        capacity,
                        policy,
                    )?;
                    if name == "up_proj" {
                        ensure!(
                            policies.first() == Some(&policy),
                            "EXL3 shared gate/up workspace requires identical numerical profiles"
                        );
                    }
                    policies.push(policy);
                    Exl3DeviceShard::upload(gpu, &view, &plan)
                });
        match result {
            Ok(shard) => shards.push(shard),
            Err(error) => {
                let mut cleanup = None;
                for shard in &mut shards {
                    retain_error(&mut cleanup, shard.release(gpu));
                }
                return Err(match cleanup {
                    None => error,
                    Some(e) => error.context(format!("EXL3 shard rollback failed: {e:#}")),
                });
            }
        }
    }
    let shards = shards
        .try_into()
        .map_err(|_| anyhow::anyhow!("EXL3 FFN projection count mismatch"))?;
    NativeDenseFfn::new_with_numerics(
        gpu,
        shards,
        capacity,
        stream,
        activation,
        [policies[0], policies[2]],
    )
}
