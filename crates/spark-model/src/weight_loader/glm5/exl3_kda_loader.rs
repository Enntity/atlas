// SPDX-License-Identifier: AGPL-3.0-only

//! Native fused-checkpoint KDA loading. Owns every allocation until model
//! assembly explicitly transfers projection weights and registers companions.

use anyhow::{Context, Result, ensure};
use atlas_core::{config::ModelConfig, scope::ModelResource};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::{
    Exl3CheckpointRouter,
    dense::{Exl3CheckpointDtype as Dtype, Exl3DenseDeviceTensor, Exl3DenseDtype},
    native::Exl3LinearWorkspace,
    tp::{Exl3DeviceShard, TpShardPlan},
};

use super::{exl3::Exl3ProjectionSpec, exl3_projection::Glm5Exl3Projection};
use crate::layers::{Glm5KdaWeights, Glm5Projection};
use crate::weight_map::DenseWeight;

/// Stable companion names, independent of their checkpoint/device dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3KdaDenseSlot {
    BProj,
    FAProj,
    FBProj,
    GAProj,
    GBProj,
    Conv,
    ALog,
    DtBias,
    ONorm,
    /// Owned `[B; FA; GA]` rows; the three source slots are retired.
    AuxMerged,
}

/// Owns converted/sharded dense auxiliaries alongside plain dense handles.
pub struct Glm5Exl3KdaDenseResources {
    tensors: Vec<(Exl3KdaDenseSlot, Exl3DenseDeviceTensor)>,
}
impl Glm5Exl3KdaDenseResources {
    pub fn into_tensors(self) -> Vec<Exl3DenseDeviceTensor> {
        self.tensors.into_iter().map(|(_, tensor)| tensor).collect()
    }

    pub fn tensor(&self, slot: Exl3KdaDenseSlot) -> Result<&Exl3DenseDeviceTensor> {
        self.tensors
            .iter()
            .find(|(key, _)| *key == slot)
            .map(|(_, tensor)| tensor)
            .context("EXL3 KDA companion missing")
    }
    pub fn weight(&self, slot: Exl3KdaDenseSlot) -> Result<DenseWeight> {
        if matches!(
            slot,
            Exl3KdaDenseSlot::BProj | Exl3KdaDenseSlot::FAProj | Exl3KdaDenseSlot::GAProj
        ) {
            return self.merged_view(slot);
        }
        Ok(DenseWeight {
            weight: self.tensor(slot)?.ptr()?,
        })
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3KdaDenseResources {
    fn label(&self) -> &'static str {
        "GLM EXL3 KDA dense companions"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        release_dense(&mut self.tensors, gpu)
    }
}

/// Separate Q/K/V owners slice the fused EXL3 matrix without re-encoding.
/// Caller must register this bundle before transferring any resources.
pub struct Glm5Exl3KdaResources {
    pub q_proj: Glm5Exl3Projection,
    pub k_proj: Glm5Exl3Projection,
    pub v_proj: Glm5Exl3Projection,
    pub o_proj: Glm5Exl3Projection,
    pub dense: Glm5Exl3KdaDenseResources,
}
impl Glm5Exl3KdaResources {
    /// Transfer native projections into KDA's real forward weights. The
    /// returned sidecar owns all plain dense handles and must be registered
    /// for teardown alongside the layer's EXL3 projection resources.
    pub fn into_weights(
        mut self,
        gpu: &dyn GpuBackend,
    ) -> Result<(Glm5KdaWeights, Glm5Exl3KdaDenseResources)> {
        let prepared = (|| -> Result<_> {
            for projection in [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj] {
                projection.validate_batch(0)?;
            }
            let weight = |slot| self.dense.weight(slot);
            Ok((
                weight(Exl3KdaDenseSlot::BProj)?,
                weight(Exl3KdaDenseSlot::FAProj)?,
                weight(Exl3KdaDenseSlot::FBProj)?,
                weight(Exl3KdaDenseSlot::GAProj)?,
                weight(Exl3KdaDenseSlot::GBProj)?,
                weight(Exl3KdaDenseSlot::Conv)?,
                weight(Exl3KdaDenseSlot::ALog)?,
                weight(Exl3KdaDenseSlot::DtBias)?,
                weight(Exl3KdaDenseSlot::ONorm)?,
            ))
        })();
        let (b_proj, f_a_proj, f_b_proj, g_a_proj, g_b_proj, conv, a_log, dt_bias, o_norm) =
            match prepared {
                Ok(weights) => weights,
                Err(error) => {
                    return Err(match self.release(gpu) {
                        Ok(()) => error,
                        Err(cleanup) => error.context(format!(
                            "EXL3 KDA transfer cleanup also failed: {cleanup:#}"
                        )),
                    });
                }
            };
        Ok((
            Glm5KdaWeights {
                q_proj: Glm5Projection::Exl3(self.q_proj),
                k_proj: Glm5Projection::Exl3(self.k_proj),
                v_proj: Glm5Projection::Exl3(self.v_proj),
                o_proj: Glm5Projection::Exl3(self.o_proj),
                b_proj,
                f_a_proj,
                f_b_proj,
                g_a_proj,
                g_b_proj,
                conv,
                a_log,
                dt_bias,
                o_norm,
            },
            self.dense,
        ))
    }

    pub fn load(
        checkpoint: &Exl3CheckpointRouter,
        config: &ModelConfig,
        layer: usize,
        gpu: &dyn GpuBackend,
        capacity: usize,
        stream: u64,
    ) -> Result<Self> {
        Self::load_with_numerics(
            checkpoint,
            config,
            layer,
            gpu,
            capacity,
            stream,
            super::exl3_numerics::glm53_tp2_packed_numerics,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with_numerics(
        checkpoint: &Exl3CheckpointRouter,
        config: &ModelConfig,
        layer: usize,
        gpu: &dyn GpuBackend,
        capacity: usize,
        stream: u64,
        select: super::exl3_numerics::Exl3NumericsSelector,
    ) -> Result<Self> {
        let p = format!("{}.layers.{layer}.self_attn", config.weight_prefix);
        let spec = Exl3ProjectionSpec::from_rank_local_config(config, &format!("{p}.qkv_proj"))?;
        let segments = spec.output_segments();
        ensure!(
            segments.len() == 3 && segments[0] == segments[1] && segments[0] == segments[2],
            "EXL3 GLM KDA requires matching Q/K/V widths"
        );
        ensure!(
            config.linear_num_key_heads == config.linear_num_value_heads
                && config.linear_key_head_dim == config.linear_value_head_dim,
            "EXL3 GLM KDA requires matching key/value head layout"
        );
        ensure!(
            config.linear_conv_kernel_dim > 0,
            "EXL3 KDA requires convolution width"
        );
        let view = checkpoint.linear_view(
            &format!("{p}.qkv_proj"),
            spec.in_features(),
            spec.out_features(),
        )?;
        let mut projections = Vec::new();
        let mut dense = Vec::new();
        let result = (|| -> Result<()> {
            let mut ranges = Vec::new();
            for component in 0..3 {
                let plan = TpShardPlan::new_output_component(
                    &view,
                    segments,
                    component,
                    config.tp_rank,
                    config.tp_world_size,
                )?;
                ranges.push(plan.output_ranges()[0].clone());
                let logical_n = segments.iter().sum::<usize>() / config.tp_world_size;
                let offset = segments[..component].iter().sum::<usize>() / config.tp_world_size;
                let numerics = select(
                    config.tp_world_size,
                    plan.in_features(),
                    plan.out_features(),
                    plan.bits(),
                    logical_n,
                    offset,
                )?;
                Exl3LinearWorkspace::preflight_numerics(
                    gpu,
                    plan.in_features(),
                    plan.out_features(),
                    capacity,
                    numerics,
                )?;
                let shard = Exl3DeviceShard::upload(gpu, &view, &plan)?;
                projections.push(Glm5Exl3Projection::new_with_numerics(
                    gpu, shard, capacity, stream, numerics,
                )?);
            }
            let ospec = Exl3ProjectionSpec::from_rank_local_config(config, &format!("{p}.o_proj"))?;
            projections.push(ospec.load_projection(checkpoint, gpu, capacity, stream, select)?);
            let key = segments[0];
            let kd = config.linear_key_head_dim;
            let h = config.hidden_size;
            let heads = key / kd;
            let local_heads = config.linear_num_key_heads;
            let head_range = config.tp_rank * local_heads..(config.tp_rank + 1) * local_heads;
            let key_range = ranges[0].clone();
            let upload = |name: &str,
                          dtype: Dtype,
                          shape: Vec<usize>,
                          rows: Vec<std::ops::Range<usize>>,
                          output: Exl3DenseDtype|
             -> Result<Exl3DenseDeviceTensor> {
                let tensor = checkpoint.dense_view(&format!("{p}.{name}"), dtype, &shape)?;
                Exl3DenseDeviceTensor::upload(gpu, &tensor, output, &rows)
                    .with_context(|| format!("EXL3 GLM KDA {name} rank {}", config.tp_rank))
            };
            // Each slot is explicitly associated with its checkpoint tensor.
            for (slot, name, dtype, shape, rows, output) in [
                (
                    Exl3KdaDenseSlot::BProj,
                    "b_proj.weight",
                    Dtype::F16,
                    vec![heads, h],
                    vec![head_range.clone()],
                    Exl3DenseDtype::Bf16,
                ),
                (
                    Exl3KdaDenseSlot::FAProj,
                    "f_a_proj.weight",
                    Dtype::F16,
                    vec![kd, h],
                    single_range(0..kd),
                    Exl3DenseDtype::Bf16,
                ),
                (
                    Exl3KdaDenseSlot::FBProj,
                    "f_b_proj.weight",
                    Dtype::F16,
                    vec![key, kd],
                    vec![key_range.clone()],
                    Exl3DenseDtype::Bf16,
                ),
                (
                    Exl3KdaDenseSlot::GAProj,
                    "g_a_proj.weight",
                    Dtype::F16,
                    vec![kd, h],
                    single_range(0..kd),
                    Exl3DenseDtype::Bf16,
                ),
                (
                    Exl3KdaDenseSlot::GBProj,
                    "g_b_proj.weight",
                    Dtype::F16,
                    vec![key, kd],
                    vec![key_range.clone()],
                    Exl3DenseDtype::Bf16,
                ),
                (
                    Exl3KdaDenseSlot::Conv,
                    "conv1d.weight",
                    Dtype::BF16,
                    vec![spec.out_features(), 1, config.linear_conv_kernel_dim],
                    ranges,
                    Exl3DenseDtype::Bf16,
                ),
                (
                    Exl3KdaDenseSlot::ALog,
                    "A_log",
                    Dtype::F32,
                    vec![heads],
                    vec![head_range],
                    Exl3DenseDtype::F32,
                ),
                (
                    Exl3KdaDenseSlot::DtBias,
                    "dt_bias",
                    Dtype::F32,
                    vec![key],
                    vec![key_range],
                    Exl3DenseDtype::F32,
                ),
                (
                    Exl3KdaDenseSlot::ONorm,
                    "o_norm.weight",
                    Dtype::BF16,
                    vec![kd],
                    single_range(0..kd),
                    Exl3DenseDtype::Bf16,
                ),
            ] {
                dense.push((slot, upload(name, dtype, shape, rows, output)?));
            }
            aux::merge_auxiliary(&mut dense, gpu)
        })();
        if let Err(error) = result {
            let cleanup = release_all(&mut projections, &mut dense, gpu);
            return Err(match cleanup {
                Ok(()) => error,
                Err(cleanup) => {
                    error.context(format!("EXL3 KDA rollback also failed: {cleanup:#}"))
                }
            });
        }
        let mut projections = projections.into_iter();
        // Four successful pushes above are the only path to this conversion.
        Ok(Self {
            q_proj: projections
                .next()
                .context("EXL3 KDA missing Q projection")?,
            k_proj: projections
                .next()
                .context("EXL3 KDA missing K projection")?,
            v_proj: projections
                .next()
                .context("EXL3 KDA missing V projection")?,
            o_proj: projections
                .next()
                .context("EXL3 KDA missing O projection")?,
            dense: Glm5Exl3KdaDenseResources { tensors: dense },
        })
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3KdaResources {
    fn label(&self) -> &'static str {
        "GLM EXL3 KDA resources"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut error = None;
        for projection in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ] {
            if let Err(e) = projection.release(gpu)
                && error.is_none()
            {
                error = Some(e);
            }
        }
        if let Err(e) = self.dense.release(gpu)
            && error.is_none()
        {
            error = Some(e);
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
fn release_dense(
    tensors: &mut [(Exl3KdaDenseSlot, Exl3DenseDeviceTensor)],
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let mut error = None;
    for (_, tensor) in tensors.iter_mut().rev() {
        if let Err(e) = tensor.release(gpu)
            && error.is_none()
        {
            error = Some(e);
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
fn release_all(
    projections: &mut [Glm5Exl3Projection],
    dense: &mut [(Exl3KdaDenseSlot, Exl3DenseDeviceTensor)],
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let mut result = release_dense(dense, gpu);
    for projection in projections.iter_mut().rev() {
        if let Err(e) = projection.release(gpu)
            && result.is_ok()
        {
            result = Err(e);
        }
    }
    result
}

fn single_range(range: std::ops::Range<usize>) -> Vec<std::ops::Range<usize>> {
    vec![range]
}

#[path = "exl3_kda_aux.rs"]
mod aux;

#[cfg(test)]
#[path = "exl3_kda_loader_tests.rs"]
mod tests;
