// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-native GLM-5 EXL3 projection ownership. This does not enable
//! serving: the native forward path must consume these weights explicitly.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::tp::{Exl3DeviceShard, ShardAxis, TpShardPlan};
use spark_runtime::weights::exl3::{Exl3CheckpointRouter, Exl3LinearView};

/// An explicit full-checkpoint projection shape plus rank-local TP policy.
/// Routed experts are present on every rank; their intermediate channels are
/// sharded through TP rather than the legacy NVFP4 expert-TP/EP path.
#[derive(Debug, Clone)]
pub struct Exl3ProjectionSpec {
    prefix: String,
    input: usize,
    output: usize,
    axis: ShardAxis,
    segments: Vec<usize>,
    rank: usize,
    world: usize,
}

impl Exl3ProjectionSpec {
    /// Config must already have topology's **local** attention/linear head
    /// counts. Hidden, intermediate and vocabulary sizes remain **global**.
    /// No topology or dimension defaults are inferred.
    pub fn from_rank_local_config(config: &ModelConfig, prefix: &str) -> Result<Self> {
        ensure!(
            config.model_type == "glm5_next",
            "EXL3 GLM requires model_type=glm5_next"
        );
        ensure!(
            config.tp_world_size > 0 && config.tp_rank < config.tp_world_size,
            "EXL3 GLM requires explicit valid TP rank/world"
        );
        ensure!(
            config.ep_world_size == 1 && config.ep_rank == 0 && !config.expert_tp,
            "EXL3 GLM requires pure TP, ep_world_size=1, ep_rank=0, expert_tp=false"
        );
        ensure!(
            !config.weight_prefix.is_empty(),
            "EXL3 GLM requires weight_prefix"
        );
        let h = config.hidden_size;
        let mut segments = Vec::new();
        let (input, output, axis) = if prefix == "lm_head" {
            (h, config.vocab_size, ShardAxis::Output)
        } else {
            let stem = format!("{}.layers.", config.weight_prefix);
            let remainder = prefix
                .strip_prefix(&stem)
                .context("EXL3 unsupported GLM prefix")?;
            let (layer, projection) = remainder
                .split_once('.')
                .context("EXL3 invalid layer prefix")?;
            let layer: usize = layer.parse().context("EXL3 invalid layer index")?;
            ensure!(
                layer < config.num_hidden_layers,
                "EXL3 GLM layer index out of range"
            );
            ensure!(
                config.layer_types.len() == config.num_hidden_layers,
                "EXL3 GLM requires explicit layer_types for every layer"
            );
            if let Some(name) = projection.strip_prefix("self_attn.") {
                match config.layer_types[layer] {
                    LayerType::LinearAttention => {
                        let key = product(&[
                            config.linear_num_key_heads,
                            config.tp_world_size,
                            config.linear_key_head_dim,
                        ])?;
                        let value = product(&[
                            config.linear_num_value_heads,
                            config.tp_world_size,
                            config.linear_value_head_dim,
                        ])?;
                        match name {
                            "qkv_proj" => {
                                segments = vec![key, key, value];
                                (h, sum(&segments)?, ShardAxis::Output)
                            }
                            "q_proj" | "k_proj" => (h, key, ShardAxis::Output),
                            "v_proj" => (h, value, ShardAxis::Output),
                            "o_proj" => (value, h, ShardAxis::Input),
                            _ => bail!("EXL3 unsupported KDA projection {prefix}"),
                        }
                    }
                    LayerType::FullAttention => {
                        let q_head = sum(&[config.qk_nope_head_dim, config.qk_rope_head_dim])?;
                        let kv_head = sum(&[config.qk_nope_head_dim, config.v_head_dim])?;
                        match name {
                            "q_a_proj" => (h, config.q_lora_rank, ShardAxis::Replicated),
                            "kv_a_proj_with_mqa" => (
                                h,
                                sum(&[config.kv_lora_rank, config.qk_rope_head_dim])?,
                                ShardAxis::Replicated,
                            ),
                            "q_b_proj" => (
                                config.q_lora_rank,
                                product(&[
                                    config.num_attention_heads,
                                    config.tp_world_size,
                                    q_head,
                                ])?,
                                ShardAxis::Output,
                            ),
                            "kv_b_proj" => (
                                config.kv_lora_rank,
                                product(&[
                                    config.num_key_value_heads,
                                    config.tp_world_size,
                                    kv_head,
                                ])?,
                                ShardAxis::Output,
                            ),
                            "o_proj" => (
                                product(&[
                                    config.num_attention_heads,
                                    config.tp_world_size,
                                    config.v_head_dim,
                                ])?,
                                h,
                                ShardAxis::Input,
                            ),
                            // Indexer scores select one shared token set on every TP rank.
                            "indexer.wq_b" => (
                                config.q_lora_rank,
                                product(&[config.index_n_heads, config.index_head_dim])?,
                                ShardAxis::Replicated,
                            ),
                            _ => bail!("EXL3 unsupported MLA projection {prefix}"),
                        }
                    }
                    _ => bail!("EXL3 unsupported GLM layer type at {layer}"),
                }
            } else if let Some(name) = projection.strip_prefix("mlp.") {
                let dense = config.mlp_only_layers.contains(&layer);
                let (width, name) = if let Some(name) = name.strip_prefix("experts.") {
                    ensure!(!dense, "EXL3 routed expert on dense layer");
                    let (expert, name) =
                        name.split_once('.').context("EXL3 invalid expert prefix")?;
                    let expert: usize = expert.parse().context("EXL3 invalid expert index")?;
                    ensure!(
                        expert < config.num_experts,
                        "EXL3 expert index out of range"
                    );
                    (config.moe_intermediate_size, name)
                } else if let Some(name) = name.strip_prefix("shared_experts.") {
                    ensure!(!dense, "EXL3 shared expert on dense layer");
                    (config.shared_expert_intermediate_size, name)
                } else {
                    ensure!(dense, "EXL3 dense FFN projection on sparse layer");
                    (config.intermediate_size, name)
                };
                match name {
                    "gate_proj" | "up_proj" => (h, width, ShardAxis::Output),
                    "down_proj" => (width, h, ShardAxis::Input),
                    _ => bail!("EXL3 unsupported FFN projection {prefix}"),
                }
            } else {
                bail!("EXL3 unsupported GLM projection {prefix}");
            }
        };
        ensure!(
            input > 0 && output > 0,
            "EXL3 GLM projection requires positive dimensions"
        );
        Ok(Self {
            prefix: prefix.to_owned(),
            input,
            output,
            axis,
            segments,
            rank: config.tp_rank,
            world: config.tp_world_size,
        })
    }

    pub fn in_features(&self) -> usize {
        self.input
    }
    pub fn out_features(&self) -> usize {
        self.output
    }
    pub fn axis(&self) -> ShardAxis {
        self.axis
    }
    pub fn output_segments(&self) -> &[usize] {
        &self.segments
    }

    /// Build a validated H128-aligned ownership plan for checkpoint bytes.
    pub fn shard_plan(&self, view: &Exl3LinearView<'_>) -> Result<TpShardPlan> {
        ensure!(
            view.in_features() == self.input && view.out_features() == self.output,
            "EXL3 GLM projection shape mismatch for {}",
            self.prefix
        );
        if self.segments.is_empty() {
            TpShardPlan::new(view, self.axis, self.rank, self.world)
        } else {
            TpShardPlan::new_output_segments(view, &self.segments, self.rank, self.world)
        }
        .with_context(|| {
            format!(
                "EXL3 GLM {} TP rank {} of {}",
                self.prefix, self.rank, self.world
            )
        })
    }

    /// Resolve declared numerical kernels before uploading projection bytes.
    pub fn load_projection(
        &self,
        checkpoint: &Exl3CheckpointRouter,
        gpu: &dyn GpuBackend,
        capacity: usize,
        stream: u64,
        select: super::exl3_numerics::Exl3NumericsSelector,
    ) -> Result<super::exl3_projection::Glm5Exl3Projection> {
        let view = checkpoint.linear_view(&self.prefix, self.input, self.output)?;
        let plan = self.shard_plan(&view)?;
        let numerics = select(
            self.world,
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
            numerics,
        )?;
        let shard = Exl3DeviceShard::upload(gpu, &view, &plan)?;
        super::exl3_projection::Glm5Exl3Projection::new_with_numerics(
            gpu, shard, capacity, stream, numerics,
        )
        .with_context(|| format!("EXL3 GLM numerical projection {}", self.prefix))
    }

    /// Route checkpoint reads and rank-local upload through their I/O boundaries.
    /// Caller registers the returned resource for model teardown.
    pub fn load(
        &self,
        checkpoint: &Exl3CheckpointRouter,
        gpu: &dyn GpuBackend,
    ) -> Result<Exl3DeviceShard> {
        let view = checkpoint.linear_view(&self.prefix, self.input, self.output)?;
        let plan = self.shard_plan(&view)?;
        Exl3DeviceShard::upload(gpu, &view, &plan)
            .with_context(|| format!("EXL3 GLM upload {} rank {}", self.prefix, self.rank))
    }
}

fn product(values: &[usize]) -> Result<usize> {
    ensure!(
        values.iter().all(|&value| value > 0),
        "EXL3 GLM requires positive dimensions"
    );
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .context("EXL3 GLM dimension overflow")
    })
}

fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |acc, &value| {
        acc.checked_add(value)
            .context("EXL3 GLM dimension overflow")
    })
}

#[cfg(test)]
#[path = "exl3_tests.rs"]
mod tests;
