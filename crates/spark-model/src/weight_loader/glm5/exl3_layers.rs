// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM assembly with explicit execution capacity and resource transfer.
use super::{exl3_kda_loader::Glm5Exl3KdaResources, exl3_mla::load_exl3_mla_layer};
use crate::layers::glm5_exl3_ffn::{
    MoeScratch, MoeScratchGeometry, load_glm5_exl3_ffn_with_scratch,
};
use crate::{layer::TransformerLayer, layers::Glm5KdaLayer, weight_map::dense_auto};
use anyhow::{Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use atlas_core::scope::ModelResource;
use spark_runtime::{
    gpu::GpuBackend,
    kv_cache::KvCacheDtype,
    weights::{WeightStore, exl3::dense::Exl3DenseDeviceTensor},
};

pub type NativeLayers = (
    Vec<Box<dyn TransformerLayer>>,
    Vec<Exl3DenseDeviceTensor>,
    Option<MoeScratch>,
);

pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    kv_dtypes: &[KvCacheDtype],
    capacity: usize,
    stream: u64,
) -> Result<NativeLayers> {
    let checkpoint = store
        .exl3_checkpoint()
        .ok_or_else(|| anyhow::anyhow!("Native GLM assembly requires EXL3 checkpoint"))?;
    ensure!(
        capacity > 0 && config.ep_world_size == 1 && !config.expert_tp,
        "Native GLM assembly requires positive capacity and pure TP"
    );
    ensure!(
        kv_dtypes.len()
            == config
                .layer_types
                .iter()
                .filter(|&&kind| kind == LayerType::FullAttention)
                .count(),
        "EXL3 requires an explicit KV dtype for every MLA layer"
    );
    let routed_layers = (0..config.num_hidden_layers)
        .filter(|l| !config.mlp_only_layers.contains(l))
        .count();
    let mut scratch = if routed_layers > 0 {
        let geometry = MoeScratchGeometry::new(config, capacity, stream)?;
        let bytes = geometry.total_bytes()?;
        let old_bytes = bytes
            .checked_mul(routed_layers)
            .ok_or_else(|| anyhow::anyhow!("EXL3 layer scratch accounting overflow"))?;
        tracing::info!(
            capacity,
            routed_layers,
            scratch_bytes = bytes,
            prior_per_layer_scratch_bytes = old_bytes,
            recovered_bytes = old_bytes - bytes,
            "EXL3 routed scratch allocation before KV budgeting; excludes per-layer shared dense activations"
        );
        Some(MoeScratch::allocate(gpu, geometry)?)
    } else {
        None
    };
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut companions = Vec::new();
    let mut attention = 0;
    let result = (|| -> Result<()> {
        for layer in 0..config.num_hidden_layers {
            let prefix = config.layer_prefix(layer);
            let input_norm = dense_auto(store, &format!("{prefix}.input_layernorm.weight"), gpu)?;
            let post_norm = dense_auto(
                store,
                &format!("{prefix}.post_attention_layernorm.weight"),
                gpu,
            )?;
            let hc = super::components::load_hc(store, &prefix, layer, config, gpu)?;
            let ffn = load_glm5_exl3_ffn_with_scratch(
                checkpoint,
                config,
                layer,
                gpu,
                capacity,
                stream,
                crate::weight_loader::glm53_tp2_packed_numerics,
                scratch.as_ref().map(MoeScratch::borrow),
            )?;
            match config.layer_type(layer) {
                LayerType::LinearAttention => {
                    let resource = Glm5Exl3KdaResources::load(
                        checkpoint, config, layer, gpu, capacity, stream,
                    )?;
                    let (weights, dense) = resource.into_weights(gpu)?;
                    companions.extend(dense.into_tensors());
                    layers.push(Box::new(Glm5KdaLayer::new(
                        input_norm, post_norm, weights, ffn, hc, layer, config, gpu,
                    )?));
                }
                LayerType::FullAttention => {
                    layers.push(load_exl3_mla_layer(
                        store,
                        config,
                        layer,
                        attention,
                        input_norm,
                        post_norm,
                        ffn,
                        Some(hc),
                        gpu,
                        kv_dtypes[attention],
                        capacity,
                        stream,
                    )?);
                    attention += 1;
                }
                other => anyhow::bail!("EXL3 unsupported GLM layer type {other:?}"),
            }
            checkpoint.evict_cached_pages()?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        if let Err(completion) = gpu.synchronize(stream) {
            // No owner may be freed while kernel completion is unknown.
            std::mem::forget(layers);
            std::mem::forget(companions);
            std::mem::forget(scratch);
            return Err(error.context(format!("EXL3 rollback completion unknown: {completion:#}")));
        }
        let mut cleanup = None;
        for layer in &mut layers {
            let released = if let Some(kda) = layer
                .as_any_mut()
                .and_then(|l| l.downcast_mut::<Glm5KdaLayer>())
            {
                kda.release(gpu)
            } else if let Some(mla) = layer
                .as_any_mut()
                .and_then(|l| l.downcast_mut::<crate::layers::Qwen3AttentionLayer>())
            {
                mla.release_exl3_mla(gpu)
            } else {
                Ok(())
            };
            if let Err(e) = released {
                cleanup = Some(e);
            }
        }
        for tensor in &mut companions {
            if let Err(e) = tensor.release(gpu) {
                cleanup = Some(e);
            }
        }
        if let Some(owner) = &mut scratch {
            if let Err(e) = owner.release(gpu) {
                cleanup = Some(e);
            }
        }
        return Err(match cleanup {
            None => error,
            Some(e) => error.context(format!("EXL3 assembly rollback failed: {e:#}")),
        });
    }
    Ok((layers, companions, scratch))
}
