// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-address persistent state owned by GLM-5.3 sparse-MLA layers.
//!
//! KDA reuses the outer recurrent/conv pool's slot lifecycle, while this
//! module owns the DSA latent/index image that generic paged KV cannot model.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use atlas_core::glm5::Glm53FlashPlan;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::GlmSparseMlaStatePointers;

struct LayerPools {
    layers: Vec<DevicePtr>,
    allocations: Vec<DevicePtr>,
}

impl LayerPools {
    fn allocate(gpu: &dyn GpuBackend, layers: usize, layer_bytes: usize) -> Result<Self> {
        if layers == 0 || layer_bytes == 0 {
            return Ok(Self {
                layers: vec![DevicePtr::NULL; layers],
                allocations: Vec::new(),
            });
        }

        if let Some(total) = layer_bytes.checked_mul(layers)
            && let Ok(base) = gpu.alloc(total)
        {
            gpu.memset(base, 0, total)?;
            return Ok(Self {
                layers: (0..layers)
                    .map(|layer| base.offset(layer * layer_bytes))
                    .collect(),
                allocations: vec![base],
            });
        }

        tracing::warn!(
            "GLM state pool: {layers} x {layer_bytes} B did not fit one contiguous allocation; using per-layer allocations"
        );
        let mut layer_ptrs = Vec::with_capacity(layers);
        for _ in 0..layers {
            let ptr = gpu.alloc(layer_bytes)?;
            gpu.memset(ptr, 0, layer_bytes)?;
            layer_ptrs.push(ptr);
        }
        Ok(Self {
            allocations: layer_ptrs.clone(),
            layers: layer_ptrs,
        })
    }

    fn at(&self, layer: usize, slot: usize, slot_bytes: usize) -> DevicePtr {
        self.layers[layer].offset(slot * slot_bytes)
    }

    fn release(&mut self, gpu: &dyn GpuBackend, first_error: &mut Option<anyhow::Error>) {
        for ptr in self.allocations.drain(..) {
            if let Err(error) = gpu.free(ptr)
                && first_error.is_none()
            {
                *first_error = Some(error);
            }
        }
        self.layers.clear();
    }
}

/// GLM-only geometry and device allocations attached to the common slot pool.
pub(super) struct GlmStatePools {
    latent: LayerPools,
    pooled_keys: LayerPools,
    tail_keys: LayerPools,
    tail_gates: LayerPools,
    metadata: LayerPools,
    checkpoint_tail_keys: LayerPools,
    checkpoint_tail_gates: LayerPools,
    checkpoint_metadata: LayerPools,
    intermediate_tail_keys: LayerPools,
    intermediate_tail_gates: LayerPools,
    intermediate_metadata: LayerPools,
    pub(super) kda_history_bytes: usize,
    latent_slot_bytes: usize,
    pooled_slot_bytes: usize,
    tail_slot_bytes: usize,
    metadata_slot_bytes: usize,
    dsa_layers: usize,
    verify_slots: usize,
    num_intermediates: usize,
}

impl GlmStatePools {
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        max_seq_len: usize,
        has_mtp: bool,
        num_intermediates: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let plan = Glm53FlashPlan::from_config(config)?;
        let total_slots = max_slots
            .checked_add(1)
            .context("GLM state slot count overflow")?;
        let bf16_bytes = 2usize;
        let fp32_bytes = 4usize;
        let history = plan
            .kda_conv_kernel
            .checked_sub(1)
            .context("GLM KDA convolution kernel must be non-zero")?;
        let kda_history_bytes =
            checked_product(&[plan.kda_heads, plan.kda_head_dim, history, fp32_bytes])?;
        let latent_slot_bytes = checked_product(&[max_seq_len, plan.mla_latent_dim, bf16_bytes])?;
        let pooled_slot_bytes = checked_product(&[
            max_seq_len / plan.index_kpool,
            plan.index_key_dim,
            bf16_bytes,
        ])?;
        let tail_slot_bytes =
            checked_product(&[plan.index_kpool - 1, plan.index_key_dim, bf16_bytes])?;
        let metadata_slot_bytes = 4 * size_of::<i32>();
        let dsa_layers = plan.dsa_layers.len();

        let layer_bytes = |slot_bytes: usize| {
            total_slots
                .checked_mul(slot_bytes)
                .context("GLM DSA layer pool size overflow")
        };
        let latent = LayerPools::allocate(gpu, dsa_layers, layer_bytes(latent_slot_bytes)?)?;
        let pooled_keys = LayerPools::allocate(gpu, dsa_layers, layer_bytes(pooled_slot_bytes)?)?;
        let tail_keys = LayerPools::allocate(gpu, dsa_layers, layer_bytes(tail_slot_bytes)?)?;
        let tail_gates = LayerPools::allocate(gpu, dsa_layers, layer_bytes(tail_slot_bytes)?)?;
        let metadata = LayerPools::allocate(gpu, dsa_layers, layer_bytes(metadata_slot_bytes)?)?;
        // DSA's latent cache and complete four-token pools are append-only and
        // are overwritten when verification resumes at a rejected position.
        // Exact rollback therefore snapshots only the incomplete-pool tail
        // and its metadata. This keeps a gamma-wide rollback image in KiB,
        // rather than copying the multi-GiB latent cache after every token.
        let verify_slots = if has_mtp { total_slots } else { 0 };
        let checkpoint_elems = verify_slots;
        let intermediate_elems = verify_slots
            .checked_mul(num_intermediates)
            .context("GLM DSA intermediate count overflow")?;
        let checkpoint_tail_keys =
            LayerPools::allocate(gpu, dsa_layers, checkpoint_elems * tail_slot_bytes)?;
        let checkpoint_tail_gates =
            LayerPools::allocate(gpu, dsa_layers, checkpoint_elems * tail_slot_bytes)?;
        let checkpoint_metadata =
            LayerPools::allocate(gpu, dsa_layers, checkpoint_elems * metadata_slot_bytes)?;
        let intermediate_tail_keys =
            LayerPools::allocate(gpu, dsa_layers, intermediate_elems * tail_slot_bytes)?;
        let intermediate_tail_gates =
            LayerPools::allocate(gpu, dsa_layers, intermediate_elems * tail_slot_bytes)?;
        let intermediate_metadata =
            LayerPools::allocate(gpu, dsa_layers, intermediate_elems * metadata_slot_bytes)?;

        let total_bytes = plan.state_pool_bytes(max_slots, 1, max_seq_len, bf16_bytes)?;
        tracing::info!(
            "GLM persistent state: {max_slots} claimable + 1 padding slot x {} KDA / {dsa_layers} DSA layers = {:.2} GiB",
            plan.kda_layers.len(),
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        Ok(Self {
            latent,
            pooled_keys,
            tail_keys,
            tail_gates,
            metadata,
            checkpoint_tail_keys,
            checkpoint_tail_gates,
            checkpoint_metadata,
            intermediate_tail_keys,
            intermediate_tail_gates,
            intermediate_metadata,
            kda_history_bytes,
            latent_slot_bytes,
            pooled_slot_bytes,
            tail_slot_bytes,
            metadata_slot_bytes,
            dsa_layers,
            verify_slots,
            num_intermediates,
        })
    }

    pub(super) fn dsa_state(&self, dsa_layer: usize, slot: usize) -> GlmSparseMlaStatePointers {
        debug_assert!(dsa_layer < self.dsa_layers);
        GlmSparseMlaStatePointers {
            latent_cache: self.latent.at(dsa_layer, slot, self.latent_slot_bytes),
            pooled_keys: self.pooled_keys.at(dsa_layer, slot, self.pooled_slot_bytes),
            tail_keys: self.tail_keys.at(dsa_layer, slot, self.tail_slot_bytes),
            tail_gates: self.tail_gates.at(dsa_layer, slot, self.tail_slot_bytes),
            tail_metadata: self.metadata.at(dsa_layer, slot, self.metadata_slot_bytes),
        }
    }

    pub(super) fn dsa_checkpoint(
        &self,
        dsa_layer: usize,
        slot: usize,
    ) -> GlmSparseMlaStatePointers {
        debug_assert!(slot < self.verify_slots);
        let current = self.dsa_state(dsa_layer, slot);
        GlmSparseMlaStatePointers {
            latent_cache: current.latent_cache,
            pooled_keys: current.pooled_keys,
            tail_keys: self
                .checkpoint_tail_keys
                .at(dsa_layer, slot, self.tail_slot_bytes),
            tail_gates: self
                .checkpoint_tail_gates
                .at(dsa_layer, slot, self.tail_slot_bytes),
            tail_metadata: self
                .checkpoint_metadata
                .at(dsa_layer, slot, self.metadata_slot_bytes),
        }
    }

    pub(super) fn dsa_intermediate(
        &self,
        dsa_layer: usize,
        slot: usize,
        token_idx: usize,
    ) -> GlmSparseMlaStatePointers {
        debug_assert!(slot < self.verify_slots);
        debug_assert!(token_idx < self.num_intermediates);
        let current = self.dsa_state(dsa_layer, slot);
        let image = slot * self.num_intermediates + token_idx;
        GlmSparseMlaStatePointers {
            latent_cache: current.latent_cache,
            pooled_keys: current.pooled_keys,
            tail_keys: self
                .intermediate_tail_keys
                .at(dsa_layer, image, self.tail_slot_bytes),
            tail_gates: self
                .intermediate_tail_gates
                .at(dsa_layer, image, self.tail_slot_bytes),
            tail_metadata: self.intermediate_metadata.at(
                dsa_layer,
                image,
                self.metadata_slot_bytes,
            ),
        }
    }

    pub(super) fn zero_slot(&self, slot: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        for layer in 0..self.dsa_layers {
            let state = self.dsa_state(layer, slot);
            for (ptr, bytes) in [
                (state.latent_cache, self.latent_slot_bytes),
                (state.pooled_keys, self.pooled_slot_bytes),
                (state.tail_keys, self.tail_slot_bytes),
                (state.tail_gates, self.tail_slot_bytes),
                (state.tail_metadata, self.metadata_slot_bytes),
            ] {
                gpu.memset_async(ptr, 0, bytes, stream)?;
            }
        }
        Ok(())
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        for layer in 0..self.dsa_layers {
            let state = self.dsa_state(layer, slot);
            for (ptr, bytes) in [
                (state.latent_cache, self.latent_slot_bytes),
                (state.pooled_keys, self.pooled_slot_bytes),
                (state.tail_keys, self.tail_slot_bytes),
                (state.tail_gates, self.tail_slot_bytes),
                (state.tail_metadata, self.metadata_slot_bytes),
            ] {
                gpu.memset(ptr, 0, bytes)?;
            }
        }
        Ok(())
    }

    pub(super) fn copy_slot(
        &self,
        from: usize,
        to: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for layer in 0..self.dsa_layers {
            let source = self.dsa_state(layer, from);
            let target = self.dsa_state(layer, to);
            for (src, dst, bytes) in [
                (
                    source.latent_cache,
                    target.latent_cache,
                    self.latent_slot_bytes,
                ),
                (
                    source.pooled_keys,
                    target.pooled_keys,
                    self.pooled_slot_bytes,
                ),
                (source.tail_keys, target.tail_keys, self.tail_slot_bytes),
                (source.tail_gates, target.tail_gates, self.tail_slot_bytes),
                (
                    source.tail_metadata,
                    target.tail_metadata,
                    self.metadata_slot_bytes,
                ),
            ] {
                gpu.copy_d2d_async(src, dst, bytes, stream)?;
            }
        }
        Ok(())
    }

    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first_error = None;
        for pools in [
            &mut self.latent,
            &mut self.pooled_keys,
            &mut self.tail_keys,
            &mut self.tail_gates,
            &mut self.metadata,
            &mut self.checkpoint_tail_keys,
            &mut self.checkpoint_tail_gates,
            &mut self.checkpoint_metadata,
            &mut self.intermediate_tail_keys,
            &mut self.intermediate_tail_gates,
            &mut self.intermediate_metadata,
        ] {
            pools.release(gpu, &mut first_error);
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn checked_product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, factor| {
        product
            .checked_mul(*factor)
            .context("GLM state pool size overflow")
    })
}
