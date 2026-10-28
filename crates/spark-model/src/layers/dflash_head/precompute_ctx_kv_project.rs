// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl BlockDiffusionDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ctx_kv_project(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        rows: u32,
        hidden: u32,
        target_hidden: u32,
        fc_dst: DevicePtr,
        fused_dst: DevicePtr,
        stream: u64,
        dump: Option<&dyn Fn(&str, DevicePtr, usize) -> Result<()>>,
    ) -> Result<()> {
        let Some(fused_kv) = self.fused_kv_weight else {
            anyhow::bail!("DFlash ctx_kv_project called without fused_kv_weight");
        };
        let bf16 = 2usize;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let layers = self.num_layers;
        self.ctx_projection(
            gpu,
            0,
            &self.fc,
            src,
            fc_dst,
            rows,
            hidden,
            target_hidden,
            stream,
        )?;
        if let Some(dump) = dump {
            dump("fc_proj", fc_dst, rows as usize * hidden as usize * bf16)?;
        }
        crate::layers::ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            fc_dst,
            &self.hidden_norm,
            fc_dst,
            rows,
            hidden,
            self.rms_norm_eps,
            stream,
        )?;
        if let Some(dump) = dump {
            dump(
                "fc_proj_normed",
                fc_dst,
                rows as usize * hidden as usize * bf16,
            )?;
        }
        let fused_weight = DenseWeight { weight: fused_kv };
        self.ctx_projection(
            gpu,
            1,
            &fused_weight,
            fc_dst,
            fused_dst,
            rows,
            layers as u32 * 2 * kv_dim,
            hidden,
            stream,
        )?;
        if let Some(dump) = dump {
            dump(
                "fused_kv_out",
                fused_dst,
                rows as usize * layers * 2 * kv_dim as usize * bf16,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ctx_kv_scatter(
        &self,
        gpu: &dyn GpuBackend,
        fused_src: DevicePtr,
        new_ctx_count: usize,
        slot_positions: &[i32],
        slot_mapping_dev: DevicePtr,
        _ctx: &ForwardContext,
        stream: u64,
        commit: bool,
        scratch: &DflashScratch,
        dump: Option<&dyn Fn(&str, DevicePtr, usize) -> Result<()>>,
    ) -> Result<()> {
        let bf16 = 2usize;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let rows = new_ctx_count as u32;
        let layers = self.num_layers;
        let kv_row_bytes = kv_dim as usize * bf16;
        let row_stride = layers * 2 * kv_row_bytes;
        let dump_buf = |label: &str, ptr: DevicePtr, bytes: usize| -> Result<()> {
            dump.map_or(Ok(()), |dump| dump(label, ptr, bytes))
        };
        anyhow::ensure!(
            slot_positions.len() == new_ctx_count,
            "DFlash ctx scatter positions {} != rows {new_ctx_count}",
            slot_positions.len()
        );
        let positions: Vec<u8> = slot_positions
            .iter()
            .copied()
            .cycle()
            .take(layers * new_ctx_count)
            .flat_map(i32::to_le_bytes)
            .collect();
        gpu.copy_h2d(&positions, scratch.norm_buf)?;

        let all_k_stage = scratch.mlp_intermediate;
        for layer in 0..layers {
            gpu.copy_d2d_2d_async(
                fused_src.offset(layer * 2 * kv_row_bytes),
                row_stride,
                all_k_stage.offset(layer * new_ctx_count * kv_row_bytes),
                kv_row_bytes,
                kv_row_bytes,
                new_ctx_count,
                stream,
            )?;
        }
        for layer in 0..layers {
            let k_layer = all_k_stage.offset(layer * new_ctx_count * kv_row_bytes);
            crate::layers::ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                k_layer,
                &self.layers[layer].k_norm,
                k_layer,
                rows * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;
        }
        crate::layers::ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            all_k_stage,
            all_k_stage,
            scratch.norm_buf,
            layers as u32 * rows,
            0,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;
        if dump.is_some() {
            dump_buf(
                "layer0_k_post_rope",
                all_k_stage,
                new_ctx_count * kv_row_bytes,
            )?;
        }
        let v_stage = scratch.v_buf;
        for layer in 0..layers {
            let k_layer = all_k_stage.offset(layer * new_ctx_count * kv_row_bytes);
            gpu.copy_d2d_2d_async(
                fused_src.offset(layer * 2 * kv_row_bytes + kv_row_bytes),
                row_stride,
                v_stage,
                kv_row_bytes,
                kv_row_bytes,
                new_ctx_count,
                stream,
            )?;
            if dump.is_some() && layer == 0 {
                dump_buf("layer0_v", v_stage, new_ctx_count * kv_row_bytes)?;
            }
            if commit {
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(layer), cache.v_pool_ptr(layer))
                };
                crate::layers::ops::reshape_and_cache(
                    gpu,
                    self.kernels.reshape_cache_bf16,
                    k_layer,
                    v_stage,
                    k_pool,
                    v_pool,
                    slot_mapping_dev,
                    rows,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16,
                    kv_dim,
                    kv_dim,
                    0,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
