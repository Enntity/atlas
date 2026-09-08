// SPDX-License-Identifier: AGPL-3.0-only
//! Private construction phases; no partial-layout serving entry point.

use super::*;

impl MoeLayer {
    pub(super) fn transpose_unified_shared_gate_up(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        let h = config.hidden_size;
        let shared_inter = config.shared_expert_intermediate_size;
        // Shared expert (tiny, do unconditionally — fits regardless).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
        }
        Ok(())
    }

    pub(super) fn release_unified_shared_gate_up(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        let shared_inter = config.shared_expert_intermediate_size;
        if !self.weights.shared_expert.gate_proj.weight.is_null() && shared_inter > 0 {
            gpu.free(self.weights.shared_expert.gate_proj.weight)?;
            gpu.free(self.weights.shared_expert.gate_proj.weight_scale)?;
            self.weights.shared_expert.gate_proj.weight = DevicePtr::NULL;
            self.weights.shared_expert.gate_proj.weight_scale = DevicePtr::NULL;
            gpu.free(self.weights.shared_expert.up_proj.weight)?;
            gpu.free(self.weights.shared_expert.up_proj.weight_scale)?;
            self.weights.shared_expert.up_proj.weight = DevicePtr::NULL;
            self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL;
        }
        Ok(())
    }

    pub(super) fn transpose_unified_down_phase(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        routed_group: usize,
        keep_originals: bool,
        keep_shared_originals: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;
        // ── Phase C: transpose down routed experts ──
        let down_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.down_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.down_proj
                }
            })
            .collect();
        let down_t = self.transpose_experts_gpu(gpu, &down_src, h, inter, routed_group)?;
        self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        if !keep_originals {
            // ── Phase D: free down untransposed ──
            for expert in &mut self.weights.experts {
                if !expert.down_proj.weight.is_null() {
                    gpu.free(expert.down_proj.weight)?;
                    gpu.free(expert.down_proj.weight_scale)?;
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !keep_shared_originals
                && !self.weights.shared_expert.down_proj.weight.is_null()
                && shared_inter > 0
            {
                gpu.free(self.weights.shared_expert.down_proj.weight)?;
                gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
                self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        }
        Ok(())
    }
}
