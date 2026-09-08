// SPDX-License-Identifier: AGPL-3.0-only

//! Setters + transposes + transpose_for_prefill_unified_inner.

use super::*;

#[path = "helpers_unified_phases.rs"]
mod unified_phases;
#[cfg(test)]
#[path = "helpers_unified_tests.rs"]
mod unified_tests;

impl MoeLayer {
    /// Transpose MoE weights for coalesced prefill GEMM reads.
    ///
    /// Transposes per-expert routed weights [N, K/2] → [K/2, N] to enable
    /// the cp.async pipelined FP8-MMA K64 kernels. This doubles expert
    /// memory (~17 GB for 35B, ~30 GB for 122B) but eliminates the
    /// catastrophic uncoalesced B reads in the fallback grouped GEMM,
    /// cutting MoE prefill time by ~2x.
    /// Set pre-expert norm (Gemma-4 26B: pre_feedforward_layernorm_2).
    /// Applied to input AFTER routing but BEFORE expert dispatch.
    pub fn set_pre_expert_norm(&mut self, norm: crate::weight_map::DenseWeight) {
        self.pre_expert_norm = Some(norm);
    }

    /// Set GeGLU activation for MoE experts (Gemma-4 26B).
    /// Replaces SiLU with GELU in the sorted/unfused path and forces decode
    /// to use the sorted path (avoiding fused SiLU kernels).
    pub fn set_gelu_activation(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.moe_act_mul = gpu.kernel("gelu", "gelu_mul")?;
        self.gelu_activation = true;
        Ok(())
    }

    pub fn transpose_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, true)
    }

    /// Transpose only the gate+up routed weights, leaving the down projection
    /// in its original layout. Cuts the transpose memory cost from ~3×
    /// (gate+up+down) to ~2× per expert. Used by MiniMax M2.7-NVFP4 EP=2
    /// when the full transpose doesn't fit but gate+up does — the fused
    /// `moe_w4a16_fused_gate_up_k64_n128` kernel still runs (capturing the
    /// dominant gate+up bandwidth savings), while down stays on the
    /// uncoalesced grouped-GEMM path.
    pub fn transpose_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, false)
    }

    pub(super) fn transpose_for_prefill_impl(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        include_down: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        // Transpose per-expert routed weights for coalesced prefill GEMM reads.
        let num_experts = self.weights.experts.len();
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        let mut down_t = Vec::with_capacity(num_experts);

        // ARM-2 Phase-K Family C: native-MXFP4 routed experts have per-32 E8M0
        // scales ([N, K/32]); NVFP4 is per-16. The scale transpose must use the
        // matching block size or the E8M0 kernels read a mis-shaped scale table.
        let routed_gs =
            if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                32
            } else {
                16
            };
        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
                if include_down {
                    down_t.push(QuantizedWeight::null());
                }
            } else {
                gate_t.push(
                    expert
                        .gate_proj
                        .transpose_for_gemm_gs(gpu, inter, h, routed_gs)?,
                );
                up_t.push(
                    expert
                        .up_proj
                        .transpose_for_gemm_gs(gpu, inter, h, routed_gs)?,
                );
                if include_down {
                    down_t.push(
                        expert
                            .down_proj
                            .transpose_for_gemm_gs(gpu, h, inter, routed_gs)?,
                    );
                }
            }
        }

        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        if include_down {
            self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        }

        // Transpose shared expert weights (tiny: ~5 MB per layer).
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
            if include_down {
                self.shared_down_t =
                    Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                        gpu,
                        h,
                        shared_inter,
                    )?);
            }
        }

        Ok(())
    }

    /// Phase 8a unified-layout transpose pass: build persistent transposed
    /// gate/up/down for all experts, freeing the untransposed copies between
    /// phases so the entire pass fits in tight memory budgets that the
    /// non-unified `transpose_for_prefill_impl(true)` would reject.
    ///
    /// Phased flow (memory math for MiniMax M2.7-NVFP4 EP=2 ≈ 47 GB free):
    ///   A. Transpose gate+up               (allocs +39 GB; free ≈ 8 GB)
    ///   B. Free gate+up untransposed       (frees 39 GB; free ≈ 47 GB)
    ///   C. Transpose down                  (allocs +20 GB; free ≈ 27 GB)
    ///   D. Free down untransposed          (frees 20 GB; free ≈ 47 GB)
    ///
    /// Net memory: all originals are freed and the layout is transposed-only.
    /// Models that need a decode-native shared expert use the explicitly named
    /// `transpose_for_prefill_unified_keep_shared` variant below.
    ///
    /// Caller responsibilities:
    ///   1. Set `ATLAS_UNIFIED_MOE_LAYOUT=1` so `MoeLayer::use_t_layout_for_decode()`
    ///      returns true at dispatch time.
    ///   2. Call this method INSTEAD of `transpose_for_prefill` /
    ///      `transpose_gate_up_for_prefill`.
    pub fn transpose_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, false, false)
    }

    /// Unified routed-expert layout while retaining only the small shared
    /// expert's decode-native weights. GLM K=4 verification uses this to run
    /// an exact-M=4 shared projection without paying hybrid layout's cost for
    /// every routed expert.
    pub fn transpose_for_prefill_unified_keep_shared(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, false, true)
    }

    /// Hybrid-layout transpose pass — analogue of `transpose_for_prefill_unified`
    /// that **keeps** the untransposed originals so decode + MTP verify dispatch
    /// can continue using the warp-reduction kernels. Allocates ~58 GB
    /// transposed alongside the existing ~58 GB originals on MiniMax M2.7-NVFP4
    /// EP=2; fits in 122 GB GB10 with KV-cache headroom up to ~32K context.
    /// Caller is responsible for memory-fit gating (factory checks free memory
    /// before invoking this).
    pub fn transpose_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, true, true)
    }

    /// Phased build of the transposed weight set. When `keep_originals` is true
    /// (hybrid-layout mode), Phase B and Phase D frees are skipped so decode
    /// paths still find the untransposed weights. When false (unified-layout
    /// mode), routed originals are freed between phases. The independent
    /// `keep_shared_originals` bit exempts only the shared expert.
    pub(super) fn transpose_for_prefill_unified_inner(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        keep_originals: bool,
        keep_shared_originals: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;

        // ── Phase A: transpose gate+up routed experts ──
        // ARM-2 Phase-K Family C: native-MXFP4 routed experts are per-32 E8M0.
        let routed_gs =
            if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                32
            } else {
                16
            };
        let gate_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.gate_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.gate_proj
                }
            })
            .collect();
        let up_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.gate_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.up_proj
                }
            })
            .collect();
        let gate_t = self.transpose_experts_gpu(gpu, &gate_src, inter, h, routed_gs)?;
        let up_t = self.transpose_experts_gpu(gpu, &up_src, inter, h, routed_gs)?;
        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        self.transpose_unified_shared_gate_up(gpu, config)?;

        if !keep_originals {
            // ── Phase B: free gate+up untransposed ──
            // The previous gate_ptrs / up_ptrs device-side pointer tables now
            // contain stale addresses, but the unified dispatch never reads
            // them (gated by `use_t_layout_for_decode()`).
            for expert in &mut self.weights.experts {
                if !expert.gate_proj.weight.is_null() {
                    gpu.free(expert.gate_proj.weight)?;
                    gpu.free(expert.gate_proj.weight_scale)?;
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                if !expert.up_proj.weight.is_null() {
                    gpu.free(expert.up_proj.weight)?;
                    gpu.free(expert.up_proj.weight_scale)?;
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !keep_shared_originals {
                self.release_unified_shared_gate_up(gpu, config)?;
            }
        }
        self.transpose_unified_down_phase(
            gpu,
            config,
            routed_gs,
            keep_originals,
            keep_shared_originals,
        )
    }

    /// Transpose one projection across ALL routed experts on the GPU, into a
    /// single slab allocation per buffer.
    /// Replaces a per-expert `QuantizedWeight::transpose_for_gemm_gs`, which
    /// round-trips each expert through the host and creates two allocations.
    /// The batched kernel avoids those transfers and allocation pressure.
    /// `src` supplies the per-expert untransposed `[n, k/2]` packed bytes and
    /// `[n, k/group_size]` scales; the returned `QuantizedWeight`s point into
    /// the two slabs and carry the source's scale metadata unchanged.
    #[allow(clippy::too_many_arguments)]
    fn transpose_experts_gpu(
        &self,
        gpu: &dyn GpuBackend,
        src: &[QuantizedWeight],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<QuantizedWeight>> {
        let num_experts = src.len();
        let (local_experts, compact_slots) =
            super::compact_layout::compact_slot_map(src.iter().map(|w| !w.is_null()));
        let packed_each = n * (k / 2);
        let scale_each = n * (k / group_size);
        anyhow::ensure!(
            packed_each > 0 && scale_each > 0,
            "transpose_experts_gpu: zero-sized projection (n={n} k={k} gs={group_size})"
        );

        if local_experts == 0 {
            return Ok(vec![QuantizedWeight::null(); num_experts]);
        }
        let packed_slab = gpu.alloc(local_experts * packed_each)?;
        let scale_slab = gpu.alloc(local_experts * scale_each)?;

        // Destinations carve the slabs; a NULL source keeps a NULL slot so the
        // kernel's own NULL guard skips that expert (EP-remote convention).
        let mut out = Vec::with_capacity(num_experts);
        for (w, compact_slot) in src.iter().zip(compact_slots) {
            if let Some(slot) = compact_slot {
                out.push(QuantizedWeight {
                    weight: packed_slab.offset(slot * packed_each),
                    weight_scale: scale_slab.offset(slot * scale_each),
                    weight_scale_2: w.weight_scale_2,
                    input_scale: w.input_scale,
                    weight_scale_2_vec: w.weight_scale_2_vec,
                });
            } else {
                out.push(QuantizedWeight::null());
            }
        }

        let src_tbl = build_ptr_table_from_qw(src, gpu)?;
        let dst_tbl = build_ptr_table_from_qw(&out, gpu)?;
        let stream = gpu.default_stream();
        // Packed [n, k/2] -> [k/2, n].
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.packed_ptrs,
            dst_tbl.packed_ptrs,
            n as u32,
            (k / 2) as u32,
            num_experts as u32,
            stream,
        )?;
        // Scales [n, k/group_size] -> [k/group_size, n].
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.scale_ptrs,
            dst_tbl.scale_ptrs,
            n as u32,
            (k / group_size) as u32,
            num_experts as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        // The pointer tables were scratch for the launch only.
        gpu.free(src_tbl.packed_ptrs)?;
        gpu.free(src_tbl.scale_ptrs)?;
        gpu.free(src_tbl.scale2_vals)?;
        gpu.free(dst_tbl.packed_ptrs)?;
        gpu.free(dst_tbl.scale_ptrs)?;
        gpu.free(dst_tbl.scale2_vals)?;
        Ok(out)
    }

    /// Build per-expert swizzled SFB weight-scale tables for the CUTLASS grouped
    /// NVFP4 path (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`). For each expert, swizzle the
    /// `[K/16,N]` `gate_ptrs_t`/`up_ptrs_t` scale into the CUTLASS SFB atom via
    /// `pack_weight_sfb`, then upload the per-expert pointer arrays. The grouped
    /// kernel pairs these with `gate_ptrs.packed` (`[N,K/2]`) + the real per-expert
    /// `scale2`. Requires FAST_MOE=full (gate_ptrs_t/up_ptrs_t present); no-op else.
    pub fn build_cutlass_grouped_sfb(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let num = self.weights.experts.len();
        // Swizzled SFB atom size (bytes): round_up(N,128) * round_up(K/16,4).
        let sfb_len = |n: usize, k: usize| n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4;
        // Prefer the Atlas-transposed [K/16,N] scales when they exist. Without
        // them (a checkpoint served straight from its native tables, e.g.
        // Laguna with the unified transpose disabled) fall back to the
        // ORIGINAL [N,K/16] scales and tell the packer to read N-major — the
        // SFB output is identical, so this avoids materialising a transposed
        // copy purely to feed the swizzle.
        let (gate_scale_dev, up_scale_dev, src_n_major) =
            match (self.gate_ptrs_t.as_ref(), self.up_ptrs_t.as_ref()) {
                (Some(g), Some(u)) => (g.scale_ptrs, u.scale_ptrs, false),
                _ => (self.gate_ptrs.scale_ptrs, self.up_ptrs.scale_ptrs, true),
            };
        if gate_scale_dev.is_null() || up_scale_dev.is_null() {
            return Ok(());
        }
        let down_scale_dev = match self.down_ptrs_t.as_ref() {
            Some(d) => Some(d.scale_ptrs),
            None if !self.down_ptrs.scale_ptrs.is_null() => Some(self.down_ptrs.scale_ptrs),
            None => None,
        };
        let mut owned: Vec<DevicePtr> = Vec::new();
        // Swizzle each expert's [K/16,N] scale into the CUTLASS SFB atom. `n`/`k`
        // are the projection's GEMM dims: gate/up = (inter, hidden); down = (hidden, inter).
        // Returns the HOST vector of per-expert SFB pointers: the grouped C entry
        // consumes pointer values host-side, so no device copy of this table is
        // ever made — the values go straight into the layer-owned snapshot below.
        let mut build_one = |scale_ptrs_dev: DevicePtr, n: usize, k: usize| -> Result<Vec<u64>> {
            let len = sfb_len(n, k);
            let sp = crate::layers::ops::read_expert_ptrs_u64(gpu, scale_ptrs_dev, num)?;
            let mut sfb_ptrs = vec![0u64; num];
            for (e, &sptr) in sp.iter().enumerate() {
                if sptr == 0 {
                    continue; // remote/placeholder expert
                }
                let sfb = gpu.alloc(len)?;
                spark_runtime::cutlass::pack_weight_sfb(
                    sptr,
                    sfb.0,
                    n as u32,
                    k as u32,
                    src_n_major,
                    stream,
                )?;
                sfb_ptrs[e] = sfb.0;
                owned.push(sfb);
            }
            gpu.synchronize(stream)?;
            Ok(sfb_ptrs)
        };
        let gate_sfb = build_one(gate_scale_dev, inter, h)?;
        let up_sfb = build_one(up_scale_dev, inter, h)?;
        let down = match down_scale_dev {
            Some(ds) => Some((
                self.down_ptrs.packed_ptrs,
                build_one(ds, h, inter)?,
                self.down_ptrs.scale2_vals,
            )),
            None => None,
        };
        self.cutlass_grouped_host = Some(crate::layers::ops::MoeCutlassHostTables::snapshot(
            gpu,
            num,
            self.gate_ptrs.packed_ptrs,
            gate_sfb,
            self.gate_ptrs.scale2_vals,
            self.up_ptrs.packed_ptrs,
            up_sfb,
            self.up_ptrs.scale2_vals,
            down,
        )?);
        self._cutlass_sfb_owned = owned;
        tracing::info!(
            "CUTLASS grouped SFB: built {num} experts gate/up (N={inter} K={h}) + down (N={h} K={inter})"
        );
        Ok(())
    }
}
