// SPDX-License-Identifier: AGPL-3.0-only

//! Causal prompt-batched launch contracts for GLM-5.3 sparse DSA.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

pub const GLM53_DSA_PREFILL_MODULE: &str = "glm53_dsa_prefill";
pub const GLM53_DSA_PREFILL_FAST_MODULE: &str = "glm53_dsa_prefill_fast";
pub const GLM53_DSA_SCORE_PREFILL_ENTRY: &str = "glm53_dsa_score_prefill";
pub const GLM53_DSA_TOPK_PREFILL_ENTRY: &str = "glm53_dsa_topk_expand_prefill";
pub const GLM53_DSA_SPARSE_MLA_PREFILL_ENTRY: &str = "glm53_dsa_sparse_mla_prefill";
pub const GLM53_DSA_CAUSAL_MLA_PREFILL_ENTRY: &str = "glm53_dsa_causal_mla_prefill";
pub const GLM53_DSA_SPARSE_MLA_PREFILL_WARP_ENTRY: &str = "glm53_dsa_sparse_mla_prefill_warp";

fn pointers(values: &[(&str, DevicePtr)]) -> Result<()> {
    for (name, pointer) in values {
        ensure!(!pointer.is_null(), "GLM DSA prefill {name} pointer is null");
    }
    Ok(())
}

pub struct Glm53DsaScorePrefillArgs {
    pub query: DevicePtr,
    pub head_weights: DevicePtr,
    pub positions: DevicePtr,
    pub valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub scores: DevicePtr,
    pub max_pools: u32,
    pub total_tokens: u32,
    pub num_sequences: u32,
}

impl Glm53DsaScorePrefillArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.max_pools > 0, "GLM DSA prefill requires pool capacity");
        ensure!(self.total_tokens > 0, "GLM DSA prefill requires tokens");
        ensure!(self.num_sequences > 0, "GLM DSA prefill requires sequences");
        pointers(&[
            ("query", self.query),
            ("head weights", self.head_weights),
            ("positions", self.positions),
            ("valid", self.valid),
            ("state table", self.sequence_state_ptrs),
            ("cu_seqlens", self.cu_seqlens),
            ("scores", self.scores),
        ])
    }
}

pub fn glm53_dsa_score_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaScorePrefillArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.max_pools, args.total_tokens, 1])
        .block([128, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.head_weights)
        .arg_ptr(args.positions)
        .arg_ptr(args.valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.scores)
        .arg_u32(args.max_pools)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .launch(stream)
}

pub struct Glm53DsaTopkPrefillArgs {
    pub scores: DevicePtr,
    pub positions: DevicePtr,
    pub valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub output: DevicePtr,
    pub max_pools: u32,
    pub total_tokens: u32,
    pub num_sequences: u32,
}

impl Glm53DsaTopkPrefillArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.max_pools > 0, "GLM DSA prefill requires pool capacity");
        ensure!(self.total_tokens > 0, "GLM DSA prefill requires tokens");
        ensure!(self.num_sequences > 0, "GLM DSA prefill requires sequences");
        pointers(&[
            ("scores", self.scores),
            ("positions", self.positions),
            ("valid", self.valid),
            ("state table", self.sequence_state_ptrs),
            ("cu_seqlens", self.cu_seqlens),
            ("selected output", self.output),
        ])
    }
}

pub fn glm53_dsa_topk_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaTopkPrefillArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.total_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(args.scores)
        .arg_ptr(args.positions)
        .arg_ptr(args.valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.output)
        .arg_u32(args.max_pools)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .launch(stream)
}

pub struct Glm53DsaSparseMlaPrefillArgs {
    pub absorbed_query: DevicePtr,
    pub selected_indices: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub output: DevicePtr,
    pub num_heads: u32,
    pub total_tokens: u32,
    pub num_sequences: u32,
    pub attention_scale: f32,
}

pub struct Glm53DsaCausalMlaPrefillArgs {
    pub absorbed_query: DevicePtr,
    pub positions: DevicePtr,
    pub valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub output: DevicePtr,
    pub num_heads: u32,
    pub total_tokens: u32,
    pub num_sequences: u32,
    pub attention_scale: f32,
}

impl Glm53DsaCausalMlaPrefillArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_heads > 0, "GLM DSA prefill requires heads");
        ensure!(self.total_tokens > 0, "GLM DSA prefill requires tokens");
        ensure!(self.num_sequences > 0, "GLM DSA prefill requires sequences");
        ensure!(
            self.attention_scale.is_finite() && self.attention_scale > 0.0,
            "GLM DSA prefill scale must be finite and positive"
        );
        pointers(&[
            ("absorbed query", self.absorbed_query),
            ("positions", self.positions),
            ("valid", self.valid),
            ("state table", self.sequence_state_ptrs),
            ("cu_seqlens", self.cu_seqlens),
            ("output", self.output),
        ])
    }
}

pub fn glm53_dsa_causal_mla_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaCausalMlaPrefillArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    let tiles = div_ceil(args.total_tokens, 8);
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, args.num_sequences * tiles, 1])
        .block([256, 1, 1])
        .arg_ptr(args.absorbed_query)
        .arg_ptr(args.positions)
        .arg_ptr(args.valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.output)
        .arg_u32(args.num_heads)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .arg_f32(args.attention_scale)
        .launch(stream)
}

impl Glm53DsaSparseMlaPrefillArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_heads > 0, "GLM DSA prefill requires heads");
        ensure!(self.total_tokens > 0, "GLM DSA prefill requires tokens");
        ensure!(self.num_sequences > 0, "GLM DSA prefill requires sequences");
        ensure!(
            self.attention_scale.is_finite() && self.attention_scale > 0.0,
            "GLM DSA prefill scale must be finite and positive"
        );
        pointers(&[
            ("absorbed query", self.absorbed_query),
            ("selected indices", self.selected_indices),
            ("state table", self.sequence_state_ptrs),
            ("cu_seqlens", self.cu_seqlens),
            ("output", self.output),
        ])
    }
}

pub fn glm53_dsa_sparse_mla_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaSparseMlaPrefillArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, args.total_tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(args.absorbed_query)
        .arg_ptr(args.selected_indices)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.output)
        .arg_u32(args.num_heads)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .arg_f32(args.attention_scale)
        .launch(stream)
}

pub fn glm53_dsa_sparse_mla_prefill_warp(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaSparseMlaPrefillArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, div_ceil(args.total_tokens, 8), 1])
        .block([256, 1, 1])
        .arg_ptr(args.absorbed_query)
        .arg_ptr(args.selected_indices)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.output)
        .arg_u32(args.num_heads)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .arg_f32(args.attention_scale)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_dsa_prefill.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");
    const FAST_SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_dsa_prefill_fast.cu");

    #[test]
    fn causal_prefill_entrypoints_are_registered() {
        for entry in [
            GLM53_DSA_SCORE_PREFILL_ENTRY,
            GLM53_DSA_TOPK_PREFILL_ENTRY,
            GLM53_DSA_SPARSE_MLA_PREFILL_ENTRY,
        ] {
            assert!(SOURCE.contains(&format!("void {entry}(")));
        }
        assert!(REGISTRY.contains("glm53_dsa_prefill = \"glm53_dsa_prefill\""));
    }

    #[test]
    fn fast_prefill_tiles_queries_and_keeps_sparse_fallback_on_device() {
        assert!(FAST_SOURCE.contains("void glm53_dsa_causal_mla_prefill("));
        assert!(FAST_SOURCE.contains("void glm53_dsa_sparse_mla_prefill_warp("));
        assert!(FAST_SOURCE.contains("constexpr unsigned int QUERY_TILE = 8"));
        assert!(FAST_SOURCE.contains("const unsigned int warp = threadIdx.x >> 5"));
        assert!(REGISTRY.contains("glm53_dsa_prefill_fast = \"glm53_dsa_prefill_fast\""));
    }

    #[test]
    fn tensor_core_prefill_shares_latent_tiles_across_all_tp2_heads() {
        let source =
            include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_dsa_prefill_tc.cu");
        assert!(source.contains("void glm53_dsa_prefill_tc_scores("));
        assert!(source.contains("void glm53_dsa_prefill_tc_values("));
        assert!(source.contains("wmma::mma_sync"));
        assert!(source.contains("HEADS = 32"));
        assert!(source.contains("selected[CAUSAL_WIDTH] < 0"));
        assert!(FAST_SOURCE.contains("within_causal_tile"));
        assert!(REGISTRY.contains("glm53_dsa_prefill_tc = \"glm53_dsa_prefill_tc\""));
    }

    #[test]
    fn causal_prefill_maps_each_packed_token_to_its_sequence_state() {
        assert!(SOURCE.contains("sequence_for_token("));
        assert!(SOURCE.contains("cu_seqlens[sequence + 1]"));
        assert!(SOURCE.contains("static_cast<unsigned long long>(sequence) * 5"));
        assert!(SOURCE.contains("pool_count <= TOP_POOLS"));
        assert!(SOURCE.contains("metadata[0] + index"));
    }

    #[test]
    fn invalid_prefill_contract_fails_closed() {
        let args = Glm53DsaScorePrefillArgs {
            query: DevicePtr(1),
            head_weights: DevicePtr(2),
            positions: DevicePtr(3),
            valid: DevicePtr(4),
            sequence_state_ptrs: DevicePtr(5),
            cu_seqlens: DevicePtr(6),
            scores: DevicePtr(7),
            max_pools: 0,
            total_tokens: 1,
            num_sequences: 1,
        };
        assert!(
            args.validate()
                .unwrap_err()
                .to_string()
                .contains("capacity")
        );
    }
}
