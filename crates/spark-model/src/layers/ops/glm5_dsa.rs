// SPDX-License-Identifier: AGPL-3.0-only

//! Launch contracts for GLM-5.3-Flash's learned sparse-index core.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_DSA_MODULE: &str = "glm53_dsa";
pub const GLM53_DSA_POOL_ENTRY: &str = "glm53_dsa_pool_append";
pub const GLM53_DSA_SCORE_ENTRY: &str = "glm53_dsa_score";
pub const GLM53_DSA_TOPK_ENTRY: &str = "glm53_dsa_topk_expand_decode";
pub const GLM53_DSA_SPARSE_MLA_ENTRY: &str = "glm53_dsa_sparse_mla_decode";
pub const GLM53_DSA_INDEX_DIM: u32 = 128;
pub const GLM53_DSA_POOL_THREADS: u32 = 128;
pub const GLM53_DSA_SCORE_THREADS: u32 = 128;
pub const GLM53_DSA_TOPK_THREADS: u32 = 256;
pub const GLM53_DSA_OUTPUT_WIDTH: u32 = 2051;
pub const GLM53_DSA_MLA_DIM: u32 = 512;

fn ensure_pointers(pointers: &[(&str, DevicePtr)]) -> Result<()> {
    for (name, pointer) in pointers {
        ensure!(!pointer.is_null(), "GLM DSA {name} pointer is null");
    }
    Ok(())
}

/// Packed key/gate chunks and per-sequence semantic-index state.
///
/// `sequence_state_ptrs` is device `u64[num_sequences,5]` in
/// latent-cache/pooled-key/tail-key/tail-gate/metadata order. Metadata is
/// device `i32[4]`: first-valid, valid-count, complete-pool-count, tail-length.
pub struct Glm53DsaPoolAppendArgs {
    pub keys: DevicePtr,
    pub gates: DevicePtr,
    pub ape: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub positions: DevicePtr,
    pub valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub num_sequences: u32,
}

impl Glm53DsaPoolAppendArgs {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_sequences > 0,
            "GLM DSA pool append requires sequences"
        );
        ensure_pointers(&[
            ("keys", self.keys),
            ("gates", self.gates),
            ("ape", self.ape),
            ("cu_seqlens", self.cu_seqlens),
            ("positions", self.positions),
            ("valid", self.valid),
            ("sequence_state_ptrs", self.sequence_state_ptrs),
        ])
    }
}

pub fn glm53_dsa_pool_append(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaPoolAppendArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_sequences, 1, 1])
        .block([GLM53_DSA_POOL_THREADS, 1, 1])
        .arg_ptr(args.keys)
        .arg_ptr(args.gates)
        .arg_ptr(args.ape)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.positions)
        .arg_ptr(args.valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_u32(args.num_sequences)
        .launch(stream)
}

pub struct Glm53DsaScoreArgs {
    pub query: DevicePtr,
    /// BF16 output of `indexer.weights_proj`; the CUDA kernel promotes it to
    /// FP32 and applies the official `1/sqrt(index_heads)` factor.
    pub head_weights: DevicePtr,
    pub query_valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub scores: DevicePtr,
    pub max_pools: u32,
    pub num_sequences: u32,
}

impl Glm53DsaScoreArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_sequences > 0, "GLM DSA scoring requires sequences");
        ensure!(self.max_pools > 0, "GLM DSA scoring requires pool capacity");
        ensure_pointers(&[
            ("query", self.query),
            ("head_weights", self.head_weights),
            ("query_valid", self.query_valid),
            ("sequence_state_ptrs", self.sequence_state_ptrs),
            ("scores", self.scores),
        ])
    }
}

pub fn glm53_dsa_score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaScoreArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.max_pools, args.num_sequences, 1])
        .block([GLM53_DSA_SCORE_THREADS, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.head_weights)
        .arg_ptr(args.query_valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.scores)
        .arg_u32(args.max_pools)
        .arg_u32(args.num_sequences)
        .launch(stream)
}

pub struct Glm53DsaTopkArgs {
    pub scores: DevicePtr,
    pub query_valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub output: DevicePtr,
    pub max_pools: u32,
    pub num_sequences: u32,
}

impl Glm53DsaTopkArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_sequences > 0, "GLM DSA top-k requires sequences");
        ensure!(self.max_pools > 0, "GLM DSA top-k requires pool capacity");
        ensure_pointers(&[
            ("scores", self.scores),
            ("query_valid", self.query_valid),
            ("sequence_state_ptrs", self.sequence_state_ptrs),
            ("output", self.output),
        ])
    }
}

pub fn glm53_dsa_topk_expand_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaTopkArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_sequences, 1, 1])
        .block([GLM53_DSA_TOPK_THREADS, 1, 1])
        .arg_ptr(args.scores)
        .arg_ptr(args.query_valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.output)
        .arg_u32(args.max_pools)
        .arg_u32(args.num_sequences)
        .launch(stream)
}

pub struct Glm53DsaSparseMlaArgs {
    pub absorbed_query: DevicePtr,
    pub selected_indices: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub output: DevicePtr,
    pub num_heads: u32,
    pub num_sequences: u32,
    pub attention_scale: f32,
}

impl Glm53DsaSparseMlaArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_heads > 0, "GLM sparse MLA requires heads");
        ensure!(self.num_sequences > 0, "GLM sparse MLA requires sequences");
        ensure!(
            self.attention_scale.is_finite() && self.attention_scale > 0.0,
            "GLM sparse MLA scale must be finite and positive"
        );
        ensure_pointers(&[
            ("absorbed_query", self.absorbed_query),
            ("selected_indices", self.selected_indices),
            ("sequence_state_ptrs", self.sequence_state_ptrs),
            ("output", self.output),
        ])
    }
}

pub fn glm53_dsa_sparse_mla_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaSparseMlaArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, args.num_sequences, 1])
        .block([GLM53_DSA_TOPK_THREADS, 1, 1])
        .arg_ptr(args.absorbed_query)
        .arg_ptr(args.selected_indices)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.output)
        .arg_u32(args.num_heads)
        .arg_u32(args.num_sequences)
        .arg_f32(args.attention_scale)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_dsa_index.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    fn pool_args() -> Glm53DsaPoolAppendArgs {
        Glm53DsaPoolAppendArgs {
            keys: DevicePtr(1),
            gates: DevicePtr(2),
            ape: DevicePtr(3),
            cu_seqlens: DevicePtr(4),
            positions: DevicePtr(5),
            valid: DevicePtr(6),
            sequence_state_ptrs: DevicePtr(7),
            num_sequences: 2,
        }
    }

    fn score_args() -> Glm53DsaScoreArgs {
        Glm53DsaScoreArgs {
            query: DevicePtr(1),
            head_weights: DevicePtr(2),
            query_valid: DevicePtr(3),
            sequence_state_ptrs: DevicePtr(4),
            scores: DevicePtr(5),
            max_pools: 8192,
            num_sequences: 2,
        }
    }

    #[test]
    fn official_geometry_and_entrypoints_are_pinned() {
        assert_eq!(GLM53_DSA_INDEX_DIM, 128);
        assert_eq!(GLM53_DSA_OUTPUT_WIDTH, 2051);
        for entry in [
            GLM53_DSA_POOL_ENTRY,
            GLM53_DSA_SCORE_ENTRY,
            GLM53_DSA_TOPK_ENTRY,
            GLM53_DSA_SPARSE_MLA_ENTRY,
        ] {
            assert!(SOURCE.contains(&format!("void {entry}(")));
        }
        assert!(REGISTRY.contains("glm53_dsa_index = \"glm53_dsa\""));
        assert!(SOURCE.contains("pool_count <= TOP_POOLS"));
        assert!(SOURCE.contains("metadata[0] + index"));
        assert!(SOURCE.contains("rank < selected_count"));
        assert!(SOURCE.contains("metadata[2]), TOP_POOLS) * KPOOL"));
    }

    #[test]
    fn launch_contracts_fail_closed() {
        pool_args().validate().unwrap();
        score_args().validate().unwrap();

        let mut pool = pool_args();
        pool.sequence_state_ptrs = DevicePtr::NULL;
        assert!(
            pool.validate()
                .unwrap_err()
                .to_string()
                .contains("state_ptrs")
        );

        let mut score = score_args();
        score.max_pools = 0;
        assert!(
            score
                .validate()
                .unwrap_err()
                .to_string()
                .contains("capacity")
        );
    }
}
