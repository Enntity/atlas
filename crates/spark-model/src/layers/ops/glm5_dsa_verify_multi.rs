// SPDX-License-Identifier: AGPL-3.0-only

//! Launch contracts for depth-major multi-sequence GLM DSA verification.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_DSA_VERIFY_MULTI_MODULE: &str = "glm53_dsa_verify_multi";
pub const GLM53_DSA_VERIFY_MULTI_LATENT_ENTRY: &str = "glm53_dsa_verify_multi_latent_append";
pub const GLM53_DSA_VERIFY_MULTI_POOL_ENTRY: &str = "glm53_dsa_verify_multi_pool_append";
pub const GLM53_DSA_VERIFY_MULTI_SCORE_ENTRY: &str = "glm53_dsa_verify_multi_score";
pub const GLM53_DSA_VERIFY_MULTI_SPARSE_ENTRY: &str = "glm53_dsa_verify_multi_sparse_mla";
pub const GLM53_DSA_VERIFY_MULTI_SNAPSHOT_ENTRY: &str = "glm53_dsa_verify_multi_snapshot_tail";

fn pointers(values: &[(&str, DevicePtr)]) -> Result<()> {
    for (name, pointer) in values {
        ensure!(!pointer.is_null(), "GLM DSA multi-verify {name} is null");
    }
    Ok(())
}

fn shape(rows_per_sequence: u32, num_sequences: u32, token_depth: u32) -> Result<()> {
    ensure!(rows_per_sequence > 0, "GLM DSA multi-verify needs rows");
    ensure!(num_sequences > 1, "GLM DSA multi-verify needs concurrency");
    ensure!(
        token_depth < rows_per_sequence,
        "GLM DSA multi-verify depth exceeds row width"
    );
    Ok(())
}

pub struct Glm53DsaVerifyMultiLatentArgs {
    pub latent: DevicePtr,
    pub positions: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub rows_per_sequence: u32,
    pub num_sequences: u32,
    pub latent_capacity: u32,
}

pub fn glm53_dsa_verify_multi_latent_append(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaVerifyMultiLatentArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        args.rows_per_sequence > 0,
        "GLM DSA multi-verify needs rows"
    );
    ensure!(
        args.num_sequences > 1,
        "GLM DSA multi-verify needs concurrency"
    );
    ensure!(args.latent_capacity > 0, "GLM DSA latent capacity is zero");
    pointers(&[
        ("latent", args.latent),
        ("positions", args.positions),
        ("state table", args.sequence_state_ptrs),
    ])?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_sequences * args.rows_per_sequence, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(args.latent)
        .arg_ptr(args.positions)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_u32(args.rows_per_sequence)
        .arg_u32(args.num_sequences)
        .arg_u32(args.latent_capacity)
        .launch(stream)
}

pub struct Glm53DsaVerifyMultiPoolArgs {
    pub keys: DevicePtr,
    pub gates: DevicePtr,
    pub ape: DevicePtr,
    pub positions: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub rows_per_sequence: u32,
    pub num_sequences: u32,
    pub token_depth: u32,
}

pub fn glm53_dsa_verify_multi_pool_append(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaVerifyMultiPoolArgs,
    stream: u64,
) -> Result<()> {
    shape(args.rows_per_sequence, args.num_sequences, args.token_depth)?;
    pointers(&[
        ("keys", args.keys),
        ("gates", args.gates),
        ("APE", args.ape),
        ("positions", args.positions),
        ("state table", args.sequence_state_ptrs),
    ])?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_sequences, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(args.keys)
        .arg_ptr(args.gates)
        .arg_ptr(args.ape)
        .arg_ptr(args.positions)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_u32(args.rows_per_sequence)
        .arg_u32(args.num_sequences)
        .arg_u32(args.token_depth)
        .launch(stream)
}

pub struct Glm53DsaVerifyMultiScoreArgs {
    pub query: DevicePtr,
    pub head_weights: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub scores: DevicePtr,
    pub max_pools: u32,
    pub rows_per_sequence: u32,
    pub num_sequences: u32,
    pub token_depth: u32,
}

pub fn glm53_dsa_verify_multi_score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaVerifyMultiScoreArgs,
    stream: u64,
) -> Result<()> {
    shape(args.rows_per_sequence, args.num_sequences, args.token_depth)?;
    ensure!(args.max_pools > 0, "GLM DSA multi-score needs pools");
    pointers(&[
        ("query", args.query),
        ("head weights", args.head_weights),
        ("state table", args.sequence_state_ptrs),
        ("scores", args.scores),
    ])?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.max_pools, args.num_sequences, 1])
        .block([128, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.head_weights)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.scores)
        .arg_u32(args.max_pools)
        .arg_u32(args.rows_per_sequence)
        .arg_u32(args.num_sequences)
        .arg_u32(args.token_depth)
        .launch(stream)
}

pub struct Glm53DsaVerifyMultiSparseArgs {
    pub absorbed_query: DevicePtr,
    pub selected_indices: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub output: DevicePtr,
    pub num_heads: u32,
    pub rows_per_sequence: u32,
    pub num_sequences: u32,
    pub token_depth: u32,
    pub attention_scale: f32,
    pub direct_selection: bool,
}

pub fn glm53_dsa_verify_multi_sparse_mla(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaVerifyMultiSparseArgs,
    stream: u64,
) -> Result<()> {
    shape(args.rows_per_sequence, args.num_sequences, args.token_depth)?;
    ensure!(args.num_heads > 0, "GLM sparse MLA needs heads");
    ensure!(
        args.attention_scale.is_finite() && args.attention_scale > 0.0,
        "GLM sparse MLA scale must be finite and positive"
    );
    pointers(&[
        ("absorbed query", args.absorbed_query),
        ("selected indices", args.selected_indices),
        ("state table", args.sequence_state_ptrs),
        ("output", args.output),
    ])?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, args.num_sequences, 1])
        .block([256, 1, 1])
        .arg_ptr(args.absorbed_query)
        .arg_ptr(args.selected_indices)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.output)
        .arg_u32(args.num_heads)
        .arg_u32(args.rows_per_sequence)
        .arg_u32(args.num_sequences)
        .arg_u32(args.token_depth)
        .arg_f32(args.attention_scale)
        .arg_u32(u32::from(args.direct_selection))
        .launch(stream)
}

pub struct Glm53DsaVerifyMultiSnapshotArgs {
    pub state_ptrs: DevicePtr,
    pub num_sequences: u32,
    pub snapshot_depth: u32,
    pub tail_elements: u32,
}

pub fn glm53_dsa_verify_multi_snapshot_tail(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaVerifyMultiSnapshotArgs,
    stream: u64,
) -> Result<()> {
    ensure!(args.num_sequences > 1, "GLM DSA snapshot needs concurrency");
    ensure!(args.tail_elements > 0, "GLM DSA snapshot tail is empty");
    pointers(&[("snapshot state table", args.state_ptrs)])?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_sequences, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(args.state_ptrs)
        .arg_u32(args.num_sequences)
        .arg_u32(args.snapshot_depth)
        .arg_u32(args.tail_elements)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_dsa_verify_multi.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    #[test]
    fn fixed_depth_major_contract_is_registered() {
        for entry in [
            GLM53_DSA_VERIFY_MULTI_LATENT_ENTRY,
            GLM53_DSA_VERIFY_MULTI_POOL_ENTRY,
            GLM53_DSA_VERIFY_MULTI_SCORE_ENTRY,
            GLM53_DSA_VERIFY_MULTI_SPARSE_ENTRY,
            GLM53_DSA_VERIFY_MULTI_SNAPSHOT_ENTRY,
        ] {
            assert!(SOURCE.contains(&format!("void {entry}(")));
        }
        assert!(SOURCE.contains("sequence * rows_per_sequence + token_depth"));
        assert!(SOURCE.contains("rank < selected_count"));
        assert!(SOURCE.contains("metadata[2]), TOP_POOLS) * KPOOL"));
        assert!(REGISTRY.contains("glm53_dsa_verify_multi = \"glm53_dsa_verify_multi\""));
    }

    #[test]
    fn invalid_depth_is_rejected_before_launch() {
        assert!(shape(8, 4, 7).is_ok());
        assert!(shape(8, 4, 8).is_err());
        assert!(shape(8, 1, 0).is_err());
    }
}
