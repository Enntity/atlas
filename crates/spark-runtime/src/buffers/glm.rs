// SPDX-License-Identifier: AGPL-3.0-only

//! One shared transient workspace for GLM-5.3 KDA and sparse MLA.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;

const KDA_CHUNK: usize = 16;
const KDA_DIM: usize = 128;
/// Maximum target width reserved by the fixed GLM DFlash2 verifier.
pub const GLM53_VERIFY_MAX_ROWS: usize = 17;
/// The appliance admits four live GLM streams. DFlash verification stacks
/// each stream's causal block into one expert-weight sweep.
pub const GLM53_VERIFY_MAX_SEQS: usize = 4;
pub const GLM53_VERIFY_MAX_BATCH_ROWS: usize = GLM53_VERIFY_MAX_ROWS * GLM53_VERIFY_MAX_SEQS;
const GLM53_KDA_STATE_WORDS: usize = 4;
const DSA_OUTPUT_WIDTH: usize = 2051;
const DSA_HEADS: usize = 32;
const DSA_TC_WIDTH: usize = 2064;
// Match the two-Spark appliance's 7,168-row physical arena. A large prompt can
// then route once for the whole slab, making >128-row experts visible to the
// fat EXL3 GEMM instead of hiding them behind repeated 512-row windows.
const EXL3_ROWS: usize = 7168;
// Decode uses four expert groups (one per expected local EP2 route). Prompt
// cohorts touch most local experts, so they use eight groups. Allocate once
// for the larger prompt geometry; the hot-path launcher selects 4 or 8.
const EXL3_CONCURRENCY: usize = 8;
const EXL3_HIDDEN: usize = 4096;
const EXL3_INTERMEDIATE: usize = 2048;
const EXL3_EXPERTS: usize = 288;
const EXL3_TOPK: usize = 8;
const EXL3_LOCK_INTS: usize = 1024 * 1024 + 2 * 1024;
pub const GLM53_EXL3_LOCK_BYTES: usize = EXL3_LOCK_INTS * 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmWorkspaceLayout {
    pub state_ptrs: usize,
    pub state_ptrs_stride: usize,
    pub state_ptrs_layers: usize,
    pub cu_seqlens_i32: usize,
    pub cu_seqlens_i64: usize,
    pub positions: usize,
    pub valid: usize,
    pub state_slot_ids: usize,
    pub dynamic: usize,
    pub mhc_sqsum: usize,
    pub mhc_mix: usize,
    pub mhc_bytes: usize,
    pub dsa_scores: usize,
    pub dsa_selected: usize,
    pub dsa_attention_scores: usize,
    pub dsa_attention_weights: usize,
    pub exl3_hidden_fp16: usize,
    pub exl3_output_fp32: usize,
    pub exl3_temp_state_g: usize,
    pub exl3_temp_state_u: usize,
    pub exl3_temp_intermediate_g: usize,
    pub exl3_temp_intermediate_u: usize,
    pub exl3_expert_count: usize,
    pub exl3_fat_descriptors: usize,
    pub exl3_token_sorted: usize,
    pub exl3_weight_sorted: usize,
    pub exl3_locks: usize,
    pub exl3_bytes: usize,
    pub flash_kda_bytes: usize,
    pub dsa_max_pools: usize,
    pub latent_capacity: usize,
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub total_bytes: usize,
}

impl GlmWorkspaceLayout {
    pub fn from_config(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        max_batch_size: usize,
    ) -> Result<Self> {
        if config.model_type != "glm5_next" {
            return Ok(Self::default());
        }
        let mut cursor = 0usize;
        // One stable device pointer table per transformer layer. Decode CUDA
        // graphs dereference these tables at replay time; sharing a single
        // table across layers would leave every captured layer reading the
        // final layer's state. The tables are tiny (45 * 4 * 5 * 8 = 7 KiB
        // for GLM-5.3 EP2) and remove transient host pointers from capture.
        // Decode batches need five pointers per live sequence. Verification
        // instead needs one KDA current image plus K-1 rollback images. The
        // fixed DFlash2 engine verifies at most 17 rows, so reserve the larger
        // table once per layer and keep every graph-visible address stable.
        let decode_state_words = product(&[max_batch_size, 5])?;
        let verify_state_words = product(&[GLM53_VERIFY_MAX_BATCH_ROWS, GLM53_KDA_STATE_WORDS])?;
        let state_ptrs_stride = product(&[decode_state_words.max(verify_state_words), 8])?;
        let state_ptrs_layers = config.num_hidden_layers;
        let state_ptrs = take(
            &mut cursor,
            product(&[state_ptrs_layers, state_ptrs_stride])?,
            8,
        )?;
        let cu_seqlens_i32 = take(&mut cursor, product(&[max_batch_size + 1, 4])?, 4)?;
        let cu_seqlens_i64 = take(&mut cursor, product(&[max_batch_size + 1, 8])?, 8)?;
        let positions = take(&mut cursor, product(&[max_batch_tokens, 4])?, 4)?;
        let valid = take(&mut cursor, max_batch_tokens, 1)?;
        let state_slot_ids = take(&mut cursor, product(&[max_batch_size, 4])?, 4)?;
        let dynamic = align(cursor, 256)?;

        // Tensor-core mHC pre emits one RMS sum and 24 learned mix logits per
        // row. They live only until hc_pre_finish, before either attention or
        // FFN uses the dynamic workspace, so overlay them on that arena.
        let mut mhc_cursor = dynamic;
        let mhc_sqsum = take(&mut mhc_cursor, product(&[max_batch_tokens, 4])?, 256)?;
        let mhc_mix = take(
            &mut mhc_cursor,
            product(&[max_batch_tokens, (2 + config.hc_mult) * config.hc_mult, 4])?,
            256,
        )?;
        let mhc_bytes = mhc_cursor
            .checked_sub(dynamic)
            .context("GLM mHC workspace offset underflow")?;

        let max_pools = max_seq_len.div_ceil(config.index_kpool).max(1);
        // Decode needs one row per sequence; causal DSA prefill needs one row
        // per prompt token so all queries can be scored/top-k'd in parallel.
        // This remains smaller than FlashKDA's workspace at the 2k-token GB10
        // envelope, so the shared dynamic region does not grow in practice.
        let dsa_scores_bytes = product(&[max_batch_tokens, max_pools, 4])?;
        let dsa_selected_bytes = product(&[max_batch_tokens, DSA_OUTPUT_WIDTH, 4])?;
        // The selected rows must survive the indexer score pass and the
        // attention pass. Put them first, then reuse the remaining arena:
        // indexer FP32 scores are dead before tensor-core attention writes its
        // larger QK scores and normalized BF16 weights.
        let dsa_selected = dynamic;
        let dsa_scores = dsa_selected
            .checked_add(dsa_selected_bytes)
            .context("GLM DSA workspace offset overflow")?;
        let dsa_attention_scores = dsa_scores;
        let dsa_attention_score_bytes = product(&[max_batch_tokens, DSA_HEADS, DSA_TC_WIDTH, 4])?;
        let dsa_attention_weights = dsa_attention_scores
            .checked_add(dsa_attention_score_bytes)
            .context("GLM DSA attention workspace offset overflow")?;
        let dsa_attention_weight_bytes = product(&[max_batch_tokens, DSA_HEADS, DSA_TC_WIDTH, 2])?;
        let index_bytes = dsa_selected_bytes
            .checked_add(dsa_scores_bytes)
            .context("GLM DSA index workspace size overflow")?;
        let attention_bytes = dsa_selected_bytes
            .checked_add(dsa_attention_score_bytes)
            .and_then(|value| value.checked_add(dsa_attention_weight_bytes))
            .context("GLM DSA attention workspace size overflow")?;
        let dsa_bytes = index_bytes.max(attention_bytes);

        // EXL3 runs the routed experts in <=7,168-token windows. The scratch is
        // shared by every MoE layer and reuses the KDA/DSA dynamic region: the
        // attention phase has completed before the FFN phase begins.
        let mut exl3_cursor = dynamic;
        let exl3_hidden_fp16 = take(
            &mut exl3_cursor,
            product(&[EXL3_ROWS, EXL3_HIDDEN, 2])?,
            256,
        )?;
        let exl3_output_fp32 = take(
            &mut exl3_cursor,
            product(&[EXL3_ROWS, EXL3_HIDDEN, 4])?,
            256,
        )?;
        let state_bytes = product(&[EXL3_CONCURRENCY, EXL3_ROWS, EXL3_HIDDEN, 2])?;
        let intermediate_bytes = product(&[EXL3_CONCURRENCY, EXL3_ROWS, EXL3_INTERMEDIATE, 2])?;
        let exl3_temp_state_g = take(&mut exl3_cursor, state_bytes, 256)?;
        let exl3_temp_state_u = take(&mut exl3_cursor, state_bytes, 256)?;
        let exl3_temp_intermediate_g = take(&mut exl3_cursor, intermediate_bytes, 256)?;
        let exl3_temp_intermediate_u = take(&mut exl3_cursor, intermediate_bytes, 256)?;
        let exl3_expert_count = take(&mut exl3_cursor, (EXL3_EXPERTS + 1) * 8, 8)?;
        let exl3_fat_descriptors = take(&mut exl3_cursor, EXL3_EXPERTS * 4 * 4, 16)?;
        let exl3_token_sorted = take(&mut exl3_cursor, EXL3_ROWS * EXL3_TOPK * 8, 8)?;
        let exl3_weight_sorted = take(&mut exl3_cursor, EXL3_ROWS * EXL3_TOPK * 2, 2)?;
        let exl3_locks = take(&mut exl3_cursor, GLM53_EXL3_LOCK_BYTES, 256)?;
        let exl3_bytes = exl3_cursor
            .checked_sub(dynamic)
            .context("GLM EXL3 workspace offset underflow")?;

        let total_tiles = max_batch_tokens.div_ceil(KDA_CHUNK) + max_batch_size;
        let per_tile = 3 * KDA_CHUNK * KDA_DIM * 2 + KDA_DIM * 4 + 2 * KDA_CHUNK * KDA_CHUNK * 2;
        let prefix = align(product(&[max_batch_size + 1, 4])?, 128)?;
        let flash_kda_bytes = product(&[config.linear_num_key_heads, total_tiles, per_tile])?
            .checked_add(prefix)
            .context("GLM FlashKDA workspace size overflow")?;
        let total_bytes = dynamic
            .checked_add(
                dsa_bytes
                    .max(flash_kda_bytes)
                    .max(exl3_bytes)
                    .max(mhc_bytes),
            )
            .context("GLM workspace size overflow")?;
        Ok(Self {
            state_ptrs,
            state_ptrs_stride,
            state_ptrs_layers,
            cu_seqlens_i32,
            cu_seqlens_i64,
            positions,
            valid,
            state_slot_ids,
            dynamic,
            mhc_sqsum,
            mhc_mix,
            mhc_bytes,
            dsa_scores,
            dsa_selected,
            dsa_attention_scores,
            dsa_attention_weights,
            exl3_hidden_fp16,
            exl3_output_fp32,
            exl3_temp_state_g,
            exl3_temp_state_u,
            exl3_temp_intermediate_g,
            exl3_temp_intermediate_u,
            exl3_expert_count,
            exl3_fat_descriptors,
            exl3_token_sorted,
            exl3_weight_sorted,
            exl3_locks,
            exl3_bytes,
            flash_kda_bytes,
            dsa_max_pools: max_pools,
            latent_capacity: max_seq_len,
            max_batch_size,
            max_batch_tokens,
            total_bytes,
        })
    }

    pub fn state_table_offset(&self, layer_idx: usize) -> Result<usize> {
        ensure!(
            layer_idx < self.state_ptrs_layers,
            "GLM state-table layer {layer_idx} exceeds {} layers",
            self.state_ptrs_layers
        );
        self.state_ptrs
            .checked_add(layer_idx * self.state_ptrs_stride)
            .context("GLM state-table offset overflow")
    }
}

fn take(cursor: &mut usize, bytes: usize, alignment: usize) -> Result<usize> {
    let offset = align(*cursor, alignment)?;
    *cursor = offset
        .checked_add(bytes)
        .context("GLM workspace offset overflow")?;
    Ok(offset)
}

fn align(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|v| v / alignment * alignment)
        .context("GLM workspace alignment overflow")
}

fn product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .context("GLM workspace size overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_glm_has_no_workspace() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        assert_eq!(
            GlmWorkspaceLayout::from_config(&config, 2048, 32768, 8).unwrap(),
            GlmWorkspaceLayout::default()
        );
    }

    #[test]
    fn glm_layout_separates_metadata_and_reuses_dynamic_region() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".to_string();
        config.num_hidden_layers = 45;
        config.linear_num_key_heads = 64;
        config.index_kpool = 4;
        config.hc_mult = 4;
        let layout = GlmWorkspaceLayout::from_config(&config, 2048, 32768, 8).unwrap();
        assert!(layout.dynamic > layout.state_slot_ids);
        assert_eq!(layout.state_ptrs_layers, config.num_hidden_layers);
        assert_eq!(
            layout.state_ptrs_stride,
            GLM53_VERIFY_MAX_BATCH_ROWS * 4 * 8
        );
        let dsa_verify_words = GLM53_VERIFY_MAX_SEQS * (5 + (GLM53_VERIFY_MAX_ROWS - 1) * 3);
        assert!(dsa_verify_words * size_of::<u64>() <= layout.state_ptrs_stride);
        assert!(layout.state_table_offset(44).unwrap() < layout.cu_seqlens_i32);
        assert!(layout.state_table_offset(45).is_err());
        assert_eq!(layout.dsa_selected, layout.dynamic);
        assert_eq!(layout.mhc_sqsum, layout.dynamic);
        assert!(layout.mhc_mix > layout.mhc_sqsum);
        assert!(layout.mhc_bytes >= 2048 * (1 + 24) * 4);
        assert!(layout.dsa_scores > layout.dsa_selected);
        assert_eq!(layout.dsa_attention_scores, layout.dsa_scores);
        assert!(layout.dsa_attention_weights > layout.dsa_attention_scores);
        assert!(layout.flash_kda_bytes > 100 * 1024 * 1024);
        assert!(layout.exl3_bytes > 16 * 1024 * 1024);
        assert!(layout.exl3_bytes > 1024 * 1024 * 1024);
        assert_eq!(layout.exl3_hidden_fp16, layout.dynamic);
        assert!(layout.total_bytes >= layout.dynamic + layout.flash_kda_bytes);
    }
}
