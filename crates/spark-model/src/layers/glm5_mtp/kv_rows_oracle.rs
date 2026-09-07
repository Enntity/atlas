// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off resident BF16 writer diagnostic; no private weights or GPU arena.

use super::kv_rows_plan::{DeviceSpan, KvRowsPlan};
use super::*;
use anyhow::{Context, ensure};

const ROW_BYTES: usize = 1024;
const BLOCK_ROWS: usize = 16;

pub(super) fn enabled(rows: usize) -> bool {
    (1..=4).contains(&rows)
        && std::env::var("ATLAS_GLM_MTP_KV_REPAIR_VERIFY").is_ok_and(|v| v == "1")
}

struct OracleLayout {
    actual: Vec<i64>,
    reference: Vec<i64>,
    blocks: Vec<u32>,
}

fn validate_pool_spans(k: DeviceSpan, v: DeviceSpan) -> Result<()> {
    k.end()?;
    v.end()?;
    ensure!(k.ptr.0 != 0 && v.ptr.0 != 0, "oracle cache pool is absent");
    ensure!(!k.overlaps(v)?, "oracle K/V pools overlap");
    Ok(())
}

impl OracleLayout {
    fn new(
        cache: &PagedKvCache,
        actual: &[i64],
        reference_block: u32,
        capable: bool,
    ) -> Result<Self> {
        let c = cache.config();
        ensure!(capable, "oracle requires a KV-only MLA body");
        ensure!((1..=4).contains(&actual.len()), "oracle requires 1..4 rows");
        ensure!(
            c.block_size == BLOCK_ROWS
                && c.num_layers == 1
                && c.num_kv_heads == 1
                && c.head_dim == 512
                && c.dtype == KvCacheDtype::Bf16
                && c.layer_dtypes.iter().all(|d| *d == KvCacheDtype::Bf16)
                && c.layer_dims.is_empty(),
            "oracle requires BF16 NoPE512 blocks of16"
        );
        let bytes = cache
            .num_blocks()
            .checked_mul(BLOCK_ROWS * ROW_BYTES)
            .context("oracle pool extent overflow")?;
        validate_pool_spans(
            DeviceSpan {
                ptr: cache.k_cache_ptr(0, 0),
                bytes,
            },
            DeviceSpan {
                ptr: cache.v_cache_ptr(0, 0),
                bytes,
            },
        )?;
        let mut blocks = vec![reference_block];
        let mut seen = std::collections::HashSet::new();
        for &slot in actual {
            ensure!(
                slot >= 0 && seen.insert(slot),
                "oracle repeated or negative slot"
            );
            let block = u32::try_from(slot / BLOCK_ROWS as i64).context("oracle block overflow")?;
            if !blocks.contains(&block) {
                blocks.push(block);
            }
        }
        ensure!(blocks.len() <= 3, "oracle touches more than three blocks");
        for &block in &blocks {
            ensure!(
                (block as usize) < cache.num_blocks() && cache.ref_count(block) == 1,
                "oracle requires exclusive allocated cache blocks"
            );
        }
        Ok(Self {
            actual: actual.to_vec(),
            reference: (0..actual.len())
                .map(|r| reference_block as i64 * 16 + r as i64)
                .collect(),
            blocks,
        })
    }
}

#[derive(Debug, PartialEq)]
struct BlockSnapshot {
    block: u32,
    k: Vec<u8>,
    v: Vec<u8>,
}

fn snapshot(
    cache: &PagedKvCache,
    gpu: &dyn GpuBackend,
    blocks: &[u32],
) -> Result<Vec<BlockSnapshot>> {
    blocks
        .iter()
        .map(|&block| {
            let (k, v) = cache.read_block(0, block, gpu)?;
            Ok(BlockSnapshot { block, k, v })
        })
        .collect()
}

fn restore(
    cache: &PagedKvCache,
    gpu: &dyn GpuBackend,
    original: &[BlockSnapshot],
    stream: u64,
) -> Result<()> {
    // Drain failed asynchronous work before overwriting. Attempt every side even
    // if another copy fails; a CUDA fault can still make restoration impossible.
    let mut errors = Vec::new();
    if let Err(e) = gpu.synchronize(stream) {
        errors.push(format!("synchronize: {e:#}"));
    }
    for block in original {
        for (ptr, bytes) in [
            (cache.k_cache_ptr(0, block.block), &block.k),
            (cache.v_cache_ptr(0, block.block), &block.v),
        ] {
            if let Err(e) = gpu.copy_h2d(bytes, ptr) {
                errors.push(format!("block {}: {e:#}", block.block));
            }
        }
    }
    ensure!(
        errors.is_empty(),
        "oracle restoration failed: {}",
        errors.join("; ")
    );
    Ok(())
}

fn check_guards(before: &[BlockSnapshot], after: &[BlockSnapshot], written: &[i64]) -> Result<()> {
    ensure!(before.len() == after.len(), "oracle snapshot shape changed");
    for (old, new) in before.iter().zip(after) {
        ensure!(old.block == new.block, "oracle snapshot block changed");
        for row in 0..BLOCK_ROWS {
            if written.contains(&(old.block as i64 * 16 + row as i64)) {
                continue;
            }
            let r = row * ROW_BYTES..(row + 1) * ROW_BYTES;
            ensure!(
                old.k[r.clone()] == new.k[r.clone()] && old.v[r.clone()] == new.v[r],
                "oracle untouched guard changed in block {} row {row}",
                old.block
            );
        }
    }
    Ok(())
}

fn rows(snapshot: &[BlockSnapshot], slots: &[i64]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    slots
        .iter()
        .map(|&slot| {
            let block = snapshot
                .iter()
                .find(|b| b.block as i64 == slot / 16)
                .context("oracle row missing from snapshot")?;
            let r = slot as usize % 16 * ROW_BYTES..(slot as usize % 16 + 1) * ROW_BYTES;
            Ok((block.k[r.clone()].to_vec(), block.v[r].to_vec()))
        })
        .collect()
}

fn verify(
    cache: &mut PagedKvCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    layout: &OracleLayout,
    reference: impl FnOnce(&mut PagedKvCache) -> Result<()>,
    candidate: impl FnOnce(&mut PagedKvCache) -> Result<()>,
) -> Result<()> {
    gpu.synchronize(stream)?;
    let original = snapshot(cache, gpu, &layout.blocks)?;
    let run = (|| {
        reference(cache)?;
        gpu.synchronize(stream)?;
        let reference_result = snapshot(cache, gpu, &layout.blocks)?;
        check_guards(&original, &reference_result, &layout.reference)?;
        let expected = rows(&reference_result, &layout.reference)?;
        for (k, v) in &expected {
            ensure!(
                k.chunks_exact(2)
                    .chain(v.chunks_exact(2))
                    .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
                "oracle reference contains non-finite BF16"
            );
        }
        restore(cache, gpu, &original, stream)?;
        candidate(cache)?;
        gpu.synchronize(stream)?;
        let actual = snapshot(cache, gpu, &layout.blocks)?;
        check_guards(&original, &actual, &layout.actual)?;
        ensure!(
            rows(&actual, &layout.actual)? == expected,
            "GLM KV resident BF16 oracle mismatch"
        );
        Ok(())
    })();
    if let Err(error) = run {
        return match restore(cache, gpu, &original, stream) {
            Ok(()) => Err(error),
            Err(restore_error) => {
                Err(error.context(format!("{restore_error:#}; GPU state must not resume")))
            }
        };
    }
    tracing::info!(
        rows = layout.actual.len(),
        blocks = layout.blocks.len(),
        "GLM MTP resident BF16 KV oracle passed (diagnostic, not timing)"
    );
    Ok(())
}

impl Glm5MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_kv_rows(
        &self,
        tokens: &[u32],
        plan: &KvRowsPlan,
        source: DeviceSpan,
        blocks: &[u32],
        cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            !ctx.graph_capture && !ctx.gpu.stream_is_capturing(stream),
            "GLM KV oracle cannot run during stream capture"
        );
        let reference_block = *blocks.first().context("oracle empty block table")?;
        let layout = OracleLayout::new(
            cache,
            &plan.slots,
            reference_block,
            self.module.body.supports_mla_kv_only(),
        )?;
        // Both plans validated before snapshots or any reference overwrite.
        let forbidden = self.validate_kv_inputs(tokens, source, ctx, cache)?;
        KvRowsPlan::new(
            tokens,
            source,
            ctx.config.hidden_size,
            ctx.config.vocab_size,
            0,
            cache.block_size(),
            blocks,
            cache.num_blocks(),
            ctx.buffers.max_batch_tokens(),
            ctx.buffers.scratch_bytes(),
            &forbidden,
        )?;
        verify(
            cache,
            ctx.gpu,
            stream,
            &layout,
            |cache| self.oracle_reference_prefix(tokens, source, blocks, cache, ctx, stream),
            |cache| self.execute_kv_rows(plan, source, cache, ctx, stream),
        )
    }
}

impl Glm5MtpHead {
    // Frozen BF16 pre-extraction chain: independent embedding and slot addressing.
    // Keep dispatch/reduction order identical; this is not the NVFP4 proposer.
    #[allow(clippy::too_many_arguments)]
    fn oracle_reference_prefix(
        &self,
        tokens: &[u32],
        source: DeviceSpan,
        blocks: &[u32],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let rows_total = tokens.len();
        let chunk_rows = ctx.buffers.max_batch_tokens();
        let mut done = 0usize;
        while done < rows_total {
            let n = (rows_total - done).min(chunk_rows);
            let embed = ctx.buffers.ssm_deinterleaved();
            for i in 0..n {
                ctx.gpu.copy_d2d_async(
                    self.embed_tokens
                        .weight
                        .offset(tokens[done + i] as usize * h * bf16),
                    embed.offset(i * h * bf16),
                    h * bf16,
                    stream,
                )?;
            }

            let normed_embed = ctx.buffers.attn_output();
            let normed_hidden = ctx.buffers.residual();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                embed,
                &self.module.enorm,
                normed_embed,
                n as u32,
                h as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                source.ptr.offset(done * h * bf16),
                &self.module.hnorm,
                normed_hidden,
                n as u32,
                h as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;

            let eh_input = ctx.buffers.ssm_qkvz();
            for row in 0..n {
                ops::bf16_concat(
                    ctx.gpu,
                    self.bf16_concat_k,
                    normed_embed.offset(row * h * bf16),
                    normed_hidden.offset(row * h * bf16),
                    eh_input.offset(row * 2 * h * bf16),
                    h as u32,
                    stream,
                )?;
            }
            let h_in = ctx.buffers.hidden_states();
            if ctx.dispatch.cublas_gemm && n > 1 {
                ops::cublas_bf16_proj_dense(
                    eh_input,
                    self.module.eh_proj.weight,
                    h_in,
                    n as u32,
                    h as u32,
                    (2 * h) as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    eh_input,
                    &self.module.eh_proj,
                    h_in,
                    n as u32,
                    h as u32,
                    (2 * h) as u32,
                    stream,
                )?;
            }

            let slots_dev = ctx.buffers.scratch().offset(MTP_META_OFFSET);
            let slots: Vec<i64> = (done..done + n)
                .map(|row| {
                    (blocks[row / kv_cache.block_size()] as usize * kv_cache.block_size()
                        + row % kv_cache.block_size()) as i64
                })
                .collect();
            let slot_bytes =
                unsafe { std::slice::from_raw_parts(slots.as_ptr().cast::<u8>(), slots.len() * 8) };
            if let Err(error) = ctx.gpu.copy_h2d_async(slot_bytes, slots_dev, stream) {
                let _ = ctx.gpu.synchronize(stream);
                return Err(error);
            }

            let mtp_ctx = ForwardContext {
                ssm_batch: None,
                buffers: ctx.buffers,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: None,
                profile: ctx.profile,
                comm: None,
                graph_capture: false,
                gdn_exact_replay: false,
                token_ids: ctx.token_ids,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            };
            let result = self
                .module
                .body
                .prefill_mla_kv_only(h_in, n, kv_cache, slots_dev, &mtp_ctx, stream);
            // This reference owns pageable slot storage; keep it alive until
            // upload/write completion, including the error path.
            let sync = ctx.gpu.synchronize(stream);
            ensure!(
                result?,
                "GLM MTP appended layer does not support MLA KV-only prefill"
            );
            sync?;
            done += n;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "kv_rows_oracle_tests.rs"]
mod tests;
