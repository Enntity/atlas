// SPDX-License-Identifier: AGPL-3.0-only
//! Native kernel oracle: rejected future rows stay invisible until overwritten.
//! No checkpoint required. Run only on an explicitly available CUDA GPU.
#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn bf16(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}
struct Pool {
    tail: DevicePtr,
    keys: DevicePtr,
    blocks: usize,
}
fn pool(gpu: &dyn GpuBackend, blocks: usize) -> Result<Pool> {
    let tail = gpu.alloc(blocks * 8192)?;
    let keys = gpu.alloc(blocks * 1024)?;
    gpu.memset(tail, 0, blocks * 8192)?;
    gpu.memset(keys, 0, blocks * 1024)?;
    Ok(Pool { tail, keys, blocks })
}
fn write(
    gpu: &dyn GpuBackend,
    pool: &Pool,
    ape: DevicePtr,
    blocks: usize,
    start: usize,
    rows: usize,
    poison_from: Option<usize>,
) -> Result<()> {
    let stream = gpu.default_stream();
    let mut keys = vec![];
    let mut slots = vec![];
    for pos in start..start + rows {
        // Each four-token pool gets a distinct exactly representable BF16
        // value. Equal keys/gates within a pool make the weighted mean exact;
        // positive uniform queries produce strictly increasing pooled scores.
        // This avoids the kernel's explicitly permitted threshold-tie choice.
        let bits = if poison_from.is_some_and(|p| pos >= p) {
            0x6000u16 // Greater than every canonical score, still finite.
        } else {
            0x2800u16 + (pos / 4) as u16
        };
        for _ in 0..128 {
            keys.extend(bits.to_le_bytes());
        }
        // Reverse physical block order catches accidental logical addressing.
        let physical = blocks - 1 - pos / 16;
        slots.extend(((physical * 16 + pos % 16) as i64).to_le_bytes());
    }
    let key = upload(gpu, &keys)?;
    let gate = gpu.alloc(rows * 128 * 2)?;
    gpu.memset(gate, 0, rows * 128 * 2)?;
    let slot = upload(gpu, &slots)?;
    ops::glm_index_tail_write(
        gpu,
        gpu.kernel("glm_indexer", "glm_index_tail_write_bf16")?,
        key,
        gate,
        pool.tail,
        slot,
        rows as u32,
        16,
        4,
        128,
        8192,
        stream,
    )?;
    ops::glm_index_kpool_finalize(
        gpu,
        gpu.kernel("glm_indexer", "glm_index_kpool_finalize_bf16")?,
        pool.tail,
        ape,
        pool.keys,
        slot,
        rows as u32,
        16,
        4,
        128,
        8192,
        1024,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.free(key)?;
    gpu.free(gate)?;
    gpu.free(slot)?;
    Ok(())
}
#[derive(Debug, PartialEq, Eq)]
struct Selection {
    visible_keys: Vec<u8>,
    visible_logits: Vec<u8>,
    sorted_ids: Vec<i32>,
}
fn selected(
    gpu: &dyn GpuBackend,
    pool: &Pool,
    table: DevicePtr,
    length: usize,
) -> Result<Selection> {
    let stream = gpu.default_stream();
    let q = upload(gpu, &(0..128).flat_map(|_| bf16(1.0)).collect::<Vec<_>>())?;
    let w = upload(gpu, &bf16(1.0))?;
    let stride = length.div_ceil(4);
    let logits = gpu.alloc(stride * 4)?;
    let out = gpu.alloc(2051 * 4)?;
    // Sentries make incomplete scorer/selector coverage observable.
    gpu.copy_h2d(&vec![0xff; stride * 4], logits)?;
    gpu.memset(out, 0x55, 2051 * 4)?;
    ops::glm_index_logits(
        gpu,
        gpu.kernel("glm_indexer", "glm_index_logits_bf16")?,
        q,
        w,
        pool.keys,
        logits,
        table,
        1,
        (length - 1) as u32,
        stride as u32,
        1,
        128,
        4,
        16,
        1024,
        1,
        8,
        stream,
    )?;
    ops::glm_index_topk_expand(
        gpu,
        gpu.kernel("glm_indexer", "glm_index_topk_expand")?,
        logits,
        out,
        1,
        (length - 1) as u32,
        stride as u32,
        2048,
        4,
        2051,
        stream,
    )?;
    let mut bytes = vec![0; 2051 * 4];
    gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
    let mut scores = vec![0; stride * 4];
    gpu.copy_d2h_on_stream(logits, &mut scores, stream)?;
    let mut physical_keys = vec![0; pool.blocks * 1024];
    gpu.copy_d2h_on_stream(pool.keys, &mut physical_keys, stream)?;
    let count = length / 4;
    let mut visible_keys = Vec::with_capacity(count * 256);
    for logical_pool in 0..count {
        let offset = (pool.blocks - 1 - logical_pool / 4) * 1024 + (logical_pool % 4) * 256;
        visible_keys.extend_from_slice(&physical_keys[offset..offset + 256]);
    }
    // Compare every causally visible pooled key and score, including pools
    // outside top-K. Future incomplete pools must be masked by the scorer.
    for raw in scores[count * 4..].chunks_exact(4) {
        ensure!(
            f32::from_le_bytes(raw.try_into().unwrap()) == f32::NEG_INFINITY,
            "future incomplete pool was scored at length={length}"
        );
    }
    let ids: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        ids.iter()
            .all(|&id| id == -1 || (id >= 0 && (id as usize) < length)),
        "selector emitted invalid or future token at length={length}"
    );
    let mut sorted_ids: Vec<_> = ids.iter().copied().filter(|&id| id >= 0).collect();
    sorted_ids.sort_unstable();
    ensure!(
        sorted_ids.windows(2).all(|p| p[0] != p[1]),
        "duplicate selected token"
    );
    let selected_pools = count.min(512);
    let mut expected: Vec<i32> = ((count - selected_pools) * 4..count * 4)
        .map(|i| i as i32)
        .collect();
    expected.extend((count * 4..length).map(|i| i as i32));
    if sorted_ids != expected {
        let missing: Vec<_> = expected
            .iter()
            .filter(|id| sorted_ids.binary_search(id).is_err())
            .take(12)
            .collect();
        let excess: Vec<_> = sorted_ids
            .iter()
            .filter(|id| expected.binary_search(id).is_err())
            .take(12)
            .collect();
        let values: Vec<_> = scores[..count * 4]
            .chunks_exact(4)
            .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
            .collect();
        let nonfinite = values.iter().filter(|v| !v.is_finite()).count();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let key_at =
            |p: usize| u16::from_le_bytes(visible_keys[p * 256..p * 256 + 2].try_into().unwrap());
        let sample: Vec<_> = [0, count / 2, count - 1]
            .into_iter()
            .map(|p| (p, key_at(p), values[p]))
            .collect();
        anyhow::bail!(
            "wrong top-K at length={length}: actual={} expected={} missing={missing:?} excess={excess:?} ids_head={:?} ids_tail={:?} nonfinite={nonfinite} score_min={min:e} score_max={max:e} pool_key_score_samples={sample:?}",
            sorted_ids.len(),
            expected.len(),
            &ids[..12],
            &ids[ids.len() - 12..]
        );
    }
    ensure!(
        ids.iter().filter(|&&id| id == -1).count() == 2051 - expected.len(),
        "invalid selector padding at length={length}"
    );
    // Atomic placement intentionally makes output order nondeterministic.
    for p in [q, w, logits, out] {
        gpu.free(p)?;
    }
    Ok(Selection {
        visible_keys,
        visible_logits: scores[..count * 4].to_vec(),
        sorted_ids,
    })
}
#[test]
#[ignore = "requires an available GB10 GPU and compiled GLM kernels; no model weights"]
fn rejected_pool_and_block_boundaries_match_clean_history() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let ape = upload(&gpu, &vec![0; 4 * 128 * 2])?;
    for base in [2046usize, 2047, 2048, 32750, 32751] {
        let blocks = (base + 8).div_ceil(16);
        let table = upload(
            &gpu,
            &(0..blocks)
                .flat_map(|b| ((blocks - 1 - b) as i32).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        for accepted_drafts in 0..=2 {
            let actual = pool(&gpu, blocks)?;
            let clean = pool(&gpu, blocks)?;
            write(&gpu, &actual, ape, blocks, 0, base, None)?;
            write(&gpu, &clean, ape, blocks, 0, base + 7, None)?;
            let kept = 1 + accepted_drafts;
            write(&gpu, &actual, ape, blocks, base, 3, Some(base + kept))?;
            // At rollback, future completed pools must be excluded by length.
            ensure!(
                selected(&gpu, &actual, table, base + kept)?
                    == selected(&gpu, &clean, table, base + kept)?,
                "rollback base={base} accepted={accepted_drafts}"
            );
            // Replacements republish every raw row; a pool-ending replacement
            // finalizes the canonical pool before it becomes visible again.
            for pos in base + kept..base + 7 {
                write(&gpu, &actual, ape, blocks, pos, 1, None)?;
                ensure!(
                    selected(&gpu, &actual, table, pos + 1)?
                        == selected(&gpu, &clean, table, pos + 1)?,
                    "replacement pos={pos} accepted={accepted_drafts}"
                );
            }
            for p in [actual.tail, actual.keys, clean.tail, clean.keys] {
                gpu.free(p)?;
            }
        }
        gpu.free(table)?;
    }
    gpu.free(ape)?;
    Ok(())
}
