// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only preparation for future indexed KDA decode; not wired to execution.

#![allow(dead_code)] // Foundation only: runtime promotion requires the separate GPU gate.

use anyhow::{Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::buffers::{DECODE_META_MAX_ROWS, DECODE_META_MIN_ROWS, DecodeMetaLayout};
use spark_runtime::gpu::DevicePtr;

use super::ssm_pool::SsmStatePool;
use crate::layer::ssm_batch::{SsmBatchView, SsmPoolView, checked_span};
use crate::layer::{LayerState, SsmLayerState};
use crate::traits::SequenceState;

/// Host-only upload payload plus stable device view. The caller must upload
/// `slot_bytes` at `view.layer(0)?.slots()` on every rank before future dispatch.
pub(crate) struct PreparedSsmBatch<'a> {
    pub(crate) view: SsmBatchView<'a>,
    pub(crate) slot_bytes: Vec<u8>,
}

pub(crate) fn ssm_ordinal(layer_types: &[LayerType], global_layer: usize) -> Result<usize> {
    ensure!(
        layer_types.get(global_layer) == Some(&LayerType::LinearAttention),
        "indexed SSM requested for a non-SSM layer"
    );
    Ok(layer_types[..global_layer]
        .iter()
        .filter(|&&kind| kind == LayerType::LinearAttention)
        .count())
}

/// `metadata_base` already points to the decode metadata block, not scratch[0].
/// Absence is the existing fallback; any present but inconsistent pool is an error.
pub(crate) fn prepare_indexed_decode<'a>(
    pool: Option<&'a SsmStatePool>,
    sequences: &[&SequenceState],
    layer_types: &[LayerType],
    metadata_base: DevicePtr,
    metadata_bytes: usize,
    layout: DecodeMetaLayout,
) -> Result<Option<PreparedSsmBatch<'a>>> {
    let Some(pool) = pool else { return Ok(None) };
    ensure!(
        pool.h_prefill_stage_pool.is_none(),
        "indexed SSM cannot use staged H storage"
    );
    ensure!(
        pool.num_ssm_layers == pool.h_state_pools.len(),
        "SSM pool layer count mismatch"
    );
    let view = SsmPoolView::new(
        &pool.h_state_pools,
        &pool.conv_state_pools,
        pool.h_bytes,
        pool.h_stored_bytes,
        pool.conv_bytes,
        pool.max_slots,
    )?;
    let rows: Vec<_> = sequences
        .iter()
        .map(|seq| SsmRow {
            slot: seq.ssm_slot_idx(),
            states: &seq.layer_states,
        })
        .collect();
    Ok(Some(prepare_rows(
        view,
        &rows,
        layer_types,
        metadata_base,
        metadata_bytes,
        layout,
    )?))
}

struct SsmRow<'a> {
    slot: Option<usize>,
    states: &'a [Box<dyn LayerState>],
}

fn prepare_rows<'a>(
    pool: SsmPoolView<'a>,
    rows: &[SsmRow<'_>],
    layer_types: &[LayerType],
    metadata_base: DevicePtr,
    metadata_bytes: usize,
    layout: DecodeMetaLayout,
) -> Result<PreparedSsmBatch<'a>> {
    ensure!(
        (1..=4).contains(&rows.len()),
        "indexed SSM supports one to four rows"
    );
    ensure!(
        layer_types
            .iter()
            .filter(|&&kind| kind == LayerType::LinearAttention)
            .count()
            == pool.layer_count(),
        "SSM model layer map disagrees with pools"
    );
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let slot = row
            .slot
            .ok_or_else(|| anyhow::anyhow!("live sequence lacks an SSM pool guard"))?;
        ids.push(i32::try_from(slot)?);
        ensure!(
            row.states.len() == layer_types.len(),
            "sequence layer-state count mismatch"
        );
        for (global, kind) in layer_types.iter().enumerate() {
            if *kind != LayerType::LinearAttention {
                continue;
            }
            let state = row.states[global]
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("missing live SSM layer state"))?;
            ensure!(
                !state.h_is_f16 && state.h_prefill_stage.is_none(),
                "indexed SSM requires live FP32 state"
            );
            let (h, conv) = pool.row_pointers(ssm_ordinal(layer_types, global)?, slot)?;
            ensure!(
                state.h_state == h && state.conv_state == conv,
                "live SSM pointers disagree with pool base, layer ordinal, or slot guard"
            );
        }
    }
    let (slots, slot_bytes) = slot_upload(metadata_base, metadata_bytes, layout, &ids)?;
    pool.validate_disjoint(slots, slot_bytes.len())?;
    let view = SsmBatchView::new(pool, slots, &ids)?;
    Ok(PreparedSsmBatch { view, slot_bytes })
}

fn slot_upload(
    metadata_base: DevicePtr,
    metadata_bytes: usize,
    layout: DecodeMetaLayout,
    ids: &[i32],
) -> Result<(DevicePtr, Vec<u8>)> {
    // Check the row ceiling before any offset calculation (the legacy layout
    // constructor intentionally does not impose policy or checked arithmetic).
    ensure!(
        (DECODE_META_MIN_ROWS..=DECODE_META_MAX_ROWS).contains(&layout.rows()),
        "unsupported SSM metadata row capacity"
    );
    ensure!(
        (1..=4).contains(&ids.len()) && ids.len() <= layout.rows(),
        "invalid active SSM row count"
    );
    checked_span(metadata_base, metadata_bytes)?;
    ensure!(
        metadata_bytes >= layout.block_table_off(),
        "SSM metadata region is truncated"
    );
    let slots = metadata_base.offset(layout.ssm_slots_off());
    let mut bytes = vec![255; layout.rows() * 4];
    for (target, id) in bytes.chunks_exact_mut(4).zip(ids) {
        target.copy_from_slice(&id.to_le_bytes());
    }
    Ok((slots, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(h: u64, conv: u64) -> Box<dyn LayerState> {
        Box::new(SsmLayerState {
            h_state: DevicePtr(h),
            conv_state: DevicePtr(conv),
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: vec![],
            conv_state_intermediates: vec![],
            h_is_f16: false,
            h_prefill_stage: None,
        })
    }

    #[test]
    fn live_pointers_follow_nonprefix_slots_and_global_layer_ordinals() {
        let h = [DevicePtr(0x1000), DevicePtr(0x3000)];
        let c = [DevicePtr(0x5000), DevicePtr(0x7000)];
        let pool = SsmPoolView::new(&h, &c, 64, 64, 32, 8).unwrap();
        let kinds = [
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ];
        let a = vec![
            state(0x1180, 0x50c0),
            Box::new(crate::layer::EmptyLayerState) as Box<dyn LayerState>,
            state(0x3180, 0x70c0),
        ];
        let b = vec![
            state(0x1040, 0x5020),
            Box::new(crate::layer::EmptyLayerState) as Box<dyn LayerState>,
            state(0x3040, 0x7020),
        ];
        let rows = [
            SsmRow {
                slot: Some(6),
                states: &a,
            },
            SsmRow {
                slot: Some(1),
                states: &b,
            },
        ];
        let layout = DecodeMetaLayout::for_max_batch_size(4);
        let prepared = prepare_rows(pool, &rows, &kinds, DevicePtr(0x9000), 4096, layout).unwrap();
        assert_eq!(prepared.view.layer(1).unwrap().h_base(), h[1]);
        assert_eq!(&prepared.slot_bytes[..8], &[6, 0, 0, 0, 1, 0, 0, 0]);
        // Reordered and drained active rows must rewrite all padded IDs every step.
        let drained =
            prepare_rows(pool, &rows[1..], &kinds, DevicePtr(0x9000), 4096, layout).unwrap();
        assert_eq!(&drained.slot_bytes[..4], &[1, 0, 0, 0]);
        assert!(drained.slot_bytes[4..].iter().all(|&v| v == 255));
        let swapped = [
            SsmRow {
                slot: Some(1),
                states: &b,
            },
            SsmRow {
                slot: Some(6),
                states: &a,
            },
        ];
        let reordered =
            prepare_rows(pool, &swapped, &kinds, DevicePtr(0xa000), 4096, layout).unwrap();
        assert_eq!(&reordered.slot_bytes[..8], &[1, 0, 0, 0, 6, 0, 0, 0]);
        assert_eq!(reordered.view.layer(0).unwrap().slots(), DevicePtr(0xa280));
        for slot in [None, Some(1), Some(8), Some(usize::MAX)] {
            assert!(
                prepare_rows(
                    pool,
                    &[SsmRow { slot, states: &a }],
                    &kinds,
                    DevicePtr(0x9000),
                    4096,
                    layout
                )
                .is_err()
            );
        }
        assert!(prepare_rows(pool, &rows, &kinds[..2], DevicePtr(0x9000), 4096, layout).is_err());
    }

    #[test]
    fn present_non_fp32_missing_and_stale_layer_states_fail_closed() {
        let h = [DevicePtr(0x1000)];
        let c = [DevicePtr(0x3000)];
        let pool = SsmPoolView::new(&h, &c, 64, 64, 32, 8).unwrap();
        let kinds = [LayerType::LinearAttention];
        let layout = DecodeMetaLayout::for_max_batch_size(4);
        let mut states = vec![state(0x1000, 0x3000)];
        for mode in 0..5 {
            let st = states[0]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .unwrap();
            st.h_is_f16 = mode == 0;
            st.h_prefill_stage = (mode == 1).then_some(DevicePtr(0x9000));
            st.h_state = DevicePtr(if mode == 2 { 0x1040 } else { 0x1000 });
            st.conv_state = DevicePtr(if mode == 3 { 0x3020 } else { 0x3000 });
            let live = if mode == 4 { &states[..0] } else { &states[..] };
            assert!(
                prepare_rows(
                    pool,
                    &[SsmRow {
                        slot: Some(0),
                        states: live
                    }],
                    &kinds,
                    DevicePtr(0x9000),
                    4096,
                    layout
                )
                .is_err()
            );
        }
        let empty: Vec<Box<dyn LayerState>> = vec![Box::new(crate::layer::EmptyLayerState)];
        assert!(
            prepare_rows(
                pool,
                &[SsmRow {
                    slot: Some(0),
                    states: &empty
                }],
                &kinds,
                DevicePtr(0x9000),
                4096,
                layout
            )
            .is_err()
        );
    }

    #[test]
    fn absent_pool_is_fallback_not_malformed_metadata() {
        assert!(
            prepare_indexed_decode(
                None,
                &[],
                &[],
                DevicePtr::NULL,
                0,
                DecodeMetaLayout::for_max_batch_size(4)
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn padded_slot_payload_cannot_overlap_live_state() {
        let h = [DevicePtr(0x1000)];
        let c = [DevicePtr(0x3000)];
        let pool = SsmPoolView::new(&h, &c, 64, 64, 32, 8).unwrap();
        let states = vec![state(0x1000, 0x3000)];
        let row = [SsmRow {
            slot: Some(0),
            states: &states,
        }];
        let kinds = [LayerType::LinearAttention];
        let layout = DecodeMetaLayout::for_max_batch_size(4);
        // Active ID ends before H, but the complete 32-row padded upload overlaps it.
        assert!(prepare_rows(pool, &row, &kinds, DevicePtr(0xd04), 4096, layout).is_err());
        // Moving four bytes earlier makes the padded region exactly adjacent.
        assert!(prepare_rows(pool, &row, &kinds, DevicePtr(0xd00), 4096, layout).is_ok());
    }

    #[test]
    fn layer_ordinals_skip_attention_and_reject_wrong_layer_kinds() {
        let kinds = [
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ];
        assert_eq!(ssm_ordinal(&kinds, 0).unwrap(), 0);
        assert_eq!(ssm_ordinal(&kinds, 2).unwrap(), 1);
        assert!(ssm_ordinal(&kinds, 1).is_err());
        assert!(ssm_ordinal(&kinds, 3).is_err());
    }

    #[test]
    fn slot_region_is_bounded_and_preserves_legacy_metadata() {
        for rows in [32, 64, 128] {
            let layout = DecodeMetaLayout::for_max_batch_size(rows);
            let (ptr, bytes) =
                slot_upload(DevicePtr(0x1000), layout.meta_bytes(2), layout, &[7, 2]).unwrap();
            assert_eq!(ptr, DevicePtr(0x1000 + (20 * rows) as u64));
            assert_eq!(bytes.len(), rows * 4);
            let ids: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(&ids[..2], &[7, 2]);
            assert!(ids[2..].iter().all(|&id| id == -1));
            assert!(slot_upload(DevicePtr(0x1000), 24 * rows - 1, layout, &[0]).is_err());
        }
        let layout = DecodeMetaLayout::for_max_batch_size(4);
        assert!(slot_upload(DevicePtr::NULL, 4096, layout, &[0]).is_err());
        assert!(slot_upload(DevicePtr(u64::MAX - 3), 4096, layout, &[0]).is_err());
        assert!(
            slot_upload(
                DevicePtr(0x1000),
                4096,
                DecodeMetaLayout::for_max_batch_size(usize::MAX),
                &[0]
            )
            .is_err()
        );
    }
}
