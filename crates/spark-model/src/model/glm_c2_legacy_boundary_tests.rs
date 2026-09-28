// SPDX-License-Identifier: AGPL-3.0-only
//! Actual old writer entries must not accept request-owned paired state.
//!
//! Nested under glm_c2_handoff_tests by the implementation author after review.
//! Each API has its own behavioral RED: otherwise-valid cold input, real paired
//! reserve, and actual model context. These are recording-backend ownership
//! tests, not numerical or full worker/acceptance qualification.
use super::{fixture::*, isolated};
use crate::layers::glm5_mtp::Glm5MtpProposerState;
use crate::speculative::DraftProposer;
use crate::speculative::glm_repair::{GlmPairRepair, RepairInput, RepairSpan};
use spark_runtime::gpu::DevicePtr;

#[derive(Debug, PartialEq, Eq)]
struct Cursors(Vec<(usize, usize, Vec<u32>)>);

fn cursors(f: &Fixture) -> Cursors {
    Cursors(
        f.seqs
            .iter()
            .map(|seq| {
                let state = seq
                    .proposer_state
                    .as_ref()
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Glm5MtpProposerState>()
                    .unwrap();
                (
                    state.seq_len,
                    state.last_num_drafted,
                    state.block_table.clone(),
                )
            })
            .collect(),
    )
}

fn seed_spans(f: &Fixture) -> Vec<(DevicePtr, Vec<u8>)> {
    // Entire handoff slab includes both owners, four accepted rows, and bonus.
    // Distinct initialized input bytes make unintended stores observable.
    [
        (f.gpu.slab(), SLAB_BYTES, 0xa5),
        (f.model.mtp_prefill_hidden, 4 * ROW_BYTES, 0x31),
        (
            f.model.buffers.norm_output(),
            f.model.buffers.sizes().norm_output,
            0x52,
        ),
        (f.model.mtp_hidden_save, ROW_BYTES, 0x73),
    ]
    .into_iter()
    .map(|(ptr, bytes, value)| {
        let data = vec![value; bytes];
        f.gpu.write_span(ptr, &data);
        (ptr, data)
    })
    .collect()
}

fn unchanged(f: &Fixture, before: &Cursors, spans: &[(DevicePtr, Vec<u8>)]) {
    assert!(
        f.gpu.trace().is_empty(),
        "old API must refuse before any recorded backend work: {:?}",
        f.gpu.trace()
    );
    assert_eq!(&cursors(f), before, "both owned cursors/reserves unchanged");
    for (ptr, bytes) in spans {
        assert_eq!(f.gpu.read_span(*ptr, bytes.len()), *bytes);
    }
}

fn raw_input(f: &Fixture) -> RepairInput<'static> {
    RepairInput {
        token: 6,
        tokens: &[1, 2, 3, 4, 5],
        prompt_len: 4,
        position: 5,
        drafts: 4,
        generation: 1,
        capture_generation: 1,
        captured_rows: 4,
        context_tokens: 2044,
        capture: RepairSpan {
            ptr: f.model.mtp_prefill_hidden,
            bytes: f.model.mtp_prefill_capacity * ROW_BYTES,
        },
        normalized: RepairSpan {
            ptr: f.model.buffers.norm_output(),
            bytes: f.model.buffers.sizes().norm_output,
        },
        bonus: RepairSpan {
            ptr: f.model.mtp_hidden_save,
            bytes: ROW_BYTES,
        },
        hidden_row: 0,
    }
}

#[test]
fn raw_repair_validation_refuses_paired_owner_before_work() {
    if isolated("legacy_boundary_tests::raw_repair_validation_refuses_paired_owner_before_work") {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let f = Fixture::new(rank);
            let before = cursors(&f);
            let spans = seed_spans(&f);
            let input = raw_input(&f);
            let ctx = f.model.glm_repair_context();
            f.gpu.clear();
            let result = f.head.validate_prepare(
                &input,
                f.seqs[owner].proposer_state.as_ref().unwrap().as_ref(),
                &ctx,
            );
            let error = result.expect_err("raw RepairInput accepted actual paired owner");
            assert!(format!("{error:#}").contains("paired"), "{error:#}");
            unchanged(&f, &before, &spans);
        }
    }
}

#[test]
fn raw_repair_writer_refuses_paired_owner_before_work() {
    if isolated("legacy_boundary_tests::raw_repair_writer_refuses_paired_owner_before_work") {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let mut f = Fixture::new(rank);
            let before = cursors(&f);
            let spans = seed_spans(&f);
            let input = raw_input(&f);
            let ctx = f.model.glm_repair_context();
            f.gpu.clear();
            let result = f.head.prepare(
                &input,
                f.seqs[owner].proposer_state.as_mut().unwrap().as_mut(),
                &ctx,
                DEFAULT,
            );
            let error = result.expect_err("raw repair writer accepted actual paired owner");
            assert!(format!("{error:#}").contains("paired"), "{error:#}");
            unchanged(&f, &before, &spans);
        }
    }
}

#[test]
fn legacy_batched_prefill_refuses_paired_owner_before_work() {
    if isolated("legacy_boundary_tests::legacy_batched_prefill_refuses_paired_owner_before_work") {
        return;
    }
    assert_eq!(
        std::env::var("ATLAS_GLM_MTP_BATCHED_PREFILL").as_deref(),
        Ok("1"),
        "exercise the real enabled legacy batched writer"
    );
    for rank in 0..2 {
        for owner in 0..2 {
            let mut f = Fixture::new(rank);
            let before = cursors(&f);
            let spans = seed_spans(&f);
            let ctx = f.model.glm_repair_context();
            f.gpu.clear();
            let result = f.head.prefill_drafter(
                &[1, 2, 3, 4],
                f.model.mtp_prefill_hidden,
                f.seqs[owner].proposer_state.as_mut().unwrap().as_mut(),
                &ctx,
                DEFAULT,
            );
            let error = result.expect_err("legacy batched writer accepted actual paired owner");
            assert!(format!("{error:#}").contains("paired"), "{error:#}");
            unchanged(&f, &before, &spans);
        }
    }
}
