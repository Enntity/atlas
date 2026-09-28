// SPDX-License-Identifier: AGPL-3.0-only
//! Unsupported real entry points must not bypass a live K5 output lease.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::model::types::TransformerModel;
use crate::traits::Model;
use anyhow::Result;

#[derive(Clone, Copy, Debug)]
enum Escape {
    DraftTrait,
    DraftInherent,
    VerifyGeneric,
    VerifyK2,
    VerifyK3,
    VerifyK4,
    VerifyBatched,
    VerifyFused,
    Rollback,
    RestoreSnapshot,
    Normalize,
    RollbackCheckpoint,
    RestoreSequence,
    Compact,
}

#[derive(Default)]
struct ReadSpy(usize);
impl std::io::Read for ReadSpy {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        self.0 += 1;
        Ok(0)
    }
}

fn attempt(f: &mut Fixture, owner: usize, path: Escape, reader: &mut ReadSpy) -> Result<()> {
    let model = &f.model;
    match path {
        Escape::DraftTrait => Model::decode_draft(model, 1, &mut f.seqs[owner], CALLER).map(|_| ()),
        Escape::DraftInherent => {
            TransformerModel::decode_draft(model, 1, &mut f.seqs[owner], CALLER).map(|_| ())
        }
        Escape::VerifyGeneric => model
            .decode_verify(&[1, 2], &mut f.seqs[owner], CALLER)
            .map(|_| ()),
        Escape::VerifyK2 => model
            .decode_verify_graphed(&[1, 2], &mut f.seqs[owner], CALLER)
            .map(|_| ()),
        Escape::VerifyK3 => model
            .decode_verify_graphed_k3(&[1, 2, 3], &mut f.seqs[owner], CALLER)
            .map(|_| ()),
        Escape::VerifyK4 => model
            .decode_verify_graphed_k4(&[1, 2, 3, 4], &mut f.seqs[owner], CALLER)
            .map(|_| ()),
        Escape::VerifyBatched => {
            let mut seqs: Vec<_> = f.seqs.iter_mut().collect();
            model
                .decode_verify_batched(&[1, 2, 3, 4], &[2, 2], &mut seqs, CALLER)
                .map(|_| ())
        }
        Escape::VerifyFused => model
            .decode_and_verify_fused(&[1, 2], &mut f.seqs[owner], CALLER)
            .map(|_| ()),
        Escape::Rollback => model.rollback_ssm_states(&mut f.seqs[owner], 1),
        Escape::RestoreSnapshot => model.restore_decode_ssm_snapshot(&f.seqs[owner], 0),
        Escape::Normalize => model.normalize_ssm_states(&f.seqs[owner], CALLER),
        Escape::RollbackCheckpoint => {
            model.start_rollback_and_checkpoint_async(&mut f.seqs[owner], 1)
        }
        Escape::RestoreSequence => model.restore_sequence_state(&mut f.seqs[owner], 1, reader),
        Escape::Compact => model.compact_sequence(&mut f.seqs[owner], 1 - owner),
    }
}

fn check(name: &str, path: Escape) {
    if flow::isolated(&format!("verdict_escape_tests::{name}")) {
        return;
    }
    for rank in 0..2 {
        for producer in 0..2 {
            for attempted_owner in 0..2 {
                let (mut f, histories) = flow::prepare(rank, [producer, 1 - producer]);
                let h = &histories[producer];
                f.model
                    .decode_verify_graphed_kgamma(&h.issued, &mut f.seqs[producer], CALLER)
                    .unwrap();
                let norm = f
                    .gpu
                    .read_span(f.model.buffers.norm_output(), 5 * ROW_BYTES);
                let slab = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
                let cursors = f.seqs.each_ref().map(|s| (s.seq_len, s.tokens.clone()));
                let blocks = f.seqs.each_ref().map(|s| s.block_table.clone());
                let slots = f.seqs.each_ref().map(|s| s.slot_idx);
                let mut reader = ReadSpy::default();
                f.gpu.clear();
                let result = attempt(&mut f, attempted_owner, path, &mut reader);
                let error = result.expect_err(&format!("{path:?} bypassed active output lease"));
                assert!(
                    format!("{error:#}").contains("paired"),
                    "{path:?} failed for an unrelated reason: {error:#}"
                );
                assert!(f.gpu.trace().is_empty(), "{path:?} performed backend work");
                assert_eq!(reader.0, 0, "refusal must precede restore I/O");
                assert_eq!(
                    f.gpu.read_span(f.model.buffers.norm_output(), norm.len()),
                    norm
                );
                assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), slab);
                assert_eq!(
                    f.seqs.each_ref().map(|s| (s.seq_len, s.tokens.clone())),
                    cursors
                );
                assert_eq!(f.seqs.each_ref().map(|s| s.block_table.clone()), blocks);
                assert_eq!(f.seqs.each_ref().map(|s| s.slot_idx), slots);

                // A rejected alternate entry must not consume or poison the
                // valid producer's output. Consume it through the real hook.
                f.seqs[producer].seq_len = h.base + 3;
                f.seqs[producer].tokens.truncate(h.base + 3);
                f.model
                    .record_glm_mtp_verified(&mut f.seqs[producer], h.base, &h.issued, 2)
                    .unwrap();
                flow::detached(&f, producer, h, 2);
            }
        }
    }
}

macro_rules! escape_test {
    ($name:ident, $path:ident) => {
        #[test]
        fn $name() {
            check(stringify!($name), Escape::$path);
        }
    };
}
escape_test!(produced_refuses_draft_trait, DraftTrait);
escape_test!(produced_refuses_draft_inherent, DraftInherent);
escape_test!(produced_refuses_generic_verify, VerifyGeneric);
escape_test!(produced_refuses_k2_verify, VerifyK2);
escape_test!(produced_refuses_k3_verify, VerifyK3);
escape_test!(produced_refuses_k4_verify, VerifyK4);
escape_test!(produced_refuses_batched_verify, VerifyBatched);
escape_test!(produced_refuses_fused_verify, VerifyFused);
escape_test!(produced_refuses_legacy_rollback, Rollback);
escape_test!(produced_refuses_snapshot_restore, RestoreSnapshot);
escape_test!(produced_refuses_ssm_normalization, Normalize);
escape_test!(produced_refuses_rollback_checkpoint, RollbackCheckpoint);
escape_test!(produced_refuses_sequence_restore, RestoreSequence);
escape_test!(produced_refuses_target_slot_compaction, Compact);
