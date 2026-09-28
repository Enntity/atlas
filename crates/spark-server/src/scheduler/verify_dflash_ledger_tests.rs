// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const DRAFTS: [u32; 4] = [10, 11, 12, 13];
const RAW: [u32; 5] = [20, 21, 22, 23, 24];

fn prepared(ledger: &Ledger) -> Option<Prepared> {
    ledger.prepare(true, 17, 7, &DRAFTS, &RAW, 100)
}

#[test]
fn k5_ledger_preserves_raw_selected_and_every_acceptance_count() {
    for accepted in 0..=4 {
        let mut ledger = Ledger::default();
        let mut selected = [99; 5];
        selected[..accepted].copy_from_slice(&DRAFTS[..accepted]);
        let capture = prepared(&ledger);
        let record = ledger
            .finish(capture, &selected, accepted)
            .expect("valid K5 record");
        assert_eq!(
            record,
            Record {
                ordinal: 1,
                position: 17,
                seed: 7,
                drafts: DRAFTS,
                raw: RAW,
                selected,
                accepted,
            }
        );
    }
}

#[test]
fn k5_ledger_rejects_disabled_shape_bounds_and_overflow() {
    let ledger = Ledger::default();
    assert!(ledger.prepare(false, 17, 7, &DRAFTS, &RAW, 100).is_none());
    for drafts in [&DRAFTS[..3], &[1, 2, 3, 4, 5][..]] {
        assert!(ledger.prepare(true, 17, 7, drafts, &RAW, 100).is_none());
    }
    for raw in [&RAW[..4], &[1, 2, 3, 4, 5, 6][..]] {
        assert!(ledger.prepare(true, 17, 7, &DRAFTS, raw, 100).is_none());
    }
    assert!(
        ledger
            .prepare(true, usize::MAX - 4, 7, &DRAFTS, &RAW, 100)
            .is_none()
    );
    assert!(ledger.prepare(true, 17, 100, &DRAFTS, &RAW, 100).is_none());
    assert!(
        ledger
            .prepare(true, 17, 7, &[10, 100, 12, 13], &RAW, 100)
            .is_none()
    );
    assert!(
        ledger
            .prepare(true, 17, 7, &DRAFTS, &[20, 21, 22, 23, 100], 100)
            .is_none()
    );
    assert!(ledger.prepare(true, 17, 7, &DRAFTS, &RAW, 0).is_none());
    assert_eq!(ledger.emitted, 0);
}

#[test]
fn k5_ledger_rejected_verdict_does_not_spend_budget() {
    let mut ledger = Ledger::default();
    for (selected, accepted) in [
        (&RAW[..4], 0),
        (&RAW[..], 5),
        (&RAW[..], 1),
        (&[10, 11, 99, 99, 99][..], 1),
        (&[99, 99, 99, 99, 100][..], 0),
    ] {
        let capture = prepared(&ledger);
        assert!(ledger.finish(capture, selected, accepted).is_none());
        assert_eq!(ledger.emitted, 0);
    }
    assert!(ledger.finish(None, &RAW, 0).is_none());
}

#[test]
fn k5_ledger_is_bounded_per_request_and_default_resets_it() {
    let mut ledger = Ledger::default();
    for ordinal in 1..=8 {
        let capture = prepared(&ledger);
        assert_eq!(ledger.finish(capture, &RAW, 0).unwrap().ordinal, ordinal);
    }
    assert!(prepared(&ledger).is_none());
    assert_eq!(ledger.emitted, 8);
    let snapshot = ledger;
    assert!(prepared(&snapshot).is_none());
    ledger = Ledger::default();
    let capture = prepared(&ledger);
    assert_eq!(ledger.finish(capture, &RAW, 0).unwrap().ordinal, 1);
}

#[test]
fn k5_ledger_requires_explicit_flag_and_native_state() {
    for value in [None, Some("0"), Some("true"), Some(""), Some("2")] {
        assert!(!flag_enabled(value));
    }
    assert!(flag_enabled(Some("1")));
    let mut seq = spark_model::traits::SequenceState::host_only(0);
    assert!(!native_glm(&seq));
    struct OtherProposer;
    impl spark_model::speculative::ProposerState for OtherProposer {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }
    seq.proposer_state = Some(Box::new(OtherProposer));
    assert!(
        !native_glm(&seq),
        "other proposer families must not be diagnosed as GLM"
    );
}
