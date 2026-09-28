// SPDX-License-Identifier: AGPL-3.0-only
//! Actual scalar Model boundary; sentinels prove ownership, not CUDA numerics.
use super::fixture::*;
use crate::layers::glm5_mtp::Glm5MtpProposerState;
use crate::traits::Model;
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::Ordering;

fn isolated(name: &str, profiles: &[(&str, &str)]) -> bool {
    if std::env::var("ATLAS_EAGER_BOOTSTRAP_CHILD").as_deref() == Ok("1") {
        return false;
    }
    for (ep, gdn) in profiles {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            &format!("model::glm_c2_handoff_tests::eager_bootstrap_error_tests::{name}"),
            "--nocapture",
        ])
        .env("ATLAS_EAGER_BOOTSTRAP_CHILD", "1")
        .env("ATLAS_EP_GRAPHS", ep)
        .env("ATLAS_GDN_DECODE_GRAPH", gdn)
        .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
        .env("ATLAS_GLM_MTP_REPAIR", "0")
        .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
        .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
        .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1");
        for key in [
            "ATLAS_DEBUG_NO_GRAPH",
            "ATLAS_DIAG_GEMMA4",
            "ATLAS_SSM_SAVE_DUMP",
            "ATLAS_GLM_MULTI_SEQ_SPARSE",
            "ATLAS_NO_MTP_EAGER_DRAFTER",
            "ATLAS_NO_MTP_DRAFTER_CONTEXT",
            "ATLAS_MTP_CARRY_DRAFTER",
            "ATLAS_GLM_MTP_SERIAL_PREFILL",
            "ATLAS_GLM_MTP_FUSED_EH_NORM",
            "ATLAS_GLM_MTP_PROFILE",
            "ATLAS_MTP_CATCHUP",
            "ATLAS_MTP_ACCEPT_DEBUG",
        ] {
            cmd.env_remove(key);
        }
        assert!(
            cmd.status().unwrap().success(),
            "{name}: EP={ep}, GDN={gdn}"
        );
    }
    true
}

fn state(f: &Fixture, owner: usize) -> &Glm5MtpProposerState {
    f.seqs[owner]
        .proposer_state
        .as_ref()
        .unwrap()
        .as_any()
        .downcast_ref::<Glm5MtpProposerState>()
        .unwrap()
}
fn slot(f: &Fixture, owner: usize) -> DevicePtr {
    f.gpu.slab().offset(owner * 6 * ROW_BYTES)
}
fn prepared(rank: usize, paired: bool) -> Fixture {
    let mut f = if paired {
        Fixture::new(rank)
    } else {
        Fixture::new_legacy(rank)
    };
    if paired {
        f.gpu.write_span(f.gpu.slab(), &vec![0xa5; SLAB_BYTES]);
    }
    for owner in 0..if paired { 2 } else { 1 } {
        let prompt = if owner == 0 {
            [1, 2, 3, 4]
        } else {
            [4, 3, 2, 1]
        };
        f.model
            .prefill(&prompt, &mut f.seqs[owner], CALLER)
            .unwrap();
        assert_eq!(f.seqs[owner].seq_len, 4);
        assert_eq!(state(&f, owner).seq_len, 3);
    }
    assert!(!f.model.profile);
    assert!(!f.model.suppress_graphs.load(Ordering::Relaxed));
    f.gpu.clear();
    f
}
fn target_ordinal(events: &[Event]) -> usize {
    let found: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| **e == Event::Target(1, 4, DEFAULT))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(found.len(), 1, "actual target boundary: {events:?}");
    found[0]
}
fn decode(f: &mut Fixture, owner: usize) -> anyhow::Result<DevicePtr> {
    f.model.decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
}
fn successful_bonus(f: &Fixture, owner: usize) {
    assert_eq!(f.seqs[owner].seq_len, 5);
    assert_eq!(state(f, owner).seq_len, 3);
    assert_eq!(
        f.gpu
            .read_span(slot(f, owner).offset(5 * ROW_BYTES), ROW_BYTES),
        vec![5 + owner as u8 + 0x24; ROW_BYTES]
    );
    let events = f.gpu.trace();
    let copy = Event::Copy(
        f.model.buffers.norm_output(),
        slot(f, owner).offset(5 * ROW_BYTES),
        ROW_BYTES,
        DEFAULT,
    );
    let at = events.iter().position(|e| *e == copy).unwrap();
    assert_eq!(events.get(at + 1), Some(&Event::Sync(DEFAULT)));
    assert!(target_ordinal(&events) <= at);
}

#[test]
fn selected_body_error_has_no_inner_graph_cleanup() {
    if isolated(
        "selected_body_error_has_no_inner_graph_cleanup",
        &[("0", "0")],
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let peer = 1 - owner;
            let mut control = prepared(rank, true);
            decode(&mut control, peer).unwrap();
            control.gpu.clear();
            decode(&mut control, owner).unwrap();
            successful_bonus(&control, owner);
            let ordinal = target_ordinal(&control.gpu.trace());

            let mut f = prepared(rank, true);
            decode(&mut f, peer).unwrap();
            let slab = f.gpu.read_span(slot(&f, peer), 6 * ROW_BYTES);
            let tokens = f.seqs[peer].tokens.clone();
            let blocks = state(&f, peer).block_table.clone();
            let rows = f
                .head
                .paired_test_kv_rows(
                    f.seqs[peer].proposer_state.as_ref().unwrap().as_ref(),
                    f.model.gpu.as_ref(),
                    3,
                )
                .unwrap();
            let kv: Vec<_> = rows
                .iter()
                .map(|(k, v)| (f.gpu.read_span(*k, 1024), f.gpu.read_span(*v, 1024)))
                .collect();
            f.gpu.clear();
            f.gpu.fail.store(ordinal, Ordering::Relaxed);
            let error = decode(&mut f, owner).unwrap_err();
            assert!(format!("{error:#}").contains("injected fixture operation failure"));
            let events = f.gpu.trace();
            assert_eq!(events.get(ordinal - 1), Some(&Event::Target(1, 4, DEFAULT)));
            assert_eq!(
                events.len(),
                ordinal,
                "selected body Err reached, but inner cleanup/follow-on ran: {events:?}"
            );
            f.gpu.clear();
            f.model.gpu.synchronize(DEFAULT).unwrap();
            f.gpu.clear();
            assert!(decode(&mut f, owner).is_err());
            assert!(f.gpu.trace().is_empty(), "failed lease retried work");
            assert_eq!(f.gpu.read_span(slot(&f, peer), 6 * ROW_BYTES), slab);
            assert_eq!(f.seqs[peer].tokens, tokens);
            assert_eq!(state(&f, peer).block_table, blocks);
            assert_eq!(state(&f, peer).seq_len, 3);
            for ((k, v), (kb, vb)) in rows.iter().zip(&kv) {
                assert_eq!(f.gpu.read_span(*k, 1024), *kb);
                assert_eq!(f.gpu.read_span(*v, 1024), *vb);
            }
        }
    }
}

#[test]
fn selected_bootstrap_is_eager_despite_inherited_graph_flags() {
    if isolated(
        "selected_bootstrap_is_eager_despite_inherited_graph_flags",
        &[("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")],
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let mut f = prepared(rank, true);
            decode(&mut f, owner).unwrap();
            successful_bonus(&f, owner);
            assert!(
                !f.gpu.trace().iter().any(|e| matches!(
                    e,
                    Event::BeginCapture(_)
                        | Event::EndCapture(_)
                        | Event::AbortCapture(_)
                        | Event::LaunchGraph(..)
                )),
                "selected scalar path attempted graph work: {:?}",
                f.gpu.trace()
            );
        }
    }
}

#[test]
fn actual_legacy_decode_retains_graph_and_error_behavior() {
    if isolated(
        "actual_legacy_decode_retains_graph_and_error_behavior",
        &[("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")],
    ) {
        return;
    }
    let graphs = std::env::var("ATLAS_EP_GRAPHS").unwrap() == "1"
        || std::env::var("ATLAS_GDN_DECODE_GRAPH").unwrap() == "1";
    for rank in 0..2 {
        let mut control = prepared(rank, false);
        assert!(control.model.glm_paired_execution().is_none());
        decode(&mut control, 0).unwrap();
        let events = control.gpu.trace();
        assert_eq!(
            events
                .iter()
                .filter(|e| **e == Event::BeginCapture(DEFAULT))
                .count(),
            usize::from(graphs)
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| **e == Event::EndCapture(DEFAULT))
                .count(),
            usize::from(graphs)
        );
        let ordinal = target_ordinal(&events);
        let mut f = prepared(rank, false);
        f.gpu.fail.store(ordinal, Ordering::Relaxed);
        let error = decode(&mut f, 0).unwrap_err();
        assert!(format!("{error:#}").contains("injected fixture operation failure"));
        assert_eq!(
            f.gpu.trace().get(ordinal - 1),
            Some(&Event::Target(1, 4, DEFAULT))
        );
        assert_eq!(&f.gpu.trace()[ordinal..], &[Event::AbortCapture(DEFAULT)]);
    }
}
