// SPDX-License-Identifier: AGPL-3.0-only

//! Row masks of a strict speculative verify, after vLLM's regression cases
//! for #14702 / #44297: one mask per row from the matcher state after the
//! drafts before it, the bonus row masked, trimming at the first rejected
//! draft or `-1` pad, an exact rollback, and the `</think>` boundary.

use super::{StrictVerify, prepare, row_masks, width_word};
use crate::grammar::GrammarState;
use crate::scheduler::test_support::{
    GRAMMAR_VOCAB as VOCAB, THINK_END, THINK_START, grammar_engine, test_seq,
};

const WORDS: usize = VOCAB.div_ceil(32);
const TAGS: (Option<u32>, Option<u32>) = (Some(THINK_START), Some(THINK_END));
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,
    "properties":{"bridge":{"type":"string"}},"required":["bridge"]}"#;

fn strict() -> GrammarState {
    let mut engine = grammar_engine();
    let compiled = engine.compile_json_schema(SCHEMA).unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .strict_output(false)
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// The mask a fresh matcher reports after `prefix`, tags cleared: what each
/// row must equal.
fn mask_after(prefix: &[u32]) -> Vec<u32> {
    let mut gs = strict();
    for &t in prefix {
        assert!(gs.accept_token(t), "fixture prefix {prefix:?}");
    }
    assert!(gs.fill_bitmask());
    let mut m: Vec<u32> = gs.bitmask_data().iter().map(|&w| w as u32).collect();
    for t in [THINK_START, THINK_END] {
        m[t as usize / 32] &= !(1 << (t % 32));
    }
    m
}

fn rows(v: &StrictVerify) -> Vec<&[u32]> {
    assert_eq!(v.masks.len(), (v.drafts.len() + 1) * WORDS);
    v.masks.chunks(WORDS).collect()
}

fn allowed(row: &[u32], t: u32) -> bool {
    row[t as usize / 32] >> (t % 32) & 1 == 1
}

#[test]
fn every_row_and_the_bonus_row_carry_the_state_after_the_drafts_before_them() {
    let drafts = bytes(r#"{"bridge":"#);
    let mut gs = strict();
    let v = row_masks(&mut gs, false, TAGS, &drafts, VOCAB).unwrap();
    assert_eq!(v.drafts, drafts, "every draft is legal: none trimmed");
    for (r, row) in rows(&v).into_iter().enumerate() {
        assert_eq!(row, mask_after(&drafts[..r]), "row {r}");
    }
    // The walk is rolled back: the live matcher still starts the object.
    assert_eq!(gs.num_history_steps(), 0);
    assert!(gs.accept_token(b'{' as u32));
}

#[test]
fn a_rejected_draft_trims_the_window_and_rolls_back_exactly() {
    let mut gs = strict();
    assert!(gs.accept_token(b'{' as u32));
    let before = gs.num_history_steps();
    let drafts = bytes(r#""bZ"x"#);
    let v = row_masks(&mut gs, false, TAGS, &drafts, VOCAB).unwrap();
    // `"b` is a legal key prefix of "bridge"; `Z` is not.
    assert_eq!(v.drafts, bytes(r#""b"#));
    let r = rows(&v);
    assert!(
        !allowed(r[2], b'Z' as u32),
        "the bonus row refuses the rejected draft"
    );
    assert_eq!(gs.num_history_steps(), before);
    assert!(
        gs.accept_token(b'"' as u32),
        "matcher back at its pre-walk state"
    );
}

#[test]
fn a_pad_draft_is_no_draft_never_an_unconstrained_slot() {
    let mut gs = strict();
    let v = row_masks(
        &mut gs,
        false,
        TAGS,
        &[b'{' as u32, u32::MAX, b'"' as u32],
        VOCAB,
    )
    .unwrap();
    assert_eq!(v.drafts, [b'{' as u32]);
    assert_eq!(
        rows(&v)[1],
        mask_after(&[b'{' as u32]),
        "bonus row stays masked"
    );
    // A pad first: the verify keeps width 2 with a draft row 0 refuses, so
    // the step emits only its masked bonus.
    let v = row_masks(&mut gs, false, TAGS, &[u32::MAX, b'{' as u32], VOCAB).unwrap();
    let r = rows(&v);
    assert_eq!(v.drafts.len(), 1);
    assert!(!allowed(r[0], v.drafts[0]));
    assert_eq!(r[0], mask_after(&[]));
    assert_eq!(r[1], r[0]);
    assert_eq!(gs.num_history_steps(), 0);
}

#[test]
fn think_end_inside_the_window_switches_masks_at_the_next_row() {
    // vLLM #44297: rows inside reasoning are unmasked, the matcher never
    // consumes reasoning, and the row after `</think>` starts the grammar.
    let mut drafts = bytes("ok");
    drafts.push(THINK_END);
    drafts.extend(bytes(r#"{""#));
    let mut gs = strict();
    let v = row_masks(&mut gs, true, TAGS, &drafts, VOCAB).unwrap();
    assert_eq!(v.drafts, drafts);
    let r = rows(&v);
    for (i, row) in r[..3].iter().enumerate() {
        assert!(
            row.iter().all(|&w| w == u32::MAX),
            "reasoning row {i} unmasked"
        );
    }
    assert_eq!(r[3], mask_after(&[]), "start state after </think>");
    assert_eq!(r[4], mask_after(&[b'{' as u32]));
    assert_eq!(r[5], mask_after(&bytes(r#"{""#)));
    assert_eq!(gs.num_history_steps(), 0);
}

#[test]
fn thinking_off_rows_refuse_the_think_tags_as_the_serial_path_does() {
    let mut gs = strict();
    let v = row_masks(&mut gs, false, TAGS, &[THINK_START, b'{' as u32], VOCAB).unwrap();
    assert!(!allowed(rows(&v)[0], THINK_START));
    assert_eq!(
        v.drafts.len(),
        1,
        "the tag is trimmed, a dead draft keeps width 2"
    );
    assert!(!allowed(rows(&v)[0], v.drafts[0]));
}

#[test]
fn only_strict_sequences_mask_their_verify() {
    // A batch mixes strict and other sequences: theirs keep today's verify,
    // with no flag on the width word and nothing staged.
    let drafts = bytes("{\"");
    let (mut plain, _rx) = test_seq(vec![1], 20, None, 8);
    assert_eq!(prepare(&mut plain, &drafts, VOCAB).unwrap(), None);
    assert_eq!(width_word(3, &None), 3);
    let mut engine = grammar_engine();
    let compiled = engine.compile_json_grammar().unwrap();
    let (mut tool, _rx) = test_seq(vec![1], 20, None, 8);
    tool.grammar_state = Some(GrammarState::new(&compiled, VOCAB).unwrap());
    assert_eq!(prepare(&mut tool, &drafts, VOCAB).unwrap(), None);
    let (mut a, _rx) = test_seq(vec![1], 20, None, 8);
    a.grammar_state = Some(strict());
    let v = prepare(&mut a, &drafts, VOCAB).unwrap();
    assert_eq!(v.as_ref().map(|v| v.drafts.clone()), Some(drafts));
    assert_eq!(width_word(3, &v), 3 | spark_model::model::MASKED_VERIFY);
}
