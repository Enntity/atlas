// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The defaults with copy drafts on.
fn on() -> Settings {
    Settings::parse(|name| (name == "ATLAS_DFLASH_COPY_DRAFTS").then(|| "1".into())).unwrap()
}

fn ctx(tokens: &[u32]) -> Context<'_> {
    let (&pending, history) = tokens.split_last().unwrap();
    Context { history, pending }
}

/// Proposal of a fresh index over `tokens` (pending token last).
fn propose(n: usize, prompt: usize, tokens: &[u32], k: usize, reply_match: usize) -> Vec<u32> {
    Index::new(n, prompt).propose(ctx(tokens), k, reply_match)
}

#[test]
fn settings_are_off_unless_asked_and_fall_back_on_bad_values() {
    assert_eq!(Settings::parse(|_| None), None);
    assert_eq!(Settings::parse(|_| Some("0".into())), None);
    let s = on();
    assert_eq!(
        (s.match_len, s.max, s.reply_match, s.miss_max),
        (8, MAX_DRAFTS, 0, 0)
    );
    let s = Settings::parse(|name| {
        let value = match name {
            "ATLAS_DFLASH_COPY_MATCH" => "4",
            "ATLAS_DFLASH_COPY_MAX" => "99",
            // Not more than the match: off.
            "ATLAS_DFLASH_COPY_REPLY_MATCH" => "3",
            "ATLAS_DFLASH_COPY_MISS_MAX" => "2",
            _ => "1",
        };
        Some(value.into())
    })
    .unwrap();
    assert_eq!(
        (s.match_len, s.max, s.reply_match, s.miss_max),
        (4, MAX_DRAFTS, 0, 2)
    );
}

#[test]
fn a_quoted_span_proposes_what_followed_it() {
    // Prompt 10..=19; the reply has quoted 12, 13, 14 and is pending on 14.
    let tokens = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 12, 13, 14];
    assert_eq!(propose(3, 11, &tokens, 4, 0), [15, 16, 17, 18]);
    // A longer copy reads on through the prompt into the reply.
    assert_eq!(propose(3, 11, &tokens, 7, 0), [15, 16, 17, 18, 19, 1, 12]);
    // The last `n` tokens never occurred before: nothing.
    assert!(propose(4, 11, &tokens, 4, 0).is_empty());
    assert!(propose(3, 11, &[12, 13, 14], 4, 0).is_empty());
    assert!(propose(3, 11, &tokens, 0, 0).is_empty());
}

#[test]
fn the_latest_occurrence_with_a_full_continuation_wins() {
    // `1 2` occurs three times, followed by 7, then 8, then 9 then the end.
    let tokens = [1, 2, 7, 7, 1, 2, 8, 8, 1, 2, 9, 1, 2];
    assert_eq!(propose(2, 0, &tokens, 1, 0), [9]);
    assert_eq!(propose(2, 0, &tokens, 2, 0), [9, 1]);
    // Four tokens after the latest would run past the pending token: the
    // occurrence before it has them.
    assert_eq!(propose(2, 0, &tokens, 4, 0), [8, 8, 1, 2]);
}

#[test]
fn a_short_repeat_continues_over_its_own_copy() {
    // Period 2, and no occurrence has five tokens after it.
    let tokens = [5, 6, 5, 6, 5, 6];
    assert_eq!(propose(2, 0, &tokens, 5, 0), [5, 6, 5, 6, 5]);
}

#[test]
fn reply_occurrences_need_the_longer_reply_match() {
    // `3 4` follows 9 9 in the prompt and 8 8 in the reply (prompt is 6).
    let tokens = [0, 9, 9, 3, 4, 5, 1, 8, 8, 3, 4, 6, 2, 8, 8, 3, 4];
    // Without the refinement the latest occurrence (in the reply) wins.
    assert_eq!(propose(2, 6, &tokens, 1, 0), [6]);
    // With it, the reply occurrence matches four tokens and still wins...
    assert_eq!(propose(2, 6, &tokens, 1, 4), [6]);
    // ...but needing five, it does not, and the prompt's two-token match is
    // taken.
    assert_eq!(propose(2, 6, &tokens, 1, 5), [5]);
}

#[test]
fn the_incremental_index_agrees_with_a_fresh_one() {
    // A pseudo-random text over a small vocabulary, so grams repeat.
    let mut x = 7u32;
    let tokens: Vec<u32> = (0..600)
        .map(|_| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (x >> 16) % 5
        })
        .collect();
    let mut index = Index::new(3, 100);
    let mut copies = 0;
    for len in 4..=tokens.len() {
        let grown = index.propose(ctx(&tokens[..len]), 4, 0);
        assert_eq!(grown, propose(3, 100, &tokens[..len], 4, 0), "len {len}");
        copies += usize::from(!grown.is_empty());
    }
    // Each of the 125 grams misses once, on its first occurrence.
    assert!(copies > 400, "{copies}");
    // A rewritten context (a rollback, then other tokens) is re-indexed.
    let mut other = tokens[..300].to_vec();
    other.extend(tokens[..300].iter().map(|t| t + 5));
    let rewritten = ctx(&other);
    assert_eq!(
        index.propose(rewritten, 4, 0),
        propose(3, 100, &other, 4, 0)
    );
}

#[test]
fn a_long_prompt_is_indexed_over_several_steps() {
    // A quoted span at the start of a prompt longer than one step's budget.
    let mut tokens: Vec<u32> = (0..INDEX_BUDGET as u32 + 1000).map(|t| t + 100).collect();
    tokens.extend([100, 101, 102]);
    let mut index = Index::new(3, tokens.len() - 3);
    // The first step indexes the budget, which already holds the span.
    assert_eq!(index.propose(ctx(&tokens), 2, 0), [103, 104]);
    assert_eq!(index.indexed, INDEX_BUDGET + 2);
    // The next one finishes the prompt.
    tokens.push(103);
    assert_eq!(index.propose(ctx(&tokens), 2, 0), [104, 105]);
    assert_eq!(index.indexed, tokens.len() - 1);
}

#[test]
fn a_copy_replaces_only_a_block_that_differs() {
    let (mut drafts, mut conf) = (vec![1, 2, 3, 4], vec![-0.1, -0.2, -0.3, -0.4]);
    // No copy, or one the block already starts with: the block stands.
    assert!(!merge(&[], &mut drafts, &mut conf));
    assert!(!merge(&[1, 2], &mut drafts, &mut conf));
    assert!(!merge(&[1, 2, 3, 4], &mut drafts, &mut conf));
    assert_eq!(
        (drafts.as_slice(), conf.len()),
        ([1, 2, 3, 4].as_slice(), 4)
    );
    // A copy that parts from it replaces it, confidences and all.
    assert!(merge(&[1, 2, 9, 9], &mut drafts, &mut conf));
    assert_eq!(drafts, [1, 2, 9, 9]);
    assert_eq!(conf, [COPY_CONF; 4]);
    // So does a shorter one (a narrowed copy after a miss).
    let mut unmeasured = Vec::new();
    assert!(merge(&[5], &mut drafts, &mut unmeasured));
    assert_eq!((drafts, unmeasured), (vec![5], vec![COPY_CONF]));
}

/// A request over prompt `prompt` that has emitted `reply` (its last token
/// pending) with DFlash2's `drafts` pending.
fn seq(prompt: &[u32], reply: &[u32], drafts: &[u32]) -> ActiveSeq {
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(reply.to_vec(), 8, None, 0);
    a.seq.tokens = prompt
        .iter()
        .chain(&reply[..reply.len() - 1])
        .copied()
        .collect();
    a.pending_drafts = drafts.to_vec();
    a.pending_draft_conf = vec![-0.5; drafts.len()];
    a
}

#[test]
fn an_offer_never_lengthens_the_block_and_counts_its_rounds() {
    let s = Settings {
        match_len: 3,
        ..on()
    };
    let prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
    let mut a = seq(&prompt, &[12, 13, 14], &[40, 41]);
    offer_with(&mut a, &s);
    assert_eq!(a.pending_drafts, [15, 16]);
    assert_eq!(a.draft_conf(), [COPY_CONF; 2]);
    // One offer a round: a second at the same context changes nothing.
    a.pending_drafts = vec![40, 41];
    offer_with(&mut a, &s);
    assert_eq!(a.pending_drafts, [40, 41]);
    // The verify kept one of two copies: a miss.
    settle(&mut a.spec_adapt.copy, &[COPY_CONF; 2], 2, 1);
    // DFlash2's own rounds are not copy rounds.
    settle(&mut a.spec_adapt.copy, &[-0.5; 2], 2, 0);
    let c = &a.spec_adapt.copy;
    assert_eq!((c.rounds, c.drafted, c.accepted, c.missed), (1, 2, 1, true));
}

#[test]
fn after_a_miss_the_next_copy_is_narrow_until_one_is_kept_whole() {
    let s = Settings {
        match_len: 3,
        miss_max: 1,
        ..on()
    };
    let prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
    let mut a = seq(&prompt, &[12, 13, 14], &[40, 41, 42]);
    a.spec_adapt.copy.missed = true;
    offer_with(&mut a, &s);
    assert_eq!(a.pending_drafts, [15]);
    settle(&mut a.spec_adapt.copy, &[COPY_CONF], 1, 1);
    assert!(!a.spec_adapt.copy.missed);
}

#[test]
fn draftless_requests_get_no_copy() {
    let s = Settings {
        match_len: 3,
        ..on()
    };
    let prompt = [10, 11, 12, 13, 14, 15];
    let mut a = seq(&prompt, &[12, 13, 14], &[]);
    offer_with(&mut a, &s);
    assert!(a.pending_drafts.is_empty());
    assert!(a.spec_adapt.copy.index.is_none());
}
