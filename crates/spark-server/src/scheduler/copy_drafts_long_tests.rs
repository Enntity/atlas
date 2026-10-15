// SPDX-License-Identifier: AGPL-3.0-only

//! Long copies (`ATLAS_DFLASH_COPY_LONG_MAX`).

use super::*;

/// A prompt of 40 distinct tokens (100..140) re-emitted from its start: the
/// reply has copied `copied` of them, the last pending.
fn reemit(copied: usize) -> (Vec<u32>, Vec<u32>) {
    let prompt: Vec<u32> = (100..140).collect();
    (prompt.clone(), prompt[..copied].to_vec())
}

fn long_on(long_max: usize) -> Settings {
    Settings {
        match_len: 3,
        miss_max: 3,
        reply_match: 16,
        long_max,
        ..on()
    }
}

#[test]
fn after_a_copy_kept_whole_a_prompt_copy_goes_long() {
    let s = long_on(15);
    let (prompt, reply) = reemit(5);
    let mut a = seq(&prompt, &reply, &[1; 7]);
    // No copy round settled yet: DFlash2's block length, as shipped.
    offer_with(&mut a, &s);
    assert_eq!(a.pending_drafts, (105..112).collect::<Vec<_>>());
    settle(&mut a.spec_adapt.copy, &[COPY_CONF; 7], 7, 7);
    // Kept whole: the next copy reads 15 past DFlash2's 7 drafts.
    let (_, reply) = reemit(13);
    let mut b = seq(&prompt, &reply, &[1; 7]);
    b.spec_adapt.copy = std::mem::take(&mut a.spec_adapt.copy);
    offer_with(&mut b, &s);
    assert_eq!(b.pending_drafts, (113..128).collect::<Vec<_>>());
    assert_eq!(b.draft_conf(), [COPY_CONF; 15]);
    // A batch or a prefill ride counts it as DFlash2's block.
    assert_eq!(ride_len(&b), MAX_DRAFTS);
    // Missed (kept 4 of 15): the next is narrow (miss_max), not long.
    settle(&mut b.spec_adapt.copy, &[COPY_CONF; 15], 15, 4);
    let (_, reply) = reemit(18);
    let mut c = seq(&prompt, &reply, &[1; 7]);
    c.spec_adapt.copy = std::mem::take(&mut b.spec_adapt.copy);
    offer_with(&mut c, &s);
    assert_eq!(c.pending_drafts, [118, 119, 120]);
}

#[test]
fn a_long_copy_near_the_prompts_end_continues_over_the_reply() {
    // 31 past the occurrence runs off the prompt: the copy reads on into
    // the reply (the occurrence still starts in the prompt).
    let s = long_on(31);
    let (prompt, reply) = reemit(20);
    let mut a = seq(&prompt, &reply, &[1; 7]);
    a.spec_adapt.copy.rounds = [1, 0];
    offer_with(&mut a, &s);
    let mut want: Vec<u32> = (120..140).collect();
    want.extend(100..111);
    assert_eq!(a.pending_drafts, want);
}

#[test]
fn copies_of_the_reply_never_go_long() {
    let s = long_on(15);
    // 20 tokens that occur only in the reply, repeated.
    let prompt = [7u32, 8, 9];
    let body: Vec<u32> = (200..220).collect();
    let mut reply = body.clone();
    reply.extend(&body[..17]);
    let mut a = seq(&prompt, &reply, &[1; 7]);
    a.spec_adapt.copy.rounds = [0, 1];
    offer_with(&mut a, &s);
    assert_eq!(a.pending_drafts, [217, 218, 219, 200, 201, 202, 203]);
    assert_eq!(a.draft_conf(), [COPY_REPLY_CONF; 7]);
}

#[test]
fn long_copies_off_offer_exactly_as_shipped() {
    // Same request, same kept-whole history: long_max 0 is the shipped offer.
    let (prompt, reply) = reemit(13);
    for drafts in [&[1u32; 7][..], &[1; 3][..]] {
        let mut a = seq(&prompt, &reply, drafts);
        a.spec_adapt.copy.rounds = [1, 0];
        let mut b = seq(&prompt, &reply, drafts);
        b.spec_adapt.copy.rounds = [1, 0];
        offer_with(&mut a, &long_on(0));
        offer_with(
            &mut b,
            &Settings {
                long_max: 0,
                ..long_on(15)
            },
        );
        assert_eq!(a.pending_drafts, b.pending_drafts);
        assert_eq!(
            a.pending_drafts,
            (113..113 + drafts.len() as u32).collect::<Vec<_>>()
        );
        assert_eq!(ride_len(&a), drafts.len());
    }
}

/// A long copy verified by the per-sequence tail and kept only in part: the
/// sequence rolls back to the accepted prefix plus the target's token, the
/// model commits that many rows, and the round books a miss.
#[test]
fn a_long_copy_kept_in_part_rolls_back_to_the_accepted_prefix() {
    let s = long_on(15);
    let (prompt, reply) = reemit(13);
    let (mut a, _caller) = seq_with_caller(&prompt, &reply, &[1; 7]);
    a.pending_draft_conf.clear();
    a.spec_adapt.copy.rounds = [1, 0];
    (a.finished, a.min_tokens, a.remaining) = (false, 0, 64);
    offer_with(&mut a, &s);
    let (drafts, conf) = a.take_drafts();
    assert_eq!(drafts.len(), 15);
    let pre = a.seq.tokens.len();
    let tokens: Vec<u32> = std::iter::once(a.last_token)
        .chain(drafts.clone())
        .collect();
    a.seq.tokens.extend_from_slice(&tokens);
    a.seq.seq_len = a.seq.tokens.len();
    // The target keeps nine copies (113..=121), then edits: 7 where 122 was.
    let mut verified: Vec<u32> = (113..122).collect();
    verified.push(7);
    verified.extend([9; 6]);
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    crate::scheduler::verify_dflash_step::verify_dflash_tail(
        &StubModel::default(),
        &mut a,
        &sched,
        &drafts,
        &conf,
        drafts.len(),
        &ctx,
        true,
        &tokens,
        verified,
        false,
        0.0,
        true,
        true,
    );
    let mut want = reply.clone();
    want.extend(113..122);
    want.push(7);
    assert_eq!(a.output_tokens, want);
    assert_eq!(a.last_token, 7);
    // Rolled back to the prefix, the nine copies and the target's token.
    assert_eq!(a.seq.seq_len, pre + 10);
    assert_eq!(a.seq.tokens.len(), pre + 10);
    let commits = COMMITS.with(|c| std::mem::take(&mut *c.borrow_mut()));
    assert_eq!(commits, [(pre + 10, 10, 16)]);
    let c = &a.spec_adapt.copy;
    assert_eq!(
        (c.rounds, c.drafted, c.accepted, c.missed),
        ([2, 0], [15, 0], [9, 0], true)
    );
}

#[test]
fn a_long_copy_needs_the_rows_its_lone_verify_may_hold_now() {
    // `Model::lone_verify_draft_capacity`: 31 alone, DFlash2's 7 once another
    // sequence holds the rows a long verify borrows.
    let s = long_on(31);
    assert_eq!(s.lone(31), s);
    assert_eq!(s.lone(15).long_max, 15);
    assert_eq!(s.lone(MAX_DRAFTS).long_max, 0);
    assert_eq!(long_on(0).lone(31).long_max, 0);
    let (prompt, reply) = reemit(5);
    let offered = |cap| {
        let mut a = seq(&prompt, &reply, &[1; 7]);
        a.spec_adapt.copy.rounds = [1, 0];
        offer_with(&mut a, &s.lone(cap));
        a.pending_drafts
    };
    assert_eq!(offered(31), (105..136).collect::<Vec<_>>());
    assert_eq!(offered(MAX_DRAFTS), (105..112).collect::<Vec<_>>());
}
