// SPDX-License-Identifier: AGPL-3.0-only

//! The prompt delta: what the head sends, what the worker rebuilds from it,
//! that the wire is unchanged with the switch off, and that a rank out of
//! step fails instead of prefilling another prompt. Then the trace line.

// The real model and byte transport of `idle_command_tests`.
#[allow(clippy::duplicate_mod)]
#[path = "impl_a2/idle_command_fixture.rs"]
mod wire;

use super::*;

/// The turns of a conversation as prefill commands carry them: a cold prompt
/// in two chunks, the next turn (the prompt, its answer, a new message) in
/// three, a regenerated answer (a shorter prefix, then other tokens), an
/// unrelated prompt, the first conversation again, and an empty prompt.
fn prompts() -> Vec<Vec<u32>> {
    let turn1: Vec<u32> = (100..140).collect();
    let turn2: Vec<u32> = turn1.iter().copied().chain(900..917).collect();
    let regenerated: Vec<u32> = turn2[..47].iter().copied().chain(700..703).collect();
    vec![
        turn1.clone(),
        turn1.clone(),
        turn2.clone(),
        turn2.clone(),
        turn2.clone(),
        turn2[..47].to_vec(),
        regenerated,
        vec![5, 6, 7],
        turn1,
        vec![],
    ]
}

#[test]
fn the_hash_tells_length_and_content_apart() {
    assert_ne!(prompt_hash(&[]), prompt_hash(&[0]));
    assert_ne!(prompt_hash(&[0]), prompt_hash(&[0, 0]));
    assert_ne!(prompt_hash(&[1, 2]), prompt_hash(&[2, 1]));
    assert_eq!(prompt_hash(&[1, 2, 3]), prompt_hash(&[1, 2, 3]));
}

#[test]
fn the_shared_prefix_stops_at_the_first_difference_or_the_shorter_end() {
    assert_eq!(common_prefix(&[], &[1, 2]), 0);
    assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4, 3]), 2);
    assert_eq!(common_prefix(&[1, 2, 3], &[1, 2]), 2);
    assert_eq!(common_prefix(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(common_prefix(&[9], &[1]), 0);
}

/// The worker's side of one announced prompt: its words, then its suffix.
fn receive(
    worker: &mut PromptMirrors,
    words: [u32; ANNOUNCE_WORDS],
    p: &[u32],
    from: usize,
) -> usize {
    let n = worker.suffix_len(words, p.len()).unwrap();
    assert_eq!(n, p.len() - from);
    assert_eq!(*worker.rebuild(words, &p[from..]).unwrap(), p);
    n
}

/// Head and worker mirrors through every prompt in one slot: the worker
/// rebuilds each one, and only the tokens the slot's previous prompt lacked
/// cross the wire.
#[test]
fn the_worker_rebuilds_every_prompt_from_what_the_last_one_lacked() {
    let (mut head, mut worker) = (PromptMirrors::default(), PromptMirrors::default());
    let mut sent = Vec::new();
    for p in prompts() {
        let (words, from) = head.advance(0, &p);
        assert_eq!(words[..3], [0, 0, from as u32]);
        sent.push(receive(&mut worker, words, &p, from));
    }
    // cold 40; its second chunk 0; the turn's 17 new tokens; two more chunks
    // 0; the shorter prefix 0; the regenerated tail 3; an unrelated prompt in
    // full; the first conversation again in full; the empty prompt.
    assert_eq!(sent, [40, 0, 17, 0, 0, 0, 3, 3, 40, 0]);
}

/// Two conversations prefill in turn in slots 0 and 1: each slot's chunk
/// commands share its own prompt. Their next turns land in the other slot
/// and still share the conversation's previous prompt, wherever it ran.
#[test]
fn a_prompt_extends_whichever_slot_shares_the_most() {
    let (mut head, mut worker) = (PromptMirrors::default(), PromptMirrors::default());
    let a: Vec<u32> = (100..160).collect();
    let b: Vec<u32> = (100..110).chain(500..540).collect(); // shares a 10-token preamble with `a`
    let mut step = |slot: usize, p: &[u32]| {
        let (words, from) = head.advance(slot, p);
        assert_eq!(words[0], slot as u32);
        (words[1], receive(&mut worker, words, p, from))
    };
    // (base slot, tokens sent)
    assert_eq!(step(0, &a), (0, 60));
    assert_eq!(step(1, &b), (0, 40), "the preamble comes from slot 0");
    for _ in 0..3 {
        assert_eq!(step(0, &a), (0, 0));
        assert_eq!(step(1, &b), (1, 0));
    }
    let a2: Vec<u32> = a.iter().copied().chain(900..905).collect();
    let b2: Vec<u32> = b.iter().copied().chain(950..953).collect();
    assert_eq!(step(1, &a2), (0, 5), "the next turn of `a`, now in slot 1");
    assert_eq!(step(0, &b2), (0, 43), "slot 1 no longer holds `b`");
    assert_eq!(
        step(2, &b2),
        (0, 0),
        "a third slot takes it whole from slot 0"
    );
    // A tie goes to the slot's own mirror.
    assert_eq!(step(2, &b2), (2, 0));
}

#[test]
fn a_rank_out_of_step_fails_before_it_prefills() {
    let p: Vec<u32> = (0..32).collect();
    let mut head = PromptMirrors::default();
    head.advance(0, &p[..20]);
    let (words, from) = head.advance(0, &p);
    assert_eq!((&words[..3], from), (&[0, 0, 20][..], 20));

    // A worker that missed the first command holds nothing to share.
    let behind = PromptMirrors::default();
    let e = behind.suffix_len(words, p.len()).unwrap_err();
    assert!(format!("{e:#}").contains("out of step"), "{e:#}");

    // A worker whose slot holds another prompt under the shared length
    // rebuilds another prompt: the hash catches it.
    let mut other = PromptMirrors::default();
    other.advance(0, &[vec![9; 20], vec![1]].concat());
    assert_eq!(other.suffix_len(words, p.len()).unwrap(), 12);
    let e = other.rebuild(words, &p[from..]).unwrap_err();
    assert!(format!("{e:#}").contains("out of step"), "{e:#}");

    // More shared tokens than the prompt has, and a word that is no slot.
    let e = head.suffix_len([0, 0, 3, 0], 2).unwrap_err();
    assert!(format!("{e:#}").contains("out of step"), "{e:#}");
    let e = head.suffix_len([1 << 20, 0, 0, 0], 2).unwrap_err();
    assert!(format!("{e:#}").contains("out of step"), "{e:#}");
}

/// The payload words the head put on the wire, in order.
fn words_sent(f: &wire::Fixture) -> Vec<u32> {
    let cmd = f.model.ep_cmd_buf.0;
    f.events()
        .iter()
        .filter_map(|e| match e {
            wire::Event::H2d(ptr, bytes) if *ptr != cmd => Some(bytes.clone()),
            _ => None,
        })
        .flat_map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The real head and worker entry points over the real broadcast helpers: a
/// worker fed exactly the words the head sent rebuilds every prompt and
/// leaves no word unread.
#[test]
fn actual_head_and_worker_carry_the_delta() {
    let mut head = wire::Fixture::new(0, true, &[]);
    head.model.warm.prompt_delta = true;
    for p in prompts() {
        head.model.ep_broadcast_prompt_dispatch(0, &p).unwrap();
    }
    let sent = words_sent(&head);
    // The announce words of each command, plus the tokens the mirror test
    // counts.
    assert_eq!(
        sent.len(),
        ANNOUNCE_WORDS * prompts().len() + 40 + 17 + 3 + 3 + 40
    );

    let mut worker = wire::Fixture::new(1, true, &sent);
    worker.model.warm.prompt_delta = true;
    for p in prompts() {
        assert_eq!(*worker.model.ep_recv_prompt(p.len()).unwrap(), p);
    }
    assert!(worker.record.words.lock().is_empty());
}

/// With the switch off both entry points are the bulk broadcast they
/// replaced: the same device copies, collectives and syncs, in order.
#[test]
fn actual_wire_is_the_bulk_broadcast_with_the_switch_off() {
    let p: Vec<u32> = (100..140).collect();
    let (head, plain) = (
        wire::Fixture::new(0, true, &[]),
        wire::Fixture::new(0, true, &[]),
    );
    assert!(!head.model.warm.prompt_delta);
    for _ in 0..2 {
        head.model.ep_broadcast_prompt_dispatch(3, &p).unwrap();
        plain.model.ep_broadcast_tokens(&p).unwrap();
    }
    assert_eq!(words_sent(&head), [p.clone(), p.clone()].concat());
    // The fixtures' device addresses differ; the payloads and their order do not.
    assert_eq!(words_sent(&head), words_sent(&plain));
    assert_eq!(head.events().len(), plain.events().len());

    let words = [p.clone(), p.clone()].concat();
    let worker = wire::Fixture::new(1, true, &words);
    let plain = wire::Fixture::new(1, true, &words);
    for _ in 0..2 {
        assert_eq!(*worker.model.ep_recv_prompt(p.len()).unwrap(), p);
        assert_eq!(plain.model.ep_broadcast_tokens(&[0; 40]).unwrap(), p);
    }
    assert_eq!(worker.events().len(), plain.events().len());
    assert!(worker.record.words.lock().is_empty());
}

/// A worker whose mirror is behind the head's reads the two announce words
/// and fails; it does not wait for tokens the head never sends.
#[test]
fn actual_worker_behind_the_head_fails_on_the_announce_words() {
    let mut worker = wire::Fixture::new(1, true, &[0, 0, 20, 0xdead_beef]);
    worker.model.warm.prompt_delta = true;
    let e = worker.model.ep_recv_prompt(32).unwrap_err();
    assert!(format!("{e:#}").contains("out of step"), "{e:#}");
    assert!(worker.record.words.lock().is_empty());
}

#[test]
fn the_trace_line_sums_a_requests_chunks_and_forgets_them() {
    let ms = Duration::from_millis;
    let mut warm = WarmTurn::from_env().unwrap();
    warm.trace = true;
    let began = Instant::now();
    let shape = || RequestShape {
        rank: 1,
        prompt: 40,
        matched: 32,
        restored: 16,
    };
    warm.charge_transfer(began - ms(2));
    // Two cached chunks: no rows, a lookup and a block vote.
    let cached = [ms(0), ms(0), ms(3), ms(1), ms(0), ms(0), ms(0)];
    assert_eq!(warm.note_chunk(7, began, 0, cached, None), None);
    assert_eq!(warm.note_chunk(7, began, 0, cached, None), None);
    // Another slot's chunk in between stays out of this request's line.
    assert_eq!(warm.note_chunk(2, began, 99, [ms(50); 7], None), None);
    let pass = [ms(17), ms(2), ms(0), ms(1), ms(1), ms(100), ms(5)];
    assert_eq!(warm.note_chunk(7, began, 8, pass, None), None);
    let line = warm.note_chunk(7, began, 16, pass, Some(shape())).unwrap();
    assert!(
        line.starts_with(
            "warm-turn rank=1 slot=7 tokens=40 matched=32 restored=16 chunks=4 \
             cached_chunks=2 rows=24 ms: transfer="
        ),
        "{line}"
    );
    for span in [
        "zero=34.0",
        "embed=4.0",
        "lookup=6.0",
        "blocks=4.0",
        "meta=2.0",
        "forward=200.0",
        "finish=10.0",
    ] {
        assert!(line.contains(span), "{span}: {line}");
    }
    // The transfer was charged to the first chunk noted after it (about
    // 2 ms), and the wall clock starts with it.
    let ms_of = |key: &str| -> f64 {
        let at = line.find(key).unwrap() + key.len();
        line[at..].split(' ').next().unwrap().parse().unwrap()
    };
    assert!((2.0..50.0).contains(&ms_of("transfer=")), "{line}");
    assert!(ms_of("wall=") >= ms_of("transfer="), "{line}");
    // The request is gone; the other slot's chunk is still pending.
    let next = warm.note_chunk(7, began, 1, pass, Some(shape())).unwrap();
    assert!(next.contains("chunks=1 cached_chunks=0 rows=1 "), "{next}");
    let other = warm.note_chunk(2, began, 1, pass, Some(shape())).unwrap();
    assert!(
        other.contains("chunks=2 cached_chunks=0 rows=100 "),
        "{other}"
    );

    // With the switch off no transfer time is kept.
    warm.trace = false;
    warm.charge_transfer(began - ms(2));
    assert_eq!(*warm.transfer.lock(), (Duration::ZERO, None));
}

#[test]
fn zero_rows_is_off_trim_or_check_and_refuses_anything_else() {
    use ZeroRows::*;
    assert_eq!(ZeroRows::parse(None, None).unwrap(), Off);
    assert_eq!(ZeroRows::parse(Some("0"), None).unwrap(), Off);
    assert_eq!(ZeroRows::parse(Some("1"), None).unwrap(), Trim(2048));
    assert_eq!(ZeroRows::parse(Some("check"), None).unwrap(), Check(2048));
    assert_eq!(
        ZeroRows::parse(Some("1"), Some("4096")).unwrap(),
        Trim(4096)
    );
    assert_eq!(
        ZeroRows::parse(Some("check"), Some("256")).unwrap(),
        Check(256)
    );
    // A floor without the switch is only read.
    assert_eq!(ZeroRows::parse(None, Some("512")).unwrap(), Off);
    for bad in ["true", "2", "", "CHECK"] {
        let e = ZeroRows::parse(Some(bad), None).unwrap_err();
        assert!(
            format!("{e:#}").contains("ATLAS_GLM_ZERO_ROWS must"),
            "{e:#}"
        );
    }
    // A floor under a batched verify's rows could not cover a decode step.
    for bad in ["255", "0", "rows", "-1"] {
        let e = ZeroRows::parse(Some("1"), Some(bad)).unwrap_err();
        assert!(format!("{e:#}").contains("_FLOOR must"), "{e:#}");
    }
}
