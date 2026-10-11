// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `stop_match` (split out to keep the module under the
//! 500-LoC cap).

use super::*;
use std::sync::Arc;

fn stops(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn earliest_completion_wins_across_multiple_stops() {
    let s = stops(&["stop", "WORLD"]);
    // "WORLD" completes first even though "stop" is listed first.
    assert_eq!(
        find_stop_string("hello WORLD stop it", 0, &s),
        Some((6, "WORLD"))
    );
}

#[test]
fn earliest_end_beats_earlier_start() {
    // Token-by-token arrival hits "bc" (end 3) before "abcd" (end 4),
    // although "abcd" starts first (vLLM detokenizer.py:431).
    let s = stops(&["abcd", "bc"]);
    assert_eq!(find_stop_string("xabcd", 0, &s), Some((2, "bc")));
}

#[test]
fn equal_end_tie_goes_to_list_order() {
    assert_eq!(
        find_stop_string("AB", 0, &stops(&["AB", "B"])),
        Some((0, "AB"))
    );
    assert_eq!(
        find_stop_string("AB", 0, &stops(&["B", "AB"])),
        Some((1, "B"))
    );
}

#[test]
fn search_from_skips_prefix_and_never_splits_a_char() {
    let s = stops(&["é"]);
    // Byte 1 is inside the first 'é' (2 bytes): snapped down to 0.
    assert_eq!(find_stop_string("éé", 1, &s), Some((0, "é")));
    assert_eq!(find_stop_string("é x é", 3, &s), Some((5, "é")));
    assert_eq!(find_stop_string("abc", 99, &s), None);
}

#[test]
fn empty_stop_strings_are_ignored() {
    assert_eq!(find_stop_string("abc", 0, &stops(&[""])), None);
    assert_eq!(
        find_stop_string("abc", 0, &stops(&["", "c"])),
        Some((2, "c"))
    );
}

#[test]
fn single_token_stop_maps_back_to_its_string() {
    let map = vec![
        (7u32, "END".to_string()),
        (9, "STOP".to_string()),
        (7, "dup".into()),
    ];
    assert_eq!(stop_string_for_token(&map, 9), Some("STOP"));
    assert_eq!(stop_string_for_token(&map, 7), Some("END"));
    assert_eq!(stop_string_for_token(&map, 1), None);
}

#[test]
fn single_token_stop_hit_is_reported() {
    let map = vec![(7u32, "END".to_string())];
    // Old code: the stop token was dropped before decoding, so no text
    // match existed and `matched_stop` stayed None.
    assert_eq!(
        resolve_matched_stop(None, Some(7), &map).as_deref(),
        Some("END")
    );
    // Model EOS (not a client stop) and non-token finishes report nothing.
    assert_eq!(resolve_matched_stop(None, Some(2), &map), None);
    assert_eq!(resolve_matched_stop(None, None, &map), None);
    // A text match found earlier in the content wins.
    assert_eq!(
        resolve_matched_stop(Some("Obs:".into()), Some(7), &map).as_deref(),
        Some("Obs:")
    );
}

#[test]
fn holdback_stream_cuts_at_stop_and_flushes_tail_otherwise() {
    let mut h = StopHoldback::new(stops(&["Obs:"]), 0, false);
    let mut out = h.push("hello Ob");
    assert!(!h.triggered());
    out += &h.push("s: tail");
    assert!(h.triggered());
    out += &h.push("more");
    out += &h.finish();
    assert_eq!(out, "hello ");

    let mut h = StopHoldback::new(stops(&["Obs:"]), 0, false);
    let mut out = h.push("hello O");
    out += &h.finish();
    assert_eq!(out, "hello O", "unmatched hold-back must be released");

    let mut h = StopHoldback::new(Vec::new(), 0, false);
    assert_eq!(h.push("Obs:"), "Obs:");
    assert_eq!(h.finish(), "");
}

/// Toy decoder: token id = one char ('a' + id), 100 = a skipped special.
fn toy_decode() -> Arc<StopDecodeFn> {
    Arc::new(|ids: &[u32]| {
        ids.iter()
            .filter(|&&t| t != 100)
            .map(|&t| char::from(b'a' + t as u8))
            .collect()
    })
}

fn ids(s: &str) -> Vec<u32> {
    s.bytes().map(|b| (b - b'a') as u32).collect()
}

#[test]
fn watch_detects_stop_spanning_token_boundaries() {
    let mut w = StopStringWatch::new(&stops(&["cde"]), 0, toy_decode()).unwrap();
    let mut out = Vec::new();
    for (i, t) in ids("abcdef").into_iter().enumerate() {
        out.push(t);
        let hit = w.observe(&out, false);
        // "cde" spans three single-char tokens; completes at index 4.
        assert_eq!(hit, i == 4, "token {i}");
        if hit {
            break;
        }
    }
}

#[test]
fn watch_hit_is_sticky() {
    let mut w = StopStringWatch::new(&stops(&["ab"]), 0, toy_decode()).unwrap();
    assert!(w.observe(&ids("ab"), false));
    // Same tokens again (post-sampling recheck): still a hit.
    assert!(w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("ab"), true));
}

#[test]
fn watch_detects_stop_in_a_multi_token_step() {
    // Speculative decode can append several tokens between checkpoints.
    let mut w = StopStringWatch::new(&stops(&["cd"]), 0, toy_decode()).unwrap();
    assert!(!w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("abcde"), false));
}

#[test]
fn watch_ignores_reasoning_tokens() {
    let mut w = StopStringWatch::new(&stops(&["cd"]), 0, toy_decode()).unwrap();
    let mut out = ids("cd");
    // Stop text inside the thinking block must not end generation.
    assert!(!w.observe(&out, true));
    out.extend(ids("ab"));
    assert!(!w.observe(&out, false));
    out.extend(ids("cd"));
    assert!(w.observe(&out, false));
}

#[test]
fn watch_window_spans_skipped_special_tokens() {
    let mut w = StopStringWatch::new(&stops(&["ab"]), 0, toy_decode()).unwrap();
    let mut out = ids("a");
    assert!(!w.observe(&out, false));
    out.push(100);
    out.push(100);
    assert!(!w.observe(&out, false));
    out.extend(ids("b"));
    assert!(w.observe(&out, false));
}

#[test]
fn watch_tolerates_rollback_truncation() {
    let mut w = StopStringWatch::new(&stops(&["xy"]), 0, toy_decode()).unwrap();
    assert!(!w.observe(&ids("abcdx"), false));
    // Rollback dropped "cdx"; regrowth must rescan, not index past len.
    assert!(!w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("abxy"), false));
}

#[test]
fn watch_needs_a_non_empty_stop() {
    assert!(StopStringWatch::new(&[], 0, toy_decode()).is_none());
    assert!(StopStringWatch::new(&stops(&[""]), 0, toy_decode()).is_none());
}

/// vLLM `_verify_args` (sampling_params.py:763): an empty stop string is a
/// 400, alone or among others. Atlas used to accept it and silently skip it.
#[test]
fn empty_stop_string_is_rejected() {
    assert_eq!(
        validate_stop_params(&stops(&[""]), 0, 16).unwrap_err().0,
        "stop"
    );
    assert_eq!(
        validate_stop_params(&stops(&["ok", ""]), 0, 16)
            .unwrap_err()
            .0,
        "stop"
    );
    assert!(validate_stop_params(&stops(&["ok"]), 0, 16).is_ok());
    assert!(validate_stop_params(&[], 0, 16).is_ok());
}

/// vLLM `_verify_args` (sampling_params.py:729): min_tokens may equal but
/// not exceed max_tokens.
#[test]
fn min_tokens_above_max_tokens_is_rejected() {
    assert!(validate_stop_params(&[], 16, 16).is_ok());
    let (param, msg) = validate_stop_params(&[], 17, 16).unwrap_err();
    assert_eq!(param, "min_tokens");
    assert!(msg.contains("max_tokens=16"), "{msg}");
}

/// Client `stop_token_ids` join the single-token stop strings in the
/// scheduler's stop set (sorted, deduplicated); before, they were dropped.
#[test]
fn stop_token_ids_merge_into_the_stop_set() {
    assert_eq!(merge_stop_token_ids(vec![9, 3], &[7, 3, 7]), vec![3, 7, 9]);
    assert_eq!(merge_stop_token_ids(Vec::new(), &[5]), vec![5]);
    assert!(merge_stop_token_ids(Vec::new(), &[]).is_empty());
}

// ── streaming hold-back (moved from chat_stream::handle_token) ──

/// One hold-back step with vLLM-default options (no floor, cut before).
fn hold(st: &mut HoldbackState, delta: &str, stops: &[String], buffer_len: usize) -> String {
    let p = HoldbackParams {
        stops,
        buffer_len,
        include_stop_str: false,
    };
    apply_stop_string_holdback(delta, false, p, st).0
}

/// Stop string spanning a chunk boundary must not leak the partial prefix
/// in the first delta. When the suffix arrives in the next chunk the full
/// output up to (but excluding) the stop string is emitted; the stop
/// literal itself is consumed.
#[test]
fn stop_string_spanning_chunk_boundary_does_not_leak() {
    let s = stops(&["<|im_start|>"]);
    let buffer_len = holdback_len(&s, false);
    assert_eq!(buffer_len, 11);
    let mut st = HoldbackState::default();
    // Entirely inside the hold-back window: nothing emitted.
    assert_eq!(hold(&mut st, "hello ", &s, buffer_len), "");
    assert!(!st.triggered);
    // Partial stop string: acc is 13 bytes, emit up to 13 - 11 = 2.
    let out = hold(&mut st, "<|im_st", &s, buffer_len);
    assert_eq!(out, "he");
    assert!(!st.triggered);
    // Completes the stop at byte 6: emit "llo " and consume the literal.
    assert_eq!(hold(&mut st, "art|>", &s, buffer_len), "llo ");
    assert_eq!(st.acc, "hello ");
    assert!(st.triggered);
}

/// No stop strings: `buffer_len = 0` and the hold-back is a pass-through.
#[test]
fn no_stop_strings_is_zero_behavior_change() {
    let mut st = HoldbackState::default();
    assert_eq!(holdback_len(&[], false), 0);
    assert_eq!(hold(&mut st, "hello ", &[], 0), "hello ");
    assert_eq!(hold(&mut st, "<|im_start|>", &[], 0), "<|im_start|>");
    assert!(!st.triggered);
}

/// Two stops in one delta: the one that COMPLETES first wins even if
/// another starts earlier (vLLM check_stop_strings).
#[test]
fn earliest_completion_wins_within_one_delta() {
    let s = stops(&["abcd", "bc"]);
    let mut st = HoldbackState::default();
    let p = HoldbackParams {
        stops: &s,
        buffer_len: 3,
        include_stop_str: false,
    };
    let (out, matched) = apply_stop_string_holdback("xabcd", false, p, &mut st);
    assert!(st.triggered);
    assert_eq!(matched.as_deref(), Some("bc"));
    assert_eq!(out, "xa");
}

/// Multi-byte UTF-8 inside the hold-back window is never sliced
/// mid-codepoint: "aébc" with a 3-byte hold-back emits only "a".
#[test]
fn utf8_boundary_safety_in_holdback() {
    let mut st = HoldbackState::default();
    let out = hold(&mut st, "aébc", &stops(&["STOP"]), 3);
    assert_eq!(out, "a");
    assert!(!st.triggered);
}

// ── vLLM min_tokens: stops dormant over the first min_tokens tokens ──

/// A match counts only when it ENDS past the floor; one straddling the floor
/// fires (vLLM advances `stop_check_offset`, detokenizer.py:194-196).
#[test]
fn floor_admits_only_matches_ending_past_it() {
    let s = stops(&["STOP"]);
    let hit = |floor| find_stop_hit("abcSTOP x STOP", 0, floor, &s).map(|h| (h.start, h.end));
    assert_eq!(hit(0), Some((3, 7)));
    assert_eq!(hit(5), Some((3, 7)), "straddles the floor");
    assert_eq!(hit(7), Some((10, 14)), "first match ends on the floor");
    assert_eq!(hit(14), None);
    // Earliest END among stops that pass the floor.
    let s = stops(&["x", "STOP"]);
    let h = find_stop_hit("abcSTOP x STOP", 0, 7, &s).unwrap();
    assert_eq!((h.start, h.stop), (8, "x"));
    // Snapping never panics on a multi-byte floor.
    assert_eq!(
        find_stop_hit("ééé", 0, 3, &stops(&["é"])).map(|h| h.start),
        Some(2)
    );
}

/// Blocking truncation: the old matcher cut at the first "STOP" even
/// when it came from the first `min_tokens` tokens.
#[test]
fn truncate_at_stop_respects_the_floor() {
    let s = stops(&["STOP"]);
    assert_eq!(
        truncate_at_stop("a STOP b STOP".into(), &s, 0, false),
        ("a ".to_string(), Some("STOP".to_string()))
    );
    assert_eq!(
        truncate_at_stop("a STOP b STOP".into(), &s, 6, false),
        ("a STOP b ".to_string(), Some("STOP".to_string()))
    );
    assert_eq!(
        truncate_at_stop("a STOP".into(), &s, 6, false),
        ("a STOP".to_string(), None)
    );
}

/// Streaming: a stop produced by a token within `min_tokens` is not a
/// match; the same stop after the floor is.
#[test]
fn holdback_ignores_stops_within_min_tokens() {
    let s = stops(&["STOP"]);
    let p = HoldbackParams {
        stops: &s,
        buffer_len: holdback_len(&s, false),
        include_stop_str: false,
    };
    let mut st = HoldbackState::default();
    let mut out = apply_stop_string_holdback("STOP", true, p, &mut st).0;
    assert!(!st.triggered);
    out += &apply_stop_string_holdback(" x ", false, p, &mut st).0;
    assert!(!st.triggered);
    let (tail, matched) = apply_stop_string_holdback("STOP!", false, p, &mut st);
    out += &tail;
    assert!(st.triggered);
    assert_eq!(matched.as_deref(), Some("STOP"));
    assert_eq!(out, "STOP x ");
}

/// Legacy completions stream: `min_tokens` counts pushes (tokens).
#[test]
fn stop_holdback_counts_tokens_for_min_tokens() {
    let mut h = StopHoldback::new(stops(&["b"]), 1, false);
    let mut out = h.push("ab");
    assert!(!h.triggered(), "token 1 is within min_tokens");
    out += &h.push("cb");
    assert!(h.triggered());
    out += &h.finish();
    assert_eq!(out, "abc");
}

/// Blocking scheduler watch: dormant through `min_tokens`, then a match
/// must end past their text (one straddling it fires).
#[test]
fn watch_respects_min_tokens() {
    let mut w = StopStringWatch::new(&stops(&["cd"]), 2, toy_decode()).unwrap();
    let mut out = ids("cd");
    assert!(!w.observe(&out, false), "within min_tokens");
    out.extend(ids("x"));
    assert!(!w.observe(&out, false), "match ends inside the floor");
    out.extend(ids("cd"));
    assert!(w.observe(&out, false));

    let mut w = StopStringWatch::new(&stops(&["bc"]), 2, toy_decode()).unwrap();
    assert!(!w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("abc"), false), "straddles the floor");
}

/// The min_tokens floor keeps its distance from the end of the text when
/// mapped onto post-processed content (reasoning split off / trimmed).
#[test]
fn content_floor_maps_from_the_end() {
    // full "RRRR</think>abcdef" (18), floor after 14 bytes, content "abcdef".
    assert_eq!(content_floor(6, 18, 14), 2);
    // Floor inside the reasoning: no content floor.
    assert_eq!(content_floor(6, 18, 3), 0);
    // Floor at the very end: nothing in content can match.
    assert_eq!(content_floor(6, 18, 18), 6);
}

// ── vLLM include_stop_str_in_output ──

/// Blocking: the cut moves to the END of the earliest match (vLLM
/// check_stop_strings `include_in_output`); the old strip always cut before.
#[test]
fn include_keeps_the_matched_stop_string() {
    let s = stops(&["STOP"]);
    assert_eq!(
        truncate_at_stop("a STOP b STOP".into(), &s, 0, true),
        ("a STOP".to_string(), Some("STOP".to_string()))
    );
    assert_eq!(
        truncate_at_stop("a STOP".into(), &s, 0, true),
        ("a STOP".to_string(), Some("STOP".to_string()))
    );
}

/// Streaming: nothing is held back (vLLM `stop_buffer_length = 0`) and the
/// match is emitted through its end.
#[test]
fn include_streams_without_holdback_through_the_match() {
    let s = stops(&["Obs:"]);
    assert_eq!(holdback_len(&s, true), 0);
    let mut h = StopHoldback::new(s, 0, true);
    assert_eq!(h.push("hello Ob"), "hello Ob");
    assert_eq!(h.push("s: tail"), "s:");
    assert!(h.triggered());
    assert_eq!(h.push("more"), "");
    assert_eq!(h.finish(), "");
}

/// Only a CLIENT stop token's text is returned, and only when asked.
#[test]
fn included_stop_token_is_client_only() {
    let client = [7u32, 9];
    assert_eq!(included_stop_token(true, Some(9), &client), Some(9));
    assert_eq!(included_stop_token(false, Some(9), &client), None);
    // Model EOS (not in the client set) and non-token finishes.
    assert_eq!(included_stop_token(true, Some(2), &client), None);
    assert_eq!(included_stop_token(true, None, &client), None);
}

/// The included stop token is decoded in context: a toy SentencePiece-like
/// decoder drops the leading space of a sequence-initial word piece, so the
/// lone decode ("END") differs from the in-context text (" END").
#[test]
fn stop_token_text_is_decoded_in_context() {
    let pieces = ["▁the", "▁END", "s"];
    let decode = |ids: &[u32]| {
        let joined: String = ids.iter().map(|&i| pieces[i as usize]).collect();
        joined.replace('▁', " ").trim_start().to_string()
    };
    assert_eq!(decode(&[1]), "END");
    assert_eq!(token_text_after(decode, &[0], 1), " END");
    assert_eq!(token_text_after(decode, &[0, 1], 2), "s");
    // No context: the lone decode.
    assert_eq!(token_text_after(decode, &[], 1), "END");
}
