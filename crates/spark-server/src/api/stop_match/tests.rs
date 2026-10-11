// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `stop_match`. Split out for the 500-LoC cap.

use super::*;

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
    let mut h = StopHoldback::new(stops(&["Obs:"]));
    let mut out = h.push("hello Ob");
    assert!(!h.triggered());
    out += &h.push("s: tail");
    assert!(h.triggered());
    out += &h.push("more");
    out += &h.finish();
    assert_eq!(out, "hello ");

    let mut h = StopHoldback::new(stops(&["Obs:"]));
    let mut out = h.push("hello O");
    out += &h.finish();
    assert_eq!(out, "hello O", "unmatched hold-back must be released");

    let mut h = StopHoldback::new(Vec::new());
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
    let mut w = StopStringWatch::new(&stops(&["cde"]), toy_decode()).unwrap();
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
    let mut w = StopStringWatch::new(&stops(&["ab"]), toy_decode()).unwrap();
    assert!(w.observe(&ids("ab"), false));
    // Same tokens again (post-sampling recheck): still a hit.
    assert!(w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("ab"), true));
}

#[test]
fn watch_detects_stop_in_a_multi_token_step() {
    // Speculative decode can append several tokens between checkpoints.
    let mut w = StopStringWatch::new(&stops(&["cd"]), toy_decode()).unwrap();
    assert!(!w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("abcde"), false));
}

#[test]
fn watch_ignores_reasoning_tokens() {
    let mut w = StopStringWatch::new(&stops(&["cd"]), toy_decode()).unwrap();
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
    let mut w = StopStringWatch::new(&stops(&["ab"]), toy_decode()).unwrap();
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
    let mut w = StopStringWatch::new(&stops(&["xy"]), toy_decode()).unwrap();
    assert!(!w.observe(&ids("abcdx"), false));
    // Rollback dropped "cdx"; regrowth must rescan, not index past len.
    assert!(!w.observe(&ids("ab"), false));
    assert!(w.observe(&ids("abxy"), false));
}

#[test]
fn watch_needs_a_non_empty_stop() {
    assert!(StopStringWatch::new(&[], toy_decode()).is_none());
    assert!(StopStringWatch::new(&stops(&[""]), toy_decode()).is_none());
}
