// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// qwen4_exp at the checkpoint bound: 2048x2048 = 16384 patches, merge 2.
const P_MAX: usize = 16_384;
const FULL: usize = 4096; // one 2048x2048 image, merged rows
/// `--max-prefill-tokens 16384 --max-num-seqs 4` → max_batch_tokens.
const CHUNK: usize = 16_388;

fn limits(request_rows: usize) -> VisionRowLimits {
    VisionRowLimits {
        window_rows: window_rows(P_MAX, 2, CHUNK),
        chunk_tokens: CHUNK,
        request_rows,
    }
}

#[test]
fn the_window_covers_a_chunk_plus_two_cut_images() {
    assert_eq!(window_rows(P_MAX, 2, CHUNK), CHUNK + 2 * FULL);
    // Never below the historical one-batch buffer, which every request that
    // fit before relied on.
    assert_eq!(
        window_rows(P_MAX, 2, 0),
        P_MAX,
        "unknown chunk keeps the size"
    );
    assert_eq!(window_rows(P_MAX, 2, 1024), P_MAX);
    assert_eq!(window_rows(6400, 2, 0), 6400);
}

#[test]
fn worst_chunk_rows_is_the_two_largest_plus_what_fits_between() {
    assert_eq!(worst_chunk_rows(&[], CHUNK), 0);
    assert_eq!(worst_chunk_rows(&[FULL], CHUNK), FULL, "one item: its rows");
    // Few items: the whole request can sit in one chunk.
    assert_eq!(worst_chunk_rows(&[FULL; 3], CHUNK), 3 * FULL);
    // The reporter's agent history, 16680 rows: e.g. four full-size
    // screenshots and a 256-row thumbnail.
    let report = [FULL, FULL, FULL, FULL, 296];
    assert_eq!(report.iter().sum::<usize>(), 16_680);
    assert_eq!(worst_chunk_rows(&report, CHUNK), 16_680);
    // Many: two cut at the ends plus a chunk's worth between.
    assert_eq!(worst_chunk_rows(&[FULL; 20], CHUNK), 2 * FULL + CHUNK - 2);
    // A tiny chunk still needs the two largest items whole.
    assert_eq!(worst_chunk_rows(&[10, FULL, 7, 3000], 2), FULL + 3000);
}

#[test]
fn every_still_image_request_within_the_request_limit_is_servable() {
    // The window is sized so the encoder never decides: any mix of still
    // images under the request limit passes the window check.
    let l = limits(usize::MAX);
    for n in [1, 2, 4, 5, 7, 16, 64, 1024] {
        assert_eq!(l.check(&vec![FULL; n]), Ok(()), "{n} full-size images");
        assert_eq!(l.check(&vec![1024; n]), Ok(()), "{n} 1024x1024 images");
    }
}

#[test]
fn a_request_over_the_old_one_batch_cap_is_accepted() {
    // Issue #180: 16680 rows used to fail at prefill start against 16384.
    let l = limits(32_768);
    assert_eq!(l.check(&[FULL, FULL, FULL, FULL, 296]), Ok(()));
    assert_eq!(l.check(&[FULL; 8]), Ok(()), "exactly the request limit");
}

#[test]
fn the_request_limit_rejects_with_counts_and_the_limit() {
    let err = limits(16_384)
        .check(&[FULL, FULL, FULL, FULL, 296])
        .unwrap_err();
    assert!(err.contains("5 images"), "{err}");
    assert!(err.contains("16680"), "{err}");
    assert!(err.contains("16384"), "{err}");
}

#[test]
fn items_past_the_protocol_limit_are_rejected() {
    let l = limits(usize::MAX);
    assert_eq!(l.check(&vec![1; MAX_VISION_ITEMS]), Ok(()));
    let err = l.check(&vec![1; MAX_VISION_ITEMS + 1]).unwrap_err();
    assert!(err.contains("1025 images"), "{err}");
}

#[test]
fn long_videos_that_cannot_share_a_chunk_are_rejected() {
    // Videos are items of many temporal groups; two that large cut by one
    // chunk would overflow the window, so the request is refused up front.
    let l = limits(usize::MAX);
    let video = 12_000;
    assert_eq!(l.check(&[video]), Ok(()), "one video fits on its own");
    assert_eq!(
        l.check(&[video, 100]),
        Ok(()),
        "the whole request fits one window"
    );
    let err = l.check(&[video, video, FULL]).unwrap_err();
    assert!(err.contains("window"), "{err}");
    assert!(err.contains(&l.window_rows.to_string()), "{err}");
    // A single item larger than the window can never be encoded.
    assert!(l.check(&[l.window_rows + 1]).is_err());
}

#[test]
fn a_request_that_fit_the_old_cap_still_fits() {
    // Any request of <= 16384 rows (the historical per-request bound) passes,
    // whatever its items: its worst chunk is at most its total.
    let l = limits(16_384);
    assert_eq!(l.check(&[16_384]), Ok(()));
    assert_eq!(l.check(&[8000, 8000, 384]), Ok(()));
}

#[test]
fn messages_carry_no_request_content() {
    // Counts and limits only: digits, ASCII words, punctuation.
    let err = limits(10).check(&[FULL]).unwrap_err();
    assert!(err.is_ascii());
    assert!(err.starts_with("vision input too large"));
}

#[test]
fn the_request_limit_is_the_knob_capped_by_the_context() {
    let window = window_rows(P_MAX, 2, CHUNK);
    let l = VisionRowLimits::resolve(window, CHUNK, 32_768, 262_144);
    assert_eq!(l, limits(32_768));
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 0, 262_144).request_rows,
        262_144,
        "0 = the context alone"
    );
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 1 << 20, 8192).request_rows,
        8192,
        "never past the context"
    );
}
