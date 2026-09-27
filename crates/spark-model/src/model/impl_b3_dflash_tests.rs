// SPDX-License-Identifier: AGPL-3.0-only

use super::impl_b3_dflash::{dflash_prefill_capture_span, dflash_prefill_window};

/// Replays a chunked prefill of a `prompt_len` prompt the way prefill_b runs
/// it — chunk `[s, s + len)` computes rows from `max(s, lo)`, a pass of at
/// least `sp_min` rows keeps only its upper half resident — and returns the
/// prompt position each accumulator slot ends up holding.
fn replay(chunks: &[(usize, usize)], prompt_len: usize, lo: usize, sp_min: usize) -> Vec<usize> {
    let max_ctx = 2048;
    let window = dflash_prefill_window(prompt_len, lo, max_ctx);
    let mut acc = vec![usize::MAX; max_ctx];
    for &(s, len) in chunks {
        let row0 = s.max(lo);
        let (row0, rows) = match s + len {
            end if row0 < end => (row0, end - row0),
            // Exact hit: the last chunk re-runs only its final token.
            end => (end - 1, 1),
        };
        let local0 = if rows >= sp_min { rows / 2 } else { 0 };
        if let Some((row, n, slot)) = dflash_prefill_capture_span(&window, row0, rows, local0) {
            for i in 0..n {
                acc[slot + i] = row0 + local0 + row + i;
            }
        }
    }
    acc.truncate(window.len());
    acc
}

fn chunked(prompt_len: usize, chunk: usize) -> Vec<(usize, usize)> {
    (0..prompt_len)
        .step_by(chunk)
        .map(|s| (s, chunk.min(prompt_len - s)))
        .collect()
}

/// prefill_b's tail-checkpoint split: the last chunk is cut one block below
/// the last block boundary under the prompt end.
fn tail_split(prompt_len: usize, chunk: usize, bs: usize) -> Vec<(usize, usize)> {
    let mut chunks = chunked(prompt_len, chunk);
    let (s, len) = chunks.pop().unwrap();
    let cut = ((prompt_len - 1) / bs * bs).saturating_sub(bs);
    if cut > s {
        chunks.push((s, cut - s));
        chunks.push((cut, prompt_len - cut));
    } else {
        chunks.push((s, len));
    }
    chunks
}

fn assert_tail(acc: &[usize], first: usize, prompt_len: usize) {
    let want: Vec<usize> = (first..prompt_len).collect();
    assert_eq!(acc, want.as_slice());
}

#[test]
fn cold_chunked_prefill_keeps_the_prompt_tail() {
    for p in [
        1000, 2048, 3000, 8192, 9000, 12000, 16383, 16384, 20000, 32768,
    ] {
        for c in [1024, 4096, 8192] {
            let acc = replay(&chunked(p, c), p, 0, usize::MAX);
            assert_tail(&acc, p.saturating_sub(2048), p);
        }
    }
}

#[test]
fn tail_checkpoint_split_keeps_the_last_blocks() {
    for p in [12000, 16384, 20000] {
        for bs in [16, 64] {
            let acc = replay(&tail_split(p, 8192, bs), p, 0, usize::MAX);
            assert_tail(&acc, p - 2048, p);
        }
    }
}

#[test]
fn warm_hit_window_starts_at_the_first_computed_position() {
    // Suffix shorter than the window: only computed positions, from lo.
    let acc = replay(&chunked(12000, 8192), 12000, 11000, usize::MAX);
    assert_tail(&acc, 11000, 12000);
    // Suffix longer than the window: the usual tail.
    let acc = replay(&chunked(20000, 8192), 20000, 3000, usize::MAX);
    assert_tail(&acc, 20000 - 2048, 20000);
}

#[test]
fn exact_hit_captures_nothing() {
    // lo == prompt_len: the re-run last token is not a context row.
    assert!(replay(&chunked(4000, 8192), 4000, 4000, usize::MAX).is_empty());
    assert!(dflash_prefill_window(4000, 4000, 2048).is_empty());
}

#[test]
fn sequence_parallel_passes_capture_from_the_resident_upper_half() {
    for p in [8192, 16384, 20000] {
        let acc = replay(&chunked(p, 8192), p, 0, 4096);
        assert_tail(&acc, p - 2048, p);
    }
}
