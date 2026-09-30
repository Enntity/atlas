// SPDX-License-Identifier: AGPL-3.0-only

//! The warm-turn switches: the trace line and its logits hash, and what each
//! environment value parses to.

use super::*;

/// A `WarmTurn` with every switch off but the trace `mode`.
fn tracing(mode: TraceMode) -> WarmTurn {
    let mut warm = WarmTurn::from_env().unwrap();
    warm.trace = mode;
    warm
}

#[test]
fn the_trace_is_off_a_hash_line_or_a_synced_line_and_refuses_anything_else() {
    use TraceMode::*;
    assert_eq!(TraceMode::parse(None).unwrap(), Off);
    assert_eq!(TraceMode::parse(Some("0")).unwrap(), Off);
    assert_eq!(TraceMode::parse(Some("1")).unwrap(), Spans);
    assert_eq!(TraceMode::parse(Some("hash")).unwrap(), Hash);
    for bad in ["true", "2", "", "HASH"] {
        let e = TraceMode::parse(Some(bad)).unwrap_err();
        assert!(
            format!("{e:#}").contains("ATLAS_GLM_WARM_TRACE must"),
            "{e:#}"
        );
    }
}

/// The fingerprint tells apart rows that differ in any byte, in length, or
/// only past the last whole word.
#[test]
fn the_logits_hash_covers_every_byte_and_the_length() {
    let row: Vec<u8> = (0..=255u8).cycle().take(1003).collect();
    let h = logits_hash(&row);
    assert_eq!(h, logits_hash(&row.clone()));
    for at in [0, 7, 8, 500, 999, 1000, 1002] {
        let mut other = row.clone();
        other[at] ^= 1;
        assert_ne!(logits_hash(&other), h, "byte {at}");
    }
    assert_ne!(logits_hash(&row[..1002]), h);
    assert_ne!(logits_hash(&[]), logits_hash(&[0]));
    assert_ne!(logits_hash(&[0; 8]), logits_hash(&[0; 9]));
}

fn shape() -> RequestShape {
    RequestShape {
        rank: 1,
        prompt: 40,
        matched: 32,
        restored: 16,
        logits: 0xabc,
    }
}

#[test]
fn the_trace_line_sums_a_requests_chunks_and_forgets_them() {
    let ms = Duration::from_millis;
    let warm = tracing(TraceMode::Spans);
    let began = Instant::now();
    *warm.transfer.lock() += ms(2);
    // Two cached chunks: no rows, a lookup and a block vote.
    let cached = [ms(0), ms(0), ms(3), ms(1), ms(0), ms(0), ms(0)];
    assert_eq!(warm.note_chunk(7, began, true, 0, cached, None), None);
    assert_eq!(warm.note_chunk(7, began, false, 0, cached, None), None);
    // Another slot's chunk in between stays out of this request's line.
    assert_eq!(warm.note_chunk(2, began, true, 99, [ms(50); 7], None), None);
    let pass = [ms(17), ms(2), ms(0), ms(1), ms(1), ms(100), ms(5)];
    assert_eq!(warm.note_chunk(7, began, false, 8, pass, None), None);
    let line = warm
        .note_chunk(7, began, false, 16, pass, Some(shape()))
        .unwrap();
    assert!(
        line.starts_with(
            "warm-turn rank=1 slot=7 tokens=40 matched=32 restored=16 chunks=4 \
             cached_chunks=2 rows=24 ms: transfer=2.0 zero=34.0 embed=4.0 lookup=6.0 \
             blocks=4.0 meta=2.0 forward=200.0 finish=10.0 wall="
        ),
        "{line}"
    );
    assert!(line.ends_with(" logits=0000000000000abc"), "{line}");
    // The wall clock starts with the first chunk's transfer.
    let at = line.find("wall=").unwrap() + 5;
    let wall: f64 = line[at..].split(' ').next().unwrap().parse().unwrap();
    assert!(wall >= 2.0, "{line}");
    // The request is gone; the other slot's chunk is still pending.
    let next = warm
        .note_chunk(7, began, false, 1, pass, Some(shape()))
        .unwrap();
    assert!(next.contains("chunks=1 cached_chunks=0 rows=1 "), "{next}");
    let other = warm
        .note_chunk(2, began, false, 1, pass, Some(shape()))
        .unwrap();
    assert!(
        other.contains("chunks=2 cached_chunks=0 rows=100 "),
        "{other}"
    );
}

/// A request that failed or was preempted before its last chunk leaves a
/// trace behind; the slot's next request starts over at its first chunk.
#[test]
fn a_first_chunk_drops_what_an_unfinished_request_left() {
    let ms = Duration::from_millis;
    let warm = tracing(TraceMode::Hash);
    let pass = [ms(1); 7];
    let long_ago = Instant::now() - ms(60_000);
    assert_eq!(warm.note_chunk(3, long_ago, true, 8, pass, None), None);
    let line = warm
        .note_chunk(3, Instant::now(), true, 4, pass, Some(shape()))
        .unwrap();
    assert!(line.contains("chunks=1 cached_chunks=0 rows=4 "), "{line}");
    let at = line.find("wall=").unwrap() + 5;
    let wall: f64 = line[at..].split(' ').next().unwrap().parse().unwrap();
    assert!(wall < 30_000.0, "{line}");
}

/// Bulk token broadcasts are timed only while the trace is on, and charged
/// to the next chunk noted.
#[test]
fn a_transfer_is_timed_only_with_the_trace_on() {
    let off = tracing(TraceMode::Off);
    assert!(off.transfer_span().is_none());
    for mode in [TraceMode::Hash, TraceMode::Spans] {
        let warm = tracing(mode);
        let span = warm.transfer_span().unwrap();
        std::thread::sleep(Duration::from_millis(2));
        drop(span);
        assert!(*warm.transfer.lock() >= Duration::from_millis(2));
        let line = warm
            .note_chunk(
                0,
                Instant::now(),
                true,
                1,
                [Duration::ZERO; 7],
                Some(shape()),
            )
            .unwrap();
        assert!(!line.contains("transfer=0.0 "), "{line}");
        assert_eq!(*warm.transfer.lock(), Duration::ZERO);
    }
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
