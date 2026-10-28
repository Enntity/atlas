// SPDX-License-Identifier: AGPL-3.0-only

//! Dense ctx-window capture planning for DFlash prefill.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slide {
    pub drop_n: usize,
    pub move_rows: usize,
    pub carried_drop: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapturePlan {
    pub slide: Option<Slide>,
    pub skip: usize,
    pub first_row: usize,
    pub base: usize,
    pub origin: usize,
}

pub(crate) fn plan_capture(
    base: usize,
    origin: usize,
    chunk_start: usize,
    proc_count: usize,
    acc_rows: usize,
    max_ctx_len: usize,
) -> CapturePlan {
    let skip0 = origin.saturating_sub(chunk_start).min(proc_count);
    let cursor0 = base + chunk_start.saturating_sub(origin);
    let needed = cursor0 + (proc_count - skip0);
    debug_assert!(cursor0 <= acc_rows);
    if needed > acc_rows {
        let keep = max_ctx_len / 2;
        let drop_n = needed - keep;
        let move_rows = cursor0.saturating_sub(drop_n);
        debug_assert!(move_rows <= drop_n);
        debug_assert!(move_rows == 0 || drop_n + move_rows <= acc_rows);
        let carried_drop = drop_n.min(base);
        let base = base - carried_drop;
        let origin = origin + (drop_n - carried_drop);
        let skip = origin.saturating_sub(chunk_start).min(proc_count);
        let first_row = base + chunk_start.saturating_sub(origin);
        CapturePlan {
            slide: Some(Slide {
                drop_n,
                move_rows,
                carried_drop,
            }),
            skip,
            first_row,
            base,
            origin,
        }
    } else {
        CapturePlan {
            slide: None,
            skip: skip0,
            first_row: cursor0,
            base,
            origin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gb10_oversized_chunk_stays_in_bounds() {
        let p = plan_capture(0, 11984, 11984, 8193, 4096, 4096);
        assert_eq!(
            p.slide,
            Some(Slide {
                drop_n: 6145,
                move_rows: 0,
                carried_drop: 0,
            })
        );
        assert_eq!((p.skip, p.first_row), (6145, 0));
        assert_eq!((p.base, p.origin), (0, 18129));
        assert!(p.first_row + (8193 - p.skip) <= 4096);
        let p2 = plan_capture(p.base, p.origin, 20177, 353, 4096, 4096);
        assert_eq!(p2.slide, None);
        assert_eq!(p2.skip, 0);
        assert_eq!(p2.first_row, 2048);
        assert_eq!(p2.base + (20177 + 353 - p2.origin), 2401);
    }

    #[test]
    fn replan_is_idempotent() {
        let p = plan_capture(0, 11984, 11984, 8193, 4096, 4096);
        let p2 = plan_capture(p.base, p.origin, 11984, 8193, 4096, 4096);
        assert_eq!(p2.slide, None);
        assert_eq!((p2.skip, p2.first_row), (p.skip, p.first_row));
        let q = plan_capture(3500, 12000, 12000, 1024, 4096, 4096);
        assert!(q.slide.is_some());
        let q2 = plan_capture(q.base, q.origin, 12000, 1024, 4096, 4096);
        assert_eq!(q2.slide, None);
        assert_eq!((q2.skip, q2.first_row), (q.skip, q.first_row));
    }

    #[test]
    fn strix_regime_matches_the_old_formula() {
        for win in [8192usize, 12288] {
            let keep = win / 2;
            for mut base in [0usize, 3000] {
                let mut origin = 0usize;
                for chunk_start in (0..25000).step_by(1024) {
                    if origin == 0 {
                        origin = chunk_start;
                    }
                    let proc = 1024usize.min(25000 - chunk_start);
                    let p = plan_capture(base, origin, chunk_start, proc, win, win);
                    let cursor = base + chunk_start.saturating_sub(origin);
                    let needed = cursor + proc;
                    if needed <= win {
                        assert_eq!(p.slide, None);
                        assert_eq!(p.first_row, cursor);
                        assert_eq!(p.skip, 0);
                    } else {
                        let drop_n = needed - keep;
                        assert_eq!(
                            p.slide.unwrap(),
                            Slide {
                                drop_n,
                                move_rows: cursor - drop_n,
                                carried_drop: drop_n.min(base),
                            }
                        );
                        assert_eq!(p.first_row, cursor - drop_n);
                        assert_eq!(p.skip, 0);
                        base = p.base;
                        origin = p.origin;
                    }
                }
            }
        }
    }

    #[test]
    fn carried_base_with_oversized_chunk() {
        let p = plan_capture(2147, 12000, 12000, 8193, 4096, 4096);
        assert_eq!(
            p.slide,
            Some(Slide {
                drop_n: 8292,
                move_rows: 0,
                carried_drop: 2147,
            })
        );
        assert_eq!((p.base, p.origin, p.skip, p.first_row), (0, 18145, 6145, 0));
    }

    #[test]
    fn grid_simulation() {
        for win in [2048usize, 4096, 8192, 12288] {
            for chunk in [512usize, 1024, 2048, 4096, 8193] {
                for base0 in [0usize, 1000, win / 2] {
                    for replay in [100usize, 3000, 9000, 25000] {
                        simulate(win, chunk, base0, replay);
                    }
                }
            }
        }
    }

    fn simulate(win: usize, chunk: usize, base0: usize, replay: usize) {
        let mut rows: Vec<Option<usize>> = vec![None; win];
        let mut origin = 10000usize;
        for i in 0..base0.min(win) {
            rows[i] = Some(origin - base0 + i);
        }
        let mut base = base0.min(win);
        let mut chunk_start = origin;
        let end = chunk_start + replay;
        while chunk_start < end {
            let proc = chunk.min(end - chunk_start);
            let p = plan_capture(base, origin, chunk_start, proc, win, win);
            if let Some(s) = p.slide {
                assert!(s.move_rows == 0 || s.drop_n + s.move_rows <= win);
                assert!(s.move_rows <= s.drop_n);
                if s.move_rows > 0 {
                    rows.copy_within(s.drop_n..s.drop_n + s.move_rows, 0);
                }
                for row in &mut rows[s.move_rows..] {
                    *row = None;
                }
                base = p.base;
                origin = p.origin;
            }
            for t in p.skip..proc {
                let row = p.first_row + (t - p.skip);
                assert!(row < win);
                rows[row] = Some(chunk_start + t);
            }
            chunk_start += proc;
        }
        let n = rows.iter().take_while(|row| row.is_some()).count();
        assert!(rows[n..].iter().all(|row| row.is_none()));
        let mut prev = None;
        for (i, row) in rows.iter().take(n).enumerate() {
            if let Some(pos) = row {
                if let Some(previous) = prev {
                    assert_eq!(*pos, previous + 1, "row {i} position gap");
                }
                prev = Some(*pos);
            }
        }
        if n > 0 {
            assert_eq!(rows[n - 1].unwrap(), end - 1);
            assert_eq!(rows[0].unwrap(), origin - base);
        }
        assert_eq!(n, base + (end - origin));
    }
}
