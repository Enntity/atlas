// SPDX-License-Identifier: AGPL-3.0-only

//! Query-row split of the prefill semantic selection across the TP pair
//! (`ATLAS_GLM_INDEX_SPLIT=1`).
//!
//! Both ranks hold the same pooled index keys and project the same queries.
//! Each row's logits and top-k are independent of the other rows in its
//! launch: a row group only decides which key tiles are wholly future, and
//! those score -inf for each of its rows either way. So each rank selects a
//! subset of an owner's rows and swaps the finished token-id rows with its
//! peer, and the result is bit-identical to the replicated selection.
//!
//! A row's work grows with its causal extent, so the rows are cut into
//! zigzag quarters: rank 0 selects Q0 and Q3, rank 1 selects Q1 and Q2
//! (equal work), then the equal-sized pairs Q0<->Q1 and Q3<->Q2 are
//! exchanged on the copy-engine pair. The projections and the index-cache
//! update stay replicated. Every eligibility input is mirrored on both ranks,
//! since one rank splitting alone would deadlock the pair.
//!
//! `ATLAS_GLM_INDEX_SPLIT_CHECK=1` also selects every row into scratch and
//! fails the request on any difference.

use std::ops::Range;

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::glm_sp;

/// Owners below this many rows stay replicated (verify, short appends): the
/// two exchanges' fixed cost outweighs the halved selection.
const MIN_ROWS: usize = 256;

/// `ATLAS_GLM_INDEX_SPLIT_MIN_CTX` default: the history per row at which
/// the halved selection starts to outweigh the row's exchange.
const DEFAULT_MIN_CTX: usize = 4096;

fn flag(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// Whether `ATLAS_GLM_INDEX_SPLIT=1`.
fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_GLM_INDEX_SPLIT"))
}

/// Whether `ATLAS_GLM_INDEX_SPLIT_CHECK=1`.
fn check_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_GLM_INDEX_SPLIT_CHECK"))
}

fn parse_min_ctx(value: Option<&str>) -> Result<usize, String> {
    value.map_or(Ok(DEFAULT_MIN_CTX), |v| {
        v.parse()
            .map_err(|_| format!("ATLAS_GLM_INDEX_SPLIT_MIN_CTX must be a token count, got {v:?}"))
    })
}

/// History (`seq_len_start`) an owner needs to split,
/// `ATLAS_GLM_INDEX_SPLIT_MIN_CTX`.
fn min_ctx() -> Result<usize> {
    static N: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
    N.get_or_init(|| {
        parse_min_ctx(
            std::env::var("ATLAS_GLM_INDEX_SPLIT_MIN_CTX")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Whether an owner of `rows` rows continuing at `seq_len_start` splits:
/// equal quarters and enough history per row.
fn admits(rows: usize, seq_len_start: usize, min_ctx: usize) -> bool {
    rows >= MIN_ROWS && rows.is_multiple_of(4) && seq_len_start >= min_ctx
}

/// One exchanged pair of quarters: this rank's selected rows at `own` go to
/// the peer, whose rows land at `peer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Swap {
    own: usize,
    peer: usize,
    rows: usize,
}

/// This rank's share of one owner's selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IndexSplit {
    swaps: [Swap; 2],
    check: bool,
}

impl IndexSplit {
    /// This rank's split of an owner of `rows` rows continuing at
    /// `seq_len_start` with `row_bytes`-wide selections, or `None` to select
    /// every row here. Both ranks reach the same answer.
    pub(super) fn plan(
        rows: usize,
        seq_len_start: usize,
        row_bytes: usize,
        ctx: &ForwardContext,
    ) -> Result<Option<Self>> {
        let Some(comm) = ctx.comm.filter(|_| requested()) else {
            return Ok(None);
        };
        let min_ctx = min_ctx()?;
        let eligible = !ctx.graph_capture
            && ctx.config.tp_world_size == 2
            && comm.world_size() == 2
            && admits(rows, seq_len_start, min_ctx)
            && comm.supports_exchange_async(rows / 4 * row_bytes);
        if !eligible {
            return Ok(None);
        }
        let check = check_requested();
        ensure!(
            !check || ctx.buffers.sizes().expert_down_out >= 2 * rows * row_bytes,
            "ATLAS_GLM_INDEX_SPLIT_CHECK: no scratch for {rows} replicated rows"
        );
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "GLM index split: rank {} selects zigzag quarters of owners from {min_ctx} history tokens (check={check})",
                comm.rank()
            )
        });
        Ok(Some(Self {
            swaps: Self::zigzag(rows, comm.rank()),
            check,
        }))
    }

    /// Zigzag quarters of an owner's `rows` (a multiple of 4) for `rank`:
    /// pair k is (rank 0's quarter, rank 1's quarter).
    fn zigzag(rows: usize, rank: usize) -> [Swap; 2] {
        let q = rows / 4;
        [(0, q), (3 * q, 2 * q)].map(|(r0, r1)| {
            let (own, peer) = if rank == 0 { (r0, r1) } else { (r1, r0) };
            Swap { own, peer, rows: q }
        })
    }

    /// The rows this rank selects.
    fn own(&self) -> [Range<usize>; 2] {
        self.swaps.map(|s| s.own..s.own + s.rows)
    }

    /// Swap this rank's finished rows of `selected` for the peer's. Under the
    /// check, then compare all `rows` with the replicated rows in `scratch`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn exchange(
        &self,
        selected: DevicePtr,
        scratch: DevicePtr,
        rows: usize,
        row_bytes: usize,
        layer: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        for s in self.swaps {
            glm_sp::exchange_rows(
                selected.offset(s.own * row_bytes),
                selected.offset(s.peer * row_bytes),
                s.rows,
                row_bytes,
                false,
                ctx,
                stream,
            )?;
        }
        if !self.check {
            return Ok(());
        }
        let bytes = rows * row_bytes;
        let (mut split, mut replicated) = (vec![0u8; bytes], vec![0u8; bytes]);
        ctx.gpu.copy_d2h_on_stream(selected, &mut split, stream)?;
        ctx.gpu
            .copy_d2h_on_stream(scratch, &mut replicated, stream)?;
        if let Some(row) = first_mismatch(&split, &replicated, row_bytes) {
            bail!(
                "ATLAS_GLM_INDEX_SPLIT_CHECK: layer {layer} row {row} of {rows} differs from the replicated selection"
            );
        }
        tracing::info!("ATLAS_GLM_INDEX_SPLIT_CHECK ok layer={layer} rows={rows}");
        Ok(())
    }
}

/// The selection passes of an owner of `rows` rows, as row ranges and the
/// output each lands in: every row into `selected` when replicated; split,
/// this rank's quarters, then (check) every row again into `scratch`.
pub(super) fn passes(
    split: Option<IndexSplit>,
    rows: usize,
    selected: DevicePtr,
    scratch: DevicePtr,
) -> Vec<(Range<usize>, DevicePtr)> {
    let Some(split) = split else {
        return vec![(0..rows, selected)];
    };
    let mut passes: Vec<_> = split.own().into_iter().map(|r| (r, selected)).collect();
    if split.check {
        passes.push((0..rows, scratch));
    }
    passes
}

/// `(row_start, rows, output)` tiles of at most `tile_rows` rows over
/// `passes`.
pub(super) fn tiles(
    passes: &[(Range<usize>, DevicePtr)],
    tile_rows: usize,
) -> impl Iterator<Item = (usize, usize, DevicePtr)> + '_ {
    passes.iter().flat_map(move |(range, out)| {
        range
            .clone()
            .step_by(tile_rows)
            .map(move |start| (start, tile_rows.min(range.end - start), *out))
    })
}

fn first_mismatch(a: &[u8], b: &[u8], row_bytes: usize) -> Option<usize> {
    a.chunks(row_bytes)
        .zip(b.chunks(row_bytes))
        .position(|(x, y)| x != y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owner row counts the production pieces produce: the 8196-row first
    /// chunk's 4096 and 2052 pieces, full 4096 pieces, warm appends.
    const ROWS: [usize; 6] = [256, 512, 1024, 2052, 3000, 4096];

    fn split(rows: usize, rank: usize, check: bool) -> IndexSplit {
        IndexSplit {
            swaps: IndexSplit::zigzag(rows, rank),
            check,
        }
    }

    #[test]
    fn quarters_tile_the_owner_once_and_pair_symmetrically() {
        for rows in ROWS.into_iter().filter(|r| r % 4 == 0) {
            let (r0, r1) = (IndexSplit::zigzag(rows, 0), IndexSplit::zigzag(rows, 1));
            let mut seen = vec![0u8; rows];
            for s in r0.iter().chain(&r1) {
                assert_eq!(s.rows, rows / 4);
                seen[s.own..s.own + s.rows].iter_mut().for_each(|c| *c += 1);
            }
            assert!(seen.iter().all(|&c| c == 1), "rows {rows}");
            // Pair k: what one rank sends lands where the other expects it,
            // and both ranks exchange the same byte count in the same order.
            for k in 0..2 {
                assert_eq!((r0[k].own, r0[k].peer), (r1[k].peer, r1[k].own));
                assert_eq!(r0[k].rows, r1[k].rows);
            }
        }
        let q = 1024;
        assert_eq!(
            IndexSplit::zigzag(4 * q, 0),
            [
                Swap {
                    own: 0,
                    peer: q,
                    rows: q
                },
                Swap {
                    own: 3 * q,
                    peer: 2 * q,
                    rows: q
                },
            ]
        );
    }

    #[test]
    fn zigzag_balances_causal_work_exactly() {
        // A row's logits and top-k scale with its causal extent.
        for rows in ROWS.into_iter().filter(|r| r % 4 == 0) {
            for start in [0, 2048, 6144, 8196, 100_000, 524_288] {
                let work = |rank| -> usize {
                    split(rows, rank, false)
                        .own()
                        .into_iter()
                        .flatten()
                        .map(|row| start + row + 1)
                        .sum()
                };
                assert_eq!(work(0), work(1), "rows {rows} start {start}");
            }
        }
    }

    #[test]
    fn admits_long_aligned_owners_only() {
        let min = 4096;
        // Later 8K-chunk pieces and the first chunk's 2052-row tail piece.
        assert!(admits(4096, 8196, min));
        assert!(admits(2052, 6144, min));
        // A warm 512-row append at long context.
        assert!(admits(512, 200_000, min));
        // The first chunk's 4096-row piece has too little history.
        assert!(!admits(4096, 2048, min));
        // Unequal quarters, verify-sized owners and 4-row tails.
        assert!(!admits(2050, 8196, min));
        assert!(!admits(3001, 8196, min));
        assert!(!admits(64, 100_000, min));
        assert!(!admits(4, 16_388, min));
        assert!(admits(4096, 0, 0));
    }

    #[test]
    fn min_ctx_defaults_to_4096_and_rejects_junk() {
        assert_eq!(parse_min_ctx(None), Ok(4096));
        assert_eq!(parse_min_ctx(Some("16384")), Ok(16384));
        assert_eq!(parse_min_ctx(Some("0")), Ok(0));
        assert!(parse_min_ctx(Some("4k")).is_err());
    }

    #[test]
    fn passes_cover_own_rows_and_the_check_recompute() {
        let (sel, scratch) = (DevicePtr(0x1000), DevicePtr(0x9000));
        assert_eq!(passes(None, 4096, sel, scratch), [(0..4096, sel)]);
        assert_eq!(
            passes(Some(split(4096, 0, false)), 4096, sel, scratch),
            [(0..1024, sel), (3072..4096, sel)]
        );
        assert_eq!(
            passes(Some(split(4096, 1, true)), 4096, sel, scratch),
            [(1024..2048, sel), (2048..3072, sel), (0..4096, scratch)]
        );
    }

    #[test]
    fn tiles_bound_each_pass_like_the_replicated_loop() {
        let (a, b) = (DevicePtr(0x10), DevicePtr(0x20));
        let t: Vec<_> = tiles(&[(0..700, a)], 300).collect();
        assert_eq!(t, [(0, 300, a), (300, 300, a), (600, 100, a)]);
        let t: Vec<_> = tiles(&[(0..513, a), (1539..2052, a), (0..2052, b)], 2052).collect();
        assert_eq!(t, [(0, 513, a), (1539, 513, a), (0, 2052, b)]);
    }

    #[test]
    fn mismatch_reports_the_first_differing_row() {
        let a = [1u8, 2, 3, 4, 5, 6];
        assert_eq!(first_mismatch(&a, &a, 2), None);
        assert_eq!(first_mismatch(&a, &[1, 2, 3, 4, 5, 7], 2), Some(2));
        assert_eq!(first_mismatch(&a, &[1, 2, 0, 4, 0, 6], 2), Some(1));
    }
}
