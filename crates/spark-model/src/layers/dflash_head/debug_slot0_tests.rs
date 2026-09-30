// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_DEBUG_SLOT0`: the parser, the pure `keep` decision replayed
//! over the prefill shapes it has to tell apart, and the end-of-prefill
//! `zero` on the mock backend.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::first_append_tests::{SLOT, WINDOW, dstate, head, slot};
use super::{DebugSlot0, FirstAppend, Slot0};
use crate::speculative::{DraftProposer, ProposerState};

const ALL: [DebugSlot0; 3] = [DebugSlot0::AsIs, DebugSlot0::Zero, DebugSlot0::Keep];
/// Capture layers: the capture runs once per layer per pass.
const LAYERS: usize = 5;
/// The production window, for the replays (they touch no GPU).
const CTX: usize = 2048;

#[test]
fn parser_defaults_to_asis_and_accepts_the_three_variants() {
    assert_eq!(DebugSlot0::parse(None).unwrap(), DebugSlot0::AsIs);
    assert_eq!(DebugSlot0::default(), DebugSlot0::AsIs);
    assert_eq!(Slot0::default().variant, DebugSlot0::AsIs);
    for (raw, variant) in [
        ("asis", DebugSlot0::AsIs),
        ("zero", DebugSlot0::Zero),
        ("keep", DebugSlot0::Keep),
    ] {
        assert_eq!(DebugSlot0::parse(Some(raw)).unwrap(), variant, "{raw}");
    }
}

#[test]
fn parser_rejects_every_other_value() {
    for raw in ["", "0", "1", "Keep", "zero ", "off", "as-is", "zero,keep"] {
        let error = DebugSlot0::parse(Some(raw)).unwrap_err().to_string();
        assert!(
            error.contains("ATLAS_DFLASH_DEBUG_SLOT0"),
            "{raw:?}: {error}"
        );
        assert!(error.contains(&format!("{raw:?}")), "{raw:?}: {error}");
    }
}

/// Only `keep` ever skips a row or records anything.
#[test]
fn asis_and_zero_never_skip() {
    for variant in [DebugSlot0::AsIs, DebugSlot0::Zero] {
        let mut slot0 = Slot0 {
            variant,
            kept: None,
        };
        for fits in [false, true] {
            for pos in [0, 1, 80] {
                for slot in [0, 1] {
                    assert_eq!(slot0.keep_skip(fits, pos, slot), 0, "{variant:?}");
                }
            }
        }
        assert_eq!(slot0.kept, None, "{variant:?}");
    }
}

/// One prefill pass through the capture's index math
/// (`try_dflash_prefill_capture_layer`): `rows` rows whose row 0 is prompt
/// position `row0_pos` and whose first `sp_row0` rows live on the other rank,
/// after `done` tokens of a `prompt`-token prompt. `acc[slot]` is the prompt
/// position the slot holds.
fn pass(
    slot0: &mut Slot0,
    acc: &mut [Option<usize>],
    prompt: usize,
    (done, row0_pos, rows, sp_row0): (usize, usize, usize, usize),
) {
    let window_start = done.max(rows).saturating_sub(CTX);
    let first = window_start.max(sp_row0);
    for _ in 0..LAYERS {
        let skip = slot0.keep_skip(prompt <= CTX, row0_pos + first, first - window_start);
        for row in first + skip..rows {
            acc[row - window_start] = Some(row0_pos + row);
        }
    }
}

/// The accumulator after a prefill made of `passes`, and the `keep` record.
fn replay(
    variant: DebugSlot0,
    prompt: usize,
    passes: &[(usize, usize, usize, usize)],
) -> (Vec<Option<usize>>, Option<usize>) {
    let mut slot0 = Slot0 {
        variant,
        kept: None,
    };
    let mut acc = vec![None; CTX];
    for &shape in passes {
        pass(&mut slot0, &mut acc, prompt, shape);
    }
    (acc, slot0.kept)
}

/// A cold 100-token prompt, tail cut at 80.
const SPLIT: [(usize, usize, usize, usize); 2] = [(0, 0, 80, 0), (80, 80, 20, 0)];

#[test]
fn keep_lets_the_position_0_row_survive_the_tail_pass_and_changes_nothing_else() {
    let (asis, _) = replay(DebugSlot0::AsIs, 100, &SPLIT);
    // Today: the tail's rows sit at slots 0..20 over the first pass's head.
    assert_eq!(asis[0], Some(80));
    assert_eq!(asis[19], Some(99));
    assert_eq!(asis[20], Some(20));
    assert_eq!(asis[79], Some(79));
    assert_eq!(asis[80], None);

    let (keep, kept) = replay(DebugSlot0::Keep, 100, &SPLIT);
    assert_eq!(keep[0], Some(0));
    assert_eq!(keep[1..], asis[1..]);
    // One skipped write per capture layer of the tail pass.
    assert_eq!(kept, Some(LAYERS));

    assert_eq!(replay(DebugSlot0::Zero, 100, &SPLIT), (asis, None));
}

/// `keep` has nothing to do for a single pass, for a request whose first pass
/// does not start at prompt position 0 (a warm turn), for a first pass whose
/// position-0 row lives on the other rank (sequence-parallel), or for a
/// prompt longer than the window.
#[test]
fn keep_is_a_no_op_without_a_position_0_row_in_slot_0() {
    let one_pass = [(0, 0, 100, 0)];
    let warm = [(0, 48, 32, 0), (80, 80, 20, 0)];
    let warm_tail_only = [(80, 80, 20, 0)];
    let sequence_parallel = [(0, 0, 80, 40), (80, 80, 20, 0)];
    let longer = [(0, 0, 2032, 0), (2032, 2032, 28, 0)];
    let much_longer = [(0, 0, 4976, 0), (4976, 4976, 24, 0)];
    for (prompt, passes) in [
        (100, &one_pass[..]),
        (100, &warm[..]),
        (100, &warm_tail_only[..]),
        (100, &sequence_parallel[..]),
        (2060, &longer[..]),
        (5000, &much_longer[..]),
    ] {
        let (asis, _) = replay(DebugSlot0::AsIs, prompt, passes);
        let (keep, kept) = replay(DebugSlot0::Keep, prompt, passes);
        assert_eq!(keep, asis, "{prompt} {passes:?}");
        assert!(kept.is_none_or(|skipped| skipped == 0), "{passes:?}");
    }
}

/// A retried prefill starts over at position 0: the row is written again.
#[test]
fn keep_rewrites_slot_0_when_a_pass_starts_at_position_0_again() {
    let mut slot0 = Slot0 {
        variant: DebugSlot0::Keep,
        kept: Some(LAYERS),
    };
    assert_eq!(slot0.keep_skip(true, 0, 0), 0);
    assert_eq!(slot0.keep_skip(true, 80, 0), 1);
    // A write that does not start at slot 0 is never touched.
    assert_eq!(slot0.keep_skip(true, 81, 1), 0);
}

/// A sequence after prefill: every slot dirty, slot 0 stamped `stamp`.
fn after_prefill(
    variant: DebugSlot0,
    stamp: Option<i32>,
    gpu: &MockGpuBackend,
) -> Box<dyn ProposerState> {
    let mut head = head(FirstAppend::Legacy);
    head.startup.diagnostics.debug_slot0 = variant;
    let mut state = head.alloc_state(gpu).unwrap();
    let d = dstate(&mut state);
    assert_eq!(
        d.slot0,
        Slot0 {
            variant,
            kept: None
        }
    );
    // The switch reserves nothing.
    assert_eq!(
        gpu.read_alloc(d.ctx_hidden_acc).unwrap().len(),
        WINDOW * SLOT
    );
    d.ctx_positions = stamp.map_or(Vec::new(), |first| (first..first + 2).collect());
    d.ctx_len = d.ctx_positions.len();
    d.slot0.kept = Some(LAYERS);
    gpu.memset(d.ctx_hidden_acc, 0xEE, WINDOW * SLOT).unwrap();
    state
}

#[test]
fn zero_fills_exactly_slot_0_when_it_is_stamped_position_0() {
    let gpu = MockGpuBackend::new();
    let mut state = after_prefill(DebugSlot0::Zero, Some(0), &gpu);
    let d = dstate(&mut state);
    d.end_prefill_slot0(&gpu, 7).unwrap();
    assert_eq!(slot(&gpu, d, 0), vec![0; SLOT]);
    for index in 1..WINDOW {
        assert_eq!(slot(&gpu, d, index), vec![0xEE; SLOT], "slot {index}");
    }
    assert_eq!(d.slot0.kept, None);
}

/// Everything else issues no GPU operation at the end of prefill: the switch
/// unset, `keep`, and `zero` on a window that does not start at position 0
/// (a prompt longer than the window) or that holds nothing.
#[test]
fn nothing_else_touches_the_gpu_at_the_end_of_prefill() {
    for variant in ALL {
        for stamp in [Some(0), Some(12), None] {
            if (variant, stamp) == (DebugSlot0::Zero, Some(0)) {
                continue;
            }
            let gpu = MockGpuBackend::new();
            let mut state = after_prefill(variant, stamp, &gpu);
            let d = dstate(&mut state);
            let (ops, launches) = (gpu.d2d_count() + gpu.memset_count(), gpu.launch_count());
            d.end_prefill_slot0(&gpu, 7).unwrap();
            let case = format!("{variant:?} {stamp:?}");
            assert_eq!(gpu.d2d_count() + gpu.memset_count(), ops, "{case}");
            assert_eq!(gpu.launch_count(), launches, "{case}");
            assert_eq!(slot(&gpu, d, 0), vec![0xEE; SLOT], "{case}");
            // The `keep` record is per prefill.
            assert_eq!(d.slot0.kept, None, "{case}");
        }
    }
}
