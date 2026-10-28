// SPDX-License-Identifier: AGPL-3.0-only

/// Phase-C: maximum number of watchdog-triggered rollbacks a single
/// sequence may perform before the watchdog reverts to a hard stop.
/// Bounds the worst case where re-steering re-enters the same attractor
/// — without this a degenerate sequence could rollback indefinitely.
pub const ROLLBACK_RESTEER_CAP: u32 = 2;

/// Phase-C: number of boundary SSM-state snapshots retained per sequence in
/// the decode-rollback ring (hybrid GDN/Mamba models). DECOUPLED from
/// [`ROLLBACK_RESTEER_CAP`]: the cap bounds how many times we re-steer, but
/// the ring must retain enough *boundary* snapshots that a clean PRE-loop
/// boundary survives long enough to roll back to. Sizing it at the old
/// `CAP + 1 = 3` meant a loop spanning ≥3 sentence/newline boundaries evicted
/// the clean boundary before the fuzzy detector (3 repeats) fired, forcing a
/// `NoSsmSnapshot` decline → hard-stop (observed: Claude-Code @ nvfp4-head,
/// 2026-06-07). 8 covers the 3-repeat detector with margin at modest cost
/// (8 × max_batch × per-layer GDN state, allocated once). Pure-attention
/// models ignore this (their ring is 0; they roll back to any boundary).
pub const DECODE_ROLLBACK_RING_SLOTS: usize = 8;

/// Domain salt folded into every decode cold-tier key so a decode-ring blob can
/// never collide with a Marconi prefix-hash key on a shared store/peer.
pub const DECODE_DOMAIN: u64 = 0xD3C0_DE12_A5B6_C7D8;
