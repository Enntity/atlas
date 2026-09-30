// SPDX-License-Identifier: AGPL-3.0-only

//! Sequences sharing one prefix cache, replayed on the host through the
//! production lookup, adoption, block allocation, cache insert and release.
//! Every KV row write checks that the writer holds the block alone, and every
//! read checks that the row was computed for the reader's own token prefix.

use std::collections::HashMap;

use super::adopt_prefix_match;
use crate::model::block_mgmt::{
    cache_acquires_refs, ensure_blocks_through_decode, ensure_blocks_through_prefill,
};
use crate::traits::SequenceState;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{
    KvCacheConfig, KvCacheDtype, PagedKvCache, SparseIndexCacheConfig, TailSlotPlan,
};
use spark_runtime::prefix_cache::{PrefixCache, PrefixMatch};
use spark_runtime::radix_tree::RadixTree;

const BS: usize = 16;
/// A prompt that ends five tokens into its third block.
const PROMPT: usize = 2 * BS + 5;

fn toks(range: std::ops::Range<u32>) -> Vec<u32> {
    range.collect()
}

fn join(parts: &[&[u32]]) -> Vec<u32> {
    parts.concat()
}

/// Stands in for the KV a row holds: a function of the whole token prefix.
fn prefix_hash(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325, |h, &t| {
        (h ^ u64::from(t)).wrapping_mul(0x0100_0000_01b3)
    })
}

struct World {
    gpu: MockGpuBackend,
    kv: PagedKvCache,
    cache: RadixTree,
    /// `(block, row) -> prefix_hash` of the tokens the row was written for.
    rows: HashMap<(u32, usize), u64>,
}

impl World {
    /// A GLM-style cache with slotted index tails, so a tail stripped from a
    /// block its sequence still writes fails the step.
    fn new() -> Self {
        let gpu = MockGpuBackend::new();
        let config = KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut kv = PagedKvCache::new(config, 96, &gpu).unwrap();
        let plan = TailSlotPlan {
            lag_blocks: 2,
            sequences: 4,
        };
        kv.attach_sparse_index_with_tail_slots(
            SparseIndexCacheConfig::bf16(4, 128),
            Some(plan),
            &gpu,
        )
        .unwrap();
        Self {
            gpu,
            kv,
            cache: RadixTree::new(),
            rows: HashMap::new(),
        }
    }

    /// Write the row for `fed[pos]` (computed over `fed[..=pos]`).
    fn write_row(&mut self, seq: &SequenceState, fed: &[u32], pos: usize) {
        let block = seq.block_table[pos / BS];
        assert_eq!(
            self.kv.ref_count(block),
            1,
            "position {pos} is written into block {block}, which has another holder"
        );
        self.rows
            .insert((block, pos % BS), prefix_hash(&fed[..=pos]));
    }

    /// Attention reads every committed row of the sequence.
    fn read_rows(&self, seq: &SequenceState) {
        for pos in 0..seq.seq_len {
            let block = seq.block_table[pos / BS];
            assert_eq!(
                self.rows.get(&(block, pos % BS)),
                Some(&prefix_hash(&seq.tokens[..=pos])),
                "row {pos} (block {block}) was not computed for this sequence's tokens"
            );
        }
    }

    /// Chunk-0 lookup through the end-of-prefill cache insert.
    fn prefill(&mut self, prompt: &[u32]) -> SequenceState {
        let mut seq = SequenceState::host_only(0);
        let m = self.cache.lookup_whole_blocks(prompt, BS, 0, 0);
        adopt_prefix_match(&mut seq, &m, &mut self.kv);
        seq.prompt_len = prompt.len();
        let last = (prompt.len() - 1) / BS;
        ensure_blocks_through_prefill(
            &mut seq,
            last,
            &mut self.kv,
            &self.cache,
            &self.gpu,
            0,
            false,
        )
        .unwrap();
        for pos in m.matched_tokens..prompt.len() {
            self.write_row(&seq, prompt, pos);
        }
        seq.tokens = prompt.to_vec();
        seq.seq_len = prompt.len();
        self.read_rows(&seq);
        let acquired = self.cache.insert(
            prompt,
            &seq.block_table,
            &[],
            BS,
            seq.cached_prefix_tokens,
            0,
        );
        cache_acquires_refs(&acquired, &mut self.kv);
        seq
    }

    fn grow(&mut self, seq: &mut SequenceState, pos: usize) {
        ensure_blocks_through_decode(
            seq,
            pos / BS,
            &mut self.kv,
            &self.cache,
            &self.gpu,
            0,
            false,
        )
        .unwrap();
    }

    fn decode(&mut self, seq: &mut SequenceState, tokens: &[u32]) {
        for &token in tokens {
            self.read_rows(seq);
            let pos = seq.seq_len;
            self.grow(seq, pos);
            seq.tokens.push(token);
            self.write_row(seq, &seq.tokens, pos);
            seq.seq_len += 1;
        }
    }

    /// A speculative verify: writes one row per draft, rejected ones included,
    /// then commits the first `accepted`.
    fn verify(&mut self, seq: &mut SequenceState, drafts: &[u32], accepted: usize) {
        self.read_rows(seq);
        let base = seq.seq_len;
        let fed = join(&[&seq.tokens, drafts]);
        for t in 0..drafts.len() {
            self.grow(seq, base + t);
            self.write_row(seq, &fed, base + t);
        }
        seq.tokens.extend_from_slice(&drafts[..accepted]);
        seq.seq_len = base + accepted;
    }

    /// `cache_sequence` then `free_sequence`.
    fn retire(&mut self, seq: SequenceState) {
        self.read_rows(&seq);
        if seq.tokens.len() >= BS {
            let acquired =
                self.cache
                    .insert(&seq.tokens, &seq.block_table, &[], BS, seq.prompt_len, 0);
            cache_acquires_refs(&acquired, &mut self.kv);
        }
        self.cache.release(&seq.tokens, BS, 0);
        self.kv.free_blocks(&seq.block_table);
    }
}

/// Live donor: the same prompt arrives again while its first sequence is
/// still decoding. Both then decode, and either may finish first.
#[test]
fn live_donor_and_its_twin_write_their_own_blocks() {
    let prompt = toks(0..PROMPT as u32);
    for twin_leaves_first in [true, false] {
        let mut w = World::new();
        let mut donor = w.prefill(&prompt);
        let mut twin = w.prefill(&prompt);
        assert_eq!(twin.cached_prefix_tokens, 2 * BS);
        assert_eq!(twin.block_table[..2], donor.block_table[..2]);
        assert_ne!(twin.block_table[2], donor.block_table[2]);

        w.decode(&mut donor, &[500]);
        w.verify(&mut twin, &[600, 601, 602], 1);
        w.decode(&mut donor, &[501, 502]);
        w.verify(&mut twin, &toks(610..630), 14);

        let (leaver, mut stayer) = if twin_leaves_first {
            (twin, donor)
        } else {
            (donor, twin)
        };
        w.retire(leaver);
        // The stayer's frontier block kept its index tail and its rows.
        w.decode(&mut stayer, &toks(700..740));
        w.retire(stayer);
    }
}

/// Retired donor: the choices of a blocking `n > 1` request run one after
/// another on one prompt, each after the previous one retired.
#[test]
fn serial_choices_of_one_prompt_do_not_touch_each_other() {
    let prompt = toks(0..PROMPT as u32);
    let mut w = World::new();
    let mut first = w.prefill(&prompt);
    let answer = toks(500..530);
    w.decode(&mut first, &answer);
    w.retire(first);

    for choice in 1..4u32 {
        let mut seq = w.prefill(&prompt);
        assert_eq!(seq.cached_prefix_tokens, 2 * BS, "choice {choice}");
        w.verify(&mut seq, &toks(choice * 1000..choice * 1000 + 8), 3);
        w.decode(&mut seq, &toks(choice * 2000..choice * 2000 + 5 * choice));
        w.retire(seq);
    }

    // The first choice's conversation continues and reads its own rows.
    let turn2 = join(&[&prompt, &answer, &toks(900..920)]);
    let mut seq = w.prefill(&turn2);
    w.decode(&mut seq, &toks(950..960));
    w.retire(seq);
}

/// Strict-prefix retry: turn 1 is sent again after turn 2, a longer prompt
/// with the same start, was cached. The retry's prompt ends inside a block
/// that holds turn-2 rows; the conversation's next turn must still read them.
#[test]
fn strict_prefix_retry_leaves_the_longer_prompt_intact() {
    let turn1 = toks(0..PROMPT as u32);
    let answer1 = toks(500..520);
    let turn2 = join(&[&turn1, &answer1, &toks(900..930)]);
    let answer2 = toks(600..625);
    for turn2_is_live in [false, true] {
        let mut w = World::new();
        let mut seq = w.prefill(&turn1);
        w.decode(&mut seq, &answer1);
        w.retire(seq);
        let mut conversation = w.prefill(&turn2);

        let retry_now = |w: &mut World| {
            let mut retry = w.prefill(&turn1);
            assert_eq!(retry.cached_prefix_tokens, 2 * BS);
            w.verify(&mut retry, &toks(7000..7008), 2);
            w.decode(&mut retry, &toks(7100..7140));
            w.retire(retry);
        };
        if turn2_is_live {
            retry_now(&mut w);
        }
        w.decode(&mut conversation, &answer2);
        w.retire(conversation);
        if !turn2_is_live {
            retry_now(&mut w);
        }

        let turn3 = join(&[&turn2, &answer2, &toks(950..970)]);
        let mut seq = w.prefill(&turn3);
        assert!(seq.cached_prefix_tokens >= turn2.len() / BS * BS);
        w.decode(&mut seq, &toks(980..990));
        w.retire(seq);
    }
}

/// The four-request chain: A finishes, B re-sends A's prompt and generates
/// the same answer, A sends turn 2 (whose insert finds the third chunk's
/// node already cached) and turn 3 (which matches through that node).
#[test]
fn four_request_chain_reads_only_rows_written_for_it() {
    let prompt = toks(0..PROMPT as u32);
    let answer = toks(500..520);
    let mut w = World::new();
    let mut a1 = w.prefill(&prompt);
    w.decode(&mut a1, &answer);
    let a1_third_block = a1.block_table[2];
    w.retire(a1);

    let mut b = w.prefill(&prompt);
    assert_ne!(b.block_table[2], a1_third_block);
    // Rejected drafts are written too, then the accepted tokens over them.
    w.verify(&mut b, &[500, 8001, 8002, 8003], 1);
    w.decode(&mut b, &answer[1..]);
    w.retire(b);

    let turn2 = join(&[&prompt, &answer, &toks(900..925)]);
    let answer2 = toks(600..630);
    let mut a2 = w.prefill(&turn2);
    w.decode(&mut a2, &answer2);
    w.retire(a2);

    let turn3 = join(&[&turn2, &answer2, &toks(950..975)]);
    let mut a3 = w.prefill(&turn3);
    assert!(a3.cached_prefix_tokens >= turn2.len() / BS * BS);
    assert_eq!(a3.block_table[2], a1_third_block);
    w.decode(&mut a3, &toks(980..990));
    w.retire(a3);
}

/// A prompt that ends on a block boundary is matched in full; the twins
/// still decode into separate blocks.
#[test]
fn block_aligned_prompts_share_every_prompt_block() {
    let prompt = toks(0..(2 * BS) as u32);
    let mut w = World::new();
    let mut donor = w.prefill(&prompt);
    let mut twin = w.prefill(&prompt);
    assert_eq!(twin.cached_prefix_tokens, 2 * BS);
    assert_eq!(twin.block_table, donor.block_table);
    w.decode(&mut donor, &[500, 501]);
    w.verify(&mut twin, &[600, 601, 602], 2);
    assert_ne!(twin.block_table[2], donor.block_table[2]);
    w.retire(donor);
    w.decode(&mut twin, &toks(700..720));
    w.retire(twin);
}

#[test]
#[should_panic(expected = "whole blocks")]
fn a_match_that_ends_inside_a_block_is_not_adopted() {
    let mut w = World::new();
    let block = w.kv.alloc_block().unwrap();
    let inside = PrefixMatch {
        matched_blocks: vec![block],
        matched_tokens: 5,
        ..PrefixMatch::empty()
    };
    adopt_prefix_match(&mut SequenceState::host_only(0), &inside, &mut w.kv);
}
