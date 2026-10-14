// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix caching for vision prompts (issue #58).
//!
//! # Keys
//!
//! Every image of a prompt is the same run of placeholder tokens, so the radix
//! tree cannot be keyed on the prompt as the model sees it. At chunk 0 each
//! rank stamps `SequenceState::vision_cache_key`: the prompt with every
//! placeholder replaced by an id derived from its item's content hash and its
//! index in the item ([`vision_cache_key`]). Every prefix-cache call takes its
//! tokens through [`TransformerModel::pc_key`]; embedding, the splice, MRoPE
//! and the drafters keep reading the real tokens. The head computes the hashes
//! (`VisionItem::content_hash`) and sends them with the grids
//! (`EP_CMD_VISION_STATE`), so a worker rebuilds the same key and its radix
//! walk agrees with the head's before the EP min-sync.
//!
//! Only GLM-5 prompts are keyed. Its MRoPE walker rebuilds positions from the
//! complete prompt, so a prefill that starts after a cached image positions
//! the rest exactly as a full one would; the generic walker is chunk-local.
//!
//! The generated tokens of a vision sequence are not cached: decode writes
//! them at linear positions while a prefill of the same text writes the
//! compressed MRoPE positions, so their KV would not equal a recompute.
//! A next turn reuses the previous prompt and re-prefills the reply.
//!
//! # Encoding only what the prefill computes
//!
//! On the chunked path the head stages the items into the SEQUENCE instead of
//! encoding them ([`TransformerModel::prepare_vision_embed_deferred_dispatch`])
//! and the command carries only the grids and hashes. Once the chunk-0 lookup
//! has the match every rank agreed on, [`TransformerModel::vision_resolve_deferred`]
//! encodes only the items whose first computed placeholder falls inside chunk
//! 0's computed range, and on a multi-rank world broadcasts those rows. Every
//! later chunk encodes its own items the same way, right before its embed.
//! Leading items wholly inside the cached prefix skip the ViT; their grids
//! stay for MRoPE. The splice offsets its rows by the rows it skipped
//! (`skip_rows`) and indexes the rows still resident in `buf_out`
//! (`rows_encoded`/`rows_base`).

use std::borrow::Cow;
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::prefix_cache::vision_key::vision_cache_key;

use super::types::TransformerModel;
use crate::VisionItem;
use crate::traits::SequenceState;

/// `ATLAS_VISION_PREFIX_CACHE=0` turns vision prefix caching off (vision
/// prompts then neither look up nor insert, as before). Read once.
fn switch_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VISION_PREFIX_CACHE").as_deref() != Ok("0"))
}

/// Leading items whose every placeholder sits before `computed_from`, and
/// their placeholder (= encoder row) count. `counts` holds each item's
/// placeholders in prompt order.
pub(crate) fn cached_items(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
    computed_from: usize,
) -> (usize, usize) {
    let (mut item, mut rows, mut seen) = (0usize, 0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() || pos >= computed_from {
            break;
        }
        if is_pad(t) {
            seen += 1;
            if seen == counts[item] {
                rows += seen;
                item += 1;
                seen = 0;
            }
        }
    }
    (item, rows)
}

/// The items a chunk must encode: those with a placeholder AT OR AFTER
/// `computed_from` inside `[from, to)` (the chunk's computed range), as
/// indices into the item list. An item straddling a chunk boundary is
/// RE-encoded at every chunk holding one of its placeholders: the encoder
/// writes its packed output at `buf_out` row 0, so a later chunk could not
/// index the earlier chunk's resident tail under one window base — and the
/// re-encode makes the resident window exactly this chunk's splice range.
/// The re-encoded head rows were already spliced; only the pads this chunk
/// computes are read. An item straddling `computed_from` (its cached head is
/// skipped) is encoded at the chunk that first computes one of its
/// placeholders.
///
/// `first_pad[i]`/`last_pad[i]` are item i's first/last placeholder positions
/// in the prompt; `computed_from` is the first prompt position the prefill
/// computes.
pub(crate) fn items_in_range(
    first_pad: &[usize],
    last_pad: &[usize],
    computed_from: usize,
    from: usize,
    to: usize,
) -> Vec<usize> {
    first_pad
        .iter()
        .zip(last_pad)
        .enumerate()
        .filter(|&(_, (&first, &last))| {
            let first = first.max(computed_from);
            first < to && last >= from
        })
        .map(|(i, _)| i)
        .collect()
}

/// First placeholder position of every item, in prompt order. Items' pad runs
/// may be split by text (a GLM video's timestamp tokens), so the first pad
/// after the previous item's last one is the item's own first — the same walk
/// [`cached_items`] uses to attribute pads to items. An item with no pad in
/// `tokens` gets `usize::MAX` (never inside any chunk range).
pub(crate) fn first_pad_positions(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
) -> Vec<usize> {
    let mut out = Vec::with_capacity(counts.len());
    let (mut item, mut seen) = (0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() {
            break;
        }
        if is_pad(t) {
            if seen == 0 {
                out.push(pos);
            }
            seen += 1;
            if seen == counts[item] {
                item += 1;
                seen = 0;
            }
        }
    }
    out.resize(counts.len(), usize::MAX);
    out
}

/// Last placeholder position of every item, in prompt order — the same walk
/// [`first_pad_positions`] uses, recording the position each item's run ends
/// on. An item with no pad in `tokens` gets 0 (never passes a `>= from` test
/// against a real chunk start).
pub(crate) fn last_pad_positions(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
) -> Vec<usize> {
    let mut out = vec![0; counts.len()];
    let (mut item, mut seen) = (0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() {
            break;
        }
        if is_pad(t) {
            seen += 1;
            if seen == counts[item] {
                out[item] = pos;
                item += 1;
                seen = 0;
            }
        }
    }
    out
}

impl TransformerModel {
    /// Whether vision prompts are keyed for the prefix cache (module docs).
    pub(super) fn vision_prefix_cache_ok(&self) -> bool {
        self.prefix_cache.is_active()
            && self.vision_encoder.is_some()
            && self.config.vision.as_ref().is_some_and(|v| v.is_glm5_next)
            && switch_on()
    }

    /// Stamp `seq`'s cache key from the sequence's own vision state. Chunk 0
    /// only; a prompt without placeholders, or whose items cannot be keyed,
    /// keeps an empty key (and therefore stays uncached if it has
    /// placeholders).
    pub(in crate::model) fn stamp_vision_cache_key(&self, tokens: &[u32], seq: &mut SequenceState) {
        seq.vision_cache_key.clear();
        if !self.vision_prefix_cache_ok() || !self.tokens_have_vision_pad(tokens) {
            return;
        }
        let grids = &seq.vision_grids;
        let hashes = &seq.vision_hashes;
        if grids.len() != hashes.len() {
            return;
        }
        let items: Vec<(u128, usize)> = grids
            .iter()
            .zip(hashes)
            .map(|(&(t, h, w), &hash)| (hash, t * h * w))
            .collect();
        if let Some(key) = vision_cache_key(tokens, |t| self.is_vision_pad(t), &items) {
            seq.vision_cache_key = key;
        }
    }

    /// `tokens` (a prefix of `seq`'s stream from position 0) with the
    /// stamped key over the positions it covers.
    pub(in crate::model) fn pc_key_tokens<'a>(
        &self,
        seq: &SequenceState,
        tokens: &'a [u32],
    ) -> Cow<'a, [u32]> {
        let key = &seq.vision_cache_key;
        if key.is_empty() {
            return Cow::Borrowed(tokens);
        }
        let n = key.len().min(tokens.len());
        let mut out = Vec::with_capacity(tokens.len());
        out.extend_from_slice(&key[..n]);
        out.extend_from_slice(&tokens[n..]);
        Cow::Owned(out)
    }

    /// The prefix-cache key for `tokens` (a prefix of `seq`'s stream from
    /// position 0), or `None` when it holds a placeholder the key does not
    /// cover: such a prefix must not be looked up or cached.
    pub(in crate::model) fn pc_key<'a>(
        &self,
        seq: &SequenceState,
        tokens: &'a [u32],
    ) -> Option<Cow<'a, [u32]>> {
        let covered = seq.vision_cache_key.len().min(tokens.len());
        (!self.tokens_have_vision_pad(&tokens[covered..])).then(|| self.pc_key_tokens(seq, tokens))
    }

    /// Whether the current request's encode waits for its chunk-0 lookup.
    pub(in crate::model) fn vision_encode_deferred(seq: &SequenceState) -> bool {
        seq.vision_deferred
    }

    /// Clear a sequence's per-request vision state (the sequence is leaving
    /// the prefill path).
    pub(in crate::model) fn clear_seq_vision_state(seq: &mut SequenceState) {
        seq.vision_items.clear();
        seq.vision_grids.clear();
        seq.vision_hashes.clear();
        seq.vision_skip_rows = 0;
        seq.vision_computed_from = 0;
        seq.vision_deferred = false;
        seq.vision_rows_encoded = 0;
        seq.vision_rows_base = 0;
        seq.vision_row_base = 0;
        seq.vision_grid_base = 0;
        seq.vision_owned_images = 0;
        seq.vision_slice_rows = 0;
    }

    /// Derive the per-item post-merge grids and hashes of `images` into `seq`.
    /// Shared by every staging path: the key, MRoPE and the worker's command
    /// need them before any encode runs.
    fn stage_vision_grids(seq: &mut SequenceState, images: &[VisionItem], sms: usize) {
        seq.vision_grids = images
            .iter()
            .map(|it| (it.t_len(), it.grid_h / sms, it.grid_w / sms))
            .collect();
        seq.vision_hashes = images.iter().map(|it| it.content_hash).collect();
    }

    /// Trait entry for [`Self::stage_vision_grids`] (the co-dispatch path's
    /// rows are already encoded; only the geometry is staged).
    pub(in crate::model) fn stage_vision_grids_dispatch(
        &self,
        seq: &mut SequenceState,
        images: &[VisionItem],
    ) {
        let sms = self
            .config
            .vision
            .as_ref()
            .map_or(1, |v| v.spatial_merge_size.max(1));
        Self::stage_vision_grids(seq, images, sms);
    }

    /// Head, chunked path: stage `images` into `seq` for the encode the
    /// chunk-0 lookup resolves. The grids and hashes are set now (the key,
    /// MRoPE and the worker's command need them); `buf_out` holds nothing
    /// yet. Encodes at once when vision prompts are not cached.
    pub(in crate::model) fn prepare_vision_embed_deferred_dispatch(
        &self,
        seq: &mut SequenceState,
        images: &[VisionItem],
    ) -> Result<()> {
        if !self.vision_prefix_cache_ok() {
            return self.prepare_vision_embed_dispatch(seq, images);
        }
        let sms = self
            .config
            .vision
            .as_ref()
            .map_or(1, |v| v.spatial_merge_size.max(1));
        seq.vision_rows_encoded = 0;
        seq.vision_rows_base = 0;
        seq.vision_skip_rows = 0;
        seq.vision_computed_from = 0;
        Self::stage_vision_grids(seq, images, sms);
        seq.vision_items = images.to_vec();
        seq.vision_deferred = true;
        Ok(())
    }

    /// Run a deferred encode once the chunk-0 lookup has fixed the first
    /// computed position (module docs). Every rank calls this at the same
    /// point. The ranks min-reduce `computed_from` first, so they size the
    /// payload broadcast alike even if their restore depths ever differed;
    /// encoding from an earlier position is always correct. No-op otherwise.
    ///
    /// Encodes only the items whose first computed placeholder falls inside
    /// `[computed_from, chunk_end)`; a later chunk encodes its own items the
    /// same way (`vision_encode_chunk_items`).
    pub(in crate::model) fn vision_resolve_deferred(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        computed_from: usize,
        chunk_end: usize,
    ) -> Result<()> {
        if !seq.vision_deferred {
            return Ok(());
        }
        // `vision_deferred` stays set: it now means "this prefill encodes
        // per chunk" (later chunks encode their own items right before their
        // embed), not "the chunk-0 lookup is still pending".
        let computed_from = if self.multi_rank_protocol_active() {
            let local = u32::try_from(computed_from).unwrap_or(u32::MAX);
            self.ep_min_u32(local)? as usize
        } else {
            computed_from
        };
        let counts: Vec<usize> = seq
            .vision_grids
            .iter()
            .map(|&(t, h, w)| t * h * w)
            .collect();
        let (skip, skip_rows) =
            cached_items(tokens, |t| self.is_vision_pad(t), &counts, computed_from);
        // Before the encode: `vision_encode_items` subtracts `skip_rows` when
        // it places the run's window (`rows_base`), so the cached leading rows
        // must already be recorded here.
        seq.vision_skip_rows = skip_rows;
        seq.vision_computed_from = computed_from;
        let items: Vec<usize> = items_in_range(
            &first_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            &last_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            computed_from,
            computed_from,
            chunk_end,
        );
        let rows = self.vision_encode_items(seq, &items)?;
        tracing::info!(
            "Vision prefix cache: {skip} of {} items cached ({skip_rows} rows), encoded {rows} rows \
             for positions from {computed_from}",
            counts.len()
        );
        Ok(())
    }

    /// Encode the items of `seq` whose placeholder range intersects this
    /// chunk's range `[chunk_start, chunk_end)` (module docs). Called for
    /// every chunk after the first, right before its embed; chunk 0's encode
    /// ran inside the lookup (`vision_resolve_deferred`). Only a DEFERRED
    /// prefill reaches here: an immediate or co-dispatched encode already
    /// wrote every row this prefill splices, and re-encoding would clobber
    /// the shared `buf_out` window.
    pub(in crate::model) fn vision_encode_chunk_items(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        chunk_start: usize,
        chunk_end: usize,
    ) -> Result<()> {
        // Grids, not items: a worker holds no pixels but must reach the
        // encode's broadcast (its rows arrive by receive, `vision_encode_items`).
        if !seq.vision_deferred || seq.vision_grids.is_empty() {
            return Ok(());
        }
        let counts: Vec<usize> = seq
            .vision_grids
            .iter()
            .map(|&(t, h, w)| t * h * w)
            .collect();
        let items: Vec<usize> = items_in_range(
            &first_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            &last_pad_positions(tokens, |t| self.is_vision_pad(t), &counts),
            seq.vision_computed_from,
            chunk_start,
            chunk_end,
        );
        self.vision_encode_items(seq, &items)?;
        Ok(())
    }

    /// Encode the named items of `seq` into `buf_out` at row 0 and broadcast
    /// them on a multi-rank world. Returns the row count written. The rows
    /// are complete before the splice reads them on the prefill stream and
    /// before the communicator's stream sends them.
    ///
    /// A worker (receive-only for vision) holds no pixels: it skips the
    /// encode and receives the head's broadcast into its own `buf_out`
    /// instead, so both ranks end with the same resident rows.
    fn vision_encode_items(&self, seq: &mut SequenceState, items: &[usize]) -> Result<usize> {
        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision encode without an encoder"))?;
        let grids = &seq.vision_grids;
        // The encoded run's place in the prompt-order packed non-cached
        // output: every item before the first encoded one contributes its
        // rows (cached items' rows are skipped by the splice, not packed).
        // Computed on BOTH ranks (the worker has no pixels but must slide
        // its window identically), so it cannot depend on the items alone.
        let a: usize = items
            .first()
            .map_or(seq.vision_rows_base, |&first| {
                (0..first)
                    .map(|i| {
                        let (t, h, w) = grids[i];
                        t * h * w
                    })
                    .sum()
            })
            .saturating_sub(seq.vision_skip_rows);
        let rows: usize = items
            .iter()
            .map(|&i| {
                let (t, h, w) = grids[i];
                t * h * w
            })
            .sum();
        if !items.is_empty() && rows > 0 && !seq.vision_items.is_empty() {
            // The head alone holds pixels: a worker (receive-only for vision)
            // skips the encode and takes the rows from the broadcast below.
            ensure!(
                grids.len() == seq.vision_items.len(),
                "vision items and grids disagree"
            );
            let refs: Vec<&VisionItem> = items.iter().map(|&i| &seq.vision_items[i]).collect();
            let per_group =
                ve.forward_items(&refs, self.gpu.as_ref(), self.gpu.default_stream())?;
            // The grids were derived before the encode (the key and the
            // worker needed them); the encoder's own geometry must agree.
            let mut group = 0usize;
            for (&i, (t_len, h, w)) in items.iter().zip(items.iter().map(|&i| grids[i])) {
                let item = &seq.vision_items[i];
                let (eh, ew, _) = per_group[group];
                ensure!(
                    (item.t_len(), eh, ew) == (t_len, h, w),
                    "deferred vision grid {:?} disagrees with the encoder's {:?}",
                    (t_len, h, w),
                    (item.t_len(), eh, ew)
                );
                group += item.t_len();
            }
            let encoded: usize = per_group.iter().map(|&(_, _, m)| m).sum();
            ensure!(
                encoded == rows,
                "deferred vision encode produced {encoded} rows, the grids expect {rows}"
            );
        }
        // The window is exactly this chunk's splice range: every item it
        // holds has a placeholder in this chunk, and every item of earlier
        // chunks was either fully spliced there or is re-encoded here. So
        // `rows_base` is the run's virtual start (in the prompt-order packed
        // non-cached output) and `rows_encoded` the run's rows — the previous
        // window is obsolete. An empty selection encodes nothing and leaves
        // the window alone (the chunk splices no pads).
        if rows > 0 {
            seq.vision_rows_base = a;
            seq.vision_rows_encoded = rows;
        }
        self.gpu.synchronize(self.gpu.default_stream())?;
        if rows > 0 && self.multi_rank_protocol_active() {
            ensure!(
                rows <= ve.output_rows(),
                "deferred vision rows exceed encoder capacity"
            );
            let comm = self
                .comm
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("deferred vision payload without communicator"))?;
            comm.broadcast(ve.buf_out.0, rows * ve.out_hidden_size * 2, 0)?;
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::{cached_items, first_pad_positions, items_in_range, last_pad_positions};

    const PAD: u32 = 9;

    #[test]
    fn only_items_wholly_before_the_computed_position_are_cached() {
        // text, item 0 (2 rows) at 1..3, text, item 1 (3 rows) at 4..7, text.
        let t = [1, PAD, PAD, 2, PAD, PAD, PAD, 3];
        let pad = |x| x == PAD;
        assert_eq!(cached_items(&t, pad, &[2, 3], 0), (0, 0));
        assert_eq!(
            cached_items(&t, pad, &[2, 3], 2),
            (0, 0),
            "item 0 is half computed"
        );
        assert_eq!(cached_items(&t, pad, &[2, 3], 3), (1, 2));
        assert_eq!(
            cached_items(&t, pad, &[2, 3], 6),
            (1, 2),
            "item 1 is partly computed"
        );
        assert_eq!(cached_items(&t, pad, &[2, 3], 7), (2, 5));
        assert_eq!(cached_items(&t, pad, &[2, 3], 99), (2, 5));
    }

    #[test]
    fn split_video_runs_count_by_item() {
        // One item of 3 placeholders split by timestamp text.
        let t = [PAD, 5, PAD, PAD, 6];
        assert_eq!(cached_items(&t, |x| x == PAD, &[3], 3), (0, 0));
        assert_eq!(cached_items(&t, |x| x == PAD, &[3], 4), (1, 3));
    }

    fn walks(t: &[u32], counts: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let pad = |x: u32| x == PAD;
        (
            first_pad_positions(t, pad, counts),
            last_pad_positions(t, pad, counts),
        )
    }

    #[test]
    fn a_straddling_item_is_re_encoded_at_every_chunk_it_spans() {
        // item 0 (4 rows) pads at 2..6, item 1 (2 rows) pads at 7..9. Chunks
        // [0,5) and [5,10): item 0 straddles (pads 2,3,4 in chunk 0; pad 5 in
        // chunk 1), so chunk 1 re-encodes it together with item 1.
        let t = [1, 2, PAD, PAD, PAD, PAD, 3, PAD, PAD, 4];
        let (first, last) = walks(&t, &[4, 2]);
        assert_eq!(first, vec![2, 7]);
        assert_eq!(last, vec![5, 8]);
        assert_eq!(items_in_range(&first, &last, 0, 0, 5), vec![0]);
        assert_eq!(
            items_in_range(&first, &last, 0, 5, 10),
            vec![0, 1],
            "item 0's tail pad sits in chunk 1 — re-encode it there"
        );
        // A chunk past every pad encodes nothing.
        assert!(items_in_range(&first, &last, 0, 10, 99).is_empty());
    }

    #[test]
    fn cached_leading_items_are_never_selected() {
        // item 0 (2 rows) wholly cached (computed_from 3), item 1 (2 rows) at
        // 4..6, item 2 (2 rows) at 7..9. Chunk [3,6) selects item 1 only.
        let t = [1, PAD, PAD, 2, PAD, PAD, 3, PAD, PAD, 4];
        let (first, last) = walks(&t, &[2, 2, 2]);
        assert_eq!(items_in_range(&first, &last, 3, 3, 6), vec![1]);
        assert_eq!(
            items_in_range(&first, &last, 3, 6, 9),
            vec![2],
            "item 1's pads all sit before 6 — chunk [6,9) encodes only item 2"
        );
        assert!(items_in_range(&first, &last, 3, 9, 99).is_empty());
    }

    #[test]
    fn a_video_run_split_by_timestamp_text_is_one_item() {
        // One item of 4 placeholders split by timestamp text: [PAD, 5, PAD,
        // 6, PAD, 7, PAD]. Its pads span chunks [0,3) and [3,7) — the item is
        // re-encoded at both.
        let t = [PAD, 5, PAD, 6, PAD, 7, PAD];
        let (first, last) = walks(&t, &[4]);
        assert_eq!(first, vec![0]);
        assert_eq!(last, vec![6]);
        assert_eq!(items_in_range(&first, &last, 0, 0, 3), vec![0]);
        assert_eq!(items_in_range(&first, &last, 0, 3, 7), vec![0]);
        assert!(items_in_range(&first, &last, 0, 7, 99).is_empty());
    }

    #[test]
    fn an_item_without_pads_is_never_selected() {
        let t = [1, 2, 3];
        let (first, last) = walks(&t, &[2]);
        assert_eq!(first, vec![usize::MAX]);
        assert_eq!(last, vec![0]);
        assert!(items_in_range(&first, &last, 0, 0, 3).is_empty());
    }
}
