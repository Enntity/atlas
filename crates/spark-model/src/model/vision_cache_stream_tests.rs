// SPDX-License-Identifier: AGPL-3.0-only

//! The per-chunk encode on a real model with a recording mock backend: when it
//! may rewrite the shared encoder buffer, and how long a sequence keeps its
//! pixels.

// The real model and recording layer of `prefill_stream_tests`.
#[allow(clippy::duplicate_mod)]
#[path = "trait_impl/prefill_stream_test_fixture.rs"]
mod fixture;

use fixture::{CALLER, DEFAULT, Fixture, Order};

use crate::layers::VisionEncoder;
use crate::layers::vision_encoder::IMAGE_PAD_TOKEN_ID as PAD;
use crate::layers::vision_encoder::glm::test_tower;
use crate::traits::Model;

/// A single-rank fixture with the tiny GLM tower and one staged 64-row image
/// whose placeholders sit at positions 4..68 of a 72-token prompt.
fn staged() -> (Fixture, Vec<u32>) {
    let mut f = Fixture::new(1, 1, 0);
    let tower = VisionEncoder::new_glm(
        Some(test_tower::weights()),
        &test_tower::config(),
        f.model.gpu.as_ref(),
    )
    .unwrap();
    f.model.vision_encoder = Some(tower);
    let item = test_tower::image(16);
    f.seq.vision.grids = vec![(1, 8, 8)];
    f.seq.vision.hashes = vec![item.content_hash];
    f.seq.vision.items = vec![item];
    f.seq.vision.deferred = true;
    let tokens: Vec<u32> = [1, 2, 3, 4]
        .into_iter()
        .chain(std::iter::repeat_n(PAD, 64))
        .chain([5, 6, 7, 8])
        .collect();
    (f, tokens)
}

#[test]
fn a_chunk_encode_waits_for_the_prefill_stream_before_rewriting_buf_out() {
    // The previous chunk's splice copies out of `buf_out` on the prefill
    // stream (another sequence's, once image prompts interleave); the encoder
    // writes `buf_out` on the default stream. Draining only the default
    // stream would let the encode overwrite rows a queued splice still reads.
    let (mut f, tokens) = staged();
    f.seq.prefix_lookup_applied = true;
    f.order();
    f.model
        .vision_encode_chunk_items(&mut f.seq, &tokens, 0, tokens.len(), CALLER)
        .unwrap();
    let order = f.order();
    assert_eq!(
        order.first(),
        Some(&Order::Sync(CALLER)),
        "the prefill stream drains before the encode: {order:?}"
    );
    assert!(
        order.contains(&Order::Sync(DEFAULT)),
        "the rows complete before the splice"
    );
    assert_eq!(f.seq.vision.rows_encoded, 64);
}

#[test]
fn no_chunk_encodes_before_the_chunk_zero_lookup() {
    // Chunk 0's items are encoded inside its prefix lookup, once the cached
    // prefix is known; a batched path that embeds before its lookup must not
    // encode them early under an unresolved `computed_from`.
    let (mut f, tokens) = staged();
    f.order();
    f.model
        .vision_encode_chunk_items(&mut f.seq, &tokens, 0, tokens.len(), CALLER)
        .unwrap();
    assert!(f.order().is_empty(), "nothing launched or drained");
    assert_eq!(f.seq.vision.rows_encoded, 0);
}

#[test]
fn the_pixels_are_released_once_the_last_prompt_chunk_succeeds() {
    // A sequence keeps decoding for minutes after its prefill; its staged
    // images (tens of MB per still, far more per video) must not ride along.
    let (mut f, _) = staged();
    f.seq.vision.deferred = false;
    let text = [1, 2, 3, 4];
    f.model
        .prefill_chunk(&text, &mut f.seq, 0, text.len(), true, CALLER)
        .unwrap();
    assert!(f.seq.vision.items.is_empty());
}

#[test]
fn a_glm_prefill_encodes_per_chunk_even_with_the_prefix_cache_off() {
    // Per-chunk encoding is what keeps two image prompts from sharing
    // `buf_out` across ticks; it must not depend on caching (this fixture's
    // prefix cache is `NoPrefixCaching`). An eager encode here would put
    // every row in `buf_out` at chunk 0 for later chunks to read.
    let (mut f, _) = staged();
    f.model.config.vision = Some(test_tower::config());
    f.seq.vision = Default::default();
    let image = test_tower::image(16);
    f.order();
    f.model
        .prepare_vision_embed_deferred_dispatch(&mut f.seq, &[image])
        .unwrap();
    assert!(f.seq.vision.deferred);
    assert_eq!(f.seq.vision.rows_encoded, 0, "nothing encoded yet");
    assert_eq!(f.seq.vision.grids, vec![(1, 8, 8)]);
}

#[test]
fn every_tower_encodes_per_chunk_not_at_admission() {
    // A Qwen tower encoded at admission into the shared `buf_out`: a long
    // prompt's rows then waited there for its last chunk, and an image prompt
    // admitted meanwhile overwrote them (the long prompt answered about the
    // other image). The tower type must not matter.
    let (mut f, _) = staged();
    let mut qwen = test_tower::config();
    qwen.is_glm5_next = false;
    f.model.config.vision = Some(qwen);
    f.seq.vision = Default::default();
    let image = test_tower::image(16);
    f.order();
    f.model
        .prepare_vision_embed_deferred_dispatch(&mut f.seq, &[image])
        .unwrap();
    assert!(f.seq.vision.deferred);
    assert_eq!(f.seq.vision.rows_encoded, 0, "nothing encoded at admission");
    assert!(f.order().is_empty(), "no encoder launch at admission");
}
