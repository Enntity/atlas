// SPDX-License-Identifier: AGPL-3.0-only

//! How many vision rows a request may carry, decided once and shared by the
//! API's admission check and the engine.
//!
//! A vision row is one merged patch: one placeholder token in the prompt and
//! one `out_hidden_size` BF16 row of encoder output. The chunked prefill
//! encodes, before each chunk's embed, only the items whose placeholders fall
//! inside that chunk (`model::vision_cache`), so what has to fit the encoder's
//! output buffer is one chunk's items, not the whole request. Three bounds
//! follow:
//!
//! - **window** — rows the encoder output buffer holds. A chunk of at most
//!   `chunk_tokens` tokens touches every item whose pads it covers, including
//!   the two it may cut through at either end, so the buffer is sized
//!   [`window_rows`] = `chunk_tokens + 2 x` the largest still image, never
//!   below the historical one-batch capacity.
//! - **request** — rows one request may carry in total. Every row is a prompt
//!   token, so the context bounds it; the operator bounds it further because
//!   rank 0 holds the request's preprocessed pixels (f32, 24 KiB per merged
//!   row on the Qwen tower) from admission until its prefill ends.
//! - **items** — the rank-to-rank protocol's item limit.
//!
//! [`VisionRowLimits::check`] rejects a request that one of them would fail
//! after admission, so the client gets a 400 it can act on instead of a
//! prefill error it retries forever.

/// Items one request may carry: the rank-to-rank protocol's limit
/// (`model::vision_transport::VISION_MAX_GRID_ITEMS`).
pub const MAX_VISION_ITEMS: usize = 1024;

/// Merged rows the encoder output buffer must hold so that any prefill chunk
/// of at most `chunk_tokens` tokens can encode every item it touches.
///
/// `p_max` is the encoder's patch capacity (one batch, and the largest still
/// image); a still image yields at most `p_max / merge^2` rows. The result is
/// never below `p_max`, the buffer's historical size, so a request that fit
/// before still fits. `chunk_tokens == 0` (unknown) keeps the historical size.
pub fn window_rows(p_max: usize, spatial_merge_size: usize, chunk_tokens: usize) -> usize {
    if chunk_tokens == 0 {
        return p_max;
    }
    let merge2 = spatial_merge_size
        .max(1)
        .saturating_mul(spatial_merge_size.max(1));
    let image_rows = p_max / merge2;
    p_max.max(chunk_tokens.saturating_add(image_rows.saturating_mul(2)))
}

/// The most rows one chunk can need for a request whose items carry
/// `item_rows` (prompt order), over every chunking with chunks of at most
/// `chunk_tokens` tokens.
///
/// A chunk that touches items `i < j` holds at least one pad of each and all
/// the pads of the items between, so it needs `r_i + r_j + middle` rows with
/// `middle <= chunk_tokens - 2` and `middle <= total - r_i - r_j`. That grows
/// with `r_i + r_j`, so the two largest items give the bound; one item alone
/// needs its own rows.
pub fn worst_chunk_rows(item_rows: &[usize], chunk_tokens: usize) -> usize {
    let total: usize = item_rows.iter().sum();
    let (mut top1, mut top2) = (0usize, 0usize);
    for &r in item_rows {
        if r > top1 {
            top2 = top1;
            top1 = r;
        } else if r > top2 {
            top2 = r;
        }
    }
    let middle = chunk_tokens.saturating_sub(2).min(total - top1 - top2);
    top1 + top2 + middle
}

/// The bounds one server applies to every vision request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionRowLimits {
    /// Encoder output rows ([`window_rows`]).
    pub window_rows: usize,
    /// The longest prefill chunk, in tokens (`max_batch_tokens`).
    pub chunk_tokens: usize,
    /// Rows one request may carry.
    pub request_rows: usize,
}

impl VisionRowLimits {
    /// The limits a server applies: the encoder's window, its longest prefill
    /// chunk, and the operator's per-request rows (`0` = unset) capped by the
    /// context, since every row is a prompt token.
    pub fn resolve(
        window_rows: usize,
        chunk_tokens: usize,
        request_rows: usize,
        max_seq_len: usize,
    ) -> Self {
        let request_rows = match request_rows {
            0 => max_seq_len,
            n => n.min(max_seq_len),
        };
        Self {
            window_rows,
            chunk_tokens,
            request_rows,
        }
    }

    /// Why a request whose items carry `item_rows` (prompt order) cannot be
    /// served, or `Ok`. The message names counts and limits only, never
    /// request content.
    pub fn check(&self, item_rows: &[usize]) -> Result<(), String> {
        let n = item_rows.len();
        let items = if n == 1 {
            "1 image".to_string()
        } else {
            format!("{n} images")
        };
        if n > MAX_VISION_ITEMS {
            return Err(format!(
                "vision input too large: the request has {items}, more than the \
                 {MAX_VISION_ITEMS} one request may carry"
            ));
        }
        let total: usize = item_rows.iter().sum();
        if total > self.request_rows {
            return Err(format!(
                "vision input too large: {items} need {total} vision tokens, more than \
                 the {} this server accepts per request; send fewer or smaller images",
                self.request_rows
            ));
        }
        let worst = worst_chunk_rows(item_rows, self.chunk_tokens);
        if worst > self.window_rows {
            let largest = item_rows.iter().max().copied().unwrap_or(0);
            return Err(format!(
                "vision input too large: {items} need {total} vision tokens and a prefill \
                 chunk of up to {} tokens may need {worst} of them encoded at once (largest \
                 item {largest}), more than the encoder's {}-row window; send fewer or \
                 smaller images",
                self.chunk_tokens, self.window_rows
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "vision_limits_tests.rs"]
mod tests;
