// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Borrowed, alignment-independent LE words, constructed only from a validated
/// bulk call. Reading never transmutes bytes into an aligned/native u32 slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeWords<'a> {
    bytes: &'a [u8],
}
impl<'a> LeWords<'a> {
    pub fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub fn len(self) -> usize {
        self.bytes.len() / 4
    }
    pub fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        self.bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

#[derive(Clone, Copy)]
enum HeaderKind {
    Decode { slot: u32, token: u32 },
    Prefill { slot: u32, start: u32, count: u32 },
    Batch,
    Replace { slot: u32 },
    Shutdown,
}

/// Header acceptance fixes every following bulk byte count before any bulk
/// access. No live request state, lifetime generation, or peer vote is implied.
pub struct ValidatedHeader {
    kind: HeaderKind,
    bulk_bytes: [usize; 2],
    bulk_count: usize,
    wire_bytes: usize,
    limits: LegacyWireLimits,
}

impl ValidatedHeader {
    pub fn bulk_byte_lengths(&self) -> &[usize] {
        &self.bulk_bytes[..self.bulk_count]
    }
    pub fn wire_bytes(&self) -> usize {
        self.wire_bytes
    }
    /// Caller must receive only the exact bounded calls prescribed by this
    /// header. This parser itself allocates no payload buffer and borrows input.
    pub fn parse_payloads<'a>(&self, calls: &[&'a [u8]]) -> Result<LegacyCommand<LeWords<'a>>> {
        ensure!(
            calls.len() == self.bulk_count,
            "legacy codec bulk call count mismatch"
        );
        let mut words = [LeWords { bytes: &[] }; 2];
        for (index, &bytes) in calls.iter().enumerate() {
            ensure!(
                bytes.len() == self.bulk_bytes[index],
                "legacy codec bulk byte length mismatch"
            );
            words[index] = LeWords { bytes };
        }
        Ok(match self.kind {
            HeaderKind::Decode { slot, token } => LegacyCommand::Decode { slot, token },
            HeaderKind::Prefill { slot, start, count } => {
                for token in words[0].iter() {
                    self.limits.token(token)?;
                }
                LegacyCommand::Prefill {
                    slot,
                    chunk_start: start,
                    chunk_len: count,
                    prompt: words[0],
                }
            }
            HeaderKind::Batch => {
                validate_batch(
                    words[0].iter(),
                    words[1].iter(),
                    words[0].len(),
                    self.limits,
                )?;
                LegacyCommand::DecodeBatch {
                    slots: words[0],
                    tokens: words[1],
                }
            }
            HeaderKind::Replace { slot } => LegacyCommand::ReplaceSlot { slot },
            HeaderKind::Shutdown => LegacyCommand::Shutdown,
        })
    }
}

pub(super) fn validate_batch(
    slots: impl Iterator<Item = u32>,
    tokens: impl Iterator<Item = u32>,
    count: usize,
    limits: LegacyWireLimits,
) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    // Metadata only, after the staged shape limit. No quadratic duplicate scan.
    seen.try_reserve(count)?;
    for slot in slots {
        limits.slot(slot, LegacyDialect::V2)?;
        ensure!(seen.insert(slot), "legacy codec duplicate batch slot");
    }
    for token in tokens {
        limits.token(token)?;
    }
    Ok(())
}

/// Parse ONLY fixed scalar calls. Every supplied call must contain exactly four
/// LE bytes; flattening scalar calls or appending bulk data is a framing error.
pub fn parse_header(
    dialect: LegacyDialect,
    scalar_calls: &[&[u8]],
    limits: LegacyWireLimits,
) -> Result<ValidatedHeader> {
    ensure!(
        scalar_calls.len() <= 5,
        "legacy codec too many scalar header calls"
    );
    let mut words = [0u32; 5];
    for (index, bytes) in scalar_calls.iter().enumerate() {
        ensure!(
            bytes.len() == 4,
            "legacy codec scalar call must contain exactly four bytes"
        );
        words[index] = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    parse_words_header(dialect, &words[..scalar_calls.len()], limits)
}

pub(super) fn parse_words_header(
    dialect: LegacyDialect,
    words: &[u32],
    limits: LegacyWireLimits,
) -> Result<ValidatedHeader> {
    limits.validate()?;
    let base = if dialect == LegacyDialect::V2 { 2 } else { 1 };
    ensure!(words.len() >= base, "legacy codec truncated preamble");
    let slot = if dialect == LegacyDialect::V2 {
        words[0]
    } else {
        0
    };
    let opcode = words[base - 1];
    let mut bulk_words = [0usize; 2];
    let (kind, extra, bulk_count) = match opcode {
        SHUTDOWN => (HeaderKind::Shutdown, 0, 0), // existing worker ignores its preamble slot
        BATCH => {
            ensure!(
                dialect == LegacyDialect::V2 && slot == 0,
                "legacy codec E0 requires canonical V2 sentinel0"
            );
            ensure!(
                words.len() == base + 1,
                "legacy codec batch header length mismatch"
            );
            let n = words[base];
            ensure!(
                n > 0 && n <= limits.max_decode_rows,
                "legacy codec batch row count outside bounds"
            );
            let n = usize::try_from(n)?;
            bulk_words = [n, n];
            (HeaderKind::Batch, 1, 2)
        }
        PREFILL => {
            limits.slot(slot, dialect)?;
            ensure!(
                words.len() == base + 3,
                "legacy codec prefill header length mismatch"
            );
            let (count, start, full) = (words[base], words[base + 1], words[base + 2]);
            ensure!(
                count > 0
                    && count <= limits.max_chunk_tokens
                    && full > 0
                    && full <= limits.max_prompt_tokens,
                "legacy codec prefill count/full prompt outside bounds"
            );
            let end = start
                .checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("legacy codec prefill end overflow"))?;
            ensure!(end <= full, "legacy codec chunk exceeds full prompt");
            bulk_words[0] = usize::try_from(full)?;
            (HeaderKind::Prefill { slot, start, count }, 3, 1)
        }
        REPLACE => {
            limits.slot(slot, dialect)?;
            (HeaderKind::Replace { slot }, 0, 0)
        }
        token => {
            limits.slot(slot, dialect)?;
            limits.token(token)?;
            (HeaderKind::Decode { slot, token }, 0, 0)
        }
    };
    ensure!(
        words.len() == base + extra,
        "legacy codec scalar header length mismatch"
    );
    let mut bulk_bytes = [0usize; 2];
    let mut wire_bytes = checked_bytes(words.len())?;
    for index in 0..bulk_count {
        let bytes = checked_bytes(bulk_words[index])?;
        ensure!(
            bytes <= limits.staging_bytes,
            "legacy codec bulk exceeds staging capacity"
        );
        wire_bytes = wire_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("legacy codec command byte overflow"))?;
        bulk_bytes[index] = bytes;
    }
    ensure!(
        wire_bytes <= limits.max_control_bytes,
        "legacy codec command exceeds control budget"
    );
    Ok(ValidatedHeader {
        kind,
        bulk_bytes,
        bulk_count,
        wire_bytes,
        limits,
    })
}
