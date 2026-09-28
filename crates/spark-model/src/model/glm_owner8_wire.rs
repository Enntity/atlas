// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed E8 words only; decoded facts never grant producer or dispatch authority.
use super::glm_owner_wire::{Bounds, Mode, OwnerRecord};
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use anyhow::{Context, Result, ensure};

pub(crate) const EP_GLM_OWNER8_VERIFY: u32 = 0xffff_ffe8;
pub(crate) const PAYLOAD_WORDS: usize = 92;
pub(crate) const VERDICT_WORDS: usize = 10;
const VERSION: u32 = 1;
const RECORD_WORDS: usize = 11;

fn count(shape: GlmOwnerBatchShape) -> Result<usize> {
    let count = shape.owners();
    ensure!(
        (5..=8).contains(&count),
        "E8 requires five through eight owners"
    );
    Ok(count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Packet {
    pub shape: GlmOwnerBatchShape,
    pub mode: Mode,
    pub owners: [Option<OwnerRecord>; 8],
}

impl Packet {
    pub(crate) fn encode(&self, bounds: Bounds) -> Result<[u32; PAYLOAD_WORDS]> {
        let count = count(self.shape)?;
        bounds.validate(count).context("E8 bounds")?;
        ensure!(
            self.owners[count..].iter().all(Option::is_none),
            "E8 unused owner record is present"
        );
        let mut words = [0; PAYLOAD_WORDS];
        words[..4].copy_from_slice(&[
            VERSION,
            count as u32,
            self.shape.rows() as u32,
            self.mode.word(),
        ]);
        let mut previous = None;
        for (ordinal, owner) in self.owners[..count].iter().enumerate() {
            let owner = owner.context("E8 live owner record missing")?;
            owner.validate(bounds, previous).context("E8 owner")?;
            previous = Some(owner.slot);
            let start = 4 + ordinal * RECORD_WORDS;
            words[start..start + 6].copy_from_slice(&[
                owner.slot,
                owner.generation as u32,
                (owner.generation >> 32) as u32,
                owner.attempt as u32,
                (owner.attempt >> 32) as u32,
                owner.base,
            ]);
            words[start + 6..start + RECORD_WORDS].copy_from_slice(&owner.tokens);
        }
        Ok(words)
    }

    pub(crate) fn decode(words: &[u32; PAYLOAD_WORDS], bounds: Bounds) -> Result<Self> {
        ensure!(words[0] == VERSION, "E8 payload version mismatch");
        // Bound externally supplied count before any fixed-array slice.
        ensure!(
            (5..=8).contains(&words[1]),
            "E8 requires five through eight owners"
        );
        let shape = GlmOwnerBatchShape::new(words[1] as usize)?;
        bounds.validate(shape.owners()).context("E8 bounds")?;
        ensure!(
            words[2] == shape.rows() as u32,
            "E8 payload row count mismatch"
        );
        let mode = Mode::decode(words[3]).context("E8 mode")?;
        let mut owners = [None; 8];
        for (ordinal, owner) in owners[..shape.owners()].iter_mut().enumerate() {
            let start = 4 + ordinal * RECORD_WORDS;
            let mut tokens = [0; 5];
            tokens.copy_from_slice(&words[start + 6..start + RECORD_WORDS]);
            *owner = Some(OwnerRecord {
                slot: words[start],
                generation: u64::from(words[start + 1]) | (u64::from(words[start + 2]) << 32),
                attempt: u64::from(words[start + 3]) | (u64::from(words[start + 4]) << 32),
                base: words[start + 5],
                tokens,
            });
        }
        let packet = Self {
            shape,
            mode,
            owners,
        };
        ensure!(
            packet.encode(bounds)? == *words,
            "E8 noncanonical payload padding"
        );
        Ok(packet)
    }
}

pub(crate) fn encode_verdict(
    shape: GlmOwnerBatchShape,
    accepted: [usize; 8],
) -> Result<[u32; VERDICT_WORDS]> {
    let count = count(shape)?;
    ensure!(
        accepted[..count].iter().all(|&value| value <= 4)
            && accepted[count..].iter().all(|&value| value == 0),
        "E8 invalid accepted count or unused verdict padding"
    );
    let mut words = [0; VERDICT_WORDS];
    words[..2].copy_from_slice(&[VERSION, count as u32]);
    for (ordinal, value) in accepted.into_iter().enumerate() {
        words[ordinal + 2] = value as u32;
    }
    Ok(words)
}

pub(crate) fn decode_verdict(
    words: &[u32; VERDICT_WORDS],
    expected_shape: GlmOwnerBatchShape,
) -> Result<[usize; 8]> {
    let accepted = std::array::from_fn(|ordinal| words[ordinal + 2] as usize);
    ensure!(
        encode_verdict(expected_shape, accepted)? == *words,
        "E8 verdict version/owner count mismatch"
    );
    Ok(accepted)
}

#[cfg(test)]
#[path = "glm_owner8_wire_tests.rs"]
mod tests;
