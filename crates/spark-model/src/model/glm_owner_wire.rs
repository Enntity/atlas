// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed E7 words only; decoded facts are not producer or dispatch authority.
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use anyhow::{Context, Result, bail, ensure};

pub(crate) const EP_GLM_OWNER_VERIFY: u32 = 0xffff_ffe7;
pub(crate) const PAYLOAD_WORDS: usize = 48;
pub(crate) const VERDICT_WORDS: usize = 6;
const VERSION: u32 = 1;
const RECORD_WORDS: usize = 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    OwnersJoint,
}

impl Mode {
    fn word(self) -> u32 {
        match self {
            Self::OwnersJoint => 1,
        }
    }

    fn decode(word: u32) -> Result<Self> {
        match word {
            1 => Ok(Self::OwnersJoint),
            _ => bail!("E7 unknown owner compute mode"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub capacity: usize,
    pub vocab_size: usize,
    pub context_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnerRecord {
    pub slot: u32,
    pub generation: u64,
    pub attempt: u64,
    pub base: u32,
    pub tokens: [u32; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Packet {
    pub shape: GlmOwnerBatchShape,
    pub mode: Mode,
    /// Live records occupy exactly the shape's prefix; C3 has no fourth owner.
    pub owners: [Option<OwnerRecord>; 4],
}

impl Packet {
    pub(crate) fn encode(&self, bounds: Bounds) -> Result<[u32; PAYLOAD_WORDS]> {
        let count = self.shape.owners();
        ensure!(
            (2..=4).contains(&bounds.capacity)
                && count <= bounds.capacity
                && bounds.vocab_size > 0
                && bounds.vocab_size <= u32::MAX as usize
                && (5..=2048).contains(&bounds.context_tokens),
            "E7 invalid explicit owner/vocabulary/context bounds"
        );
        ensure!(
            self.owners[count..].iter().all(Option::is_none),
            "E7 unused owner record is present"
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
            let owner = owner.context("E7 live owner record missing")?;
            ensure!(
                (owner.slot as usize) < bounds.capacity
                    && previous.is_none_or(|slot| slot < owner.slot)
                    && owner.generation != 0
                    && owner.attempt != 0,
                "E7 invalid owner ordering/identity"
            );
            let end = owner.base.checked_add(5).context("E7 owner end overflow")?;
            ensure!(
                (end as usize) <= bounds.context_tokens
                    && owner
                        .tokens
                        .iter()
                        .all(|&token| (token as usize) < bounds.vocab_size),
                "E7 owner position/token outside explicit bounds"
            );
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
        ensure!(words[0] == VERSION, "E7 payload version mismatch");
        let shape = GlmOwnerBatchShape::new(words[1] as usize)?;
        ensure!(
            words[2] == shape.rows() as u32,
            "E7 payload row count mismatch"
        );
        let mode = Mode::decode(words[3])?;
        let mut owners = [None; 4];
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
        // One canonical validator for both directions; comparison also rejects
        // every nonzero word in C3's unused fourth record.
        ensure!(
            packet.encode(bounds)? == *words,
            "E7 noncanonical payload padding"
        );
        Ok(packet)
    }
}

pub(crate) fn encode_verdict(
    shape: GlmOwnerBatchShape,
    accepted: [usize; 4],
) -> Result<[u32; VERDICT_WORDS]> {
    let count = shape.owners();
    ensure!(
        accepted[..count].iter().all(|&value| value <= 4)
            && accepted[count..].iter().all(|&value| value == 0),
        "E7 invalid accepted count or unused verdict padding"
    );
    Ok([
        VERSION,
        count as u32,
        accepted[0] as u32,
        accepted[1] as u32,
        accepted[2] as u32,
        accepted[3] as u32,
    ])
}

pub(crate) fn decode_verdict(
    words: &[u32; VERDICT_WORDS],
    expected_shape: GlmOwnerBatchShape,
) -> Result<[usize; 4]> {
    let accepted = std::array::from_fn(|ordinal| words[ordinal + 2] as usize);
    ensure!(
        encode_verdict(expected_shape, accepted)? == *words,
        "E7 verdict version/owner count mismatch"
    );
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDS: Bounds = Bounds {
        capacity: 4,
        vocab_size: 32,
        context_tokens: 2048,
    };

    fn fixture(count: usize) -> (Packet, [u32; PAYLOAD_WORDS]) {
        let shape = GlmOwnerBatchShape::new(count).unwrap();
        let slots = if count == 3 {
            [0, 2, 3, 0]
        } else {
            [0, 1, 2, 3]
        };
        let mut packet = Packet {
            shape,
            mode: Mode::OwnersJoint,
            owners: [None; 4],
        };
        let mut words = [0; PAYLOAD_WORDS];
        words[..4].copy_from_slice(&[1, count as u32, (count * 5) as u32, 1]);
        for ordinal in 0..count {
            let low = ordinal as u32 + 1;
            packet.owners[ordinal] = Some(OwnerRecord {
                slot: slots[ordinal],
                generation: (7u64 << 32) | u64::from(low),
                attempt: (9u64 << 32) | u64::from(low + 10),
                base: 2043 - ordinal as u32,
                tokens: [low, low + 1, low + 2, low + 3, low + 4],
            });
            let start = 4 + ordinal * 11;
            words[start..start + 11].copy_from_slice(&[
                slots[ordinal],
                low,
                7,
                low + 10,
                9,
                2043 - ordinal as u32,
                low,
                low + 1,
                low + 2,
                low + 3,
                low + 4,
            ]);
        }
        (packet, words)
    }

    #[test]
    fn canonical_three_noncontiguous_roundtrip() {
        let (packet, words) = fixture(3);
        assert_eq!(EP_GLM_OWNER_VERIFY, 0xffff_ffe7);
        assert_eq!(packet.encode(BOUNDS).unwrap(), words);
        assert_eq!(Packet::decode(&words, BOUNDS).unwrap(), packet);
        assert_eq!(&words[37..], &[0; 11]);
    }

    #[test]
    fn canonical_four_roundtrip() {
        let (packet, words) = fixture(4);
        assert_eq!(packet.encode(BOUNDS).unwrap(), words);
        assert_eq!(Packet::decode(&words, BOUNDS).unwrap(), packet);
        let mut contiguous = fixture(3).0;
        for (slot, record) in contiguous.owners[..3].iter_mut().enumerate() {
            record.as_mut().unwrap().slot = slot as u32;
        }
        let bounds = Bounds {
            capacity: 3,
            ..BOUNDS
        };
        let words = contiguous.encode(bounds).unwrap();
        assert_eq!(Packet::decode(&words, bounds).unwrap(), contiguous);
    }

    #[test]
    fn rejects_headers_last_record_and_padding() {
        for count in [3, 4] {
            let (_, original) = fixture(count);
            for (index, bad) in [(0, 0), (1, 2), (1, 5), (2, 10), (3, 0), (3, 2)] {
                let mut words = original;
                words[index] = bad;
                assert!(Packet::decode(&words, BOUNDS).is_err(), "header {index}");
            }
            let last = 4 + (count - 1) * 11;
            for (offset, bad) in [(0, 4), (0, 0), (5, 2044), (5, u32::MAX), (10, 32)] {
                let mut words = original;
                words[last + offset] = bad;
                assert!(
                    Packet::decode(&words, BOUNDS).is_err(),
                    "last field {offset}"
                );
            }
            for offset in [1, 3] {
                let mut words = original;
                words[last + offset..last + offset + 2].fill(0);
                assert!(Packet::decode(&words, BOUNDS).is_err(), "zero identity");
            }
        }
        for index in 37..48 {
            let (_, mut words) = fixture(3);
            words[index] = 1;
            assert!(Packet::decode(&words, BOUNDS).is_err(), "padding {index}");
        }
    }

    #[test]
    fn rejects_noncanonical_records_and_explicit_bounds() {
        let (packet, words) = fixture(4);
        for bounds in [
            Bounds {
                capacity: 1,
                ..BOUNDS
            },
            Bounds {
                capacity: 2,
                ..BOUNDS
            },
            Bounds {
                capacity: 3,
                ..BOUNDS
            },
            Bounds {
                capacity: 5,
                ..BOUNDS
            },
            Bounds {
                vocab_size: 0,
                ..BOUNDS
            },
            Bounds {
                vocab_size: 7,
                ..BOUNDS
            },
            Bounds {
                context_tokens: 2047,
                ..BOUNDS
            },
            Bounds {
                context_tokens: 2049,
                ..BOUNDS
            },
            Bounds {
                context_tokens: 0,
                ..BOUNDS
            },
        ] {
            assert!(packet.encode(bounds).is_err());
            assert!(Packet::decode(&words, bounds).is_err());
        }
        let mut missing = packet;
        missing.owners[2] = None;
        assert!(missing.encode(BOUNDS).is_err());
        let mut extra = fixture(3).0;
        extra.owners[3] = packet.owners[3];
        assert!(extra.encode(BOUNDS).is_err());
        let mut swapped = packet;
        swapped.owners.swap(2, 3);
        assert!(swapped.encode(BOUNDS).is_err());
        let mut duplicate = packet;
        duplicate.owners[3] = duplicate.owners[2];
        assert!(duplicate.encode(BOUNDS).is_err());
    }

    #[test]
    fn verdict_exact_roundtrips_and_rejects_last_count() {
        for count in [3, 4] {
            let shape = GlmOwnerBatchShape::new(count).unwrap();
            let accepted = if count == 3 {
                [0, 2, 4, 0]
            } else {
                [0, 2, 4, 1]
            };
            let words = [1, count as u32, 0, 2, 4, accepted[3] as u32];
            assert_eq!(encode_verdict(shape, accepted).unwrap(), words);
            assert_eq!(decode_verdict(&words, shape).unwrap(), accepted);
            let mut invalid = accepted;
            invalid[count - 1] = 5;
            assert!(encode_verdict(shape, invalid).is_err());
            invalid[count - 1] = usize::MAX;
            assert!(encode_verdict(shape, invalid).is_err());
            for (index, value) in [(0, 0), (1, 2), (1, 7), (count + 1, 5), (5, u32::MAX)] {
                let mut bad = words;
                bad[index] = value;
                assert!(decode_verdict(&bad, shape).is_err());
            }
        }
        let three = GlmOwnerBatchShape::new(3).unwrap();
        assert!(encode_verdict(three, [0, 0, 0, 1]).is_err());
        assert!(decode_verdict(&[1, 3, 0, 0, 0, 1], three).is_err());
    }
}
