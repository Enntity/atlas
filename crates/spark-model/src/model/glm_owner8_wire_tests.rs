// SPDX-License-Identifier: AGPL-3.0-only
//! Exact word-layout tests, not communication or producer authority.
use super::*;
use crate::model::glm_owner_wire as e7;

const BOUNDS: Bounds = Bounds {
    capacity: 8,
    vocab_size: 64,
    context_tokens: 2048,
};

fn fixture(count: usize) -> (Packet, [u32; PAYLOAD_WORDS]) {
    let mut packet = Packet {
        shape: GlmOwnerBatchShape::new(count).unwrap(),
        mode: Mode::OwnersJoint,
        owners: [None; 8],
    };
    let mut words = [0; PAYLOAD_WORDS];
    words[..4].copy_from_slice(&[1, count as u32, count as u32 * 5, 1]);
    for ordinal in 0..count {
        // The final physical slot is always7, including noncontiguous C5..7.
        let slot = if ordinal + 1 == count {
            7
        } else {
            ordinal as u32
        };
        let low = ordinal as u32 + 1;
        let tokens = [low, low + 1, low + 2, low + 3, low + 4];
        packet.owners[ordinal] = Some(OwnerRecord {
            slot,
            generation: (7u64 << 32) | u64::from(low),
            attempt: (9u64 << 32) | u64::from(low + 10),
            base: 2043 - ordinal as u32,
            tokens,
        });
        let start = 4 + ordinal * 11;
        words[start..start + 6].copy_from_slice(&[
            slot,
            low,
            7,
            low + 10,
            9,
            2043 - ordinal as u32,
        ]);
        words[start + 6..start + 11].copy_from_slice(&tokens);
    }
    (packet, words)
}

#[test]
fn five_through_eight_exact_payload_roundtrips() {
    assert_eq!(EP_GLM_OWNER8_VERIFY, 0xffff_ffe8);
    assert_eq!((PAYLOAD_WORDS, VERDICT_WORDS), (92, 10));
    for count in 5..=8 {
        let (packet, words) = fixture(count);
        assert_eq!(packet.encode(BOUNDS).unwrap(), words);
        assert_eq!(Packet::decode(&words, BOUNDS).unwrap(), packet);
        assert!(words[4 + count * 11..].iter().all(|&word| word == 0));
        let mut contiguous = packet;
        for (slot, record) in contiguous.owners[..count].iter_mut().enumerate() {
            record.as_mut().unwrap().slot = slot as u32;
        }
        let bounds = Bounds {
            capacity: count,
            ..BOUNDS
        };
        let encoded = contiguous.encode(bounds).unwrap();
        assert_eq!(Packet::decode(&encoded, bounds).unwrap(), contiguous);
    }
}

#[test]
fn headers_last_owner_and_padding_are_strict() {
    for count in 5..=8 {
        let (packet, original) = fixture(count);
        for (index, bad) in [
            (0, 0),
            (1, 3),
            (1, 4),
            (1, 9),
            (1, u32::MAX),
            (2, 10),
            (3, 0),
            (3, 2),
        ] {
            let mut words = original;
            words[index] = bad;
            assert!(Packet::decode(&words, BOUNDS).is_err(), "header{index}");
        }
        let last = 4 + (count - 1) * 11;
        for (offset, bad) in [(0, 8), (0, 0), (5, 2044), (5, u32::MAX), (10, 64)] {
            let mut words = original;
            words[last + offset] = bad;
            assert!(
                Packet::decode(&words, BOUNDS).is_err(),
                "owner field{offset}"
            );
        }
        for offset in [1, 3] {
            let mut words = original;
            words[last + offset..last + offset + 2].fill(0);
            assert!(Packet::decode(&words, BOUNDS).is_err());
        }
        for index in 4 + count * 11..PAYLOAD_WORDS {
            let mut words = original;
            words[index] = 1;
            assert!(Packet::decode(&words, BOUNDS).is_err(), "padding{index}");
        }
        let mut missing = packet;
        missing.owners[count - 1] = None;
        assert!(missing.encode(BOUNDS).is_err());
        let mut swapped = packet;
        swapped.owners.swap(0, count - 1);
        assert!(swapped.encode(BOUNDS).is_err());
        if count < 8 {
            let mut extra = packet;
            extra.owners[count] = packet.owners[0];
            assert!(extra.encode(BOUNDS).is_err());
        }
    }
    for count in [3, 4] {
        let mut packet = fixture(5).0;
        packet.shape = GlmOwnerBatchShape::new(count).unwrap();
        assert!(packet.encode(BOUNDS).is_err());
    }
}

#[test]
fn explicit_registry_vocabulary_and_context_bounds() {
    let (packet, words) = fixture(8);
    for bounds in [
        Bounds {
            capacity: 1,
            ..BOUNDS
        },
        Bounds {
            capacity: 7,
            ..BOUNDS
        },
        Bounds {
            capacity: 9,
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
            context_tokens: 4,
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
    ] {
        assert!(packet.encode(bounds).is_err());
        assert!(Packet::decode(&words, bounds).is_err());
    }
}

#[test]
fn verdict_exact_roundtrips_and_final_owner_refusal() {
    for count in 5..=8 {
        let shape = GlmOwnerBatchShape::new(count).unwrap();
        let accepted = std::array::from_fn(|i| if i < count { i % 5 } else { 0 });
        let mut words = [0; VERDICT_WORDS];
        words[..2].copy_from_slice(&[1, count as u32]);
        for i in 0..8 {
            words[i + 2] = accepted[i] as u32;
        }
        assert_eq!(encode_verdict(shape, accepted).unwrap(), words);
        assert_eq!(decode_verdict(&words, shape).unwrap(), accepted);
        for index in [0, 1, count + 1] {
            let mut bad = words;
            bad[index] = 99;
            assert!(decode_verdict(&bad, shape).is_err());
        }
        let mut bad = accepted;
        bad[count - 1] = 5;
        assert!(encode_verdict(shape, bad).is_err());
        for i in count..8 {
            let mut bad = words;
            bad[i + 2] = 1;
            assert!(decode_verdict(&bad, shape).is_err());
        }
    }
    for count in [3, 4] {
        assert!(encode_verdict(GlmOwnerBatchShape::new(count).unwrap(), [0; 8]).is_err());
    }
}

#[test]
fn e7_keeps_layout_with_high_physical_slots_in_eight_owner_registry() {
    for slots in [&[0u32, 4, 7][..], &[1u32, 4, 6, 7][..]] {
        let count = slots.len();
        let source = fixture(5).0;
        let mut packet = e7::Packet {
            shape: GlmOwnerBatchShape::new(count).unwrap(),
            mode: Mode::OwnersJoint,
            owners: [None; 4],
        };
        for (i, &slot) in slots.iter().enumerate() {
            let mut record = source.owners[i].unwrap();
            record.slot = slot;
            packet.owners[i] = Some(record);
        }
        let words = packet.encode(BOUNDS).unwrap();
        assert_eq!((words.len(), e7::VERDICT_WORDS), (48, 6));
        assert_eq!(e7::Packet::decode(&words, BOUNDS).unwrap(), packet);
        assert_eq!(words[4 + (count - 1) * 11], 7);
        for wider in 5..=8 {
            let mut bad = words;
            bad[1] = wider;
            bad[2] = wider * 5;
            assert!(e7::Packet::decode(&bad, BOUNDS).is_err());
        }
    }
}
