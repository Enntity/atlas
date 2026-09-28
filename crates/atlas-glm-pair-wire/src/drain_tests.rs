// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn bytes() -> Vec<u8> {
    let mut bytes = 52u32.to_be_bytes().to_vec();
    bytes.extend_from_slice(b"GLP2");
    bytes.extend_from_slice(&[0, 1, 0x16, 0, 0, 0, 0, 0]);
    bytes.extend_from_slice(&[7; 32]);
    bytes.extend_from_slice(&1u64.to_be_bytes());
    bytes
}

#[test]
fn actual_rank0_drain_codec_has_exact_direction_length_and_fields() {
    let bytes = bytes();
    assert_eq!(bytes.len(), 56);
    assert_eq!(
        control_frame_len([0, 0, 0, 52], Direction::ControllerToGuard).unwrap(),
        56
    );
    let value = Frame::decode(&bytes, Direction::ControllerToGuard).unwrap();
    assert_eq!(
        value,
        Frame {
            rank: 0,
            body: Body::DrainRequest(DrainRequest {
                pair_digest: [7; 32],
                epoch: 1
            })
        }
    );
    assert_eq!(value.encode().unwrap().as_slice(), bytes);
    for direction in [
        Direction::StartupFile,
        Direction::ChildToGuard,
        Direction::GuardToChild,
        Direction::GuardToController,
    ] {
        assert!(Frame::decode(&bytes, direction).is_err());
    }
    for index in [0, 10, 11, 12, 55] {
        let mut bad = bytes.clone();
        bad[index] ^= 1;
        assert!(Frame::decode(&bad, Direction::ControllerToGuard).is_err());
    }
    for end in 0..bytes.len() {
        assert!(Frame::decode(&bytes[..end], Direction::ControllerToGuard).is_err());
    }
    let mut bad = bytes.clone();
    bad[16..48].fill(0);
    assert!(Frame::decode(&bad, Direction::ControllerToGuard).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(Frame::decode(&trailing, Direction::ControllerToGuard).is_err());
    assert!(
        Frame {
            rank: 1,
            body: value.body
        }
        .encode()
        .is_err()
    );
}

#[test]
fn actual_drain_receipt_binds_current_manifest_only() {
    let manifest = Manifest {
        pair_session: [17; 32],
        policy_digest: policy_digest(&policy()).unwrap(),
        ranks: [rank(0), rank(1)],
    };
    let mut drain = DrainRequest {
        pair_digest: manifest_digest(&manifest).unwrap(),
        epoch: DRAIN_EPOCH,
    };
    drain.validate(&manifest).unwrap();
    drain.pair_digest[0] ^= 1;
    assert!(drain.validate(&manifest).is_err());
    drain.pair_digest = manifest_digest(&manifest).unwrap();
    drain.epoch = 2;
    assert!(drain.validate(&manifest).is_err());
}
