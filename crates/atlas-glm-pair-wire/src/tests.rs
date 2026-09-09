// SPDX-License-Identifier: AGPL-3.0-only

use crate::*;

#[test]
fn actual_child_hello_codec_matches_fixed_independent_bytes() {
    let hello = ChildHello {
        boot_id: [0x11; 16],
        pid_namespace_device: 2,
        pid_namespace_inode: 3,
        guard_pid: 1,
        child_pid: 2,
        guard_start_ticks: 4,
        child_start_ticks: 5,
        server_challenge: [0x22; 32],
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&100u32.to_be_bytes());
    bytes.extend_from_slice(b"GLP2");
    bytes.extend_from_slice(&[0, 1, 0x12, 1, 0, 0, 0, 0]);
    bytes.extend_from_slice(&[0x11; 16]);
    for v in [2u64, 3] {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    for v in [1u32, 2] {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    for v in [4u64, 5] {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    bytes.extend_from_slice(&[0x22; 32]);
    assert_eq!(bytes.len(), 104);
    let frame = Frame::decode(&bytes, Direction::ChildToGuard).expect("valid postexec HELLO");
    assert_eq!(
        frame,
        Frame {
            rank: 1,
            body: Body::ChildHello(hello)
        }
    );
    assert_eq!(frame.encode().unwrap().as_slice(), bytes);
}

fn policy() -> Policy {
    Policy {
        startup: 30000,
        lease: 30000,
        challenge: 5000,
        frame: 3000,
        campaign: 60000,
        poll: 250,
        reap: 10000,
        child_handshake: 30000,
        quiescent_wait: 10000,
        exit: 5000,
    }
}
fn rank(n: u8) -> RankRecord {
    RankRecord {
        container_id: [1 + n; 32],
        image_digest: [3; 32],
        recipe_digest: [4 + n; 32],
        server_elf_digest: [6; 32],
        guard_elf_digest: [7; 32],
        local_control_session: [8 + n; 32],
        guard_instance: [10 + n; 32],
        original_startup_challenge: [12 + n; 32],
        process: ProcessIdentity {
            boot_id: [14 + n; 16],
            pid_namespace_device: 1,
            pid_namespace_inode: 100 + u64::from(n),
            guard_pid: 1,
            child_pid: 2,
            guard_start_ticks: 123,
            child_start_ticks: 124,
            child_instance: [16 + n; 32],
        },
    }
}
fn manifest() -> Manifest {
    Manifest {
        pair_session: [18; 32],
        policy_digest: policy_digest(&policy()).unwrap(),
        ranks: [rank(0), rank(1)],
    }
}
fn startup(rank: u8) -> StartupRecord {
    let m = manifest();
    let r = m.ranks[usize::from(rank)];
    StartupRecord {
        pair_session: m.pair_session,
        container_id: r.container_id,
        image_digest: r.image_digest,
        recipe_digest: r.recipe_digest,
        server_elf_digest: r.server_elf_digest,
        guard_elf_digest: r.guard_elf_digest,
        policy: policy(),
    }
}
fn hello(rank: u8) -> ChildHello {
    let p = manifest().ranks[usize::from(rank)].process;
    ChildHello {
        boot_id: p.boot_id,
        pid_namespace_device: p.pid_namespace_device,
        pid_namespace_inode: p.pid_namespace_inode,
        guard_pid: p.guard_pid,
        child_pid: p.child_pid,
        guard_start_ticks: p.guard_start_ticks,
        child_start_ticks: p.child_start_ticks,
        server_challenge: [20 + rank; 32],
    }
}
fn ticket(rank: u8) -> ChildTicket {
    ChildTicket {
        manifest: manifest(),
        echoed_server_challenge: hello(rank).server_challenge,
        guard_ticket_challenge: [22 + rank; 32],
    }
}
fn receipt(rank: u8) -> Quiescent {
    Quiescent {
        pair_digest: manifest_digest(&manifest()).unwrap(),
        child_instance: manifest().ranks[usize::from(rank)].process.child_instance,
        epoch: 1,
        last_command: SHUTDOWN_COMMAND,
        receipt_nonce: [24 + rank; 32],
    }
}
fn release() -> PairRelease {
    let receipts = [
        QuiescentFrame {
            rank: 0,
            receipt: receipt(0),
        },
        QuiescentFrame {
            rank: 1,
            receipt: receipt(1),
        },
    ];
    PairRelease {
        pair_digest: manifest_digest(&manifest()).unwrap(),
        epoch: 1,
        receipt_digests: receipts.map(|r| quiescent_digest(&r).unwrap()),
        receipts,
    }
}
fn frames(rank: u8) -> [(Frame, Direction, usize, u8); 7] {
    [
        (
            Frame {
                rank,
                body: Body::Startup(startup(rank)),
            },
            Direction::StartupFile,
            288,
            0x01,
        ),
        (
            Frame {
                rank,
                body: Body::GatedReport(GatedReport {
                    pair_session: manifest().pair_session,
                    record: manifest().ranks[usize::from(rank)],
                }),
            },
            Direction::GuardToController,
            392,
            0x10,
        ),
        (
            Frame {
                rank,
                body: Body::PairedStart(manifest()),
            },
            Direction::ControllerToGuard,
            768,
            0x11,
        ),
        (
            Frame {
                rank,
                body: Body::ChildHello(hello(rank)),
            },
            Direction::ChildToGuard,
            104,
            0x12,
        ),
        (
            Frame {
                rank,
                body: Body::ChildTicket(ticket(rank)),
            },
            Direction::GuardToChild,
            832,
            0x13,
        ),
        (
            Frame {
                rank,
                body: Body::Quiescent(receipt(rank)),
            },
            Direction::ChildToGuard,
            128,
            0x14,
        ),
        (
            Frame {
                rank,
                body: Body::PairRelease(release()),
            },
            Direction::GuardToChild,
            376,
            0x15,
        ),
    ]
}

#[test]
fn all_seven_records_have_exact_headers_lengths_and_directional_roundtrips() {
    for rank in 0..2 {
        for (frame, direction, len, kind) in frames(rank) {
            let bytes = frame.encode().unwrap();
            let bytes = bytes.as_slice();
            assert_eq!(bytes.len(), len);
            assert_eq!(&bytes[..4], &((len - 4) as u32).to_be_bytes());
            assert_eq!(&bytes[4..], &frame.encode().unwrap().as_slice()[4..]);
            assert_eq!(
                &bytes[4..16],
                &[b'G', b'L', b'P', b'2', 0, 1, kind, rank, 0, 0, 0, 0]
            );
            assert_eq!(Frame::decode(bytes, direction).unwrap(), frame);
            if kind == 0x14 {
                assert_eq!(
                    Frame::decode(bytes, Direction::GuardToController).unwrap(),
                    frame
                );
            }
            if kind == 0x15 {
                assert_eq!(
                    Frame::decode(bytes, Direction::ControllerToGuard).unwrap(),
                    frame
                );
            }
            for bad in [
                Direction::StartupFile,
                Direction::ControllerToGuard,
                Direction::GuardToController,
                Direction::ChildToGuard,
                Direction::GuardToChild,
            ] {
                let allowed = bad == direction
                    || kind == 0x14 && bad == Direction::GuardToController
                    || kind == 0x15 && bad == Direction::ControllerToGuard;
                if !allowed {
                    assert!(Frame::decode(bytes, bad).is_err());
                }
            }
        }
    }
}

#[test]
fn all_truncations_trailing_bytes_and_bad_headers_are_refused() {
    for (frame, direction, _, _) in frames(0) {
        let bytes = frame.encode().unwrap();
        let bytes = bytes.as_slice();
        for cut in 0..bytes.len() {
            assert!(Frame::decode(&bytes[..cut], direction).is_err());
        }
        let mut extra = bytes.to_vec();
        extra.push(0);
        assert!(Frame::decode(&extra, direction).is_err());
        for offset in [0, 4, 8, 9, 10, 11, 12, 15] {
            let mut corrupt = bytes.to_vec();
            corrupt[offset] = 255;
            assert!(
                Frame::decode(&corrupt, direction).is_err(),
                "header byte {offset}"
            );
        }
    }
    assert!(Frame::decode(&[0; 4097], Direction::ChildToGuard).is_err());
    let mut q = Frame {
        rank: 0,
        body: Body::Quiescent(receipt(0)),
    }
    .encode()
    .unwrap()
    .as_slice()
    .to_vec();
    q[92] = 1; // Header16 + digest32 + instance32 + epoch8 + command4.
    assert!(Frame::decode(&q, Direction::ChildToGuard).is_err());
    let mut r = Frame {
        rank: 0,
        body: Body::PairRelease(release()),
    }
    .encode()
    .unwrap()
    .as_slice()
    .to_vec();
    r[56 + 12] = 1; // First embedded frame's reserved header.
    assert!(Frame::decode(&r, Direction::GuardToChild).is_err());
}

#[test]
fn legacy_control_prefix_is_only_bounded_not_reinterpreted() {
    for direction in [Direction::ControllerToGuard, Direction::GuardToController] {
        assert_eq!(
            control_frame_len(108u32.to_be_bytes(), direction).unwrap(),
            112
        );
        assert!(Frame::decode(&[0; 112], direction).is_err());
        for bad in [0, 100, 124 + 1, 284, 4096, u32::MAX] {
            assert!(control_frame_len(bad.to_be_bytes(), direction).is_err());
        }
    }
    assert_eq!(
        control_frame_len(764u32.to_be_bytes(), Direction::ControllerToGuard).unwrap(),
        768
    );
    assert_eq!(
        control_frame_len(388u32.to_be_bytes(), Direction::GuardToController).unwrap(),
        392
    );
    assert!(control_frame_len(108u32.to_be_bytes(), Direction::ChildToGuard).is_err());
}

#[test]
fn policy_all_ten_fields_are_explicit_and_digest_matches_external_vector() {
    let p = policy();
    p.validate().unwrap();
    // Python hashlib + struct.pack('>10Q', ...) independently generated this KAT.
    let got = policy_digest(&p)
        .unwrap()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect::<String>();
    assert_eq!(
        got,
        "cf52674c11512efddaf4b921786521656ee009362e159dc9eb1dacb7dead0fac"
    );
    for index in 0..10 {
        let mut q = p;
        let fields = [
            &mut q.startup,
            &mut q.lease,
            &mut q.challenge,
            &mut q.frame,
            &mut q.campaign,
            &mut q.poll,
            &mut q.reap,
            &mut q.child_handshake,
            &mut q.quiescent_wait,
            &mut q.exit,
        ];
        *fields.into_iter().nth(index).unwrap() = 0;
        assert!(q.validate().is_err());
    }
    for q in [
        Policy { exit: 30001, ..p },
        Policy { poll: 251, ..p },
        Policy { reap: 10001, ..p },
        Policy { frame: 5000, ..p },
        Policy {
            challenge: 30000,
            ..p
        },
        Policy {
            campaign: 86400001,
            ..p
        },
    ] {
        assert!(q.validate().is_err());
    }
    assert_ne!(recipe_digest(&[1; 80]).unwrap(), policy_digest(&p).unwrap());
    assert!(recipe_digest(&[]).is_err());
    assert!(recipe_digest(&[0; 65537]).is_err());
}

#[test]
fn ticket_binds_actual_supplied_local_records_and_independent_hello_challenge() {
    for rank in 0..2 {
        let m = manifest();
        let t = ticket(rank);
        let h = hello(rank);
        m.validate_local(rank, &startup(rank), &m.ranks[usize::from(rank)])
            .unwrap();
        t.validate_hello(rank, &h).unwrap();
        assert!(t.validate_hello(1 - rank, &h).is_err());
        assert!(
            t.validate_hello(
                rank,
                &ChildHello {
                    server_challenge: [99; 32],
                    ..h
                }
            )
            .is_err()
        );
        assert!(
            t.validate_hello(
                rank,
                &ChildHello {
                    child_start_ticks: h.child_start_ticks + 1,
                    ..h
                }
            )
            .is_err()
        );
        let mut changed = startup(rank);
        changed.container_id = [99; 32];
        assert!(
            m.validate_local(rank, &changed, &m.ranks[usize::from(rank)])
                .is_err()
        );
        assert_ne!(
            ticket_digest(rank, &t).unwrap(),
            manifest_digest(&m).unwrap()
        );
    }
    let mut m = manifest();
    m.ranks[1] = m.ranks[0];
    assert!(m.validate().is_err());
    let h = hello(0);
    assert!(ChildHello { guard_pid: 12, ..h }.validate().is_err());
}

#[test]
fn release_requires_both_exact_ordered_receipts_and_the_pending_local_one() {
    let m = manifest();
    let r = release();
    for rank in 0..2 {
        r.validate(&m, rank, &receipt(rank)).unwrap();
    }
    let mut forged = r;
    forged.receipt_digests[1] = [99; 32];
    assert!(forged.validate_fields().is_err());
    forged = r;
    forged.receipts.swap(0, 1);
    assert!(forged.validate_fields().is_err());
    forged = r;
    forged.receipts[1].receipt.child_instance = [99; 32];
    forged.receipt_digests[1] = quiescent_digest(&forged.receipts[1]).unwrap();
    // Recomputed hashes do not authorize a foreign child.
    assert!(forged.validate(&m, 0, &receipt(0)).is_err());
    let pending = Quiescent {
        receipt_nonce: [99; 32],
        ..receipt(0)
    };
    assert!(r.validate(&m, 0, &pending).is_err());
    assert!(
        Quiescent {
            last_command: 0,
            ..receipt(0)
        }
        .validate_fields()
        .is_err()
    );
    assert!(
        Quiescent {
            epoch: 2,
            ..receipt(0)
        }
        .validate_fields()
        .is_err()
    );
    assert!(r.validate(&m, 2, &receipt(0)).is_err());
    let d = release_digest(&r).unwrap();
    assert_ne!(d, quiescent_digest(&r.receipts[0]).unwrap());
    let a = Frame {
        rank: 0,
        body: Body::PairRelease(r),
    }
    .encode()
    .unwrap();
    let b = Frame {
        rank: 1,
        body: Body::PairRelease(r),
    }
    .encode()
    .unwrap();
    assert_ne!(a.as_slice()[11], b.as_slice()[11]);
    assert_eq!(&a.as_slice()[16..], &b.as_slice()[16..]);
}
