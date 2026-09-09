// SPDX-License-Identifier: AGPL-3.0-only

//! Untrusted-record binding tests, NOT actual PID1/channel/Model authority proof.
use super::*;
use atlas_glm_pair_wire::{Manifest, ProcessIdentity, RankRecord};

fn records(rank: u8) -> (ExpectedSession, ChildHello, ChildTicket) {
    let policy = Policy {
        startup: 30_000,
        lease: 30_000,
        challenge: 5000,
        frame: 3000,
        campaign: 60_000,
        poll: 250,
        reap: 10_000,
        child_handshake: 30_000,
        quiescent_wait: 10_000,
        exit: 5000,
    };
    let ranks = std::array::from_fn(|i| RankRecord {
        container_id: [1 + i as u8; 32],
        image_digest: [3; 32],
        recipe_digest: [4 + i as u8; 32],
        server_elf_digest: [6; 32],
        guard_elf_digest: [7; 32],
        local_control_session: [8 + i as u8; 32],
        guard_instance: [10 + i as u8; 32],
        original_startup_challenge: [12 + i as u8; 32],
        process: ProcessIdentity {
            boot_id: [14 + i as u8; 16],
            pid_namespace_device: 4,
            pid_namespace_inode: 99,
            guard_pid: 1,
            child_pid: 2,
            guard_start_ticks: 300,
            child_start_ticks: 400,
            child_instance: [16 + i as u8; 32],
        },
    });
    let r = ranks[rank as usize];
    let p = r.process;
    let expected = ExpectedSession {
        rank,
        pair_session: [18; 32],
        policy,
        container_id: r.container_id,
        image_digest: r.image_digest,
        recipe_digest: r.recipe_digest,
        server_elf_digest: r.server_elf_digest,
        guard_elf_digest: r.guard_elf_digest,
        max_executable_bytes: 512 << 20,
    };
    let hello = ChildHello {
        boot_id: p.boot_id,
        pid_namespace_device: p.pid_namespace_device,
        pid_namespace_inode: p.pid_namespace_inode,
        guard_pid: p.guard_pid,
        child_pid: p.child_pid,
        guard_start_ticks: p.guard_start_ticks,
        child_start_ticks: p.child_start_ticks,
        server_challenge: [19; 32],
    };
    let ticket = ChildTicket {
        manifest: Manifest {
            pair_session: expected.pair_session,
            policy_digest: policy_digest(&policy).unwrap(),
            ranks,
        },
        echoed_server_challenge: hello.server_challenge,
        guard_ticket_challenge: [20; 32],
    };
    (expected, hello, ticket)
}

#[test]
fn matching_hello_and_explicit_launch_record_bind_both_ranks() {
    for rank in 0..2 {
        let (expected, hello, ticket) = records(rank);
        expected.validate(30_000).unwrap();
        // Codec control passes before invoking the new production checker.
        ticket.validate_hello(rank, &hello).unwrap();
        let frame = Frame {
            rank,
            body: Body::ChildTicket(ticket),
        };
        assert_eq!(check_ticket(&expected, &hello, frame).unwrap(), ticket);
    }
}

#[test]
fn wrong_local_identity_challenge_or_launch_refuses() {
    for rank in 0..2 {
        let (expected, hello, ticket) = records(rank);
        for which in 0..16 {
            let mut bad = ticket;
            let r = &mut bad.manifest.ranks[rank as usize];
            match which {
                0 => r.container_id[0] ^= 128,
                1 => r.image_digest[0] ^= 128,
                2 => r.recipe_digest[0] ^= 128,
                3 => r.server_elf_digest[0] ^= 128,
                4 => r.guard_elf_digest[0] ^= 128,
                5 => r.process.boot_id[0] ^= 128,
                6 => r.process.pid_namespace_device += 1,
                7 => r.process.pid_namespace_inode += 1,
                8 => r.process.child_pid += 1,
                9 => r.process.guard_start_ticks += 1,
                10 => r.process.child_start_ticks += 1,
                11 => bad.manifest.pair_session[0] ^= 128,
                12 => bad.manifest.policy_digest[0] ^= 128,
                13 => bad.echoed_server_challenge[0] ^= 128,
                14 => bad.guard_ticket_challenge = [0; 32],
                15 => r.process.guard_pid = 3,
                _ => unreachable!(),
            }
            assert!(
                check_ticket(
                    &expected,
                    &hello,
                    Frame {
                        rank,
                        body: Body::ChildTicket(bad)
                    }
                )
                .is_err(),
                "rank{rank} mismatch{which}"
            );
        }
        assert!(check_ticket(
            &expected,
            &hello,
            Frame {
                rank: 1 - rank,
                body: Body::ChildTicket(ticket)
            }
        )
        .is_err());
        assert!(check_ticket(
            &expected,
            &hello,
            Frame {
                rank,
                body: Body::ChildHello(hello)
            }
        )
        .is_err());
    }
}

#[test]
fn invalid_explicit_bounds_refuse_before_fd_consumption() {
    let (expected, _, _) = records(0);
    for duration in [0, 30_001, u64::MAX] {
        assert!(unsafe { InheritedSession::receive(expected, duration) }.is_err());
    }
    // No reset-for-tests: these immutable failures never spent the FD entry.
    assert!(!INHERITED_SPENT.load(Ordering::Acquire));
    assert!(Deadline::new(u64::MAX).is_err());
    let mut expired = Deadline { at: 0, last: 0 };
    assert!(expired.check().is_err());
}
