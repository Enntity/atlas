// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn config() -> Config {
    Config {
        policy: wire::Policy {
            startup: 1000,
            lease: 1000,
            challenge: 100,
            frame: 50,
            campaign: 10000,
            poll: 10,
            reap: 100,
            child_handshake: 500,
            quiescent_wait: 500,
            exit: 200,
        },
        readiness_ms: 2000,
        workload_ms: 2000,
        drain_ms: 1000,
        status_max_age_ms: 400,
    }
}
// Synthetic protocol records only: production state requires independently
// observed caller input. These tests do not claim proc, Docker or Model proof.
fn record(r: u8) -> wire::RankRecord {
    wire::RankRecord {
        container_id: [1 + r; 32],
        image_digest: [3; 32],
        recipe_digest: [4 + r; 32],
        server_elf_digest: [6; 32],
        guard_elf_digest: [7; 32],
        local_control_session: [8 + r; 32],
        guard_instance: [10 + r; 32],
        original_startup_challenge: [12 + r; 32],
        process: wire::ProcessIdentity {
            boot_id: [14 + r; 16],
            pid_namespace_device: 1,
            pid_namespace_inode: 100 + u64::from(r),
            guard_pid: 1,
            child_pid: 2,
            guard_start_ticks: 123,
            child_start_ticks: 124,
            child_instance: [16 + r; 32],
        },
    }
}
fn expected(r: u8) -> ExpectedRank {
    let v = record(r);
    let p = v.process;
    ExpectedRank {
        startup: wire::StartupRecord {
            pair_session: [18; 32],
            container_id: v.container_id,
            image_digest: v.image_digest,
            recipe_digest: v.recipe_digest,
            server_elf_digest: v.server_elf_digest,
            guard_elf_digest: v.guard_elf_digest,
            policy: config().policy,
        },
        process: ObservedProcess {
            boot_id: p.boot_id,
            pid_namespace_device: p.pid_namespace_device,
            pid_namespace_inode: p.pid_namespace_inode,
            guard_pid: p.guard_pid,
            child_pid: p.child_pid,
            guard_start_ticks: p.guard_start_ticks,
            child_start_ticks: p.child_start_ticks,
        },
    }
}
fn new() -> State {
    State::new(0, config(), [expected(0), expected(1)]).unwrap()
}
fn hello(r: u8) -> legacy::Frame {
    let v = record(r);
    legacy::Frame {
        kind: legacy::HELLO,
        session: v.local_control_session,
        instance: v.guard_instance,
        ordinal: 0,
        challenge: v.original_startup_challenge,
    }
}
fn gated(s: &mut State, r: u8) {
    // Exercise the real legacy and shared wire codec, not a copied frame layout.
    s.accept_hello(1, r, legacy::Frame::decode(&hello(r).encode()).unwrap())
        .unwrap();
    let f = wire::Frame {
        rank: r,
        body: wire::Body::GatedReport(wire::GatedReport {
            pair_session: [18; 32],
            record: record(r),
        }),
    };
    let decoded = wire::Frame::decode(
        f.encode().unwrap().as_slice(),
        wire::Direction::GuardToController,
    )
    .unwrap();
    let wire::Body::GatedReport(report) = decoded.body else {
        panic!("actual report");
    };
    s.accept_gated(1, r, report).unwrap();
}
fn health(s: &mut State, now: u64) {
    for r in 0..2 {
        let v = record(r);
        s.observe_running(now, r, v.container_id, v.image_digest, false)
            .unwrap();
    }
}
fn running() -> State {
    let mut s = new();
    gated(&mut s, 1);
    gated(&mut s, 0);
    let starts = s
        .paired_start(2)
        .expect("both actual reports must produce paired start");
    for (rank, f) in starts.into_iter().enumerate() {
        assert_eq!(f.rank, rank as u8);
        assert!(matches!(
            wire::Frame::decode(
                f.encode().unwrap().as_slice(),
                wire::Direction::ControllerToGuard
            )
            .unwrap()
            .body,
            wire::Body::PairedStart(_)
        ));
    }
    assert_eq!(s.phase(), Phase::Starting);
    s.sent_start(3, 1).unwrap();
    assert_eq!(s.phase(), Phase::Starting);
    s.sent_start(3, 0).unwrap();
    assert_eq!(s.phase(), Phase::Running);
    health(&mut s, 3);
    s
}
fn draining() -> State {
    let mut s = running();
    s.mark_ready(4).unwrap();
    let f = s.workload_complete(5).unwrap();
    assert_eq!(f.rank, 0);
    assert!(matches!(
        wire::Frame::decode(
            f.encode().unwrap().as_slice(),
            wire::Direction::ControllerToGuard
        )
        .unwrap()
        .body,
        wire::Body::DrainRequest(_)
    ));
    s
}
fn receipt(s: &State, r: u8) -> wire::Quiescent {
    wire::Quiescent {
        pair_digest: wire::manifest_digest(s.manifest.as_ref().unwrap()).unwrap(),
        child_instance: record(r).process.child_instance,
        epoch: wire::DRAIN_EPOCH,
        last_command: wire::SHUTDOWN_COMMAND,
        receipt_nonce: [24 + r; 32],
    }
}
fn releasing() -> State {
    let mut s = draining();
    for r in [1, 0] {
        let q = receipt(&s, r);
        s.accept_quiescent(6, r, q).unwrap();
    }
    let release = s.release(7).unwrap();
    for (r, f) in release.into_iter().enumerate() {
        let wire::Body::PairRelease(p) = f.body else {
            panic!("release");
        };
        p.validate(s.manifest.as_ref().unwrap(), r as u8, &receipt(&s, r as u8))
            .unwrap();
    }
    s
}
fn sticky(s: &mut State) {
    assert_eq!(s.phase(), Phase::Failed);
    assert!(s.tick(0).is_err());
    assert!(s.paired_start(0).is_err());
    assert_eq!(s.phase(), Phase::Failed);
}
#[test]
fn controller_requires_both_real_reports_then_both_releases_and_exact_exits() {
    let mut s = releasing();
    s.sent_release(8, 0).unwrap();
    s.relay_eof(9, 0).unwrap();
    let a = record(0);
    s.observe_exit(9, 0, a.container_id, a.image_digest, 0, false)
        .unwrap();
    assert_eq!(s.phase(), Phase::Releasing);
    s.sent_release(10, 1).unwrap();
    s.relay_eof(10, 1).unwrap();
    assert_eq!(
        s.phase(),
        Phase::Releasing,
        "relay EOF is not a success certificate"
    );
    let b = record(1);
    s.observe_exit(11, 1, b.container_id, b.image_digest, 0, false)
        .unwrap();
    assert_eq!(s.phase(), Phase::Succeeded);
}

#[test]
fn in_flight_release_challenges_are_valid_but_never_renew_an_exited_pair() {
    let mut s = releasing();
    for r in 0..2 {
        let mut f = hello(r);
        f.kind = legacy::CHALLENGE;
        f.ordinal = 1;
        f.challenge = [30 + r; 32];
        s.accept_challenge(8, r, f)
            .expect("guard can challenge until actual exit");
    }
    s.renew(9).expect("both peers still healthy during release");
    s.sent_release(10, 0).unwrap();
    let a = record(0);
    s.observe_exit(11, 0, a.container_id, a.image_digest, 0, false)
        .unwrap();
    for r in 0..2 {
        let mut f = hello(r);
        f.kind = legacy::CHALLENGE;
        f.ordinal = 2;
        f.challenge = [32 + r; 32];
        // A queued frame may be read after an independent exit observation.
        s.accept_challenge(12, r, f).unwrap();
    }
    assert!(
        s.renew(13).is_err(),
        "no renewal after either exact peer exited"
    );
}

#[test]
fn release_eof_stops_renewal_without_certifying_exit() {
    let mut s = releasing();
    s.sent_release(8, 0).unwrap();
    s.relay_eof(9, 0).unwrap();
    assert_eq!(s.phase(), Phase::Releasing);
    for r in 0..2 {
        let mut f = hello(r);
        f.kind = legacy::CHALLENGE;
        f.ordinal = 1;
        f.challenge = [34 + r; 32];
        s.accept_challenge(10, r, f).unwrap();
    }
    assert!(
        s.renew(11).is_err(),
        "closed relay cannot refresh pair leases"
    );
}
#[test]
fn startup_identity_fields_are_compared_not_invented() {
    for field in 0..8 {
        let mut s = new();
        s.accept_hello(1, 0, hello(0)).unwrap();
        let mut r = record(0);
        match field {
            0 => r.container_id[0] ^= 1,
            1 => r.image_digest[0] ^= 1,
            2 => r.process.boot_id[0] ^= 1,
            3 => r.process.child_start_ticks += 1,
            4 => r.local_control_session[0] ^= 1,
            5 => r.guard_instance[0] ^= 1,
            6 => r.original_startup_challenge[0] ^= 1,
            _ => r.process.child_instance = [0; 32],
        }
        assert!(s
            .accept_gated(
                2,
                0,
                wire::GatedReport {
                    pair_session: [18; 32],
                    record: r
                }
            )
            .is_err());
        sticky(&mut s);
    }
    let mut s = new();
    let mut r = record(0);
    r.process.child_instance = [99; 32];
    s.accept_hello(1, 0, hello(0)).unwrap();
    s.accept_gated(
        2,
        0,
        wire::GatedReport {
            pair_session: [18; 32],
            record: r,
        },
    )
    .unwrap();
    assert_eq!(
        s.reports[0].unwrap().process.child_instance,
        [99; 32],
        "nonzero instance comes from guard"
    );
}
#[test]
fn startup_replay_missing_peer_and_exact_deadline_are_terminal() {
    let mut s = new();
    gated(&mut s, 0);
    assert!(s.paired_start(2).is_err());
    sticky(&mut s);
    let mut s = new();
    gated(&mut s, 0);
    assert!(s.accept_hello(2, 0, hello(0)).is_err());
    sticky(&mut s);
    let mut s = new();
    assert!(s.tick(1000).is_err());
    sticky(&mut s);
    let mut s = new();
    assert!(s.accept_hello(1, 2, hello(0)).is_err());
    sticky(&mut s);
    assert!(State::new(u64::MAX, config(), [expected(0), expected(1)]).is_err());
}
#[test]
fn fresh_paired_challenges_and_health_produce_exact_renewals() {
    let mut s = running();
    for r in 0..2 {
        let mut f = hello(r);
        f.kind = legacy::CHALLENGE;
        f.ordinal = 1;
        f.challenge = [30 + r; 32];
        s.accept_challenge(100, r, f).unwrap();
    }
    health(&mut s, 101);
    let renew = s.renew(102).unwrap();
    for (r, f) in renew.into_iter().enumerate() {
        assert_eq!(f.kind, legacy::RENEW);
        assert_eq!(f.ordinal, 1);
        assert_eq!(f.challenge, [30 + r as u8; 32]);
        assert_eq!(f.session, hello(r as u8).session);
    }
    assert!(s.renew(103).is_err());
    sticky(&mut s);
}
#[test]
fn no_renewal_after_missing_peer_stale_status_or_late_challenge() {
    for failure in 0..3 {
        let mut s = running();
        for r in 0..if failure == 0 { 1 } else { 2 } {
            let mut f = hello(r);
            f.kind = legacy::CHALLENGE;
            f.ordinal = 1;
            f.challenge = [30 + r; 32];
            s.accept_challenge(500, r, f).unwrap();
        }
        if failure != 1 {
            health(&mut s, 501);
        }
        assert!(s.renew(if failure == 2 { 550 } else { 502 }).is_err());
        sticky(&mut s);
    }
}
#[test]
fn startup_exit_readiness_and_workload_order_cannot_grant_drain() {
    let mut s = running();
    assert!(s.workload_complete(4).is_err());
    sticky(&mut s);
    let mut s = new();
    let r = record(0);
    assert!(s
        .observe_exit(1, 0, r.container_id, r.image_digest, 0, false)
        .is_err());
    sticky(&mut s);
    let mut s = draining();
    assert!(s.workload_complete(6).is_err());
    sticky(&mut s);
    let mut s = running();
    assert!(s.relay_eof(4, 0).is_err());
    sticky(&mut s);
    let mut s = running();
    assert!(s.tick(2003).is_err());
    sticky(&mut s);
}
#[test]
fn foreign_or_duplicate_receipts_missing_peer_and_late_release_refuse() {
    for failure in 0..4 {
        let mut s = draining();
        let mut q = receipt(&s, 0);
        if failure == 0 {
            q.child_instance = record(1).process.child_instance;
        }
        if failure == 0 {
            assert!(s.accept_quiescent(6, 0, q).is_err());
        } else {
            s.accept_quiescent(6, 0, q).unwrap();
            match failure {
                1 => assert!(s.accept_quiescent(7, 0, q).is_err()),
                2 => assert!(s.release(7).is_err()),
                _ => assert!(s.tick(506).is_err()),
            }
        }
        sticky(&mut s);
    }
}
#[test]
fn exit_needs_release_delivery_full_identity_zero_code_and_no_oom() {
    for failure in 0..6 {
        let mut s = releasing();
        let mut r = record(0);
        if failure != 0 {
            s.sent_release(8, 0).unwrap();
        }
        if failure == 1 {
            r.container_id[0] ^= 1;
        }
        if failure == 2 {
            r.image_digest[0] ^= 1;
        }
        assert!(s
            .observe_exit(
                if failure == 5 { 207 } else { 9 },
                0,
                r.container_id,
                r.image_digest,
                if failure == 3 { 74 } else { 0 },
                failure == 4
            )
            .is_err());
        sticky(&mut s);
    }
}
