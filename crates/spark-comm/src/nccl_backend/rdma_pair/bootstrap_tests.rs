// SPDX-License-Identifier: AGPL-3.0-only

//! Bootstrap tests: the exchange over a scripted stream, then rank 0's
//! listener and the worker's dial over loopback TCP.

use super::*;
use std::net::Ipv4Addr;

const LOCAL: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// One end of the bootstrap connection: `peer` is what the other rank
/// sent, `sent` what we wrote.
struct Wire {
    peer: std::io::Cursor<Vec<u8>>,
    sent: Vec<u8>,
}

impl Read for Wire {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.peer.read(buf)
    }
}

impl Write for Wire {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.sent.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn head(rank: usize) -> Head {
    Head {
        rank,
        rails: 2,
        chain: false,
        segments: 4,
        capacity: 1 << 20,
        oneshot: [1 << 20, 1 << 20],
    }
}

/// A region base and two rails, recognisable per rank.
fn ident(rank: usize) -> Vec<u8> {
    vec![0xa0 + rank as u8; 8 + 2 * 28]
}

/// Everything `h`'s rank writes when the exchange succeeds.
fn script(h: &Head) -> Vec<u8> {
    [&h.wire()[..], &ident(h.rank)].concat()
}

/// `local`'s side against a peer that sent `peer`: the result and what we
/// wrote.
fn run(local: &Head, peer: Vec<u8>) -> (Result<Vec<u8>>, Vec<u8>) {
    let mut wire = Wire {
        peer: std::io::Cursor::new(peer),
        sent: Vec::new(),
    };
    let got = exchange(&mut wire, local, &ident(local.rank));
    (got, wire.sent)
}

fn why<T>(got: Result<T>) -> String {
    format!("{:#}", got.err().expect("the bootstrap must fail"))
}

#[test]
fn matching_ranks_swap_identities() {
    for rank in 0..2 {
        let (local, peer) = (head(rank), head(1 - rank));
        let (got, sent) = run(&local, script(&peer));
        assert_eq!(got.unwrap(), ident(1 - rank));
        assert_eq!(sent, script(&local));
    }
}

#[test]
fn each_differing_field_is_named_on_both_sides() {
    type Edit = fn(&mut Head);
    let cases: [(Edit, &str, &str); 6] = [
        (
            |h| h.rails = 1,
            "rail count: here 2, peer 1 - check ATLAS_RDMA_RAILS on both nodes",
            "rail count: here 1, peer 2 - check ATLAS_RDMA_RAILS on both nodes",
        ),
        (
            |h| h.chain = true,
            "PAIR_CHAIN: here 0, peer 1 - check ATLAS_RDMA_PAIR_CHAIN",
            "PAIR_CHAIN: here 1, peer 0",
        ),
        (
            |h| h.segments = 1,
            "segment count: here 4, peer 1 - check ATLAS_RDMA_PAIR_SEGMENTS",
            "segment count: here 1, peer 4",
        ),
        (
            |h| h.capacity = 2 << 20,
            "capacity: here 1048576, peer 2097152 - max_batch_tokens and the model must match",
            "capacity: here 2097152, peer 1048576",
        ),
        (
            |h| h.oneshot[0] = 0,
            "one-shot max: here 1048576, peer 0 - check ATLAS_RDMA_ONESHOT",
            "one-shot max: here 0, peer 1048576",
        ),
        (
            |h| h.oneshot[1] = 0,
            "one-shot stripe min: here 1048576, peer 0 - check ATLAS_RDMA_ONESHOT_STRIPE_MIN",
            "one-shot stripe min: here 0, peer 1048576",
        ),
    ];
    for (edit, here, there) in cases {
        for rank in 0..2 {
            let (good, mut odd) = (head(rank), head(1 - rank));
            edit(&mut odd);
            // Each side names the field and has sent its head, nothing more.
            for (local, peer, want) in [(&good, &odd, here), (&odd, &good, there)] {
                let (got, sent) = run(local, script(peer));
                let said = why(got);
                assert!(said.contains(want), "{said}");
                assert_eq!(sent, local.wire());
            }
        }
    }
}

#[test]
fn a_bad_or_reflected_head_gets_no_identity() {
    // Not this format (a scanner, the previous build): rank 0 says nothing,
    // the worker has sent its head only.
    for junk in [b"GET / HTTP/1.1\r\n".repeat(8), b"ATLPAIR2".repeat(16)] {
        let (got, sent) = run(&head(0), junk.clone());
        assert!(why(got).contains("different build"));
        assert!(sent.is_empty());
        let (got, sent) = run(&head(1), junk);
        assert!(why(got).contains("different build"));
        assert_eq!(sent, head(1).wire());
    }
    // A rank's own head played back to it.
    for rank in 0..2 {
        let local = head(rank);
        let (got, sent) = run(&local, script(&local));
        let said = why(got);
        assert!(
            said.contains(&format!("rank: here {rank}, peer {rank}")),
            "{said}"
        );
        assert_eq!(sent, local.wire());
    }
    // A peer that hangs up mid-head.
    let (got, sent) = run(&head(0), head(1).wire()[..20].to_vec());
    assert!(why(got).contains("closed the connection"));
    assert!(sent.is_empty());
}

#[test]
fn the_boot_timeout_defaults_to_two_minutes_and_is_never_zero() {
    let secs = |v: &str| boot_timeout(Some(v.to_owned())).as_secs();
    assert_eq!(boot_timeout(None).as_secs(), 120);
    assert_eq!((secs("30"), secs("0"), secs("soon")), (30, 1, 120));
}

/// A loopback address nothing listens on yet.
fn free_addr() -> SocketAddr {
    let probe = TcpListener::bind((LOCAL, 0)).unwrap();
    probe.local_addr().unwrap()
}

/// Connect as soon as rank 0 listens.
fn reach(at: SocketAddr) -> TcpStream {
    loop {
        if let Ok(stream) = TcpStream::connect(at) {
            return stream;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// What a connection that writes `say` is sent before rank 0 drops it.
fn probe(at: SocketAddr, say: &[u8]) -> Vec<u8> {
    let mut stream = reach(at);
    // Dropped with our bytes unread, the connection ends in a reset.
    let _ = stream.write_all(say);
    let mut got = Vec::new();
    let _ = stream.read_to_end(&mut got);
    got
}

fn rank0(
    at: SocketAddr,
    worker: Option<IpAddr>,
    timeout: Duration,
) -> std::thread::JoinHandle<Result<Vec<u8>>> {
    std::thread::spawn(move || Ok(listen(&head(0), &ident(0), at, worker, timeout)?.1))
}

#[test]
fn a_pair_boots_over_tcp() {
    let at = free_addr();
    let rank = |r: usize| {
        let worker = Some(LOCAL);
        std::thread::spawn(move || open(&head(r), &ident(r), at, worker).unwrap())
    };
    let (zero, one) = (rank(0), rank(1));
    let ((mut s0, got0), (mut s1, got1)) = (zero.join().unwrap(), one.join().unwrap());
    assert_eq!((got0, got1), (ident(1), ident(0)));
    // The stream then carries the barrier.
    let mut byte = [0u8];
    s0.write_all(&[1]).unwrap();
    s1.read_exact(&mut byte).unwrap();
    assert_eq!(byte, [1]);
}

#[test]
fn rejected_connections_do_not_stop_a_later_valid_peer() {
    let (at, timeout) = (free_addr(), Duration::from_secs(12));
    let zero = rank0(at, None, timeout);
    // A scanner and a connection that never speaks are sent nothing; a
    // worker with one rail is sent the head that tells it so.
    assert!(probe(at, &[0x55; HEAD_WIRE]).is_empty());
    assert!(probe(at, &[]).is_empty());
    let mut odd = head(1);
    odd.rails = 1;
    assert_eq!(probe(at, &odd.wire()), head(0).wire());
    let (_, got1) = dial(&head(1), &ident(1), at, timeout).unwrap();
    assert_eq!((zero.join().unwrap().unwrap(), got1), (ident(1), ident(0)));
}

#[test]
fn a_rail_mismatch_ends_both_ranks_with_its_name() {
    let at = free_addr();
    let zero = rank0(at, None, Duration::from_millis(1200));
    drop(reach(at));
    let mut odd = head(1);
    odd.rails = 1;
    let said = why(dial(&odd, &ident(1), at, Duration::from_secs(5)));
    let want = "rail count: here 1, peer 2 - check ATLAS_RDMA_RAILS on both nodes";
    assert!(said.contains(want), "{said}");
    let said = why(zero.join().unwrap());
    let want = "rail count: here 2, peer 1 - check ATLAS_RDMA_RAILS on both nodes";
    assert!(said.contains(want), "{said}");
}

#[test]
fn a_silent_peer_times_out() {
    // The worker, against a listener that never answers.
    let quiet = TcpListener::bind((LOCAL, 0)).unwrap();
    let at = quiet.local_addr().unwrap();
    let got = dial(&head(1), &ident(1), at, Duration::from_millis(200));
    assert!(why(got).contains("timed out"));
    // Rank 0, against a connection that never speaks.
    let at = free_addr();
    let zero = rank0(at, None, Duration::from_millis(1200));
    let held = reach(at);
    let said = why(zero.join().unwrap());
    assert!(
        said.contains("no valid peer") && said.contains("timed out"),
        "{said}"
    );
    drop(held);
}

#[test]
fn only_the_worker_of_the_nccl_bootstrap_is_admitted() {
    let (at, other) = (free_addr(), Ipv4Addr::new(127, 0, 0, 2).into());
    let zero = rank0(at, Some(other), Duration::from_millis(600));
    assert!(probe(at, &head(1).wire()).is_empty());
    let said = why(zero.join().unwrap());
    assert!(said.contains("not the worker"), "{said}");
}

#[test]
fn rank0_listens_on_the_given_address_or_not_at_all() {
    // TEST-NET-1 is nobody's interface: no fallback to every interface.
    let at = SocketAddr::new(Ipv4Addr::new(192, 0, 2, 1).into(), free_addr().port());
    let got = listen(&head(0), &ident(0), at, None, Duration::from_millis(100));
    assert!(why(got).contains("bind 192.0.2.1"));
}
