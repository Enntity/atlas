// SPDX-License-Identifier: AGPL-3.0-only

//! TCP bootstrap of the RDMA pair (rank 0 listens). Both ranks address the
//! peer's region with their own layout, so each first sends a fixed-size
//! [`Head`] with everything the layout and the flag protocol depend on. Only
//! a peer whose head agrees is sent the identity: the region base, then per
//! rail the QPN, PSN, GID and region rkey, which together grant remote writes.
//!
//! The ranks get here seconds apart: the NCCL bootstrap and
//! `ncclCommInitRank` before it have already absorbed any difference in model
//! load time, and what remains is each rank pinning and registering its
//! region. The boot fails after `ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S` (default
//! 120) rather than hang.

use anyhow::{Context, Result, anyhow, bail, ensure};
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// Names this wire format; bump the digit when the head or layout changes.
const MAGIC: [u8; 8] = *b"ATLPAIR3";
/// What both ranks must agree on, in wire order, and where to look when
/// they do not.
const FIELDS: [(&str, &str); 6] = [
    ("rail count", "check ATLAS_RDMA_RAILS on both nodes"),
    ("PAIR_CHAIN", "check ATLAS_RDMA_PAIR_CHAIN on both nodes"),
    (
        "segment count",
        "check ATLAS_RDMA_PAIR_SEGMENTS on both nodes",
    ),
    ("capacity", "max_batch_tokens and the model must match"),
    (
        "one-shot max",
        "check ATLAS_RDMA_ONESHOT and ATLAS_RDMA_ONESHOT_MAX on both nodes",
    ),
    (
        "one-shot stripe min",
        "check ATLAS_RDMA_ONESHOT_STRIPE_MIN on both nodes",
    ),
];
/// The rank, then the [`FIELDS`].
const WORDS: usize = 1 + FIELDS.len();
/// Magic, then the words as little-endian u64.
const HEAD_WIRE: usize = 8 * (1 + WORDS);

/// What a rank's use of the pair depends on, besides its own identity.
pub(super) struct Head {
    pub(super) rank: usize,
    pub(super) rails: usize,
    pub(super) chain: bool,
    pub(super) segments: usize,
    pub(super) capacity: usize,
    /// One-shot `max` and `stripe_min`, zero when off.
    pub(super) oneshot: [u64; 2],
}

impl Head {
    fn words(&self) -> [u64; WORDS] {
        [
            self.rank as u64,
            self.rails as u64,
            self.chain as u64,
            self.segments as u64,
            self.capacity as u64,
            self.oneshot[0],
            self.oneshot[1],
        ]
    }

    fn wire(&self) -> [u8; HEAD_WIRE] {
        let mut w = [0u8; HEAD_WIRE];
        w[..8].copy_from_slice(&MAGIC);
        for (at, word) in w[8..].chunks_exact_mut(8).zip(self.words()) {
            at.copy_from_slice(&word.to_le_bytes());
        }
        w
    }
}

/// Names a failed bootstrap read or write; a timeout reads as one, and a
/// hang-up as the rejection it is (the wire has no error frame).
pub(super) fn io(what: &str) -> impl Fn(std::io::Error) -> anyhow::Error {
    move |e| {
        let why = match e.kind() {
            ErrorKind::WouldBlock | ErrorKind::TimedOut => "timed out".into(),
            ErrorKind::UnexpectedEof => {
                "the peer closed the connection (its log says why it rejected this rank)".into()
            }
            _ => e.to_string(),
        };
        anyhow!("RDMA pair: {what}: {why}")
    }
}

/// The peer's head words. The magic is read first and alone: a peer on a
/// build with another format fails here on both sides -- it reads a foreign
/// or missing magic too -- rather than being misparsed.
fn read_head(stream: &mut impl Read) -> Result<[u64; WORDS]> {
    let mut magic = [0u8; 8];
    stream.read_exact(&mut magic).map_err(io("peer head"))?;
    ensure!(
        magic == MAGIC,
        "RDMA pair: the peer runs a different build (bootstrap magic {magic:02x?})"
    );
    let mut rest = [0u8; HEAD_WIRE - 8];
    stream.read_exact(&mut rest).map_err(io("peer head"))?;
    Ok(std::array::from_fn(|i| {
        u64::from_le_bytes(rest[8 * i..8 * i + 8].try_into().unwrap())
    }))
}

/// Agree on the heads, then send `ident` and return the peer's identity (as
/// many bytes). Rank 0 listens, so it stays silent until the connecting side
/// has shown this format, and neither rank writes its identity to a peer
/// whose head differs or is its own reflected.
pub(super) fn exchange(
    stream: &mut (impl Read + Write),
    head: &Head,
    ident: &[u8],
) -> Result<Vec<u8>> {
    let (ours, wire) = (head.words(), head.wire());
    if head.rank != 0 {
        stream.write_all(&wire).map_err(io("send head"))?;
    }
    let theirs = read_head(stream)?;
    if head.rank == 0 {
        stream.write_all(&wire).map_err(io("send head"))?;
    }
    ensure!(
        theirs[0] == ours[0] ^ 1,
        "RDMA pair: rank: here {}, peer {} - a reflected head, or both nodes run the same --rank",
        ours[0],
        theirs[0]
    );
    for ((name, hint), (here, peer)) in FIELDS.iter().zip(ours.iter().zip(&theirs).skip(1)) {
        ensure!(
            here == peer,
            "RDMA pair: {name}: here {here}, peer {peer} - {hint}"
        );
    }
    stream.write_all(ident).map_err(io("send identity"))?;
    let mut remote = vec![0u8; ident.len()];
    stream
        .read_exact(&mut remote)
        .map_err(io("peer identity"))?;
    Ok(remote)
}

/// `ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S`; never zero, which a socket timeout
/// rejects.
fn boot_timeout(secs: Option<String>) -> Duration {
    Duration::from_secs(secs.and_then(|v| v.parse().ok()).unwrap_or(120).max(1))
}

/// The bootstrap stream to the peer, exchanged, with the peer's identity.
/// `at` is rank 0's address on the link between the ranks: rank 0 listens
/// there only, and the worker dials it.
pub(super) fn open(
    head: &Head,
    ident: &[u8],
    at: SocketAddr,
    worker: Option<IpAddr>,
) -> Result<(TcpStream, Vec<u8>)> {
    let timeout = boot_timeout(std::env::var("ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S").ok());
    if head.rank == 0 {
        listen(head, ident, at, worker, timeout)
    } else {
        dial(head, ident, at, timeout)
    }
}

/// Blocking (an accepted socket can inherit the listener's non-blocking
/// mode), unbuffered, and bounded per read or write.
fn configure(stream: &TcpStream, timeout: Duration) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))
}

/// Rank 0: take connections on `at` until one completes the exchange. One
/// that does not -- a scanner, a stale or misconfigured rank, anyone but
/// `worker` when that is known -- is dropped and logged; it cannot end the
/// boot, only `timeout` does. Each gets a twelfth of `timeout` per read or
/// write, so silent ones cannot hold the listener for the whole window.
fn listen(
    head: &Head,
    ident: &[u8],
    at: SocketAddr,
    worker: Option<IpAddr>,
    timeout: Duration,
) -> Result<(TcpStream, Vec<u8>)> {
    let listener = TcpListener::bind(at).with_context(|| format!("RDMA pair: bind {at}"))?;
    // std has no accept timeout: poll.
    listener.set_nonblocking(true)?;
    let deadline = Instant::now() + timeout;
    let mut last = "no connection".to_owned();
    while Instant::now() < deadline {
        let (mut stream, from) = match listener.accept() {
            Ok(conn) => conn,
            // Nothing pending, or a connection that already went away.
            Err(e) => {
                if e.kind() != ErrorKind::WouldBlock {
                    last = format!("accept: {e}");
                }
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        let admitted = (|| {
            ensure!(
                worker.is_none_or(|w| w.to_canonical() == from.ip().to_canonical()),
                "not the worker of the NCCL bootstrap"
            );
            configure(&stream, timeout / 12)?;
            exchange(&mut stream, head, ident)
        })();
        match admitted {
            Ok(remote) => return Ok((stream, remote)),
            Err(e) => {
                tracing::warn!("RDMA pair: rejected a bootstrap connection from {from}: {e:#}");
                last = format!("rejected {from}: {e:#}");
            }
        }
    }
    bail!("RDMA pair: no valid peer on {at} within {timeout:?} (last: {last})")
}

/// The worker: reach rank 0, which may not be listening yet, and exchange.
fn dial(
    head: &Head,
    ident: &[u8],
    at: SocketAddr,
    timeout: Duration,
) -> Result<(TcpStream, Vec<u8>)> {
    let deadline = Instant::now() + timeout;
    let mut stream = loop {
        match TcpStream::connect_timeout(&at, timeout) {
            Ok(s) => break s,
            Err(e) if Instant::now() < deadline => {
                tracing::debug!("RDMA pair connect {at}: {e}; retrying");
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => return Err(e).with_context(|| format!("RDMA pair: connect {at}")),
        }
    };
    configure(&stream, timeout)?;
    let remote = exchange(&mut stream, head, ident)?;
    Ok((stream, remote))
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
