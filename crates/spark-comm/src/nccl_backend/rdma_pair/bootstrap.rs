// SPDX-License-Identifier: AGPL-3.0-only

//! Bootstrap exchange of the RDMA pair: each rank sends a header, then per
//! rail its QPN, PSN, GID and region rkey. Both ranks address the peer's
//! region with their own layout, so everything the layout depends on must
//! match before either posts a WRITE.

use anyhow::{Result, ensure};
use std::io::{Read, Write};

/// Magic, region base, legacy capacity, one-shot settings.
pub(super) const HEAD_WIRE: usize = 8 + 8 + 8 + 16;
/// Names this wire format; bump the digit when the header or layout changes.
const MAGIC: [u8; 8] = *b"ATLPAIR2";

pub(super) fn head(base: u64, capacity: usize, oneshot: [u8; 16]) -> Vec<u8> {
    let mut w = Vec::with_capacity(HEAD_WIRE);
    w.extend_from_slice(&MAGIC);
    w.extend_from_slice(&base.to_le_bytes());
    w.extend_from_slice(&(capacity as u64).to_le_bytes());
    w.extend_from_slice(&oneshot);
    w
}

/// Send `local` (a [`head`] plus the rails) and return the peer's bytes and
/// its region base. The magic goes first and alone: a peer on a build without
/// it (or with another format) fails here on both sides -- it reads a short
/// header and we read a foreign one -- rather than misparsing identities.
pub(super) fn exchange(stream: &mut (impl Read + Write), local: &[u8]) -> Result<(Vec<u8>, u64)> {
    let mut remote = vec![0u8; local.len()];
    stream.write_all(&local[..8])?;
    stream.read_exact(&mut remote[..8])?;
    ensure!(
        remote[..8] == local[..8],
        "RDMA pair: the peer runs a different build (bootstrap magic {:02x?})",
        &remote[..8]
    );
    stream.write_all(&local[8..])?;
    stream.read_exact(&mut remote[8..])?;
    let word = |w: &[u8], at: usize| u64::from_le_bytes(w[at..at + 8].try_into().unwrap());
    ensure!(
        word(&remote, 16) == word(local, 16),
        "RDMA pair: the peer's capacity is {} B, ours {} B (max_batch_tokens and the model must match)",
        word(&remote, 16),
        word(local, 16)
    );
    ensure!(
        remote[24..HEAD_WIRE] == local[24..HEAD_WIRE],
        "RDMA pair: the peer's one-shot settings differ (ATLAS_RDMA_ONESHOT, \
         ATLAS_RDMA_ONESHOT_MAX and ATLAS_RDMA_ONESHOT_STRIPE_MIN must match)"
    );
    let base = word(&remote, 8);
    Ok((remote, base))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn run(local: &[u8], peer: Vec<u8>) -> (Result<(Vec<u8>, u64)>, usize) {
        let mut wire = Wire {
            peer: std::io::Cursor::new(peer),
            sent: Vec::new(),
        };
        let got = exchange(&mut wire, local);
        (got, wire.sent.len())
    }

    fn ident(base: u64, capacity: usize, oneshot: u8) -> Vec<u8> {
        let mut w = head(base, capacity, [oneshot; 16]);
        w.extend_from_slice(&[base as u8; 28]);
        w
    }

    #[test]
    fn matching_ranks_learn_the_peer_base_and_rails() {
        let (local, peer) = (ident(0x1000, 1 << 20, 0), ident(0x2000, 1 << 20, 0));
        assert_eq!(local.len(), HEAD_WIRE + 28);
        let (got, sent) = run(&local, peer.clone());
        assert_eq!(got.unwrap(), (peer, 0x2000));
        assert_eq!(sent, local.len());
    }

    #[test]
    fn a_different_layout_fails_at_bootstrap() {
        let local = ident(0x1000, 1 << 20, 1);
        let why = |peer| run(&local, peer).0.unwrap_err().to_string();
        assert!(why(ident(0x2000, 2 << 20, 1)).contains("capacity is 2097152 B"));
        assert!(why(ident(0x2000, 1 << 20, 0)).contains("one-shot settings differ"));
    }

    #[test]
    fn a_peer_without_the_magic_fails_on_both_sides() {
        // A build from before the header: region base, then the rails.
        let local = ident(0x1000, 1 << 20, 0);
        let mut old = 0x7f00_dead_0000u64.to_le_bytes().to_vec();
        old.extend_from_slice(&[7; 28]);
        let (got, sent) = run(&local, old);
        assert!(got.unwrap_err().to_string().contains("different build"));
        // We sent the magic only, so the old side's read of 36 bytes hits EOF.
        assert_eq!(sent, 8);
    }
}
