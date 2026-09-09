// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

enum Control {
    Lease(frame::Frame),
    Pair(wire::Frame),
}
fn receive(socket: &mut UnixStream) -> Result<Control> {
    let mut prefix = [0; 4];
    socket.read_exact(&mut prefix)?;
    let size = wire::control_frame_len(prefix, wire::Direction::GuardToController)?;
    let mut bytes = vec![0; size];
    bytes[..4].copy_from_slice(&prefix);
    socket.read_exact(&mut bytes[4..])?;
    if size == frame::LEN {
        Ok(Control::Lease(
            frame::Frame::decode(bytes.as_slice().try_into()?).map_err(linux::error)?,
        ))
    } else {
        Ok(Control::Pair(wire::Frame::decode(
            &bytes,
            wire::Direction::GuardToController,
        )?))
    }
}
fn alive(nodes: &mut [Namespace; 2]) -> Result<()> {
    for node in nodes {
        ensure!(
            node.process.try_wait()?.is_none(),
            "guard exited before current pair receipts"
        );
    }
    Ok(())
}

pub(super) fn run(nodes: &mut [Namespace; 2], manifest: &wire::Manifest, mode: &str) -> Result<()> {
    for (rank, node) in nodes.iter_mut().enumerate() {
        write_frame(
            &mut node.socket,
            wire::Frame {
                rank: rank as u8,
                body: wire::Body::PairedStart(*manifest),
            },
        )?;
    }
    if matches!(
        mode,
        "unreleased-zero"
            | "bad-environment"
            | "registered-wrong-rank"
            | "registered-missing-capability"
            | "registered-unhealthy"
    ) {
        for node in nodes {
            node.finish_expected(74)?;
        }
        return Ok(());
    }
    let deadline = identity::boot_time_ms()? + 15000;
    let mut pending = [None, None];
    let mut challenges: [Option<frame::Frame>; 2] = [None, None];
    let mut ordinals = [0, 0];
    let mut renewal_pairs = 0;
    while pending.iter().any(Option::is_none) {
        identity::check_deadline(deadline)?;
        alive(nodes)?;
        let mut polls = [
            libc::pollfd {
                fd: nodes[0].socket.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: nodes[1].socket.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        linux::Io::poll(&mut polls, 25)?;
        for rank in 0..2 {
            ensure!(
                polls[rank].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) == 0,
                "guard control failed"
            );
            if polls[rank].revents & libc::POLLIN == 0 {
                continue;
            }
            match receive(&mut nodes[rank].socket)? {
                Control::Lease(challenge) => {
                    let record = &manifest.ranks[rank];
                    ensure!(
                        challenge.kind == frame::CHALLENGE
                            && challenge.session == record.local_control_session
                            && challenge.instance == record.guard_instance
                            && challenge.ordinal == ordinals[rank] + 1
                            && challenges[rank].is_none(),
                        "fresh exact guard challenge"
                    );
                    challenges[rank] = Some(challenge);
                }
                Control::Pair(frame) => {
                    ensure!(
                        frame.rank as usize == rank && pending[rank].is_none(),
                        "unique current receipt"
                    );
                    let wire::Body::Quiescent(receipt) = frame.body else {
                        bail!("expected quiescent receipt")
                    };
                    receipt.validate(manifest, rank as u8)?;
                    pending[rank] = Some(wire::QuiescentFrame {
                        rank: rank as u8,
                        receipt,
                    });
                }
            }
        }
        if challenges.iter().all(Option::is_some) {
            // Renew only after both fresh challenges and a fresh check of both
            // exact wrappers; this fixture never runs a detached renewal thread.
            alive(nodes)?;
            for rank in 0..2 {
                let mut response = challenges[rank].take().unwrap();
                ordinals[rank] = response.ordinal;
                response.kind = frame::RENEW;
                nodes[rank].socket.write_all(&response.encode())?;
            }
            renewal_pairs += 1;
        }
    }
    if mode == "delayed" {
        ensure!(
            renewal_pairs >= 1,
            "delayed peer must exercise both guard renewals"
        );
    }
    let receipts = [pending[0].unwrap(), pending[1].unwrap()];
    let release = wire::PairRelease {
        pair_digest: wire::manifest_digest(manifest)?,
        epoch: wire::DRAIN_EPOCH,
        receipts,
        receipt_digests: [
            wire::quiescent_digest(&receipts[0])?,
            wire::quiescent_digest(&receipts[1])?,
        ],
    };
    for (rank, node) in nodes.iter_mut().enumerate() {
        let frame = wire::Frame {
            rank: rank as u8,
            body: wire::Body::PairRelease(release),
        };
        let mut bytes = frame.encode()?.as_slice().to_vec();
        if mode == "bad-release" {
            *bytes.last_mut().unwrap() ^= 128;
        }
        if mode == "replayed-release" {
            bytes.extend_from_within(..);
        }
        node.socket.write_all(&bytes)?;
    }
    for node in nodes {
        node.finish_expected(
            if matches!(
                mode,
                "valid" | "delayed" | "reused-session" | "registered-valid"
            ) {
                0
            } else {
                74
            },
        )?;
    }
    println!("observed renewal pairs={renewal_pairs}");
    Ok(())
}
