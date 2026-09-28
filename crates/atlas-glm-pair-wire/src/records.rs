// SPDX-License-Identifier: AGPL-3.0-only

use crate::codec::{Fixed, Reader, Writer, pair, record};
use crate::{Digest, Error, Result};

record!(Policy {
    startup: u64,
    lease: u64,
    challenge: u64,
    frame: u64,
    campaign: u64,
    poll: u64,
    reap: u64,
    child_handshake: u64,
    quiescent_wait: u64,
    exit: u64,
});
record!(ProcessIdentity {
    boot_id: [u8; 16],
    pid_namespace_device: u64,
    pid_namespace_inode: u64,
    guard_pid: u32,
    child_pid: u32,
    guard_start_ticks: u64,
    child_start_ticks: u64,
    child_instance: Digest,
});
record!(RankRecord {
    container_id: Digest,
    image_digest: Digest,
    recipe_digest: Digest,
    server_elf_digest: Digest,
    guard_elf_digest: Digest,
    local_control_session: Digest,
    guard_instance: Digest,
    original_startup_challenge: Digest,
    process: ProcessIdentity,
});
pair!(RankRecord);
record!(Manifest {
    pair_session: Digest,
    policy_digest: Digest,
    ranks: [RankRecord; 2]
});
record!(StartupRecord {
    pair_session: Digest,
    container_id: Digest,
    image_digest: Digest,
    recipe_digest: Digest,
    server_elf_digest: Digest,
    guard_elf_digest: Digest,
    policy: Policy,
});
record!(GatedReport {
    pair_session: Digest,
    record: RankRecord
});
record!(ChildHello {
    boot_id: [u8; 16],
    pid_namespace_device: u64,
    pid_namespace_inode: u64,
    guard_pid: u32,
    child_pid: u32,
    guard_start_ticks: u64,
    child_start_ticks: u64,
    server_challenge: Digest,
});
record!(ChildTicket {
    manifest: Manifest,
    echoed_server_challenge: Digest,
    guard_ticket_challenge: Digest,
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quiescent {
    pub pair_digest: Digest,
    pub child_instance: Digest,
    pub epoch: u64,
    pub last_command: u32,
    pub receipt_nonce: Digest,
}
impl Fixed for Quiescent {
    const LEN: usize = 112;
    fn put(&self, w: &mut Writer<'_>) -> Result<()> {
        w.field(&self.pair_digest)?;
        w.field(&self.child_instance)?;
        w.field(&self.epoch)?;
        w.field(&self.last_command)?;
        w.field(&0u32)?;
        w.field(&self.receipt_nonce)
    }
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        let pair_digest = r.field()?;
        let child_instance = r.field()?;
        let epoch = r.field()?;
        let last_command = r.field()?;
        if r.field::<u32>()? != 0 {
            return Err(Error("receipt reserved bits"));
        }
        Ok(Self {
            pair_digest,
            child_instance,
            epoch,
            last_command,
            receipt_nonce: r.field()?,
        })
    }
}

/// A complete embedded QUIESCENT frame, never a standalone authority object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuiescentFrame {
    pub rank: u8,
    pub receipt: Quiescent,
}
impl Fixed for QuiescentFrame {
    const LEN: usize = 128;
    fn put(&self, w: &mut Writer<'_>) -> Result<()> {
        crate::frame::put_header(w, Self::LEN, 0x14, self.rank)?;
        self.receipt.put(w)
    }
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        let rank = crate::frame::get_header(r, Self::LEN, 0x14)?;
        Ok(Self {
            rank,
            receipt: r.field()?,
        })
    }
}
pair!(QuiescentFrame);
pair!(Digest);
record!(PairRelease {
    pair_digest: Digest,
    epoch: u64,
    receipts: [QuiescentFrame; 2],
    receipt_digests: [Digest; 2],
});

record!(DrainRequest {
    pair_digest: Digest,
    epoch: u64,
});
