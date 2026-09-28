// SPDX-License-Identifier: AGPL-3.0-only

use crate::*;

pub(crate) fn nonzero(bytes: &[u8]) -> Result<()> {
    if bytes.iter().all(|&b| b == 0) {
        return Err(Error("missing identity or nonce"));
    }
    Ok(())
}

fn process(boot: &[u8; 16], inode: u64, guard: u32, child: u32) -> Result<()> {
    nonzero(boot)?;
    if inode == 0 || guard != 1 || child <= 1 || child > i32::MAX as u32 {
        return Err(Error("invalid PID1 child identity"));
    }
    Ok(())
}

impl Policy {
    pub fn validate(&self) -> Result<()> {
        let all = [
            self.startup,
            self.lease,
            self.challenge,
            self.frame,
            self.campaign,
            self.poll,
            self.reap,
            self.child_handshake,
            self.quiescent_wait,
            self.exit,
        ];
        if all
            .iter()
            .any(|&v| v == 0 || v > self.campaign || v > 86_400_000)
            || self.poll > 250
            || self.reap > 10_000
            || self.frame >= self.challenge
            || self.challenge >= self.lease
            || [self.child_handshake, self.quiescent_wait, self.exit]
                .iter()
                .any(|&v| v > self.lease)
        {
            return Err(Error("invalid explicit policy"));
        }
        Ok(())
    }
}
impl ProcessIdentity {
    pub fn validate(&self) -> Result<()> {
        process(
            &self.boot_id,
            self.pid_namespace_inode,
            self.guard_pid,
            self.child_pid,
        )?;
        nonzero(&self.child_instance)
    }
}
impl RankRecord {
    pub fn validate(&self) -> Result<()> {
        for id in [
            &self.container_id,
            &self.image_digest,
            &self.recipe_digest,
            &self.server_elf_digest,
            &self.guard_elf_digest,
            &self.local_control_session,
            &self.guard_instance,
            &self.original_startup_challenge,
        ] {
            nonzero(id)?;
        }
        self.process.validate()
    }
}
impl StartupRecord {
    pub fn validate(&self) -> Result<()> {
        for id in [
            &self.pair_session,
            &self.container_id,
            &self.image_digest,
            &self.recipe_digest,
            &self.server_elf_digest,
            &self.guard_elf_digest,
        ] {
            nonzero(id)?;
        }
        self.policy.validate()
    }
}
impl Manifest {
    pub fn validate(&self) -> Result<()> {
        nonzero(&self.pair_session)?;
        nonzero(&self.policy_digest)?;
        for r in &self.ranks {
            r.validate()?;
        }
        let [a, b] = &self.ranks;
        if a.container_id == b.container_id
            || a.local_control_session == b.local_control_session
            || a.guard_instance == b.guard_instance
            || a.process.child_instance == b.process.child_instance
            || (a.process.boot_id == b.process.boot_id
                && a.process.pid_namespace_device == b.process.pid_namespace_device
                && a.process.pid_namespace_inode == b.process.pid_namespace_inode)
        {
            return Err(Error("pair repeats a local instance"));
        }
        Ok(())
    }
    /// Compare supplied data with independently established local records.
    /// Does not establish their provenance or replace live process/channel checks.
    pub fn validate_local(
        &self,
        rank: u8,
        startup: &StartupRecord,
        actual: &RankRecord,
    ) -> Result<()> {
        self.validate()?;
        startup.validate()?;
        let r = self
            .ranks
            .get(usize::from(rank))
            .ok_or(Error("invalid rank"))?;
        if r != actual
            || self.pair_session != startup.pair_session
            || self.policy_digest != policy_digest(&startup.policy)?
            || r.container_id != startup.container_id
            || r.image_digest != startup.image_digest
            || r.recipe_digest != startup.recipe_digest
            || r.server_elf_digest != startup.server_elf_digest
            || r.guard_elf_digest != startup.guard_elf_digest
        {
            return Err(Error("manifest disagrees with local records"));
        }
        Ok(())
    }
}
impl ChildHello {
    pub fn validate(&self) -> Result<()> {
        process(
            &self.boot_id,
            self.pid_namespace_inode,
            self.guard_pid,
            self.child_pid,
        )?;
        nonzero(&self.server_challenge)
    }
    pub fn validate_process(&self, expected: &ProcessIdentity) -> Result<()> {
        self.validate()?;
        expected.validate()?;
        if self.boot_id != expected.boot_id
            || self.pid_namespace_device != expected.pid_namespace_device
            || self.pid_namespace_inode != expected.pid_namespace_inode
            || self.guard_pid != expected.guard_pid
            || self.child_pid != expected.child_pid
            || self.guard_start_ticks != expected.guard_start_ticks
            || self.child_start_ticks != expected.child_start_ticks
        {
            return Err(Error("HELLO process identity mismatch"));
        }
        Ok(())
    }
}
impl ChildTicket {
    pub fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        nonzero(&self.echoed_server_challenge)?;
        nonzero(&self.guard_ticket_challenge)
    }
    pub fn validate_hello(&self, rank: u8, hello: &ChildHello) -> Result<()> {
        self.validate()?;
        let r = self
            .manifest
            .ranks
            .get(usize::from(rank))
            .ok_or(Error("invalid rank"))?;
        hello.validate_process(&r.process)?;
        if self.echoed_server_challenge != hello.server_challenge {
            return Err(Error("HELLO challenge mismatch"));
        }
        Ok(())
    }
}
impl Quiescent {
    pub fn validate_fields(&self) -> Result<()> {
        nonzero(&self.pair_digest)?;
        nonzero(&self.child_instance)?;
        nonzero(&self.receipt_nonce)?;
        if self.epoch != DRAIN_EPOCH || self.last_command != SHUTDOWN_COMMAND {
            return Err(Error("not the one-shot shutdown receipt"));
        }
        Ok(())
    }
    pub fn validate(&self, manifest: &Manifest, rank: u8) -> Result<()> {
        self.validate_fields()?;
        let r = manifest
            .ranks
            .get(usize::from(rank))
            .ok_or(Error("invalid rank"))?;
        if self.pair_digest != manifest_digest(manifest)?
            || self.child_instance != r.process.child_instance
        {
            return Err(Error("receipt manifest/child mismatch"));
        }
        Ok(())
    }
}
impl PairRelease {
    pub fn validate_fields(&self) -> Result<()> {
        nonzero(&self.pair_digest)?;
        if self.epoch != DRAIN_EPOCH {
            return Err(Error("release epoch"));
        }
        for (rank, q) in self.receipts.iter().enumerate() {
            q.receipt.validate_fields()?;
            if usize::from(q.rank) != rank
                || q.receipt.epoch != self.epoch
                || q.receipt.pair_digest != self.pair_digest
                || quiescent_digest(q)? != self.receipt_digests[rank]
            {
                return Err(Error("release receipt ordering/digest mismatch"));
            }
        }
        if self.receipts[0].receipt.child_instance == self.receipts[1].receipt.child_instance {
            return Err(Error("release repeats child instance"));
        }
        Ok(())
    }
    /// Requires the exact pending local receipt, not just its digest or epoch.
    /// Callers still enforce one-shot phase, sender credentials and deadlines.
    pub fn validate(&self, manifest: &Manifest, local_rank: u8, pending: &Quiescent) -> Result<()> {
        self.validate_fields()?;
        let local = self
            .receipts
            .get(usize::from(local_rank))
            .ok_or(Error("invalid rank"))?;
        for q in &self.receipts {
            q.receipt.validate(manifest, q.rank)?;
        }
        if local.receipt != *pending {
            return Err(Error("release is not the pending local receipt"));
        }
        Ok(())
    }
}

impl DrainRequest {
    pub fn validate_fields(&self) -> Result<()> {
        nonzero(&self.pair_digest)?;
        if self.epoch != DRAIN_EPOCH {
            return Err(Error("drain request epoch"));
        }
        Ok(())
    }
    /// Only checks supplied data. Guard owns one-shot phase, deadline and
    /// actual held-child checks before signaling its pidfd.
    pub fn validate(&self, manifest: &Manifest) -> Result<()> {
        self.validate_fields()?;
        if self.pair_digest != manifest_digest(manifest)? {
            return Err(Error("drain request manifest mismatch"));
        }
        Ok(())
    }
}
