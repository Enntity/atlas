// SPDX-License-Identifier: AGPL-3.0-only

//! Pure deployment controller decisions. Observations are supplied by the
//! bounded I/O adapter; this module neither authenticates them nor performs I/O.
use crate::frame as legacy;
use atlas_glm_pair_wire as wire;

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy)]
pub struct ObservedProcess {
    pub boot_id: [u8; 16],
    pub pid_namespace_device: u64,
    pub pid_namespace_inode: u64,
    pub guard_pid: u32,
    pub child_pid: u32,
    pub guard_start_ticks: u64,
    pub child_start_ticks: u64,
}
impl ObservedProcess {
    fn matches(&self, p: &wire::ProcessIdentity) -> bool {
        self.boot_id == p.boot_id
            && self.pid_namespace_device == p.pid_namespace_device
            && self.pid_namespace_inode == p.pid_namespace_inode
            && self.guard_pid == p.guard_pid
            && self.child_pid == p.child_pid
            && self.guard_start_ticks == p.guard_start_ticks
            && self.child_start_ticks == p.child_start_ticks
    }
}
#[derive(Clone, Copy)]
pub struct ExpectedRank {
    pub startup: wire::StartupRecord,
    pub process: ObservedProcess,
}
#[derive(Clone, Copy)]
pub struct Config {
    pub policy: wire::Policy,
    pub readiness_ms: u64,
    pub workload_ms: u64,
    pub drain_ms: u64,
    pub status_max_age_ms: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Gated,
    Starting,
    Running,
    Workload,
    Draining,
    Releasing,
    Succeeded,
    Failed,
}

pub struct State {
    config: Config,
    expected: [ExpectedRank; 2],
    phase: Phase,
    last_now: u64,
    campaign_end: u64,
    phase_end: u64,
    hello: [Option<legacy::Frame>; 2],
    reports: [Option<wire::RankRecord>; 2],
    manifest: Option<wire::Manifest>,
    sent: [bool; 2],
    status: [Option<u64>; 2],
    ordinal: [u64; 2],
    challenge: [Option<(u64, legacy::Frame)>; 2],
    last_challenge: [[u8; 32]; 2],
    receipts: [Option<wire::Quiescent>; 2],
    exited: [bool; 2],
    relay_closed: [bool; 2],
}
fn rank(rank: u8) -> Result<usize> {
    if rank < 2 {
        Ok(usize::from(rank))
    } else {
        Err("invalid rank")
    }
}
fn after(now: u64, duration: u64) -> Result<u64> {
    now.checked_add(duration).ok_or("time overflow")
}
impl State {
    pub fn new(now: u64, config: Config, expected: [ExpectedRank; 2]) -> Result<Self> {
        config.policy.validate().map_err(|_| "invalid policy")?;
        for v in [
            config.readiness_ms,
            config.workload_ms,
            config.drain_ms,
            config.status_max_age_ms,
        ] {
            if v == 0 || v > config.policy.campaign {
                return Err("invalid explicit controller bound");
            }
        }
        for r in &expected {
            r.startup.validate().map_err(|_| "invalid startup")?;
            if r.startup.policy != config.policy {
                return Err("policy mismatch");
            }
        }
        if expected[0].startup.pair_session != expected[1].startup.pair_session
            || expected[0].startup.container_id == expected[1].startup.container_id
        {
            return Err("invalid expected pair");
        }
        Ok(Self {
            config,
            expected,
            phase: Phase::Gated,
            last_now: now,
            campaign_end: after(now, config.policy.campaign)?,
            phase_end: after(now, config.policy.startup)?,
            hello: [None, None],
            reports: [None, None],
            manifest: None,
            sent: [false; 2],
            status: [None; 2],
            ordinal: [0; 2],
            challenge: [None, None],
            last_challenge: [[0; 32]; 2],
            receipts: [None; 2],
            exited: [false; 2],
            relay_closed: [false; 2],
        })
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn fail(&mut self, reason: &'static str) -> Result<()> {
        self.phase = Phase::Failed;
        Err(reason)
    }
    pub fn tick(&mut self, now: u64) -> Result<()> {
        if matches!(self.phase, Phase::Failed | Phase::Succeeded)
            || now < self.last_now
            || now >= self.campaign_end
            || now >= self.phase_end
        {
            return self.fail("terminal, late, or regressed clock");
        }
        self.last_now = now;
        Ok(())
    }
    fn op<T>(&mut self, now: u64, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.tick(now)?;
        let result = f(self);
        if result.is_err() {
            self.phase = Phase::Failed;
        }
        result
    }
    fn require(&self, phase: Phase) -> Result<()> {
        if self.phase == phase {
            Ok(())
        } else {
            Err("wrong controller phase")
        }
    }
    fn stage(&mut self, now: u64, phase: Phase, duration: u64) -> Result<()> {
        self.phase_end = after(now, duration)?.min(self.campaign_end);
        self.phase = phase;
        Ok(())
    }
    pub fn accept_hello(&mut self, now: u64, r: u8, f: legacy::Frame) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Gated)?;
            let r = rank(r)?;
            if s.hello[r].is_some()
                || f.kind != legacy::HELLO
                || f.ordinal != 0
                || [f.session, f.instance, f.challenge].contains(&[0; 32])
            {
                return Err("invalid or repeated HELLO");
            }
            s.last_challenge[r] = f.challenge;
            s.hello[r] = Some(f);
            Ok(())
        })
    }
    pub fn accept_gated(&mut self, now: u64, r: u8, report: wire::GatedReport) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Gated)?;
            let r = rank(r)?;
            let expected = &s.expected[r];
            let h = s.hello[r].as_ref().ok_or("report before HELLO")?;
            let record = report.record;
            record.validate().map_err(|_| "invalid report")?;
            let startup = &expected.startup;
            if s.reports[r].is_some()
                || report.pair_session != startup.pair_session
                || record.container_id != startup.container_id
                || record.image_digest != startup.image_digest
                || record.recipe_digest != startup.recipe_digest
                || record.server_elf_digest != startup.server_elf_digest
                || record.guard_elf_digest != startup.guard_elf_digest
                || !expected.process.matches(&record.process)
                || record.local_control_session != h.session
                || record.guard_instance != h.instance
                || record.original_startup_challenge != h.challenge
            {
                return Err("gated report observation mismatch or replay");
            }
            s.reports[r] = Some(record);
            Ok(())
        })
    }
    pub fn paired_start(&mut self, now: u64) -> Result<[wire::Frame; 2]> {
        self.op(now, |s| {
            s.require(Phase::Gated)?;
            let manifest = wire::Manifest {
                pair_session: s.expected[0].startup.pair_session,
                policy_digest: wire::policy_digest(&s.config.policy)
                    .map_err(|_| "policy digest")?,
                ranks: [
                    s.reports[0].ok_or("missing rank0 report")?,
                    s.reports[1].ok_or("missing rank1 report")?,
                ],
            };
            for r in 0..2 {
                manifest
                    .validate_local(r as u8, &s.expected[r].startup, &manifest.ranks[r])
                    .map_err(|_| "invalid observed pair")?;
            }
            s.manifest = Some(manifest);
            // Both complete local writes must finish within the original frame
            // budget; no child ACK or remote completion is inferred here.
            s.stage(now, Phase::Starting, s.config.policy.frame)?;
            Ok([0, 1].map(|rank| wire::Frame {
                rank,
                body: wire::Body::PairedStart(manifest),
            }))
        })
    }
    /// Adapter confirms a complete local write, never a remote completion ACK.
    pub fn sent_start(&mut self, now: u64, r: u8) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Starting)?;
            let r = rank(r)?;
            if s.sent[r] {
                return Err("start write replay");
            }
            s.sent[r] = true;
            if s.sent == [true; 2] {
                s.stage(now, Phase::Running, s.config.readiness_ms)?;
            }
            Ok(())
        })
    }
    pub fn observe_running(
        &mut self,
        now: u64,
        r: u8,
        container: wire::Digest,
        image: wire::Digest,
        oom: bool,
    ) -> Result<()> {
        self.op(now, |s| {
            let r = rank(r)?;
            s.identity(r, container, image)?;
            if oom || s.exited[r] {
                return Err("unhealthy container");
            }
            s.status[r] = Some(now);
            Ok(())
        })
    }
    fn identity(&self, r: usize, container: wire::Digest, image: wire::Digest) -> Result<()> {
        let e = &self.expected[r].startup;
        if container == e.container_id && image == e.image_digest {
            Ok(())
        } else {
            Err("container identity changed")
        }
    }
    fn healthy(&self, now: u64) -> Result<()> {
        for observed in self.status {
            legacy::check_deadline(
                observed.ok_or("missing health observation")?,
                now,
                self.config.status_max_age_ms,
            )?;
        }
        Ok(())
    }
    pub fn accept_challenge(&mut self, now: u64, r: u8, f: legacy::Frame) -> Result<()> {
        self.op(now, |s| {
            if s.phase == Phase::Gated {
                return Err("challenge outside active phase");
            }
            let r = rank(r)?;
            let h = s.hello[r].as_ref().ok_or("missing HELLO")?;
            if f.kind != legacy::CHALLENGE
                || f.session != h.session
                || f.instance != h.instance
                || f.ordinal != s.ordinal[r].checked_add(1).ok_or("ordinal overflow")?
                || f.challenge == [0; 32]
                || f.challenge == s.last_challenge[r]
                || s.challenge[r].is_some()
            {
                return Err("invalid challenge or replay");
            }
            s.challenge[r] = Some((now, f));
            Ok(())
        })
    }
    pub fn renew(&mut self, now: u64) -> Result<[legacy::Frame; 2]> {
        self.op(now, |s| {
            if s.phase == Phase::Gated
                || s.exited.iter().any(|v| *v)
                || s.relay_closed.iter().any(|v| *v)
            {
                return Err("renew outside active phase");
            }
            s.healthy(now)?;
            let mut replies = [
                s.challenge[0].clone().ok_or("missing rank0 challenge")?,
                s.challenge[1].clone().ok_or("missing rank1 challenge")?,
            ];
            for (r, (received, f)) in replies.iter_mut().enumerate() {
                legacy::check_deadline(*received, now, s.config.policy.frame)?;
                s.ordinal[r] = f.ordinal;
                s.last_challenge[r] = f.challenge;
                f.kind = legacy::RENEW;
            }
            s.challenge = [None, None];
            Ok(replies.map(|(_, f)| f))
        })
    }
    /// Called only after successful actual HTTP readiness, not process startup.
    pub fn mark_ready(&mut self, now: u64) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Running)?;
            s.healthy(now)?;
            s.stage(now, Phase::Workload, s.config.workload_ms)
        })
    }
    /// Successful workload completion grants only one shutdown request.
    pub fn workload_complete(&mut self, now: u64) -> Result<wire::Frame> {
        self.op(now, |s| {
            s.require(Phase::Workload)?;
            s.healthy(now)?;
            let manifest = s.manifest.as_ref().ok_or("missing manifest")?;
            let request = wire::DrainRequest {
                pair_digest: wire::manifest_digest(manifest).map_err(|_| "manifest digest")?,
                epoch: wire::DRAIN_EPOCH,
            };
            s.stage(now, Phase::Draining, s.config.drain_ms)?;
            Ok(wire::Frame {
                rank: 0,
                body: wire::Body::DrainRequest(request),
            })
        })
    }
    pub fn accept_quiescent(&mut self, now: u64, r: u8, receipt: wire::Quiescent) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Draining)?;
            let r = rank(r)?;
            if s.receipts[r].is_some() {
                return Err("receipt replay");
            }
            receipt
                .validate(s.manifest.as_ref().ok_or("missing manifest")?, r as u8)
                .map_err(|_| "foreign receipt")?;
            s.phase_end = s.phase_end.min(after(now, s.config.policy.quiescent_wait)?);
            s.receipts[r] = Some(receipt);
            Ok(())
        })
    }
    pub fn release(&mut self, now: u64) -> Result<[wire::Frame; 2]> {
        self.op(now, |s| {
            s.require(Phase::Draining)?;
            s.healthy(now)?;
            let receipts = [
                wire::QuiescentFrame {
                    rank: 0,
                    receipt: s.receipts[0].ok_or("missing rank0 receipt")?,
                },
                wire::QuiescentFrame {
                    rank: 1,
                    receipt: s.receipts[1].ok_or("missing rank1 receipt")?,
                },
            ];
            let release = wire::PairRelease {
                pair_digest: wire::manifest_digest(s.manifest.as_ref().ok_or("missing manifest")?)
                    .map_err(|_| "manifest digest")?,
                epoch: wire::DRAIN_EPOCH,
                receipt_digests: [
                    wire::quiescent_digest(&receipts[0]).map_err(|_| "receipt digest")?,
                    wire::quiescent_digest(&receipts[1]).map_err(|_| "receipt digest")?,
                ],
                receipts,
            };
            release.validate_fields().map_err(|_| "invalid release")?;
            s.sent = [false; 2];
            s.stage(now, Phase::Releasing, s.config.policy.exit)?;
            Ok([0, 1].map(|rank| wire::Frame {
                rank,
                body: wire::Body::PairRelease(release),
            }))
        })
    }
    pub fn sent_release(&mut self, now: u64, r: u8) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Releasing)?;
            let r = rank(r)?;
            if s.sent[r] {
                return Err("release write replay");
            }
            s.sent[r] = true;
            Ok(())
        })
    }
    pub fn relay_eof(&mut self, now: u64, r: u8) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Releasing)?;
            let r = rank(r)?;
            if !s.sent[r] || s.relay_closed[r] {
                return Err("EOF before release delivery");
            }
            s.relay_closed[r] = true;
            Ok(())
        })
    }
    pub fn observe_exit(
        &mut self,
        now: u64,
        r: u8,
        container: wire::Digest,
        image: wire::Digest,
        exit_code: i32,
        oom: bool,
    ) -> Result<()> {
        self.op(now, |s| {
            s.require(Phase::Releasing)?;
            let r = rank(r)?;
            s.identity(r, container, image)?;
            if !s.sent[r] || s.exited[r] || exit_code != 0 || oom {
                return Err("unhealthy or premature exit");
            }
            s.exited[r] = true;
            if s.exited == [true; 2] && s.sent == [true; 2] {
                s.phase = Phase::Succeeded;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
