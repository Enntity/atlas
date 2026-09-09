// SPDX-License-Identifier: AGPL-3.0-only
//! Connected one-shot driver; only observed exits complete a campaign.
use crate::{docker, error, frame, node, prepared, process, relay, remote, state, wire};
use atlas_glm_pair_io::identity::boot_time_ms;
use serde_json::{json, Value};
use std::io;

#[path = "run_io.rs"]
mod io_adapter;
use io_adapter::{pause, remote_job, remote_spec, Job, Journal};
#[path = "run_loop.rs"]
mod serving;
#[path = "run_setup.rs"]
mod setup;
#[path = "run_workload.rs"]
mod workload;

struct Driver<'a> {
    prepared: &'a prepared::Prepared,
    policy: wire::Policy,
    started: u64,
    ids: [Option<wire::Digest>; 2],
    relays: [Option<relay::Relay>; 2],
    machine: Option<state::State>,
    expected: [Option<node::NodeProcess>; 2],
    journal: Journal<'a>,
    jobs: [Option<Job>; 2],
    last_status: [u64; 2],
    eof: [bool; 2],
    exited: [bool; 2],
    release_written: [bool; 2],
    challenge_started: [Option<u64>; 2],
    quiescent: [bool; 2],
    workload: Option<Job>,
    readiness: Option<Job>,
    next_probe: u64,
    pinned_workload: io_adapter::WorkloadExecutable,
}
impl<'a> Driver<'a> {
    fn tick(&mut self) -> io::Result<u64> {
        let now = boot_time_ms()?;
        frame::check_deadline(self.started, now, self.policy.campaign).map_err(error)?;
        if let Some(machine) = &mut self.machine {
            machine.tick(now).map_err(error)?;
        }
        Ok(now)
    }
    fn state(&mut self) -> io::Result<&mut state::State> {
        self.machine
            .as_mut()
            .ok_or_else(|| error("paired controller not initialized"))
    }
    fn id(&self, r: usize) -> io::Result<wire::Digest> {
        self.ids[r].ok_or_else(|| error("missing recorded container ID"))
    }
    fn check_memory(&self, obs: &node::NodeObservation) -> io::Result<()> {
        // Fixed selected-deployment floor, matching the minimum server OOM guard.
        if obs.swap_used_kib != 0 || obs.mem_available_kib < 4 * 1024 * 1024 {
            return Err(error("node swap use or less than4GiB available headroom"));
        }
        Ok(())
    }
    fn cleanup(&mut self) {
        let began = boot_time_ms();
        if let Some(machine) = &mut self.machine {
            let _ = machine.fail("latched driver failure");
        }
        // Keep each exact local child owner until try_wait confirms exit or the
        // explicit cleanup bound expires. Abort does not itself prove reaping.
        let mut retained = [
            self.jobs[0].take(),
            self.jobs[1].take(),
            self.workload.take(),
            self.readiness.take(),
        ];
        for (index, job) in retained.iter_mut().enumerate() {
            if let Some(job) = job {
                if let Err(e) = job.process.abort() {
                    let _ = self.journal.record(
                        "cleanup-local-abort-failed",
                        &json!({"owner":index,"error":e.to_string()}),
                    );
                }
            }
        }
        for (rank, relay) in self.relays.iter_mut().enumerate() {
            if let Some(relay) = relay {
                if let Err(e) = relay.abort() {
                    let _ = self.journal.record(
                        "cleanup-relay-abort-failed",
                        &json!({"rank":rank,"error":e.to_string()}),
                    );
                }
            }
        }
        let began = match began {
            Ok(now) => now,
            Err(e) => {
                let _ = self.journal.record("cleanup-unconfirmed-clock",
                    &json!({"error":e.to_string(),"local_exit_confirmed":false,"remote_exit_confirmed":false}));
                return;
            }
        };
        let bound = self.prepared.launch.controller.cleanup_ms;
        let mut kills: [Option<Job>; 2] = [None, None];
        let mut kill_failed = [false; 2];
        let mut reap_error = [false; 6];
        for (rank, slot) in kills.iter_mut().enumerate() {
            if !boot_time_ms().is_ok_and(|now| frame::check_deadline(began, now, bound).is_ok()) {
                break;
            }
            if let Some(id) = self.ids[rank] {
                match remote_job(self.prepared, rank, remote::Verb::Kill, Some(id), vec![]) {
                    Ok(job) => *slot = Some(job),
                    Err(e) => {
                        let _ = self.journal.record(
                            "cleanup-spawn-failed",
                            &json!({"rank":rank,"error":e.to_string()}),
                        );
                    }
                }
            }
        }
        loop {
            let Ok(now) = boot_time_ms() else {
                break;
            };
            if frame::check_deadline(began, now, bound).is_err() {
                break;
            }
            for (rank, slot) in kills.iter_mut().enumerate() {
                let Some(job) = slot else {
                    continue;
                };
                if kill_failed[rank] {
                    if let Ok(Some(status)) = job.process.reap() {
                        let _ = self.journal.record(
                            "cleanup-kill-child-reaped",
                            &json!({"rank":rank,"status":format!("{status:?}")}),
                        );
                        *slot = None;
                    }
                    continue;
                }
                match job.poll() {
                    Ok(None) => {}
                    Ok(Some(bytes)) => {
                        let _ = self.journal.record(
                            "cleanup-result",
                            &json!({"rank":rank,"reply":String::from_utf8_lossy(&bytes)}),
                        );
                        *slot = None; // poll success includes actual reaped exit.
                    }
                    Err(e) => {
                        let _ = self.journal.record(
                            "cleanup-result",
                            &json!({"rank":rank,"error":e.to_string()}),
                        );
                        kill_failed[rank] = true;
                        if let Err(e) = job.process.abort() {
                            let _ = self.journal.record(
                                "cleanup-kill-abort-failed",
                                &json!({"rank":rank,"error":e.to_string()}),
                            );
                        }
                        // Keep the failed kill-command owner for later reaping.
                    }
                }
            }
            for (index, slot) in retained.iter_mut().enumerate() {
                let Some(job) = slot else {
                    continue;
                };
                match job.process.reap() {
                    Ok(Some(status)) => {
                        let _ = self.journal.record(
                            "cleanup-local-reaped",
                            &json!({"owner":index,"status":format!("{status:?}")}),
                        );
                        *slot = None;
                    }
                    Err(e) if !reap_error[index] => {
                        reap_error[index] = true;
                        let _ = self.journal.record(
                            "cleanup-local-reap-failed",
                            &json!({"owner":index,"error":e.to_string()}),
                        );
                    }
                    _ => {}
                }
            }
            for (rank, slot) in self.relays.iter_mut().enumerate() {
                let Some(relay) = slot else {
                    continue;
                };
                match relay.reap() {
                    Ok(Some(status)) => {
                        let _ = self.journal.record(
                            "cleanup-relay-reaped",
                            &json!({"rank":rank,"status":format!("{status:?}")}),
                        );
                        *slot = None;
                    }
                    Err(e) if !reap_error[4 + rank] => {
                        reap_error[4 + rank] = true;
                        let _ = self.journal.record(
                            "cleanup-relay-reap-failed",
                            &json!({"rank":rank,"error":e.to_string()}),
                        );
                    }
                    _ => {}
                }
            }
            if kills.iter().all(Option::is_none)
                && retained.iter().all(Option::is_none)
                && self.relays.iter().all(Option::is_none)
            {
                break;
            }
            let Ok(now) = boot_time_ms() else {
                break;
            };
            let Some(remaining) = now
                .checked_sub(began)
                .and_then(|elapsed| bound.checked_sub(elapsed))
            else {
                break;
            };
            if pause(self.policy.poll.min(remaining)).is_err() {
                break;
            }
        }
        for job in kills.iter_mut().flatten() {
            let _ = job.process.abort();
        }
        let _ = self.journal.record(
            "cleanup-finished",
            &json!({
                "unconfirmed_local_jobs":retained.iter().map(Option::is_some).collect::<Vec<_>>(),
                "unconfirmed_kill_children":kills.iter().map(Option::is_some).collect::<Vec<_>>(),
                "unconfirmed_relays":self.relays.iter().map(Option::is_some).collect::<Vec<_>>(),
                "remote_exit_confirmed":false
            }),
        );
        // Bound exhaustion is recorded as uncertainty, not successful cleanup.
        // Remote resources remain retained; no group kill or force-removal.
    }
}

pub fn execute(p: &prepared::Prepared) -> io::Result<()> {
    let policy = p.launch.policy.to_wire()?;
    let pinned_workload = io_adapter::pin_workload(p)?;
    p.consume()?;
    let mut driver = Driver {
        prepared: p,
        policy,
        started: boot_time_ms()?,
        ids: [None, None],
        relays: [None, None],
        machine: None,
        expected: [None, None],
        journal: Journal::new(p),
        jobs: [None, None],
        last_status: [0; 2],
        eof: [false; 2],
        exited: [false; 2],
        release_written: [false; 2],
        challenge_started: [None, None],
        quiescent: [false; 2],
        workload: None,
        readiness: None,
        next_probe: 0,
        pinned_workload,
    };
    let result = (|| {
        driver.setup()?;
        driver.serve()
    })();
    if let Err(e) = &result {
        let _ = driver
            .journal
            .record("failed", &json!({"error":e.to_string()}));
        driver.cleanup();
    }
    result
}
