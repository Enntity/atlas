// SPDX-License-Identifier: AGPL-3.0-only
//! Startup barriers use actual command completions and original receipt clocks.
use super::*;

// Shared by the actual command path and finite-result ordering tests. The
// closures perform I/O; this helper supplies no process or Model authority.
fn pair_barrier<C>(
    context: &mut C,
    mut submit: impl FnMut(&mut C, usize) -> io::Result<()>,
    mut poll: impl FnMut(&mut C, usize) -> io::Result<Option<Vec<u8>>>,
    mut idle: impl FnMut(&mut C) -> io::Result<()>,
) -> io::Result<[Vec<u8>; 2]> {
    for rank in 0..2 {
        submit(context, rank)?;
    }
    let mut results = [None, None];
    loop {
        for (rank, result) in results.iter_mut().enumerate() {
            if result.is_none() {
                *result = poll(context, rank)?;
            }
        }
        if results.iter().all(Option::is_some) {
            return Ok(results.map(Option::unwrap));
        }
        idle(context)?;
    }
}

impl Driver<'_> {
    fn setup_tick(&mut self) -> io::Result<u64> {
        let now = self.tick()?;
        frame::check_deadline(self.started, now, self.policy.startup).map_err(error)?;
        Ok(now)
    }

    fn setup_pair(
        &mut self,
        verb: remote::Verb,
        label: &'static str,
        mut inputs: [Vec<u8>; 2],
    ) -> io::Result<[Vec<u8>; 2]> {
        pair_barrier(
            self,
            |driver, r| {
                driver.setup_tick()?;
                let id = match verb {
                    remote::Verb::Prepare | remote::Verb::Create => None,
                    _ => Some(driver.id(r)?),
                };
                driver.jobs[r] = Some(remote_job(
                    driver.prepared,
                    r,
                    verb,
                    id,
                    std::mem::take(&mut inputs[r]),
                )?);
                Ok(())
            },
            |driver, r| {
                driver.setup_tick()?;
                let job = driver.jobs[r]
                    .as_mut()
                    .ok_or_else(|| error("missing setup job"))?;
                let original_started = job.started;
                let Some(bytes) = job.poll()? else {
                    return Ok(None);
                };
                driver.jobs[r] = None;
                let value: Value = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
                if matches!(verb, remote::Verb::Create) {
                    let id = docker::parse_id(
                        value["container_id"]
                            .as_str()
                            .ok_or_else(|| error("create omitted full ID"))?,
                    )?;
                    // Retain exact cleanup authority before inspection or journal I/O.
                    driver.ids[r] = Some(id);
                    docker::inspect(
                        &driver.prepared.recipes[r],
                        &driver.prepared.launch.nodes[r].guard_container_path,
                        "runc",
                        &id,
                        docker::Stage::Created,
                        &value["inspect"],
                    )?;
                } else if matches!(verb, remote::Verb::Observe | remote::Verb::Socket) {
                    driver.last_status[r] = original_started;
                } else if value != json!({}) {
                    return Err(error("unexpected setup acknowledgement"));
                }
                driver
                    .journal
                    .record(label, &json!({"rank":r,"reply":value}))?;
                driver.setup_tick()?;
                Ok(Some(bytes))
            },
            |driver| {
                driver.setup_tick()?;
                driver.quiet_setup_relays()?;
                pause(driver.policy.poll)
            },
        )
    }

    // Once a report has been received, no further guard frame is valid until
    // START is queued. Continue servicing pipes while host observations run.
    fn quiet_setup_relays(&mut self) -> io::Result<()> {
        for r in 0..2 {
            if self.relays[r].is_none() {
                continue;
            }
            let progress = self.relays[r].as_mut().unwrap().poll()?;
            self.setup_progress(r, &progress)?;
            if progress.incoming.is_some() {
                return Err(error("unexpected additional gated frame"));
            }
        }
        Ok(())
    }

    fn setup_progress(&mut self, r: usize, progress: &relay::Progress) -> io::Result<()> {
        if !progress.stderr.is_empty() {
            self.journal.record(
                "relay-stderr",
                &json!({"rank":r,"bytes":String::from_utf8_lossy(&progress.stderr)}),
            )?;
        }
        if progress.eof_idle
            || progress.exit.is_some()
            || progress.done
            || progress.sent_live.is_some()
        {
            return Err(error(
                "relay ended or wrote unexpectedly during gated startup",
            ));
        }
        Ok(())
    }

    fn observation(&self, r: usize, bytes: &[u8]) -> io::Result<node::NodeObservation> {
        let obs: node::NodeObservation = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        self.check_memory(&obs)?;
        let pid = docker::inspect(
            &self.prepared.recipes[r],
            &self.prepared.launch.nodes[r].guard_container_path,
            "runc",
            &self.id(r)?,
            docker::Stage::Running,
            &obs.inspect,
        )?;
        if let Some(process) = &obs.process {
            if process.host_guard_pid != pid
                || process.host_child_pid <= 1
                || process.host_child_pid == pid
                || process.guard_pid != 1
                || process.child_pid <= 1
                || process.uid != 0
                || process.gid != 0
            {
                return Err(error("host process observation differs from Docker init"));
            }
        }
        Ok(obs)
    }

    pub(super) fn setup(&mut self) -> io::Result<()> {
        let mut inputs = [vec![], vec![]];
        for (r, input) in inputs.iter_mut().enumerate() {
            let p = self.prepared;
            let n = &p.launch.nodes[r];
            let recipe_hex: String = p.recipes[r]
                .encode()
                .map_err(io::Error::other)?
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            *input = serde_json::to_vec(&node::PrepareInput {
                recipe_hex,
                metadata: node::Metadata {
                    command_ms: p.launch.controller.command_ms,
                    policy: p.launch.policy.clone(),
                    relay_path: n.relay_path.clone(),
                    relay_sha256: n.relay_sha256.clone(),
                    guard_container_path: n.guard_container_path.clone(),
                },
            })
            .map_err(io::Error::other)?;
        }
        self.setup_pair(remote::Verb::Prepare, "node-prepared", inputs)?;
        self.setup_pair(remote::Verb::Create, "node-created", [vec![], vec![]])?;
        let startup = [0, 1].map(|r| {
            let recipe = &self.prepared.recipes[r];
            Ok(wire::StartupRecord {
                pair_session: self.prepared.session,
                container_id: self.id(r)?,
                image_digest: recipe.image_digest,
                recipe_digest: recipe.digest().map_err(io::Error::other)?,
                server_elf_digest: recipe.server_elf_digest,
                guard_elf_digest: recipe.guard_elf_digest,
                policy: self.policy,
            })
        });
        let [a, b]: [io::Result<wire::StartupRecord>; 2] = startup;
        let startup = [a?, b?];
        let mut inputs = [vec![], vec![]];
        for r in 0..2 {
            inputs[r] = wire::Frame {
                rank: r as u8,
                body: wire::Body::Startup(startup[r]),
            }
            .encode()
            .map_err(io::Error::other)?
            .as_slice()
            .to_vec();
        }
        self.setup_pair(remote::Verb::Seal, "node-sealed", inputs)?;
        self.setup_pair(remote::Verb::Start, "node-started", [vec![], vec![]])?;
        loop {
            let bytes = self.setup_pair(remote::Verb::Socket, "node-socket", [vec![], vec![]])?;
            let observations = [
                self.observation(0, &bytes[0])?,
                self.observation(1, &bytes[1])?,
            ];
            if observations.iter().all(|v| v.socket_ready) {
                break;
            }
            self.setup_tick()?;
            pause(self.policy.poll)?;
        }
        for r in 0..2 {
            self.setup_tick()?;
            let spec = remote_spec(
                self.prepared,
                r,
                remote::Verb::Relay,
                Some(self.id(r)?),
                vec![],
            )?;
            self.relays[r] = Some(relay::Relay::spawn(
                spec,
                r as u8,
                relay::Limits {
                    frame_ms: self.policy.frame,
                    poll_ms: self.policy.poll,
                    campaign_ms: self.policy.campaign,
                    stderr_bytes: 65536,
                    stderr_per_turn: 16384,
                    reap_ms: self.policy.reap,
                },
                self.started,
            )?);
            self.journal.record("relay-spawned", &json!({"rank":r}))?;
        }
        let mut hellos: [Option<(frame::Frame, u64)>; 2] = [None, None];
        let mut reports: [Option<(wire::GatedReport, u64)>; 2] = [None, None];
        while reports.iter().any(Option::is_none) {
            self.setup_tick()?;
            for r in 0..2 {
                let progress = self.relays[r].as_mut().unwrap().poll()?;
                self.setup_progress(r, &progress)?;
                if let Some((incoming, started)) = progress.incoming {
                    frame::check_deadline(started, boot_time_ms()?, self.policy.frame)
                        .map_err(error)?;
                    match incoming {
                        relay::Incoming::Legacy(f)
                            if f.kind == frame::HELLO
                                && hellos[r].is_none()
                                && reports[r].is_none() =>
                        {
                            self.journal.record(
                                "guard-hello",
                                &json!({"rank":r,"started":started,"frame":f.encode().as_slice()}),
                            )?;
                            hellos[r] = Some((f, started));
                        }
                        relay::Incoming::Live(wire::Frame {
                            rank,
                            body: wire::Body::GatedReport(report),
                        }) if usize::from(rank) == r
                            && hellos[r].is_some()
                            && reports[r].is_none() =>
                        {
                            reports[r] = Some((report, started));
                            let encoded = wire::Frame {
                                rank,
                                body: wire::Body::GatedReport(report),
                            }
                            .encode()
                            .map_err(io::Error::other)?;
                            self.journal.record(
                                "guard-report",
                                &json!({"rank":r,"started":started,"frame":encoded.as_slice()}),
                            )?;
                        }
                        _ => return Err(error("unexpected or duplicate startup frame")),
                    }
                }
            }
            for receipt in hellos
                .iter()
                .flatten()
                .map(|(_, n)| n)
                .chain(reports.iter().flatten().map(|(_, n)| n))
            {
                frame::check_deadline(*receipt, boot_time_ms()?, self.policy.frame)
                    .map_err(error)?;
            }
            if reports.iter().any(Option::is_none) {
                pause(self.policy.poll)?;
            }
        }
        let bytes = self.setup_pair(
            remote::Verb::Observe,
            "node-gated-observation",
            [vec![], vec![]],
        )?;
        let mut expected = [None, None];
        for r in 0..2 {
            let obs = self.observation(r, &bytes[r])?;
            if !obs.socket_ready {
                return Err(error("control socket disappeared"));
            }
            let p = obs
                .process
                .ok_or_else(|| error("gated child observation absent"))?;
            expected[r] = Some(state::ExpectedRank {
                startup: startup[r],
                process: state::ObservedProcess {
                    boot_id: p.boot_id,
                    pid_namespace_device: p.pid_namespace_device,
                    pid_namespace_inode: p.pid_namespace_inode,
                    guard_pid: p.guard_pid,
                    child_pid: p.child_pid,
                    guard_start_ticks: p.guard_start_ticks,
                    child_start_ticks: p.child_start_ticks,
                },
            });
            self.expected[r] = Some(p);
        }
        let c = &self.prepared.launch.controller;
        let status_max_age = c.status_max_age_ms;
        self.machine = Some(
            state::State::new(
                self.started,
                state::Config {
                    policy: self.policy,
                    readiness_ms: c.readiness_ms,
                    workload_ms: c.workload_ms,
                    drain_ms: c.drain_ms,
                    status_max_age_ms: c.status_max_age_ms,
                },
                [expected[0].unwrap(), expected[1].unwrap()],
            )
            .map_err(error)?,
        );
        for r in 0..2 {
            let (hello, hello_started) = hellos[r].take().unwrap();
            let (report, report_started) = reports[r].unwrap();
            let now = self.setup_tick()?;
            frame::check_deadline(hello_started, now, self.policy.frame).map_err(error)?;
            self.state()?
                .accept_hello(now, r as u8, hello)
                .map_err(error)?;
            let now = self.setup_tick()?;
            frame::check_deadline(report_started, now, self.policy.frame).map_err(error)?;
            self.state()?
                .accept_gated(now, r as u8, report)
                .map_err(error)?;
        }
        // Check health with the original command launch time, not a newly
        // minted observation age after journal/parsing work.
        for r in 0..2 {
            let now = self.tick()?;
            frame::check_deadline(self.last_status[r], now, status_max_age).map_err(error)?;
            let id = self.id(r)?;
            let image = self.prepared.recipes[r].image_digest;
            self.state()?
                .observe_running(now, r as u8, id, image, false)
                .map_err(error)?;
        }
        let oldest = reports[0].unwrap().1.min(reports[1].unwrap().1);
        let now = self.setup_tick()?;
        frame::check_deadline(oldest, now, self.policy.frame).map_err(error)?;
        let starts = self.state()?.paired_start(now).map_err(error)?;
        self.journal
            .record("paired-start-queued", &json!({"original_started":oldest}))?;
        for (r, start) in starts.iter().enumerate() {
            self.relays[r].as_mut().unwrap().queue_live(start, oldest)?;
        }
        // Do not wait for writes here: the live loop must service both leases.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Finite {
        events: Vec<(u8, usize)>,
        polls: [usize; 2],
    }
    #[test]
    fn actual_setup_barrier_waits_for_both_finite_results() {
        let mut c = Finite::default();
        for phase in [0, 4] {
            pair_barrier(
                &mut c,
                |c, r| {
                    c.events.push((phase, r));
                    Ok(())
                },
                |c, r| {
                    c.events.push((phase + 1, r));
                    c.polls[r] += 1;
                    Ok((r == 1 || c.polls[r] >= 3).then(|| vec![r as u8]))
                },
                |c| {
                    c.events.push((phase + 2, 0));
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(&c.events[..2], &[(0, 0), (0, 1)]);
        let next = c.events.iter().position(|v| v.0 == 4).unwrap();
        assert_eq!(c.events[..next].iter().filter(|v| **v == (1, 0)).count(), 3);
        assert_eq!(c.events[..next].iter().filter(|v| **v == (1, 1)).count(), 1);
        assert_eq!(&c.events[next..next + 2], &[(4, 0), (4, 1)]);
    }
    #[test]
    fn actual_setup_barrier_failure_prevents_following_phase() {
        let mut c = Finite::default();
        let result = pair_barrier(
            &mut c,
            |c, r| {
                c.events.push((0, r));
                Ok(())
            },
            |c, r| {
                c.events.push((1, r));
                Err(error("finite command failure"))
            },
            |_| Ok(()),
        );
        assert_eq!(result.unwrap_err().to_string(), "finite command failure");
        assert_eq!(c.events, vec![(0, 0), (0, 1), (1, 0)]);
    }
}
