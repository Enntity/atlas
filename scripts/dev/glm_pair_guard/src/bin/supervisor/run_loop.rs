// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
impl Driver<'_> {
    pub(super) fn serve(&mut self) -> io::Result<()> {
        loop {
            self.tick()?;
            self.pump_relays()?;
            self.observe_nodes()?;
            if self.state()?.phase() == state::Phase::Succeeded {
                self.journal
                    .record("paired-exit-zero", &json!({"exited":self.exited}))?;
                return self.finish_relays();
            }
            if self.challenge_started.iter().all(Option::is_some)
                && !self.eof.iter().any(|v| *v)
                && !self.exited.iter().any(|v| *v)
                && self.last_status.iter().all(|v| *v != 0)
            {
                let now = self.tick()?;
                self.fresh_status(now)?;
                let renew = self.state()?.renew(now).map_err(error)?;
                for (r, value) in renew.iter().enumerate() {
                    let began = self.challenge_started[r]
                        .take()
                        .ok_or_else(|| error("missing challenge clock"))?;
                    self.relays[r]
                        .as_mut()
                        .ok_or_else(|| error("missing relay"))?
                        .queue_lease(value, began)?;
                }
                self.journal.record(
                    "paired-renew-queued",
                    &json!({"ordinals":renew.each_ref().map(|f|f.ordinal)}),
                )?;
            }
            self.work()?;
            if self.quiescent == [true; 2] && self.state()?.phase() == state::Phase::Draining {
                let now = self.tick()?;
                self.fresh_status(now)?;
                let release = self.state()?.release(now).map_err(error)?;
                for (r, value) in release.iter().enumerate() {
                    self.relays[r]
                        .as_mut()
                        .ok_or_else(|| error("missing relay"))?
                        .queue_live(value, now)?;
                }
                self.journal
                    .record("release-queued", &json!({"ranks":[0,1]}))?;
            }
            pause(self.policy.poll)?;
        }
    }
    pub(super) fn fresh_status(&self, now: u64) -> io::Result<()> {
        for began in self.last_status {
            if began == 0 {
                return Err(error("missing original health observation clock"));
            }
            frame::check_deadline(
                began,
                now,
                self.prepared.launch.controller.status_max_age_ms,
            )
            .map_err(error)?;
        }
        Ok(())
    }
    fn finish_relays(&mut self) -> io::Result<()> {
        let began = boot_time_ms()?;
        loop {
            let now = boot_time_ms()?;
            frame::check_deadline(began, now, self.policy.reap).map_err(error)?;
            for r in 0..2 {
                let Some(relay) = self.relays[r].as_mut() else {
                    continue;
                };
                // Actual remote exits were already independently certified.
                // Drain the retained transport without granting more leases.
                let progress = relay.poll()?;
                self.journal.record("post-exit-transport",&json!({"rank":r,"stderr":String::from_utf8_lossy(&progress.stderr),
                    "stdout_eof":progress.eof_idle,"local_exit":progress.exit.map(|s|s.code()),"done":progress.done}))?;
                if progress.sent_live.is_some() {
                    return Err(error("pending output after certified exits"));
                }
                if let Some((incoming, began)) = progress.incoming {
                    frame::check_deadline(began, boot_time_ms()?, self.policy.frame)
                        .map_err(error)?;
                    match incoming {
                        relay::Incoming::Legacy(value) if value.kind == frame::CHALLENGE => {
                            self.journal.record(
                                "unrenewed-post-exit-challenge",
                                &json!({"rank":r,"frame":value.encode().as_slice()}),
                            )?;
                        }
                        _ => return Err(error("unexpected post-exit live frame")),
                    }
                }
                if progress.done {
                    self.relays[r] = None;
                }
            }
            if self.relays.iter().all(Option::is_none) {
                return Ok(());
            }
            pause(self.policy.poll)?;
        }
    }
    fn pump_relays(&mut self) -> io::Result<()> {
        for r in 0..2 {
            if self.relays[r].is_none() {
                continue;
            }
            let progress = self.relays[r]
                .as_mut()
                .ok_or_else(|| error("missing active relay"))?
                .poll()?;
            if !progress.stderr.is_empty() {
                self.journal.record(
                    "relay-stderr",
                    &json!({"rank":r,"text":String::from_utf8_lossy(&progress.stderr)}),
                )?;
            }
            if let Some(written) = progress.sent_live {
                let now = self.tick()?;
                frame::check_deadline(written.started, now, self.policy.frame).map_err(error)?;
                match written.kind {
                    0x11 => self.state()?.sent_start(now, r as u8).map_err(error)?,
                    0x15 => {
                        self.state()?.sent_release(now, r as u8).map_err(error)?;
                        self.release_written[r] = true;
                    }
                    0x16 => {}
                    _ => return Err(error("unexpected live output completion")),
                }
                self.journal.record("local-write-complete",&json!({"rank":r,"kind":written.kind,"began":written.started,"completed":written.completed}))?;
            }
            if let Some((incoming, began)) = progress.incoming {
                let now = self.tick()?;
                frame::check_deadline(began, now, self.policy.frame).map_err(error)?;
                match incoming {
                    relay::Incoming::Legacy(value) => {
                        self.state()?
                            .accept_challenge(now, r as u8, value)
                            .map_err(error)?;
                        self.challenge_started[r] = Some(began);
                    }
                    relay::Incoming::Live(value) => {
                        if value.rank != r as u8 {
                            return Err(error("relay rank mismatch"));
                        }
                        let wire::Body::Quiescent(receipt) = value.body else {
                            return Err(error("unexpected running live frame"));
                        };
                        self.state()?
                            .accept_quiescent(now, r as u8, receipt)
                            .map_err(error)?;
                        self.quiescent[r] = true;
                        self.journal.record("quiescent",&json!({"rank":r,"frame":value.encode().map_err(io::Error::other)?.as_slice()}))?;
                    }
                }
            }
            if progress.eof_idle {
                let now = self.tick()?;
                self.state()?.relay_eof(now, r as u8).map_err(error)?;
                self.eof[r] = true;
                self.journal
                    .record("relay-eof", &json!({"rank":r,"remote_completion":false}))?;
            }
            if let Some(exit) = progress.exit {
                if !self.eof[r] {
                    return Err(error("relay exit before authorized EOF"));
                }
                self.journal
                    .record("local-relay-exit", &json!({"rank":r,"code":exit.code()}))?;
            }
            if progress.done {
                self.relays[r] = None;
            }
        }
        Ok(())
    }
    fn observe_nodes(&mut self) -> io::Result<()> {
        for r in 0..2 {
            if self.exited[r] {
                continue;
            }
            let now = self.tick()?;
            if self.jobs[r].is_none()
                && now.saturating_sub(self.last_status[r])
                    >= self.prepared.launch.controller.status_interval_ms
            {
                self.jobs[r] = Some(remote_job(
                    self.prepared,
                    r,
                    remote::Verb::Observe,
                    Some(self.id(r)?),
                    vec![],
                )?);
            }
            let Some(job) = &mut self.jobs[r] else {
                continue;
            };
            let began = job.started;
            let Some(bytes) = job.poll()? else { continue };
            self.jobs[r] = None;
            let obs: node::NodeObservation = serde_json::from_slice(&bytes)?;
            self.check_memory(&obs)?;
            let recipe = &self.prepared.recipes[r];
            let id = self.id(r)?;
            let image = recipe.image_digest;
            let stage = match obs.inspect.pointer("/State/Status").and_then(Value::as_str) {
                Some("running") => docker::Stage::Running,
                Some("exited") => docker::Stage::Exited,
                _ => return Err(error("unexpected observed Docker state")),
            };
            let pid = docker::inspect(
                recipe,
                &self.prepared.launch.nodes[r].guard_container_path,
                "runc",
                &id,
                stage,
                &obs.inspect,
            )?;
            self.journal.record(
                "node-observation",
                &json!({"rank":r,"began":began,"observation":obs}),
            )?;
            let now = self.tick()?;
            frame::check_deadline(
                began,
                now,
                self.prepared.launch.controller.status_max_age_ms,
            )
            .map_err(error)?;
            match stage {
                docker::Stage::Running => {
                    let expected = self.expected[r]
                        .as_ref()
                        .ok_or_else(|| error("missing pinned host process"))?;
                    if pid != expected.host_guard_pid {
                        return Err(error("host guard changed"));
                    }
                    match obs.process.as_ref() {
                        Some(actual) if actual == expected => {
                            self.state()?
                                .observe_running(now, r as u8, id, image, false)
                                .map_err(error)?;
                            self.last_status[r] = began;
                        }
                        None if self.release_written[r] => {
                            self.last_status[r] = 0;
                        }
                        _ => {
                            return Err(error(
                                "active server process identity changed or disappeared",
                            ))
                        }
                    }
                }
                docker::Stage::Exited => {
                    if !self.release_written[r] || obs.process.is_some() {
                        return Err(error("premature observed guard exit"));
                    }
                    self.state()?
                        .observe_exit(now, r as u8, id, image, 0, false)
                        .map_err(error)?;
                    self.exited[r] = true;
                }
                docker::Stage::Created => unreachable!(),
            }
        }
        Ok(())
    }
}
