// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
impl Driver<'_> {
    pub(super) fn work(&mut self) -> io::Result<()> {
        let now = self.tick()?;
        if self.state()?.phase() == state::Phase::Running {
            if self.readiness.is_none() && now >= self.next_probe {
                let ready = &self.prepared.launch.readiness;
                let seconds = format!(
                    "{}.{:03}",
                    ready.per_attempt_ms / 1000,
                    ready.per_attempt_ms % 1000
                );
                let spec = process::Spec {
                    program: ready.curl.clone(),
                    args: [
                        "--disable",
                        "--silent",
                        "--show-error",
                        "--noproxy",
                        "*",
                        "--max-time",
                        &seconds,
                        "--proto",
                        "=http,https",
                        "--write-out",
                        "\n%{http_code}",
                        "--url",
                        &ready.url,
                    ]
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                    env: vec![],
                    stdin: vec![],
                };
                self.readiness = Some(Job::spawn(spec, io_adapter::limits(self.prepared))?);
            }
            if let Some(job) = &mut self.readiness {
                if let Some(done) = job.poll_any()? {
                    self.readiness = None;
                    let ready = crate::readiness::ready(
                        done.exit.code(),
                        &done.stdout,
                        &self.prepared.launch.readiness.expected_model,
                    )?;
                    self.journal.record("http-readiness",&json!({"ready":ready,"code":done.exit.code(),"stdout":String::from_utf8_lossy(&done.stdout)}))?;
                    let now = self.tick()?;
                    if ready {
                        self.fresh_status(now)?;
                        self.pinned_workload.revalidate()?;
                        self.state()?.mark_ready(now).map_err(error)?;
                        let workload = &self.prepared.launch.workload;
                        let spec = process::Spec {
                            program: self.pinned_workload.program(),
                            args: workload.argv.iter().map(Into::into).collect(),
                            env: workload
                                .environment
                                .iter()
                                .map(|(k, v)| (k.into(), v.into()))
                                .collect(),
                            stdin: self.prepared.workload_input.clone(),
                        };
                        self.workload = Some(Job::spawn(spec, workload.limits.to_process())?);
                        self.journal.record(
                            "workload-started",
                            &json!({"input_sha256":workload.input_sha256}),
                        )?;
                    } else {
                        self.next_probe = now
                            .checked_add(self.prepared.launch.controller.status_interval_ms)
                            .ok_or_else(|| error("probe time overflow"))?;
                    }
                }
            }
        }
        if self.state()?.phase() == state::Phase::Workload {
            let job = self
                .workload
                .as_mut()
                .ok_or_else(|| error("missing workload process"))?;
            if let Some(done) = job.poll_any()? {
                self.journal.record("workload-result",&json!({"exit":done.exit.code(),"stdout":String::from_utf8_lossy(&done.stdout),"stderr":String::from_utf8_lossy(&done.stderr)}))?;
                if !done.exit.success() {
                    return Err(error("workload failed; healthy drain not authorized"));
                }
                self.workload = None;
                let now = self.tick()?;
                self.fresh_status(now)?;
                let drain = self.state()?.workload_complete(now).map_err(error)?;
                self.relays[0]
                    .as_mut()
                    .ok_or_else(|| error("missing head relay"))?
                    .queue_live(&drain, now)?;
                self.journal.record("drain-queued", &json!({"rank":0}))?;
            }
        }
        Ok(())
    }
}
