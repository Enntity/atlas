// SPDX-License-Identifier: AGPL-3.0-only
//! One bounded stream/read/write turn; no internal wait or remote success rule.
use super::*;

impl Relay {
    pub(super) fn poll_inner(&mut self) -> io::Result<Progress> {
        self.tick()?;
        self.observe_exit()?;
        if self.exit.is_some() && self.output.pending() {
            return Err(error("relay exited with pending writes"));
        }
        let (mut incoming, mut eof_idle) = (None, false);
        if let Some(stdout) = &self.stdout {
            let fd = stdout.as_raw_fd();
            let began = self.tick()?;
            let read = self.reader.read(fd, began);
            let observed = self.tick()?;
            match read {
                Ok(Some((packet, original))) => {
                    frame::check_deadline(original, observed, self.limits.frame_ms)
                        .map_err(error)?;
                    match &packet {
                        Incoming::Legacy(value)
                            if !matches!(value.kind, frame::HELLO | frame::CHALLENGE) =>
                        {
                            return Err(error("relay received invalid legacy direction"))
                        }
                        Incoming::Live(value) if value.rank != self.rank => {
                            return Err(error("relay received foreign rank"))
                        }
                        _ => {}
                    }
                    incoming = Some((packet, original));
                }
                Ok(None) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof && self.reader.idle() => {
                    if self.output.pending() {
                        return Err(error("relay EOF with pending writes"));
                    }
                    self.stdout.take();
                    eof_idle = true;
                }
                Err(e) => return Err(e),
            }
        }
        let stderr = self.stderr_delta()?;
        self.tick()?;
        self.observe_exit()?;
        // try_wait does not discard already-buffered stdout frames. Only a
        // subsequent actual EOF can distinguish a complete stream from truncation.
        if self.exit.is_some() && self.output.pending() {
            return Err(error("relay exited with pending writes"));
        }
        let mut sent_live = None;
        if self.output.pending() {
            let fd = self
                .stdin
                .as_ref()
                .ok_or_else(|| error("missing relay stdin"))?
                .as_raw_fd();
            self.tick()?;
            self.output.send(fd, self.limits.frame_ms)?;
            let completed = self.tick()?;
            if !self.output.live_pending() {
                if let Some((kind, started)) = self.live_kind.take() {
                    sent_live = Some(Written {
                        kind,
                        started,
                        completed,
                    });
                }
            }
        }
        self.observe_exit()?;
        let accepted = self.tick()?;
        if self.exit.is_some() && self.output.pending() {
            return Err(error("relay exited with pending writes"));
        }
        let exit = if self.stdout.is_none() && !self.exit_reported {
            if self.exit.is_some() {
                self.exit_reported = true;
                self.stdin.take();
            }
            self.exit
        } else {
            None
        };
        if let Some((_, original)) = &incoming {
            // Preserve the first-read clock through all I/O and final status
            // observation before publication to the controller's State.
            frame::check_deadline(*original, accepted, self.limits.frame_ms).map_err(error)?;
        }
        Ok(Progress {
            incoming,
            sent_live,
            eof_idle,
            exit,
            stderr,
            done: self.exit.is_some() && self.stdout.is_none() && self.stderr.is_none(),
        })
    }

    fn stderr_delta(&mut self) -> io::Result<Vec<u8>> {
        let Some(stderr) = &self.stderr else {
            return Ok(Vec::new());
        };
        let remaining = self
            .limits
            .stderr_bytes
            .checked_sub(self.stderr_count)
            .ok_or_else(|| error("relay stderr counter exceeds cap"))?;
        // Read at most one byte past the total cap to detect, not truncate,
        // overflow; this one read is also limited by the explicit turn bound.
        let mut bytes = vec![0; (remaining + 1).min(self.limits.stderr_per_turn)];
        let read = Io::read(stderr.as_raw_fd(), &mut bytes)?;
        self.tick()?;
        match read {
            None => bytes.clear(),
            Some(0) => {
                self.stderr.take();
                bytes.clear();
            }
            Some(n) => {
                if n > remaining {
                    return Err(error("relay stderr exceeded explicit cap"));
                }
                self.stderr_count += n;
                bytes.truncate(n);
            }
        }
        Ok(bytes)
    }
}
