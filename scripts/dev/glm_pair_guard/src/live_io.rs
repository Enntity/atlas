// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed bounded stream I/O. No resynchronization or partial interleave.
use super::*;

#[allow(clippy::large_enum_variant)] // Fixed <=832-byte storage avoids a watchdog heap allocation.
pub(super) enum Incoming {
    Legacy(frame::Frame),
    Live(wire::Frame),
}
pub(super) struct Reader {
    bytes: [u8; wire::MAX_ENCODED],
    used: usize,
    target: usize,
    started: Option<u64>,
    direction: wire::Direction,
}
impl Reader {
    pub(super) fn new() -> Self {
        Self::directional(wire::Direction::ControllerToGuard)
    }
    pub(super) fn directional(direction: wire::Direction) -> Self {
        Self {
            bytes: [0; wire::MAX_ENCODED],
            used: 0,
            target: 4,
            started: None,
            direction,
        }
    }
    pub(super) fn idle(&self) -> bool {
        self.used == 0
    }
    pub(super) fn check(&self, now: u64, limit: u64) -> io::Result<()> {
        if let Some(started) = self.started {
            frame::check_deadline(started, now, limit).map_err(error)?;
        }
        Ok(())
    }
    pub(super) fn read(&mut self, fd: i32, now: u64) -> io::Result<Option<(Incoming, u64)>> {
        match Io::read(fd, &mut self.bytes[self.used..self.target])? {
            Some(0) => return Err(error("LIVE control EOF")),
            None => return Ok(None),
            Some(n) => {
                self.started.get_or_insert(now);
                self.used += n;
            }
        }
        if self.used == 4 && self.target == 4 {
            self.target =
                wire::control_frame_len(self.bytes[..4].try_into().unwrap(), self.direction)
                    .map_err(wire_error)?;
        }
        if self.used != self.target {
            return Ok(None);
        }
        let decoded = if self.target == frame::LEN {
            Incoming::Legacy(
                frame::Frame::decode(self.bytes[..frame::LEN].try_into().unwrap())
                    .map_err(error)?,
            )
        } else {
            Incoming::Live(
                wire::Frame::decode(&self.bytes[..self.target], self.direction)
                    .map_err(wire_error)?,
            )
        };
        let started = self
            .started
            .take()
            .ok_or_else(|| error("missing input clock"))?;
        self.used = 0;
        self.target = 4;
        Ok(Some((decoded, started)))
    }
}

pub(super) struct Output {
    bytes: [u8; wire::MAX_ENCODED],
    len: usize,
    sent: usize,
    started: u64,
}
impl Output {
    pub(super) fn new(bytes: &[u8], now: u64) -> io::Result<Self> {
        if bytes.is_empty() || bytes.len() > wire::MAX_ENCODED {
            return Err(error("LIVE output bound"));
        }
        let mut output = Self {
            bytes: [0; wire::MAX_ENCODED],
            len: bytes.len(),
            sent: 0,
            started: now,
        };
        output.bytes[..bytes.len()].copy_from_slice(bytes);
        Ok(output)
    }
    pub(super) fn check(&self, now: u64, limit: u64) -> io::Result<()> {
        frame::check_deadline(self.started, now, limit).map_err(error)
    }
    pub(super) fn packet(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}
pub(super) struct Outputs {
    lease: Option<Output>,
    live: Option<Output>,
}
impl Outputs {
    pub(super) fn new() -> Self {
        Self {
            lease: None,
            live: None,
        }
    }
    pub(super) fn pending(&self) -> bool {
        self.lease.is_some() || self.live.is_some()
    }
    pub(super) fn live_pending(&self) -> bool {
        self.live.is_some()
    }
    pub(super) fn lease(&mut self, output: Output) -> io::Result<()> {
        if self.lease.is_some() {
            return Err(error("LIVE lease output slot occupied"));
        }
        self.lease = Some(output);
        Ok(())
    }
    pub(super) fn live(&mut self, output: Output) -> io::Result<()> {
        if self.live.is_some() {
            return Err(error("LIVE one-shot output slot occupied"));
        }
        self.live = Some(output);
        Ok(())
    }
    pub(super) fn check(&self, now: u64, limit: u64) -> io::Result<()> {
        for output in [&self.lease, &self.live].into_iter().flatten() {
            output.check(now, limit)?;
        }
        Ok(())
    }
    pub(super) fn send(&mut self, fd: i32, limit: u64) -> io::Result<()> {
        // A partially sent LIVE frame finishes first; otherwise lease wins.
        let slot = if self.live.as_ref().is_some_and(|o| o.sent != 0) {
            &mut self.live
        } else if self.lease.is_some() {
            &mut self.lease
        } else {
            &mut self.live
        };
        let Some(output) = slot.as_mut() else {
            return Ok(());
        };
        output.check(Io::now()?, limit)?;
        let written = Io::write(fd, &output.bytes[output.sent..output.len])?;
        output.check(Io::now()?, limit)?;
        if let Some(n) = written {
            if n == 0 {
                return Err(error("LIVE zero output write"));
            }
            output.sent += n;
            if output.sent == output.len {
                *slot = None;
            }
        }
        Ok(())
    }
}

pub(super) fn signals(fd: i32) -> io::Result<()> {
    // All four blocked signal kinds fit in one bounded read. SIGCHLD merely
    // asks the loop to inspect its held pidfd; it is not itself a failure.
    let mut entries = unsafe { std::mem::zeroed::<[libc::signalfd_siginfo; 4]>() };
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            entries.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(&entries),
        )
    };
    if let Some(n) = Io::read(fd, bytes)? {
        let size = std::mem::size_of::<libc::signalfd_siginfo>();
        if n == 0 || n % size != 0 {
            return Err(error("LIVE invalid signal read"));
        }
        for entry in &entries[..n / size] {
            if entry.ssi_signo != libc::SIGCHLD as u32 {
                return Err(error("LIVE explicit terminal signal"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "live_io_tests.rs"]
mod tests;
