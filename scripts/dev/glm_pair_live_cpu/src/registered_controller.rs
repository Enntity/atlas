// SPDX-License-Identifier: AGPL-3.0-only
//! CPU-only coordination/readiness fixture. It exercises the real controller,
//! inherited session and Model owner; it is not inference or NCCL evidence.
use super::selected::SelectedModel;
use anyhow::{ensure, Context, Result};
use atlas_glm_pair_io::identity::{boot_time_ms, check_deadline};
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::sync::atomic::Ordering;

struct Markers(File);
impl Markers {
    fn open() -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open("/run/atlas-cpu")?;
        let value = Self(file);
        value.check()?;
        Ok(value)
    }
    fn check(&self) -> Result<()> {
        let m = self.0.metadata()?;
        let path = std::fs::symlink_metadata("/run/atlas-cpu")?;
        ensure!(
            m.is_dir()
                && m.uid() == 0
                && m.gid() == 0
                && m.mode() & 0o7777 == 0o700
                && m.dev() == path.dev()
                && m.ino() == path.ino()
                && path.is_dir(),
            "CPU marker directory must remain pinned root0700"
        );
        Ok(())
    }
    fn file(&self, name: &CStr, create: bool) -> io::Result<File> {
        let flags = libc::O_CLOEXEC
            | libc::O_NOFOLLOW
            | libc::O_NONBLOCK
            | if create {
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL
            } else {
                libc::O_RDONLY
            };
        let fd = unsafe { libc::openat(self.0.as_raw_fd(), name.as_ptr(), flags, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_fd(fd) })
    }
    fn mark(&self, name: &CStr) -> Result<()> {
        self.check()?;
        // An empty marker is atomically visible at exclusive creation. There
        // is no partially written payload to confuse the other live process.
        let file = self.file(name, true)?;
        file.sync_all()?;
        self.0.sync_all()?;
        self.check()
    }
    fn present(&self, name: &CStr) -> Result<bool> {
        self.check()?;
        let file = match self.file(name, false) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let m = file.metadata()?;
        ensure!(
            m.is_file()
                && m.uid() == 0
                && m.gid() == 0
                && m.mode() & 0o7777 == 0o600
                && m.nlink() == 1
                && m.len() == 0,
            "invalid CPU coordination marker"
        );
        self.check()?;
        Ok(true)
    }
}

struct Request {
    stream: TcpStream,
    input: Vec<u8>,
    output: Vec<u8>,
    written: usize,
    deadline: u64,
}
impl Request {
    fn new(stream: TcpStream) -> Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            input: vec![],
            output: vec![],
            written: 0,
            deadline: boot_time_ms()?
                .checked_add(2000)
                .context("HTTP deadline overflow")?,
        })
    }
    fn step(&mut self, ready: bool) -> Result<bool> {
        check_deadline(self.deadline)?;
        if self.output.is_empty() {
            let mut bytes = [0; 1024];
            match self.stream.read(&mut bytes) {
                Ok(0) => return Ok(true),
                Ok(n) => {
                    ensure!(
                        n <= 4096 - self.input.len(),
                        "CPU HTTP header exceeds4096 bytes"
                    );
                    self.input.extend_from_slice(&bytes[..n]);
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(e.into()),
            }
            if self.input.windows(4).any(|v| v == b"\r\n\r\n") {
                let line = self.input.split(|&b| b == b'\n').next().unwrap_or_default();
                ensure!(
                    matches!(line, b"GET /health HTTP/1.1\r" | b"GET /health HTTP/1.0\r"),
                    "CPU fixture serves only GET /health"
                );
                let (status, body) = if ready {
                    (
                        "200 OK",
                        "{\"status\":\"ready\",\"model\":\"glm-pair-cpu\"}",
                    )
                } else {
                    (
                        "503 Service Unavailable",
                        "{\"status\":\"loading\",\"model\":\"glm-pair-cpu\"}",
                    )
                };
                self.output = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).into_bytes();
            }
        }
        if !self.output.is_empty() {
            match self.stream.write(&self.output[self.written..]) {
                Ok(0) => anyhow::bail!("zero CPU HTTP response write"),
                Ok(n) => self.written += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(e.into()),
            }
        }
        check_deadline(self.deadline)?;
        Ok(!self.output.is_empty() && self.written == self.output.len())
    }
}

pub(super) fn wait(owner: &SelectedModel, rank: u8) -> Result<()> {
    ensure!(rank < 2, "CPU controller rank");
    let address = std::env::var("ATLAS_PAIR_CPU_HTTP_ADDR")?;
    ensure!(
        address == "127.0.0.1:18761"
            && std::env::var("ATLAS_PAIR_CPU_HTTP_MODEL")? == "glm-pair-cpu"
            && std::env::var("ATLAS_PAIR_CPU_WAIT_MS")? == "60000",
        "explicit CPU controller fixture profile"
    );
    let deadline = boot_time_ms()?
        .checked_add(60000)
        .context("CPU wait deadline overflow")?;
    let markers = Markers::open()?;
    owner.check_health();
    markers.mark(if rank == 0 {
        c"rank0-registered"
    } else {
        c"rank1-registered"
    })?;
    let listener = if rank == 0 {
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        Some(listener)
    } else {
        None
    };
    let mut request: Option<Request> = None;
    loop {
        check_deadline(deadline)?;
        owner.check_health();
        if rank == 0 && super::DRAIN_SIGNAL.load(Ordering::Acquire) {
            ensure!(
                markers.present(c"rank1-registered")?,
                "drain before both CPU registrations"
            );
            markers.mark(c"head-drain")?;
            return Ok(());
        }
        if rank == 1 && markers.present(c"head-drain")? {
            return Ok(());
        }
        if let Some(listener) = &listener {
            if request.is_none() {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        ensure!(peer.ip().is_loopback(), "CPU HTTP client must be loopback");
                        request = Some(Request::new(stream)?);
                    }
                    Err(e)
                        if matches!(
                            e.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                        ) => {}
                    Err(e) => return Err(e.into()),
                }
            }
            if let Some(active) = &mut request {
                // No detached health thread; real owner health precedes every
                // bounded response step on this same fixture control thread.
                let ready = markers.present(c"rank1-registered")?;
                if active.step(ready)? {
                    request = None;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
