// SPDX-License-Identifier: AGPL-3.0-only

//! Actual local subprocesses only; no Docker identity, T2 or GPU proof.
#[allow(dead_code)] // Reuse the exact wire codec; Reader is exercised by lease.rs.
#[path = "../src/frame.rs"]
mod frame;
use frame::{Frame, LEN, RENEW, START};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Run {
    guard: Child,
    control: UnixStream,
    dir: PathBuf,
    ready: PathBuf,
    witness: PathBuf,
    hello: Frame,
    child_fd: Option<OwnedFd>,
}
fn wait_until(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() {
        assert!(Instant::now() < deadline, "bounded subprocess deadline");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn random_directory() -> PathBuf {
    let mut template = b"/tmp/atlas-guard-cpu-XXXXXX\0".to_vec();
    let p = unsafe { libc::mkdtemp(template.as_mut_ptr().cast()) };
    assert!(!p.is_null(), "mkdtemp: {}", std::io::Error::last_os_error());
    PathBuf::from(unsafe { std::ffi::CStr::from_ptr(p) }.to_str().unwrap())
}
impl Run {
    fn new(mode: &str) -> Self {
        Self::with_executable(
            mode,
            std::path::Path::new(env!("CARGO_BIN_EXE_probe_child")),
        )
    }
    fn with_executable(mode: &str, executable: &std::path::Path) -> Self {
        let dir = random_directory();
        let ready = dir.join("ready");
        let witness = dir.join("cleanup");
        let (mut control, inherited) = UnixStream::pair().unwrap();
        control
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        control
            .set_write_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let fd = inherited.as_raw_fd();
        let mut command = Command::new(env!("CARGO_BIN_EXE_glm-pair-guard"));
        command
            .args(["3", "1000", "1200", "200", "300", "5000", "10", "500"])
            .arg(executable)
            .arg(&ready)
            .arg(&witness)
            .args([mode, "cpu-only"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut guard = command.spawn().unwrap();
        drop(inherited);
        let mut bytes = [0; LEN];
        if let Err(e) = control.read_exact(&mut bytes) {
            let status = guard.try_wait().unwrap();
            let _ = guard.kill();
            let _ = guard.wait();
            panic!("actual guard setup failed, not a behavioral pass: {e}; {status:?}");
        }
        let hello = Frame::decode(&bytes).unwrap();
        assert_eq!(hello.kind, frame::HELLO);
        Self {
            guard,
            control,
            dir,
            ready,
            witness,
            hello,
            child_fd: None,
        }
    }
    fn send(&mut self, frame: &Frame) {
        self.control.write_all(&frame.encode()).unwrap();
    }
    fn start(&mut self) {
        let mut start = self.hello.clone();
        start.kind = START;
        self.send(&start);
        let mut pid = None;
        wait_until(|| {
            pid = std::fs::read_to_string(&self.ready)
                .ok()
                .and_then(|s| s.parse::<i32>().ok());
            pid.is_some()
        });
        let pid = pid.unwrap();
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        assert!(
            fd >= 0,
            "actual child pidfd unavailable: {}",
            std::io::Error::last_os_error()
        );
        self.child_fd = Some(unsafe { OwnedFd::from_raw_fd(fd) });
    }
    fn frame(&mut self) -> Frame {
        let mut bytes = [0; LEN];
        self.control.read_exact(&mut bytes).unwrap();
        Frame::decode(&bytes).unwrap()
    }
    fn exited(&mut self) {
        wait_until(|| self.guard.try_wait().unwrap().is_some());
        assert_eq!(self.guard.wait().unwrap().code(), Some(74));
    }
    fn child_dead(&self) {
        let fd = self.child_fd.as_ref().unwrap().as_raw_fd();
        wait_until(|| {
            let mut poll = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            assert!(unsafe { libc::poll(&mut poll, 1, 0) } >= 0);
            poll.revents & libc::POLLIN != 0
        });
    }
    fn observe_gated_child(&mut self) {
        // Kernel-owned relationship of this exact std::process::Child, not a
        // user-supplied PID or broad process-name discovery. HELLO is post-ready.
        let guard = self.guard.id();
        let path = format!("/proc/{guard}/task/{guard}/children");
        let children = std::fs::read_to_string(path).unwrap();
        let ids: Vec<i32> = children
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(ids.len(), 1);
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, ids[0], 0) } as i32;
        assert!(
            fd >= 0,
            "gated child pidfd: {}",
            std::io::Error::last_os_error()
        );
        self.child_fd = Some(unsafe { OwnedFd::from_raw_fd(fd) });
    }
    fn no_cleanup(&self) {
        assert!(
            !self.witness.with_extension("drop").exists(),
            "forbidden child Drop ran"
        );
        assert!(
            !self.witness.with_extension("atexit").exists(),
            "forbidden child atexit ran"
        );
    }
}
impl Drop for Run {
    fn drop(&mut self) {
        let _ = self.guard.kill();
        let _ = self.guard.wait();
        if let Some(fd) = &self.child_fd {
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    libc::SIGKILL,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                );
            }
        }
        // This exact mkdtemp directory contains only our CPU witness files.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn gate_prevents_exec_until_actual_start_then_expiry_has_no_cleanup() {
    let mut run = Run::new("wait");
    std::thread::sleep(Duration::from_millis(80));
    assert!(!run.ready.exists(), "child escaped pre-exec gate");
    run.start();
    run.exited();
    run.child_dead();
    run.no_cleanup();
}

#[test]
fn renewal_uses_real_challenge_then_loss_kills_only_owned_child() {
    let decoy_dir = random_directory();
    let mut decoy = Command::new(env!("CARGO_BIN_EXE_probe_child"))
        .arg(decoy_dir.join("ready"))
        .arg(decoy_dir.join("cleanup"))
        .args(["wait", "cpu-only"])
        .spawn()
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut run = Run::new("wait");
        run.start();
        let mut challenge = run.frame();
        assert_eq!(challenge.kind, frame::CHALLENGE);
        challenge.kind = RENEW;
        run.send(&challenge);
        let next = run.frame();
        assert_eq!(next.ordinal, challenge.ordinal + 1);
        assert_ne!(next.challenge, challenge.challenge);
        assert!(run.guard.try_wait().unwrap().is_none());
        run.exited();
        run.child_dead();
        assert!(
            decoy.try_wait().unwrap().is_none(),
            "unrelated child signalled"
        );
        run.no_cleanup();
    }));
    let _ = decoy.kill();
    let _ = decoy.wait();
    let _ = std::fs::remove_dir_all(decoy_dir);
    result.unwrap();
}

#[test]
fn parent_death_after_exec_and_while_gated_prevents_child_survival() {
    let mut run = Run::new("wait");
    run.start();
    run.guard.kill().unwrap();
    run.guard.wait().unwrap();
    run.child_dead();
    run.no_cleanup();
    let mut gated = Run::new("wait");
    gated.observe_gated_child();
    gated.guard.kill().unwrap();
    gated.guard.wait().unwrap();
    gated.child_dead();
    assert!(!gated.ready.exists());
}

#[test]
fn actual_exit_zero_and_nonzero_are_terminal_not_clean_release() {
    for mode in ["exit0", "exit9"] {
        let mut run = Run::new(mode);
        let mut start = run.hello.clone();
        start.kind = START;
        run.send(&start);
        run.exited();
        assert!(run.ready.exists());
    }
}

#[test]
fn explicit_signal_ingress_never_gracefully_stops_child() {
    for signal in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        let mut run = Run::new("wait");
        run.start();
        // Signal this exact std::process::Child; guard itself targets pidfd only.
        assert_eq!(unsafe { libc::kill(run.guard.id() as _, signal) }, 0);
        run.exited();
        run.child_dead();
        run.no_cleanup();
    }
}

#[test]
fn malformed_truncated_and_stale_frames_terminal_before_retry() {
    for fault in 0..6 {
        let mut run = Run::new("wait");
        run.start();
        let mut challenge = run.frame();
        challenge.kind = RENEW;
        match fault {
            0 => {
                run.control.write_all(&4097u32.to_be_bytes()).unwrap();
            }
            1 => {
                run.control.write_all(&challenge.encode()[..5]).unwrap();
            }
            2 => {
                challenge.instance[0] ^= 1;
                run.send(&challenge);
            }
            3 => {
                challenge.session[0] ^= 1;
                run.send(&challenge);
            }
            4 => {
                run.send(&challenge);
                run.send(&challenge);
            }
            _ => {
                run.control.shutdown(std::net::Shutdown::Both).unwrap();
            }
        }
        run.exited();
        run.child_dead();
        run.no_cleanup();
    }
}

#[test]
fn actual_parent_dies_before_pdeathsig_is_established() {
    let dir = random_directory();
    let receipt = dir.join("race");
    let mut probe = Command::new(env!("CARGO_BIN_EXE_probe_child"))
        .arg("parent-race")
        .arg(&receipt)
        .spawn()
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wait_until(|| probe.try_wait().unwrap().is_some());
        assert!(probe.wait().unwrap().success());
        assert_eq!(
            std::fs::read(&receipt).unwrap(),
            b"actual-dead-parent:exit75"
        );
    }));
    let _ = probe.kill();
    let _ = probe.wait();
    let _ = std::fs::remove_dir_all(dir);
    result.unwrap();
}

#[test]
fn trickle_flood_and_delayed_renewal_cannot_extend_terminal_deadline() {
    for fault in 0..3 {
        let mut run = Run::new("wait");
        run.start();
        let mut response = run.frame();
        response.kind = RENEW;
        let bytes = response.encode();
        let began = Instant::now();
        match fault {
            0 => {
                // Keep each inter-byte gap below the frame timeout; total must
                // still expire from the first byte, not the most recent byte.
                for byte in bytes.iter().take(10) {
                    if run.control.write_all(&[*byte]).is_err() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(70));
                }
            }
            1 => {
                // One valid renewal then a bounded repeated-frame flood.
                run.control.set_nonblocking(true).unwrap();
                for _ in 0..1024 {
                    if run.control.write(&bytes).is_err() {
                        break;
                    }
                }
            }
            _ => {
                std::thread::sleep(Duration::from_millis(1400));
                let _ = run.control.write_all(&bytes);
            }
        }
        run.exited();
        run.child_dead();
        run.no_cleanup();
        assert!(began.elapsed() < Duration::from_secs(4));
    }
}

#[test]
fn blocked_outbound_channel_expires_without_releasing_gated_child() {
    let dir = random_directory();
    let ready = dir.join("ready");
    let (control, mut inherited) = UnixStream::pair().unwrap();
    inherited.set_nonblocking(true).unwrap();
    let size = 1024i32;
    assert_eq!(
        unsafe {
            libc::setsockopt(
                inherited.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                (&size as *const i32).cast(),
                4,
            )
        },
        0
    );
    let mut filled = 0;
    loop {
        match inherited.write(&[0x55; 1024]) {
            Ok(n) => {
                filled += n;
                assert!(filled < 65536);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            other => panic!("buffer-fill control: {other:?}"),
        }
    }
    assert!(filled > 0);
    let fd = inherited.as_raw_fd();
    let mut command = Command::new(env!("CARGO_BIN_EXE_glm-pair-guard"));
    command
        .args(["3", "1000", "1200", "200", "300", "5000", "10", "500"])
        .arg(env!("CARGO_BIN_EXE_probe_child"))
        .arg(&ready)
        .arg(dir.join("cleanup"))
        .args(["wait", "cpu-only"]);
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut guard = command.spawn().unwrap();
    drop(inherited);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wait_until(|| guard.try_wait().unwrap().is_some());
        assert_eq!(guard.wait().unwrap().code(), Some(74));
        assert!(!ready.exists());
        // Keep receive endpoint open and unread until after terminal evidence.
        assert!(control.peer_addr().is_ok());
    }));
    let _ = guard.kill();
    let _ = guard.wait();
    drop(control);
    let _ = std::fs::remove_dir_all(dir);
    result.unwrap();
}

#[test]
fn actual_exec_failure_is_terminal_and_validated_inode_is_pinned() {
    use std::os::unix::fs::PermissionsExt;
    let dir = random_directory();
    let executable = dir.join("elf");
    std::fs::write(&executable, b"\x7fELFbroken").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut run = Run::with_executable("wait", &executable);
        run.observe_gated_child();
        let mut start = run.hello.clone();
        start.kind = START;
        run.send(&start);
        run.exited();
        run.child_dead();
        assert!(!run.ready.exists());
        std::fs::copy(env!("CARGO_BIN_EXE_probe_child"), &executable).unwrap();
        let mut pinned = Run::with_executable("wait", &executable);
        std::fs::rename(&executable, dir.join("validated-original")).unwrap();
        std::fs::write(&executable, b"\x7fELFreplaced").unwrap();
        pinned.start();
        pinned.exited();
        pinned.child_dead();
        pinned.no_cleanup();
    }));
    let _ = std::fs::remove_dir_all(dir);
    result.unwrap();
}
