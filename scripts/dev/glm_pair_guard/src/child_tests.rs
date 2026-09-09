// SPDX-License-Identifier: AGPL-3.0-only
//! Same Child fork/exec primitive, not PID1 or LIVE release authority.
use super::*;
use std::process::Command;

fn argv() -> Vec<String> {
    vec![
        std::env::current_exe().unwrap().to_str().unwrap().into(),
        "--exact".into(),
        "child::tests::exec_probe".into(),
        "--nocapture".into(),
    ]
}
fn environment(mode: &str) -> Vec<(String, String)> {
    vec![
        ("ATLAS_CHILD_TEST_PROBE".into(), mode.into()),
        ("ATLAS_GLM_PAIR_FD".into(), "3".into()),
        ("PATH".into(), "/usr/bin:/bin".into()),
    ]
}
fn mask() -> libc::sigset_t {
    let mut mask = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::sigprocmask(libc::SIG_SETMASK, std::ptr::null(), &mut mask) },
        0
    );
    mask
}
fn wait_status(child: &Child) -> ExitStatus {
    let until = Io::now().unwrap() + 3000;
    loop {
        if let Some(status) = child.exit_status().unwrap() {
            return status;
        }
        assert!(Io::now().unwrap() < until, "bounded child exit");
        Io::poll(
            &mut [libc::pollfd {
                fd: child.fd(),
                events: libc::POLLIN,
                revents: 0,
            }],
            10,
        )
        .unwrap();
    }
}
// Tests retain the pidfd even when an assertion fails, not a numeric PID kill.
struct Held(Child);
impl Drop for Held {
    fn drop(&mut self) {
        let _ = self.0.terminate();
        let until = Io::now().unwrap_or(0) + 3000;
        while Io::now().unwrap_or(until) < until {
            match self.0.reap() {
                Ok(true) | Err(_) => break,
                Ok(false) => {}
            }
            let _ = Io::poll(
                &mut [libc::pollfd {
                    fd: self.0.fd(),
                    events: libc::POLLIN,
                    revents: 0,
                }],
                10,
            );
        }
    }
}

#[test]
fn exec_probe() {
    let Ok(mode) = std::env::var("ATLAS_CHILD_TEST_PROBE") else {
        return;
    };
    assert_eq!(std::env::var("PATH").unwrap(), "/usr/bin:/bin");
    assert!(std::env::var_os("ATLAS_CHILD_AMBIENT").is_none());
    assert_eq!(std::env::var("ATLAS_GLM_PAIR_FD").unwrap(), "3");
    let channel =
        unsafe { Channel::consume_inherited(3) }.expect("real exec must inherit private FD3");
    assert_eq!(unsafe { libc::fcntl(3, libc::F_GETFD) }, -1);
    assert_ne!(
        unsafe { libc::fcntl(channel.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
        0
    );
    // An ordinary descendant exec cannot retain the consumed private channel.
    let mut command = Command::new("/usr/bin/test");
    let status = command
        .args(["!", "-e", &format!("/proc/self/fd/{}", channel.as_raw_fd())])
        .status()
        .unwrap();
    assert!(status.success());
    let report = format!(
        "{}:{}:{}:{}",
        std::process::id(),
        unsafe { libc::getppid() },
        unsafe { libc::getuid() },
        unsafe { libc::getgid() }
    );
    assert!(channel.send(report.as_bytes()).unwrap());
    match mode.as_str() {
        "exit0" => unsafe { libc::_exit(0) },
        "exit9" => unsafe { libc::_exit(9) },
        "signal" => unsafe {
            libc::raise(libc::SIGKILL);
            libc::_exit(88)
        },
        _ => panic!("explicit probe mode"),
    }
}

#[test]
fn live_channel_gated_identity_environment_and_exact_exit() {
    if std::env::var_os("ATLAS_CHILD_TEST_PARENT").is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "child::tests::live_channel_gated_identity_environment_and_exact_exit",
                "--nocapture",
            ])
            .env("ATLAS_CHILD_TEST_PARENT", "1")
            .env("ATLAS_CHILD_AMBIENT", "must-not-forward")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    for mode in ["exit0", "exit9", "signal"] {
        // Open ELF before pair: in this isolated process FD3 is the collision
        // that LIVE relocation must preserve rather than overwrite at exec.
        let spec = Spec::with_environment(&argv(), &environment(mode)).unwrap();
        assert_eq!(spec.executable.as_raw_fd(), 3, "isolated ELF occupies FD3");
        let (parent, endpoint) = Channel::pair().unwrap();
        let (child, parent) =
            Child::prepare_live(spec, &mask(), Io::now().unwrap() + 3000, parent, endpoint)
                .unwrap();
        let held = Held(child);
        assert!(held.0.pid() > 0);
        assert_eq!(held.0.credentials().pid, held.0.pid());
        assert!(parent.receive(held.0.credentials()).unwrap().is_none());
        assert!(held.0.exit_status().unwrap().is_none());
        // Move ownership only to call release; pidfd remains held throughout.
        let mut held = held;
        held.0.release().unwrap();
        assert!(held.0.release().is_err());
        let until = Io::now().unwrap() + 3000;
        let packet = loop {
            if let Some(packet) = parent
                .receive(held.0.credentials())
                .expect("post-exec private packet")
            {
                break packet;
            }
            assert!(Io::now().unwrap() < until, "bounded child packet");
            Io::poll(
                &mut [libc::pollfd {
                    fd: parent.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                }],
                10,
            )
            .unwrap();
        };
        assert_eq!(
            String::from_utf8(packet).unwrap(),
            format!(
                "{}:{}:{}:{}",
                held.0.pid(),
                std::process::id(),
                held.0.credentials().uid,
                held.0.credentials().gid
            )
        );
        let status = wait_status(&held.0);
        assert_eq!(
            status,
            ExitStatus {
                code: if mode == "signal" {
                    libc::CLD_KILLED
                } else {
                    libc::CLD_EXITED
                },
                status: match mode {
                    "exit0" => 0,
                    "exit9" => 9,
                    _ => libc::SIGKILL,
                }
            }
        );
        assert_eq!(
            held.0.exit_status().unwrap(),
            Some(status),
            "WNOWAIT retains exact status"
        );
        assert!(held.0.reap().unwrap());
    }
}

#[test]
fn explicit_environment_refuses_noncanonical_or_injection() {
    for case in 0..8 {
        let mut env = environment("exit0");
        match case {
            0 => env.swap(0, 1),
            1 => env.insert(1, env[0].clone()),
            2 => env[1].1 = "4".into(),
            3 => env[2].1 = "/tmp".into(),
            4 => env.insert(2, ("LD_PRELOAD".into(), "x".into())),
            5 => env[0].1 = "bad\0value".into(),
            6 => env.insert(2, ("LD_LIBRARY_PATH".into(), "/usr/lib::/tmp".into())),
            _ => env[0].1 = "x".repeat(4097),
        }
        assert!(Spec::with_environment(&argv(), &env).is_err(), "case{case}");
    }
    let (a, b) = Channel::pair().unwrap();
    assert!(Child::prepare_live(
        Spec::new(&argv()).unwrap(),
        &mask(),
        Io::now().unwrap() + 3000,
        a,
        b
    )
    .is_err());
}
