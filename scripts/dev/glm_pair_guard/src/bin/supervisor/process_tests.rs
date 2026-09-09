// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::time::{Duration, Instant};

fn limits() -> Limits {
    Limits {
        timeout_ms: 1000,
        stdout_bytes: 4096,
        stderr_bytes: 4096,
        stdin_bytes: 4096,
        bytes_per_turn: 16,
    }
}
fn spec(program: &str, args: &[&str]) -> Spec {
    Spec {
        program: program.into(),
        args: args.iter().map(OsString::from).collect(),
        env: vec![],
        stdin: vec![],
    }
}
fn shell(command: &str) -> Spec {
    spec("/bin/sh", &["-c", command])
}
fn collect(p: &mut Process, now: u64) -> io::Result<(Vec<u8>, Vec<u8>, ExitStatus)> {
    let end = Instant::now() + Duration::from_secs(3);
    let (mut out, mut err) = (Vec::new(), Vec::new());
    while Instant::now() < end {
        let progress = p.poll(now)?;
        assert!(progress.stdout.len() <= p.limits.bytes_per_turn);
        assert!(progress.stderr.len() <= p.limits.bytes_per_turn);
        out.extend(progress.stdout);
        err.extend(progress.stderr);
        if progress.done {
            return Ok((out, err, progress.exit.expect("actual reaped status")));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Err(error("bounded CPU fixture collection timed out"))
}
fn cleanup(p: &mut Process) -> ExitStatus {
    p.abort().unwrap();
    let end = Instant::now() + Duration::from_secs(3);
    while Instant::now() < end {
        if let Some(status) = p.reap().unwrap() {
            return status;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("actual retained child did not reap within CPU fixture bound");
}
fn ready(p: &mut Process) {
    let end = Instant::now() + Duration::from_secs(3);
    let mut output = Vec::new();
    while Instant::now() < end {
        output.extend(p.poll(1).unwrap().stdout);
        if output == b"ready" {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("actual child readiness marker absent");
}
#[test]
fn actual_child_deadline_equality_is_terminal_before_more_io() {
    let mut p = Process::spawn(shell("printf ready; exec /bin/sleep 30"), limits(), 0).unwrap();
    ready(&mut p);
    assert_eq!(p.deadline, 1000);
    let expired = p.poll(1000);
    let failed = p.failed;
    cleanup(&mut p);
    assert!(
        expired.is_err(),
        "exact original deadline must refuse, not return pending"
    );
    assert!(failed, "deadline must latch before caller abort");
    assert!(p.poll(1).is_err());
}
#[test]
fn actual_clock_regression_cannot_restore_a_process_budget() {
    let mut p = Process::spawn(shell("printf ready; exec /bin/sleep 30"), limits(), 0).unwrap();
    ready(&mut p);
    p.poll(2).unwrap();
    let regressed = p.poll(1);
    cleanup(&mut p);
    assert!(regressed.is_err(), "supplied BOOTTIME regression must fail");
}
#[test]
fn actual_environment_is_cleared_and_only_literal_values_survive() {
    let mut s = spec("/usr/bin/env", &[]);
    s.env = vec![("ATLAS_CPU_PROCESS_FIXTURE".into(), "literal value".into())];
    let mut p = Process::spawn(s, limits(), 0).unwrap();
    let (out, err, status) = collect(&mut p, 1).unwrap();
    assert_eq!(out, b"ATLAS_CPU_PROCESS_FIXTURE=literal value\n");
    assert!(err.is_empty());
    assert!(status.success());
    assert!(
        p.abort().unwrap().unwrap().success(),
        "already reaped child is never signaled"
    );
    assert!(!p.kill_sent);
}
#[test]
fn actual_stdout_stderr_and_nonzero_exit_are_bounded_and_preserved() {
    let mut l = limits();
    l.bytes_per_turn = 3;
    let mut p = Process::spawn(
        shell("printf 1234567890; printf abcdefghij >&2; exit 7"),
        l,
        0,
    )
    .unwrap();
    let (out, err, status) = collect(&mut p, 1).unwrap();
    assert_eq!(out, b"1234567890");
    assert_eq!(err, b"abcdefghij");
    assert_eq!(status.code(), Some(7));
}
#[test]
fn actual_input_is_chunked_and_all_pipes_are_nonblocking() {
    let mut s = spec("/bin/cat", &[]);
    s.stdin = vec![0x35; 4096];
    let mut l = limits();
    l.bytes_per_turn = 31;
    let mut p = Process::spawn(s, l, 0).unwrap();
    for fd in [
        p.stdin.as_ref().unwrap().as_raw_fd(),
        p.stdout.as_ref().unwrap().as_raw_fd(),
        p.stderr.as_ref().unwrap().as_raw_fd(),
    ] {
        assert_ne!(
            unsafe { libc::fcntl(fd, libc::F_GETFL) } & libc::O_NONBLOCK,
            0
        );
    }
    let (out, err, status) = collect(&mut p, 1).unwrap();
    assert_eq!(out, vec![0x35; 4096]);
    assert!(err.is_empty());
    assert!(status.success());
    assert_eq!(p.written, 4096);
}
#[test]
fn actual_output_overflow_is_error_never_successful_truncation() {
    for stderr in [false, true] {
        let mut l = limits();
        l.stdout_bytes = 3;
        l.stderr_bytes = 3;
        let mut p = Process::spawn(
            shell(if stderr {
                "printf 1234 >&2"
            } else {
                "printf 1234"
            }),
            l,
            0,
        )
        .unwrap();
        let result = collect(&mut p, 1);
        cleanup(&mut p);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeded explicit cap"));
        assert!(p.poll(2).is_err());
    }
}
#[test]
fn abort_uses_only_retained_child_and_leaves_outside_process_alive() {
    let mut victim = Process::spawn(spec("/bin/sleep", &["30"]), limits(), 0).unwrap();
    let mut decoy = Process::spawn(spec("/bin/sleep", &["30"]), limits(), 0).unwrap();
    for p in [&victim, &decoy] {
        assert_eq!(
            unsafe { libc::getpgid(p.child.id() as i32) },
            p.child.id() as i32
        );
    }
    let status = cleanup(&mut victim);
    assert!(!status.success());
    assert!(victim.kill_sent);
    assert!(
        decoy.reap().unwrap().is_none(),
        "outside live child was not targeted"
    );
    assert_eq!(victim.abort().unwrap(), Some(status));
    cleanup(&mut decoy);
}
#[test]
fn explicit_invalid_bounds_and_environment_refuse_before_spawn() {
    let mut s = spec("sleep", &[]);
    assert!(Process::spawn(s, limits(), 0).is_err());
    s = spec("/bin/true", &[]);
    s.env = vec![("X".into(), "1".into()), ("X".into(), "2".into())];
    assert!(Process::spawn(s, limits(), 0).is_err());
    let mut l = limits();
    l.bytes_per_turn = 0;
    assert!(Process::spawn(spec("/bin/true", &[]), l, 0).is_err());
    assert!(Process::spawn(spec("/bin/true", &[]), limits(), u64::MAX).is_err());
    let mut s = spec("/bin/true", &[]);
    s.stdin = vec![0; 4097];
    assert!(Process::spawn(s, limits(), 0).is_err());
}
