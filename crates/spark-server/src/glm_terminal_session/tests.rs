// SPDX-License-Identifier: AGPL-3.0-only

//! Production core/sink/hooks in subprocesses; no Model/protocol proof.
use super::*;
use std::io::Write;
use std::os::fd::IntoRawFd;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    OnceLock,
    atomic::{AtomicI32, Ordering},
};

const CHILD: &str = "ATLAS_TEST_GLM_TERMINAL_CASE";
const PATH: &str = "ATLAS_TEST_GLM_TERMINAL_MARKERS";
static MARKERS: OnceLock<PathBuf> = OnceLock::new();
static EXIT_FD: AtomicI32 = AtomicI32::new(-1);

fn mark(text: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(MARKERS.get().unwrap())
        .unwrap();
    file.write_all(text.as_bytes()).unwrap();
}
struct DropMarker(&'static str);
impl Drop for DropMarker {
    fn drop(&mut self) {
        mark(self.0);
    }
}

#[derive(Debug)]
struct DropError;
impl std::fmt::Display for DropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("fixture error")
    }
}
impl std::error::Error for DropError {}
impl Drop for DropError {
    fn drop(&mut self) {
        mark("error-drop\n");
    }
}

extern "C" fn at_exit() {
    let bytes = b"atexit\n";
    // Pre-opened append-only test descriptor; the real sink never calls this.
    unsafe {
        libc::write(
            EXIT_FD.load(Ordering::Relaxed),
            bytes.as_ptr().cast(),
            bytes.len(),
        );
    }
}

fn setup_observers() {
    MARKERS
        .set(PathBuf::from(std::env::var(PATH).unwrap()))
        .unwrap();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(MARKERS.get().unwrap())
        .unwrap();
    EXIT_FD.store(file.into_raw_fd(), Ordering::Relaxed);
    assert_eq!(unsafe { libc::atexit(at_exit) }, 0);
    std::panic::set_hook(Box::new(|_| mark("prior-hook\n")));
    install_panic_ingress();
}

fn nested_caught_panic() {
    let result = std::panic::catch_unwind(|| {
        let _inner = DropMarker("inner-drop\n");
        panic!("actual nested test panic");
    });
    assert!(result.is_err());
    mark("resumed\n");
}

#[test]
fn child_entry() {
    let Ok(mode) = std::env::var(CHILD) else {
        return;
    };
    setup_observers();
    let _outer = DropMarker("outer-drop\n");
    if mode == "inert" || mode == "inert-tui" {
        if mode == "inert-tui" {
            crate::tui::terminal_guard::install_panic_hook();
        }
        nested_caught_panic();
        return;
    }
    let key = CORE.activate().unwrap();
    match mode.as_str() {
        "clean" => {
            assert!(CORE.activate().is_err());
            let op = key.begin().unwrap();
            assert!(key.begin().is_err());
            // A second real core cannot complete this core's operation.
            let other = core::TerminalCore::new();
            let other_key = other.activate().unwrap();
            other_key.begin().unwrap().complete();
            other_key.close().unwrap();
            assert!(key.begin().is_err());
            assert_eq!(op.require(Ok(42)), 42);
            op.complete();
            key.begin().unwrap().complete();
            key.close().unwrap();
            assert!(CORE.activate().is_err());
            nested_caught_panic();
        }
        "error" => {
            let op = key.begin().unwrap();
            op.require::<()>(Err(anyhow::Error::new(DropError)));
            mark("continued-after-error\n");
        }
        "panic" | "panic-tui" => {
            let op = key.begin().unwrap();
            if mode == "panic-tui" {
                crate::tui::terminal_guard::install_panic_hook();
            }
            nested_caught_panic();
            op.complete();
            key.close().unwrap();
        }
        "panic-idle" => {
            nested_caught_panic();
            key.close().unwrap();
        }
        "panic-thread-idle" => {
            let worker = std::thread::spawn(|| {
                let _thread = DropMarker("thread-drop\n");
                nested_caught_panic();
            });
            let _ = worker.join();
            mark("joined\n");
            key.close().unwrap();
        }
        "abandon-op" => {
            drop(key.begin().unwrap());
        }
        "abandon-session" => {
            drop(key);
        }
        "forgotten-op" => {
            std::mem::forget(key.begin().unwrap());
            assert!(key.begin().is_err());
            let _ = key.close();
        }
        _ => panic!("unknown child test mode"),
    }
}

fn child(mode: &str) -> (std::process::Output, String) {
    let dir = tempfile::tempdir().unwrap();
    let markers = dir.path().join("events.txt");
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "glm_terminal_session::tests::child_entry",
            "--nocapture",
        ])
        .env(CHILD, mode)
        .env(PATH, &markers)
        .output()
        .unwrap();
    let marks = std::fs::read_to_string(&markers).unwrap();
    (output, marks)
}

#[test]
fn actual_core_positive_and_inert_hooks_run_normal_destructors() {
    for mode in ["clean", "inert", "inert-tui"] {
        let (out, marks) = child(mode);
        assert!(
            out.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            marks, "prior-hook\ninner-drop\nresumed\nouter-drop\natexit\n",
            "{mode}"
        );
        if mode == "inert-tui" {
            assert!(String::from_utf8_lossy(&out.stderr).contains("atlas-tui: panic"));
        }
    }
}

#[test]
fn actual_error_and_abandonment_skip_rust_and_c_destructors() {
    for mode in ["error", "abandon-op", "abandon-session", "forgotten-op"] {
        let (out, marks) = child(mode);
        assert_eq!(out.status.code(), Some(EXIT_GLM_PAIRED_UNCERTAIN), "{mode}");
        assert!(marks.is_empty(), "{mode}: destructors ran: {marks:?}");
    }
}

#[test]
fn actual_panic_ingress_precedes_nested_drop_catch_and_prior_hooks() {
    for mode in ["panic", "panic-idle", "panic-tui", "panic-thread-idle"] {
        let (out, marks) = child(mode);
        assert_eq!(
            out.status.code(),
            Some(EXIT_GLM_PAIRED_UNCERTAIN),
            "{mode}: {marks:?}"
        );
        assert!(marks.is_empty(), "{mode}: unwind/hook activity: {marks:?}");
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("atlas-tui: panic"),
            "late TUI hook ran before terminal interception"
        );
    }
}
