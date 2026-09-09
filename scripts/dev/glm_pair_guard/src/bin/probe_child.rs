// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit CPU test executable, never a serving child or GPU test.
#[allow(dead_code)] // CPU race adapter reuses the real pre-exec primitive only.
#[path = "../child.rs"]
mod child;
#[allow(dead_code)] // Same syscall boundary; unrelated runner methods aren't used.
#[path = "../linux.rs"]
mod linux;
use std::sync::atomic::{AtomicBool, Ordering};
static TERMINATE: AtomicBool = AtomicBool::new(false);
static AT_EXIT_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
extern "C" fn signal(_: i32) {
    TERMINATE.store(true, Ordering::Relaxed);
}
extern "C" fn at_exit() {
    if let Some(path) = AT_EXIT_PATH.get() {
        let _ = std::fs::write(path, b"atexit");
    }
}
struct Witness(String);
impl Drop for Witness {
    fn drop(&mut self) {
        let _ = std::fs::write(&self.0, b"drop");
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("parent-race") {
        parent_race(&args[2]);
        return;
    }
    assert_eq!(args.len(), 5);
    AT_EXIT_PATH.set(format!("{}.atexit", args[2])).unwrap();
    let _witness = Witness(format!("{}.drop", args[2]));
    unsafe {
        let mut action = std::mem::zeroed::<libc::sigaction>();
        action.sa_sigaction = signal as *const () as usize;
        libc::sigemptyset(&mut action.sa_mask);
        assert_eq!(
            libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()),
            0
        );
        assert_eq!(libc::atexit(at_exit), 0);
    }
    std::fs::write(&args[1], std::process::id().to_string()).unwrap();
    match args[3].as_str() {
        "exit0" => return,
        "exit9" => std::process::exit(9),
        "wait" => {}
        _ => panic!("unknown harmless child mode"),
    }
    // Literal argv sentinel demonstrates real exec and no inherited LD controls.
    assert_eq!(args[4], "cpu-only");
    while !TERMINATE.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// A CPU-only rendezvous, not a guard runtime flag. The intermediate process
/// actually exits before the leaf invokes the same production parent check.
fn parent_race(receipt: &str) {
    use std::os::fd::AsRawFd;
    unsafe {
        assert_eq!(libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1), 0);
        let mut gate = [-1; 2];
        let mut report = [-1; 2];
        assert_eq!(libc::pipe2(gate.as_mut_ptr(), libc::O_CLOEXEC), 0);
        assert_eq!(libc::pipe2(report.as_mut_ptr(), libc::O_CLOEXEC), 0);
        let controller = libc::getpid();
        let parent = libc::fork();
        assert!(parent >= 0);
        if parent == 0 {
            libc::close(gate[1]);
            libc::close(report[0]);
            child::protect_parent(controller);
            let expected = libc::getpid();
            let leaf = libc::fork();
            if leaf < 0 {
                libc::_exit(77);
            }
            if leaf == 0 {
                libc::close(report[1]);
                let mut byte = 0u8;
                if libc::read(gate[0], (&mut byte as *mut u8).cast(), 1) != 1 {
                    libc::_exit(78);
                }
                child::protect_parent(expected);
                libc::_exit(0); // Wrong: production check must terminate first.
            }
            if libc::write(report[1], (&leaf as *const libc::pid_t).cast(), 4) != 4 {
                libc::_exit(79);
            }
            libc::_exit(0);
        }
        libc::close(gate[0]);
        libc::close(report[1]);
        let gate = linux::owned(gate[1]).unwrap();
        let report = linux::owned(report[0]).unwrap();
        let mut p = libc::pollfd {
            fd: report.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(libc::poll(&mut p, 1, 3000), 1);
        let mut leaf = 0 as libc::pid_t;
        assert_eq!(
            libc::read(
                report.as_raw_fd(),
                (&mut leaf as *mut libc::pid_t).cast(),
                4
            ),
            4
        );
        let fd = linux::owned(libc::syscall(libc::SYS_pidfd_open, leaf, 0) as i32).unwrap();
        let deadline = linux::Io::now().unwrap() + 3000;
        let mut status = 0;
        loop {
            let result = libc::waitpid(parent, &mut status, libc::WNOHANG);
            assert!(result >= 0);
            if result == parent {
                break;
            }
            assert!(linux::Io::now().unwrap() < deadline);
            libc::poll(std::ptr::null_mut(), 0, 2);
        }
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        let mut p = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            libc::poll(&mut p, 1, 0),
            0,
            "leaf must still await rendezvous"
        );
        assert_eq!(libc::write(gate.as_raw_fd(), b"R".as_ptr().cast(), 1), 1);
        assert_eq!(libc::poll(&mut p, 1, 3000), 1);
        let mut info = std::mem::zeroed::<libc::siginfo_t>();
        assert_eq!(
            libc::waitid(
                libc::P_PIDFD,
                fd.as_raw_fd() as _,
                &mut info,
                libc::WEXITED | libc::WNOHANG
            ),
            0
        );
        assert_eq!(info.si_code, libc::CLD_EXITED);
        assert_eq!(
            info.si_status(),
            75,
            "real parent race must refuse before exec"
        );
        std::fs::write(receipt, b"actual-dead-parent:exit75").unwrap();
    }
}
