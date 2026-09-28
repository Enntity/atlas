// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn actual_local_identity_and_pinned_elf_are_stable() {
    let before = LocalIdentity::observe().unwrap();
    let deadline = boot_time_ms().unwrap() + 30_000;
    let executable =
        PinnedExecutable::open_process(before.child.pid as u32, 512 << 20, deadline).unwrap();
    assert_ne!(executable.digest(), [0; 32]);
    executable
        .revalidate_process(before.child.pid as u32)
        .unwrap();
    assert_eq!(before, LocalIdentity::observe().unwrap());
    assert_ne!(fresh_nonce().unwrap(), fresh_nonce().unwrap());
    // This ordinary controller control is not successful PID1 consumer proof.
    if before.parent.pid != 1 {
        assert!(before.require_guard_parent().is_err());
    }
}

#[test]
fn explicit_executable_and_time_bounds_refuse() {
    let pid = unsafe { libc::getpid() } as u32;
    assert!(PinnedExecutable::open_process(pid, 1, boot_time_ms().unwrap() + 1000).is_err());
    assert!(PinnedExecutable::open_process(pid, 512 << 20, boot_time_ms().unwrap()).is_err());
    assert!(check_deadline(0).is_err());
}

#[test]
fn stat_parser_handles_spaces_and_parentheses_without_shift() {
    let fields = (3..=22)
        .map(|n| if n == 22 { "987" } else { "1" })
        .collect::<Vec<_>>()
        .join(" ");
    let text = format!("123 (a tricky ) name) {fields}");
    assert_eq!(start_ticks(&text, 123).unwrap(), 987);
    assert!(start_ticks(&text, 124).is_err());
    assert!(start_ticks("123 (name) S 1", 123).is_err());
}
