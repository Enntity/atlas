// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn retained_tail_rejects_foreign_generation_and_prompt() {
    let tail = OwnedRepairRows {
        ptr: DevicePtr(8192),
        row_bytes: 8192,
        generation: 17,
        prompt: 15810,
    };
    assert!(tail.validate(17, 15810, 8192).is_ok());
    assert!(tail.validate(18, 15810, 8192).is_err());
    assert!(tail.validate(17, 15811, 8192).is_err());
    assert!(tail.validate(17, 15810, 1024).is_err());
    assert_eq!(tail.staging().ptr, DevicePtr(16384));
    assert_eq!(tail.staging().bytes, 16384);
}
