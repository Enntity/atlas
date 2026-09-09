// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[test]
fn exact_node_argv_and_bounded_hex() {
    let session = "11".repeat(32);
    let hash = "22".repeat(32);
    let id = "33".repeat(32);
    for verb in ["node-prepare", "node-create"] {
        let args = vec![
            verb.into(),
            session.clone(),
            "1".into(),
            "--self-sha256".into(),
            hash.clone(),
        ];
        assert_eq!(Command::parse(&args).unwrap().rank, 1);
    }
    for verb in [
        "node-seal",
        "node-start",
        "node-observe",
        "node-socket",
        "node-relay",
        "node-kill",
    ] {
        let args = vec![
            verb.into(),
            session.clone(),
            "0".into(),
            id.clone(),
            "--self-sha256".into(),
            hash.clone(),
        ];
        assert_eq!(Command::parse(&args).unwrap().id, Some([0x33; 32]));
        let mut bad = args.clone();
        bad[3] = "short".into();
        assert!(Command::parse(&bad).is_err());
        let mut bad = args;
        bad[2] = "00".into();
        assert!(Command::parse(&bad).is_err());
    }
    assert_eq!(decode_hex("00ff", 2).unwrap(), [0, 255]);
    for s in ["fff", "FF", "000000"] {
        assert!(decode_hex(s, 2).is_err());
    }
}
#[test]
fn actual_proc_memory_and_unrelated_process_refusal() {
    let (available, _) = proc::memory().unwrap();
    assert!(available > 0);
    // The host caller is deliberately not the Docker PID1 in a private
    // namespace. Never reinterpret it using caller-parent identity helpers.
    assert!(proc::pair(std::process::id(), false).is_err());
}
