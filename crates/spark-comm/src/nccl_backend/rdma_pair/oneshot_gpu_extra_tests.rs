// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Every fault must poison the channel and trap, never hang. By
/// `ATLAS_ONESHOT_FAULT`: `timeout` (no peer), `mismatch` (the peer flags
/// another size), `desync` (its flag is two ops ahead), `proxy` (the proxy
/// failed earlier), `proxy-stage` / `proxy-spin` (it fails while the stream
/// waits for `stage` / the kernel for the peer), `launch` (a launch fails
/// after its fenced stage).
#[test]
#[ignore = "kills the CUDA context: run alone with ATLAS_ONESHOT_FAULT"]
fn oneshot_trap_poisons_instead_of_hanging() {
    let Ok(fault) = std::env::var("ATLAS_ONESHOT_FAULT") else {
        return;
    };
    let Some(g) = gpu() else {
        return;
    };
    let (mode, want) = match fault.as_str() {
        "timeout" => (None, "op 1: the peer never arrived"),
        "mismatch" => (Some(Peer::Lie), "op 1: the peer sent a different size"),
        "desync" => (Some(Peer::Ahead), "op 1: the peer's flag is more than one"),
        "launch" => (None, "op 0: a kernel launch failed"),
        _ => (None, "op 1: the RDMA proxy failed"),
    };
    let cfg = Config {
        // Only `timeout` may take the time limit; the others must beat it.
        timeout_ns: 1_000_000 * if fault == "timeout" { 200 } else { 60_000 },
        stage_fence: fault == "launch" || test_cfg(SPLIT_MIN).stage_fence,
        ..test_cfg(SPLIT_MIN)
    };
    let os = channel(&g, cfg, 1);
    let stop = Arc::new(AtomicBool::new(false));
    let peer = mode.map(|m| spawn_peer(&os, m, stop.clone()));
    let buf = alloc(8192);
    upload(&g, buf, &local_payload(1, 4096));
    let proxy = Channel::new(os.cfg, os.host, 0);
    // SAFETY: the region outlives the test.
    let stage = unsafe { &*((os.host + ctrl_off(os.cfg.max, 1) + STAGE) as *const AtomicU32) };
    match fault.as_str() {
        "proxy" => proxy.fail(),
        // An earlier send the proxy never finishes.
        "proxy-stage" => stage.store(8192, Ordering::Release),
        "launch" => {
            // A stream is no function, so the launch fails behind the stage
            // write; the proxy would serve and clear that orphan stage.
            os.set_kernel(g.stream);
            assert!(os.enqueue(buf, buf, 8192, true, g.stream).is_err());
            os.set_kernel(g.oneshot);
            ck!(cuStreamSynchronize(g.stream));
            stage.store(0, Ordering::Release);
        }
        _ => {}
    }
    let t0 = Instant::now();
    assert!(os.enqueue(buf, buf, 8192, true, g.stream).unwrap());
    if fault.starts_with("proxy-") {
        std::thread::sleep(Duration::from_millis(100));
        proxy.fail();
    }
    let status = unsafe { cuStreamSynchronize(g.stream) };
    let took = t0.elapsed();
    stop.store(true, Ordering::Release);
    if let Some(p) = peer {
        p.join().unwrap();
    }
    let why = os.poisoned();
    println!("fault {fault}: stream status {status} after {took:?}; poison: {why:?}");
    assert_ne!(status, 0, "the kernel must trap");
    assert!(why.expect("poison word set").contains(want));
    assert!(took.as_millis() < 5000);
}
