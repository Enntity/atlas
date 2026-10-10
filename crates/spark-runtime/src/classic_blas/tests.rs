// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use std::sync::Mutex;
#[derive(Default)]
struct FakeIo {
    events: Mutex<Vec<String>>,
    fail: Mutex<Option<&'static str>>,
    profiles: Mutex<Vec<Bf16GemmNumerics>>,
}
impl FakeIo {
    fn step(&self, name: &'static str) -> Result<()> {
        self.events.lock().unwrap().push(name.into());
        anyhow::ensure!(*self.fail.lock().unwrap() != Some(name), "injected {name}");
        Ok(())
    }
}
impl ClassicBlasIo for FakeIo {
    fn check_context(&self, _: u64) -> Result<()> {
        self.step("context")
    }
    fn create(&self) -> Result<u64> {
        self.step("create")?;
        Ok(1)
    }
    fn alloc(&self, _: usize) -> Result<u64> {
        self.step("alloc")?;
        Ok(2)
    }
    fn configure(&self, _: u64, _: u64, _: u64, numerics: Bf16GemmNumerics) -> Result<()> {
        self.profiles.lock().unwrap().push(numerics);
        self.step("configure")
    }
    fn launch(&self, _: u64, _: Bf16Gemm) -> Result<()> {
        self.step("launch")
    }
    fn synchronize(&self, _: u64, _: u64) -> Result<()> {
        self.step("sync")
    }
    fn cleanup(&self, _: u64, _: u64, handle: u64, workspace: u64, pending: bool) -> Result<()> {
        let mut e = self.events.lock().unwrap();
        if pending {
            e.push("sync".into());
        }
        if workspace != 0 {
            e.push("free".into());
        }
        if handle != 0 {
            e.push("destroy".into());
        }
        Ok(())
    }
}
fn request(stream: u64) -> Bf16Gemm {
    Bf16Gemm {
        input: crate::gpu::DevicePtr(1),
        weight: crate::gpu::DevicePtr(2),
        output: crate::gpu::DevicePtr(3),
        rows: 128,
        columns: 2048,
        inner: 4096,
        stream,
        numerics: Bf16GemmNumerics::Glm53Classic,
    }
}
#[test]
fn allocation_failure_destroys_created_handle() {
    let io = Arc::new(FakeIo::default());
    *io.fail.lock().unwrap() = Some("alloc");
    let pool = ClassicBf16Pool::new(7, io.clone());
    assert!(pool.launch(request(9)).is_err());
    assert_eq!(
        *io.events.lock().unwrap(),
        ["context", "create", "alloc", "destroy"]
    );
}
#[test]
fn configuration_failure_frees_workspace_and_handle() {
    let io = Arc::new(FakeIo::default());
    *io.fail.lock().unwrap() = Some("configure");
    let pool = ClassicBf16Pool::new(7, io.clone());
    assert!(pool.launch(request(9)).is_err());
    assert_eq!(
        *io.events.lock().unwrap(),
        ["context", "create", "alloc", "configure", "free", "destroy"]
    );
}
#[test]
fn streams_have_owned_resources_and_drop_waits_before_free() {
    let io = Arc::new(FakeIo::default());
    {
        let pool = ClassicBf16Pool::new(7, io.clone());
        pool.launch(request(9)).unwrap();
        pool.launch(request(9)).unwrap();
        pool.launch(request(10)).unwrap();
    }
    let events = io.events.lock().unwrap();
    assert_eq!(events.iter().filter(|s| *s == "create").count(), 2);
    assert_eq!(events.iter().filter(|s| *s == "launch").count(), 3);
    assert_eq!(events.iter().filter(|s| *s == "sync").count(), 2);
    assert_eq!(events.iter().filter(|s| *s == "free").count(), 2);
    for (index, event) in events.iter().enumerate() {
        if event == "free" {
            assert_eq!(events[index - 1], "sync");
            assert_eq!(events[index + 1], "destroy");
        }
    }
}
#[test]
fn wrong_context_and_i32_overflow_fail_before_handle_creation() {
    let io = Arc::new(FakeIo::default());
    let pool = ClassicBf16Pool::new(7, io.clone());
    let mut invalid = request(9);
    invalid.rows = u32::MAX;
    assert!(pool.launch(invalid).is_err());
    assert!(io.events.lock().unwrap().is_empty());
    *io.fail.lock().unwrap() = Some("context");
    assert!(pool.launch(request(9)).is_err());
    assert_eq!(*io.events.lock().unwrap(), ["context"]);
}
#[test]
fn failed_launch_still_waits_before_reclaiming_resources() {
    let io = Arc::new(FakeIo::default());
    *io.fail.lock().unwrap() = Some("launch");
    {
        let pool = ClassicBf16Pool::new(7, io.clone());
        assert!(pool.launch(request(9)).is_err());
    }
    let events = io.events.lock().unwrap();
    assert_eq!(&events[events.len() - 3..], ["sync", "free", "destroy"]);
}
#[test]
fn lt_request_never_creates_classic_resources() {
    let io = Arc::new(FakeIo::default());
    let pool = ClassicBf16Pool::new(7, io.clone());
    let mut req = request(9);
    req.numerics = Bf16GemmNumerics::CublasLt;
    assert!(pool.launch(req).is_err());
    assert!(io.events.lock().unwrap().is_empty());
}

#[test]
fn quiescence_failure_preserves_pending_until_success() {
    let io = Arc::new(FakeIo::default());
    let pool = ClassicBf16Pool::new(7, io.clone());
    pool.launch(request(9)).unwrap();
    *io.fail.lock().unwrap() = Some("sync");
    assert!(pool.quiesce().is_err());
    assert!(!io.events.lock().unwrap().iter().any(|s| s == "free"));
    *io.fail.lock().unwrap() = None;
    pool.quiesce().unwrap();
    pool.quiesce().unwrap();
    let syncs = io
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|s| *s == "sync")
        .count();
    assert_eq!(syncs, 2);
    drop(pool);
    assert_eq!(
        io.events
            .lock()
            .unwrap()
            .iter()
            .filter(|s| *s == "sync")
            .count(),
        2
    );
}
#[test]
fn one_stream_reconfigures_profiles_and_retries_failed_configuration() {
    let io = Arc::new(FakeIo::default());
    let pool = ClassicBf16Pool::new(7, io.clone());
    pool.launch(request(9)).unwrap();
    let mut auxiliary = request(9);
    auxiliary.numerics = Bf16GemmNumerics::Glm53AuxiliaryClassic;
    *io.fail.lock().unwrap() = Some("configure");
    assert!(pool.launch(auxiliary).is_err());
    assert_eq!(
        io.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| *e == "launch")
            .count(),
        1
    );
    *io.fail.lock().unwrap() = None;
    pool.launch(request(9)).unwrap();
    pool.launch(auxiliary).unwrap();
    pool.launch(auxiliary).unwrap();
    pool.launch(request(9)).unwrap();
    assert_eq!(
        *io.profiles.lock().unwrap(),
        [
            Bf16GemmNumerics::Glm53Classic,
            Bf16GemmNumerics::Glm53AuxiliaryClassic,
            Bf16GemmNumerics::Glm53Classic,
            Bf16GemmNumerics::Glm53AuxiliaryClassic,
            Bf16GemmNumerics::Glm53Classic,
        ]
    );
    let events = io.events.lock().unwrap();
    assert_eq!(events.iter().filter(|e| *e == "create").count(), 1);
    assert_eq!(events.iter().filter(|e| *e == "alloc").count(), 1);
    assert_eq!(events.iter().filter(|e| *e == "configure").count(), 5);
    assert_eq!(events.iter().filter(|e| *e == "launch").count(), 5);
}
#[test]
fn coherent_provider_profile_rejects_mixed_version_or_directory() {
    let shared = Bf16GemmNumerics::Glm53Classic;
    let classic = std::path::Path::new("/pinned/libcublas.so.13");
    let lt = std::path::Path::new("/pinned/libcublasLt.so.13");
    super::validate_provider_pair(shared, 130101, 130101, classic, lt).unwrap();
    assert!(super::validate_provider_pair(shared, 130000, 130101, classic, lt).is_err());
    assert!(super::validate_provider_pair(shared, 130101, 130000, classic, lt).is_err());
    assert!(super::validate_provider_pair(
        shared,
        130101,
        130101,
        classic,
        std::path::Path::new("/other/libcublasLt.so.13")
    )
    .is_err());
    assert!(
        super::validate_provider_pair(Bf16GemmNumerics::CublasLt, 130101, 130101, classic, lt)
            .is_err()
    );
}
