// SPDX-License-Identifier: AGPL-3.0-only
//! Process-environment and load-boundary tests for the substitution knob.
//! Split from `norm_substitution_tests.rs` to keep both under the 500 LoC
//! file cap. The `from_env` tests mutate the knob env vars under the
//! process-wide env lock (repo ENV_LOCK pattern: `spark-server/src/refusal.rs`,
//! `spark-server/src/tool_parser/tests/streaming_frag_env.rs`) because cargo
//! runs these tests in parallel threads of one binary.

use super::tests::{PAYLOAD, manifest_json, payload_entry, write_fixture};
use super::*;
use std::sync::{Mutex, MutexGuard, OnceLock};

// STATIC, DELIBERATELY — it guards the PROCESS environment, which cannot be
// scoped: any test reading or writing the knob vars must hold this lock.
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Shared with `tests::knob_off_*`, which reads the knob env and must not
/// observe a transient engaged state while these tests mutate it.
pub(super) fn lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

pub(super) struct EnvGuard {
    prev: Vec<(&'static str, Option<String>)>,
    _lock: MutexGuard<'static, ()>,
}

/// Set (or, with `None`, remove) the given env vars for the guard's
/// duration, restoring every previous value on drop.
pub(super) fn env(vars: &[(&'static str, Option<&str>)]) -> EnvGuard {
    let _lock = lock();
    let mut prev = Vec::new();
    for (key, value) in vars {
        prev.push((*key, std::env::var(key).ok()));
        // SAFETY: serialised by ENV_LOCK; no other thread touches these vars
        // without holding the same lock.
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    EnvGuard { prev, _lock }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.prev {
            // SAFETY: still holding ENV_LOCK via `_lock`.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn engaged_vars<'a>(fixture: &'a Path, records: &'a Path) -> Vec<(&'static str, Option<&'a str>)> {
    vec![
        (ENV_SUBSTITUTE_DIR, Some(fixture.to_str().unwrap())),
        (ENV_RECORD_DIR, Some(records.to_str().unwrap())),
        ("ATLAS_NO_DECODE_GRAPHS", Some("1")),
        ("ATLAS_NO_DECODE_GRAPHS_MULTISEQ", Some("1")),
        ("ATLAS_DEBUG_NO_GRAPH", Some("1")),
    ]
}

fn fixture_with_entry(tag: &str) -> (PathBuf, PathBuf) {
    let manifest = manifest_json(vec![payload_entry("req-a", 0)]);
    write_fixture(tag, &manifest, &[("L0-rank0-gated.bin", &PAYLOAD)])
}

#[test]
fn from_env_engaged_parses_manifest_and_stages_payloads() {
    let (fixture, records) = fixture_with_entry("env-positive");
    let _g = env(&engaged_vars(&fixture, &records));
    let sub = NormSubstitution::from_env(0, 2)
        .unwrap()
        .expect("engaged env must produce the substitution table");
    assert_eq!(sub.entries.len(), 1);
    assert!(sub.reject_multi_seq().is_err());
}

#[test]
fn from_env_unset_or_empty_substitute_dir_is_off() {
    let (fixture, records) = fixture_with_entry("env-off");
    for off in [None, Some("")] {
        let mut vars = engaged_vars(&fixture, &records);
        vars[0] = (ENV_SUBSTITUTE_DIR, off);
        let _g = env(&vars);
        assert!(NormSubstitution::from_env(0, 2).unwrap().is_none());
    }
}

#[test]
fn from_env_requires_every_graph_interlock_flag_at_one() {
    let (fixture, records) = fixture_with_entry("env-interlock");
    for flag in GRAPH_FLAGS {
        for value in [Some("0"), None] {
            let mut vars = engaged_vars(&fixture, &records);
            for var in &mut vars {
                if var.0 == flag {
                    *var = (flag, value);
                }
            }
            let _g = env(&vars);
            let err = NormSubstitution::from_env(0, 2)
                .map(|_| ())
                .expect_err("a non-1 graph interlock flag must abort the load");
            assert!(err.to_string().contains(flag), "{err:#}");
        }
    }
}

#[test]
fn from_env_requires_explicit_record_dir_when_engaged() {
    let (fixture, records) = fixture_with_entry("env-record-dir");
    for absent in [None, Some("")] {
        let mut vars = engaged_vars(&fixture, &records);
        vars[1] = (ENV_RECORD_DIR, absent);
        let _g = env(&vars);
        let err = NormSubstitution::from_env(0, 2)
            .map(|_| ())
            .expect_err("an engaged knob without a record dir must abort the load");
        assert!(err.to_string().contains(ENV_RECORD_DIR), "{err:#}");
    }
}

#[test]
fn load_rejects_record_dir_nested_inside_fixture_dir() {
    let (fixture, _records) = fixture_with_entry("nested-records");
    let nested = fixture.join("records");
    std::fs::create_dir(&nested).unwrap();
    let err = NormSubstitution::load(&fixture, &nested, 0, 2)
        .map(|_| ())
        .expect_err("a record dir nested under the fixture dir must be rejected");
    assert!(err.to_string().contains("nested under it"), "{err:#}");
}

#[test]
fn load_allows_adjacent_record_dir_sharing_the_name_prefix() {
    // Path-component semantics: `/x/fixtures-adjacent` must not be mistaken
    // for nested-under `/x/fixtures` by a string-prefix comparison.
    let (fixture, _records) = fixture_with_entry("adjacent-records");
    let adjacent = fixture.parent().unwrap().join("fixtures-adjacent");
    std::fs::create_dir(&adjacent).unwrap();
    NormSubstitution::load(&fixture, &adjacent, 0, 2)
        .expect("a shared name prefix must not count as nesting");
}
