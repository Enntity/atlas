// SPDX-License-Identifier: AGPL-3.0-only
//! Real Model capability + T1 operation; not inherited-ticket registration proof.

use super::*;
use spark_model::model::glm_c2_test_support::Fixture;
use std::io::Write;

#[path = "../scheduler/glm_c2_fixture_test_process.rs"]
mod process;

const TEST: &str =
    "glm_terminal_session::selected::tests::actual_completion_health_precedes_return";
const MODE: &str = "ATLAS_C2_OWNER_OPERATION_CASE";
const MARKER: &str = "ATLAS_C2_OWNER_OPERATION_MARKER";

fn mark(text: &str) {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::var(MARKER).unwrap())
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
}

struct Witness;
impl Drop for Witness {
    fn drop(&mut self) {
        mark("drop\n");
    }
}

#[test]
fn actual_completion_health_precedes_return() {
    if process::isolated(TEST) {
        return;
    }
    let Ok(mode) = std::env::var(MODE) else {
        for mode in [
            "healthy",
            "health-failure",
            "missing-capability",
            "operation-error",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let marker = directory.path().join("marker");
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--nocapture"])
                .env(MODE, mode)
                .env(MARKER, &marker)
                .output()
                .unwrap();
            let text = std::fs::read_to_string(marker).unwrap_or_default();
            if mode == "healthy" {
                assert!(output.status.success(), "{mode}: {output:?}");
                assert_eq!(text, "completed\ndrop\n");
            } else {
                assert_eq!(output.status.code(), Some(74), "{mode}: {output:?}");
                assert!(text.is_empty(), "{mode}: returned or dropped: {text}");
            }
        }
        return;
    };
    let mut fixture = if mode == "missing-capability" {
        Fixture::legacy(0)
    } else {
        Fixture::paired(0)
    };
    let _wire = fixture.install_wire();
    let (mut model, mut sequences, observer) = fixture.into_parts();
    crate::glm_terminal_session::install_panic_ingress();
    let key = crate::glm_terminal_session::CORE.activate().unwrap();
    let _witness = Witness;
    let operation = SelectedOperation::begin(&model, &key);
    observer.clear();
    if mode == "health-failure" {
        // Actual Wire::is_healthy calls the existing backend event recorder.
        observer.fail_at(1);
    }
    if mode == "operation-error" {
        operation.require::<()>(Err(anyhow::anyhow!("actual operation boundary error")));
    }
    operation.complete();
    mark("completed\n");
    // Test-only healthy control closes its local operation core; production
    // SelectedModel never disarms and never takes this cleanup path.
    key.close().unwrap();
    observer.clear();
    for sequence in &mut sequences {
        model.free_sequence(sequence).unwrap();
    }
    drop(sequences);
    model.teardown().unwrap();
}
