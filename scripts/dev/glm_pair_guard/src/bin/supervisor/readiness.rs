// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded curl response interpretation; no readiness inferred from process liveness.
use crate::error;
use std::io;
pub fn ready(code: Option<i32>, stdout: &[u8], model: &str) -> io::Result<bool> {
    if code == Some(7) && stdout == b"\n000" {
        return Ok(false);
    }
    if code != Some(0) {
        return Err(error("HTTP readiness transport failed"));
    }
    let split = stdout
        .iter()
        .rposition(|b| *b == b'\n')
        .ok_or_else(|| error("missing HTTP status"))?;
    match &stdout[split + 1..] {
        b"503" => Ok(false),
        b"200" => {
            let value: serde_json::Value = serde_json::from_slice(&stdout[..split])?;
            if value.get("status").and_then(serde_json::Value::as_str) != Some("ready")
                || value.get("model").and_then(serde_json::Value::as_str) != Some(model)
            {
                return Err(error("readiness model/status mismatch"));
            }
            Ok(true)
        }
        _ => Err(error("unexpected HTTP readiness response")),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_health_shape_requires_success_status_and_exact_model() {
        assert!(ready(
            Some(0),
            br#"{"status":"ready","model":"glm"}
200"#,
            "glm"
        )
        .unwrap());
        assert!(!ready(Some(7), b"\n000", "glm").unwrap());
        assert!(!ready(Some(0), b"loading\n503", "glm").unwrap());
        for (code, bytes) in [
            (
                Some(0),
                b"{\"status\":\"ready\",\"model\":\"foreign\"}\n200".as_slice(),
            ),
            (Some(0), b"{}\n200"),
            (Some(0), b"{}\n302"),
            (Some(28), b"\n000"),
            (None, b"\n000"),
        ] {
            assert!(ready(code, bytes, "glm").is_err());
        }
    }
}
