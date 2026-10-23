// SPDX-License-Identifier: AGPL-3.0-only

//! Read resolved still-image policy without coupling benchmarks to GPU code.
use super::*;

pub async fn image_processor(target: &TargetEndpoint, timeout: Duration) -> Result<Value> {
    parse(&get_models(target, timeout).await?, &target.model)
}

fn parse(raw: &str, model: &str) -> Result<Value> {
    let (headers, body) = raw.split_once("\r\n\r\n").context("missing HTTP headers")?;
    ensure_success(headers)?;
    let decoded = if headers.lines().any(|line| {
        line.split_once(':').is_some_and(|(key, value)| {
            key.eq_ignore_ascii_case("transfer-encoding")
                && value.trim().eq_ignore_ascii_case("chunked")
        })
    }) {
        dechunk(body.as_bytes())
    } else {
        body.to_owned()
    };
    let doc: Value = serde_json::from_str(&decoded).context("invalid model profile JSON")?;
    let entry = doc["data"]
        .as_array()
        .context("missing model list")?
        .iter()
        .find(|row| row["id"].as_str() == Some(model))
        .with_context(|| format!("served model {model} is absent from /v1/models"))?;
    entry
        .get("atlas_image_processor")
        .filter(|v| v.is_object())
        .cloned()
        .context("model has no supported resolved still-image processor profile")
}

fn ensure_success(headers: &str) -> Result<()> {
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1));
    if status != Some("200") {
        bail!("/v1/models returned HTTP status {status:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_chunk_boundaries_and_selects_the_requested_model() {
        let body = r#"{"data":[{"id":"other"},{"id":"served","atlas_image_processor":{"algorithm":"qwen-smart-resize-v1"}}]}"#;
        let split = 53;
        let raw = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            split,
            &body[..split],
            body.len() - split,
            &body[split..]
        );
        assert_eq!(
            parse(&raw, "served").unwrap()["algorithm"],
            "qwen-smart-resize-v1"
        );
        assert!(parse(&raw, "other").is_err());
        assert!(parse(&raw, "missing").is_err());
        assert!(parse(&raw.replace("200 OK", "500 Error"), "served").is_err());
    }
}
